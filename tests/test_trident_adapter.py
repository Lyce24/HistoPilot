"""TRIDENT plans stay pure; the standalone runner executes only fixture commands."""

import argparse
import ast
import json
import os
import signal
import subprocess
import sys
import time
import venv
from pathlib import Path

import pytest
from pydantic import ValidationError

from histopilot.adapters.trident import (
    TridentOptions,
    build_command,
    discover_runtime,
    option_catalog,
    output_layout,
)

RUNNER = Path(__file__).parents[1] / "histopilot/adapters/trident/runner.py"
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def fixture_checkout(tmp_path, extra=""):
    root = tmp_path / "TRIDENT checkout"
    root.mkdir()
    (root / "run_batch_of_slides.py").write_text(
        "import torch\nraise RuntimeError('must never be imported')\n"
        "parser.add_argument('--wsi_dir')\nparser.add_argument('--job_dir')\n"
        "parser.add_argument('--task', choices=['seg', 'coords', 'feat', 'all'])\n"
        "parser.add_argument('--patch_encoder')\n"
        "parser.add_argument('--patch_size', type=int)\n"
        "parser.add_argument('--mag', type=int)\n"
        "parser.add_argument('--custom_list_of_wsis')\n"
        "parser.add_argument('--gpu', type=int)\n" + extra
    )
    registry = root / "trident/patch_encoder_models"
    registry.mkdir(parents=True)
    (registry / "load.py").write_text("encoder_registry = {'uni_v1': NeverImportedModel}\n")
    return root


def test_catalog_covers_current_batch_flags_and_has_typed_controls():
    names = {
        "gpu",
        "gpus",
        "task",
        "skip_errors",
        "clear_dead_locks",
        "dead_lock_max_age_hours",
        "max_workers",
        "batch_size",
        "wsi_cache",
        "cache_batch_size",
        "wsi_ext",
        "custom_mpp_keys",
        "custom_list_of_wsis",
        "reader_type",
        "search_nested",
        "segmenter",
        "seg_conf_thresh",
        "remove_holes",
        "remove_artifacts",
        "remove_penmarks",
        "seg_batch_size",
        "mag",
        "patch_size",
        "overlap",
        "min_tissue_proportion",
        "coords_dir",
        "dump_patches",
        "dump_patches_max",
        "dump_patches_format",
        "dump_patches_jpeg_quality",
        "patch_encoder",
        "patch_encoder_ckpt_path",
        "patch_encoder_img_size",
        "slide_encoder",
        "feat_batch_size",
    }
    catalog = option_catalog()
    assert {item["name"] for item in catalog["options"]} == names
    assert set(catalog["managedOptions"]) == {"wsi_dir", "job_dir"}
    assert catalog["defaults"]["patch_size"] == 256
    assert catalog["defaults"]["patch_encoder"] == "uni_v1"
    by_name = {item["name"]: item for item in catalog["options"]}
    assert by_name["gpus"]["type"] == "integer[]"
    assert by_name["custom_mpp_keys"]["type"] == "string[]"
    assert by_name["remove_penmarks"]["advanced"]
    assert "otsu" in by_name["segmenter"]["choices"]
    assert "czi" in by_name["reader_type"]["choices"]


@pytest.mark.parametrize(
    "values",
    [
        {"overlap": 256},
        {"mag": 0},
        {"mag": float("nan")},
        {"gpu": -2},
        {"gpus": []},
        {"gpus": [-3]},
        {"batch_size": 0},
        {"seg_conf_thresh": 1.1},
        {"coords_dir": "../escape"},
        {"coords_dir": "/outside"},
        {"wsi_ext": ["--skip_errors"]},
        {"unknown_flag": True},
        {"patch_encoder_img_size": 225},
        {"patch_encoder_img_size": 224, "patch_encoder": "resnet50"},
        {"slide_encoder": "titan", "patch_encoder_ckpt_path": "/weights/uni.pth"},
        {"remove_holes": "true"},
        {"patch_size": True},
    ],
)
def test_invalid_options_are_rejected_before_execution(values):
    with pytest.raises(ValidationError):
        TridentOptions(**values)


def test_native_layout_matches_blca_and_preserves_fractional_mag(tmp_path):
    result = output_layout(TridentOptions(), tmp_path)
    assert result["featuresDir"] == str(tmp_path / "20x_256px_0px_overlap/features_uni_v1")
    assert result["coordinatePattern"].endswith("patches/{slide}_patches.h5")
    result = output_layout(TridentOptions(mag=2.5, slide_encoder="titan"), tmp_path)
    assert result["featuresDir"].endswith("2.5x_256px_0px_overlap/slide_features_titan")
    assert result["featureKind"] == "slide"


def test_gpfm_input_resolution_uses_fourteen_pixel_model_stride():
    assert TridentOptions(patch_encoder="gpfm", patch_encoder_img_size=238)
    with pytest.raises(ValidationError, match="multiple of 14"):
        TridentOptions(patch_encoder="gpfm", patch_encoder_img_size=240)


def test_catalog_and_generated_command_match_local_upstream_parser(tmp_path):
    # Optional integration check against the user's checkout; execute only its
    # argparse builder, with model registries replaced by inert dictionaries.
    root = Path(__file__).parents[1] / ".local/TRIDENT"
    source = root / "run_batch_of_slides.py"
    if not source.is_file():
        pytest.skip("A local TRIDENT checkout is not configured")
    from histopilot.adapters.trident.runtime import _parser_options

    runtime = discover_runtime(sys.executable, root)
    catalog = option_catalog()
    assert set(_parser_options(source)) == (
        {item["name"] for item in catalog["options"]} | set(catalog["managedOptions"])
    )
    assert set(runtime["patchEncoders"]) == set(catalog["patchEncoders"])
    assert set(runtime["slideEncoders"]) == set(catalog["slideEncoders"])
    tree = ast.parse(source.read_text())
    builder = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_parser"
    )
    namespace = {
        "argparse": argparse,
        "patch_encoder_registry": dict.fromkeys(runtime["patchEncoders"]),
        "slide_encoder_registry": dict.fromkeys(runtime["slideEncoders"]),
    }
    exec(compile(ast.Module(body=[builder], type_ignores=[]), str(source), "exec"), namespace)
    options = TridentOptions(
        gpus=[0, 1],
        remove_artifacts=True,
        dump_patches=True,
        dump_patches_format="jpg",
        mag=2.5,
        seg_batch_size=8,
        feat_batch_size=32,
    )
    command = build_command(
        options,
        python_path=sys.executable,
        trident_root=root,
        wsi_dir=tmp_path / "slides",
        job_dir=tmp_path / "output",
    )
    parsed = namespace["build_parser"]().parse_args(command[3:])
    for name, value in options.model_dump().items():
        assert getattr(parsed, name) == value


def test_runtime_and_plan_inspect_source_without_importing_models(tmp_path):
    root = fixture_checkout(tmp_path)
    before = set(sys.modules)
    runtime = discover_runtime(sys.executable, root)
    assert runtime["available"]
    assert runtime["patchEncoders"] == ["uni_v1"]
    command = build_command(
        TridentOptions(),
        python_path=sys.executable,
        trident_root=root,
        wsi_dir=tmp_path / "slides with spaces",
        job_dir=tmp_path / "output",
        custom_list_of_wsis=tmp_path / "selected.csv",
    )
    assert command[command.index("--mag") + 1] == "20"
    assert command[command.index("--wsi_dir") + 1].endswith("slides with spaces")
    assert "--custom_list_of_wsis" in command
    assert not any(name.split(".")[0] in {"torch", "trident"} for name in set(sys.modules) - before)


def test_old_runtime_refuses_new_options_and_unknown_encoder(tmp_path):
    root = fixture_checkout(tmp_path)
    for options, match in [
        (TridentOptions(dump_patches=True), "dump_patches"),
        (TridentOptions(patch_encoder="absent"), "Unknown patch encoder"),
        (TridentOptions(mag=2.5), "requires an integer"),
    ]:
        with pytest.raises(ValueError, match=match):
            build_command(
                options,
                python_path=sys.executable,
                trident_root=root,
                wsi_dir=tmp_path,
                job_dir=tmp_path / "out",
            )


def test_runtime_does_not_execute_arbitrary_editable_pth(tmp_path):
    interpreter = tmp_path / "env/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    site = tmp_path / "env/lib/python3.10/site-packages"
    site.mkdir(parents=True)
    (site / "trident.pth").write_text(
        "import pathlib; pathlib.Path('/tmp/do-not-run').touch()\n/missing/trident\n"
    )
    assert not discover_runtime(interpreter, tmp_path / "missing")["available"]


def make_plan(tmp_path, command):
    plan = {
        "command": command,
        "logPath": str(tmp_path / "worker.log"),
        "resultPath": str(tmp_path / "result.json"),
        "cancelPath": str(tmp_path / "cancel"),
        "processPath": str(tmp_path / "process.json"),
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    return path, plan


@pytest.mark.parametrize("exit_code", [0, 7])
def test_standalone_runner_records_real_exit_and_persistent_logs(tmp_path, exit_code):
    path, plan = make_plan(
        tmp_path, [sys.executable, "-c", f"print('fixture worker'); raise SystemExit({exit_code})"]
    )
    completed = subprocess.run([sys.executable, "-S", str(RUNNER), str(path)], timeout=10)
    assert completed.returncode == (0 if exit_code == 0 else 1)
    result = json.loads(Path(plan["resultPath"]).read_text())
    assert result["state"] == ("succeeded" if exit_code == 0 else "failed")
    assert result["exitCode"] == exit_code
    assert "fixture worker" in Path(plan["logPath"]).read_text()
    assert result["finishedAt"]


def test_precancelled_runner_never_launches_child(tmp_path):
    marker = tmp_path / "launched"
    path, plan = make_plan(tmp_path, [sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"])
    Path(plan["cancelPath"]).touch()
    subprocess.run([sys.executable, "-S", str(RUNNER), str(path)], timeout=10, check=True)
    assert not marker.exists()
    assert json.loads(Path(plan["resultPath"]).read_text())["state"] == "cancelled"


def test_runner_cancellation_terminates_process_group(tmp_path):
    ready = tmp_path / "ready"
    path, plan = make_plan(
        tmp_path,
        [
            sys.executable,
            "-c",
            f"import pathlib, os, time; pathlib.Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(90)",
        ],
    )
    runner = subprocess.Popen([sys.executable, "-S", str(RUNNER), str(path)])
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists()
        pid = int(ready.read_text())
        identity = json.loads(Path(plan["processPath"]).read_text())
        assert identity["pid"] == pid
        assert identity["startTicks"] > 0
        assert identity["bootId"] == Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        Path(plan["cancelPath"]).touch()
        runner.send_signal(signal.SIGHUP)
        runner.wait(timeout=15)
        result = json.loads(Path(plan["resultPath"]).read_text())
        assert result["state"] == "cancelled"
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        if runner.poll() is None:
            runner.kill()
            runner.wait()


@pytest.mark.parametrize("worker_exit, validator_exit", [(0, 0), (0, 3), (7, 0)])
def test_validation_runs_after_success_and_can_fail_the_job(tmp_path, worker_exit, validator_exit):
    marker = tmp_path / "validated"
    path, plan = make_plan(tmp_path, [sys.executable, "-c", f"raise SystemExit({worker_exit})"])
    plan["validationCommand"] = [
        sys.executable,
        "-c",
        f"open({str(marker)!r}, 'w').close(); raise SystemExit({validator_exit})",
    ]
    path.write_text(json.dumps(plan))
    subprocess.run([sys.executable, "-S", str(RUNNER), str(path)], timeout=10)
    result = json.loads(Path(plan["resultPath"]).read_text())
    assert marker.exists() == (worker_exit == 0)
    assert result["state"] == ("succeeded" if worker_exit == validator_exit == 0 else "failed")
    assert result["tridentExitCode"] == worker_exit
    if worker_exit == 0 and validator_exit:
        assert "Artifact validation" in result["error"]


def test_cancellation_between_stages_skips_validation(tmp_path):
    marker = tmp_path / "validated"
    path, plan = make_plan(
        tmp_path, [sys.executable, "-c", f"open({str(tmp_path / 'cancel')!r}, 'w').close()"]
    )
    plan["validationCommand"] = [sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"]
    path.write_text(json.dumps(plan))
    subprocess.run([sys.executable, "-S", str(RUNNER), str(path)], timeout=10, check=True)
    assert not marker.exists()
    assert json.loads(Path(plan["resultPath"]).read_text())["state"] == "cancelled"


@pytest.fixture
def sdpc_interpreter(tmp_path, monkeypatch):
    # Use a real, isolated interpreter to exercise the selected environment,
    # rather than whichever optional packages happen to be in the service.
    monkeypatch.delenv("PYTHONPATH", raising=False)
    environment = tmp_path / "trident environment"
    venv.EnvBuilder(with_pip=False).create(environment)
    python = environment / "bin/python"
    site = next((environment / "lib").glob("python*/site-packages"))
    return python, site


def test_sdpc_environment_preserves_parent_when_package_is_missing(sdpc_interpreter, monkeypatch):
    from histopilot.adapters.trident.runner import _worker_environment

    python, _site = sdpc_interpreter
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing/native:/another/native")
    original = os.environ.copy()
    worker = _worker_environment(str(python))
    assert all(worker[key] == value for key, value in original.items())
    assert all(worker[name] == original.get(name, "1") for name in THREAD_VARIABLES)
    assert dict(os.environ) == original


def test_sdpc_environment_leaves_missing_library_path_unset(sdpc_interpreter, monkeypatch):
    from histopilot.adapters.trident.runner import _worker_environment

    python, _site = sdpc_interpreter
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    assert "LD_LIBRARY_PATH" not in _worker_environment(str(python))


def test_sdpc_runner_discovers_selected_package_without_import_and_scopes_env(
    tmp_path, sdpc_interpreter, monkeypatch
):
    python, site = sdpc_interpreter
    package = site / "opensdpc"
    native = package / "LINUX"
    ffmpeg = native / "ffmpeg"
    ffmpeg.mkdir(parents=True)
    # Importing the package in the probing process would fail; find_spec must
    # still locate its native libraries before starting the actual worker.
    (package / "__init__.py").write_text("raise RuntimeError('do not import during discovery')\n")
    old_libraries = "/custom/native:/other/native"
    monkeypatch.setenv("LD_LIBRARY_PATH", old_libraries)
    monkeypatch.setenv("SDPC_WORKER_ENV_FIXTURE", "preserved")
    worker_env = tmp_path / "worker-env.json"
    validation_env = tmp_path / "validation-env.json"

    def write_environment(output):
        return (
            "import json, os, pathlib; "
            f"pathlib.Path({str(output)!r}).write_text(json.dumps(dict(os.environ)))"
        )

    path, plan = make_plan(tmp_path, [str(python), "-c", write_environment(worker_env)])
    plan["validationCommand"] = [sys.executable, "-c", write_environment(validation_env)]
    path.write_text(json.dumps(plan))
    subprocess.run([sys.executable, "-S", str(RUNNER), str(path)], timeout=15, check=True)
    worker = json.loads(worker_env.read_text())
    validator = json.loads(validation_env.read_text())
    assert worker["LD_LIBRARY_PATH"] == os.pathsep.join([str(native), str(ffmpeg), old_libraries])
    assert worker["SDPC_WORKER_ENV_FIXTURE"] == "preserved"
    assert validator["LD_LIBRARY_PATH"] == old_libraries
    assert os.environ["LD_LIBRARY_PATH"] == old_libraries
    assert json.loads(Path(plan["resultPath"]).read_text())["state"] == "succeeded"


def test_sdpc_environment_ignores_missing_bundled_directories(sdpc_interpreter):
    from histopilot.adapters.trident.runner import _worker_environment

    python, site = sdpc_interpreter
    package = site / "opensdpc"
    package.mkdir()
    (package / "__init__.py").write_text("raise RuntimeError('do not import during discovery')\n")
    worker = _worker_environment(str(python))
    assert all(worker[key] == value for key, value in os.environ.items())
    assert all(worker[name] == os.environ.get(name, "1") for name in THREAD_VARIABLES)


@pytest.mark.parametrize("unbuffered", [False, True])
def test_worker_bootstrap_keeps_original_argv_and_paths(tmp_path, unbuffered):
    from histopilot.adapters.trident.runner import _worker_command

    script = tmp_path / "TRIDENT checkout" / "run_batch_of_slides.py"
    script.parent.mkdir()
    script.write_text("# fixture")
    command = [sys.executable, *(["-u"] if unbuffered else []), str(script), "--job_dir", "/output path"]
    original = list(command)
    assert _worker_command(command) == [
        sys.executable, "-u", str(RUNNER.with_name("bootstrap.py")),
        str(script), "--job_dir", "/output path",
    ]
    assert command == original


@pytest.mark.parametrize("arguments", [["-c", "print('fixture')"], ["-m", "fixture"], ["other.py"], ["missing/run_batch_of_slides.py"]])
def test_worker_bootstrap_preserves_historical_nontrident_commands(arguments):
    from histopilot.adapters.trident.runner import _worker_command

    command = [sys.executable, *arguments]
    assert _worker_command(command) == command


def test_native_thread_defaults_apply_only_to_trident_child(tmp_path, monkeypatch):
    for name in THREAD_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "3")
    worker_path = tmp_path / "worker.json"
    validator_path = tmp_path / "validator.json"

    def snapshot(output):
        return [sys.executable, "-c", (
            "import json, os, pathlib; "
            f"pathlib.Path({str(output)!r}).write_text(json.dumps(dict(os.environ)))"
        )]

    plan_path, plan = make_plan(tmp_path, snapshot(worker_path))
    plan["validationCommand"] = snapshot(validator_path)
    plan_path.write_text(json.dumps(plan))
    subprocess.run([sys.executable, "-S", str(RUNNER), str(plan_path)], timeout=15, check=True)
    worker = json.loads(worker_path.read_text())
    validator = json.loads(validator_path.read_text())
    assert {name: worker[name] for name in THREAD_VARIABLES} == {
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "3", "NUMEXPR_NUM_THREADS": "1",
    }
    assert validator["OPENBLAS_NUM_THREADS"] == "3"
    for name in set(THREAD_VARIABLES) - {"OPENBLAS_NUM_THREADS"}:
        assert name not in validator and name not in os.environ
