"""Task Center Phase 3: extraction (+ validation), packing and archives as tasks.

Everything runs against the per-test Task Center store and private TMPDIR
(tests/conftest.py) with in-process runners; no tmux session, GPU or real TRIDENT run is
ever used. Registries touched here are the per-session temp ones, never /tmp's.
"""

import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pytest

from histopilot.adapters import trident
from histopilot.adapters.trident import performance
from histopilot.application.extractions import ExtractionService
from histopilot.application.feature_packs import FeaturePackService
from histopilot.application.features import FeatureService
from histopilot.schemas.extractions import ExtractionSpec
from histopilot.schemas.feature_packs import FeaturePackSpec
from histopilot.schemas.features import FeatureSpec
from histopilot.storage.filesystem import LocalFilesystem
from histopilot.storage.scientific import ScientificStore
from histopilot.taskcenter import ids, procs
from histopilot.taskcenter.adapters.base import RunnerContext
from histopilot.taskcenter.adapters.extraction import (
    ExtractionAdapter,
    ExtractionValidationAdapter,
)
from histopilot.taskcenter.adapters.packing import PackingAdapter, reap_staging
from histopilot.taskcenter.client import default_client
from histopilot.taskcenter.model import TERMINAL, utc_now_iso
from histopilot.taskcenter.runner import Runner
from histopilot.workers import packing_process
from histopilot.workers.resource_reservation import SHARED_RUNS_PER_GPU, preparation_resources

RUNNER = Path(trident.__file__).with_name("runner.py")


def load_runner_module():
    spec = importlib.util.spec_from_file_location("trident_runner_under_test", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_host():
    return {
        "cpuCount": 8,
        "totalRamGb": 64.0,
        "availableRamGb": 60.0,
        "bootId": "test",
        "kernel": "test",
        "physicalCpuCount": 4,
        "gpus": [],
    }


class Center:
    """The per-test Task Center (the one default_client() resolves) plus runners."""

    def __init__(self):
        self.client = default_client()
        self.store = self.client.store
        self.runners = []
        self.logs = []

    def runner(self):
        runner = Runner(
            self.store,
            host_probe=fake_host,
            sample_interval=0.0,
            host_interval=0.0,
            log=self.logs.append,
        )
        self.runners.append(runner)
        runner.start()
        return runner

    def state(self, task_id):
        return self.store.get(task_id)["state"]

    def tick_until(self, runner, predicate, timeout=60.0):
        deadline = time.monotonic() + timeout
        while True:
            runner.tick()
            if predicate():
                return
            if time.monotonic() > deadline:
                states = {task["id"]: task["state"] for task in self.store.list(limit=None)}
                raise AssertionError(f"Condition not reached; tasks: {states}; log: {self.logs}")
            time.sleep(0.05)

    def context(self):
        return RunnerContext(
            store=self.store,
            now=utc_now_iso,
            settings=self.store.settings(),
            host=fake_host(),
            log=self.logs.append,
        )

    def cleanup(self):
        for runner in self.runners:
            runner.close()
        for task in self.store.list(limit=None):
            if task["process"]:
                procs.kill_group(task["process"])
        for runner in self.runners:
            for child in runner._procs.values():
                try:
                    child.kill()
                    child.wait(timeout=5)
                except (OSError, subprocess.SubprocessError):
                    pass


@pytest.fixture
def center():
    value = Center()
    yield value
    value.cleanup()


# -- 1.15: GPU sharing -------------------------------------------------------------------


def test_extraction_leases_share_the_gpu_with_a_vram_estimate():
    resources = preparation_resources("extraction", {"gpu": 0, "segmenter": "hest"})
    assert resources["runsPerGpu"] == SHARED_RUNS_PER_GPU > 1
    assert resources["vramGb"] == performance.estimate_vram_gb({"gpu": 0, "segmenter": "hest"})
    assert preparation_resources("extraction", {"gpu": -1})["vramGb"] == 0.0


def test_vram_estimate_scales_with_stages_and_batches():
    hest = performance.estimate_vram_gb({"task": "seg", "segmenter": "hest"})
    assert 9.0 <= hest <= 10.0
    assert performance.estimate_vram_gb({"task": "seg", "segmenter": "otsu"}) == 2.0
    doubled = performance.estimate_vram_gb({"task": "seg", "seg_batch_size": 128})
    assert doubled == pytest.approx(2 * hest, rel=0.01)
    small = performance.estimate_vram_gb({"task": "feat", "patch_encoder": "uni_v1"})
    large = performance.estimate_vram_gb({"task": "feat", "patch_encoder": "uni_v2"})
    assert small < large
    assert performance.estimate_vram_gb({"gpu": -1}) == 0.0
    assert performance.workload_key({"task": "all"}) != performance.workload_key({"task": "seg"})


def test_managed_preparation_takes_no_lease(monkeypatch, tmp_path):
    from histopilot.workers.resource_reservation import reserve_preparation

    monkeypatch.setenv("HISTOPILOT_TASK_MANAGED", "1")
    registry = Path(os.environ["TMPDIR"]) / f"histopilot-training-{os.getuid()}"
    before = set(registry.glob("lease-*.json")) if registry.exists() else set()
    with reserve_preparation(tmp_path, "extraction", preparation_resources("extraction"), bool):
        after = set(registry.glob("lease-*.json")) if registry.exists() else set()
    assert after == before
    assert not (tmp_path / "resources.json").exists()


# -- 1.15: dead TRIDENT locks --------------------------------------------------------------


def _lock(path: Path, **payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_name(path.name + ".lock").write_text(json.dumps(payload) if payload else "")


def test_dead_lock_sweep_clears_dead_writers_and_keeps_live_ones(tmp_path):
    module = load_runner_module()
    root = tmp_path / "output"
    host = socket.gethostname()
    finished = subprocess.Popen([sys.executable, "-c", "pass"])
    finished.wait()
    now = time.time()
    # A dead writer that left a (possibly truncated) output: lock gone, output set aside.
    dead = root / "contours" / "dead.jpg"
    dead.parent.mkdir(parents=True)
    dead.write_bytes(b"partial")
    _lock(dead, pid=finished.pid, hostname=host, created_at=now - 60)
    # A live writer: this test process, which started before its lock was written.
    live = root / "contours" / "live.jpg"
    _lock(live, pid=os.getpid(), hostname=host, created_at=now)
    # A reused PID: this process started after the lock was written.
    reused = root / "features" / "reused.h5"
    _lock(reused, pid=os.getpid(), hostname=host, created_at=now - 10 * 365 * 86400)
    # Unknown owners fall back to the age threshold.
    old = root / "features" / "old.h5"
    _lock(old)
    os.utime(old.with_name("old.h5.lock"), (now - 48 * 3600, now - 48 * 3600))
    young = root / "features" / "young.h5"
    _lock(young, pid=123, hostname="another-host", created_at=now - 60)
    # A live writer whose process start reads days after its lock: the clock was stepped
    # after the host slept (btime moved forward). Its lock is never touched.
    stepped = root / "features" / "stepped.h5"
    stepped.write_bytes(b"being written")
    _lock(stepped, pid=os.getpid(), hostname=host, created_at=now - 3 * 86400)
    stats = module.clear_dead_locks(str(root), max_age_hours=24)
    assert stats["scanned"] == 6
    assert stats["removed"] == 3 and stats["kept"] == 3 and stats["movedAside"] == 1
    assert stepped.with_name("stepped.h5.lock").exists() and stepped.exists()
    assert not dead.with_name("dead.jpg.lock").exists() and not dead.exists()
    assert [path.name for path in dead.parent.glob("dead.jpg.stale-*")]
    assert live.with_name("live.jpg.lock").exists()
    assert not reused.with_name("reused.h5.lock").exists()
    assert not old.with_name("old.h5.lock").exists()
    assert young.with_name("young.h5.lock").exists()


def _plan(tmp_path, command, **extra):
    plan = {
        "command": command,
        "logPath": str(tmp_path / "worker.log"),
        "resultPath": str(tmp_path / "result.json"),
        "cancelPath": str(tmp_path / "cancelled"),
        "processPath": str(tmp_path / "process.json"),
        **extra,
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    return path, plan


def test_runner_clears_dead_locks_before_starting_trident(tmp_path):
    root = tmp_path / "output"
    finished = subprocess.Popen([sys.executable, "-c", "pass"])
    finished.wait()
    _lock(
        root / "contours" / "slide.jpg",
        pid=finished.pid,
        hostname=socket.gethostname(),
        created_at=time.time() - 5,
    )
    check = (
        f"import os, sys; sys.exit(os.path.exists({str(root / 'contours' / 'slide.jpg.lock')!r}))"
    )
    path, plan = _plan(
        tmp_path,
        [sys.executable, "-c", check],
        clearDeadLocks={"root": str(root), "maxAgeHours": 24},
    )
    completed = subprocess.run([sys.executable, "-S", str(RUNNER), str(path)], timeout=30)
    assert completed.returncode == 0
    result = json.loads(Path(plan["resultPath"]).read_text())
    assert result["state"] == "succeeded"
    assert result["deadLocks"]["removed"] == 1
    assert "Dead TRIDENT locks: removed=1" in Path(plan["logPath"]).read_text()


def _wait_for(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        time.sleep(0.02)


@pytest.mark.parametrize("cancel", [False, True])
def test_managed_runner_keeps_trident_in_its_group_and_never_mistakes_a_signal_for_cancel(
    tmp_path, cancel
):
    ready = tmp_path / "ready"
    code = (
        "import os, pathlib, time; "
        f"pathlib.Path({str(ready)!r}).write_text(f'{{os.getpid()}} {{os.getpgid(0)}}'); "
        "time.sleep(60)"
    )
    path, plan = _plan(tmp_path, [sys.executable, "-c", code], managed=True)
    env = {**os.environ, "HISTOPILOT_TASK_ID": "task-x", "HISTOPILOT_TASK_ATTEMPT": "3"}
    runner = subprocess.Popen(
        [sys.executable, "-S", str(RUNNER), str(path)], env=env, start_new_session=True
    )
    try:
        _wait_for(ready.exists)
        _wait_for(lambda: len(ready.read_text().split()) == 2)
        pid, group = (int(value) for value in ready.read_text().split())
        assert group == runner.pid  # the task's process group, not a private session
        if cancel:
            Path(plan["cancelPath"]).write_text("{}")
        runner.send_signal(signal.SIGTERM)
        runner.wait(timeout=20)
        result = json.loads(Path(plan["resultPath"]).read_text())
        assert result["state"] == ("cancelled" if cancel else "interrupted")
        assert result["taskId"] == "task-x" and result["taskAttempt"] == 3
        assert result["cancelRequested"] is cancel
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        if runner.poll() is None:
            os.killpg(runner.pid, signal.SIGKILL)
            runner.wait()


def test_managed_runner_writes_progress_for_stall_detection(tmp_path):
    job = {
        "id": "extraction-" + "b" * 32,
        "spec": {"options": {"task": "seg"}},
        "slideCount": 2,
        "createdAt": utc_now_iso(),
    }
    (tmp_path / "job.json").write_text(json.dumps(job))
    bar = "Segmenting tissue:  50%|#####     | 1/2 [00:19<00:19, 19.05s/it]"
    code = f"import sys, time; print({bar!r}, flush=True); time.sleep(0.5)"
    peak = tmp_path / "peak.json"
    peak.write_text(json.dumps({"cudaPeakReservedBytes": 3 * 2**30}))
    path, plan = _plan(
        tmp_path,
        [sys.executable, "-c", code],
        managed=True,
        jobPath=str(tmp_path / "job.json"),
        progressPath=str(tmp_path / "progress.json"),
        peakPath=str(peak),
    )
    completed = subprocess.run([sys.executable, "-S", str(RUNNER), str(path)], timeout=30)
    assert completed.returncode == 0
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["stage"] == "segmentation"
    assert progress["completed"] == 1 and progress["total"] == 2
    assert progress["completedSlides"] == 1 and progress["totalSlides"] == 2
    assert progress["phase"] == progress["label"]
    assert progress["cudaPeakReservedBytes"] == 3 * 2**30
    result = json.loads(Path(plan["resultPath"]).read_text())
    assert result["cudaPeakReservedBytes"] == 3 * 2**30


# -- 3.1 / 3.2: extraction and validation tasks ---------------------------------------------


TRIDENT_FAKE = """
import json, pathlib, sys, time
out = pathlib.Path(sys.argv[1]) / "contours_geojson"
out.mkdir(parents=True, exist_ok=True)
print("Segmenting tissue: 100%|##########| 2/2 [00:01<00:00, 1.00it/s]", flush=True)
for name in sys.argv[2:]:
    (out / f"{name}.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": []}))
"""


@pytest.fixture
def extraction(tmp_path, monkeypatch, center):
    folder = tmp_path / "experiment"
    folder.mkdir()
    roots = [tmp_path / "drive-d", tmp_path / "oceanpath-hot"]
    slides = []
    for number, root in enumerate(roots):
        root.mkdir()
        slide = root / f"slide{number}.svs"
        slide.write_bytes(b"fixture slide")
        slides.append({"slideId": slide.stem, "patientId": f"p{number}", "slidePath": str(slide)})
    store = ScientificStore(folder, "project-extraction-tasks")
    draft = store.create_draft("import", "test", {})
    dataset = store.publish_dataset(
        draft["id"],
        expected_revision=1,
        manifest={"kind": "dataset"},
        artifacts={"records.json": json.dumps(slides).encode()},
        operation_id="dataset",
    )
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    monkeypatch.setattr(
        "histopilot.adapters.trident.discover_runtime",
        lambda: {"available": True, "pythonPath": sys.executable, "tridentRoot": str(runtime_root)},
    )
    monkeypatch.setattr(
        "histopilot.adapters.trident.probe_reader_modules",
        lambda _python, modules, **_: {"modules": dict.fromkeys(modules)},
    )
    commands = []

    def command(options, **kwargs):
        commands.append(options)
        return [
            sys.executable,
            "-c",
            kwargs.get("script", TRIDENT_FAKE),
            kwargs["job_dir"],
            "slide0",
            "slide1",
        ]

    monkeypatch.setattr("histopilot.adapters.trident.build_command", command)
    service = ExtractionService(
        store,
        LocalFilesystem(tuple(roots)),
        execution_mode="task-center",
        task_center=center.client,
    )
    spec = ExtractionSpec(
        datasetId=dataset["id"],
        outputPath=str(folder / "trident"),
        options={"task": "seg", "gpu": -1, "max_workers": 1},
    )
    return service, spec, commands


def submit_extraction(service, spec, operation="run"):
    preview = service.preview(spec)
    assert preview["canRun"], preview["findings"]
    return service.submit(spec, preview["previewHash"], operation)


def test_extraction_submits_two_tasks_with_the_published_contract(extraction, center):
    service, spec, _commands = extraction
    job = submit_extraction(service, spec)
    assert job["state"] == "queued" and job["executor"] == "task-center"
    assert job["sessionName"] is None
    folder = service.folder / job["id"]
    extraction_task = center.store.get(ids.task_id("extraction", str(folder)))
    validation_task = center.store.get(ids.task_id("extraction-validation", str(folder)))
    assert job["task"]["id"] == extraction_task["id"] == job["tasks"]["extraction"]["id"]
    assert job["tasks"]["validation"]["id"] == validation_task["id"]
    assert extraction_task["kind"] == "extraction" and extraction_task["adapter"] == "extraction"
    assert extraction_task["request"]["lane"] == "cpu"  # every device is -1
    assert extraction_task["request"]["graceSeconds"] == 30
    assert extraction_task["exclusiveKey"].startswith("trident-output:")
    assert extraction_task["labels"] == {
        "recordKind": "extraction",
        "recordId": job["id"],
        "projectId": service.store.project_id,
        "extractionId": job["id"],
        "encoder": "uni_v1",
        "slideCount": 2,
        "phase": "extraction",
    }
    assert extraction_task["command"]["env"]["HISTOPILOT_TASK_MANAGED"] == "1"
    assert validation_task["state"] == "blocked"
    assert validation_task["request"]["lane"] == "cpu"
    assert validation_task["labels"]["phase"] == "validation"
    owner = center.store.owner(extraction_task["ownerKey"])
    assert owner["kind"] == "extraction" and owner["id"] == job["id"]
    assert job["ownerKey"] == extraction_task["ownerKey"]
    plan = json.loads((folder / "plan.json").read_text())
    assert plan["managed"] is True
    assert "resources" not in plan and "validationCommand" not in plan
    assert plan["clearDeadLocks"] == {"root": spec.outputPath, "maxAgeHours": 24.0}


def test_gpu_extraction_requests_vram_and_runs_on_the_admitted_device(extraction, center):
    service, spec, commands = extraction
    spec = spec.model_copy(update={"options": {"task": "seg", "gpus": [0, 1], "max_workers": 1}})
    preview = service.preview(spec)
    assert {item["code"] for item in preview["findings"]} == {"SINGLE_GPU_TASK"}
    job = service.submit(spec, preview["previewHash"], "gpu")
    task = center.store.get(job["taskId"])
    assert task["request"]["lane"] == "gpu"
    assert task["request"]["vramGb"] == performance.estimate_vram_gb(spec.options)
    assert task["request"]["workloadKey"].startswith("trident:")
    # The runner exposes the admitted GPU as device 0; the saved spec keeps the request.
    assert commands[-1].gpu == 0 and commands[-1].gpus is None
    assert job["spec"]["options"]["gpus"] == [0, 1]


def test_extraction_and_validation_run_through_the_runner(extraction, center):
    service, spec, _commands = extraction
    job = submit_extraction(service, spec)
    runner = center.runner()
    center.tick_until(runner, lambda: center.state(job["validationTaskId"]) in TERMINAL)
    assert center.state(job["taskId"]) == "succeeded"
    assert center.state(job["validationTaskId"]) == "succeeded"
    shown = service.get(job["id"], logs=True)
    assert shown["state"] == "succeeded", shown
    assert shown["result"]["completedSlides"] == 2
    assert shown["result"]["taskAttempt"] == 1
    assert "Starting Artifact validation worker" in shown["logs"]
    assert shown["progress"]["label"] == "Extraction complete"
    folder = service.folder / job["id"]
    report = json.loads((folder / "validation.json").read_text())
    assert report["complete"] is True and report["taskId"] == job["validationTaskId"]
    progress = json.loads((folder / "validation-progress.json").read_text())
    assert progress["completedSlides"] == progress["totalSlides"] == 2


def test_incomplete_outputs_fail_validation_and_a_retry_rearms_it(extraction, center):
    service, spec, _commands = extraction
    job = submit_extraction(service, spec)
    folder = service.folder / job["id"]
    plan = json.loads((folder / "plan.json").read_text())
    plan["command"] = plan["command"][:-1]  # TRIDENT "forgets" slide1
    (folder / "plan.json").write_text(json.dumps(plan))
    runner = center.runner()
    center.tick_until(runner, lambda: center.state(job["validationTaskId"]) in TERMINAL)
    assert center.state(job["taskId"]) == "succeeded"
    assert center.state(job["validationTaskId"]) == "failed"
    shown = service.get(job["id"])
    assert shown["state"] == "failed"
    assert "1 slides lack valid outputs" in shown["error"]
    # Resuming runs TRIDENT again and then a fresh validation after it.
    plan["command"].append("slide1")
    (folder / "plan.json").write_text(json.dumps(plan))
    resumed = service.resume(job["id"])
    assert resumed["state"] == "queued"
    assert center.store.get(job["taskId"])["attempt"] == 2
    center.tick_until(runner, lambda: center.store.get(job["validationTaskId"])["attempt"] == 2)
    center.tick_until(runner, lambda: center.state(job["validationTaskId"]) in TERMINAL)
    assert center.state(job["validationTaskId"]) == "succeeded"
    assert service.get(job["id"])["state"] == "succeeded"


SLEEPER = (
    "import time; print('Segmenting tissue:   0%|          | 0/2', flush=True); time.sleep(60)"
)


def _running_extraction(service, spec, center, monkeypatch):
    original = trident.build_command
    monkeypatch.setattr(
        "histopilot.adapters.trident.build_command",
        lambda options, **kwargs: original(options, **{**kwargs, "script": SLEEPER}),
    )
    job = submit_extraction(service, spec)
    runner = center.runner()
    center.tick_until(runner, lambda: center.state(job["taskId"]) == "running")
    folder = service.folder / job["id"]
    _wait_for(lambda: (folder / "process.json").exists())
    _wait_for(lambda: "phase" in json.loads((folder / "process.json").read_text()))
    return job, runner, folder


def test_stage_cancel_stops_trident_and_records_cancelled(extraction, center, monkeypatch):
    service, spec, _commands = extraction
    job, runner, folder = _running_extraction(service, spec, center, monkeypatch)
    child = json.loads((folder / "process.json").read_text())["pid"]
    assert service.get(job["id"])["state"] == "running"
    cancelled = service.cancel(job["id"])
    assert cancelled["state"] in {"cancelling", "cancelled"}
    marker = json.loads((folder / "cancelled").read_text())
    assert marker["attempts"][job["taskId"]] == 1
    center.tick_until(runner, lambda: center.state(job["taskId"]) in TERMINAL)
    assert center.state(job["taskId"]) == "cancelled"
    assert center.state(job["validationTaskId"]) == "cancelled"
    assert service.get(job["id"])["state"] == "cancelled"
    _wait_for(lambda: _identity(child) is None)  # TRIDENT shared the task's group


def _identity(pid):
    try:
        return procs.identity(pid)
    except (OSError, ValueError):
        return None


def test_host_signal_is_an_interruption_not_a_cancel(extraction, center, monkeypatch):
    service, spec, _commands = extraction
    job, runner, _folder = _running_extraction(service, spec, center, monkeypatch)
    task = center.store.get(job["taskId"])
    os.kill(task["process"]["pid"], signal.SIGHUP)  # a closed terminal, not a cancel
    center.tick_until(runner, lambda: center.state(job["taskId"]) in TERMINAL)
    assert center.state(job["taskId"]) == "interrupted"
    shown = service.get(job["id"])
    assert shown["state"] == "interrupted"
    assert ExtractionAdapter().can_requeue(center.store.get(job["taskId"]), center.context())


def test_task_center_cancel_then_retry_is_not_blocked_by_the_old_marker(extraction, center):
    service, spec, _commands = extraction
    job = submit_extraction(service, spec)
    service.cancel(job["id"])  # both tasks were pending: cancelled at once
    assert service.get(job["id"])["state"] == "cancelled"
    folder = service.folder / job["id"]
    assert (folder / "cancelled").exists()
    for task_id in (job["taskId"], job["validationTaskId"]):
        assert center.client.requeue_task(task_id, reason="retry")
    runner = center.runner()
    center.tick_until(runner, lambda: center.state(job["validationTaskId"]) in TERMINAL)
    assert center.state(job["taskId"]) == "succeeded"
    assert not (folder / "cancelled").exists()
    assert service.get(job["id"])["state"] == "succeeded"


def test_overlapping_outputs_never_run_together_through_resume_or_retry(extraction, center):
    """The exclusive key serializes identical folders only; a resumed or retried job must
    not sweep the locks of a live job writing a nested folder."""
    from histopilot.storage.project_lock import StorageError
    from histopilot.taskcenter.adapters.base import AdapterError

    service, spec, _commands = extraction
    outer = submit_extraction(service, spec, "outer")
    service.cancel(outer["id"])
    assert service.get(outer["id"])["state"] == "cancelled"
    nested = spec.model_copy(update={"outputPath": str(Path(spec.outputPath) / "nested")})
    inner = submit_extraction(service, nested, "inner")
    assert service.get(inner["id"])["state"] == "queued"
    with pytest.raises(StorageError) as busy:
        service.resume(outer["id"])
    assert busy.value.code == "OUTPUT_BUSY"
    # A Task Center retry bypasses the stage: the adapter waits while the other one runs.
    assert center.store.transition(inner["taskId"], from_states=("queued",), to_state="running")
    assert center.client.requeue_task(outer["taskId"], reason="retry")
    retried = center.store.get(outer["taskId"])
    with pytest.raises(AdapterError) as waiting:
        ExtractionAdapter().prepare(retried, center.context())
    assert waiting.value.transient and "overlapping output folder" in str(waiting.value)
    assert center.store.transition(inner["taskId"], from_states=("running",), to_state="failed")
    assert ExtractionAdapter().prepare(retried, center.context()) is None


def test_extraction_adapter_classifies_receipts_by_attempt(extraction, center, tmp_path):
    service, spec, _commands = extraction
    job = submit_extraction(service, spec)
    folder = service.folder / job["id"]
    task = {**center.store.get(job["taskId"]), "attempt": 2}
    adapter, ctx = ExtractionAdapter(), center.context()
    exit = {"returncode": 0, "lost": False, "stopReason": None}

    def receipt(**values):
        (folder / "result.json").write_text(
            json.dumps({"taskId": task["id"], "taskAttempt": 2, **values})
        )

    receipt(state="succeeded", cudaPeakReservedBytes=4 * 2**30)
    done = adapter.on_exit(task, exit, ctx)
    assert done["state"] == "succeeded" and done["measurement"] == {"peakVramGb": 4.0}
    receipt(state="succeeded", taskAttempt=1)
    (folder / "result.json").write_text(
        json.dumps({"taskId": task["id"], "taskAttempt": 1, "state": "succeeded"})
    )
    assert adapter.on_exit(task, exit, ctx)["state"] == "interrupted"  # an older attempt's
    receipt(state="cancelled", cancelRequested=False)
    assert adapter.on_exit(task, exit, ctx)["state"] == "interrupted"
    assert adapter.on_exit(task, {**exit, "stopReason": "cancel"}, ctx)["state"] == "cancelled"
    receipt(state="failed", error="TRIDENT exited with status 1")
    with open(folder / "worker.log", "a") as log:
        log.write("[t] Starting TRIDENT worker\ntorch.OutOfMemoryError: CUDA out of memory.\n")
    assert adapter.on_exit(task, exit, ctx)["exitReason"] == "oom"
    with open(folder / "worker.log", "a") as log:
        log.write("[t] Starting TRIDENT worker\nValueError: bad slide\n")
    assert adapter.on_exit(task, exit, ctx)["exitReason"] == "error"


def test_extraction_prepare_waits_for_disk_and_refines_vram(extraction, center, monkeypatch):
    service, spec, _commands = extraction
    job = submit_extraction(service, spec)
    task = center.store.get(job["taskId"])
    task = {**task, "request": {**task["request"], "lane": "gpu", "vramGb": 10.0}}
    adapter, ctx = ExtractionAdapter(), center.context()
    usage = type("Usage", (), {"free": 1024})
    monkeypatch.setattr(
        "histopilot.taskcenter.adapters.extraction.shutil.disk_usage", lambda _p: usage
    )
    from histopilot.taskcenter.adapters.base import AdapterError

    with pytest.raises(AdapterError) as waiting:
        adapter.prepare(task, ctx)
    assert waiting.value.transient and "Waiting for disk space" in str(waiting.value)
    usage.free = 500 * 2**30
    assert adapter.prepare(task, ctx) is None  # nothing measured yet: keep the estimate
    center.store.record_measurement(
        {
            "taskId": "earlier",
            "attempt": 1,
            "kind": "extraction",
            "lane": "gpu",
            "workloadKey": task["request"]["workloadKey"],
            "peakVramGb": 6.0,
            "peakPrivateRamGb": 3.0,
            "exitReason": "ok",
        }
    )
    refined = adapter.prepare(task, center.context())["request"]
    assert refined["vramGb"] == pytest.approx(6.0 * 1.15 + 0.3)
    assert refined["ramGb"] == pytest.approx(3.5)


def test_validation_adapter_needs_its_own_complete_report(extraction, center):
    service, spec, _commands = extraction
    job = submit_extraction(service, spec)
    folder = service.folder / job["id"]
    task = center.store.get(job["validationTaskId"])
    adapter, ctx = ExtractionValidationAdapter(), center.context()
    exit = {"returncode": 0, "lost": False, "stopReason": None}
    assert adapter.on_exit(task, exit, ctx)["state"] == "interrupted"
    report = {"taskId": task["id"], "taskAttempt": 1, "complete": True}
    (folder / "validation.json").write_text(json.dumps(report))
    assert adapter.on_exit(task, exit, ctx)["state"] == "succeeded"
    report.update(complete=False, missingSlides=1, unvalidatedSlides=0)
    (folder / "validation.json").write_text(json.dumps(report))
    failed = adapter.on_exit(task, exit, ctx)
    assert failed["state"] == "failed" and "1 slides lack valid outputs" in failed["error"]


# -- 3.7: preflight ---------------------------------------------------------------------------


def test_preview_blocks_missing_slide_readers(extraction, monkeypatch):
    service, spec, _commands = extraction
    monkeypatch.setattr(
        "histopilot.adapters.trident.probe_reader_modules",
        lambda _python, modules, **_: {
            "modules": {
                name: "ModuleNotFoundError: No module named 'openslide'" for name in modules
            }
        },
    )
    preview = service.preview(spec)
    assert not preview["canRun"]
    finding = next(
        item for item in preview["findings"] if item["code"] == "SLIDE_READER_UNAVAILABLE"
    )
    assert "openslide" in finding["message"] and "2 selected slide" in finding["message"]
    monkeypatch.setattr(
        "histopilot.adapters.trident.probe_reader_modules",
        lambda *_args, **_kwargs: {"error": "Cannot check slide reader dependencies: timeout"},
    )
    preview = service.preview(spec)
    assert preview["canRun"]
    assert [item["severity"] for item in preview["findings"]] == ["warning"]


def test_slide_reader_map_follows_trident():
    assert trident.slide_readers(["a.sdpc", "b.SVS", "c.png", "d.czi", "e.zarr"]) == {
        "sdpc": 1,
        "openslide": 1,
        "image": 1,
        "czi": 1,
        "omezarr": 1,
    }
    assert trident.slide_readers(["a.svs"], "cucim") == {"cucim": 1}
    assert trident.READER_MODULES["sdpc"] == ("opensdpc",)


def test_preview_estimates_disk_space(extraction, monkeypatch):
    service, spec, _commands = extraction
    usage = type("Usage", (), {"free": 10 * 2**40})
    monkeypatch.setattr("histopilot.application.extractions.shutil.disk_usage", lambda _p: usage)
    preview = service.preview(spec)
    assert preview["estimatedBytes"] > 0 and preview["availableBytes"] == usage.free
    first = preview["previewHash"]
    usage.free = 10 * 2**40 - 12345  # free space moves; the preview stays valid
    assert service.preview(spec)["previewHash"] == first
    usage.free = 1024
    preview = service.preview(spec)
    assert not preview["canRun"]
    assert [item["code"] for item in preview["findings"]] == ["INSUFFICIENT_SPACE"]
    estimate = performance.estimate_output_bytes(
        {"task": "feat", "patch_encoder": "uni_v1"}, 100 * 2**30, 10
    )
    assert 20 * 2**30 < estimate < 30 * 2**30


def test_missing_trident_source_names_the_fix(tmp_path, monkeypatch):
    monkeypatch.delenv("HISTOPILOT_TRIDENT_ROOT", raising=False)
    runtime = trident.discover_runtime(sys.executable, tmp_path / "nowhere")
    assert not runtime["available"]
    assert "HISTOPILOT_TRIDENT_ROOT" in runtime["message"]
    assert ".local/TRIDENT" in runtime["message"]
    assert runtime["searchedRoots"] == [str(tmp_path / "nowhere")]


# -- 3.3: packing tasks -----------------------------------------------------------------------


@pytest.fixture
def packing(tmp_path, center):
    project = tmp_path / "project"
    project.mkdir()
    store = ScientificStore(project, "project-packing-tasks")
    draft = store.create_draft("import", "fixture", {})
    dataset = store.publish_dataset(
        draft["id"],
        expected_revision=1,
        manifest={"kind": "dataset"},
        artifacts={
            "records.json": json.dumps(
                [{"slideId": "001", "patientId": "p1"}, {"slideId": "002", "patientId": "p2"}]
            ).encode()
        },
        operation_id="dataset",
    )
    source = tmp_path / "source"
    source.mkdir()
    for slide in ("001", "002"):
        with h5py.File(source / f"{slide}.h5", "w") as handle:
            handle.create_dataset("features", data=np.arange(12, dtype="float32").reshape(3, 4))
            handle.create_dataset("coords", data=np.arange(6, dtype="int64").reshape(3, 2))
    filesystem = LocalFilesystem((tmp_path,))
    features = FeatureService(store, filesystem)
    feature_spec = FeatureSpec(datasetId=dataset["id"], path=str(source))
    feature = features.freeze(feature_spec, features.preview(feature_spec)["previewHash"], "f")
    service = FeaturePackService(
        store, filesystem, execution_mode="task-center", task_center=center.client
    )
    return service, feature["id"]


def submit_pack(service, spec, operation="op"):
    preview = service.preview(spec)
    assert preview["canRun"], preview["findings"]
    return service.submit(spec, preview["previewHash"], operation)


def test_pack_job_is_a_packing_task_without_a_claim(packing, center):
    service, feature = packing
    job = submit_pack(service, FeaturePackSpec(featureSetId=feature, action="pack"))
    assert job["state"] == "queued" and job["executor"] == "task-center"
    task = center.store.get(job["taskId"])
    assert task["kind"] == "packing" and task["request"]["lane"] == "cpu"
    assert task["request"]["graceSeconds"] == 60
    assert task["labels"]["action"] == "pack" and task["labels"]["recordKind"] == "feature-pack"
    assert task["exclusiveKey"].startswith("feature-source:")
    owner = center.store.owner(task["ownerKey"])
    assert owner["kind"] == "feature-pack" and owner["id"] == job["id"]
    registry = packing_process.registry_directory()
    claim = registry / f"{packing_process.output_key(Path(job['outputPath']))}.claim.json"
    assert not claim.exists()
    plan = json.loads((service.folder / job["id"] / "plan.json").read_text())
    assert plan["claimPath"] is None
    # The same output is busy for any other job while the task lives.
    again = service.preview(
        FeaturePackSpec(featureSetId=feature, action="pack", outputPath=job["outputPath"])
    )
    assert "OUTPUT_BUSY" in {item["code"] for item in again["findings"]}
    validate = submit_pack(service, FeaturePackSpec(featureSetId=feature, action="validate"), "v")
    assert center.store.get(validate["taskId"])["exclusiveKey"] == task["exclusiveKey"]


def test_pack_and_validate_run_one_at_a_time_through_the_runner(packing, center):
    service, feature = packing
    pack = submit_pack(service, FeaturePackSpec(featureSetId=feature, action="pack"))
    validate = submit_pack(service, FeaturePackSpec(featureSetId=feature, action="validate"), "v")
    runner = center.runner()
    seen_together = []

    def done():
        states = [center.state(pack["taskId"]), center.state(validate["taskId"])]
        seen_together.append(states.count("running") > 1)
        return all(state in TERMINAL for state in states)

    center.tick_until(runner, done)
    assert not any(seen_together)
    assert center.state(pack["taskId"]) == center.state(validate["taskId"]) == "succeeded"
    shown = service.get(pack["id"])
    assert shown["state"] == "succeeded" and shown["result"]["artifact"]["id"].startswith("pack-")
    assert service.get(validate["id"])["state"] == "succeeded"
    listed = service.list()
    assert [item["outputPath"] for item in listed["artifacts"]] == [pack["outputPath"]]


def test_busy_output_requeues_instead_of_failing(packing, center):
    service, feature = packing
    job = submit_pack(service, FeaturePackSpec(featureSetId=feature, action="pack"))
    runner = center.runner()
    with packing_process.output_lock(Path(job["outputPath"])):
        center.tick_until(
            runner,
            lambda: (
                center.store.get(job["taskId"])["attempt"] >= 2
                or center.state(job["taskId"]) in TERMINAL
            ),
        )
        first = center.store.list(limit=None)
    assert center.store.get(job["taskId"])["attempt"] >= 2, first
    assert center.state(job["taskId"]) not in {"failed", "cancelled"}
    center.tick_until(runner, lambda: center.state(job["taskId"]) in TERMINAL, timeout=120)
    assert center.state(job["taskId"]) == "succeeded"


def test_retry_moves_the_failed_receipt_aside_and_reaps_staging(packing, center):
    service, feature = packing
    job = submit_pack(service, FeaturePackSpec(featureSetId=feature, action="pack"))
    folder = service.folder / job["id"]
    output = Path(job["outputPath"])
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.packing-abc123"
    staging.mkdir()
    (staging / "features.bin").write_bytes(b"partial")
    task = center.store.get(job["taskId"])
    (folder / "result.json").write_text(
        json.dumps({"jobId": job["id"], "state": "failed", "taskId": task["id"], "taskAttempt": 1})
    )
    center.store.transition(task["id"], from_states=("queued",), to_state="failed")
    assert center.client.requeue_task(task["id"], reason="retry")
    retried = center.store.get(task["id"])
    assert PackingAdapter().prepare(retried, center.context()) is None
    assert not (folder / "result.json").exists()
    assert (folder / "result-attempt-1.json").exists()
    assert not staging.exists()


def test_a_retry_that_finds_the_pack_complete_reads_succeeded(packing, center):
    service, feature = packing
    job = submit_pack(service, FeaturePackSpec(featureSetId=feature, action="validate"))
    runner = center.runner()
    center.tick_until(runner, lambda: center.state(job["taskId"]) in TERMINAL)
    assert center.state(job["taskId"]) == "succeeded"
    # The attempt's conclusion is lost after its receipt was written (a cancel landing
    # during the final write, a reboot); the retry skips as already complete.
    center.store.transition(job["taskId"], from_states=("succeeded",), to_state="interrupted")
    assert center.client.requeue_task(job["taskId"], reason="retry")
    center.tick_until(runner, lambda: center.state(job["taskId"]) in TERMINAL)
    task = center.store.get(job["taskId"])
    assert task["state"] == "succeeded" and task["attempt"] == 2
    shown = service.get(job["id"])
    assert shown["state"] == "succeeded", shown.get("error")
    assert shown["result"]["taskAttempt"] == 1
    # Until that attempt concludes, the earlier receipt is not this attempt's outcome.
    center.store.transition(job["taskId"], from_states=("succeeded",), to_state="interrupted")
    assert center.client.requeue_task(job["taskId"], reason="retry")
    assert service.get(job["id"])["state"] == "queued"


def test_reap_staging_never_touches_a_live_writer(tmp_path):
    output = tmp_path / "pack"
    staging = tmp_path / ".pack.packing-live"
    staging.mkdir()
    with packing_process.output_lock(output):
        assert reap_staging(output) == 0
    assert staging.exists()
    assert reap_staging(output) == 1 and not staging.exists()


def test_reap_staging_matches_its_own_name_literally(tmp_path):
    own = tmp_path / ".pack[1].packing-abc123"
    glob_twin = tmp_path / ".pack1.packing-abc123"
    nested = tmp_path / ".pack[1].packing-x.packing-abc123"  # destination "pack[1].packing-x"
    for folder in (own, glob_twin, nested):
        folder.mkdir()
    assert reap_staging(tmp_path / "pack[1]") == 1
    assert not own.exists() and glob_twin.exists() and nested.exists()


def test_a_cancel_marker_naming_no_attempt_cancels_one_attempt_not_every_retry(packing, center):
    """A stage writes ``attempts: {}`` when the Task Center could not say which attempt
    runs; the attempt it cancels binds it, so the next retry runs."""
    service, feature = packing
    job = submit_pack(service, FeaturePackSpec(featureSetId=feature, action="validate"))
    marker = service.folder / job["id"] / "cancelled"
    marker.write_text(json.dumps({"requestedAt": utc_now_iso(), "attempts": {}}))
    task = center.store.get(job["taskId"])
    skipped = PackingAdapter().prepare(task, center.context())["skip"]
    assert (skipped["state"], skipped["error"]) == ("cancelled", "Cancelled before start.")
    assert json.loads(marker.read_text())["attempts"] == {task["id"]: 1}
    assert center.store.transition(task["id"], from_states=("queued",), to_state="cancelled")
    assert center.client.requeue_task(task["id"], reason="retry")
    runner = center.runner()
    center.tick_until(runner, lambda: center.state(job["taskId"]) in TERMINAL)
    assert center.state(job["taskId"]) == "succeeded" and not marker.exists()


def test_packing_cancel_from_the_stage_records_cancelled(packing, center):
    service, feature = packing
    job = submit_pack(service, FeaturePackSpec(featureSetId=feature, action="validate"))
    cancelled = service.cancel(job["id"])
    assert cancelled["state"] == "cancelled"
    assert center.state(job["taskId"]) == "cancelled"
    marker = json.loads((service.folder / job["id"] / "cancelled").read_text())
    assert marker["attempts"] == {job["taskId"]: 1}


# -- 3.5: registry housekeeping ---------------------------------------------------------------


def test_registry_sweep_drops_dead_and_malformed_claims_and_idle_locks(tmp_path):
    registry = tmp_path / "registry"
    registry.mkdir(mode=0o700)
    old = time.time() - 48 * 3600
    jobs = tmp_path / "jobs"

    def claim(name, job_state=None, *, result=False, age=None, content=None):
        folder = jobs / name
        folder.mkdir(parents=True)
        if job_state:
            (folder / "job.json").write_text(json.dumps({"state": job_state}))
        if result:
            (folder / "result.json").write_text("{}")
        path = registry / f"{packing_process.output_key(name)}.claim.json"
        path.write_text(
            content
            if content is not None
            else json.dumps({"jobPath": str(folder / "job.json"), "outputPath": f"/out/{name}"})
        )
        if age:
            os.utime(path, (age, age))
        return path

    finished = claim("finished", "running", result=True)
    missing = claim("missing")
    stale = claim("stale", "running", age=old)
    fresh = claim("fresh", "starting")
    malformed_old = claim("malformed-old", content="{", age=old)
    malformed_new = claim("malformed-new", content="[]")
    idle_lock = registry / f"{'a' * 64}.lock"
    idle_lock.write_text("")
    os.utime(idle_lock, (old, old))
    young_lock = registry / f"{'b' * 64}.lock"
    young_lock.write_text("")
    held_lock = registry / f"{'c' * 64}.lock"
    held_lock.write_text("")
    os.utime(held_lock, (old, old))
    writer = registry / ".histopilot-write.lock"
    import fcntl

    descriptor = os.open(held_lock, os.O_RDWR)
    fcntl.flock(descriptor, fcntl.LOCK_EX)
    try:
        stats = packing_process.sweep_registry(registry)
    finally:
        os.close(descriptor)
    assert stats["complete"] and stats["claimsRemoved"] == 4 and stats["claimsKept"] == 2
    assert not finished.exists() and not missing.exists() and not stale.exists()
    assert fresh.exists() and malformed_new.exists() and not malformed_old.exists()
    assert stats["locksRemoved"] == 1
    assert not idle_lock.exists() and young_lock.exists() and held_lock.exists()
    assert writer.exists()


def test_malformed_claim_no_longer_blocks_packing(packing, monkeypatch):
    service, feature = packing
    registry = packing_process.registry_directory()
    (registry / f"{'d' * 64}.claim.json").write_text("{")
    preview = service.preview(FeaturePackSpec(featureSetId=feature, action="pack"))
    assert preview["canRun"], preview["findings"]


def test_output_lock_refuses_a_lock_file_swept_away(tmp_path, monkeypatch):
    monkeypatch.setattr(packing_process, "registry_directory", lambda: tmp_path)
    path = tmp_path / f"{packing_process.output_key('/out')}.lock"
    real_flock = packing_process.fcntl.flock

    def flock_after_sweep(descriptor, operation):
        real_flock(descriptor, operation)
        if operation & packing_process.fcntl.LOCK_EX:
            path.unlink(missing_ok=True)  # the sweep unlinked it between open and flock

    monkeypatch.setattr(packing_process.fcntl, "flock", flock_after_sweep)
    from histopilot.storage.project_lock import StorageError

    with pytest.raises(StorageError) as caught, packing_process.output_lock("/out"):
        pass
    assert caught.value.code == "OUTPUT_BUSY"


def test_receipts_are_parsed_once_per_file_version(packing, center, monkeypatch):
    from histopilot.application import feature_packs

    service, feature = packing
    job = submit_pack(service, FeaturePackSpec(featureSetId=feature, action="validate"))
    runner = center.runner()
    center.tick_until(runner, lambda: center.state(job["taskId"]) in TERMINAL)
    reads = []
    original = feature_packs._read
    monkeypatch.setattr(feature_packs, "_read", lambda path: reads.append(path) or original(path))
    service.list()
    service.list()
    assert sum(path.name == "result.json" for path in reads) <= 1


# -- 3.4: archive tasks -----------------------------------------------------------------------


@pytest.fixture
def archives(tmp_path, center):
    from fastapi.testclient import TestClient

    from histopilot.api import create_app
    from histopilot.application.portability_jobs import PortabilityJobs
    from histopilot.config import Settings

    data = tmp_path / "data"
    data.mkdir()
    settings = Settings(workspace=tmp_path / "workspace", data_roots=(data,))
    with TestClient(create_app(settings), base_url="http://127.0.0.1:8787") as client:
        client.headers["X-HistoPilot-Token"] = client.get("/api/v1/session").json()["token"]
        response = client.post(
            "/api/v1/projects",
            json={"name": "Archive tasks", "storagePath": str(settings.workspace / "study")},
        )
        assert response.status_code == 201, response.text
        projects = client.app.state.projects
        store = projects.scientific_store(response.json()["id"])
        jobs = PortabilityJobs(
            projects, projects.storage, execution_mode="task-center", task_center=center.client
        )
        yield store, projects, jobs


def _busy_packing_record(store):
    """A legacy packing record whose worker is 'alive' (this test process)."""
    from histopilot.workers.packing_process import process_metadata, write_json

    identity = "packing-" + "e" * 32
    folder = store.folder / "packing" / identity
    folder.mkdir(parents=True)
    now = utc_now_iso()
    write_json(
        folder / "job.json",
        {
            "id": identity,
            "projectId": store.project_id,
            "state": "running",
            "spec": {"action": "validate"},
            "featureSetId": "configuration-" + "f" * 64,
            "outputPath": None,
            "sessionName": "histopilot-pack-test",
            "logPath": str(folder / "worker.log"),
            "createdAt": now,
            "updatedAt": now,
        },
    )
    write_json(folder / "process.json", process_metadata())
    return folder


def test_export_is_an_archive_task_that_waits_for_an_idle_project(archives, center):
    from histopilot.schemas.operations import PortabilityRequest

    store, projects, jobs = archives
    busy = _busy_packing_record(store)
    request = PortabilityRequest(
        action="export",
        archivePath=str(projects.database.workspace / "study.zip"),
        operationId="export-task",
    )
    job = jobs.submit(store.project_id, request)
    assert job["status"] == "queued" and job["executor"] == "task-center"
    task = center.store.get(job["taskId"])
    assert task["kind"] == "archive" and task["labels"]["action"] == "export"
    assert task["labels"]["recordKind"] == "archive"
    owner = center.store.owner(task["ownerKey"])
    assert owner["kind"] == "archive" and owner["projectFolder"] == str(store.folder)
    assert job["ownerKey"] == task["ownerKey"]
    runner = center.runner()
    center.tick_until(runner, lambda: center.store.get(job["taskId"])["attempt"] >= 2)
    waiting = jobs.get(store.project_id, job["id"])
    assert waiting["status"] == "queued"
    assert "active jobs" in (waiting.get("waitingReason") or "")
    (busy / "process.json").unlink()
    (busy / "cancelled").write_text("{}")
    center.tick_until(runner, lambda: center.state(job["taskId"]) in TERMINAL, timeout=120)
    assert center.state(job["taskId"]) == "succeeded"
    done = jobs.get(store.project_id, job["id"])
    assert done["status"] == "completed" and done["result"]["verified"]


def test_extraction_and_archive_bind_a_cancel_marker_that_names_no_attempt(
    extraction, archives, center
):
    from histopilot.schemas.operations import PortabilityRequest
    from histopilot.taskcenter.adapters.archive import ArchiveAdapter

    service, spec, _commands = extraction
    job = submit_extraction(service, spec)
    marker = service.folder / job["id"] / "cancelled"
    marker.write_text("cancelled")  # a legacy or attempt-less marker
    task = center.store.get(job["taskId"])
    assert ExtractionAdapter().prepare(task, center.context())["skip"]["state"] == "cancelled"
    assert json.loads(marker.read_text())["attempts"] == {task["id"]: 1}
    assert center.store.transition(task["id"], from_states=("queued",), to_state="cancelled")
    assert center.client.requeue_task(task["id"], reason="retry")
    retried = center.store.get(task["id"])
    assert "skip" not in (ExtractionAdapter().prepare(retried, center.context()) or {})
    assert not marker.exists()

    store, projects, jobs = archives
    request = PortabilityRequest(
        action="export",
        archivePath=str(projects.database.workspace / "bound.zip"),
        operationId="export-bound",
    )
    archive = jobs.submit(store.project_id, request)
    task = center.store.get(archive["taskId"])
    marker = Path(task["adapterData"]["portabilityFolder"]) / "cancel.requested"
    marker.write_text(json.dumps({"requestedAt": utc_now_iso(), "attempts": {}}))
    assert ArchiveAdapter().prepare(task, center.context())["skip"]["state"] == "cancelled"
    assert json.loads(marker.read_text())["attempts"] == {task["id"]: 1}
    assert center.store.transition(task["id"], from_states=("queued",), to_state="cancelled")
    assert center.client.requeue_task(task["id"], reason="retry")
    assert ArchiveAdapter().prepare(center.store.get(task["id"]), center.context()) is None
    assert not marker.exists()


def test_archive_cancel_and_retry_run_the_same_task_again(archives, center):
    from histopilot.schemas.operations import PortabilityRequest

    store, projects, jobs = archives
    request = PortabilityRequest(
        action="export",
        archivePath=str(projects.database.workspace / "retry.zip"),
        operationId="export-retry",
    )
    job = jobs.submit(store.project_id, request)
    cancelled = jobs.cancel(store.project_id, job["id"])
    assert cancelled["status"] == "cancelled"
    retried = jobs.retry(store.project_id, job["id"])
    assert retried["status"] == "queued"
    assert center.store.get(job["taskId"])["attempt"] == 2
    runner = center.runner()
    center.tick_until(runner, lambda: center.state(job["taskId"]) in TERMINAL, timeout=120)
    assert center.state(job["taskId"]) == "succeeded"
    assert jobs.get(store.project_id, job["id"])["status"] == "completed"


# -- links ------------------------------------------------------------------------------------


def test_task_center_links_open_the_owning_stage(center):
    from histopilot.taskcenter.service import TaskCenterService

    hashes = TaskCenterService._preparation_hash
    assert hashes("extraction", "extraction-1", {}) == "#features?extraction=extraction-1"
    assert (
        hashes("feature-pack", "packing-1", {"featureSetId": "configuration-2"})
        == "#features?source=configuration-2&packing=packing-1"
    )
    assert hashes("archive", "portability-1", {}) == "#operations"


def test_sweep_runs_at_most_every_interval_and_resumes_an_unfinished_one(tmp_path, monkeypatch):
    registry = tmp_path / "registry"
    registry.mkdir(mode=0o700)
    calls = []
    outcome = {"complete": False}
    monkeypatch.setattr(
        packing_process, "sweep_registry", lambda folder: calls.append(folder) or dict(outcome)
    )
    assert packing_process.maybe_sweep_registry(registry) == {"complete": False}
    stamp = registry / packing_process.SWEEP_STAMP
    # An unfinished sweep continues about a minute later, not after the full interval.
    remaining = stamp.stat().st_mtime + packing_process.SWEEP_INTERVAL_SECONDS - time.time()
    assert 0 < remaining <= 61
    assert packing_process.maybe_sweep_registry(registry) is None
    os.utime(stamp, (0, 0))
    outcome["complete"] = True
    assert packing_process.maybe_sweep_registry(registry) == {"complete": True}
    assert packing_process.maybe_sweep_registry(registry) is None
    assert len(calls) == 2


def test_bootstrap_reports_peak_cuda_memory_without_importing_torch(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from histopilot.adapters.trident import bootstrap

    peaks = {0: 3 * 2**30, 1: 5 * 2**30}
    cuda = SimpleNamespace(
        is_initialized=lambda: True,
        device_count=lambda: 2,
        max_memory_reserved=lambda device: peaks[device],
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    registered = []
    monkeypatch.setattr("atexit.register", registered.append)
    target = tmp_path / "peak.json"
    stop = bootstrap.report_peak_memory(target, interval=0.01)
    try:
        _wait_for(target.exists)
    finally:
        stop.set()
    assert json.loads(target.read_text()) == {"cudaPeakReservedBytes": 5 * 2**30}
    assert registered  # a final sample is taken at exit
    monkeypatch.delitem(sys.modules, "torch")
    assert bootstrap._peak_reserved_bytes() is None  # never imports torch itself
