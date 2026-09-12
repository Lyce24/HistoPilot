"""Real tiny CPU checkpoints prove whole-bag attention and exact ensemble averaging."""

import copy
import json
import runpy
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")
Image = pytest.importorskip("PIL.Image")

from histopilot.application.predictors import checkpoint_snapshot  # noqa: E402
from histopilot.storage.attention_inputs import file_stamp, inspect_inputs  # noqa: E402
from histopilot.training.attention import _percentiles, interpret  # noqa: E402
from histopilot.training.fold import train_fold  # noqa: E402
from histopilot.training.module import MILTrainModule  # noqa: E402
from histopilot.workers.packing_process import write_json  # noqa: E402

support = runpy.run_path(str(Path(__file__).with_name("test_mil_training.py")))


def attention_plan(tmp_path):
    """Small, synthetic fixture; safe to retain for an offline UI verification."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    training = support["tiny_plan"](tmp_path)
    training["recipe"]["maxEpochs"] = 1
    result = train_fold(training, tmp_path / "fit")
    checkpoint = checkpoint_snapshot(result["bestCheckpointPath"], tmp_path / "fit")
    slide_path = tmp_path / "synthetic-slide.png"
    image = Image.new("RGB", (384, 256), color="#f6e7ef")
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    rng = np.random.default_rng(7)
    for _ in range(350):
        x, y = rng.integers([0, 0], [384, 256])
        radius = int(rng.integers(2, 7))
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(int(rng.integers(95, 160)), 72, int(rng.integers(125, 180))),
        )
    image.save(slide_path)
    features = np.array(
        [
            [2, 0, 0, 0],
            [0, 1, 0, 0],
            [0.1, 0.1, 1, 0],
            [0, 0, 0, 2],
            [1, 1, 0, 0],
            [2, 0, 2, 0],
            [0, 2, 0, 0],
        ],
        dtype=np.float32,
    )
    coords = np.array(
        [[0, 0], [96, 0], [192, 0], [288, 0], [0, 128], [96, 128], [192, 128]], dtype=np.int64
    )
    feature_path = tmp_path / "independent-features.h5"
    with h5py.File(feature_path, "w") as handle:
        handle.create_dataset("features", data=features)
        handle.create_dataset("coords", data=coords)
    slide = {
        "slideId": "independent-synthetic",
        "slidePath": str(slide_path),
        "featurePath": str(feature_path),
        "coordinatesPath": None,
        "featureKey": "features",
        "coordinatesKey": "coords",
        "confirmRowAlignment": True,
        "coordinateSpace": "level0",
        "patchWidthLevel0": 96.0,
        "patchHeightLevel0": 128.0,
        "width": 384,
        "height": 256,
        "backend": "pillow",
        "levelDownsamples": [1.0],
    }
    contract = {"encoderId": "tiny-fixture-encoder", "dimensions": 4, "dtype": "float32"}
    slide.update(inspect_inputs(slide, slide, contract), slideSource=file_stamp(slide_path))
    plan = {
        "kind": "interpretation",
        "runId": "attention-tiny",
        "method": "refit",
        "target": training["target"],
        "device": "cpu",
        "resources": {
            "gpuIds": [],
            "cpuThreadsPerRun": 1,
            "dataLoaderWorkers": 0,
            "ramGbPerRun": 1.0,
            "maxConcurrentRuns": 1,
            "runsPerGpu": 1,
        },
        "checkpoints": [checkpoint],
        "featureContract": contract,
        "slides": [slide],
        "references": [],
        "data": {"sourceStamps": slide["sourceStamps"]},
    }
    write_json(tmp_path / "attention-plan.json", plan)
    return plan


@pytest.fixture
def plan(tmp_path):
    return attention_plan(tmp_path)


def test_refit_attention_is_exact_whole_bag_forward_and_resume_reuses_evidence(
    plan, tmp_path, monkeypatch
):
    result = interpret(plan, tmp_path / "attention")
    value = json.loads((tmp_path / "attention/slide-0.json").read_text())
    model = MILTrainModule.load_from_checkpoint(plan["checkpoints"][0]["path"], weights_only=True)
    model.eval()
    with h5py.File(plan["slides"][0]["featurePath"]) as handle, torch.inference_mode():
        features = torch.tensor(handle["features"][:]).unsqueeze(0)
        expected = model.model(features, return_attention=True)
    weights = [row["weight"] for row in value["patches"]]
    assert len(weights) == 7 > model.recipe["bagSize"]
    assert weights == pytest.approx(expected["attention"][0].tolist())
    assert sum(weights) == pytest.approx(1)
    assert value["probabilities"] == pytest.approx(
        torch.softmax(expected["logits"][0].double(), -1).tolist()
    )
    assert value["attentionKind"] == "class_independent_pooling"
    assert [row["index"] for row in value["patches"]] == list(range(7))
    monkeypatch.setattr(
        MILTrainModule,
        "load_from_checkpoint",
        lambda *a, **k: pytest.fail("Completed members must be reused"),
    )
    assert interpret(plan, tmp_path / "attention")["artifacts"] == result["artifacts"]


def test_distinct_ensemble_members_average_attention_and_probabilities(plan, tmp_path):
    original = torch.load(plan["checkpoints"][0]["path"], map_location="cpu", weights_only=True)
    original["state_dict"]["model.attention_score.weight"] *= -4
    original["state_dict"]["model.classifier.bias"] += torch.tensor([-2.0, 2.0])
    other_path = tmp_path / "distinct.ckpt"
    torch.save(original, other_path)
    ensemble = {
        **plan,
        "method": "ensemble",
        "checkpoints": [*plan["checkpoints"], checkpoint_snapshot(other_path, tmp_path)],
    }
    result = interpret(ensemble, tmp_path / "ensemble")
    mean, first, second = [
        json.loads((tmp_path / "ensemble" / name).read_text())
        for name in ("slide-0.json", "slide-0-member-0.json", "slide-0-member-1.json")
    ]
    first_weights = np.array([row["weight"] for row in first["patches"]])
    second_weights = np.array([row["weight"] for row in second["patches"]])
    assert not np.allclose(first_weights, second_weights)
    assert not np.allclose(first["probabilities"], second["probabilities"])
    assert [row["weight"] for row in mean["patches"]] == pytest.approx(
        (first_weights + second_weights) / 2
    )
    assert mean["probabilities"] == pytest.approx(
        (np.array(first["probabilities"]) + second["probabilities"]) / 2
    )
    assert result["memberCount"] == 2
    assert len(result["slides"][0]["members"]) == 2
    assert _percentiles(np.array([1.0, 1.0, 1.0])).tolist() == [0.5, 0.5, 0.5]


def test_changed_source_checkpoint_cache_and_cancellation_fail_closed(plan, tmp_path):
    interpret(plan, tmp_path / "complete")
    path = tmp_path / "complete/slide-0-member-0.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="evidence changed"):
        interpret(plan, tmp_path / "complete")
    cancelled = tmp_path / "cancelled"
    cancelled.mkdir()
    (cancelled / "cancel.requested").touch()
    with pytest.raises(KeyboardInterrupt, match="cancelled"):
        interpret(plan, cancelled)
    checkpoint = Path(plan["checkpoints"][0]["path"])
    checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checkpoint changed"):
        interpret(plan, tmp_path / "changed")
    with h5py.File(plan["slides"][0]["featurePath"], "a") as handle:
        handle["features"][0, 0] = 100
    with pytest.raises(ValueError, match="changed"):
        interpret(plan, tmp_path / "source-changed")


def test_immutable_contract_detects_coords_even_when_stamp_is_replaced(plan, tmp_path):
    changed = copy.deepcopy(plan)
    feature_path = Path(changed["slides"][0]["featurePath"])
    with h5py.File(feature_path, "a") as handle:
        handle["coords"][:] = handle["coords"][:][::-1]
    changed["slides"][0]["sourceStamps"][str(feature_path)] = file_stamp(feature_path)
    with pytest.raises(ValueError, match="reviewed evidence"):
        interpret(changed, tmp_path / "wrong-rows")


def test_pinned_worker_executes_attention_without_server_or_tmux(plan, tmp_path):
    support_jobs = runpy.run_path(str(Path(__file__).with_name("test_compute_jobs.py")))
    service, initial_id, _, executor = support_jobs["job"].__wrapped__(tmp_path)
    store = service.store
    dataset_id = store.get_configuration(initial_id)["manifest"]["datasetId"]
    predictor = store.publish_configuration(
        manifest={
            "kind": "frozen-predictor",
            "datasetId": dataset_id,
            "target": plan["target"],
            "method": plan["method"],
            "checkpoints": plan["checkpoints"],
        },
        operation_id="attention-predictor",
    )
    plan["references"] = [{"id": predictor["id"], "contentHash": predictor["contentHash"]}]
    document = store.publish_configuration(
        manifest={
            "kind": "model-interpretation",
            "datasetId": dataset_id,
            "predictorId": predictor["id"],
            **{
                key: plan[key]
                for key in ("target", "slides", "featureContract", "resources", "references")
            },
        },
        operation_id="attention-study",
    )
    plan["runId"] = document["id"]
    service.launch(document["id"], plan, "attention-launch")
    _, _, plan_path, _, archive = executor.calls[0]
    result = subprocess.run(
        [sys.executable, "-m", "histopilot.workers.compute_job", str(plan_path)],
        cwd=archive,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    executor.sessions.clear()
    status = service.status(document["id"])
    assert status["status"] == "completed", status
    assert status["result"]["slideCount"] == 1
    assert status["result"] == json.loads(
        (service.folder(document["id"]) / "result.json").read_text()
    )


def test_large_attention_json_crosses_metadata_limit_and_failed_write_is_atomic(
    tmp_path, monkeypatch
):
    from histopilot.training import attention

    path = tmp_path / "large-attention.json"
    attention._write_attention_json(path, {"payload": "x" * (65 * 1024 * 1024)})
    previous_size = path.stat().st_size
    assert previous_size > 64 * 1024 * 1024
    with path.open() as stream:
        assert len(json.load(stream)["payload"]) == 65 * 1024 * 1024
    monkeypatch.setattr(attention, "MAX_ARTIFACT_BYTES", 100)
    with pytest.raises(ValueError, match="artifact budget"):
        attention._write_attention_json(path, {"payload": "a" * 101})
    assert path.stat().st_size == previous_size
    assert not list(tmp_path.glob(".large-attention.json.*"))


def packed_attention_plans(tmp_path):
    """Two independently selectable nested slides, with identical H5/packed vectors."""
    from histopilot.storage.pack_import import pack_file_stamps
    from histopilot.storage.packed import build_pack

    base = attention_plan(tmp_path / "tiny-training")
    source_dir = tmp_path / "features"
    source_dir.mkdir(parents=True)
    slide_dir = tmp_path / "slides"
    paths = [slide_dir / "cohort A" / "case-one.png", slide_dir / "cohort B" / "case-two.png"]
    with h5py.File(base["slides"][0]["featurePath"]) as handle:
        first_features, first_coords = handle["features"][:], handle["coords"][:]
    native_slides, source_files = [], []
    for index, slide_path in enumerate(paths):
        slide_path.parent.mkdir(parents=True)
        identity = slide_path.stem
        with Image.open(base["slides"][0]["slidePath"]) as image:
            image.save(slide_path)
        features = (
            first_features
            if index == 0
            else (first_features[:5][::-1] * 0.5 + 0.3).astype(np.float32)
        )
        coords = first_coords if index == 0 else first_coords[:5][::-1]
        feature_path = source_dir / f"{identity}.h5"
        with h5py.File(feature_path, "w") as handle:
            handle.create_dataset("features", data=features)
            handle["features"].attrs["encoder_id"] = base["featureContract"]["encoderId"]
            handle.create_dataset("coords", data=coords)
            handle["coords"].attrs["patch_width_level0"] = 96.0
            handle["coords"].attrs["patch_height_level0"] = 128.0
        source_files.append(
            {
                "slideId": identity,
                "path": str(feature_path),
                "coordinatePath": str(feature_path),
                "dimensions": 4,
                "dtype": "float32",
                "patchCount": len(features),
                "coordinateSpace": "level0",
            }
        )
        selection = {
            "sourceFormat": "h5",
            "slideId": identity,
            "slidePath": str(slide_path),
            "featurePath": str(feature_path),
            "coordinatesPath": None,
            "featureKey": "features",
            "coordinatesKey": "coords",
            "confirmRowAlignment": True,
            "coordinateSpace": "level0",
            "patchWidthLevel0": None,
            "patchHeightLevel0": None,
            "width": 384,
            "height": 256,
            "backend": "pillow",
            "levelDownsamples": [1.0],
        }
        selection.update(
            inspect_inputs(selection, selection, base["featureContract"]),
            slideSource=file_stamp(slide_path),
        )
        native_slides.append(selection)
    config = {
        "id": "configuration-" + "f" * 64,
        "contentHash": "a" * 64,
        "manifest": {
            "kind": "feature",
            "files": source_files,
            "provenance": [],
            "layout": {
                "featureDirectory": str(source_dir),
                "encoderId": base["featureContract"]["encoderId"],
            },
        },
    }
    pack_path = tmp_path / "shared-pack"
    manifest = build_pack(config, pack_path)
    artifact = {**manifest, "outputPath": str(pack_path), "packStamps": pack_file_stamps(pack_path)}
    packed_slides = []
    for slide in reversed(native_slides):
        packed = {
            key: value
            for key, value in slide.items()
            if key
            not in {
                "featurePath",
                "sourceStamps",
                "featureSha256",
                "coordinatesSha256",
                "alignment",
            }
        }
        packed.update(
            sourceFormat="packed",
            packPath=str(pack_path),
            packSlideId=slide["slideId"],
            packEvidence=artifact,
        )
        packed.update(inspect_inputs(packed, packed, base["featureContract"]))
        packed_slides.append(packed)
    native_plan = {
        **base,
        "runId": "native-refit",
        "slides": native_slides,
        "data": {
            "sourceStamps": {
                path: stamp
                for slide in native_slides
                for path, stamp in slide["sourceStamps"].items()
            }
        },
    }
    packed_plan = {
        **base,
        "runId": "packed-refit",
        "slides": packed_slides,
        "data": {"sourceStamps": packed_slides[0]["sourceStamps"]},
    }
    checkpoint = torch.load(base["checkpoints"][0]["path"], map_location="cpu", weights_only=True)
    checkpoint["state_dict"]["model.attention_score.weight"] *= -4
    checkpoint["state_dict"]["model.classifier.bias"] += torch.tensor([-2.0, 2.0])
    second_path = tmp_path / "distinct-ensemble-member.ckpt"
    torch.save(checkpoint, second_path)
    members = [*base["checkpoints"], checkpoint_snapshot(second_path, tmp_path)]
    plans = {
        "native-refit": native_plan,
        "packed-refit": packed_plan,
        "native-ensemble": {
            **native_plan,
            "runId": "native-ensemble",
            "method": "ensemble",
            "checkpoints": members,
        },
        "packed-ensemble": {
            **packed_plan,
            "runId": "packed-ensemble",
            "method": "ensemble",
            "checkpoints": members,
        },
    }
    for name, plan in plans.items():
        write_json(tmp_path / f"{name}-plan.json", plan)
    write_json(tmp_path / "feature-configuration.json", config)
    write_json(tmp_path / "pack-artifact.json", artifact)
    return plans


@pytest.mark.parametrize("method", ["refit", "ensemble"])
def test_packed_multi_slide_attention_matches_native_and_resumes_exact_members(
    tmp_path, method, monkeypatch
):
    plans = packed_attention_plans(tmp_path)
    native, packed = plans[f"native-{method}"], plans[f"packed-{method}"]
    native_result = interpret(native, tmp_path / "native-output")
    packed_result = interpret(packed, tmp_path / "packed-output")
    assert packed_result["slideCount"] == 2
    assert [slide["slideId"] for slide in packed_result["slides"]] == ["case-two", "case-one"]
    assert [row["packedRow"]["offset"] for row in packed["slides"]] == [7, 0]
    for result_slide in packed_result["slides"]:
        counterpart = next(
            slide
            for slide in native_result["slides"]
            if slide["slideId"] == result_slide["slideId"]
        )
        actual = json.loads(
            (tmp_path / "packed-output" / result_slide["attentionArtifact"]).read_text()
        )
        expected = json.loads(
            (tmp_path / "native-output" / counterpart["attentionArtifact"]).read_text()
        )
        assert actual == expected
        if method == "ensemble":
            first, second = [
                json.loads((tmp_path / "packed-output" / row["attentionArtifact"]).read_text())
                for row in result_slide["members"]
            ]
            assert not np.allclose(
                [row["weight"] for row in first["patches"]],
                [row["weight"] for row in second["patches"]],
            )
            assert actual["probabilities"] == pytest.approx(
                (np.array(first["probabilities"]) + second["probabilities"]) / 2
            )
    monkeypatch.setattr(
        MILTrainModule,
        "load_from_checkpoint",
        lambda *args, **kwargs: pytest.fail(
            "Resume must reuse completed packed slide/member pairs"
        ),
    )
    assert interpret(packed, tmp_path / "packed-output")["artifacts"] == packed_result["artifacts"]


def test_packed_attention_detects_mutation_before_reusing_cached_predictions(tmp_path):
    plans = packed_attention_plans(tmp_path)
    plan = plans["packed-refit"]
    interpret(plan, tmp_path / "packed-output")
    coords_path = Path(plan["slides"][0]["packPath"]) / "coords.bin"
    with coords_path.open("r+b") as stream:
        stream.write(np.array([10], dtype="<i4").tobytes())
    with pytest.raises(ValueError, match="pack changed"):
        interpret(plan, tmp_path / "packed-output")


def test_pinned_worker_executes_packed_multi_slide_attention(tmp_path):
    plan = packed_attention_plans(tmp_path)["packed-ensemble"]
    job_support = runpy.run_path(str(Path(__file__).with_name("test_compute_jobs.py")))
    service, initial_id, _, executor = job_support["job"].__wrapped__(tmp_path)
    store = service.store
    dataset_id = store.get_configuration(initial_id)["manifest"]["datasetId"]
    predictor = store.publish_configuration(
        manifest={
            "kind": "frozen-predictor",
            "datasetId": dataset_id,
            "target": plan["target"],
            "method": plan["method"],
            "checkpoints": plan["checkpoints"],
        },
        operation_id="packed-predictor",
    )
    plan["references"] = [{"id": predictor["id"], "contentHash": predictor["contentHash"]}]
    record = store.publish_configuration(
        manifest={
            "kind": "model-interpretation",
            "datasetId": dataset_id,
            "predictorId": predictor["id"],
            **{
                key: plan[key]
                for key in ("target", "slides", "featureContract", "resources", "references")
            },
        },
        operation_id="packed-attention-study",
    )
    plan["runId"] = record["id"]
    service.launch(record["id"], plan, "launch-packed")
    _, _, plan_path, _, archive = executor.calls[0]
    assert (archive / "histopilot/storage/attention_packs.py").exists()
    finished = subprocess.run(
        [sys.executable, "-m", "histopilot.workers.compute_job", str(plan_path)],
        cwd=archive,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert finished.returncode == 0, finished.stderr
    executor.sessions.clear()
    state = service.status(record["id"])
    assert state["status"] == "completed", state
    assert state["result"]["slideCount"] == 2
    assert state["result"]["memberCount"] == 2
    assert [row["slideId"] for row in state["result"]["slides"]] == ["case-two", "case-one"]
