"""Actual tiny CPU checkpoints exercise refit/ensemble test inference."""

import copy
import json
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

from histopilot.application.predictors import checkpoint_snapshot  # noqa: E402
from histopilot.storage.packed import _stamp  # noqa: E402
from histopilot.training.fold import train_fold  # noqa: E402
from histopilot.training.inference import (  # noqa: E402
    _decisions,
    evaluate,
    evaluation_metrics,
    patient_predictions,
)

support = runpy.run_path(str(Path(__file__).with_name("test_mil_training.py")))


@pytest.fixture
def plan(tmp_path):
    training = support["tiny_plan"](tmp_path)
    training["recipe"].update(maxEpochs=1)
    result = train_fold(training, tmp_path / "fit")
    data = copy.deepcopy(training["data"])
    data["memberships"] = [row for row in data["memberships"] if row["partition"] == "test"]
    selected = {row["slideId"] for row in data["memberships"]}
    data["featureFiles"] = {
        key: row for key, row in data["featureFiles"].items() if key in selected
    }
    for entry in data["featureFiles"].values():
        entry.update(_stamp(Path(entry["path"]).stat()))
    data["sourceStamps"] = {row["path"]: row for row in data["featureFiles"].values()}
    data["memberships"][-1]["label"] = None
    return {
        "kind": "evaluation",
        "runId": "evaluation-tiny",
        "method": "refit",
        "target": training["target"],
        "data": data,
        "resources": training["resources"],
        "device": "cpu",
        "checkpoints": [checkpoint_snapshot(result["bestCheckpointPath"], tmp_path / "fit")],
        "inference": {
            "loadingPolicy": "per_slide",
            "batchSize": 2,
            "numWorkers": 0,
            "precision": "float32",
            "patientAggregation": "mean",
            "decisionThreshold": 0.75,
        },
    }


def test_refit_and_ensemble_predict_exact_whole_bags_and_preserve_unlabeled_rows(plan, tmp_path):
    first = evaluate(plan, tmp_path / "refit")
    ensemble = {**plan, "method": "ensemble", "checkpoints": plan["checkpoints"] * 2}
    second = evaluate(ensemble, tmp_path / "ensemble")
    one = json.loads((tmp_path / "refit/predictions.json").read_text())
    two = json.loads((tmp_path / "ensemble/predictions.json").read_text())
    assert first["slideCount"] == second["slideCount"] == 4
    for left, right in zip(one["records"], two["records"], strict=True):
        assert left["probabilities"] == pytest.approx(right["probabilities"])
        assert left["logProbabilities"] == pytest.approx(right["logProbabilities"])
    assert [row["slideId"] for row in one["records"]] == [
        row["slideId"] for row in plan["data"]["memberships"]
    ]
    assert first["metrics"]["slide"]["count"] == 3
    assert first["metrics"]["slide"]["unlabeledCount"] == 1
    assert (tmp_path / "refit/slide-predictions.csv").read_text().count("\n") == 5
    # A completed-member restart reuses the same deterministic output.
    assert evaluate(plan, tmp_path / "refit")["artifacts"] == first["artifacts"]


def test_changed_checkpoint_and_feature_sources_block_inference(plan, tmp_path):
    checkpoint = Path(plan["checkpoints"][0]["path"])
    checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checkpoint changed"):
        evaluate(plan, tmp_path / "changed")
    path = Path(next(iter(plan["data"]["featureFiles"].values()))["path"])
    np.save(path, np.zeros((7, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="changed"):
        evaluate(plan, tmp_path / "source-changed")


def test_threshold_uses_frozen_positive_class_and_patient_mean_with_unlabeled_slides():
    target = support["target"]()
    rows = [
        {
            "slideId": "a",
            "patientId": "p",
            "labelIndex": 0,
            "label": "class-0",
            "probabilities": [0.6, 0.4],
        },
        {
            "slideId": "b",
            "patientId": "p",
            "labelIndex": None,
            "label": None,
            "probabilities": [0.2, 0.8],
        },
    ]
    _decisions(rows, target, 0.7)
    assert rows[0]["predictedLabel"] == "class-1"
    metrics = evaluation_metrics(rows, target, 0.7)
    assert metrics["accuracy"] == 0 and metrics["count"] == 1
    patients = patient_predictions(rows)
    assert patients[0]["probabilities"] == pytest.approx([0.4, 0.6])
    assert patients[0]["labelIndex"] == 0


@pytest.mark.parametrize("threshold,expected_f1", [(0.5, 0.5), (0.7, 1 / 3)])
def test_evaluation_macro_f1_keeps_all_frozen_classes_after_thresholding(threshold, expected_f1):
    target = support["target"]()
    rows = [
        {
            "slideId": f"slide-{index}",
            "patientId": f"patient-{index}",
            "labelIndex": 0 if index < 2 else None,
            "label": "class-0" if index < 2 else None,
            "probabilities": probabilities,
        }
        for index, probabilities in enumerate(([0.9, 0.1], [0.6, 0.4], [0.1, 0.9]))
    ]
    metrics = evaluation_metrics(_decisions(rows, target, threshold), target, threshold)
    assert metrics["macroF1"] == pytest.approx(expected_f1)
    assert metrics["missingClasses"] == ["class-1"]
    assert metrics["count"] == 2
    assert metrics["unlabeledCount"] == 1
    assert metrics["confusionMatrix"] == (
        [[2, 0], [0, 0]] if threshold == 0.5 else [[1, 1], [0, 0]]
    )


def test_patient_conflicts_and_invalid_refit_member_counts_rejected(plan, tmp_path):
    invalid = {**plan, "checkpoints": plan["checkpoints"] * 2}
    with pytest.raises(ValueError, match="one checkpoint"):
        evaluate(invalid, tmp_path / "invalid")
    rows = [
        {"slideId": "a", "patientId": "p", "labelIndex": 0},
        {"slideId": "b", "patientId": "p", "labelIndex": 1},
    ]
    with pytest.raises(ValueError, match="consistent labels"):
        patient_predictions(rows)


def test_pinned_worker_executes_real_inference_and_publishes_identical_receipt(plan, tmp_path):
    job_support = runpy.run_path(str(Path(__file__).with_name("test_compute_jobs.py")))
    service, initial_id, _, executor = job_support["job"].__wrapped__(tmp_path)
    store = service.store
    dataset_id = store.get_configuration(initial_id)["manifest"]["datasetId"]

    def publish(kind, values):
        return store.publish_configuration(
            manifest={"kind": kind, "datasetId": dataset_id, **values}, operation_id=kind
        )

    feature = publish("feature", {"files": list(plan["data"]["featureFiles"].values())})
    predictor = publish(
        "frozen-predictor",
        {"target": plan["target"], "method": plan["method"], "checkpoints": plan["checkpoints"]},
    )
    cohort = publish("evaluation-cohort", {"memberships": plan["data"]["memberships"]})
    evaluation = publish(
        "model-evaluation",
        {
            "predictorId": predictor["id"],
            "cohortId": cohort["id"],
            "features": {
                "feature": {"id": feature["id"]},
                "dimensions": plan["data"]["featureDim"],
            },
            "target": plan["target"],
            "inference": plan["inference"],
        },
    )
    identity = evaluation["id"]
    plan["resources"].update(gpuIds=[], ramGbPerRun=0.01, maxConcurrentRuns=1, runsPerGpu=1)
    service.launch(identity, plan, "launch")
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
    state = service.status(identity)
    assert state["status"] == "completed", state
    assert state["result"] == json.loads((service.folder(identity) / "result.json").read_text())
    assert state["result"]["slideCount"] == len(plan["data"]["memberships"])


def test_ensemble_averages_probabilities_and_preserves_extreme_finite_loss(plan, tmp_path):
    checkpoints = []
    for index, bias in enumerate(([2.0, -2.0], [-1.0, 1.0])):
        value = torch.load(plan["checkpoints"][0]["path"], map_location="cpu", weights_only=True)
        value["state_dict"]["model.classifier.weight"].zero_()
        value["state_dict"]["model.classifier.bias"] = torch.tensor(bias)
        path = tmp_path / f"member-{index}.ckpt"
        torch.save(value, path)
        checkpoints.append(checkpoint_snapshot(path, tmp_path))
    evaluate({**plan, "method": "ensemble", "checkpoints": checkpoints}, tmp_path / "mean")
    rows = json.loads((tmp_path / "mean/predictions.json").read_text())["records"]
    expected = (
        torch.softmax(torch.tensor([2.0, -2.0]).double(), 0)
        + torch.softmax(torch.tensor([-1.0, 1.0]).double(), 0)
    ) / 2
    assert rows[0]["probabilities"] == pytest.approx(expected.tolist())
    # Do not clamp an extreme but finite log loss to log(1e-300).
    value["state_dict"]["model.classifier.bias"] = torch.tensor([-1000.0, 0.0])
    path = tmp_path / "extreme.ckpt"
    torch.save(value, path)
    result = evaluate(
        {**plan, "checkpoints": [checkpoint_snapshot(path, tmp_path)]}, tmp_path / "extreme"
    )
    assert result["metrics"]["slide"]["loss"] == pytest.approx(2000 / 3)
