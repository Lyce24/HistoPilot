"""Patient, threshold and OOF invariants across the complete MIL runtime."""

import copy
import json
import runpy
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

from histopilot.datasets.datamodule import MILDataModule  # noqa: E402
from histopilot.datasets.mil import validate_memberships  # noqa: E402
from histopilot.scoring import patient_predictions  # noqa: E402
from histopilot.training.inference import _decisions, evaluation_metrics  # noqa: E402
from histopilot.training.module import classification_metrics, prediction_rows  # noqa: E402
from histopilot.training.refit import RefitDataModule  # noqa: E402
from histopilot.workers.train_batch import collect_results  # noqa: E402

support = runpy.run_path(str(Path(__file__).with_name("test_mil_training.py")))


@pytest.mark.parametrize("positive_first", [True, False])
@pytest.mark.parametrize("threshold", [0.5, 0.8])
@pytest.mark.parametrize("aggregation", ["mean_probabilities", "mean_logits"])
def test_frozen_threshold_gives_identical_development_and_external_metrics(
    positive_first, threshold, aggregation
):
    classes = ["positive", "negative"] if positive_first else ["negative", "positive"]
    target = {
        "task": "binary_classification",
        "classes": classes,
        "positiveClass": "positive",
        "unit": "patient",
    }
    records = []
    for index, positive_probability in enumerate([0.45, 0.95, 0.5, 0.5]):
        probabilities = [positive_probability, 1 - positive_probability]
        if not positive_first:
            probabilities.reverse()
        label = "positive" if index < 2 else "negative"
        records.append(
            {
                "slideId": f"s{index}",
                "patientId": f"p{index // 2}",
                "label": label,
                "labelIndex": classes.index(label),
                "probabilities": probabilities,
                "logProbabilities": np.log(probabilities).tolist(),
            }
        )
    development = classification_metrics(records, target, aggregation, decision_threshold=threshold)
    patients = patient_predictions(
        records, "mean" if aggregation == "mean_probabilities" else aggregation
    )
    for unit, rows in (("slide", records), ("patient", patients)):
        external = evaluation_metrics(
            _decisions(copy.deepcopy(rows), target, threshold), target, threshold
        )
        for metric, expected in development[unit].items():
            assert external[metric] == expected
    assert development["decisionThreshold"] == threshold
    alternative = classification_metrics(records, target, aggregation, decision_threshold=0.2)
    for unit in ("slide", "patient"):
        assert development[unit]["auroc"] == alternative[unit]["auroc"]
        assert development[unit]["auprc"] == alternative[unit]["auprc"]


def test_patient_identity_provenance_survives_dataset_collation_and_predictions(tmp_path):
    plan = support["tiny_plan"](tmp_path)
    plan["target"]["unit"] = "slide"
    plan["recipe"]["analysis"] = {"bootstrapResamples": 200}
    for row in plan["data"]["memberships"]:
        row["patientIdSource"] = "slide_fallback"
    data = MILDataModule({**plan, **plan["data"]})
    try:
        batch = next(iter(data.val_dataloader()))
        records = prediction_rows(batch, torch.zeros(len(batch["labels"]), 2), plan["target"])
        assert {row["patientIdSource"] for row in records} == {"slide_fallback"}
        metrics = classification_metrics(
            records, plan["target"], analysis=plan["recipe"]["analysis"]
        )
        assert metrics["slide"]["available"]
        assert not metrics["patient"]["available"]
        assert not metrics["patientAnalysis"]["uncertainty"]["available"]
        assert "verified patient" in metrics["patientAnalysis"]["uncertainty"]["reason"]
        with pytest.raises(ValueError, match="fallback"):
            patient_predictions(records)
    finally:
        data.teardown()


def test_patient_targets_cannot_use_fallback_identity_even_without_bootstrap(tmp_path):
    plan = support["tiny_plan"](tmp_path)
    for row in plan["data"]["memberships"]:
        row["patientIdSource"] = "slide_fallback"
    plan["recipe"]["analysis"] = None
    with pytest.raises(ValueError, match="verified patient"):
        validate_memberships({**plan, **plan["data"]})
    for row in plan["data"]["memberships"]:
        row.update(partition="train", phase="refit")
    with pytest.raises(ValueError, match="verified patient"):
        RefitDataModule(plan)


def test_default_slide_training_assigns_equal_total_weight_to_patients(tmp_path):
    plan = support["tiny_plan"](tmp_path)
    rows = [row for row in plan["data"]["memberships"] if row["partition"] == "train"]
    for index, row in enumerate(rows):
        row["patientId"] = "many-slides" if index < 3 else "single-slide"
        row["label"] = "class-0" if index < 3 else "class-1"
    data = MILDataModule({**plan, **plan["data"]})
    try:
        data.setup("fit")
        totals = {}
        for row in rows:
            totals[row["patientId"]] = (
                totals.get(row["patientId"], 0) + data.train_dataset.loss_weights[row["slideId"]]
            )
        assert totals["many-slides"] == pytest.approx(totals["single-slide"])
        assert sum(totals.values()) == pytest.approx(len(rows))
    finally:
        data.teardown()


def oof_evidence(tmp_path):
    target = support["target"]()
    runs, memberships, predictions = [], {}, {}
    for fold in range(2):
        split = f"fold-{fold}"
        rows = [
            {
                "slideId": f"s{fold}-{label}",
                "patientId": f"p{fold}-{label}",
                "patientIdSource": "crosswalk",
                "label": f"class-{label}",
                "labelIndex": label,
                "probabilities": [0.7, 0.3] if label == 0 else [0.2, 0.8],
            }
            for label in range(2)
        ]
        path = tmp_path / f"assessment-{fold}.json"
        path.write_text(json.dumps({"classOrder": target["classes"], "records": rows}))
        predictions[split] = path
        memberships[split] = [{**row, "partition": "test"} for row in rows]
        runs.append(
            {
                "id": split,
                "candidateId": "candidate",
                "splitPlanId": split,
                "trainingSeed": 42,
                "status": "completed",
                "result": {"predictions": {"assessment": str(path)}},
            }
        )
    plan = {
        "batchId": "batch",
        "protocolId": "protocol",
        "target": target,
        "configurations": [
            {
                "id": "candidate",
                "number": 1,
                "recipe": {"decisionThreshold": 0.8, "analysis": {"bootstrapResamples": 200}},
            }
        ],
        "runs": [
            {key: row[key] for key in ("id", "candidateId", "trainingSeed", "splitPlanId")}
            for row in runs
        ],
        "memberships": memberships,
        "splitPlans": [{"id": f"fold-{fold}", "seed": 42} for fold in range(2)],
    }
    return plan, {"status": "completed", "runs": runs}, predictions


def test_oof_completeness_uses_every_frozen_run_not_only_present_state(tmp_path):
    plan, state, _ = oof_evidence(tmp_path)
    state["runs"].pop()
    collect_results(plan, state, tmp_path)
    candidate = json.loads((tmp_path / "results.json").read_text())["candidates"][0]
    assert not candidate["complete"]
    assert candidate["completedRuns"] == 1 and candidate["totalRuns"] == 2
    assert candidate["metrics"] is None


@pytest.mark.parametrize("tamper", ["patient_crosses_folds", "patient_source", "run_identity"])
def test_oof_refuses_fold_or_provenance_drift(tmp_path, tamper):
    plan, state, paths = oof_evidence(tmp_path)
    if tamper == "run_identity":
        state["runs"][1]["trainingSeed"] = 99
    else:
        rows = json.loads(paths["fold-1"].read_text())
        if tamper == "patient_crosses_folds":
            patient = plan["memberships"]["fold-0"][0]["patientId"]
            plan["memberships"]["fold-1"][0]["patientId"] = patient
            rows["records"][0]["patientId"] = patient
        else:
            rows["records"][0]["patientIdSource"] = "different-source"
        paths["fold-1"].write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="fold|source|seed"):
        collect_results(plan, state, tmp_path)


def test_oof_restores_identity_provenance_and_honors_frozen_threshold(tmp_path):
    plan, state, paths = oof_evidence(tmp_path)
    plan["target"]["unit"] = "slide"
    for split, path in paths.items():
        payload = json.loads(path.read_text())
        for row, membership in zip(payload["records"], plan["memberships"][split], strict=True):
            row.pop("patientIdSource")  # Older prediction artifacts omitted this field.
            membership["patientIdSource"] = "slide_fallback"
        path.write_text(json.dumps(payload))
    collect_results(plan, state, tmp_path)
    candidate = json.loads((tmp_path / "results.json").read_text())["candidates"][0]
    assert candidate["metricDetails"]["decisionThreshold"] == 0.8
    assert not candidate["metricDetails"]["patient"]["available"]
    assert not candidate["metricDetails"]["patientAnalysis"]["uncertainty"]["available"]
