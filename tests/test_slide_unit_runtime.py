"""Explicit slide experiments never group or score by patient metadata."""
import copy
import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("lightning")

from test_mil_training import tiny_plan  # noqa: E402

from histopilot.datasets.mil import MILDataError, validate_memberships  # noqa: E402
from histopilot.schemas.training_controls import resolve_stopping, validate_split_unit  # noqa: E402
from histopilot.training import module  # noqa: E402
from histopilot.training.fold import _validate_plan, train_fold  # noqa: E402
from histopilot.training.refit import RefitDataModule  # noqa: E402
from histopilot.workers.packing_process import write_json  # noqa: E402
from histopilot.workers.train_batch import collect_results  # noqa: E402


def slide_plan(tmp_path):
    plan = tiny_plan(tmp_path)
    plan["splitUnit"] = plan["target"]["unit"] = "slide"
    for index, row in enumerate(plan["data"]["memberships"]):
        row["patientId"] = None if index % 3 == 0 else "shared-patient"
    return plan


def forbid_patient_work(*args, **kwargs):
    raise AssertionError("A slide-level experiment invoked patient aggregation")


def test_slide_memberships_preserve_exact_rows_and_ignore_patient_metadata(tmp_path):
    plan = slide_plan(tmp_path)
    before = copy.deepcopy(plan)
    _validate_plan(plan)
    grouped = validate_memberships({**plan, **plan["data"]})
    assert plan == before
    assert {row["slideId"] for rows in grouped.values() for row in rows} == set(plan["data"]["featureFiles"])
    assert any(row["patientId"] is None for row in grouped["train"])
    damaged = copy.deepcopy(plan)
    damaged["data"]["memberships"].append(damaged["data"]["memberships"][0])
    with pytest.raises(ValueError, match="same slide"):
        _validate_plan(damaged)
    with pytest.raises(MILDataError, match="Duplicate slide"):
        validate_memberships({**damaged, **damaged["data"]})
    legacy = copy.deepcopy(plan)
    legacy.pop("splitUnit")
    for row in legacy["data"]["memberships"]:
        row["patientId"] = "shared-patient"
    with pytest.raises(ValueError, match="Patient groups overlap"):
        _validate_plan(legacy)


def test_slide_validation_positive_threshold_counts_slides(tmp_path):
    plan = slide_plan(tmp_path)
    recipe = {**plan["recipe"], "minValidationPositives": 2, "fixedEpochBudget": 1}
    rows = plan["data"]["memberships"]
    for row in rows:
        row["patientId"] = "one-patient"
    assert resolve_stopping(recipe, plan["target"], rows, split_unit="slide")[1] is None
    assert resolve_stopping(recipe, plan["target"], rows)[1]["positivePatients"] == 1
    decision = resolve_stopping({**recipe, "minValidationPositives": 3}, plan["target"], rows, split_unit="slide")[1]
    assert decision["positiveSlides"] == 2
    assert decision["reason"] == "insufficient_validation_positive_slides"


@pytest.mark.parametrize("recipe", [{"samplingStrategy": "patient_natural"}, {"samplingStrategy": "cohort_balanced"}, {"inputMode": "clinical"}])
def test_slide_mode_rejects_patient_computation_controls(recipe):
    with pytest.raises(ValueError, match="[Pp]atient"):
        validate_split_unit(recipe, {"unit": "slide"}, "slide")


def test_slide_cpu_fit_scoring_checkpoint_and_refit_never_aggregate_patients(tmp_path, monkeypatch):
    plan = slide_plan(tmp_path)
    plan["recipe"]["maxEpochs"] = 1
    plan["recipe"]["analysis"] = {"bootstrapResamples": 200}
    monkeypatch.setattr(module, "aggregate_patients", forbid_patient_work)
    import histopilot.statistics
    monkeypatch.setattr(histopilot.statistics, "patient_analysis", forbid_patient_work)
    result = train_fold(plan, tmp_path / "run")
    for role in ("validation", "assessment"):
        document = json.loads(open(result["predictions"][role]).read())
        assert len(document["records"]) == 4
        assert document["patientRecords"] is None
        assert result["metrics"][role]["unit"] == "slide"
        assert result["metrics"][role]["patient"]["available"] is False
        assert "patientAnalysis" not in result["metrics"][role]
    model = module.MILTrainModule.load_from_checkpoint(result["bestCheckpointPath"], map_location="cpu", weights_only=True)
    assert model.split_unit == "slide"
    refit = copy.deepcopy(plan)
    for row in refit["data"]["memberships"]:
        row.update(partition="train", phase="refit")
    data = RefitDataModule(refit)
    assert len(data.memberships["train"]) == 12
    assert data.trainingObjective == "slide_cross_entropy"


def test_oof_slides_from_shared_patient_across_folds_are_complete(tmp_path, monkeypatch):
    target = {"unit": "slide", "task": "binary_classification", "classes": ["a", "b"], "positiveClass": "b"}
    runs, memberships = [], {}
    for fold in range(2):
        split = f"fold-{fold}"
        rows = [{"slideId": f"s-{fold}-{i}", "patientId": "shared" if i else None,
                 "label": target["classes"][i], "labelIndex": i,
                 "probabilities": [0.8, 0.2] if i == 0 else [0.2, 0.8]} for i in range(2)]
        memberships[split] = [{**row, "partition": "test"} for row in rows]
        path = tmp_path / f"{split}.json"
        write_json(path, {"classOrder": target["classes"], "records": rows})
        runs.append({"id": split, "candidateId": "candidate", "splitPlanId": split, "trainingSeed": 42,
                     "status": "completed", "result": {"predictions": {"assessment": str(path)}}})
    plan = {"batchId": "batch", "protocolId": "protocol", "target": target, "splitUnit": "slide",
            "configurations": [{"id": "candidate", "recipe": {"analysis": {"bootstrapResamples": 200}}}],
            "memberships": memberships, "splitPlans": [{"id": f"fold-{i}", "seed": 42} for i in range(2)]}
    monkeypatch.setattr(module, "aggregate_patients", forbid_patient_work)
    collect_results(plan, {"status": "completed", "runs": runs}, tmp_path)
    candidate = json.loads((tmp_path / "results.json").read_text())["candidates"][0]
    assert candidate["assessmentSlideCount"] == 4
    assert candidate["metricDetails"]["patient"]["available"] is False
    assert "patientAnalysis" not in candidate["metricDetails"]
