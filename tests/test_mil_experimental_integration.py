"""Synthetic fits verify experimental controls survive publication and inference."""

import copy
import json
import runpy
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

from histopilot.application.clinical import _patient_records  # noqa: E402
from histopilot.application.predictors import checkpoint_snapshot  # noqa: E402
from histopilot.storage.packed import _stamp  # noqa: E402
from histopilot.training.fold import train_fold  # noqa: E402
from histopilot.training.inference import evaluate  # noqa: E402
from histopilot.training.module import MILTrainModule  # noqa: E402
from histopilot.training.refit import train_refit  # noqa: E402
from histopilot.workers.train_batch import collect_results  # noqa: E402

support = runpy.run_path(str(Path(__file__).with_name("test_mil_training.py")))


def test_out_of_fold_results_use_the_candidate_patient_aggregation(tmp_path):
    target = support["target"]()
    batch = {
        "batchId": "batch",
        "protocolId": "protocol",
        "target": target,
        "configurations": [{"id": "candidate", "recipe": {"patientAggregation": "mean_logits"}}],
        "splitPlans": [{"id": f"fold-{i}", "seed": 42} for i in range(2)],
        "memberships": {},
    }
    runs = []
    for index in range(2):
        rows = [
            {
                "slideId": f"s{index}-{slide}",
                "patientId": f"p{index}",
                "label": f"class-{index}",
                "labelIndex": index,
                "probabilities": [probability, 1 - probability],
                "logProbabilities": np.log([probability, 1 - probability]).tolist(),
            }
            for slide, probability in enumerate((0.99, 0.4))
        ]
        batch["memberships"][f"fold-{index}"] = [{**row, "partition": "test"} for row in rows]
        path = tmp_path / f"predictions-{index}.json"
        path.write_text(json.dumps({"classOrder": target["classes"], "records": rows}))
        runs.append(
            {
                "id": f"run-{index}",
                "candidateId": "candidate",
                "trainingSeed": 42,
                "splitPlanId": f"fold-{index}",
                "status": "completed",
                "result": {"predictions": {"assessment": str(path)}},
            }
        )
    collect_results(batch, {"status": "completed", "runs": runs}, tmp_path)
    results = json.loads((tmp_path / "results.json").read_text())
    summary = results["candidates"][0]["metricDetails"]
    assert summary["patientAggregation"] == "mean_logits"
    assert abs(summary["patient"]["loss"] - summary["slide"]["loss"]) > 0.01


def test_small_positive_validation_fallback_uses_final_epoch_and_resumes_assessment(tmp_path):
    plan = support["tiny_plan"](tmp_path)
    plan["recipe"].update(
        maxEpochs=3,
        minEpochs=0,
        earlyStopping=True,
        patience=1,
        checkpointMetric="validation_auroc",
        minValidationPositives=2,
        fixedEpochBudget=2,
        lossType="bce",
        patientAggregation="mean_logits",
    )
    for row in plan["data"]["memberships"]:
        if row["partition"] == "val":
            row["label"] = "class-1"
    result = train_fold(plan, tmp_path / "fit")
    assert result["epochsCompleted"] == result["bestEpoch"] == 2
    assert result["bestCheckpointPath"] == result["lastCheckpointPath"]
    assert result["stoppingDecision"]["positivePatients"] == 0
    assert result["stoppingDecision"]["checkpointSelection"] == "final_epoch"
    assert result["metrics"]["validation"]["selected"]["auroc"] is None
    assert result["metrics"]["validation"]["patientAggregation"] == "mean_logits"
    resumed = train_fold(plan, tmp_path / "fit", checkpoint_path=result["lastCheckpointPath"])
    assert resumed["assessmentOnlyResume"]
    assert resumed["metrics"] == result["metrics"]


def test_explicit_fixed_budget_uses_final_epoch_and_original_cosine_horizon(tmp_path):
    plan = support["tiny_plan"](tmp_path, classes=3)
    plan["recipe"].update(
        maxEpochs=5, minEpochs=0, earlyStopping=True, fixedEpochBudget=2, lrScheduler="cosine"
    )
    result = train_fold(plan, tmp_path / "fixed")
    assert result["epochsCompleted"] == result["bestEpoch"] == 2
    assert result["bestCheckpointPath"] == result["lastCheckpointPath"]
    assert result["stoppingDecision"]["reason"] == "explicit_fixed_epoch_budget"
    selected = MILTrainModule.load_from_checkpoint(result["bestCheckpointPath"], weights_only=True)
    assert selected.recipe["maxEpochs"] == 5
    assert selected.history[1]["learningRate"] > plan["recipe"]["learningRate"] * 0.5


@pytest.mark.parametrize(
    "model,loss", [("abmil", "bce"), ("mean_pool", "focal"), ("max_pool", "ce")]
)
def test_experimental_recipes_fit_reload_and_refit_without_validation(tmp_path, model, loss):
    plan = support["tiny_plan"](tmp_path)
    plan["recipe"].update(
        model=model,
        lossType=loss,
        maxEpochs=2,
        minEpochs=0,
        classWeighting="inverse_prevalence",
        samplingStrategy="patient_natural",
        instanceDropout=0.2,
        featureNoiseStd=0.05,
        bagCurriculum=True,
        bagCurriculumStart=2,
        bagCurriculumEnd=4,
        bagCurriculumWarmupEpochs=1,
        evalBagSize=3,
        evalBatchSize=1,
        lrScheduler="step",
        lrStepSize=1,
        lrGamma=0.5,
        aggregatorLearningRate=0.005,
        headLearningRate=0.01,
        patientAggregation="mean_logits",
    )
    result = train_fold(plan, tmp_path / "fit")
    loaded = MILTrainModule.load_from_checkpoint(result["bestCheckpointPath"], weights_only=True)
    assert loaded.recipe.get("lossType", "ce") == loss
    assert result["resolvedClassWeights"] == [1.0, 1.0]
    assert "patient_natural" in result["trainingObjective"]
    refit = copy.deepcopy(plan)
    refit["recipe"].update(minEpochs=2, earlyStopping=False)
    refit["epochBudget"] = {"epochs": 2, "percentile": 75}
    for row in refit["data"]["memberships"]:
        row.update(partition="train", phase="refit")
    final = train_refit(refit, tmp_path / "refit")
    assert final["epochsCompleted"] == 2
    assert final["resolvedClassWeights"] == [1.0, 1.0]
    assert not final["validationUsed"] and not final["testDataUsed"]


def test_bce_logit_ensemble_patient_scoring_cache_and_clinical_evidence(tmp_path):
    plan = support["tiny_plan"](tmp_path)
    plan["recipe"].update(maxEpochs=1, lossType="bce", patientAggregation="mean_logits")
    fit = train_fold(plan, tmp_path / "fit")
    checkpoints = []
    for index, bias in enumerate((5.0, -1.0)):
        payload = torch.load(fit["bestCheckpointPath"], weights_only=True)
        payload["state_dict"]["model.classifier.weight"].zero_()
        payload["state_dict"]["model.classifier.bias"].fill_(bias)
        path = tmp_path / f"member-{index}.ckpt"
        torch.save(payload, path)
        checkpoints.append(checkpoint_snapshot(path, tmp_path))
    data = copy.deepcopy(plan["data"])
    data["memberships"] = [row for row in data["memberships"] if row["partition"] == "test"]
    for row in data["memberships"]:
        row["patientId"] = f"patient-{row['label']}"
    selected = {row["slideId"] for row in data["memberships"]}
    data["featureFiles"] = {
        key: row for key, row in data["featureFiles"].items() if key in selected
    }
    for entry in data["featureFiles"].values():
        entry.update(_stamp(Path(entry["path"]).stat()))
    data["sourceStamps"] = {row["path"]: row for row in data["featureFiles"].values()}
    evaluation = {
        "runId": "evaluation-logits",
        "method": "ensemble",
        "aggregation": "mean_logit",
        "target": plan["target"],
        "data": data,
        "resources": plan["resources"],
        "device": "cpu",
        "checkpoints": checkpoints,
        "inference": {
            "precision": "float32",
            "numWorkers": 0,
            "batchSize": 2,
            "patientAggregation": "mean_logits",
            "decisionThreshold": 0.5,
        },
    }
    result = evaluate(evaluation, tmp_path / "evaluation")
    artifact = json.loads((tmp_path / "evaluation/predictions.json").read_text())
    expected = 1 / (1 + np.exp(-2.0))
    assert artifact["records"][0]["probabilities"][0] == pytest.approx(expected)
    assert len(artifact["records"][0]["probabilities"]) == 2
    _patient_records(
        artifact["records"], artifact["patientRecords"], plan["target"]["classes"], "mean_logits"
    )
    assert evaluate(evaluation, tmp_path / "evaluation")["artifacts"] == result["artifacts"]
    probability_plan = {**evaluation, "aggregation": "mean_probability"}
    evaluate(probability_plan, tmp_path / "probabilities")
    other = json.loads((tmp_path / "probabilities/predictions.json").read_text())
    assert abs(other["records"][0]["probabilities"][0] - expected) > 0.1
