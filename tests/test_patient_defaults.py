"""Default patient design and validation-only configuration selection."""

import copy

import pytest

from histopilot.application.development import expand_recipes
from histopilot.candidate_selection import validation_selection
from histopilot.schemas.development import DevelopmentBatchSpec, TrainingRecipe
from histopilot.schemas.protocols import SplitSpec, TargetSpec


def test_new_patient_defaults_and_legacy_round_trip():
    recipe = TrainingRecipe()
    assert (recipe.learningRate, recipe.weightDecay) == (3e-4, 1e-4)
    assert recipe.checkpointMetric == "validation_auroc"
    assert recipe.patientAggregation == "mean_probabilities"
    assert recipe.samplingStrategy == "slide_uniform" and not recipe.classWeightedSampling
    assert recipe.classWeighting == "none" and recipe.classWeights is None
    assert recipe.analysis.bootstrapResamples == 2000
    assert recipe.analysis.confidenceLevel == 0.95
    assert recipe.decisionThreshold == 0.5
    assert SplitSpec().mode == "kfold" and SplitSpec().folds == 5 and SplitSpec().stratify
    assert TargetSpec.model_fields["unit"].default == "patient"
    spec = {
        "experimentName": "study",
        "batchName": "batch",
        "inputs": {"protocolId": "protocol", "featureBundleId": "bundle"},
    }
    current = DevelopmentBatchSpec.model_validate(spec)
    assert current.selectionMetric == "validation_auroc"
    assert current.candidateSelection == "best_validation"
    legacy = DevelopmentBatchSpec.model_validate(spec, context={"legacy": True})
    saved = legacy.model_dump()
    assert "analysis" not in saved["recipe"] and "decisionThreshold" not in saved["recipe"]
    assert "candidateSelection" not in saved and "selectionMetric" not in saved
    assert legacy.recipe.checkpointMetric == "validation_loss"
    assert (
        DevelopmentBatchSpec.model_validate(saved, context={"legacy": True}).model_dump() == saved
    )
    assert all("analysis" not in row for row in expand_recipes(legacy))


def evidence():
    configurations = [
        {"id": name, "number": i + 1, "recipe": {}} for i, name in enumerate(("a", "b"))
    ]
    runs = [
        {
            "id": f"{name}-{seed}-{fold}",
            "candidateId": name,
            "trainingSeed": seed,
            "splitPlanId": f"fold-{fold}",
        }
        for name in ("a", "b")
        for seed in (42, 43)
        for fold in range(5)
    ]
    plan = {
        "target": {"unit": "patient"},
        "selectionMetric": "validation_auroc",
        "configurations": configurations,
        "runs": runs,
    }
    state = {
        "runs": [
            {
                **run,
                "status": "completed",
                "result": {
                    "metrics": {
                        "validation": {
                            "unit": "patient",
                            "patientAggregation": "mean_probabilities",
                            "patient": {"auroc": 0.75 if run["candidateId"] == "a" else 0.8},
                        },
                        "assessment": {"patient": {"auroc": 1 if run["candidateId"] == "a" else 0}},
                    }
                },
            }
            for run in runs
        ]
    }
    return plan, state


def test_configuration_selection_averages_all_folds_and_seeds_excluding_assessment():
    plan, state = evidence()
    result = validation_selection(plan, state)
    assert result["ready"] and result["selectedCandidateId"] == "b"
    assert [row["expectedScores"] for row in result["candidates"]] == [10, 10]
    assert [row["score"] for row in result["candidates"]] == pytest.approx([0.75, 0.8])
    changed = copy.deepcopy(state)
    for row in changed["runs"]:
        row["result"]["metrics"]["assessment"] = {"patient": {"auroc": 0.5}}
    assert validation_selection(plan, changed) == result


@pytest.mark.parametrize("invalid", [None, float("nan"), True, "0.9"])
def test_selection_requires_all_valid_patient_scores(invalid):
    plan, state = evidence()
    state["runs"][0]["result"]["metrics"]["validation"]["patient"]["auroc"] = invalid
    assert not validation_selection(plan, state)["ready"]
    state["runs"][0]["result"] = None
    assert not validation_selection(plan, state)["ready"]


def test_selection_rejects_unit_or_aggregation_drift_and_has_deterministic_ties():
    plan, state = evidence()
    for row in state["runs"]:
        row["result"]["metrics"]["validation"]["patient"]["auroc"] = 0.75
    assert validation_selection(plan, state)["selectedCandidateId"] == "a"
    state["runs"][0]["result"]["metrics"]["validation"]["unit"] = "slide"
    assert not validation_selection(plan, state)["ready"]
    del plan["selectionMetric"]
    assert validation_selection(plan, state) is None
