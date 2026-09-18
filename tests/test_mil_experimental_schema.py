"""New MIL controls are validated without changing historical recipe identities."""

import json

import pytest
from pydantic import ValidationError

from histopilot.application.development import expand_recipes
from histopilot.schemas.development import (
    EXPERIMENTAL_DEFAULTS,
    DevelopmentBatchSpec,
    SearchGrid,
    TrainingRecipe,
)


def batch(**changes):
    return {
        "experimentName": "Study", "batchName": "Baseline",
        "inputs": {"protocolId": "protocol", "featureBundleId": "bundle"},
        **changes,
    }


def test_classification_optimizer_defaults_preserve_historical_and_explicit_values():
    assert (TrainingRecipe().learningRate, TrainingRecipe().weightDecay) == (0.0003, 0.0001)
    assert (SearchGrid().learningRates, SearchGrid().weightDecays) == ([0.0003], [0.0001])
    old = TrainingRecipe.model_validate({}, context={"legacy": True})
    assert (old.learningRate, old.weightDecay, old.maxEpochs, old.patience) == (0.0003, 0.0001, 100, 15)
    old_batch = DevelopmentBatchSpec.model_validate(batch(), context={"legacy": True})
    assert old_batch.recipe == old
    assert old_batch.grid.learningRates == [0.0003]
    assert old_batch.grid.weightDecays == [0.0001]
    assert old_batch.grid.maxEpochs == [100]
    explicit = TrainingRecipe.model_validate(
        {"learningRate": 0.002, "weightDecay": 0, "maxEpochs": 20, "patience": 5},
        context={"legacy": True},
    )
    assert (explicit.learningRate, explicit.weightDecay, explicit.maxEpochs, explicit.patience) == (0.002, 0, 20, 5)


@pytest.mark.parametrize("context", [None, {"legacy": True}])
def test_previous_explicit_optimizer_values_survive_new_defaults(context):
    recipe = TrainingRecipe.model_validate(
        {"learningRate": 0.0001, "weightDecay": 0.005}, context=context
    )
    restored = TrainingRecipe.model_validate(recipe.model_dump(), context={"legacy": True})
    assert (restored.learningRate, restored.weightDecay) == (0.0001, 0.005)
    grid = SearchGrid.model_validate(
        {"learningRates": [0.0001], "weightDecays": [0.005]}, context=context
    )
    assert (grid.learningRates, grid.weightDecays) == ([0.0001], [0.005])


def test_unused_controls_preserve_legacy_serialization_and_candidate_identity():
    legacy = TrainingRecipe.model_validate({}, context={"legacy": True}).model_dump()
    assert not EXPERIMENTAL_DEFAULTS.keys() & legacy.keys()
    restored = TrainingRecipe.model_validate(legacy, context={"legacy": True})
    assert restored.model_dump() == legacy
    assert json.loads(restored.model_dump_json()) == legacy
    defaults_explicit = {**legacy, **EXPERIMENTAL_DEFAULTS}
    spec = DevelopmentBatchSpec.model_validate(
        batch(mode="explicit", configurations=[legacy, defaults_explicit]), context={"legacy": True}
    )
    assert len(expand_recipes(spec)) == 1
    assert spec.model_dump()["configurations"] == [legacy, legacy]


def test_nondefault_experimental_settings_round_trip_and_change_candidates():
    original = TrainingRecipe().model_dump()
    settings = {
        "lossType": "focal", "focalGamma": 1.5, "classWeights": [1, 2],
        "patientAggregation": "mean_logits", "ensembleAggregation": "mean_logit",
        "adamBetas": [0.85, 0.95], "adamEps": 1e-7,
        "lrScheduler": "step", "lrStepSize": 4, "lrGamma": 0.8,
        "aggregatorLearningRate": 0.0002, "headLearningRate": 0.001,
        "samplingStrategy": "cohort_balanced", "cohortColumn": "site",
        "instanceDropout": 0.1, "featureNoiseStd": 0.02,
        "bagCurriculum": True, "bagCurriculumStart": 100, "bagCurriculumEnd": 2000,
        "bagCurriculumWarmupEpochs": 8, "evalBagSize": 2048, "evalBatchSize": 4,
    }
    parsed = TrainingRecipe.model_validate({**original, **settings})
    saved = json.loads(parsed.model_dump_json())
    assert all(saved[key] == value for key, value in settings.items())
    assert TrainingRecipe.model_validate(saved) == parsed
    expanded = expand_recipes(DevelopmentBatchSpec.model_validate(batch(mode="explicit", configurations=[original, saved])))
    assert len(expanded) == 2
    assert expanded[0] != expanded[1]


@pytest.mark.parametrize("settings", [
    {"lossType": "unsupported"}, {"labelSmoothing": 1}, {"lossType": "bce", "labelSmoothing": 0.1},
    {"classWeights": [0, 1]}, {"classWeights": [1]}, {"classWeights": [1, 1], "classWeighting": "inverse_prevalence"},
    {"focalGamma": -1}, {"patientAggregation": "median"}, {"ensembleAggregation": "vote"},
    {"adamBetas": [0.9, 1]}, {"adamBetas": [0.9]}, {"adamEps": 0}, {"headLearningRate": 0},
    {"aggregatorLearningRate": -1}, {"lrStepSize": 0}, {"lrGamma": 1}, {"lrPlateauPatience": -1},
    {"lrScheduler": "plateau", "warmupEpochs": 2},
    {"samplingStrategy": "cohort_balanced", "classWeightedSampling": True},
    {"samplingPositivePrevalence": 0}, {"samplingPositivePrevalence": 1}, {"cohortColumn": ""},
    {"instanceDropout": 1}, {"featureNoiseStd": -1}, {"evalBagSize": 0}, {"evalBatchSize": 0},
    {"bagCurriculum": True, "bagCurriculumStart": 2000, "bagCurriculumEnd": 1000},
    {"bagCurriculumWarmupEpochs": 0}, {"minEpochs": -1},
    {"minValidationPositives": 5}, {"fixedEpochBudget": 0}, {"fixedEpochBudget": 41},
    {"minEpochs": 20, "fixedEpochBudget": 10},
    {"lrScheduler": "cosine", "warmupEpochs": 10, "fixedEpochBudget": 10},
])
def test_invalid_or_ambiguous_controls_are_rejected(settings):
    with pytest.raises(ValidationError):
        TrainingRecipe.model_validate(settings)


def test_ocean_epoch_policy_accepts_zero_minimum_and_explicit_fallback():
    recipe = TrainingRecipe(
        minEpochs=0, maxEpochs=40, minValidationPositives=5, fixedEpochBudget=20,
        lrScheduler="cosine", warmupEpochs=2,
    )
    assert recipe.model_dump()["fixedEpochBudget"] == 20
    assert recipe.model_dump()["minValidationPositives"] == 5
    assert TrainingRecipe(minEpochs=0, fixedEpochBudget=1).fixedEpochBudget == 1
    assert TrainingRecipe(model="future_model").model == "future_model"


def test_fixed_epoch_budget_is_checked_for_each_grid_epoch_limit():
    with pytest.raises(ValidationError, match="fixed epoch budget"):
        DevelopmentBatchSpec.model_validate(batch(
            recipe={"fixedEpochBudget": 20}, mode="grid", grid={"maxEpochs": [10, 40]},
        ))
