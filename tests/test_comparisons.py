"""A declared comparison may vary only the model and its inputs between arms."""

import pytest
from pydantic import ValidationError

from histopilot.application import comparisons
from histopilot.application.comparisons import comparison_findings, confounders
from histopilot.models import catalog
from histopilot.schemas.development import DevelopmentBatchSpec, TrainingRecipe

INPUTS = {"protocolId": "protocol-1", "featureBundleId": "bundle-1"}


def recipe(**values):
    return TrainingRecipe.model_validate(values).model_dump()


def spec(configurations, **values):
    return DevelopmentBatchSpec.model_validate(
        {
            "experimentName": "Ablation",
            "batchName": "Arms",
            "inputs": INPUTS,
            "mode": "explicit",
            "configurations": configurations,
            "candidateSelection": "all",
            "comparison": {"reference": 1},
            **values,
        }
    )


def test_attention_ablation_ignores_options_only_attention_models_read():
    assert confounders(recipe(model="abmil"), recipe(model="mean_pool")) == []
    assert confounders(recipe(model="abmil"), recipe(model="nnmil", attentionDim=256)) == []


def test_optimisation_differences_confound_an_architecture_comparison():
    nnmil = recipe(model="nnmil", attentionDim=256, batchSize=32, maxEpochs=100, patience=10)
    assert confounders(recipe(model="abmil"), nnmil) == ["batchSize", "maxEpochs", "patience"]


def test_a_clinical_only_arm_ignores_image_settings():
    fields = [{"field": "age", "kind": "numeric"}]
    image = recipe(model="abmil", bagSize=2048)
    clinical = recipe(inputMode="clinical", clinicalFields=fields, bagSizeMode="fixed")
    combined = recipe(model="abmil", bagSize=2048, inputMode="multimodal", clinicalFields=fields)
    assert confounders(image, clinical) == []
    assert confounders(image, combined) == []


def test_every_catalog_model_declares_its_options():
    assert set(comparisons.MODEL_OPTIONS) == set(catalog.NAMES)


def test_findings_block_duplicates_and_confounded_arms():
    arms = [recipe(model="abmil"), recipe(model="mean_pool", learningRate=1e-3)]
    batch = spec(arms)
    (finding,) = comparison_findings(batch, arms)
    assert finding["code"] == "COMPARISON_CONFOUNDED" and "learningRate" in finding["message"]
    duplicate = spec([recipe(model="abmil"), recipe(model="abmil")])
    (finding,) = comparison_findings(duplicate, [recipe(model="abmil")])
    assert finding["code"] == "COMPARISON_DUPLICATE_ARMS"


TWO_ARMS = [{"model": "abmil"}, {"model": "mean_pool"}]


@pytest.mark.parametrize(
    "arms, values, message",
    [
        (TWO_ARMS, {"candidateSelection": "best_validation"}, "build all configurations"),
        (TWO_ARMS, {"comparison": {"reference": 3}}, "reference arm"),
        ([{"model": "abmil"}], {}, "2 to 8 explicit configurations"),
    ],
)
def test_comparisons_need_explicit_arms_all_built(arms, values, message):
    with pytest.raises(ValidationError, match=message):
        spec(arms, **values)


def test_batches_without_a_comparison_keep_their_serialized_shape():
    plain = DevelopmentBatchSpec.model_validate(
        {"experimentName": "E", "batchName": "B", "inputs": INPUTS}
    ).model_dump()
    assert "comparison" not in plain
    declared = spec([{"model": "abmil"}, {"model": "mean_pool"}]).model_dump()
    assert declared["comparison"] == {"reference": 1, "primaryMetric": "auroc"}
