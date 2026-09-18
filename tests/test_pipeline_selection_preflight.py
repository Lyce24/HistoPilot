"""Checkpoint and configuration selection are both feasible before fitting."""

import pytest
from test_training_control_preview import preview_context, with_recipe

__all__ = ["preview_context"]


def test_disabling_bootstrap_does_not_turn_fallback_slides_into_verified_patients(preview_context):
    context = preview_context
    context.protocol["memberships"][0]["patientIdSource"] = "slide_fallback"
    preview = context.service.preview(with_recipe(context, analysis=None))
    assert not preview["canFreeze"] and not preview["runs"]
    assert preview["findings"][0]["code"] == "VERIFIED_PATIENTS_REQUIRED"


@pytest.mark.parametrize("model", ["abmil", "nnmil", "mean_pool", "max_pool"])
def test_every_model_checks_validation_auroc_before_freezing(preview_context, model):
    context = preview_context
    for row in context.protocol["memberships"]:
        if row["partition"] == "val":
            row["label"] = "negative"
    preview = context.service.preview(with_recipe(context, model=model))
    assert not preview["canFreeze"] and not preview["runs"]
    assert preview["findings"][0]["code"] == "TRAINING_METRIC_UNAVAILABLE"
    assert "cannot select checkpoints" in preview["findings"][0]["message"]


@pytest.mark.parametrize("controls", [
    {"checkpointMetric": "validation_loss"},
    {"minValidationPositives": 2, "fixedEpochBudget": 3},
])
def test_configuration_auroc_requires_classes_even_when_checkpoint_selection_is_defined(
    preview_context, controls
):
    context = preview_context
    for row in context.protocol["memberships"]:
        if row["partition"] == "val":
            row["label"] = "negative"
    spec = with_recipe(context, **controls)
    preview = context.service.preview(spec)
    assert not preview["canFreeze"] and not preview["runs"]
    assert "cannot rank configurations" in preview["findings"][0]["message"]
    valid = context.service.preview(spec.model_copy(update={"selectionMetric": "validation_loss"}))
    assert valid["canFreeze"], valid["findings"]
    # Explicitly keeping all candidates does not require an AUROC winner.
    all_candidates = context.service.preview(spec.model_copy(update={"candidateSelection": "all"}))
    assert all_candidates["canFreeze"], all_candidates["findings"]
