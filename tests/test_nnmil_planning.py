"""Automatic bags are fitting-only, reviewable, immutable and backward compatible."""

from copy import deepcopy

import pytest
from pydantic import ValidationError
from test_training_control_preview import preview_context

from histopilot.schemas.development import DevelopmentBatchSpec, TrainingRecipe
from histopilot.schemas.nnmil import resolve_nnmil_plan, resolve_nnmil_recipe
from histopilot.schemas.training_controls import validate_training_controls

__all__ = ["preview_context"]


def fixture_plan(**changes):
    recipe = TrainingRecipe(model="nnmil", attentionDim=256, bagSizeMode="training_median",
                            batchSize=32, **changes).model_dump()
    rows = [{"slideId": str(i), "patientId": f"p{i}", "label": str(i % 2),
             "partition": "train" if i < 3 else "val"} for i in range(4)]
    files = {str(i): {"patchCount": count, "dimensions": 1024, "path": f"{i}.h5"}
             for i, count in enumerate([1, 11583, 20000, 999999])}
    return {"recipe": recipe, "data": {"memberships": rows, "featureFiles": files}}


def test_half_median_rounds_down_and_ignores_evaluation_data():
    plan = fixture_plan()
    before = deepcopy(plan)
    resolved = resolve_nnmil_plan(plan)
    assert resolved["effectiveRecipe"]["bagSize"] == 5791
    assert resolved["recipe"]["bagSize"] == 4096
    summary = resolved["nnmilPlanning"]
    assert summary["trainingSlideCount"] == 3
    assert summary["trainingPatientCount"] == 3
    assert summary["windowCount"] == 13
    assert summary["paddedSlides"] == 1 and summary["truncatedSlides"] == 2
    assert plan == before
    for role in ["val", "test"]:
        changed = deepcopy(plan)
        changed["data"]["memberships"][-1].update(partition=role, label="changed")
        changed["data"]["featureFiles"]["3"].update(patchCount=1, dimensions=999)
        assert resolve_nnmil_plan(changed)["nnmilPlanning"] == summary


def test_override_and_full_bag_modes_preserve_user_choices():
    for cap in [17, None]:
        plan = fixture_plan()
        plan["recipe"].update(bagSizeMode="fixed", bagSize=cap)
        result = resolve_nnmil_plan(plan)
        assert result["effectiveRecipe"]["bagSize"] == cap
        assert result["nnmilPlanning"]["mode"] == "fixed"


def test_direct_plan_defaults_match_the_training_schema():
    plan = fixture_plan()
    plan["recipe"] = {"model": "nnmil"}
    resolved = resolve_nnmil_plan(plan)
    recipe = TrainingRecipe.model_validate(resolved["effectiveRecipe"], context={"legacy": True})
    summary = resolved["nnmilPlanning"]
    assert recipe.bagSize == summary["bagSize"] == 4096
    assert recipe.attentionDim == 384 and summary["windowCount"] == 8
    assert summary["inputMemoryMiB"] == recipe.batchSize * recipe.bagSize * 1024 * 4 / 1024**2


def test_fraction_and_small_bag_safeguard_are_explicit():
    plan = fixture_plan(bagSizeFraction=0.25)
    assert resolve_nnmil_plan(plan)["effectiveRecipe"]["bagSize"] == 2895
    for value in plan["data"]["featureFiles"].values():
        value["patchCount"] = 1
    assert resolve_nnmil_plan(plan)["effectiveRecipe"]["bagSize"] == 1


@pytest.mark.parametrize("change", ["count", "dimensions", "duplicate", "missing", "external"])
def test_bad_fitting_membership_is_rejected(change):
    plan = fixture_plan()
    if change == "count":
        plan["data"]["featureFiles"]["0"]["patchCount"] = 0
    elif change == "dimensions":
        plan["data"]["featureFiles"]["0"]["dimensions"] = 128
    elif change == "duplicate":
        plan["data"]["memberships"].append(plan["data"]["memberships"][0])
    elif change == "missing":
        del plan["data"]["featureFiles"]["0"]
    else:
        plan["data"]["memberships"][0]["pool"] = "external_test"
    with pytest.raises(ValueError):
        resolve_nnmil_plan(plan)


@pytest.mark.parametrize("key", ["effectiveRecipe", "nnmilPlanning"])
def test_resolution_cannot_be_changed_after_freezing(key):
    plan = resolve_nnmil_plan(fixture_plan())
    plan[key]["bagSize"] = 18
    with pytest.raises(ValueError, match="Frozen"):
        resolve_nnmil_plan(plan)


def test_legacy_shapes_and_worker_plan_remain_unchanged():
    recipe = TrainingRecipe().model_dump()
    assert not any(key.startswith("nnmil") for key in recipe)
    assert "bagSizeMode" not in recipe and "weightDecayPolicy" not in recipe
    plan = {"recipe": recipe}
    assert resolve_nnmil_plan(plan) is plan


@pytest.mark.parametrize("changes", [
    {"bagSizeFraction": 0}, {"bagSizeFraction": float("nan")},
    {"bagSizeMode": "training_median", "bagCurriculum": True},
    {"lrScheduleInterval": "step", "lrScheduler": "plateau"},
    {"nnmilWindowStrideDivisor": 0}, {"nnmilWindowSeed": True},
    {"nnmilBatchSampler": "class_balanced", "samplingStrategy": "patient_natural"},
])
def test_invalid_parameter_combinations_are_rejected(changes):
    with pytest.raises(ValidationError):
        TrainingRecipe(model="nnmil", **changes)


def test_paper_optimizer_rates_are_defaults_and_schedule_remains_configurable():
    recipe = TrainingRecipe(model="nnmil", learningRate=3e-4, weightDecay=1e-4,
                            weightDecayPolicy="weights_only", lrScheduler="cosine",
                            lrScheduleInterval="step", finalLrFraction=0)
    assert recipe.finalLrFraction == 0
    assert TrainingRecipe().learningRate == 3e-4
    assert TrainingRecipe().weightDecay == 1e-4


def test_balanced_batch_preflight_requires_each_class_and_sufficient_batch():
    plan = fixture_plan(nnmilBatchSampler="class_balanced")
    recipe = {**plan["recipe"], "batchSize": 1}
    target = {"classes": ["0", "1"], "unit": "patient", "task": "binary_classification"}
    with pytest.raises(ValueError, match="one slide per class"):
        validate_training_controls(recipe, target, plan["data"]["memberships"])


def test_preview_exposes_one_fitting_summary_per_fold(preview_context, monkeypatch):
    context = preview_context
    files = [{"slideId": row["slideId"], "patchCount": 101 if row["partition"] == "train"
              else 999999, "dimensions": 1536} for row in context.protocol["memberships"]
             if row["fold"] == 0]
    monkeypatch.setattr(context.service.store, "get_configuration",
                        lambda key: {"manifest": {"files": files} if key == "features"
                                     else deepcopy(context.protocol)})
    monkeypatch.setattr("histopilot.application.development.MILInputService.preview",
                        lambda *_: {"canPlan": True, "findings": [], "featureSetId": "features"})
    spec = DevelopmentBatchSpec.model_validate({**context.spec.model_dump(), "recipe": {
        "model": "nnmil", "attentionDim": 256, "bagSizeMode": "training_median"}})
    preview = context.service.preview(spec)
    assert preview["canFreeze"], preview["findings"]
    assert len(preview["nnmilPlanning"]) == 2
    for item in preview["nnmilPlanning"]:
        assert item["bagSize"] == 50
        assert item["windowCount"] == 21
        assert item["trainingSlideCount"] == 4
        assert item["candidateId"] == preview["configurations"][0]["id"]
    for entry in files:
        if entry["patchCount"] == 999999:
            entry["patchCount"] = 2
    assert context.service.preview(spec)["nnmilPlanning"] == preview["nnmilPlanning"]


def test_full_development_refit_resolves_its_own_fitting_population():
    plan = fixture_plan()
    _, fold = resolve_nnmil_recipe(plan["recipe"], plan["data"]["memberships"],
                                   plan["data"]["featureFiles"])
    refit_rows = [{**row, "partition": "train", "phase": "refit"}
                  for row in plan["data"]["memberships"]]
    _, refit = resolve_nnmil_recipe(plan["recipe"], refit_rows, plan["data"]["featureFiles"])
    assert refit["bagSize"] == 7895 and refit["trainingSlideCount"] == 4
    assert fold["fingerprint"] != refit["fingerprint"]
