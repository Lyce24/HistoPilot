"""Launch planning pins sampling metadata and rejects incompatible recipes."""

from copy import deepcopy

import pytest
from test_training_execution import execution, rewrite_batch

from histopilot.schemas.training_controls import validate_training_controls
from histopilot.storage.project_lock import StorageError
from histopilot.workers.train_batch import _run_plan

__all__ = ["execution"]


def changed_recipe(service, batch, **changes):
    def update(manifest):
        manifest["spec"]["recipe"].update(changes)
        for configuration in manifest["configurations"]:
            configuration["recipe"].update(changes)

    return rewrite_batch(service, batch, update)


def test_cohort_plan_pins_dataset_values_and_worker_binds_selected_column(execution):
    service, batch, executor, _ = execution
    changed = changed_recipe(service, batch, samplingStrategy="cohort_balanced")
    plan, _freshness = service._prepare(changed)
    assert set(plan["data"]["cohortValues"]) == {"cohort"}
    pinned = plan["data"]["cohortValues"]["cohort"]
    expected = {row["slideId"] for group in plan["memberships"].values() for row in group}
    assert set(pinned) == expected
    assert set(pinned.values()) == {"development"}
    before = deepcopy(plan)
    for run in plan["runs"]:
        selected = _run_plan(plan, run, None)
        assert selected["data"]["memberships"]
        assert all(row["cohort"] == "development" for row in selected["data"]["memberships"])
    assert plan == before
    assert not executor.launches


def test_default_recipe_keeps_legacy_plan_without_cohort_metadata(execution):
    service, batch, _, _ = execution
    plan, _ = service._prepare(batch)
    assert "cohortValues" not in plan["data"]
    selected = _run_plan(plan, plan["runs"][0], None)
    assert all("cohort" not in row for row in selected["data"]["memberships"])


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"samplingStrategy": "cohort_balanced", "cohortColumn": "missing_column"}, "cohort"),
        ({"classWeights": [1.0, 2.0, 3.0]}, "one class weight"),
    ],
)
def test_launch_planning_rejects_incompatible_data_controls(execution, changes, message):
    service, batch, executor, _ = execution
    changed = changed_recipe(service, batch, **changes)
    with pytest.raises(StorageError, match=message) as error:
        service._prepare(changed)
    assert error.value.code == "TRAINING_RECIPE_UNAVAILABLE"
    assert not executor.launches


def test_binary_bce_and_fold_class_weights_are_accepted_by_planner(execution):
    service, batch, executor, _ = execution
    changed = changed_recipe(service, batch, lossType="bce", classWeighting="inverse_prevalence")
    plan, _ = service._prepare(changed)
    assert plan["target"]["task"] == "binary_classification"
    assert all(config["recipe"]["lossType"] == "bce" for config in plan["configurations"])
    assert not executor.launches


def test_multiclass_bce_and_missing_training_class_are_rejected():
    target = {"task": "multiclass_classification", "classes": ["a", "b", "c"], "unit": "patient"}
    rows = [{"slideId": "s", "patientId": "p", "partition": "train", "label": "a"}]
    with pytest.raises(ValueError, match="BCE requires a binary"):
        validate_training_controls({"lossType": "bce"}, target, rows)
    for field, value in (("classWeighting", "inverse_prevalence"), ("classWeightedSampling", True)):
        with pytest.raises(ValueError, match="every target class"):
            validate_training_controls(
                {field: value, "checkpointMetric": "validation_loss"}, target, rows
            )
