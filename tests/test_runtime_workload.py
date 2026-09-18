"""Runtime advice preserves fitting-only bag planning and full evaluation budgets."""

from copy import deepcopy

import pytest

from histopilot.application.runtime_workload import workload_for_recipe, workload_key


def profile(recipe=None, *, entries=None, memberships=None):
    rows = memberships or [
        {"slideId": "a", "patientId": "a", "partition": "train"},
        {"slideId": "b", "patientId": "b", "partition": "train"},
        {"slideId": "c", "patientId": "c", "partition": "val"},
        {"slideId": "d", "patientId": "d", "partition": "test"},
    ]
    files = entries or {
        identity: {"patchCount": count, "dimensions": 1024}
        for identity, count in zip("abcd", [100, 300, 2000, 8000], strict=True)
    }
    return workload_for_recipe(recipe or {}, rows, files, "protocol", "bundle", "mmap", 2)


def test_automatic_training_cap_excludes_validation_and_assessment_sizes():
    result = profile(
        {"model": "nnmil", "bagSizeMode": "training_median", "batchSize": 32, "evalBatchSize": 1}
    )
    assert result["trainingPatches"] == 100
    assert result["evaluationPatches"] == result["sourcePatches"] == 8000
    assert result["batchSize"] == 32 and result["evalBatchSize"] == 1


def test_full_bags_and_independent_evaluation_cap():
    whole = profile({"bagSize": None, "batchSize": 3})
    assert whole["trainingPatches"] == 300 and whole["evaluationPatches"] == 8000
    assert whole["evalBatchSize"] == 3
    capped = profile({"bagSize": 50, "evalBagSize": 20})
    assert capped["trainingPatches"] == 50 and capped["evaluationPatches"] == 20
    assert capped["sourcePatches"] == 8000


def test_curriculum_uses_largest_future_bag_and_does_not_change_recipe():
    recipe = {
        "bagSize": 30,
        "bagCurriculum": True,
        "bagCurriculumStart": 20,
        "bagCurriculumEnd": 250,
    }
    before = deepcopy(recipe)
    assert profile(recipe)["trainingPatches"] == 250
    assert recipe == before


def test_memory_identity_normalizes_defaults_and_ignores_learning_rate():
    key = workload_key({}, "p", "b", 1024, "mmap", 2)
    assert key == workload_key(
        {"learningRate": 0.003, "weightDecay": 0.0, "evalBatchSize": 1}, "p", "b", 1024, "mmap", 2
    )
    for recipe in (
        {"batchSize": 2},
        {"precision": "16-mixed"},
        {"model": "nnmil"},
        {"attentionDim": 256},
        {"evalBatchSize": 2},
        {"optimizer": "sgd"},
    ):
        assert key != workload_key(recipe, "p", "b", 1024, "mmap", 2)
    assert key != workload_key({}, "different", "b", 1024, "mmap", 2)
    assert key != workload_key({}, "p", "other", 1024, "mmap", 2)
    assert key != workload_key({}, "p", "b", 1024, "native", 2)


@pytest.mark.parametrize("problem", ["duplicate", "missing", "dimension", "count", "no_fit"])
def test_incomplete_or_ambiguous_workload_is_not_guessed(problem):
    rows = [{"slideId": "a", "partition": "train"}, {"slideId": "b", "partition": "val"}]
    files = {identity: {"patchCount": 100, "dimensions": 1024} for identity in "ab"}
    if problem == "duplicate":
        rows.append(rows[0])
    elif problem == "missing":
        del files["a"]
    elif problem == "dimension":
        files["b"]["dimensions"] = 512
    elif problem == "count":
        files["b"]["patchCount"] = 0
    else:
        rows[0]["partition"] = "test"
    with pytest.raises(ValueError):
        profile(entries=files, memberships=rows)
