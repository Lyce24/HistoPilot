"""Training controls preserve fold boundaries, patient objectives, and replay."""

# ruff: noqa: E402
from collections import Counter
from copy import deepcopy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

from histopilot.datasets.datamodule import MILDataModule
from histopilot.datasets.mil import MILDataError
from histopilot.storage.packed import _stamp


@pytest.fixture
def experimental_plan(tmp_path):
    files, rows = {}, []
    people = [
        ("p0", "negative", "A", 2),
        ("p1", "negative", "A", 1),
        ("p2", "negative", "A", 1),
        ("p3", "negative", "A", 1),
        ("p4", "positive", "A", 1),
        ("p5", "positive", "A", 1),
        ("p6", "negative", "B", 1),
        ("p7", "positive", "B", 1),
    ]
    for role in ("train", "val", "test"):
        selected = (
            people
            if role == "train"
            else [
                (f"{role}-p0", "negative", "C", 1),
                (f"{role}-p1", "positive", "C", 1),
            ]
        )
        for patient, label, cohort, slide_count in selected:
            for number in range(slide_count):
                identity = f"{patient}-s{number}"
                path = tmp_path / f"{identity}.npy"
                np.save(path, np.arange(160, dtype=np.float32).reshape(40, 4))
                files[identity] = {
                    "slideId": identity,
                    "path": str(path),
                    "patchCount": 40,
                    "dimensions": 4,
                    "dtype": "float32",
                    **_stamp(path.stat()),
                }
                rows.append(
                    {
                        "slideId": identity,
                        "patientId": patient,
                        "label": label,
                        "cohort": cohort,
                        "partition": role,
                        "pool": "development",
                        "phase": "evaluation",
                        "planId": "fold:0",
                    }
                )
    return {
        "memberships": rows,
        "featureFiles": files,
        "featureDim": 4,
        "loadingPolicy": "native",
        "trainingSeed": 29,
        "target": {
            "task": "binary_classification",
            "unit": "patient",
            "classes": ["positive", "negative"],
            "positiveClass": "positive",
        },
        "recipe": {"bagSize": 12, "batchSize": 2},
        "resources": {"dataLoaderWorkers": 0},
    }


def selected_rows(module, epoch=0):
    module.setup("fit")
    module.set_epoch(epoch)
    return [module.train_dataset.rows[index] for _, index in module.train_sampler]


def observed_batches(module):
    return [
        (batch["slideIds"], batch["features"].clone(), batch["mask"].clone())
        for batch in module.train_dataloader()
    ]


def test_patient_sampling_visits_every_patient_once_without_double_weighting(experimental_plan):
    experimental_plan["recipe"]["samplingStrategy"] = "patient_natural"
    module = MILDataModule(experimental_plan)
    rows = selected_rows(module)
    assert len(rows) == 8
    assert Counter(row["patientId"] for row in rows) == dict.fromkeys(
        [f"p{i}" for i in range(8)], 1
    )
    assert all(value == 1 for value in module.train_dataset.loss_weights.values())
    assert "one_slide_per_patient" in module.trainingObjective
    assert rows == selected_rows(module)
    assert rows == selected_rows(MILDataModule(experimental_plan))
    choices = {
        row["slideId"]
        for epoch in range(20)
        for row in selected_rows(module, epoch)
        if row["patientId"] == "p0"
    }
    assert choices == {"p0-s0", "p0-s1"}


@pytest.mark.parametrize("strategy", ["cohort_balanced", "cohort_label_balanced"])
def test_cohort_sampling_balances_train_only_and_respects_positive_class(
    experimental_plan, strategy
):
    experimental_plan["recipe"].update(
        {
            "samplingStrategy": strategy,
            "samplingPositivePrevalence": 0.75,
        }
    )
    module = MILDataModule(experimental_plan)
    rows = selected_rows(module, 3)
    assert len(rows) == 8
    assert Counter(row["cohort"] for row in rows) == {"A": 4, "B": 4}
    assert all(row["partition"] == "train" for row in rows)
    if strategy == "cohort_label_balanced":
        assert Counter((row["cohort"], row["label"]) for row in rows) == {
            ("A", "positive"): 3,
            ("A", "negative"): 1,
            ("B", "positive"): 3,
            ("B", "negative"): 1,
        }
    else:
        assert len({row["patientId"] for row in rows if row["cohort"] == "A"}) == 4


def test_missing_cohort_cells_and_patient_cohort_conflicts_fail(experimental_plan):
    experimental_plan["recipe"]["samplingStrategy"] = "cohort_label_balanced"
    missing = deepcopy(experimental_plan)
    missing["memberships"] = [row for row in missing["memberships"] if row["patientId"] != "p7"]
    with pytest.raises(MILDataError, match="both classes"):
        MILDataModule(missing).setup("fit")
    experimental_plan["memberships"][0]["cohort"] = "B"
    with pytest.raises(MILDataError, match="crosses cohorts"):
        MILDataModule(experimental_plan).setup("fit")


def test_patient_sampler_rejects_conflicting_slide_labels(experimental_plan):
    experimental_plan["target"]["unit"] = "slide"
    experimental_plan["recipe"]["samplingStrategy"] = "patient_natural"
    experimental_plan["memberships"][0]["label"] = "positive"
    with pytest.raises(MILDataError, match="conflicting labels"):
        MILDataModule(experimental_plan).setup("fit")


def test_inverse_class_sampling_and_loss_weights_are_train_fold_only(experimental_plan):
    experimental_plan["target"]["unit"] = "slide"
    experimental_plan["recipe"].update(
        {
            "classWeightedSampling": True,
            "classWeighting": "inverse_prevalence",
        }
    )
    module = MILDataModule(experimental_plan)
    assert module.training_class_weights() == [1.5, 0.75]
    total = Counter(row["label"] for epoch in range(200) for row in selected_rows(module, epoch))
    assert 0.45 < total["positive"] / sum(total.values()) < 0.55
    assert selected_rows(module, 8) == selected_rows(MILDataModule(experimental_plan), 8)
    for row in experimental_plan["memberships"]:
        if row["partition"] != "train":
            row["label"] = "positive"
    assert MILDataModule(experimental_plan).training_class_weights() == [1.5, 0.75]
    experimental_plan["recipe"]["samplingStrategy"] = "patient_natural"
    with pytest.raises(MILDataError, match="cannot be combined"):
        MILDataModule(experimental_plan).setup("fit")


@pytest.mark.parametrize("unit", ["patient", "slide"])
@pytest.mark.parametrize(
    "strategy", ["slide_uniform", "patient_natural", "cohort_balanced", "cohort_label_balanced"]
)
def test_class_weighting_follows_objective_unit_before_sampling(experimental_plan, unit, strategy):
    experimental_plan["target"]["unit"] = unit
    experimental_plan["recipe"].update(
        samplingStrategy=strategy, classWeighting="inverse_prevalence"
    )
    module = MILDataModule(experimental_plan)
    patient_objective = unit == "patient" or strategy != "slide_uniform"
    # Three positive and five negative patients; one negative has two slides.
    expected = [8 / 6, 8 / 10] if patient_objective else [9 / 6, 9 / 12]
    assert module.training_class_weights() == pytest.approx(expected)
    assert module.training_class_weight_unit() == ("patient" if patient_objective else "slide")
    # Replicating one patient's slide cannot change patient prevalence.
    duplicate = {**experimental_plan["memberships"][0], "slideId": "duplicate"}
    experimental_plan["memberships"].append(duplicate)
    for row in experimental_plan["memberships"]:
        if row["partition"] != "train":
            row["label"] = "positive"
    updated = MILDataModule(experimental_plan).training_class_weights()
    assert updated == pytest.approx(expected if patient_objective else [10 / 6, 10 / 14])


def test_patient_weights_reject_inconsistent_labels_without_opening_features(experimental_plan):
    experimental_plan["target"]["unit"] = "slide"
    experimental_plan["recipe"].update(
        samplingStrategy="patient_natural", classWeighting="inverse_prevalence"
    )
    experimental_plan["memberships"][0]["label"] = "positive"
    with pytest.raises(MILDataError, match="consistent training labels"):
        MILDataModule(experimental_plan).training_class_weights()


def test_explicit_class_order_and_missing_training_class_validation(experimental_plan):
    experimental_plan["recipe"]["classWeights"] = [2.0, 0.5]
    assert MILDataModule(experimental_plan).training_class_weights() == [2.0, 0.5]
    experimental_plan["recipe"]["classWeights"] = [2.0]
    with pytest.raises(MILDataError, match="per target class"):
        MILDataModule(experimental_plan).training_class_weights()
    experimental_plan["recipe"] = {
        "bagSize": 12,
        "batchSize": 2,
        "classWeighting": "inverse_prevalence",
    }
    for row in experimental_plan["memberships"]:
        if row["partition"] == "train":
            row["label"] = "negative"
    with pytest.raises(MILDataError, match="every class"):
        MILDataModule(experimental_plan).training_class_weights()


def test_curriculum_grows_training_cap_and_eval_cap_is_epoch_independent(experimental_plan):
    experimental_plan["recipe"].update(
        {
            "bagCurriculum": True,
            "bagCurriculumStart": 4,
            "bagCurriculumEnd": 20,
            "bagCurriculumWarmupEpochs": 4,
            "evalBagSize": 7,
            "evalBatchSize": 1,
        }
    )
    module = MILDataModule(experimental_plan)
    module.setup()
    for epoch, expected in ((0, 4), (2, 12), (4, 20), (9, 20)):
        module.set_epoch(epoch)
        assert {int(size) for batch in module.train_dataloader() for size in batch["lengths"]} == {
            expected
        }
    evaluation = next(iter(module.val_dataloader()))
    assert evaluation["features"].shape == (1, 7, 4)
    module.val_dataset.set_epoch(100)
    assert torch.equal(evaluation["features"], next(iter(module.val_dataloader()))["features"])
    assert next(iter(module.test_dataloader()))["features"].shape == (1, 7, 4)


@pytest.mark.parametrize("workers", [0, 2])
def test_augmentation_replays_across_workers_and_epoch_resume(experimental_plan, workers):
    experimental_plan["recipe"].update(
        {
            "instanceDropout": 0.3,
            "featureNoiseStd": 0.1,
            "samplingStrategy": "patient_natural",
        }
    )
    experimental_plan["resources"]["dataLoaderWorkers"] = workers
    first = MILDataModule(experimental_plan)
    resumed = MILDataModule(experimental_plan)
    try:
        first.set_epoch(4)
        expected = observed_batches(first)
        first.set_epoch(5)
        changed = observed_batches(first)
        resumed.set_epoch(4)
        actual = observed_batches(resumed)
        assert len(expected) == len(actual)
        for left, right in zip(expected, actual, strict=True):
            assert left[0] == right[0]
            assert torch.equal(left[1], right[1]) and torch.equal(left[2], right[2])
        assert any(
            left[0] != right[0] or not torch.equal(left[1], right[1])
            for left, right in zip(expected, changed, strict=True)
        )
        evaluation = next(iter(first.val_dataloader()))
        np.testing.assert_array_equal(evaluation["features"][0], np.arange(160).reshape(40, 4))
    finally:
        first.teardown()
        resumed.teardown()


def test_dropout_preserves_nonempty_bags_and_never_modifies_sources(experimental_plan):
    experimental_plan["recipe"].update({"bagSize": None, "instanceDropout": 0.999999})
    module = MILDataModule(experimental_plan)
    for batch in module.train_dataloader():
        assert all(int(size) >= 1 for size in batch["lengths"])
    for entry in experimental_plan["featureFiles"].values():
        np.testing.assert_array_equal(np.load(entry["path"]), np.arange(160).reshape(40, 4))
