"""nnMIL views and task batches preserve membership, replay and explicit objectives."""

# ruff: noqa: E402
from collections import Counter, defaultdict
from copy import deepcopy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

from histopilot.datasets.datamodule import MILDataModule
from histopilot.datasets.mil import MILDataError


@pytest.fixture
def nnmil_data_plan(tmp_path):
    files, rows = {}, []
    patients = [
        ("p0", "negative", 3),
        ("p1", "negative", 2),
        ("p2", "positive", 1),
        ("p3", "positive", 1),
    ]
    for role in ("train", "val", "test"):
        people = (
            patients
            if role == "train"
            else [(f"{role}-p0", "negative", 1), (f"{role}-p1", "positive", 1)]
        )
        for patient, label, count in people:
            for number in range(count):
                identity = f"{patient}-s{number}"
                patches = 4 if identity == "p0-s0" else 40
                path = tmp_path / f"{identity}.npy"
                np.save(path, np.arange(patches * 4, dtype=np.float32).reshape(patches, 4))
                files[identity] = {
                    "slideId": identity,
                    "path": str(path),
                    "patchCount": patches,
                    "dimensions": 4,
                }
                rows.append(
                    {
                        "slideId": identity,
                        "patientId": patient,
                        "label": label,
                        "partition": role,
                        "pool": "development",
                        "phase": "evaluation",
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
        "recipe": {"model": "nnmil", "bagSize": 12, "batchSize": 6},
        "resources": {"dataLoaderWorkers": 0},
    }


def test_default_keeps_equal_patient_loss_contribution(nnmil_data_plan):
    original = deepcopy(nnmil_data_plan)
    module = MILDataModule(nnmil_data_plan)
    module.setup("fit")
    assert module.train_batch_sampler is None
    assert all(len(index) == 3 for index in module.train_sampler)
    contributions = defaultdict(float)
    for row in module.train_dataset.rows:
        contributions[row["patientId"]] += module.train_dataset.loss_weights[row["slideId"]]
    assert list(contributions.values()) == pytest.approx([7 / 4] * 4)
    assert module.trainingObjective == "patient_balanced_slide_cross_entropy"
    assert nnmil_data_plan == original


def test_balanced_batches_cycle_rare_classes_and_mask_short_bags(nnmil_data_plan):
    nnmil_data_plan["recipe"]["nnmilBatchSampler"] = "class_balanced"
    module = MILDataModule(nnmil_data_plan)
    batches = list(module.train_dataloader())
    assert len(batches) == len(module.train_batch_sampler) == 2
    positive_views = []
    for batch in batches:
        assert Counter(batch["labels"].tolist()) == {0: 3, 1: 3}
        assert torch.all(batch["lossWeights"] == 1)
        assert torch.equal(batch["mask"].sum(1), batch["lengths"])
        assert torch.all(batch["features"][~batch["mask"]] == 0)
        for index, identity in enumerate(batch["slideIds"]):
            assert identity.startswith("p")
            if identity == "p2-s0":
                positive_views.append(batch["features"][index][batch["mask"][index]])
    assert len(positive_views) >= 2
    assert not torch.equal(positive_views[0], positive_views[1])
    assert module.trainingObjective == "nnmil_class_balanced_slide_cross_entropy"
    # The minority pool is smaller than the per-batch quota; upstream's
    # stop-on-exhaustion behavior would produce no batches for this fold.
    assert sum(batch["features"].shape[0] for batch in batches) == 12


def test_auc_stratification_covers_each_slide_once_and_retains_remainder(nnmil_data_plan):
    nnmil_data_plan["recipe"]["nnmilBatchSampler"] = "auc_stratified"
    module = MILDataModule(nnmil_data_plan)
    batches = list(module.train_dataloader())
    assert [len(batch["slideIds"]) for batch in batches] == [6, 1]
    identities = [identity for batch in batches for identity in batch["slideIds"]]
    assert Counter(identities) == Counter(
        row["slideId"] for row in nnmil_data_plan["memberships"] if row["partition"] == "train"
    )
    assert Counter(batches[0]["labels"].tolist()) == {0: 2, 1: 4}
    assert all(torch.all(batch["lossWeights"] == 1) for batch in batches)


@pytest.mark.parametrize("mode", ["class_balanced", "auc_stratified"])
def test_task_sampler_class_weights_follow_slide_prevalence(nnmil_data_plan, mode):
    nnmil_data_plan["recipe"].update(nnmilBatchSampler=mode, classWeighting="inverse_prevalence")
    module = MILDataModule(nnmil_data_plan)
    assert module.training_class_weights() == pytest.approx([7 / 4, 7 / 10])
    assert module.training_class_weight_unit() == "slide"
    for row in nnmil_data_plan["memberships"]:
        if row["partition"] != "train":
            row["label"] = "positive"
    assert MILDataModule(nnmil_data_plan).training_class_weights() == pytest.approx([7 / 4, 7 / 10])


@pytest.mark.parametrize(
    "update",
    [
        {"batchSize": 1},
        {"samplingStrategy": "patient_natural"},
        {"classWeightedSampling": True},
        {"nnmilBatchSampler": "unknown"},
    ],
)
def test_invalid_task_sampler_combinations_fail(nnmil_data_plan, update):
    nnmil_data_plan["recipe"].update({"nnmilBatchSampler": "class_balanced", **update})
    with pytest.raises(MILDataError):
        MILDataModule(nnmil_data_plan).setup("fit")


def test_missing_fitting_class_fails_without_using_validation(nnmil_data_plan):
    nnmil_data_plan["recipe"]["nnmilBatchSampler"] = "auc_stratified"
    nnmil_data_plan["memberships"] = [
        row
        for row in nnmil_data_plan["memberships"]
        if row["partition"] != "train" or row["label"] != "positive"
    ]
    with pytest.raises(MILDataError, match="every target class"):
        MILDataModule(nnmil_data_plan).setup("fit")


@pytest.mark.parametrize("mode", ["patient_weighted", "class_balanced", "auc_stratified"])
def test_epoch_replay_is_independent_of_data_workers(nnmil_data_plan, mode):
    nnmil_data_plan["recipe"].update(
        nnmilBatchSampler=mode, instanceDropout=0.2, featureNoiseStd=0.01
    )
    first = MILDataModule(nnmil_data_plan)
    nnmil_data_plan["resources"]["dataLoaderWorkers"] = 2
    second = MILDataModule(nnmil_data_plan)
    try:
        first.set_epoch(5)
        expected = list(first.train_dataloader())
        first.set_epoch(6)
        changed = list(first.train_dataloader())
        second.set_epoch(5)
        replay = list(second.train_dataloader())
        for left, right in zip(expected, replay, strict=True):
            assert left["slideIds"] == right["slideIds"]
            torch.testing.assert_close(left["features"], right["features"], rtol=0, atol=0)
            assert torch.equal(left["mask"], right["mask"])
        assert any(
            not torch.equal(left["features"], right["features"])
            for left, right in zip(expected, changed, strict=True)
        )
        validation = next(iter(second.val_dataloader()))
        second.val_dataset.set_epoch(100)
        again = next(iter(second.val_dataloader()))
        assert torch.equal(validation["features"], again["features"])
        assert validation["features"].shape[1] == 40
    finally:
        first.teardown()
        second.teardown()
