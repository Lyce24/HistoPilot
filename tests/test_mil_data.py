"""Compute-runtime feature loading preserves frozen identities and evaluation bags."""

# Optional compute dependencies must be checked before importing their consumers.
# ruff: noqa: E402

import copy
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

from histopilot.datasets.datamodule import MILDataModule
from histopilot.datasets.mil import MILDataError, SlideDataset, collate_mil, validate_memberships
from histopilot.storage.pack_import import pack_file_stamps
from histopilot.storage.packed import _stamp, build_pack


@pytest.fixture
def plan(tmp_path):
    source = tmp_path / "sources"
    source.mkdir()
    specifications = [
        ("001", "patient-a", "low", "train", 12),
        ("002", "patient-a", "low", "train", 9),
        ("003", "patient-b", "high", "train", 3),
        ("004", "patient-c", "low", "val", 11),
        ("005", "patient-d", "high", "val", 5),
        ("006", "patient-e", "low", "test", 14),
        ("007", "patient-f", "high", "test", 4),
    ]
    files, memberships = {}, []
    for identity, patient, label, role, count in specifications:
        path = source / f"{identity}.h5"
        values = np.arange(count * 4, dtype="float32").reshape(count, 4) + int(identity) * 100
        with h5py.File(path, "w") as handle:
            handle.create_dataset("features", data=values)
            handle.create_dataset(
                "coords", data=np.arange(count * 2, dtype="int64").reshape(count, 2)
            )
        files[identity] = {
            "slideId": identity,
            "path": str(path),
            "patchCount": count,
            "dimensions": 4,
            "dtype": "float32",
            **_stamp(path.stat()),
        }
        memberships.append(
            {
                "slideId": identity,
                "patientId": patient,
                "label": label,
                "partition": role,
                "planId": "seed:42/fold:0",
                "seed": 42,
                "fold": 0,
                "phase": "evaluation",
                "pool": "development",
            }
        )
    return {
        "memberships": memberships,
        "featureFiles": files,
        "featureDim": 4,
        "loadingPolicy": "native",
        "packPath": None,
        "target": {
            "task": "binary_classification",
            "unit": "patient",
            "classes": ["low", "high"],
            "positiveClass": "high",
        },
        "recipe": {"bagSize": 6, "batchSize": 2},
        "trainingSeed": 71,
        "resources": {"dataLoaderWorkers": 0, "cpuThreadsPerRun": 2},
    }


def bags(loader):
    return {
        identity: item["features"][index][item["mask"][index]].clone()
        for item in loader
        for index, identity in enumerate(item["slideIds"])
    }


def test_exact_memberships_and_patient_weighting_without_resampling(plan):
    original = copy.deepcopy(plan)
    module = MILDataModule(plan)
    module.setup()
    loaded = list(module.train_dataloader())
    assert sorted(identity for batch in loaded for identity in batch["slideIds"]) == [
        "001",
        "002",
        "003",
    ]
    weights = {
        identity: weight
        for batch in loaded
        for identity, weight in zip(batch["slideIds"], batch["lossWeights"].tolist())
    }
    assert weights == {"001": 0.75, "002": 0.75, "003": 1.5}
    assert weights["001"] + weights["002"] == weights["003"]
    assert module.trainingObjective == "patient_balanced_slide_cross_entropy"
    assert plan == original
    plan["target"]["unit"] = "slide"
    assert all(
        torch.all(batch["lossWeights"] == 1) for batch in MILDataModule(plan).train_dataloader()
    )


def test_training_sampling_changes_by_epoch_but_evaluation_uses_every_patch(plan):
    first, second = MILDataModule(plan), MILDataModule(plan)
    train_first = bags(first.train_dataloader())
    assert all(
        torch.equal(values, bags(second.train_dataloader())[identity])
        for identity, values in train_first.items()
    )
    assert train_first["001"].shape == (6, 4)
    assert train_first["003"].shape == (3, 4)
    first.set_epoch(1)
    different = bags(first.train_dataloader())
    assert not torch.equal(different["001"], train_first["001"])
    first.set_epoch(0)
    assert torch.equal(bags(first.train_dataloader())["001"], train_first["001"])
    for loader, expected in [
        (first.val_dataloader(), {"004": 11, "005": 5}),
        (first.test_dataloader(), {"006": 14, "007": 4}),
    ]:
        outputs = bags(loader)
        assert {identity: len(values) for identity, values in outputs.items()} == expected
        for identity, values in outputs.items():
            with h5py.File(plan["featureFiles"][identity]["path"], "r") as handle:
                np.testing.assert_array_equal(values.numpy(), handle["features"][:])


def test_whole_bag_training_preserves_every_patch_across_seeds_and_epochs(plan):
    plan["recipe"]["bagSize"] = None
    expected = {}
    for row in plan["memberships"]:
        if row["partition"] == "train":
            with h5py.File(plan["featureFiles"][row["slideId"]]["path"], "r") as handle:
                expected[row["slideId"]] = torch.from_numpy(handle["features"][:])
    for seed in (0, 71, 99):
        module = MILDataModule({**plan, "trainingSeed": seed})
        for epoch in (0, 1, 9):
            module.set_epoch(epoch)
            actual = bags(module.train_dataloader())
            assert actual.keys() == expected.keys()
            assert all(
                torch.equal(actual[identity], values) for identity, values in expected.items()
            )


@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, "4096", 1000001])
def test_invalid_bag_size_is_rejected_by_compute_loader(plan, value):
    plan["recipe"]["bagSize"] = value
    with pytest.raises(MILDataError, match="bagSize must be"):
        MILDataModule(plan).setup()


def test_actual_cpu_whole_bag_fit_uses_all_training_patches(plan, tmp_path, monkeypatch):
    from histopilot.schemas.development import TrainingRecipe
    from histopilot.training.fold import train_fold
    from histopilot.training.module import MILTrainModule

    recipe = TrainingRecipe(
        bagSize=None,
        batchSize=2,
        maxEpochs=2,
        earlyStopping=False,
        embedDim=8,
        attentionDim=4,
        dropout=0,
    ).model_dump()
    actual_steps = []
    training_step = MILTrainModule.training_step

    def verify_full_bags(self, batch, batch_index):
        for index, identity in enumerate(batch["slideIds"]):
            values = batch["features"][index][batch["mask"][index]]
            with h5py.File(plan["featureFiles"][identity]["path"], "r") as handle:
                np.testing.assert_array_equal(values.cpu().numpy(), handle["features"][:])
            actual_steps.append((self.current_epoch, identity, len(values)))
        loss = training_step(self, batch, batch_index)
        assert loss.requires_grad and bool(torch.isfinite(loss))
        return loss

    monkeypatch.setattr(MILTrainModule, "training_step", verify_full_bags)
    output = tmp_path / "whole-bag-run"
    result = train_fold(
        {
            **plan,
            "runId": "whole-bag-fit",
            "device": "cpu",
            "recipe": recipe,
            "data": {
                key: plan[key]
                for key in (
                    "memberships",
                    "featureFiles",
                    "featureDim",
                    "loadingPolicy",
                    "packPath",
                )
            },
        },
        output,
    )
    assert result["state"] == "succeeded"
    assert sorted(actual_steps) == [
        (epoch, identity, count)
        for epoch in (0, 1)
        for identity, count in (("001", 12), ("002", 9), ("003", 3))
    ]
    checkpoint = torch.load(output / "last.ckpt", map_location="cpu", weights_only=True)
    assert checkpoint["global_step"] == 4
    assert checkpoint["optimizer_states"]


def test_padding_mask_never_treats_padding_as_a_patch(plan):
    module = MILDataModule(plan)
    batch = next(iter(module.val_dataloader()))
    assert batch["features"].shape == (2, 11, 4)
    assert batch["mask"].dtype == torch.bool
    assert batch["mask"].sum(dim=1).tolist() == [11, 5]
    assert batch["labels"].dtype == torch.long
    assert batch["labels"].tolist() == [0, 1]
    assert torch.count_nonzero(batch["features"][~batch["mask"]]).item() == 0
    assert batch["patientIds"] == ["patient-c", "patient-d"]


@pytest.mark.parametrize(
    "change,match",
    [
        ("duplicate", "Duplicate slide"),
        ("patient_overlap", "more than one partition"),
        ("conflicting_labels", "conflicting target labels"),
        ("missing_partition", "nonempty"),
        ("wrong_label", "declared target class"),
        ("different_fold", "multiple frozen split"),
        ("external_test", "external test"),
    ],
)
def test_invalid_frozen_membership_contract_is_blocked(plan, change, match):
    rows = plan["memberships"]
    if change == "duplicate":
        rows.append(dict(rows[0]))
    elif change == "patient_overlap":
        rows[3]["patientId"] = rows[0]["patientId"]
    elif change == "conflicting_labels":
        rows[1]["label"] = "high"
    elif change == "missing_partition":
        plan["memberships"] = [row for row in rows if row["partition"] != "val"]
    elif change == "wrong_label":
        rows[0]["label"] = "Positive"
    elif change == "different_fold":
        rows[-1]["fold"] = 1
    else:
        rows[-1]["pool"] = "external_test"
    with pytest.raises(MILDataError, match=match):
        MILDataModule(plan)


def test_exact_feature_ids_are_not_normalized_or_silently_skipped(plan):
    plan["featureFiles"]["1"] = plan["featureFiles"].pop("001")
    with pytest.raises(MILDataError, match="exact feature inventory.*001"):
        MILDataModule(plan).setup()


def test_source_file_change_after_preflight_blocks_loading(plan):
    module = MILDataModule(plan)
    module.setup()
    with h5py.File(plan["featureFiles"]["001"]["path"], "r+") as handle:
        handle["features"][0, 0] = 999
    with pytest.raises(MILDataError, match="changed since"):
        module.train_dataset[0]


@pytest.mark.parametrize("corruption", ["nonfinite", "dimensions", "empty", "external_hdf5"])
def test_invalid_feature_content_is_not_used(plan, tmp_path, corruption):
    entry = plan["featureFiles"]["001"]
    path = Path(entry["path"])
    with h5py.File(path, "w") as handle:
        if corruption == "external_hdf5":
            handle["features"] = h5py.ExternalLink(plan["featureFiles"]["002"]["path"], "features")
        else:
            shape = (
                (12, 5)
                if corruption == "dimensions"
                else (0, 4)
                if corruption == "empty"
                else (12, 4)
            )
            values = np.ones(shape, dtype="float32")
            if corruption == "nonfinite":
                values[:] = float("nan")
            handle.create_dataset("features", data=values)
    entry.update(_stamp(path.stat()))
    module = MILDataModule(plan)
    module.setup()
    with pytest.raises(MILDataError):
        module.train_dataset[0]


@pytest.mark.parametrize("suffix", [".npy", ".pt"])
def test_safe_native_tensor_formats_match_hdf5(plan, tmp_path, suffix):
    original = bags(MILDataModule(plan).train_dataloader())
    for identity, entry in plan["featureFiles"].items():
        with h5py.File(entry["path"], "r") as handle:
            values = handle["features"][:]
        path = tmp_path / f"{identity}{suffix}"
        if suffix == ".npy":
            np.save(path, values)
        else:
            torch.save({"features": torch.from_numpy(values)}, path)
        entry.update(path=str(path), **_stamp(path.stat()))
    loaded = bags(MILDataModule(plan).train_dataloader())
    assert all(torch.equal(loaded[identity], values) for identity, values in original.items())


@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("bag_size", [6, None])
def test_verified_native_and_oceanpath_memmap_inputs_are_equivalent(
    plan, tmp_path, external, bag_size
):
    plan["recipe"]["bagSize"] = bag_size
    destination = tmp_path / "packed"
    configuration = {
        "id": "configuration-fixture",
        "contentHash": "a" * 64,
        "manifest": {"kind": "feature", "files": list(plan["featureFiles"].values())},
    }
    build_pack(configuration, destination, dtype="preserve")
    if external:
        # A registered external OceanPath pack has the same public index/bin
        # contract without HistoPilot's optional archive sidecars.
        (destination / "manifest.json").unlink()
        (destination / "checksums.json").unlink()
    packed_plan = {
        **plan,
        "loadingPolicy": "mmap",
        "packPath": str(destination),
        "packStamps": pack_file_stamps(destination),
    }
    native, packed = MILDataModule(plan), MILDataModule(packed_plan)
    for name in ("train_dataloader", "val_dataloader", "test_dataloader"):
        expected, actual = bags(getattr(native, name)()), bags(getattr(packed, name)())
        assert expected.keys() == actual.keys()
        assert all(torch.equal(actual[identity], values) for identity, values in expected.items())
    with (destination / "features.bin").open("r+b") as handle:
        handle.write(b"\x00\x00\x00\x00")
    with pytest.raises(MILDataError, match="pack changed"):
        packed.train_dataset[0]


def test_workers_and_main_process_produce_same_epoch_specific_bags(plan):
    zero = MILDataModule(plan)
    worker_plan = copy.deepcopy(plan)
    worker_plan["resources"]["dataLoaderWorkers"] = 2
    workers = MILDataModule(worker_plan)
    loader = workers.train_dataloader()
    try:
        for epoch in (0, 2, 0):
            zero.set_epoch(epoch)
            workers.set_epoch(epoch)
            expected, actual = bags(zero.train_dataloader()), bags(loader)
            assert all(
                torch.equal(actual[identity], values) for identity, values in expected.items()
            )
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()


def test_dataset_package_is_safe_to_import_without_compute_dependencies():
    code = "import sys; import histopilot.datasets; assert 'torch' not in sys.modules; assert 'lightning' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_standalone_epoch_token_controls_patch_selection(plan):
    rows = validate_memberships(plan)["train"]
    dataset = SlideDataset(plan, rows, training=True)
    first = dataset[(0, 0)]["features"]
    other = dataset[(3, 0)]["features"]
    assert not torch.equal(first, other)
    assert torch.equal(dataset[(0, 0)]["features"], first)
    with pytest.raises(MILDataError):
        collate_mil([])


def test_loader_iteration_does_not_consume_model_random_state(plan):
    module = MILDataModule(plan)
    before = torch.get_rng_state().clone()
    bags(module.train_dataloader())
    bags(module.val_dataloader())
    bags(module.test_dataloader())
    assert torch.equal(torch.get_rng_state(), before)


def test_fit_defers_assessment_dataset_and_reuses_each_loader(plan):
    module = MILDataModule(plan)
    module.setup("fit")
    assert module.test_dataset is None
    assert module.train_dataloader() is module.train_dataloader()
    assert module.val_dataloader() is module.val_dataloader()
    assert module.test_dataset is None
    assert module.test_dataloader() is module.test_dataloader()
    module.teardown()
    assert not module._loaders


def test_singleton_whole_bag_collation_reuses_feature_storage(plan):
    plan["recipe"]["bagSize"] = None
    module = MILDataModule(plan)
    module.setup("fit")
    item = module.train_dataset[0]
    batch = collate_mil([item])
    assert batch["features"].data_ptr() == item["features"].data_ptr()
    assert batch["features"].shape == (1, 12, 4)
    assert bool(batch["mask"].all())
    torch.testing.assert_close(batch["features"][0], item["features"])


def test_worker_pools_have_bounded_prefetch_and_are_released(plan):
    plan["resources"]["dataLoaderWorkers"] = 2
    module = MILDataModule(plan)
    loader = module.train_dataloader()
    assert loader.persistent_workers and loader.prefetch_factor == 1
    iterator = iter(loader)
    next(iterator)
    workers = list(iterator._workers)
    validation = module.val_dataloader()
    assert validation.persistent_workers
    assert validation is module.val_dataloader()
    assert not module.test_dataloader().persistent_workers
    validation_iterator = iter(validation)
    validation_workers = list(validation_iterator._workers)
    list(validation_iterator)
    module.teardown()
    assert all(not worker.is_alive() for worker in validation_workers)
    assert all(not worker.is_alive() for worker in workers)
    assert loader._iterator is None
