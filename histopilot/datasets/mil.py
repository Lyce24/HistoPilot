"""Exact frozen slide memberships and lazily opened MIL feature bags.

This module belongs to the optional training runtime. No OceanPath imports or
split generation are used; its pack format is read through HistoPilot's storage
contract. Patch selection depends on seed, epoch, and exact ID; nnMIL also
carries a presentation ordinal so repeated slides receive fresh training views.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from histopilot.models import catalog
from histopilot.storage.pack_import import _layout, pack_file_stamps
from histopilot.storage.packed import PackedFeatureStore, PackedStoreError, _dataset, _source


class MILDataError(ValueError):
    """An immutable training plan cannot be read without changing its meaning."""


def stable_seed(seed: int, epoch: int, identity: str) -> int:
    encoded = json.dumps([seed, epoch, identity], ensure_ascii=False, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(encoded.encode()).digest()[:8], "big") % (2**63 - 1)


def validate_memberships(plan: dict) -> dict[str, list[dict]]:
    """Require one already resolved fold and preserve every selected membership."""
    rows = plan.get("memberships")
    if not isinstance(rows, list) or not rows:
        raise MILDataError("A run requires explicit frozen slide memberships.")
    target = plan.get("target", {})
    classes = target.get("classes", [])
    if (
        target.get("task") not in {"binary_classification", "multiclass_classification"}
        or target.get("unit") not in {"patient", "slide"}
        or len(classes) < 2
        or any(not isinstance(label, str) or not label.strip() for label in classes)
        or len(classes) != len(set(classes))
    ):
        raise MILDataError("The target must declare an ordered classification contract.")
    if target["task"] == "binary_classification" and (
        len(classes) != 2 or target.get("positiveClass") not in classes
    ):
        raise MILDataError("Binary targets require two classes and the inherited positive class.")
    partitions = {role: [] for role in ("train", "val", "test")}
    seen_slides, patients, patient_labels, plans = set(), {}, {}, set()
    for raw in rows:
        row = deepcopy(raw)
        slide, patient, role, label = (
            row.get(key) for key in ("slideId", "patientId", "partition", "label")
        )
        if any(
            not isinstance(value, str) or not value or value.strip() != value
            for value in (slide, patient)
        ):
            raise MILDataError("Every selected row requires exact nonempty slide and patient IDs.")
        if slide in seen_slides:
            raise MILDataError(
                f"Duplicate slide membership: {slide}. Supply exactly one frozen fold."
            )
        if (
            target["unit"] == "patient"
            and row.get("patientIdSource") == "slide_fallback"
        ):
            raise MILDataError(
                "Patient analysis requires verified patient IDs, not slide fallback."
            )
        seen_slides.add(slide)
        if (
            role not in partitions
            or row.get("phase") == "final"
            or row.get("pool") == "external_test"
        ):
            raise MILDataError(
                "Runs may only fit and assess one development fold; no external test or tuning rows."
            )
        if label not in classes:
            raise MILDataError(f"{slide}: label is not a declared target class.")
        if patient in patients and patients[patient] != role:
            raise MILDataError(f"Patient {patient} appears in more than one partition.")
        if (
            target["unit"] == "patient"
            and patient in patient_labels
            and patient_labels[patient] != label
        ):
            raise MILDataError(f"Patient {patient} has conflicting target labels.")
        patients[patient], patient_labels[patient] = role, label
        metadata = {
            key: row[key]
            for key in (
                "planId",
                "seed",
                "fold",
                "phase",
                "outerFold",
                "innerFold",
                "repeat",
                "domain",
            )
            if key in row
        }
        plans.add(json.dumps(metadata, sort_keys=True, separators=(",", ":")))
        row["labelIndex"] = classes.index(label)
        partitions[role].append(row)
    if len(plans) != 1:
        raise MILDataError("The run contains memberships from multiple frozen split plans.")
    if any(not values for values in partitions.values()):
        raise MILDataError(
            "A development run requires nonempty fitting, validation, and assessment partitions."
        )
    return {
        role: sorted(values, key=lambda row: row["slideId"]) for role, values in partitions.items()
    }


class _OceanPathMemmap:
    """Read verified external OceanPath v1 packs without requiring our sidecar manifest."""

    def __init__(self, path, layout):
        meta = layout["meta"]
        self.slides = {row["slideId"]: row for row in layout["slides"]}
        self.array = np.memmap(
            Path(path) / "features.bin",
            mode="r",
            dtype=np.dtype(meta["feat_dtype"]).newbyteorder("<"),
            shape=(meta["total_patches"], meta["feat_dim"]),
        )

    def read_features(self, slide_id, rows=None):
        entry = self.slides[slide_id]
        values = self.array[entry["offset"] : entry["offset"] + entry["patchCount"]]
        return np.array(values if rows is None else values[rows], copy=True)

    def close(self):
        self.array._mmap.close()


class SlideDataset(Dataset):
    """One selected slide per item with reproducible, explicitly configured views."""

    def __init__(self, plan: dict, memberships: list[dict], *, training: bool):
        self.rows = deepcopy(memberships)
        self.training = training
        self.training_seed = plan["trainingSeed"]
        self.epoch = 0
        self.recipe = deepcopy(plan["recipe"])
        self.input_mode = self.recipe.get("inputMode", "image")
        self.clinical_values = deepcopy(plan.get("clinicalValues", {}))
        if self.input_mode != "image":
            from histopilot.clinical_features import clinical_rows

            clinical_rows(self.rows, self.clinical_values, self.recipe.get("clinicalFields", []))
        self.bag_size = self.recipe["bagSize"]
        self.eval_bag_size = self.recipe.get("evalBagSize")
        self.dimensions = plan["featureDim"]
        for name, value, minimum in (
            ("trainingSeed", self.training_seed, 0),
            ("featureDim", self.dimensions, 1),
        ):
            if type(value) is not int or value < minimum:
                raise MILDataError(f"{name} must be an integer of at least {minimum}.")
        if self.bag_size is not None and (
            type(self.bag_size) is not int or not 1 <= self.bag_size <= 1000000
        ):
            raise MILDataError(
                "bagSize must be an integer from 1 to 1000000 or null for whole-bag training."
            )
        if self.eval_bag_size is not None and (
            type(self.eval_bag_size) is not int or not 1 <= self.eval_bag_size <= 1000000
        ):
            raise MILDataError("evalBagSize must be a positive integer or null for full bags.")
        self.instance_dropout = self.recipe.get("instanceDropout", 0.0)
        self.feature_noise_std = self.recipe.get("featureNoiseStd", 0.0)
        if (
            isinstance(self.instance_dropout, bool)
            or not isinstance(self.instance_dropout, (int, float))
            or not np.isfinite(self.instance_dropout)
            or not 0 <= self.instance_dropout < 1
        ):
            raise MILDataError("instanceDropout must be finite and between zero and one.")
        if (
            isinstance(self.feature_noise_std, bool)
            or not isinstance(self.feature_noise_std, (int, float))
            or not np.isfinite(self.feature_noise_std)
            or self.feature_noise_std < 0
        ):
            raise MILDataError("featureNoiseStd must be finite and nonnegative.")
        if self.recipe.get("bagCurriculum", False):
            for name in ("bagCurriculumStart", "bagCurriculumEnd", "bagCurriculumWarmupEpochs"):
                value = self.recipe.get(
                    name,
                    {
                        "bagCurriculumStart": 512,
                        "bagCurriculumEnd": 8000,
                        "bagCurriculumWarmupEpochs": 5,
                    }[name],
                )
                if type(value) is not int or value < 1:
                    raise MILDataError(f"{name} must be a positive integer.")
            if self.recipe.get("bagCurriculumStart", 512) > self.recipe.get(
                "bagCurriculumEnd", 8000
            ):
                raise MILDataError("Bag curriculum start cannot exceed its end.")
        self.files = deepcopy(plan.get("featureFiles", {}))
        self.policy = plan.get("loadingPolicy", "native")
        if self.policy not in {"native", "mmap"}:
            raise MILDataError("Resolve feature loading to native or mmap before starting a run.")
        self.pack_path = plan.get("packPath")
        if self.input_mode == "clinical":
            self.policy, self.pack_path = "native", None
        if (self.policy == "mmap") != bool(self.pack_path):
            raise MILDataError(
                "Packed loading requires a pack path; native loading must not bind a pack."
            )
        self._pack = None
        self._pack_pid = None
        self._pack_layout = None
        self.pack_stamps = None
        if self.policy == "mmap":
            self._pack_layout = _layout(Path(self.pack_path))
            self.pack_stamps = self._pack_layout["packStamps"]
            if plan.get("packStamps") is not None and plan["packStamps"] != self.pack_stamps:
                raise MILDataError("The selected feature pack changed after run preflight.")
            if self._pack_layout["meta"]["feat_dim"] != self.dimensions:
                raise MILDataError("Pack dimensions differ from the selected model features.")
            self.packed_rows = {row["slideId"]: row for row in self._pack_layout["slides"]}
        for row in self.rows:
            if self.input_mode == "clinical":
                continue
            identity = row["slideId"]
            if identity not in self.files:
                raise MILDataError(
                    f"No exact feature inventory record exists for slide {identity}."
                )
            entry = self.files[identity]
            if entry.get("slideId", identity) != identity:
                raise MILDataError(f"Feature inventory key does not match slide ID {identity}.")
            if (
                type(entry.get("patchCount")) is not int
                or entry["patchCount"] <= 0
                or entry.get("dimensions") != self.dimensions
            ):
                raise MILDataError(
                    f"{identity}: feature inventory has empty bags or incompatible dimensions."
                )
            if self.policy == "native" and (
                not isinstance(entry.get("path"), str) or not Path(entry["path"]).is_absolute()
            ):
                raise MILDataError(
                    f"{identity}: feature inventory requires an absolute source path."
                )
            if self.policy == "mmap" and (
                identity not in self.packed_rows
                or self.packed_rows[identity]["patchCount"] != entry["patchCount"]
            ):
                raise MILDataError(f"The pack lacks exact feature rows for slide {identity}.")
        patients = Counter(row["patientId"] for row in self.rows)
        self.loss_weights = {
            row["slideId"]: len(self.rows) / (len(patients) * patients[row["patientId"]])
            if training
            and plan["target"]["unit"] == "patient"
            and self.recipe.get("samplingStrategy", "slide_uniform") == "slide_uniform"
            and not (
                catalog.owns_options(self.recipe.get("model"), "nnmil")
                and self.recipe.get("nnmilBatchSampler", "patient_weighted") != "patient_weighted"
            )
            else 1.0
            for row in self.rows
        }

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch):
        if type(epoch) is not int or epoch < 0:
            raise MILDataError("Epoch must be a nonnegative integer.")
        self.epoch = epoch

    def presentation_index(self, epoch, index, ordinal):
        """Carry draw identity through worker prefetch without mutable RNG state."""
        if catalog.windowed_sampling(self.recipe.get("model")) and self.training:
            return epoch, int(index), ordinal
        return epoch, int(index)

    def _selection(self, count, identity, epoch, ordinal=0):
        cap = self.bag_size if self.training else self.eval_bag_size
        if self.training and self.recipe.get("bagCurriculum", False):
            start = self.recipe.get("bagCurriculumStart", 512)
            end = self.recipe.get("bagCurriculumEnd", 8000)
            progress = min(epoch / self.recipe.get("bagCurriculumWarmupEpochs", 5), 1.0)
            cap = int(start + (end - start) * progress)
        if cap is None or count <= cap:
            return None
        if self.training and catalog.windowed_sampling(self.recipe.get("model")):
            identity = json.dumps(["nnmil-patch-view", identity, ordinal], ensure_ascii=False)
        rng = np.random.default_rng(
            stable_seed(self.training_seed, epoch if self.training else 0, identity)
        )
        # Sorted indices satisfy HDF5's indexing contract and preserve source row order.
        return np.sort(rng.choice(count, size=cap, replace=False))

    def _augment(self, values, identity, epoch, ordinal=0):
        if not self.training or not (self.instance_dropout or self.feature_noise_std):
            return values
        if catalog.windowed_sampling(self.recipe.get("model")):
            identity = json.dumps(["nnmil-augmentation", identity, ordinal], ensure_ascii=False)
        # Local streams preserve model/dropout RNG and exact worker/resume replay.
        rng = np.random.default_rng(stable_seed(self.training_seed, epoch, f"augment:{identity}"))
        if self.instance_dropout and len(values) > 1:
            keep = rng.random(len(values)) >= self.instance_dropout
            if not keep.any():
                keep[rng.integers(len(values))] = True
            values = values[keep]
        if self.feature_noise_std:
            noise = rng.standard_normal(values.shape, dtype=np.float32)
            values = values + noise * self.feature_noise_std
        return values

    def _native(self, entry, selection):
        path = Path(entry["path"])
        with _source(path, entry) as (stream, _stamp):
            if path.suffix.lower() in {".h5", ".hdf5"}:
                with h5py.File(stream, "r") as handle:
                    source = _dataset(handle, "features")
                    if len(source.shape) == 1:
                        # One slide embedding is the single instance of its bag.
                        # Read it whole and give it a row before any selection.
                        values = np.asarray(source[:]).reshape(1, -1)
                        self._header(values.shape, values.dtype, entry)
                        return values if selection is None else values[selection]
                    self._header(source.shape, source.dtype, entry)
                    return source[:] if selection is None else source[selection]
            if path.suffix.lower() == ".npy":
                source = np.load(stream, allow_pickle=False)
            elif path.suffix.lower() == ".pt":
                source = torch.load(stream, map_location="cpu", weights_only=True)
                if isinstance(source, dict):
                    source = source.get("features")
                if (
                    not isinstance(source, torch.Tensor)
                    or source.layout != torch.strided
                    or source.requires_grad
                ):
                    raise MILDataError("A PT feature artifact must contain a plain feature tensor.")
                source = source.numpy()
            else:
                raise MILDataError(f"Unsupported native feature format: {path.suffix}.")
            if source.ndim == 1:
                source = source.reshape(1, -1)
            self._header(source.shape, source.dtype, entry)
            # Safe readers own their arrays; keep that storage until the float32
            # conversion instead of copying an entire whole bag a second time.
            return source if selection is None else source[selection]

    def _header(self, shape, dtype, entry):
        if (
            len(shape) != 2
            or shape != (entry["patchCount"], self.dimensions)
            or np.dtype(dtype).kind != "f"
        ):
            raise MILDataError(
                f"{entry.get('slideId', 'Slide')}: feature shape or floating-point dtype changed."
            )
        if entry.get("dtype") is not None and np.dtype(entry["dtype"]) != np.dtype(dtype):
            raise MILDataError("Native feature precision differs from the frozen inventory.")

    def _packed(self, identity, selection):
        if pack_file_stamps(Path(self.pack_path)) != self.pack_stamps:
            raise MILDataError("The selected feature pack changed while the run was active.")
        if self._pack is None or self._pack_pid != os.getpid():
            self.close()
            if self._pack_layout["manifest"] is not None:
                self._pack = PackedFeatureStore(Path(self.pack_path), verify=False)
            else:
                self._pack = _OceanPathMemmap(self.pack_path, self._pack_layout)
            self._pack_pid = os.getpid()
        values = self._pack.read_features(identity, selection)
        if pack_file_stamps(Path(self.pack_path)) != self.pack_stamps:
            raise MILDataError("The selected feature pack changed during a read.")
        return values

    def __getitem__(self, index):
        # Epoch travels with each sampled index across worker/prefetch boundaries.
        ordinal = 0
        if isinstance(index, tuple) and len(index) == 3:
            epoch, position, ordinal = index
            if type(ordinal) is not int or ordinal < 0:
                raise MILDataError("Presentation ordinal must be a nonnegative integer.")
        else:
            epoch, position = index if isinstance(index, tuple) else (self.epoch, index)
        row = self.rows[position]
        if self.input_mode == "clinical":
            return self._example(row, torch.zeros((1, self.dimensions), dtype=torch.float32))
        entry = self.files[row["slideId"]]
        selection = self._selection(entry["patchCount"], row["slideId"], epoch, ordinal)
        try:
            values = (
                self._native(entry, selection)
                if self.policy == "native"
                else self._packed(row["slideId"], selection)
            )
        except (PackedStoreError, OSError, KeyError) as error:
            raise MILDataError(
                f"{row['slideId']}: cannot read the frozen features: {error}"
            ) from error
        if (
            values.ndim != 2
            or not values.shape[0]
            or values.shape[1] != self.dimensions
            or not np.isfinite(values).all()
        ):
            raise MILDataError(
                f"{row['slideId']}: features must be nonempty finite [patches, dimensions]."
            )
        values = np.asarray(values, dtype=np.float32, order="C")
        values = self._augment(values, row["slideId"], epoch, ordinal)
        if not values.flags.writeable:
            values = values.copy()
        if not np.isfinite(values).all():
            raise MILDataError(
                f"{row['slideId']}: feature values exceed float32 training precision."
            )
        return self._example(row, torch.from_numpy(values))

    def _example(self, row, features):
        return {
            "features": features,
            "label": row["labelIndex"],
            "slideId": row["slideId"],
            "patientId": row["patientId"],
            **({"patientIdSource": row["patientIdSource"]} if "patientIdSource" in row else {}),
            "lossWeight": self.loss_weights[row["slideId"]],
            **({"clinical": self.clinical_values[row["slideId"]]} if self.input_mode != "image" else {}),
        }

    def close(self):
        if self._pack is not None:
            self._pack.close()
            self._pack = None

    def __getstate__(self):
        return {**self.__dict__, "_pack": None, "_pack_pid": None}


class EpochShuffleSampler(Sampler):
    """Visit every training slide once, with reproducible order and patch epoch."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.dataset.set_epoch(epoch)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        epoch = self.epoch
        generator = torch.Generator().manual_seed(
            stable_seed(self.dataset.training_seed, epoch, "training-slide-order")
        )
        for ordinal, index in enumerate(
            torch.randperm(len(self.dataset), generator=generator).tolist()
        ):
            yield self.dataset.presentation_index(epoch, index, ordinal)


def collate_mil(batch):
    if not batch:
        raise MILDataError("Cannot collate an empty MIL batch.")
    lengths = [item["features"].shape[0] for item in batch]
    dimensions = batch[0]["features"].shape[1]
    if any(
        not count or item["features"].ndim != 2 or item["features"].shape[1] != dimensions
        for count, item in zip(lengths, batch)
    ):
        raise MILDataError("MIL batches require nonempty bags with consistent feature dimensions.")
    if len(batch) == 1:
        # Most whole-bag MIL runs use one slide per batch. Add a view dimension;
        # no padding or second full feature allocation is necessary.
        features = batch[0]["features"].to(dtype=torch.float32).unsqueeze(0)
        mask = torch.ones((1, lengths[0]), dtype=torch.bool)
    else:
        features = torch.zeros((len(batch), max(lengths), dimensions), dtype=torch.float32)
        mask = torch.zeros((len(batch), max(lengths)), dtype=torch.bool)
        for index, item in enumerate(batch):
            count = lengths[index]
            features[index, :count] = item["features"]
            mask[index, :count] = True
    return {
        "features": features,
        "mask": mask,
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "lossWeights": torch.tensor([item["lossWeight"] for item in batch], dtype=torch.float32),
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "slideIds": [item["slideId"] for item in batch],
        "patientIds": [item["patientId"] for item in batch],
        **({"clinical": [item["clinical"] for item in batch]} if "clinical" in batch[0] else {}),
        **({"patientIdSources": [item.get("patientIdSource") for item in batch]}
           if any("patientIdSource" in item for item in batch) else {}),
    }
