"""Lightning loaders over one immutable, pre-resolved development fold."""

from __future__ import annotations

import math
from collections import Counter
from copy import deepcopy

import lightning as L
import torch
from torch.utils.data import DataLoader

from histopilot.datasets.mil import (
    MILDataError,
    SlideDataset,
    collate_mil,
    stable_seed,
    validate_memberships,
)
from histopilot.datasets.sampling import make_training_batch_sampler, make_training_sampler


def _worker_init(_worker_id):
    # Each run's main-process BLAS budget is configured by its worker entrypoint.
    # Data workers read and collate bags; they must not each create a BLAS pool.
    torch.set_num_threads(1)


def training_objective(target, recipe):
    objective = (
        "patient_balanced_slide_cross_entropy"
        if target["unit"] == "patient"
        else "slide_cross_entropy"
    )
    strategy = recipe.get("samplingStrategy", "slide_uniform")
    loss = recipe.get("lossType", "ce")
    nnmil_sampler = recipe.get("nnmilBatchSampler", "patient_weighted")
    if recipe.get("model", "abmil").lower() == "nnmil" and nnmil_sampler != "patient_weighted":
        loss_name = "cross_entropy" if loss == "ce" else loss
        objective = f"nnmil_{nnmil_sampler}_slide_{loss_name}"
    elif strategy != "slide_uniform":
        objective = f"{strategy}_one_slide_per_patient_{loss}"
    else:
        if loss != "ce":
            objective = objective.replace("cross_entropy", loss)
        if recipe.get("classWeightedSampling", False):
            objective = f"class_sampled_{objective}"
    if recipe.get("classWeighting", "none") != "none" or recipe.get("classWeights") is not None:
        objective = f"class_weighted_{objective}"
    return objective


class MILDataModule(L.LightningDataModule):
    def __init__(self, plan: dict):
        super().__init__()
        self.plan = deepcopy(plan)
        self.memberships = validate_memberships(self.plan)
        self._fit_clinical()
        self.batch_size = self.plan["recipe"]["batchSize"]
        self.eval_batch_size = self.plan["recipe"].get("evalBatchSize")
        if self.eval_batch_size is None:
            self.eval_batch_size = self.batch_size
        self.num_workers = self.plan.get("resources", {}).get("dataLoaderWorkers", 0)
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise MILDataError("batchSize must be a positive integer.")
        if type(self.eval_batch_size) is not int or self.eval_batch_size < 1:
            raise MILDataError("evalBatchSize must be a positive integer or null.")
        if type(self.num_workers) is not int or not 0 <= self.num_workers <= 64:
            raise MILDataError("dataLoaderWorkers must be an integer between zero and 64.")
        self.trainingObjective = training_objective(self.plan["target"], self.plan["recipe"])
        self.train_dataset = self.val_dataset = self.test_dataset = None
        self.train_sampler = None
        self.train_batch_sampler = None
        self._loaders = {}
        self.epoch = 0

    def _fit_clinical(self):
        from histopilot.clinical_features import clinical_fields, fit_clinical_preprocessor

        fields = clinical_fields(self.plan["recipe"])
        self.clinical_preprocessor = (
            fit_clinical_preprocessor(self.memberships["train"], self.plan.get("clinicalValues", {}), fields)
            if fields else None
        )

    def setup(self, stage=None):
        # The held-out development fold does not open feature stores during fit.
        if stage in (None, "fit") and self.train_dataset is None:
            self.train_dataset = SlideDataset(self.plan, self.memberships["train"], training=True)
            self._setup_training_sampler()
        if stage in (None, "fit", "validate") and self.val_dataset is None:
            self.val_dataset = SlideDataset(self.plan, self.memberships["val"], training=False)
        if stage in (None, "test", "predict") and self.test_dataset is None:
            self.test_dataset = SlideDataset(self.plan, self.memberships["test"], training=False)
        self.set_epoch(self.epoch)

    def _setup_training_sampler(self):
        recipe, target = self.plan["recipe"], self.plan["target"]
        self.train_batch_sampler = make_training_batch_sampler(self.train_dataset, target, recipe)
        self.train_sampler = (
            None
            if self.train_batch_sampler is not None
            else make_training_sampler(self.train_dataset, target, recipe)
        )

    def set_epoch(self, epoch):
        if type(epoch) is not int or epoch < 0:
            raise MILDataError("Epoch must be a nonnegative integer.")
        self.epoch = epoch
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
        if getattr(self, "train_batch_sampler", None) is not None:
            self.train_batch_sampler.set_epoch(epoch)

    def training_class_weights(self):
        """Resolve prevalence in the objective's unit, before sampler rebalancing."""
        recipe = self.plan["recipe"]
        explicit = recipe.get("classWeights")
        strategy = recipe.get("classWeighting", "none")
        classes = self.plan["target"]["classes"]
        if strategy not in {"none", "inverse_prevalence"}:
            raise MILDataError(f"Unsupported class weighting: {strategy}")
        if explicit is not None:
            if strategy != "none":
                raise MILDataError(
                    "Explicit class weights cannot be combined with automatic weights."
                )
            if (
                not isinstance(explicit, (list, tuple))
                or len(explicit) != len(classes)
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value <= 0
                    for value in explicit
                )
            ):
                raise MILDataError("Class weights need one finite positive value per target class.")
            return [float(value) for value in explicit]
        if strategy == "none":
            return None
        rows = self.memberships["train"]
        if self.training_class_weight_unit() == "patient":
            patients = {}
            for row in rows:
                patient, label = row["patientId"], row["labelIndex"]
                if patient in patients and patients[patient]["labelIndex"] != label:
                    raise MILDataError(
                        "Patient class weighting requires consistent training labels per patient."
                    )
                patients[patient] = row
            rows = list(patients.values())
        counts = Counter(row["labelIndex"] for row in rows)
        if len(counts) != len(classes):
            raise MILDataError(
                "Automatic class weighting requires every class in the training fold."
            )
        return [len(rows) / (len(classes) * counts[index]) for index in range(len(classes))]

    def training_class_weight_unit(self):
        recipe = self.plan["recipe"]
        if recipe.get("classWeighting", "none") != "inverse_prevalence":
            return None
        if (
            recipe.get("model", "abmil").lower() == "nnmil"
            and recipe.get("nnmilBatchSampler", "patient_weighted") != "patient_weighted"
        ):
            return "slide"
        # Uniform slides with a patient target already receive inverse slide
        # multiplicity loss weights. Patient samplers also contribute in patient
        # units, irrespective of the reporting target's unit.
        return (
            "patient"
            if self.plan["target"]["unit"] == "patient"
            or recipe.get("samplingStrategy", "slide_uniform") != "slide_uniform"
            else "slide"
        )

    def _loader(self, role, dataset, sampler=None):
        if role in self._loaders:
            return self._loaders[role]
        # DataLoader draws a base worker seed even with zero workers. Keep that
        # draw independent from the model's dropout RNG so constructing loaders
        # during checkpoint resume cannot change the subsequent optimization.
        generator = torch.Generator().manual_seed(
            stable_seed(
                self.plan["trainingSeed"], 0, f"loader-workers:{dataset.rows[0]['partition']}"
            )
        )
        batch_sampler = getattr(self, "train_batch_sampler", None) if role == "train" else None
        batching = (
            {"batch_sampler": batch_sampler}
            if batch_sampler is not None
            else {
                "batch_size": self.batch_size if role == "train" else self.eval_batch_size,
                "sampler": sampler,
                "shuffle": False,
                "drop_last": False,
            }
        )
        loader = DataLoader(
            dataset,
            **batching,
            num_workers=self.num_workers,
            collate_fn=collate_mil,
            generator=generator,
            worker_init_fn=_worker_init,
            # Cache exactly one fitting and one validation pool across epochs.
            # teardown closes both before best-checkpoint assessment; creating
            # fresh validation loaders must never duplicate persistent pools.
            persistent_workers=self.num_workers > 0 and role in {"train", "val"},
            **(
                {"multiprocessing_context": "spawn", "prefetch_factor": 1}
                if self.num_workers
                else {}
            ),
        )
        self._loaders[role] = loader
        return loader

    def train_dataloader(self):
        self.setup("fit")
        return self._loader("train", self.train_dataset, self.train_sampler)

    def val_dataloader(self):
        self.setup("validate")
        return self._loader("val", self.val_dataset)

    def test_dataloader(self):
        self.setup("test")
        return self._loader("test", self.test_dataset)

    def assessment_dataloader(self):
        return self.test_dataloader()

    def teardown(self, stage=None):
        # DataLoader exposes no public close method. Explicitly release its
        # persistent iterator before opening the next phase, also on failure.
        for loader in self._loaders.values():
            iterator = getattr(loader, "_iterator", None)
            if iterator is not None:
                iterator._shutdown_workers()
                loader._iterator = None
        self._loaders.clear()
        for dataset in (self.train_dataset, self.val_dataset, self.test_dataset):
            if dataset is not None:
                dataset.close()
