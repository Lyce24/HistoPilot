"""Lightning loaders over one immutable, pre-resolved development fold."""

from __future__ import annotations

from copy import deepcopy

import lightning as L
import torch
from torch.utils.data import DataLoader

from histopilot.datasets.mil import (
    EpochShuffleSampler,
    MILDataError,
    SlideDataset,
    collate_mil,
    stable_seed,
    validate_memberships,
)


def _worker_init(_worker_id):
    # Each run's main-process BLAS budget is configured by its worker entrypoint.
    # Data workers read and collate bags; they must not each create a BLAS pool.
    torch.set_num_threads(1)


class MILDataModule(L.LightningDataModule):
    def __init__(self, plan: dict):
        super().__init__()
        self.plan = deepcopy(plan)
        self.memberships = validate_memberships(self.plan)
        self.batch_size = self.plan["recipe"]["batchSize"]
        self.num_workers = self.plan.get("resources", {}).get("dataLoaderWorkers", 0)
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise MILDataError("batchSize must be a positive integer.")
        if type(self.num_workers) is not int or not 0 <= self.num_workers <= 64:
            raise MILDataError("dataLoaderWorkers must be an integer between zero and 64.")
        self.trainingObjective = (
            "patient_balanced_slide_cross_entropy"
            if self.plan["target"]["unit"] == "patient"
            else "slide_cross_entropy"
        )
        self.train_dataset = self.val_dataset = self.test_dataset = None
        self.train_sampler = None
        self._loaders = {}
        self.epoch = 0

    def setup(self, stage=None):
        # The held-out development fold does not open feature stores during fit.
        if stage in (None, "fit") and self.train_dataset is None:
            self.train_dataset = SlideDataset(self.plan, self.memberships["train"], training=True)
            self.train_sampler = EpochShuffleSampler(self.train_dataset)
        if stage in (None, "fit", "validate") and self.val_dataset is None:
            self.val_dataset = SlideDataset(self.plan, self.memberships["val"], training=False)
        if stage in (None, "test", "predict") and self.test_dataset is None:
            self.test_dataset = SlideDataset(self.plan, self.memberships["test"], training=False)
        self.set_epoch(self.epoch)

    def set_epoch(self, epoch):
        if type(epoch) is not int or epoch < 0:
            raise MILDataError("Epoch must be a nonnegative integer.")
        self.epoch = epoch
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

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
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_mil,
            drop_last=False,
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
