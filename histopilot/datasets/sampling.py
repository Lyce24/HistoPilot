"""Deterministic fold-local sampling with explicit slide or patient objectives."""

from collections import Counter, defaultdict
from math import ceil

import numpy as np
from torch.utils.data import Sampler

from histopilot.datasets.mil import EpochShuffleSampler, MILDataError, stable_seed


def make_training_sampler(dataset, target, recipe):
    if recipe.get("samplingStrategy", "slide_uniform") == "slide_uniform" and not recipe.get(
        "classWeightedSampling", False
    ):
        return EpochShuffleSampler(dataset)
    return ExperimentalSampler(dataset, target, recipe)


def make_training_batch_sampler(dataset, target, recipe):
    """An explicit nnMIL sampler owns minibatches, rather than drawing slides twice."""
    if recipe.get("model", "abmil").lower() != "nnmil":
        return None
    mode = recipe.get("nnmilBatchSampler", "patient_weighted")
    if mode == "patient_weighted":
        return None
    return NNMILBatchSampler(dataset, target, recipe)


class NNMILBatchSampler(Sampler):
    """Fold-local, epoch-replayable slide batches inspired by nnMIL's samplers.

    Class-balanced batches cycle exhausted class pools to produce ceil(N/B)
    full batches. This explicitly oversamples rare classes, avoiding upstream's
    empty epoch when a class cannot fill one batch. AUC-stratified batches visit
    every slide exactly once, approximately preserve the natural slide prior
    within each batch, and retain the final short batch. Both are slide-level
    objectives; patient evaluation remains a separate, frozen contract.
    """

    def __init__(self, dataset, target, recipe):
        self.dataset = dataset
        self.epoch = 0
        self.mode = recipe["nnmilBatchSampler"]
        self.batch_size = recipe["batchSize"]
        if self.mode not in {"class_balanced", "auc_stratified"}:
            raise MILDataError(f"Unsupported nnMIL batch sampler: {self.mode}")
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise MILDataError("nnMIL batch size must be a positive integer.")
        if recipe.get("samplingStrategy", "slide_uniform") != "slide_uniform" or recipe.get(
            "classWeightedSampling", False
        ):
            raise MILDataError("nnMIL batch samplers cannot be combined with another sampler.")
        self.pools = defaultdict(list)
        for index, row in enumerate(dataset.rows):
            self.pools[row["labelIndex"]].append(index)
        self.classes = list(range(len(target["classes"])))
        if set(self.pools) != set(self.classes):
            raise MILDataError("nnMIL batch sampling requires every target class in training.")
        if self.mode == "class_balanced" and self.batch_size < len(self.classes):
            raise MILDataError("Class-balanced nnMIL batches need at least one slide per class.")
        self.counts = np.asarray([len(self.pools[label]) for label in self.classes])
        self.priors = self.counts / self.counts.sum()

    def __len__(self):
        return ceil(len(self.dataset) / self.batch_size)

    @property
    def sampler(self):
        # Lightning propagates epoch state via batch_sampler.sampler before
        # opening a new iterator; expose the batch owner's replayable stream.
        return self

    def set_epoch(self, epoch):
        self.dataset.set_epoch(epoch)
        self.epoch = epoch

    def __iter__(self):
        epoch = self.epoch
        rng = np.random.default_rng(
            stable_seed(self.dataset.training_seed, epoch, f"nnmil-batches:{self.mode}")
        )
        pools = {label: rng.permutation(self.pools[label]).tolist() for label in self.classes}
        pointers = dict.fromkeys(self.classes, 0)
        assigned = np.zeros(len(self.classes), dtype=np.int64)
        total = 0
        ordinal = 0
        for batch_number in range(len(self)):
            if self.mode == "class_balanced":
                base, remainder = divmod(self.batch_size, len(self.classes))
                quotas = np.full(len(self.classes), base, dtype=np.int64)
                for offset in range(remainder):
                    quotas[(epoch + batch_number + offset) % len(self.classes)] += 1
            else:
                size = min(self.batch_size, len(self.dataset) - total)
                target = (total + size) * self.priors
                quotas = np.zeros(len(self.classes), dtype=np.int64)
                # Allocate each remaining slot to the greatest cumulative
                # deficit, while never exceeding a class's available slides.
                for _ in range(size):
                    deficit = target - assigned - quotas
                    deficit[assigned + quotas >= self.counts] = -np.inf
                    choices = np.flatnonzero(deficit == deficit.max())
                    quotas[int(rng.choice(choices))] += 1
            batch = []
            for label, quota in zip(self.classes, quotas, strict=True):
                for _ in range(int(quota)):
                    if pointers[label] == len(pools[label]):
                        pools[label] = rng.permutation(self.pools[label]).tolist()
                        pointers[label] = 0
                    batch.append(pools[label][pointers[label]])
                    pointers[label] += 1
            rng.shuffle(batch)
            presentations = [
                self.dataset.presentation_index(epoch, index, ordinal + offset)
                for offset, index in enumerate(batch)
            ]
            ordinal += len(batch)
            assigned += quotas
            total += len(batch)
            yield presentations


class ExperimentalSampler(Sampler):
    """Sample training rows only; every emitted index carries the frozen epoch."""

    def __init__(self, dataset, target, recipe):
        self.dataset = dataset
        self.epoch = 0
        self.strategy = recipe.get("samplingStrategy", "slide_uniform")
        self.class_weighted = recipe.get("classWeightedSampling", False)
        if self.strategy not in {
            "slide_uniform",
            "patient_natural",
            "cohort_balanced",
            "cohort_label_balanced",
        }:
            raise MILDataError(f"Unsupported sampling strategy: {self.strategy}")
        if self.class_weighted and self.strategy != "slide_uniform":
            raise MILDataError("Class-weighted sampling cannot be combined with patient sampling.")
        self.prevalence = recipe.get("samplingPositivePrevalence", 0.4)
        if (
            isinstance(self.prevalence, bool)
            or not isinstance(self.prevalence, (int, float))
            or not np.isfinite(self.prevalence)
            or not 0 < self.prevalence < 1
        ):
            raise MILDataError("Sampling positive prevalence must be between zero and one.")
        self.positive_index = None
        self.slides_by_patient = defaultdict(list)
        self.patient_labels, self.patient_cohorts = {}, {}
        self.cohort_pools, self.cell_pools = defaultdict(list), defaultdict(list)
        if self.strategy == "slide_uniform":
            counts = Counter(row["labelIndex"] for row in dataset.rows)
            if self.class_weighted and len(counts) != len(target["classes"]):
                raise MILDataError("Class-weighted sampling requires every class in training.")
            self.probabilities = np.asarray(
                [1 / counts[row["labelIndex"]] for row in dataset.rows], dtype=np.float64
            )
            self.probabilities /= self.probabilities.sum()
            self.sample_count = len(dataset)
            return
        if self.strategy == "cohort_label_balanced":
            if target["task"] != "binary_classification":
                raise MILDataError("Cohort-label balanced sampling requires a binary target.")
            self.positive_index = target["classes"].index(target["positiveClass"])
        for index, row in enumerate(dataset.rows):
            patient, label = row["patientId"], row["labelIndex"]
            if patient in self.patient_labels and self.patient_labels[patient] != label:
                raise MILDataError(
                    f"Patient {patient} has conflicting labels for patient sampling."
                )
            cohort = row.get("cohort")
            if self.strategy.startswith("cohort_"):
                if not isinstance(cohort, str) or not cohort.strip():
                    raise MILDataError("Cohort sampling requires a frozen cohort for every slide.")
                if patient in self.patient_cohorts and self.patient_cohorts[patient] != cohort:
                    raise MILDataError(f"Patient {patient} crosses cohorts in the training fold.")
            self.slides_by_patient[patient].append(index)
            self.patient_labels[patient] = label
            self.patient_cohorts[patient] = cohort
        self.patients = sorted(self.slides_by_patient)
        self.sample_count = len(self.patients)
        for patient in self.patients:
            cohort = self.patient_cohorts[patient]
            self.cohort_pools[cohort].append(patient)
            self.cell_pools[(cohort, self.patient_labels[patient])].append(patient)
        if self.strategy == "cohort_label_balanced":
            missing = [
                (cohort, target["classes"][label])
                for cohort in self.cohort_pools
                for label in range(len(target["classes"]))
                if not self.cell_pools[(cohort, label)]
            ]
            if missing:
                raise MILDataError(
                    f"Cohort-label sampling needs both classes in every training cohort: {missing}."
                )

    def __len__(self):
        return self.sample_count

    def set_epoch(self, epoch):
        self.dataset.set_epoch(epoch)
        self.epoch = epoch

    @staticmethod
    def _draw(pool, count, rng):
        # Exhaust each shuffled group before repeating a patient.
        result = []
        while len(result) < count:
            result.extend(rng.permutation(pool).tolist()[: count - len(result)])
        return result

    def __iter__(self):
        epoch = self.epoch
        rng = np.random.default_rng(
            stable_seed(self.dataset.training_seed, epoch, "experimental-training-sampler")
        )
        if self.strategy == "slide_uniform":
            if self.class_weighted:
                indices = rng.choice(
                    len(self.dataset), size=self.sample_count, replace=True, p=self.probabilities
                )
            else:
                indices = rng.permutation(len(self.dataset))
        else:
            if self.strategy == "patient_natural":
                patients = rng.permutation(self.patients).tolist()
            else:
                patients = []
                cohorts = sorted(self.cohort_pools)
                base, remainder = divmod(self.sample_count, len(cohorts))
                extra = {(epoch + offset) % len(cohorts) for offset in range(remainder)}
                for index, cohort in enumerate(cohorts):
                    quota = base + int(index in extra)
                    if self.strategy == "cohort_balanced":
                        patients.extend(self._draw(self.cohort_pools[cohort], quota, rng))
                    else:
                        positive = int(np.floor(quota * self.prevalence + 0.5))
                        # Both cells exist, so each receives at least one draw.
                        positive = min(max(positive, 1), quota - 1)
                        patients.extend(
                            self._draw(
                                self.cell_pools[(cohort, self.positive_index)], positive, rng
                            )
                        )
                        patients.extend(
                            self._draw(
                                self.cell_pools[(cohort, 1 - self.positive_index)],
                                quota - positive,
                                rng,
                            )
                        )
                patients = rng.permutation(patients).tolist()
            indices = [int(rng.choice(self.slides_by_patient[patient])) for patient in patients]
        for ordinal, index in enumerate(indices):
            yield self.dataset.presentation_index(epoch, index, ordinal)
