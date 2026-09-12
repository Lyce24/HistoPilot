"""Validation-selected Lightning training and explicit slide/patient metrics."""

import math
import random
from collections import defaultdict
from numbers import Real

import lightning as L
import numpy as np
import torch
from torch import nn

from histopilot.models.abmil import ABMIL


def _auc(labels, scores):
    positive = np.asarray(labels, dtype=bool)
    n_positive = int(positive.sum())
    n_negative = len(positive) - n_positive
    if not n_positive or not n_negative:
        return None
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2)[inverse]
    return float(
        (ranks[positive].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)
    )


def _average_precision(labels, scores):
    """Threshold-grouped average precision; ties have no arbitrary ordering."""
    positive = np.asarray(labels, dtype=bool)
    if not positive.any():
        return None
    scores = np.asarray(scores)
    order = np.argsort(-scores, kind="stable")
    cumulative = np.cumsum(positive[order])
    boundaries = np.r_[np.flatnonzero(np.diff(scores[order])), len(scores) - 1]
    true_positive = cumulative[boundaries]
    precision = true_positive / (boundaries + 1)
    return float(np.sum(np.diff(np.r_[0, true_positive]) * precision) / positive.sum())


def _log_probabilities(row):
    if "logProbabilities" in row:
        return row["logProbabilities"]
    return np.log(np.clip(row["probabilities"], 1e-300, 1)).tolist()


def _metrics(rows, target):
    classes = target["classes"]
    if (
        len(classes) < 2
        or any(not isinstance(label, str) or not label for label in classes)
        or len(set(classes)) != len(classes)
    ):
        raise ValueError("Frozen class order must contain at least two distinct class names.")
    if not rows:
        return {"available": False, "count": 0, "reason": "No assessment records."}

    def numeric_vector(value):
        return (
            isinstance(value, (list, tuple, np.ndarray))
            and len(value) == len(classes)
            and all(isinstance(item, Real) and not isinstance(item, bool) for item in value)
        )

    for row in rows:
        index = row.get("labelIndex")
        if type(index) is not int or not 0 <= index < len(classes):
            raise ValueError("Prediction labelIndex must be an integer within the frozen classes.")
        if row.get("label") != classes[index]:
            raise ValueError("Prediction label does not match its index in the frozen class order.")
        if not numeric_vector(row.get("probabilities")):
            raise ValueError("Predictions require one numeric probability per frozen class.")
        if "logProbabilities" in row and not numeric_vector(row["logProbabilities"]):
            raise ValueError("Log probabilities require one numeric value per frozen class.")
    probabilities = np.asarray([row["probabilities"] for row in rows], dtype=np.float64)
    labels = np.asarray([row["labelIndex"] for row in rows], dtype=np.int64)
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5, rtol=0)
    ):
        raise ValueError("Predictions must contain finite, normalized probabilities in [0, 1].")
    predicted = probabilities.argmax(axis=1)
    log_probabilities = np.asarray([_log_probabilities(row) for row in rows], dtype=np.float64)
    if (
        not np.isfinite(log_probabilities).all()
        or not np.allclose(np.logaddexp.reduce(log_probabilities, axis=1), 0, atol=1e-5, rtol=0)
        or not np.allclose(np.exp(log_probabilities), probabilities, atol=1e-6, rtol=1e-5)
    ):
        raise ValueError("Log probabilities must be finite, normalized and match probabilities.")
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    np.add.at(confusion, (labels, predicted), 1)
    support = confusion.sum(axis=1)
    recall = np.divide(confusion.diagonal(), support, out=np.zeros(len(classes)), where=support > 0)
    precision = np.divide(
        confusion.diagonal(),
        confusion.sum(axis=0),
        out=np.zeros(len(classes)),
        where=confusion.sum(axis=0) > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(len(classes)),
        where=(precision + recall) > 0,
    )
    if target["task"] == "binary_classification":
        positive = classes.index(target["positiveClass"])
        auroc = _auc(labels == positive, probabilities[:, positive])
        auprc = _average_precision(labels == positive, probabilities[:, positive])
    else:
        class_auroc = [
            _auc(labels == index, probabilities[:, index]) for index in range(len(classes))
        ]
        auroc = (
            float(np.mean(class_auroc)) if all(value is not None for value in class_auroc) else None
        )
        class_auprc = [
            _average_precision(labels == index, probabilities[:, index])
            for index in range(len(classes))
        ]
        auprc = (
            float(np.mean(class_auprc)) if all(value is not None for value in class_auprc) else None
        )
    return {
        "available": True,
        "count": len(rows),
        "loss": float(-log_probabilities[np.arange(len(rows)), labels].mean()),
        "accuracy": float(np.mean(predicted == labels)),
        "balancedAccuracy": float(recall[support > 0].mean()),
        "macroF1": float(f1.mean()),
        "auroc": auroc,
        "auprc": auprc,
        "classCounts": {label: int(support[index]) for index, label in enumerate(classes)},
        "missingClasses": [label for index, label in enumerate(classes) if not support[index]],
        "confusionMatrix": confusion.tolist(),
    }


def aggregate_patients(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["patientId"]].append(row)
    result = []
    for identity, slides in sorted(grouped.items()):
        if not identity or len({row["labelIndex"] for row in slides}) != 1:
            raise ValueError(
                "Patient scoring requires patient IDs and consistent labels within each patient."
            )
        result.append(
            {
                "patientId": identity,
                "slideIds": [row["slideId"] for row in slides],
                "labelIndex": slides[0]["labelIndex"],
                "label": slides[0]["label"],
                "probabilities": np.mean([row["probabilities"] for row in slides], axis=0).tolist(),
                "logProbabilities": (
                    np.logaddexp.reduce([_log_probabilities(row) for row in slides], axis=0)
                    - np.log(len(slides))
                ).tolist(),
            }
        )
    return result


def classification_metrics(rows, target):
    slide_metrics = _metrics(rows, target)
    try:
        patient_metrics = _metrics(aggregate_patients(rows), target)
    except ValueError as error:
        if target["unit"] == "patient":
            raise
        patient_metrics = {"available": False, "reason": str(error), "count": 0}
    return {
        "unit": target["unit"],
        "classOrder": target["classes"],
        "positiveClass": target.get("positiveClass"),
        "patientAggregation": "mean_probabilities",
        "slide": slide_metrics,
        "patient": patient_metrics,
        "selected": patient_metrics if target["unit"] == "patient" else slide_metrics,
    }


def prediction_rows(batch, logits, target):
    probabilities = torch.softmax(logits.detach().float(), dim=-1).cpu().tolist()
    log_probabilities = torch.log_softmax(logits.detach().double(), dim=-1).cpu().tolist()
    labels = batch["labels"].detach().cpu().tolist()
    return [
        {
            "slideId": identity,
            "patientId": patient,
            "labelIndex": label,
            "label": target["classes"][label],
            "probabilities": probability,
            "logProbabilities": log_probability,
        }
        for identity, patient, label, probability, log_probability in zip(
            batch["slideIds"],
            batch["patientIds"],
            labels,
            probabilities,
            log_probabilities,
            strict=True,
        )
    ]


class MILTrainModule(L.LightningModule):
    """Fit on training bags; checkpoint decisions inspect validation only."""

    def __init__(self, feature_dim: int, target: dict, recipe: dict):
        super().__init__()
        self.save_hyperparameters()
        self.target = target
        self.recipe = recipe
        if recipe.get("model", "abmil").lower() != "abmil":
            raise ValueError("This training backend implements ABMIL only.")
        self.model = ABMIL(
            feature_dim,
            len(target["classes"]),
            embed_dim=recipe.get("embedDim", 512),
            attention_dim=recipe.get("attentionDim", 384),
            num_fc_layers=recipe.get("numFcLayers", 1),
            gated_attention=recipe.get("gatedAttention", True),
            dropout=recipe.get("dropout", 0.25),
            input_dropout=recipe.get("inputDropout", 0.0),
            gradient_checkpointing=recipe.get("gradientCheckpointing", False),
        )
        self.loss = nn.CrossEntropyLoss(reduction="none")
        self.history = []
        self.validation_rows = []
        self._training_loss_sum = 0.0
        self._training_count = 0
        self._resume_rng_state = None

    def forward(self, features, mask=None):
        return self.model(features, mask)

    def on_train_epoch_start(self):
        self._training_loss_sum = 0.0
        self._training_count = 0
        if self.trainer.datamodule is not None:
            self.trainer.datamodule.set_epoch(int(self.current_epoch))
        self._epoch_learning_rate = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("learning_rate", self._epoch_learning_rate, on_step=False, on_epoch=True)

    def on_train_start(self):
        if self._resume_rng_state is not None:
            state = self._resume_rng_state
            random.setstate(state["python"])
            numpy_state = state["numpy"]
            np.random.set_state(
                (numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
            )
            torch.set_rng_state(state["torch"].cpu())
            if state["cuda"] and self.device.type == "cuda":
                torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
            self._resume_rng_state = None

    def training_step(self, batch, batch_idx):
        losses = self.loss(self(batch["features"], batch["mask"]).float(), batch["labels"])
        if self.target["unit"] == "patient":
            if "lossWeights" not in batch:
                raise ValueError("Patient-target fitting requires explicit per-slide loss weights.")
            losses = losses * batch["lossWeights"].to(losses)
        loss = losses.mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(
                "Nonfinite training loss; the run cannot produce a valid model."
            )
        count = len(losses)
        self._training_loss_sum += float(loss.detach().cpu()) * count
        self._training_count += count
        self.log("training_loss", loss, on_step=False, on_epoch=True, batch_size=count)
        return loss

    def on_before_optimizer_step(self, optimizer):
        # Lightning invokes this after mixed-precision gradients are unscaled.
        # One device reduction/synchronization covers every model parameter.
        norms = [
            torch.linalg.vector_norm(parameter.grad.detach().float())
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        if norms and not bool(torch.isfinite(torch.stack(norms)).all()):
            raise FloatingPointError(
                "Nonfinite training gradients; lower the learning rate or use 32-bit precision. "
                "The last complete epoch remains available for a reproducible restart."
            )

    def on_validation_epoch_start(self):
        self.validation_rows = []

    def validation_step(self, batch, batch_idx):
        logits = self(batch["features"], batch["mask"])
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("Nonfinite validation predictions.")
        self.validation_rows.extend(prediction_rows(batch, logits, self.target))

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        metrics = classification_metrics(self.validation_rows, self.target)["selected"]
        if not metrics["available"]:
            raise ValueError(
                "A nonempty frozen validation partition is required for checkpoint selection."
            )
        for metric in ("loss", "accuracy", "auroc", "auprc"):
            value = metrics[metric]
            if value is not None:
                self.log(f"validation_{metric}", value, on_epoch=True, batch_size=metrics["count"])
        if self.recipe["checkpointMetric"] == "validation_auroc" and metrics["auroc"] is None:
            raise ValueError(
                "Validation AUROC cannot select checkpoints when a frozen class is absent."
            )
        self.history.append(
            {
                "epoch": int(self.current_epoch),
                "step": int(self.global_step),
                "trainingLoss": self._training_loss_sum / max(1, self._training_count),
                "learningRate": self._epoch_learning_rate,
                "validation": metrics,
                "checkpointUnit": self.target["unit"],
            }
        )

    def configure_optimizers(self):
        options = {"lr": self.recipe["learningRate"], "weight_decay": self.recipe["weightDecay"]}
        optimizer = self.recipe["optimizer"]
        optimizers = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam, "adamw": torch.optim.AdamW}
        if optimizer not in optimizers:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        instance = optimizers[optimizer](self.parameters(), **options)
        schedule = self.recipe.get("lrScheduler", "none")
        if schedule == "none":
            return instance
        if schedule != "cosine":
            raise ValueError(f"Unsupported learning-rate schedule: {schedule}")
        warmup = self.recipe.get("warmupEpochs", 0)
        total = self.recipe["maxEpochs"]
        floor = self.recipe.get("finalLrFraction", 0.01)

        def scale(epoch):
            if epoch < warmup:
                return (epoch + 1) / (warmup + 1)
            progress = min(1, (epoch - warmup) / max(1, total - warmup - 1))
            return floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2

        # Epoch-indexed scheduling reaches the relative floor on the final
        # training epoch and is fully checkpointed with the optimizer.
        scheduler = torch.optim.lr_scheduler.LambdaLR(instance, scale)
        return {
            "optimizer": instance,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def on_save_checkpoint(self, checkpoint):
        checkpoint["metricsHistory"] = self.history
        numpy_state = np.random.get_state()
        checkpoint["trainingRngState"] = {
            "python": random.getstate(),
            "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            "torch": torch.get_rng_state(),
            # A CPU worker must not initialize CUDA contexts merely to save a
            # checkpoint on a workstation that also has visible GPUs.
            "cuda": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
        }

    def on_load_checkpoint(self, checkpoint):
        self.history = checkpoint.get("metricsHistory", [])
        self._resume_rng_state = checkpoint.get("trainingRngState")
