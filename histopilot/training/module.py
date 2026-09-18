"""Validation-selected Lightning training and explicit slide/patient metrics."""

import math
import random
from collections import defaultdict
from numbers import Real

import lightning as L
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from histopilot.models import catalog, registry
from histopilot.scoring import class_ranking_score, patient_predictions


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
    ordered = scores[order]
    # Compare directly so tied +/-infinity legacy endpoint scores stay grouped.
    boundaries = np.r_[np.flatnonzero(ordered[1:] != ordered[:-1]), len(scores) - 1]
    true_positive = cumulative[boundaries]
    precision = true_positive / (boundaries + 1)
    return float(np.sum(np.diff(np.r_[0, true_positive]) * precision) / positive.sum())


def _log_probabilities(row):
    if "logProbabilities" in row:
        return row["logProbabilities"]
    return np.log(np.clip(row["probabilities"], 1e-300, 1)).tolist()


def _metrics(rows, target, *, decision_threshold=None):
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
    if target["task"] == "binary_classification" and decision_threshold is not None:
        if (
            isinstance(decision_threshold, bool)
            or not isinstance(decision_threshold, Real)
            or not math.isfinite(decision_threshold)
            or not 0 <= decision_threshold <= 1
        ):
            raise ValueError("The binary decision threshold must be finite and between zero and one.")
        positive_index = classes.index(target["positiveClass"])
        predicted = np.where(
            probabilities[:, positive_index] >= decision_threshold,
            positive_index, 1 - positive_index,
        )
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
    scores = np.asarray(
        [[class_ranking_score(row, index) for index in range(len(classes))] for row in rows]
    )
    if target["task"] == "binary_classification":
        positive = classes.index(target["positiveClass"])
        auroc = _auc(labels == positive, scores[:, positive])
        auprc = _average_precision(labels == positive, scores[:, positive])
    else:
        class_auroc = [_auc(labels == index, scores[:, index]) for index in range(len(classes))]
        auroc = (
            float(np.mean(class_auroc)) if all(value is not None for value in class_auroc) else None
        )
        class_auprc = [
            _average_precision(labels == index, scores[:, index]) for index in range(len(classes))
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


def aggregate_patients(rows, aggregation="mean_probabilities"):
    if aggregation not in {"mean_probabilities", "mean_logits"}:
        raise ValueError(f"Unsupported patient aggregation: {aggregation}")
    grouped = defaultdict(list)
    for row in rows:
        if row.get("patientIdSource") == "slide_fallback":
            raise ValueError("Patient scoring requires verified patient IDs, not slide-ID fallback.")
        grouped[row["patientId"]].append(row)
    verified = []
    for identity, slides in sorted(grouped.items()):
        if not identity or len({row["labelIndex"] for row in slides}) != 1:
            raise ValueError(
                "Patient scoring requires patient IDs and consistent labels within each patient."
            )
        if aggregation == "mean_logits":
            # Log probabilities differ from logits only by one constant per
            # slide; averaging them gives the same patient softmax. This also
            # supports prediction artifacts produced before logits were saved.
            logits = np.asarray(
                [row.get("logits", _log_probabilities(row)) for row in slides], dtype=np.float64
            )
            if (
                logits.ndim != 2
                or logits.shape[1] != len(slides[0]["probabilities"])
                or not np.isfinite(logits).all()
            ):
                raise ValueError(
                    "Patient aggregation requires finite logits in frozen class order."
                )
            normalized = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
            if not np.allclose(
                np.exp(normalized),
                [row["probabilities"] for row in slides],
                atol=1e-6,
                rtol=1e-5,
            ):
                raise ValueError("Prediction logits must agree with their saved probabilities.")
            verified.extend(
                {**row, "logProbabilities": row.get("logProbabilities", logs.tolist())}
                for row, logs in zip(slides, normalized, strict=True)
            )
        else:
            verified.extend(slides)
    return patient_predictions(
        verified, "mean" if aggregation == "mean_probabilities" else aggregation
    )


def classification_metrics(
    rows, target, aggregation="mean_probabilities", *, analysis=None, decision_threshold=None,
):
    if aggregation not in {"mean_probabilities", "mean_logits"}:
        raise ValueError(f"Unsupported patient aggregation: {aggregation}")
    slide_metrics = _metrics(rows, target, decision_threshold=decision_threshold)
    try:
        patient_metrics = _metrics(
            aggregate_patients(rows, aggregation), target, decision_threshold=decision_threshold
        )
    except ValueError as error:
        if target["unit"] == "patient":
            raise
        patient_metrics = {"available": False, "reason": str(error), "count": 0}
    result = {
        "unit": target["unit"],
        "classOrder": target["classes"],
        "positiveClass": target.get("positiveClass"),
        "patientAggregation": aggregation,
        **({"decisionThreshold": decision_threshold} if decision_threshold is not None else {}),
        "slide": slide_metrics,
        "patient": patient_metrics,
        "selected": patient_metrics if target["unit"] == "patient" else slide_metrics,
    }
    if analysis is not None:
        from histopilot.statistics import patient_analysis

        result["patientAnalysis"] = patient_analysis(
            rows, aggregate_patients(rows, aggregation) if patient_metrics["available"] else [],
            target, analysis,
        )
        intervals = result["patientAnalysis"]["uncertainty"].get("intervals")
        if intervals:
            patient_metrics["confidenceIntervals"] = intervals
    return result


def class_logits(logits, target):
    """Expand one-logit BCE outputs into the immutable target class order."""
    if logits.ndim == 1 or logits.shape[-1] == 1:
        if target["task"] != "binary_classification" or len(target["classes"]) != 2:
            raise ValueError("Single-logit BCE predictions require a binary target.")
        positive = target["classes"].index(target["positiveClass"])
        positive_logits = logits.reshape(-1)
        logits = torch.stack(
            [
                positive_logits if index == positive else torch.zeros_like(positive_logits)
                for index in range(2)
            ],
            dim=-1,
        )
    if logits.ndim != 2 or logits.shape[-1] != len(target["classes"]):
        raise ValueError("Prediction logits must match the frozen class order.")
    return logits


def window_uncertainty_rows(uncertainty, target):
    """Serialize deterministic feature-view dispersion in frozen class order."""
    if uncertainty is None:
        return None
    vectors = {
        name: value.detach().double().cpu().tolist()
        for name, value in uncertainty.items() if name not in {"windowCount", "binaryLogit"}
    }
    if (
        target["task"] == "binary_classification"
        and target["classes"].index(target["positiveClass"]) == 0
        and uncertainty.get("binaryLogit", False)
    ):
        vectors["probabilityVariance"] = [row[::-1] for row in vectors["probabilityVariance"]]
    rows = [
        {"windowCount": uncertainty["windowCount"], **{name: values[index] for name, values in vectors.items()}}
        for index in range(len(vectors["meanWindowEntropy"]))
    ]
    # Attention member JSON is canonicalized before caching. Keep insertion
    # order identical after a cache reload for downstream receipt hashes too.
    return [{name: row[name] for name in sorted(row)} for row in rows]


def prediction_rows(batch, logits, target, *, window_uncertainty=None):
    logits = class_logits(logits.detach(), target)
    probabilities = torch.softmax(logits.detach().double(), dim=-1).cpu().tolist()
    log_probabilities = torch.log_softmax(logits.detach().double(), dim=-1).cpu().tolist()
    labels = batch["labels"].detach().cpu().tolist()
    raw_logits = logits.double().cpu().tolist()
    result = [
        {
            "slideId": identity,
            "patientId": patient,
            "labelIndex": label,
            "label": target["classes"][label],
            "probabilities": probability,
            "logProbabilities": log_probability,
            "logits": raw,
        }
        for identity, patient, label, probability, log_probability, raw in zip(
            batch["slideIds"],
            batch["patientIds"],
            labels,
            probabilities,
            log_probabilities,
            raw_logits,
            strict=True,
        )
    ]
    if "patientIdSources" in batch:
        for row, source in zip(result, batch["patientIdSources"], strict=True):
            if source is not None:
                row["patientIdSource"] = source
    if window_uncertainty is not None:
        for row, uncertainty in zip(
            result, window_uncertainty_rows(window_uncertainty, target), strict=True
        ):
            row["windowUncertainty"] = uncertainty
    return result


class _FocalLoss(nn.Module):
    """OceanPath's weighted-CE focal convention, with per-bag losses."""

    def __init__(self, gamma, weight):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits, labels):
        ce = F.cross_entropy(logits, labels, weight=self.weight, reduction="none")
        return (1 - torch.exp(-ce)).pow(self.gamma) * ce


class _BinaryLoss(nn.Module):
    def __init__(self, positive_index, weight):
        super().__init__()
        self.positive_index = positive_index
        self.register_buffer(
            "pos_weight",
            None if weight is None else weight[positive_index] / weight[1 - positive_index],
        )

    def forward(self, logits, labels):
        return F.binary_cross_entropy_with_logits(
            logits.reshape(-1),
            (labels == self.positive_index).to(logits),
            pos_weight=self.pos_weight,
            reduction="none",
        )


class MILTrainModule(L.LightningModule):
    """Fit on training bags; checkpoint decisions inspect validation only."""

    def __init__(
        self,
        feature_dim: int,
        target: dict,
        recipe: dict,
        class_weights=None,
        class_weight_unit=None,
        clinical_preprocessor=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.target = target
        self.recipe = recipe
        self.input_mode = recipe.get("inputMode", "image")
        self.clinical_preprocessor = clinical_preprocessor
        if self.input_mode not in {"image", "clinical", "multimodal"}:
            raise ValueError("Unsupported clinical/image input mode.")
        if self.input_mode != "image":
            if not clinical_preprocessor or [
                {"field": column["field"], "kind": column["kind"]}
                for column in clinical_preprocessor.get("columns", [])
            ] != recipe.get("clinicalFields"):
                raise ValueError("Clinical models require their exact fitted preprocessing schema.")
        elif clinical_preprocessor is not None:
            raise ValueError("Image-only models cannot bind clinical preprocessing.")
        loss_type = recipe.get("lossType", "ce")
        if loss_type == "bce" and (
            target["task"] != "binary_classification" or len(target["classes"]) != 2
        ):
            raise ValueError("Binary cross entropy requires a binary target with two classes.")
        if recipe.get("labelSmoothing", 0) and loss_type != "ce":
            raise ValueError("Label smoothing is supported only for cross entropy.")
        if class_weights is None:
            class_weights = recipe.get("classWeights")
        if class_weights is None and recipe.get("classWeighting", "none") != "none":
            raise ValueError("Automatic class weights must be resolved from the training fold.")
        weight = None
        if class_weights is not None:
            if len(class_weights) != len(target["classes"]) or any(
                isinstance(value, bool) or not math.isfinite(value) or value <= 0
                for value in class_weights
            ):
                raise ValueError(
                    "Class weights require one finite positive value per frozen class."
                )
            weight = torch.tensor(class_weights, dtype=torch.float32)
        output_dim = 1 if loss_type == "bce" else len(target["classes"])
        model_name = catalog.normalize(recipe.get("model"))
        if self.input_mode == "clinical":
            from histopilot.models.clinical import ClinicalClassifier

            self.model = ClinicalClassifier(clinical_preprocessor["dimensions"], output_dim)
        else:
            self.model = registry.build(model_name, feature_dim, output_dim, recipe)
        self.clinical_head = None
        if self.input_mode == "multimodal":
            self.clinical_head = nn.Linear(clinical_preprocessor["dimensions"], output_dim)
            nn.init.zeros_(self.clinical_head.weight)
            nn.init.zeros_(self.clinical_head.bias)
        if loss_type == "ce":
            self.loss = nn.CrossEntropyLoss(
                weight=weight, label_smoothing=recipe.get("labelSmoothing", 0), reduction="none"
            )
        elif loss_type == "bce":
            self.loss = _BinaryLoss(target["classes"].index(target["positiveClass"]), weight)
        elif loss_type == "focal":
            self.loss = _FocalLoss(recipe.get("focalGamma", 2), weight)
        else:
            raise ValueError(f"Unsupported training loss: {loss_type}")
        self.history = []
        self.validation_rows = []
        self._training_loss_sum = 0.0
        self._training_count = 0
        self._resume_rng_state = None
        self._optimizer_presentation_count = 0
        self._optimizer_normalization_size = recipe.get("batchSize")

    def _clinical_tensor(self, clinical, count):
        from histopilot.clinical_features import transform_clinical

        if clinical is None or len(clinical) != count:
            raise ValueError("Clinical models need one frozen covariate row per prediction.")
        return torch.tensor(transform_clinical(clinical, self.clinical_preprocessor),
                            dtype=torch.float32, device=self.device)

    def forward(self, features, mask=None, clinical=None):
        if self.input_mode == "clinical":
            return self.model(self._clinical_tensor(clinical, len(features)))
        logits = self.model(features, mask)
        if self.clinical_head is not None:
            logits = logits + self.clinical_head(self._clinical_tensor(clinical, len(features)))
        return logits

    def prediction_output(self, features, mask=None, *, return_attention=False, clinical=None):
        if self.input_mode == "clinical":
            if return_attention:
                raise ValueError("Clinical-only predictors have no image attention.")
            return {"logits": self(features, mask, clinical)}
        if catalog.structured_output(self.recipe.get("model")):
            output = self.model(
                features, mask, return_attention=return_attention, return_uncertainty=True
            )
            if "window_uncertainty" in output:
                output["window_uncertainty"]["binaryLogit"] = self.model.num_classes == 1
        elif return_attention:
            output = self.model(features, mask, return_attention=True)
        else:
            output = {"logits": self.model(features, mask)}
        if self.clinical_head is not None:
            output["logits"] = output["logits"] + self.clinical_head(
                self._clinical_tensor(clinical, len(features))
            )
            # Window probability dispersion described only the image branch.
            # Do not expose it as uncertainty of the combined predictor.
            output.pop("window_uncertainty", None)
        return output

    def on_train_epoch_start(self):
        self._training_loss_sum = 0.0
        self._training_count = 0
        self._optimizer_presentation_count = 0
        self._optimizer_normalization_size = self.recipe.get("batchSize")
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
        losses = self.loss(self(batch["features"], batch["mask"], batch.get("clinical")).float(), batch["labels"])
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
        # Lightning divides each microbatch loss by its configured accumulation
        # factor. Accumulate proportionally to presentations, then normalize by
        # their actual count at the optimizer boundary. The nominal minibatch
        # divisor keeps AMP loss scaling comparable to an ordinary batch mean.
        # This applies to every aggregator, including short final groups.
        if self._optimizer_normalization_size is None:
            self._optimizer_normalization_size = count
        self._optimizer_presentation_count += count
        return loss * (count / self._optimizer_normalization_size)

    def on_before_optimizer_step(self, optimizer):
        # Lightning invokes this after mixed-precision gradients are unscaled.
        if self._optimizer_presentation_count < 1:
            raise RuntimeError("MIL optimizer updates require accumulated training presentations.")
        factor = (
            self.trainer.accumulate_grad_batches * self._optimizer_normalization_size
            / self._optimizer_presentation_count
        )
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.grad.mul_(factor)
        self._optimizer_presentation_count = 0
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
        output = self.prediction_output(batch["features"], batch["mask"], clinical=batch.get("clinical"))
        logits = output["logits"]
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("Nonfinite validation predictions.")
        self.validation_rows.extend(prediction_rows(
            batch, logits, self.target, window_uncertainty=output.get("window_uncertainty")
        ))

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        metrics = classification_metrics(
            self.validation_rows,
            self.target,
            self.recipe.get("patientAggregation", "mean_probabilities"),
            decision_threshold=self.recipe.get("decisionThreshold", 0.5),
        )["selected"]
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
        if optimizer in {"adam", "adamw"}:
            options.update(
                betas=tuple(self.recipe.get("adamBetas", (0.9, 0.999))),
                eps=self.recipe.get("adamEps", 1e-8),
            )
        head = list(self.model.classifier.parameters())
        if self.clinical_head is not None:
            head.extend(self.clinical_head.parameters())
        head_ids = {id(parameter) for parameter in head}
        aggregator = [
            parameter for parameter in self.parameters() if id(parameter) not in head_ids
        ]
        if self.recipe.get("aggregatorLearningRate") or self.recipe.get("headLearningRate"):
            parameters = [
                {
                    "params": aggregator,
                    "lr": self.recipe.get("aggregatorLearningRate") or options["lr"],
                },
                {"params": head, "lr": self.recipe.get("headLearningRate") or options["lr"]},
            ]
        else:
            parameters = self.parameters()
        decay_policy = self.recipe.get("weightDecayPolicy", "all")
        if decay_policy not in {"all", "weights_only"}:
            raise ValueError("Weight decay policy must be all or weights_only.")
        if decay_policy == "weights_only":
            # nnMIL's parameter grouping exempts biases and one-dimensional
            # parameters. Preserve any user-specified head/aggregator rates.
            groups = parameters if isinstance(parameters, list) else [{"params": parameters}]
            parameters = []
            for group in groups:
                values = list(group["params"])
                for decay in (True, False):
                    selected = [value for value in values if (value.ndim > 1) == decay]
                    if selected:
                        parameters.append({
                            **group, "params": selected,
                            "weight_decay": options["weight_decay"] if decay else 0.0,
                        })
        instance = optimizers[optimizer](parameters, **options)
        schedule = self.recipe.get("lrScheduler", "none")
        interval = self.recipe.get("lrScheduleInterval", "epoch")
        if interval not in {"epoch", "step"}:
            raise ValueError("Learning-rate schedule interval must be epoch or step.")
        if interval == "step" and schedule not in {"none", "cosine"}:
            raise ValueError("Per-step scheduling is supported for the cosine schedule.")
        if schedule == "none":
            return instance
        if schedule == "plateau":
            monitor = self.recipe["checkpointMetric"]
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                instance,
                mode="min" if monitor == "validation_loss" else "max",
                factor=self.recipe.get("lrGamma", 0.5),
                patience=self.recipe.get("lrPlateauPatience", 5),
                min_lr=1e-7,
            )
            return {
                "optimizer": instance,
                "lr_scheduler": {"scheduler": scheduler, "monitor": monitor, "interval": "epoch"},
            }
        if schedule == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                instance,
                step_size=self.recipe.get("lrStepSize", 10),
                gamma=self.recipe.get("lrGamma", 0.5),
            )
            return {
                "optimizer": instance,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }
        if schedule != "cosine":
            raise ValueError(f"Unsupported learning-rate schedule: {schedule}")
        warmup = self.recipe.get("warmupEpochs", 0)
        total = self.recipe["maxEpochs"]
        floor = self.recipe.get("finalLrFraction", 0.01)

        if interval == "step":
            # Lightning estimates optimizer updates, including gradient
            # accumulation. Scheduling each update therefore preserves the
            # warmup duration when the physical minibatch size changes.
            estimated = self.trainer.estimated_stepping_batches
            if not math.isfinite(estimated) or estimated < 1:
                raise ValueError("Per-step cosine scheduling requires a finite training budget.")
            trainer_epochs = self.trainer.max_epochs
            if trainer_epochs is None or trainer_epochs < 1:
                raise ValueError("Per-step cosine scheduling requires a finite epoch budget.")
            updates_per_epoch = math.ceil(estimated / trainer_epochs)
            total = updates_per_epoch * self.recipe["maxEpochs"]
            warmup *= updates_per_epoch

            def scale(step):
                if step < warmup:
                    return (step + 1) / warmup
                progress = min(1, (step - warmup) / max(1, total - warmup - 1))
                return floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2

            scheduler = torch.optim.lr_scheduler.LambdaLR(instance, scale)
            return {
                "optimizer": instance,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }

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
        if checkpoint.get("hyper_parameters", {}).get("clinical_preprocessor") != self.clinical_preprocessor:
            raise ValueError("Checkpoint clinical preprocessing differs from the current training patients.")
        # Loading a checkpoint restores loss buffers as well as model weights.
        # Reject an altered training objective instead of silently overwriting
        # freshly resolved class weights (including legacy slide-count weights).
        state = checkpoint.get("state_dict", {})
        for name, value in self.loss.state_dict().items():
            saved = state.get(f"loss.{name}")
            if saved is not None and not torch.equal(saved.cpu(), value.cpu()):
                raise ValueError(
                    "Checkpoint class weights differ from the current training objective. "
                    "Start a fresh run to apply the corrected class weighting."
                )
        self.history = checkpoint.get("metricsHistory", [])
        self._resume_rng_state = checkpoint.get("trainingRngState")
