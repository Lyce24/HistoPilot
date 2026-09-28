"""Cross-validation summaries: per fold, per training seed, seed average and seed ensemble.

Pure numpy, importable by the lightweight API (no torch). Point metrics follow
``histopilot.training.module._metrics`` exactly, so a number shown here equals the one
the training worker recorded; ``tests/test_cv_summary.py`` checks the parity.

Seed-average intervals resample the experiment's independent units (slides in a
slide-level design, patients otherwise) and score every seed on the same draw, so the
interval is for the mean across seeds. Paired model differences reuse the same draws.
Like the patient bootstrap, intervals condition on the fitted models: they do not
include retraining, configuration selection or threshold uncertainty.
"""

from __future__ import annotations

import math

import numpy as np

from histopilot.scoring import class_ranking_score, patient_predictions

# Display order. Loss is reported but never resampled or used to rank models.
METRICS = ("auroc", "auprc", "balancedAccuracy", "macroF1", "accuracy", "loss")
BOOTSTRAP_METRICS = ("auroc", "auprc", "balancedAccuracy", "macroF1", "accuracy")
LOWER_IS_BETTER = frozenset({"loss"})
METHOD = "unit_percentile_bootstrap_seed_mean_v1"
LIMITATION = (
    "95% percentile intervals resample {unit}s with replacement and score every training "
    "seed on the same draw. They condition on the fitted models and do not include "
    "retraining, configuration selection or threshold uncertainty. Draws missing a class "
    "are excluded and counted."
)
_DRAW_BLOCK = 500


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def describe(values) -> dict | None:
    """Mean, sample SD (n − 1), minimum, maximum and count of the finite values."""
    finite = [float(value) for value in values if _finite(value)]
    if not finite:
        return None
    count = len(finite)
    mean = math.fsum(finite) / count
    sd = (
        math.sqrt(math.fsum((value - mean) ** 2 for value in finite) / (count - 1))
        if count > 1
        else None
    )
    return {"mean": mean, "sd": sd, "min": min(finite), "max": max(finite), "n": count}


def _log_probabilities(row) -> list[float]:
    if "logProbabilities" in row:
        return row["logProbabilities"]
    return np.log(np.clip(row["probabilities"], 1e-300, 1)).tolist()


def _predicted(probabilities: np.ndarray, target: dict, decision_threshold) -> np.ndarray:
    """Argmax, or the frozen threshold on the positive class for a binary target."""
    predicted = probabilities.argmax(axis=1)
    if target["task"] == "binary_classification" and _finite(decision_threshold):
        positive = target["classes"].index(target["positiveClass"])
        predicted = np.where(
            probabilities[:, positive] >= decision_threshold, positive, 1 - positive
        )
    return predicted


def _auc(positive: np.ndarray, scores: np.ndarray) -> float | None:
    n_positive = int(positive.sum())
    n_negative = len(positive) - n_positive
    if not n_positive or not n_negative:
        return None
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2)[inverse]
    return float(
        (ranks[positive].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)
    )


def _average_precision(positive: np.ndarray, scores: np.ndarray) -> float | None:
    if not positive.any():
        return None
    order = np.argsort(-scores, kind="stable")
    cumulative = np.cumsum(positive[order])
    ordered = scores[order]
    boundaries = np.r_[np.flatnonzero(ordered[1:] != ordered[:-1]), len(scores) - 1]
    true_positive = cumulative[boundaries]
    precision = true_positive / (boundaries + 1)
    return float(np.sum(np.diff(np.r_[0, true_positive]) * precision) / positive.sum())


def point_metrics(rows: list[dict], target: dict, decision_threshold=None) -> dict:
    """Metrics of already-validated prediction rows, plus a one-versus-rest row per class."""
    classes = target["classes"]
    if not rows:
        return {"available": False, "count": 0, "reason": "No assessment records."}
    probabilities = np.asarray([row["probabilities"] for row in rows], dtype=np.float64)
    labels = np.asarray([row["labelIndex"] for row in rows], dtype=np.int64)
    logs = np.asarray([_log_probabilities(row) for row in rows], dtype=np.float64)
    predicted = _predicted(probabilities, target, decision_threshold)
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    np.add.at(confusion, (labels, predicted), 1)
    support = confusion.sum(axis=1)
    called = confusion.sum(axis=0)
    hits = confusion.diagonal()
    recall = np.divide(hits, support, out=np.zeros(len(classes)), where=support > 0)
    precision = np.divide(hits, called, out=np.zeros(len(classes)), where=called > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(len(classes)),
        where=(precision + recall) > 0,
    )
    scores = np.asarray(
        [[class_ranking_score(row, index) for index in range(len(classes))] for row in rows]
    )
    class_auroc = [_auc(labels == index, scores[:, index]) for index in range(len(classes))]
    class_auprc = [
        _average_precision(labels == index, scores[:, index]) for index in range(len(classes))
    ]
    if target["task"] == "binary_classification":
        positive = classes.index(target["positiveClass"])
        auroc, auprc = class_auroc[positive], class_auprc[positive]
    else:
        auroc = (
            float(np.mean(class_auroc)) if all(value is not None for value in class_auroc) else None
        )
        auprc = (
            float(np.mean(class_auprc)) if all(value is not None for value in class_auprc) else None
        )
    return {
        "available": True,
        "count": len(rows),
        "loss": float(-logs[np.arange(len(rows)), labels].mean()),
        "accuracy": float(np.mean(predicted == labels)),
        "balancedAccuracy": float(recall[support > 0].mean()),
        "macroF1": float(f1.mean()),
        "auroc": auroc,
        "auprc": auprc,
        "classCounts": {label: int(support[index]) for index, label in enumerate(classes)},
        "missingClasses": [label for index, label in enumerate(classes) if not support[index]],
        "confusionMatrix": confusion.tolist(),
        "perClass": [
            {
                "label": label,
                "support": int(support[index]),
                "predicted": int(called[index]),
                "recall": float(recall[index]) if support[index] else None,
                "precision": float(precision[index]) if called[index] else None,
                "f1": float(f1[index]),
                "auroc": class_auroc[index],
                "auprc": class_auprc[index],
            }
            for index, label in enumerate(classes)
        ],
    }


def per_class_from_confusion(confusion, classes: list[str]) -> list[dict] | None:
    """Recall, precision and F1 per class from a saved fold confusion matrix."""
    matrix = np.asarray(confusion, dtype=np.float64)
    if matrix.shape != (len(classes), len(classes)):
        return None
    support, called, hits = matrix.sum(axis=1), matrix.sum(axis=0), matrix.diagonal()
    rows = []
    for index, label in enumerate(classes):
        recall = hits[index] / support[index] if support[index] else None
        precision = hits[index] / called[index] if called[index] else None
        f1 = 2 * recall * precision / (recall + precision) if recall and precision else 0.0
        rows.append(
            {
                "label": label,
                "support": int(support[index]),
                "predicted": int(called[index]),
                "recall": recall,
                "precision": precision,
                "f1": f1,
            }
        )
    return rows


def _aligned(groups: list[list[dict]], key: str) -> list[list[dict]] | None:
    """Sort every group by ``key``; None unless all groups hold the same units and labels."""
    ordered = [sorted(rows, key=lambda row: row[key]) for rows in groups]
    reference = [(row[key], row["labelIndex"]) for row in ordered[0]]
    if len({identity for identity, _ in reference}) != len(reference):
        return None
    for rows in ordered[1:]:
        if [(row[key], row["labelIndex"]) for row in rows] != reference:
            return None
    return ordered


def seed_ensemble(
    groups: list[list[dict]], aggregation: str = "mean_probability"
) -> list[dict] | None:
    """One prediction per slide: the mean across seeds' OOF predictions of that slide.

    ``mean_probability`` averages probabilities; ``mean_logit`` averages log
    probabilities and renormalises, matching the fold-ensemble predictor rules.
    """
    if len(groups) < 2:
        return None
    ordered = _aligned(groups, "slideId")
    if ordered is None:
        return None
    rows = []
    for members in zip(*ordered, strict=True):
        logs = np.asarray([_log_probabilities(row) for row in members], dtype=np.float64)
        if aggregation == "mean_logit":
            average = logs.mean(axis=0)
            normalized = average - np.logaddexp.reduce(average)
        else:
            normalized = np.logaddexp.reduce(logs, axis=0) - math.log(len(members))
        first = members[0]
        row = {
            "slideId": first["slideId"],
            "patientId": first.get("patientId"),
            "labelIndex": first["labelIndex"],
            "label": first["label"],
            "probabilities": np.exp(normalized).tolist(),
            "logProbabilities": normalized.tolist(),
        }
        if "patientIdSource" in first:
            row["patientIdSource"] = first["patientIdSource"]
        rows.append(row)
    return rows


def patient_rows(rows: list[dict], aggregation: str = "mean_probabilities") -> list[dict]:
    """Patient predictions under the recipe's rule (``mean_probabilities`` or ``mean_logits``)."""
    return patient_predictions(rows, "mean" if aggregation == "mean_probabilities" else aggregation)


class WeightedScorer:
    """Scores one set of rows under many bootstrap weight vectors at once.

    Rows are sorted by score once per class; a draw only regroups weights, so ties keep
    their exact treatment. Formulas match ``_auc``/``_average_precision`` at unit weights.
    Weights are unit multiplicities, so float32 sums stay exact integers (< 2**24).
    """

    def __init__(self, rows: list[dict], target: dict, decision_threshold=None):
        classes = target["classes"]
        labels = np.asarray([row["labelIndex"] for row in rows], dtype=np.int64)
        probabilities = np.asarray([row["probabilities"] for row in rows], dtype=np.float64)
        predicted = _predicted(probabilities, target, decision_threshold)
        ranking = (
            [classes.index(target["positiveClass"])]
            if target["task"] == "binary_classification"
            else range(len(classes))
        )
        self.groups = []
        for index in ranking:
            scores = np.asarray([class_ranking_score(row, index) for row in rows])
            if np.isnan(scores).any():
                raise ValueError("Ranking scores cannot be NaN.")
            order = np.argsort(-scores, kind="stable")
            ordered = scores[order]
            starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
            self.groups.append((order, (labels[order] == index).astype(np.float32), starts))
        identity = np.arange(len(classes))
        self.truth = (labels[:, None] == identity).astype(np.float32)
        self.calls = (predicted[:, None] == identity).astype(np.float32)
        self.hits = self.truth * (predicted == labels)[:, None]
        self.correct = (predicted == labels).astype(np.float32)

    def __call__(self, weights: np.ndarray) -> dict[str, np.ndarray]:
        weights = weights.astype(np.float32, copy=False)
        aurocs, auprcs = [], []
        for order, positive, starts in self.groups:
            ordered = weights[:, order]
            tp = ordered * positive
            fp = ordered - tp
            tied = len(starts) < len(order)
            if tied:  # tied scores share one threshold
                tp = np.add.reduceat(tp, starts, axis=1)
                fp = np.add.reduceat(fp, starts, axis=1)
            p, n = tp.sum(axis=1, dtype=np.float64), fp.sum(axis=1, dtype=np.float64)
            cfp = np.cumsum(fp, axis=1)
            # Without ties each position is all positive or all negative, so the tie term vanishes.
            below = (tp * cfp).sum(axis=1, dtype=np.float64)
            if tied:
                below -= 0.5 * (tp * fp).sum(axis=1, dtype=np.float64)
            ctp = np.cumsum(tp, axis=1)
            called = ctp + cfp
            precision = ctp / np.maximum(called, 1)
            with np.errstate(divide="ignore", invalid="ignore"):
                auc = (n * p - below) / (p * n)
                ap = (tp * precision).sum(axis=1, dtype=np.float64) / p
            valid = (p > 0) & (n > 0)
            aurocs.append(np.where(valid, auc, np.nan))
            auprcs.append(np.where(valid, ap, np.nan))
        support = (weights @ self.truth).astype(np.float64)
        called = (weights @ self.calls).astype(np.float64)
        hits = (weights @ self.hits).astype(np.float64)
        recall = np.divide(hits, support, out=np.zeros_like(hits), where=support > 0)
        precision = np.divide(hits, called, out=np.zeros_like(hits), where=called > 0)
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(hits),
            where=(precision + recall) > 0,
        )
        present = support > 0
        return {
            "auroc": np.mean(aurocs, axis=0),
            "auprc": np.mean(auprcs, axis=0),
            "accuracy": (weights @ self.correct).astype(np.float64)
            / weights.sum(axis=1, dtype=np.float64),
            "balancedAccuracy": (recall * present).sum(axis=1) / np.maximum(present.sum(axis=1), 1),
            "macroF1": f1.mean(axis=1),
            "valid": present.all(axis=1),
        }


class UnitBootstrap:
    """Shared draws over independent units, so every model and seed sees the same resample.

    ``identities`` names each unit in a fixed order; rows map onto it with
    ``unit_index``. Draws are generated once from ``seed`` and replayed per scorer.
    """

    def __init__(self, identities: list[str], resamples: int, seed: int):
        self.identities = identities
        self.resamples = resamples
        self.seed = seed
        self._blocks: list[np.ndarray] | None = None

    def draws(self):
        """Unit multiplicities per draw, in blocks; generated once and replayed."""
        if self._blocks is None:
            count = len(self.identities)
            rng = np.random.default_rng(self.seed)
            sizes = [_DRAW_BLOCK] * (self.resamples // _DRAW_BLOCK)
            if self.resamples % _DRAW_BLOCK:
                sizes.append(self.resamples % _DRAW_BLOCK)
            self._blocks = [
                rng.multinomial(count, np.full(count, 1 / count), size=size).astype(np.float64)
                for size in sizes
            ]
        yield from self._blocks

    def mean_of(self, scorers: list[tuple[WeightedScorer, np.ndarray]]) -> dict[str, np.ndarray]:
        """Per-draw mean across ``scorers`` (each with its row → unit index)."""
        blocks: dict[str, list[np.ndarray]] = {name: [] for name in (*BOOTSTRAP_METRICS, "valid")}
        for counts in self.draws():
            values = [scorer(counts[:, index]) for scorer, index in scorers]
            for name in BOOTSTRAP_METRICS:
                blocks[name].append(np.mean([value[name] for value in values], axis=0))
            blocks["valid"].append(np.logical_and.reduce([value["valid"] for value in values]))
        return {name: np.concatenate(parts) for name, parts in blocks.items()}


def interval_summary(
    draws: dict[str, np.ndarray],
    *,
    resamples: int,
    level: float = 0.95,
    difference: dict[str, np.ndarray] | None = None,
) -> dict:
    """Percentile limits over valid draws; ``difference`` (left − right) replaces ``draws``."""
    valid = draws["valid"] if difference is None else difference["valid"]
    values = draws if difference is None else difference
    count = int(valid.sum())
    base = {"validResamples": count, "excludedResamples": resamples - count}
    if count < max(100, math.ceil(resamples * 0.8)):
        return {
            **base,
            "available": False,
            "reason": "Too few bootstrap draws contain every class.",
        }
    alpha = (1 - level) / 2
    intervals = {}
    for name in BOOTSTRAP_METRICS:
        finite = values[name][valid & np.isfinite(values[name])]
        if len(finite) < max(100, math.ceil(resamples * 0.8)):
            continue
        lower, upper = np.quantile(finite, [alpha, 1 - alpha], method="linear")
        intervals[name] = {"lower": float(lower), "upper": float(upper)}
    return {
        **base,
        "available": bool(intervals),
        "intervals": intervals,
        **({} if intervals else {"reason": "No metric could be resampled."}),
    }


def unit_indices(rows: list[dict], identities: list[str], key: str) -> np.ndarray:
    """Position of each row's unit (slide or patient) in the bootstrap's identity order."""
    position = {identity: index for index, identity in enumerate(identities)}
    return np.asarray([position[row[key]] for row in rows], dtype=np.int64)


def fold_differences(
    left: dict[str, list[float]], right: dict[str, list[float]], *, higher_is_better: bool = True
) -> dict:
    """Paired per-fold differences (left − right) on the folds both sides completed.

    ``left``/``right`` map a split-plan ID to that fold's metric values across seeds;
    each fold contributes the difference of its seed means. ``better`` counts folds
    where left beats right in the metric's own direction.
    """
    values = []
    for fold in sorted(set(left) & set(right)):
        a = [value for value in left[fold] if _finite(value)]
        b = [value for value in right[fold] if _finite(value)]
        if a and b:
            values.append(math.fsum(a) / len(a) - math.fsum(b) / len(b))
    stats = describe(values)
    if stats is None:
        return {"n": 0}
    sign = 1 if higher_is_better else -1
    return {
        **stats,
        "better": sum(value * sign > 0 for value in values),
        "worse": sum(value * sign < 0 for value in values),
        "tied": sum(value == 0 for value in values),
        "values": values,
    }
