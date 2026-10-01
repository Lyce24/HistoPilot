"""Probability recalibration, fitted on development predictions and never on the cohort scored.

Binary targets use Platt scaling: p' = σ(a · logit(p) + b), fitted by maximum likelihood with
Platt's smoothed targets, so separable development predictions still give a finite map.
Multiclass targets use temperature scaling: p' = softmax(log p / T), which keeps every
decision. A Platt map with a > 0 keeps the ranking of a binary target; both maps only
rescale risks.

Pure numpy, importable by the torch-free control service.
"""

from __future__ import annotations

import math

import numpy as np

# Probabilities are kept this far from 0 and 1 before taking logarithms.
EPSILON = 1e-6
MIN_TEMPERATURE, MAX_TEMPERATURE = 0.05, 20.0
BINS = 10


def _logit(values) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), EPSILON, 1 - EPSILON)
    return np.log(clipped) - np.log1p(-clipped)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 0.5 * (1 + np.tanh(0.5 * values))


def _log_likelihood(beta, design, targets) -> float:
    linear = design @ beta
    # log σ(z) = -log(1 + e^{-z}), and log(1 - σ(z)) = -log(1 + e^{z}).
    return float(
        np.sum(-targets * np.logaddexp(0, -linear) - (1 - targets) * np.logaddexp(0, linear))
    )


def fit_platt(probabilities, labels, *, iterations: int = 100) -> dict:
    """The Platt map of positive-class ``probabilities`` given 0/1 ``labels``.

    Newton-Raphson with step halving on the two-parameter logistic model, using Platt's
    targets (N₊ + 1) / (N₊ + 2) and 1 / (N₋ + 2) so that the optimum is always finite.
    """
    scores = _logit(probabilities)
    outcomes = np.asarray(labels, dtype=np.float64)
    positives, negatives = float(outcomes.sum()), float(len(outcomes) - outcomes.sum())
    if not positives or not negatives:
        raise ValueError("Recalibration needs development units of both classes.")
    targets = np.where(outcomes == 1, (positives + 1) / (positives + 2), 1 / (negatives + 2))
    design = np.column_stack([scores, np.ones_like(scores)])
    beta = np.array([1.0, 0.0])
    current = _log_likelihood(beta, design, targets)
    for _ in range(iterations):
        fitted = _sigmoid(design @ beta)
        gradient = design.T @ (targets - fitted)
        weights = fitted * (1 - fitted)
        hessian = design.T @ (design * weights[:, None]) + 1e-9 * np.eye(2)
        step = np.linalg.solve(hessian, gradient)
        scale = 1.0
        while scale > 1e-8:
            candidate = beta + scale * step
            value = _log_likelihood(candidate, design, targets)
            if value >= current - 1e-12:
                break
            scale /= 2
        if scale <= 1e-8:
            break
        beta, previous, current = candidate, current, value
        if abs(current - previous) < 1e-10:
            break
    slope, intercept = float(beta[0]), float(beta[1])
    if not (math.isfinite(slope) and math.isfinite(intercept)):
        raise ValueError("The recalibration map did not converge.")
    return {"slope": slope, "intercept": intercept}


def apply_platt(probabilities, slope: float, intercept: float) -> np.ndarray:
    """Positive-class probabilities under a Platt map."""
    return _sigmoid(slope * _logit(probabilities) + intercept)


def _log_softmax(values: np.ndarray) -> np.ndarray:
    return values - np.logaddexp.reduce(values, axis=1, keepdims=True)


def fit_temperature(probabilities, labels, *, iterations: int = 200) -> dict:
    """The temperature T minimizing the negative log-likelihood of softmax(log p / T).

    Golden-section search on log T within [0.05, 20]; the likelihood is unimodal in T.
    """
    logs = np.log(np.clip(np.asarray(probabilities, dtype=np.float64), EPSILON, 1))
    indices = np.asarray(labels, dtype=np.int64)
    rows = np.arange(len(indices))

    def loss(log_temperature: float) -> float:
        return float(-_log_softmax(logs / math.exp(log_temperature))[rows, indices].mean())

    low, high = math.log(MIN_TEMPERATURE), math.log(MAX_TEMPERATURE)
    ratio = (math.sqrt(5) - 1) / 2
    left, right = high - ratio * (high - low), low + ratio * (high - low)
    left_loss, right_loss = loss(left), loss(right)
    for _ in range(iterations):
        if left_loss < right_loss:
            high, right, right_loss = right, left, left_loss
            left = high - ratio * (high - low)
            left_loss = loss(left)
        else:
            low, left, left_loss = left, right, right_loss
            right = low + ratio * (high - low)
            right_loss = loss(right)
        if high - low < 1e-9:
            break
    return {"temperature": math.exp((low + high) / 2)}


def apply_temperature(probabilities, temperature: float) -> np.ndarray:
    """Class probabilities under a temperature map; each row's decision is unchanged."""
    logs = np.log(np.clip(np.asarray(probabilities, dtype=np.float64), EPSILON, 1))
    return np.exp(_log_softmax(logs / temperature))


def calibration_metrics(probabilities, labels, positive: int | None = None) -> dict:
    """Calibration of class ``probabilities`` against class-index ``labels``.

    With a ``positive`` class (binary targets) the risk is that class's probability; for
    multiclass targets, the confidence of the predicted class against whether it is right.
    Brier scores sum over classes for multiclass targets. The calibration slope and
    intercept regress the outcome on logit(risk): 1 and 0 when calibrated.
    """
    values = np.asarray(probabilities, dtype=np.float64)
    indices = np.asarray(labels, dtype=np.int64)
    count = len(indices)
    rows = np.arange(count)
    onehot = np.zeros_like(values)
    onehot[rows, indices] = 1
    log_loss = float(-np.log(np.clip(values[rows, indices], EPSILON, 1)).mean())
    if positive is not None:
        risk = values[:, positive]
        outcome = (indices == positive).astype(np.float64)
        brier = float(np.mean((risk - outcome) ** 2))
    else:
        risk = values.max(axis=1)
        outcome = (values.argmax(axis=1) == indices).astype(np.float64)
        brier = float(np.mean(np.sum((values - onehot) ** 2, axis=1)))
    edges = np.linspace(0, 1, BINS + 1)
    bins, ece = [], 0.0
    for index in range(BINS):
        upper_closed = index == BINS - 1
        member = (risk >= edges[index]) & (
            (risk <= edges[index + 1]) if upper_closed else (risk < edges[index + 1])
        )
        size = int(member.sum())
        if size:
            predicted, observed = float(risk[member].mean()), float(outcome[member].mean())
            ece += size / count * abs(predicted - observed)
            bins.append(
                {
                    "lower": float(edges[index]),
                    "upper": float(edges[index + 1]),
                    "count": size,
                    "meanPredicted": predicted,
                    "observedFraction": observed,
                }
            )
    expected = float(risk.sum())
    return {
        "count": count,
        "brierScore": brier,
        "logLoss": log_loss,
        "ece": float(ece),
        "meanPredictedRisk": float(risk.mean()),
        "observedFraction": float(outcome.mean()),
        "observedExpectedRatio": float(outcome.sum() / expected) if expected > 0 else None,
        **_calibration_line(risk, outcome),
        "bins": bins,
    }


def _calibration_line(risk: np.ndarray, outcome: np.ndarray) -> dict:
    """Slope and intercept of outcome ~ logit(risk), or None where they are not estimable."""
    if outcome.min() == outcome.max():
        return {"calibrationSlope": None, "calibrationIntercept": None}
    scores = _logit(risk)
    design = np.column_stack([scores, np.ones_like(scores)])
    beta = np.array([1.0, 0.0])
    current = _log_likelihood(beta, design, outcome)
    for _ in range(100):
        fitted = _sigmoid(design @ beta)
        weights = fitted * (1 - fitted)
        hessian = design.T @ (design * weights[:, None]) + 1e-9 * np.eye(2)
        step = np.linalg.solve(hessian, design.T @ (outcome - fitted))
        scale = 1.0
        while scale > 1e-8:
            candidate = beta + scale * step
            value = _log_likelihood(candidate, design, outcome)
            if value >= current - 1e-12:
                break
            scale /= 2
        if scale <= 1e-8:
            break
        beta, previous, current = candidate, current, value
        if abs(current - previous) < 1e-10:
            break
    # Perfectly separated outcomes drive the slope to infinity; it is then not reported.
    if not np.all(np.isfinite(beta)) or abs(beta[0]) > 50:
        return {"calibrationSlope": None, "calibrationIntercept": None}
    return {"calibrationSlope": float(beta[0]), "calibrationIntercept": float(beta[1])}
