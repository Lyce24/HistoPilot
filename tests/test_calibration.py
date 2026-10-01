"""Recalibration maps recover known miscalibration and never change a decision."""

import numpy as np
import pytest

from histopilot.calibration import (
    apply_platt,
    apply_temperature,
    calibration_metrics,
    fit_platt,
    fit_temperature,
)


def overconfident(seed=0, count=4000):
    """True risks σ(z), reported as σ(2z): twice as confident as the outcomes support."""
    random = np.random.default_rng(seed)
    scores = random.normal(0, 1.5, count)
    outcomes = (random.random(count) < 1 / (1 + np.exp(-scores))).astype(int)
    return 1 / (1 + np.exp(-2 * scores)), outcomes


def test_platt_scaling_recovers_the_slope_and_restores_calibration():
    reported, outcomes = overconfident()
    parameters = fit_platt(reported, outcomes)
    assert parameters["slope"] == pytest.approx(0.5, abs=0.05)
    assert parameters["intercept"] == pytest.approx(0, abs=0.1)
    recalibrated = apply_platt(reported, **parameters)
    # A positive slope keeps every ranking, so AUROC cannot change.
    assert (np.argsort(recalibrated, kind="stable") == np.argsort(reported, kind="stable")).all()
    before = calibration_metrics(np.column_stack([1 - reported, reported]), outcomes, 1)
    after = calibration_metrics(np.column_stack([1 - recalibrated, recalibrated]), outcomes, 1)
    assert before["calibrationSlope"] == pytest.approx(0.5, abs=0.05)
    assert after["calibrationSlope"] == pytest.approx(1, abs=0.05)
    assert after["ece"] < before["ece"] / 3
    assert after["brierScore"] < before["brierScore"]
    assert after["logLoss"] < before["logLoss"]


def test_platt_targets_keep_the_map_finite_for_separable_predictions():
    parameters = fit_platt([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
    assert all(np.isfinite(value) for value in parameters.values())
    with pytest.raises(ValueError, match="both classes"):
        fit_platt([0.1, 0.2], [1, 1])


def test_temperature_scaling_recovers_the_temperature_and_keeps_every_decision():
    random = np.random.default_rng(1)
    logits = random.normal(0, 1, (3000, 3))
    truth = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    labels = np.array([random.choice(3, p=row) for row in truth])
    reported = np.exp(3 * logits) / np.exp(3 * logits).sum(axis=1, keepdims=True)
    parameters = fit_temperature(reported, labels)
    assert parameters["temperature"] == pytest.approx(3, rel=0.1)
    recalibrated = apply_temperature(reported, **parameters)
    assert (recalibrated.argmax(axis=1) == reported.argmax(axis=1)).all()
    assert np.allclose(recalibrated.sum(axis=1), 1)
    before, after = calibration_metrics(reported, labels), calibration_metrics(recalibrated, labels)
    assert after["ece"] < before["ece"] / 3 and after["logLoss"] < before["logLoss"]


def test_calibration_metrics_follow_their_definitions():
    probabilities = [[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7]]
    labels = [0, 1, 1, 1]
    metrics = calibration_metrics(probabilities, labels, positive=1)
    risk = np.array([0.1, 0.8, 0.4, 0.7])
    outcome = np.array([0, 1, 1, 1])
    assert metrics["count"] == 4
    assert metrics["brierScore"] == pytest.approx(np.mean((risk - outcome) ** 2))
    assert metrics["logLoss"] == pytest.approx(-np.mean(np.log([0.9, 0.8, 0.4, 0.7])))
    assert metrics["observedExpectedRatio"] == pytest.approx(3 / risk.sum())
    # Equal-width bins: each nonempty bin weighs its gap by its share of units.
    expected = sum(abs(value - observed) for value, observed in zip(risk, outcome, strict=True)) / 4
    assert metrics["ece"] == pytest.approx(expected)
    assert [row["count"] for row in metrics["bins"]] == [1, 1, 1, 1]
    # Multiclass risks are the predicted class's confidence against whether it is right.
    multiclass = calibration_metrics([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]], [0, 1])
    assert multiclass["observedFraction"] == 0.5
    assert multiclass["brierScore"] == pytest.approx(
        np.mean([0.3**2 + 0.2**2 + 0.1**2, 0.1**2 + 0.7**2 + 0.6**2])
    )
