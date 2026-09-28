"""Cross-validation summary statistics: parity with the training metrics and exact bootstrap."""

import math
import statistics

import numpy as np
import pytest

from histopilot import cv_summary as cv

MULTICLASS = {
    "task": "multiclass_classification",
    "unit": "slide",
    "classes": ["A", "B", "C"],
    "positiveClass": None,
}
BINARY = {
    "task": "binary_classification",
    "unit": "slide",
    "classes": ["neg", "pos"],
    "positiveClass": "pos",
}


def rows_for(target, count=40, seed=0, tie_every=0):
    rng = np.random.default_rng(seed)
    classes = target["classes"]
    rows = []
    for index in range(count):
        label = index % len(classes)
        logits = rng.normal(size=len(classes))
        logits[label] += 1.2
        if tie_every and index % tie_every == 0:
            logits = np.zeros(len(classes))  # exact ties across slides
        logs = logits - np.logaddexp.reduce(logits)
        rows.append(
            {
                "slideId": f"s{index:03d}",
                "patientId": f"p{index // 2:03d}",
                "labelIndex": label,
                "label": classes[label],
                "probabilities": np.exp(logs).tolist(),
                "logProbabilities": logs.tolist(),
            }
        )
    return rows


def test_describe_uses_sample_sd_and_ignores_missing_values():
    assert cv.describe([]) is None
    assert cv.describe([None, float("nan")]) is None
    single = cv.describe([0.8])
    assert single == {"mean": 0.8, "sd": None, "min": 0.8, "max": 0.8, "n": 1}
    stats = cv.describe([0.9, None, 0.7, 0.8])
    assert stats["n"] == 3 and stats["min"] == 0.7 and stats["max"] == 0.9
    assert stats["mean"] == pytest.approx(0.8)
    assert stats["sd"] == pytest.approx(statistics.stdev([0.9, 0.7, 0.8]))


@pytest.mark.parametrize("target,threshold", [(MULTICLASS, None), (BINARY, 0.5), (BINARY, 0.3)])
@pytest.mark.parametrize("tie_every", [0, 3])
def test_point_metrics_equal_the_training_worker_metrics(target, threshold, tie_every):
    module = pytest.importorskip("histopilot.training.module")
    rows = rows_for(target, tie_every=tie_every)
    expected = module.validated_metrics(rows, target, decision_threshold=threshold)
    actual = cv.point_metrics(rows, target, threshold)
    for name in ("count", "accuracy", "balancedAccuracy", "macroF1", "loss", "auroc", "auprc"):
        assert actual[name] == pytest.approx(expected[name], abs=1e-12), name
    assert actual["confusionMatrix"] == expected["confusionMatrix"]
    assert actual["classCounts"] == expected["classCounts"]


def test_per_class_rows_are_one_versus_rest():
    rows = rows_for(MULTICLASS, count=60, seed=3)
    metrics = cv.point_metrics(rows, MULTICLASS)
    assert [row["label"] for row in metrics["perClass"]] == ["A", "B", "C"]
    assert metrics["auroc"] == pytest.approx(
        statistics.fmean(row["auroc"] for row in metrics["perClass"])
    )
    confusion = np.asarray(metrics["confusionMatrix"])
    for index, row in enumerate(metrics["perClass"]):
        assert row["support"] == confusion[index].sum()
        assert row["recall"] == pytest.approx(confusion[index, index] / confusion[index].sum())
    from_matrix = cv.per_class_from_confusion(metrics["confusionMatrix"], MULTICLASS["classes"])
    assert [row["recall"] for row in from_matrix] == pytest.approx(
        [row["recall"] for row in metrics["perClass"]]
    )
    assert cv.per_class_from_confusion([[1]], MULTICLASS["classes"]) is None


@pytest.mark.parametrize("target,threshold", [(MULTICLASS, None), (BINARY, 0.4)])
@pytest.mark.parametrize("tie_every", [0, 4])
def test_weighted_scorer_equals_metrics_of_the_resampled_rows(target, threshold, tie_every):
    """A draw's multiplicities must score exactly like the duplicated rows themselves."""
    rows = rows_for(target, count=30, seed=5, tie_every=tie_every)
    scorer = cv.WeightedScorer(rows, target, threshold)
    rng = np.random.default_rng(1)
    weights = np.vstack(
        [np.ones(len(rows)), rng.multinomial(len(rows), np.full(len(rows), 1 / len(rows)), size=6)]
    )
    values = scorer(weights)
    for draw, multiplicity in enumerate(weights):
        expanded = [
            row for row, times in zip(rows, multiplicity, strict=True) for _ in range(int(times))
        ]
        expected = cv.point_metrics(expanded, target, threshold)
        if not values["valid"][draw]:
            assert expected["missingClasses"]
            continue
        for name in cv.BOOTSTRAP_METRICS:
            assert values[name][draw] == pytest.approx(expected[name], abs=1e-6), (name, draw)


def test_seed_ensemble_averages_probabilities_or_logits():
    first, second = rows_for(MULTICLASS, seed=1), rows_for(MULTICLASS, seed=2)
    mean = cv.seed_ensemble([first, list(reversed(second))], "mean_probability")
    assert [row["slideId"] for row in mean] == sorted(row["slideId"] for row in first)
    for row, a, b in zip(mean, first, second, strict=True):
        assert row["probabilities"] == pytest.approx(
            np.mean([a["probabilities"], b["probabilities"]], axis=0)
        )
        assert math.fsum(row["probabilities"]) == pytest.approx(1)
    logits = cv.seed_ensemble([first, second], "mean_logit")
    average = np.mean([first[0]["logProbabilities"], second[0]["logProbabilities"]], axis=0)
    assert logits[0]["logProbabilities"] == pytest.approx(average - np.logaddexp.reduce(average))
    assert cv.seed_ensemble([first], "mean_probability") is None
    assert cv.seed_ensemble([first, first[:-1]], "mean_probability") is None
    relabeled = [{**row, "labelIndex": (row["labelIndex"] + 1) % 3} for row in second]
    assert cv.seed_ensemble([first, relabeled], "mean_probability") is None


def test_bootstrap_is_reproducible_shared_and_honest_about_failures():
    rows = [rows_for(MULTICLASS, count=45, seed=seed) for seed in (1, 2, 3)]
    identities = sorted(row["slideId"] for row in rows[0])
    scorers = [
        (
            cv.WeightedScorer(sorted(group, key=lambda row: row["slideId"]), MULTICLASS),
            np.arange(len(identities)),
        )
        for group in rows
    ]
    first = cv.UnitBootstrap(identities, 400, 7).mean_of(scorers)
    again = cv.UnitBootstrap(identities, 400, 7).mean_of(scorers)
    assert np.array_equal(first["auroc"], again["auroc"], equal_nan=True)
    summary = cv.interval_summary(first, resamples=400)
    assert summary["available"] and summary["validResamples"] + summary["excludedResamples"] == 400
    lower, upper = summary["intervals"]["auroc"]["lower"], summary["intervals"]["auroc"]["upper"]
    point = statistics.fmean(cv.point_metrics(group, MULTICLASS)["auroc"] for group in rows)
    assert 0 <= lower <= point <= upper <= 1
    # A difference of a model with itself is exactly zero on every draw.
    same = {name: first[name] - first[name] for name in cv.BOOTSTRAP_METRICS}
    same["valid"] = first["valid"]
    zero = cv.interval_summary(first, resamples=400, difference=same)
    assert zero["intervals"]["auroc"] == {"lower": 0.0, "upper": 0.0}
    # Tiny cohorts miss classes in most draws; the interval is withheld, never guessed.
    tiny = rows_for(MULTICLASS, count=3, seed=9)
    ids = [row["slideId"] for row in tiny]
    sparse = cv.UnitBootstrap(ids, 200, 1).mean_of(
        [(cv.WeightedScorer(tiny, MULTICLASS), np.arange(3))]
    )
    withheld = cv.interval_summary(sparse, resamples=200)
    assert not withheld["available"] and "every class" in withheld["reason"]


def test_unit_indices_map_slides_onto_patients():
    rows = rows_for(MULTICLASS, count=6)
    patients = sorted({row["patientId"] for row in rows})
    assert cv.unit_indices(rows, patients, "patientId").tolist() == [0, 0, 1, 1, 2, 2]


def test_fold_differences_follow_metric_direction():
    left = {"f0": [0.9, 0.92], "f1": [0.8], "f2": [0.7], "only-left": [0.5]}
    right = {"f0": [0.85], "f1": [0.82], "f2": [0.7]}
    result = cv.fold_differences(left, right)
    assert (
        result["n"] == 3 and result["better"] == 1 and result["worse"] == 1 and result["tied"] == 1
    )
    assert result["values"] == pytest.approx([0.06, -0.02, 0.0])
    loss = cv.fold_differences({"f0": [0.2]}, {"f0": [0.3]}, higher_is_better=False)
    assert loss["better"] == 1 and loss["mean"] == pytest.approx(-0.1)
    assert cv.fold_differences({}, right) == {"n": 0}
