import copy

import numpy as np
import pytest

from histopilot.statistics import one_slide_per_patient, patient_analysis, patient_bootstrap

TARGET = {
    "task": "binary_classification",
    "classes": ["wt", "mut"],
    "positiveClass": "mut",
    "unit": "patient",
}
POLICY = {"bootstrapResamples": 200, "bootstrapSeed": 7, "oneSlideSeed": 42}


def records():
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    scores = [0.1, 0.4, 0.6, 0.6, 0.2, 0.5, 0.6, 0.9]
    return [
        {
            "patientId": f"p{i}",
            "slideId": f"s{i}",
            "labelIndex": y,
            "label": TARGET["classes"][y],
            "probabilities": [1 - p, p],
        }
        for i, (y, p) in enumerate(zip(labels, scores, strict=True))
    ]


def test_bootstrap_matches_explicit_sklearn_patient_resampling_with_ties():
    metrics = pytest.importorskip("sklearn.metrics")
    rows = records()
    result = patient_bootstrap(rows, TARGET, POLICY)
    y = np.asarray([row["labelIndex"] for row in rows])
    p = np.asarray([row["probabilities"][1] for row in rows])
    rng = np.random.default_rng(7)
    values = []
    for _ in range(200):
        indices = rng.integers(len(rows), size=len(rows))
        if len(np.unique(y[indices])) == 2:
            values.append(
                [
                    metrics.roc_auc_score(y[indices], p[indices]),
                    metrics.average_precision_score(y[indices], p[indices]),
                ]
            )
    assert result["validResamples"] == len(values)
    for i, name in enumerate(("auroc", "auprc")):
        low, high = np.quantile(np.asarray(values)[:, i], [0.025, 0.975])
        assert result["intervals"][name] == pytest.approx({"lower": low, "upper": high})


def test_paired_bootstrap_uses_identical_patient_draws_and_is_order_invariant():
    rows = records()
    result = patient_bootstrap(rows, TARGET, POLICY, other=rows[::-1])
    assert result["paired"] and result["available"]
    assert result["intervals"] == {name: {"lower": 0, "upper": 0} for name in ("auroc", "auprc")}
    assert result["estimates"]["auroc"]["difference"] == 0
    assert result == patient_bootstrap(rows[::-1], TARGET, POLICY, other=rows)


def test_pair_differences_match_direct_resampling_not_independent_intervals():
    metrics = pytest.importorskip("sklearn.metrics")
    rows, other = records(), records()
    for row in other:
        row["probabilities"] = row["probabilities"][::-1]
    result = patient_bootstrap(rows, TARGET, POLICY, other=other)
    rng = np.random.default_rng(7)
    labels = np.array([row["labelIndex"] for row in rows])
    p = np.array([row["probabilities"][1] for row in rows])
    q = 1 - p
    differences = []
    for _ in range(200):
        draw = rng.integers(len(rows), size=len(rows))
        if len(np.unique(labels[draw])) == 2:
            differences.append(
                [
                    function(labels[draw], p[draw]) - function(labels[draw], q[draw])
                    for function in (metrics.roc_auc_score, metrics.average_precision_score)
                ]
            )
    limits = np.quantile(differences, [0.025, 0.975], axis=0)
    for i, name in enumerate(("auroc", "auprc")):
        assert result["intervals"][name] == pytest.approx(
            {"lower": limits[0, i], "upper": limits[1, i]}
        )


@pytest.mark.parametrize("change", ["patient", "label", "duplicate"])
def test_paired_patients_and_outcomes_must_match_exactly(change):
    rows, other = records(), records()
    if change == "patient":
        other.pop()
    elif change == "label":
        other[0].update(label="mut", labelIndex=1)
    else:
        other.append(other[0])
    with pytest.raises(ValueError):
        patient_bootstrap(rows, TARGET, POLICY, other=other)


def test_one_slide_choice_depends_only_on_patient_and_slide_identity():
    rows = records()
    rows += [{**row, "slideId": row["slideId"] + "-extra"} for row in rows]
    selected = one_slide_per_patient(rows, 42)
    changed = copy.deepcopy(rows[::-1])
    for row in changed:
        row.update(label=None, labelIndex=None, probabilities=[0.99, 0.01])
    assert [row["slideId"] for row in selected] == [
        row["slideId"] for row in one_slide_per_patient(changed, 42)
    ]
    assert len(selected) == len({row["patientId"] for row in rows})


def test_patient_analysis_never_bootstraps_correlated_slides_as_patients():
    patients = records()
    slides = patients + [{**row, "slideId": row["slideId"] + "-extra"} for row in patients]
    result = patient_analysis(slides, patients, TARGET, POLICY)
    assert result["uncertainty"]["patientCount"] == 8
    assert result["oneSlidePerPatient"]["patientCount"] == 8
    assert result["oneSlidePerPatient"]["metrics"] == result["uncertainty"]["estimates"]
    slides[0]["patientIdSource"] = "slide_fallback"
    assert not patient_analysis(slides, patients, TARGET, POLICY)["uncertainty"]["available"]


def test_insufficient_class_counts_report_unavailable_instead_of_false_precision():
    rows = records()[:5]
    result = patient_bootstrap(rows, TARGET, POLICY)
    assert not result["available"] and "at least two" in result["reason"]
    assert result["estimates"]["auroc"] is not None
    assert not patient_bootstrap(rows[:4], TARGET, POLICY)["available"]


def test_saved_log_probabilities_preserve_extreme_ranking():
    rows = records()
    logits = np.array([-2000, -1999, -1998, -1997, 1997, 1998, 1999, 2000])
    for row, z in zip(rows, logits, strict=True):
        logs = np.array([0.0, z]) - np.logaddexp(0.0, z)
        row.update(logProbabilities=logs.tolist(), probabilities=np.exp(logs).tolist())
    result = patient_bootstrap(rows, TARGET, POLICY)
    assert result["estimates"] == {"auroc": 1, "auprc": 1}


def test_multiclass_macro_ovr_intervals_match_explicit_resampling():
    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(11)
    probabilities = rng.dirichlet([1, 1, 1], size=18)
    labels = np.arange(18) % 3
    target = {"task": "multiclass_classification", "unit": "patient", "classes": ["a", "b", "c"]}
    rows = [
        {
            "patientId": f"p{i:02d}",
            "labelIndex": int(y),
            "label": target["classes"][y],
            "probabilities": p.tolist(),
        }
        for i, (y, p) in enumerate(zip(labels, probabilities, strict=True))
    ]
    result = patient_bootstrap(rows, target, POLICY)
    rng = np.random.default_rng(7)
    draws = []
    for _ in range(200):
        ids = rng.integers(18, size=18)
        if len(np.unique(labels[ids])) != 3:
            continue
        draws.append(
            [
                np.mean([function(labels[ids] == c, probabilities[ids, c]) for c in range(3)])
                for function in (metrics.roc_auc_score, metrics.average_precision_score)
            ]
        )
    limits = np.quantile(draws, [0.025, 0.975], axis=0)
    for i, metric in enumerate(("auroc", "auprc")):
        assert result["intervals"][metric] == pytest.approx(
            {"lower": limits[0, i], "upper": limits[1, i]}
        )
