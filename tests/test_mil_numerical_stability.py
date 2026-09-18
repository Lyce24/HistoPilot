"""Regression evidence for saturated scores and fold-local objective weights."""

import copy
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

from histopilot.application.clinical import clinical_report  # noqa: E402
from histopilot.schemas.clinical import ClinicalSelection  # noqa: E402
from histopilot.training.inference import (  # noqa: E402
    _decisions,
    evaluation_metrics,
    patient_predictions,
)
from histopilot.training.module import (  # noqa: E402
    MILTrainModule,
    classification_metrics,
    prediction_rows,
)


def binary_rows(margins, positive_first=False):
    target = {
        "classes": ["positive", "negative"] if positive_first else ["negative", "positive"],
        "positiveClass": "positive",
        "task": "binary_classification",
        "unit": "patient",
    }
    positive = target["classes"].index("positive")
    # Two slides per patient exercise both aggregations, not just slide scoring.
    rows = prediction_rows(
        {
            "labels": torch.tensor([1 - positive, 1 - positive, positive, positive]),
            "slideIds": ["n0", "n1", "p0", "p1"],
            "patientIds": ["negative", "negative", "positive", "positive"],
        },
        torch.tensor(margins, dtype=torch.float64).repeat_interleave(2).reshape(-1, 1),
        target,
    )
    return rows, target


@pytest.mark.parametrize("margins", [(20, 30), (40, 50), (1000, 1100), (-1100, -1000)])
@pytest.mark.parametrize("positive_first", [False, True])
@pytest.mark.parametrize("aggregation", ["mean_probabilities", "mean_logits"])
def test_saturated_binary_ranking_agrees_across_training_inference_and_clinical(
    margins, positive_first, aggregation
):
    rows, target = binary_rows(margins, positive_first)
    metrics = classification_metrics(rows, target, aggregation)
    for unit in ("slide", "patient"):
        assert metrics[unit]["auroc"] == metrics[unit]["auprc"] == 1
    inference_aggregation = "mean" if aggregation == "mean_probabilities" else aggregation
    patients = patient_predictions(rows, inference_aggregation)
    for records in (rows, patients):
        evaluated = evaluation_metrics(_decisions(copy.deepcopy(records), target, 0.5), target, 0.5)
        assert evaluated["auroc"] == evaluated["auprc"] == 1
    predictions = {"classOrder": target["classes"], "records": rows, "patientRecords": patients}
    inference = {"decisionThreshold": 0.5, "patientAggregation": inference_aggregation}
    for unit in ("slide", "patient"):
        report = clinical_report(
            predictions,
            target,
            inference,
            ClinicalSelection(evaluationId="configuration-" + "a" * 64, unit=unit),
        )
        assert report["metrics"]["rocAuc"] == report["metrics"]["averagePrecision"] == 1
        assert report["curveSampling"]["distinctScores"] == 2
        assert all(0 <= point["threshold"] <= 1 for point in report["rocCurve"][1:])
        json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("aggregation", ["mean_probabilities", "mean_logits"])
def test_multiclass_ranking_uses_each_class_against_all_other_classes(aggregation):
    target = {
        "classes": ["a", "b", "c"],
        "unit": "patient",
        "task": "multiclass_classification",
        "positiveClass": None,
    }
    rows = prediction_rows(
        {
            "labels": torch.tensor([0, 2, 1]),
            "slideIds": ["a", "b", "c"],
            "patientIds": ["a", "b", "c"],
        },
        torch.tensor([[0.0, 40.0, 0.0], [0.0, 50.0, 0.0], [0.0, 60.0, 1.0]], dtype=torch.float64),
        target,
    )
    assert [row["probabilities"][1] for row in rows] == [1, 1, 1]
    # a and b have their positive ranked first; c has its positive ranked second.
    metrics = classification_metrics(rows, target, aggregation)
    for unit in ("slide", "patient"):
        assert metrics[unit]["auroc"] == pytest.approx(5 / 6)
        assert metrics[unit]["auprc"] == pytest.approx(5 / 6)


def test_probability_and_logit_aggregation_preserve_their_different_rankings():
    rows, target = binary_rows((0, 0))
    logits = torch.tensor([[10.0], [-2.0], [1.0], [1.0]], dtype=torch.float64)
    rows = prediction_rows(
        {
            "labels": torch.tensor([0, 0, 1, 1]),
            "slideIds": [row["slideId"] for row in rows],
            "patientIds": [row["patientId"] for row in rows],
        },
        logits,
        target,
    )
    assert classification_metrics(rows, target, "mean_probabilities")["selected"]["auroc"] == 1
    assert classification_metrics(rows, target, "mean_logits")["selected"]["auroc"] == 0


def test_real_ties_remain_ties_and_legacy_probabilities_still_score():
    rows, target = binary_rows((1000, 1000))
    for values in (
        rows,
        [{k: v for k, v in row.items() if k not in {"logits", "logProbabilities"}} for row in rows],
    ):
        metrics = classification_metrics(values, target)
        assert metrics["selected"]["auroc"] == metrics["selected"]["auprc"] == 0.5


def test_legacy_subnormal_probabilities_remain_ordered_and_zero_ties_stay_grouped():
    _, target = binary_rows((0, 0))
    rows = [
        {
            "slideId": str(index),
            "patientId": str(index),
            "labelIndex": label,
            "label": target["classes"][label],
            "probabilities": [1 - probability, probability],
        }
        for index, (label, probability) in enumerate([(0, 0.0), (0, 0.0), (1, 5e-324), (1, 1e-310)])
    ]
    metrics = classification_metrics(rows, target)
    for unit in ("slide", "patient"):
        assert metrics[unit]["auroc"] == metrics[unit]["auprc"] == 1
    clinical = clinical_report(
        {
            "classOrder": target["classes"],
            "records": rows,
            "patientRecords": patient_predictions(rows),
        },
        target,
        {"decisionThreshold": 0.5, "patientAggregation": "mean"},
        ClinicalSelection(evaluationId="configuration-" + "a" * 64),
    )
    assert clinical["metrics"]["rocAuc"] == clinical["metrics"]["averagePrecision"] == 1
    json.dumps(clinical, allow_nan=False)


@pytest.mark.parametrize("loss_type", ["ce", "bce", "focal"])
def test_resume_rejects_changed_class_weight_buffers_but_allows_frozen_inference(loss_type):
    _, target = binary_rows((0, 0))
    recipe = {
        "model": "mean_pool",
        "embedDim": 4,
        "dropout": 0,
        "lossType": loss_type,
        "classWeighting": "inverse_prevalence",
    }
    original = MILTrainModule(2, target, recipe, class_weights=[0.55, 5.5])
    checkpoint = {"state_dict": original.state_dict()}
    restored = MILTrainModule(**original.hparams)
    restored.on_load_checkpoint(checkpoint)
    restored.load_state_dict(checkpoint["state_dict"])
    changed = MILTrainModule(2, target, recipe, class_weights=[1, 1])
    with pytest.raises(ValueError, match="class weights differ"):
        changed.on_load_checkpoint(checkpoint)


def test_moderate_scores_match_independent_reference_metrics():
    from sklearn.metrics import average_precision_score, roc_auc_score

    rng = np.random.default_rng(72)
    target = {"classes": ["a", "b", "c"], "task": "multiclass_classification", "unit": "slide"}
    labels = np.tile(np.arange(3), 40)
    logits = torch.from_numpy(rng.normal(size=(len(labels), 3)))
    rows = prediction_rows(
        {
            "labels": torch.from_numpy(labels),
            "slideIds": list(map(str, range(len(labels)))),
            "patientIds": list(map(str, range(len(labels)))),
        },
        logits,
        target,
    )
    probabilities = torch.softmax(logits, dim=-1).numpy()
    selected = classification_metrics(rows, target)["selected"]
    assert selected["auroc"] == pytest.approx(
        roc_auc_score(labels, probabilities, multi_class="ovr")
    )
    expected_ap = np.mean(
        [average_precision_score(labels == i, probabilities[:, i]) for i in range(3)]
    )
    assert selected["auprc"] == pytest.approx(expected_ap)
