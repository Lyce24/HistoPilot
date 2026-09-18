"""Numerical contracts for losses, patient aggregation and optimizer controls."""

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")
from torch.nn import functional as F  # noqa: E402

from histopilot.models.pooling import PoolingMIL  # noqa: E402
from histopilot.training.module import (  # noqa: E402
    MILTrainModule,
    aggregate_patients,
    class_logits,
    classification_metrics,
    prediction_rows,
)


def target(*, positive_first=False, multiclass=False):
    classes = ["positive", "negative"] if positive_first else ["negative", "positive"]
    return {
        "classes": classes + ["other"] if multiclass else classes,
        "positiveClass": None if multiclass else "positive",
        "unit": "patient",
        "task": "multiclass_classification" if multiclass else "binary_classification",
    }


def module(**overrides):
    values = {
        "model": "abmil",
        "optimizer": "adamw",
        "learningRate": 1e-4,
        "weightDecay": 5e-3,
        "maxEpochs": 20,
        "checkpointMetric": "validation_loss",
        "embedDim": 4,
        "attentionDim": 3,
        "dropout": 0,
        **overrides,
    }
    return MILTrainModule(2, target(), values)


@pytest.mark.parametrize("loss_type", ["ce", "focal"])
def test_weighted_losses_match_oceanpath_per_bag_convention(loss_type):
    logits = torch.tensor([[1.3, -0.2], [-0.8, 0.7]], requires_grad=True)
    labels = torch.tensor([1, 0])
    model = module(lossType=loss_type, classWeights=[0.5, 2.0], focalGamma=1.5)
    ce = F.cross_entropy(logits, labels, weight=torch.tensor([0.5, 2.0]), reduction="none")
    expected = ce if loss_type == "ce" else (1 - torch.exp(-ce)).pow(1.5) * ce
    actual = model.loss(logits, labels)
    torch.testing.assert_close(actual, expected)
    actual.mean().backward()
    assert torch.isfinite(logits.grad).all()


def test_label_smoothing_affects_training_only_and_assessment_stays_unweighted():
    model = module(classWeights=[0.5, 2.0], labelSmoothing=0.2)
    logits = torch.tensor([[1.3, -0.2], [-0.8, 0.7]])
    labels = torch.tensor([1, 0])
    torch.testing.assert_close(
        model.loss(logits, labels),
        F.cross_entropy(
            logits,
            labels,
            weight=torch.tensor([0.5, 2.0]),
            label_smoothing=0.2,
            reduction="none",
        ),
    )
    rows = prediction_rows(
        {"labels": labels, "slideIds": ["a", "b"], "patientIds": ["a", "b"]}, logits, target()
    )
    assert classification_metrics(rows, target())["selected"]["loss"] == pytest.approx(
        F.cross_entropy(logits.double(), labels).item()
    )


@pytest.mark.parametrize("positive_first", [True, False])
def test_binary_loss_and_predictions_honor_frozen_positive_class_order(positive_first):
    frozen = target(positive_first=positive_first)
    positive = frozen["classes"].index("positive")
    weights = [4.0 if index == positive else 2.0 for index in range(2)]
    recipe = {**module().recipe, "lossType": "bce", "classWeights": weights}
    model = MILTrainModule(2, frozen, recipe)
    assert model.model.classifier.out_features == 1
    logits = torch.tensor([[2.0], [-3.0]])
    labels = torch.tensor([positive, 1 - positive])
    expected = F.binary_cross_entropy_with_logits(
        logits[:, 0], torch.tensor([1.0, 0.0]), pos_weight=torch.tensor(2.0), reduction="none"
    )
    torch.testing.assert_close(model.loss(logits, labels), expected)
    rows = prediction_rows(
        {"labels": labels, "slideIds": ["a", "b"], "patientIds": ["a", "b"]}, logits, frozen
    )
    assert rows[0]["probabilities"][positive] == pytest.approx(torch.sigmoid(logits[0, 0]).item())
    assert rows[0]["logits"][positive] == 2
    assert classification_metrics(rows, frozen)["selected"]["accuracy"] == 1
    assert classification_metrics(rows, frozen)["selected"]["auroc"] == 1


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"lossType": "bce", "labelSmoothing": 0.1}, "smoothing"),
        ({"lossType": "focal", "labelSmoothing": 0.1}, "smoothing"),
        ({"classWeighting": "inverse_prevalence"}, "training fold"),
        ({"classWeights": [1]}, "one finite positive"),
        ({"classWeights": [1, float("nan")]}, "one finite positive"),
        ({"classWeights": [1, 0]}, "one finite positive"),
    ],
)
def test_invalid_loss_configurations_fail_before_fitting(changes, message):
    with pytest.raises(ValueError, match=message):
        module(**changes)


def test_binary_loss_rejects_multiclass_targets():
    with pytest.raises(ValueError, match="binary target"):
        MILTrainModule(2, target(multiclass=True), {**module().recipe, "lossType": "bce"})


def test_binary_logit_expansion_preserves_precision_and_gradient_for_interpretation():
    logits = torch.tensor([[2.0]], dtype=torch.float64, requires_grad=True)
    expanded = class_logits(logits, target(positive_first=True))
    assert expanded.dtype == logits.dtype
    torch.testing.assert_close(expanded, torch.tensor([[2.0, 0.0]], dtype=torch.float64))
    expanded.sum().backward()
    torch.testing.assert_close(logits.grad, torch.ones_like(logits))


def test_fold_resolved_class_weights_are_checkpoint_hyperparameters():
    recipe = {**module().recipe, "classWeighting": "inverse_prevalence"}
    model = MILTrainModule(2, target(), recipe, class_weights=[0.75, 1.5])
    restored = MILTrainModule(**model.hparams)
    restored.load_state_dict(model.state_dict())
    assert restored.hparams["class_weights"] == [0.75, 1.5]
    torch.testing.assert_close(restored.loss.weight, torch.tensor([0.75, 1.5]))


def test_patient_logit_aggregation_is_distinct_from_mean_probability_and_shift_invariant():
    logits = torch.tensor([[0.0, 4.0], [0.0, -1.0]], dtype=torch.float64)
    rows = prediction_rows(
        {"labels": torch.tensor([1, 1]), "slideIds": ["a", "b"], "patientIds": ["p", "p"]},
        logits,
        target(),
    )
    actual = aggregate_patients(rows, "mean_logits")[0]
    expected = torch.softmax(logits.mean(dim=0), dim=-1).tolist()
    assert actual["probabilities"] == pytest.approx(expected)
    assert actual["probabilities"] != pytest.approx(aggregate_patients(rows)[0]["probabilities"])
    legacy = [{key: value for key, value in row.items() if key != "logits"} for row in rows]
    assert aggregate_patients(legacy, "mean_logits")[0]["probabilities"] == pytest.approx(expected)
    shifted = copy.deepcopy(rows)
    shifted[0]["logits"] = [value + 1000 for value in shifted[0]["logits"]]
    assert aggregate_patients(shifted, "mean_logits")[0]["probabilities"] == pytest.approx(expected)
    metrics = classification_metrics(rows, target(), "mean_logits")
    assert metrics["patientAggregation"] == "mean_logits"
    assert metrics["selected"]["loss"] == pytest.approx(-np.log(expected[1]))


def test_patient_logit_aggregation_rejects_corrupted_raw_logits():
    rows = [
        {
            "patientId": "a",
            "slideId": "a",
            "labelIndex": 1,
            "label": "positive",
            "probabilities": [0.5, 0.5],
            "logits": [0, 3],
        }
    ]
    with pytest.raises(ValueError, match="agree"):
        aggregate_patients(rows, "mean_logits")
    rows[0]["logits"] = [0, float("inf")]
    with pytest.raises(ValueError, match="finite logits"):
        aggregate_patients(rows, "mean_logits")


@pytest.mark.parametrize("pooling", ["mean", "max"])
def test_pooling_baselines_exclude_padding_and_preserve_patch_permutation(pooling):
    torch.manual_seed(22)
    model = PoolingMIL(2, 2, pooling=pooling, embed_dim=3, dropout=0)
    features = torch.randn(1, 3, 2)
    expected = model(features)
    padded = torch.cat([features, torch.full((1, 2, 2), float("nan"))], dim=1)
    mask = torch.tensor([[True, True, True, False, False]])
    torch.testing.assert_close(model(padded, mask), expected)
    torch.testing.assert_close(model(features[:, [2, 0, 1]]), expected)
    with pytest.raises(ValueError, match="learned attention"):
        model(features, return_attention=True)
    with pytest.raises(ValueError, match="at least one"):
        model(features, torch.zeros(1, 3, dtype=torch.bool))
    classifier = module(model=f"{pooling}_pool", lossType="bce")
    assert classifier(features).shape == (1, 1)


def test_separate_learning_rates_cover_parameters_once_and_apply_adam_options():
    model = module(
        aggregatorLearningRate=2e-5, headLearningRate=8e-4, adamBetas=[0.8, 0.95], adamEps=1e-6
    )
    optimizer = model.configure_optimizers()
    assert [group["lr"] for group in optimizer.param_groups] == [2e-5, 8e-4]
    assert optimizer.defaults["betas"] == (0.8, 0.95)
    assert optimizer.defaults["eps"] == 1e-6
    all_ids = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    assert len(all_ids) == len(set(all_ids))
    assert set(all_ids) == {id(parameter) for parameter in model.parameters()}
    assert {id(parameter) for parameter in optimizer.param_groups[1]["params"]} == {
        id(parameter) for parameter in model.model.classifier.parameters()
    }


@pytest.mark.parametrize(
    "metric,values",
    [
        ("validation_loss", [1.0, 1.2]),
        ("validation_auroc", [0.7, 0.6]),
    ],
)
def test_plateau_uses_checkpoint_direction_and_configured_patience(metric, values):
    configured = module(
        lrScheduler="plateau", checkpointMetric=metric, lrPlateauPatience=0, lrGamma=0.2
    ).configure_optimizers()
    assert configured["lr_scheduler"]["monitor"] == metric
    scheduler = configured["lr_scheduler"]["scheduler"]
    for value in values:
        scheduler.step(value)
    assert configured["optimizer"].param_groups[0]["lr"] == pytest.approx(2e-5)


def test_step_scheduler_resume_preserves_learning_rate_trajectory():
    configured = module(lrScheduler="step", lrStepSize=2, lrGamma=0.3).configure_optimizers()
    optimizer, scheduler = configured["optimizer"], configured["lr_scheduler"]["scheduler"]
    for _ in range(3):
        optimizer.step()
        scheduler.step()
    state = copy.deepcopy(scheduler.state_dict())
    optim_state = copy.deepcopy(optimizer.state_dict())
    resumed = module(lrScheduler="step", lrStepSize=2, lrGamma=0.3).configure_optimizers()
    resumed["optimizer"].load_state_dict(optim_state)
    resumed["lr_scheduler"]["scheduler"].load_state_dict(state)
    for _ in range(5):
        optimizer.step()
        scheduler.step()
        resumed["optimizer"].step()
        resumed["lr_scheduler"]["scheduler"].step()
        assert resumed["optimizer"].param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]


def test_cosine_preserves_differential_learning_rate_ratio_to_final_floor():
    configured = module(
        lrScheduler="cosine",
        maxEpochs=5,
        aggregatorLearningRate=2e-5,
        headLearningRate=8e-4,
        finalLrFraction=0.01,
    ).configure_optimizers()
    optimizer = configured["optimizer"]
    for _ in range(4):
        optimizer.step()
        configured["lr_scheduler"]["scheduler"].step()
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([2e-7, 8e-6])
