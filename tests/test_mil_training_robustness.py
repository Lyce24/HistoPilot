"""Adversarial checkpoint selection and actual GPU optimizer/resume checks."""

import copy
import runpy
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

from lightning.pytorch.utilities.exceptions import SIGTERMException  # noqa: E402

from histopilot.application.refits import epoch_budget  # noqa: E402
from histopilot.datasets.datamodule import MILDataModule  # noqa: E402
from histopilot.training.fold import train_fold  # noqa: E402
from histopilot.training.module import MILTrainModule  # noqa: E402
from histopilot.training.refit import train_refit  # noqa: E402

support = runpy.run_path(str(Path(__file__).with_name("test_mil_training.py")))


def prescribed_loss(monkeypatch, scores):
    original = MILTrainModule.on_validation_epoch_end

    def validation(module):
        probability = float(np.exp(-scores[module.current_epoch]))
        for row in module.validation_rows:
            row["probabilities"] = [
                probability if index == row["labelIndex"] else 1 - probability for index in range(2)
            ]
            row["logProbabilities"] = np.log(row["probabilities"]).tolist()
            row["logits"] = row["logProbabilities"]
        original(module)

    monkeypatch.setattr(MILTrainModule, "on_validation_epoch_end", validation)


def refit_plan(plan, epochs):
    refit = copy.deepcopy(plan)
    refit["recipe"].update(maxEpochs=epochs, minEpochs=epochs, earlyStopping=False)
    refit["recipe"].pop("fixedEpochBudget", None)
    refit["recipe"].pop("minValidationPositives", None)
    refit["epochBudget"] = {"epochs": epochs, "percentile": 75}
    for row in refit["data"]["memberships"]:
        row.update(partition="train", phase="refit")
    return refit


def test_minimum_twenty_retains_epoch_three_and_refits_for_three(tmp_path, monkeypatch):
    plan = support["tiny_plan"](tmp_path)
    plan["recipe"].update(maxEpochs=22, minEpochs=20, patience=3, earlyStopping=True)
    prescribed_loss(monkeypatch, [0.5, 0.4, 0.2] + [0.6] * 19)
    result = train_fold(plan, tmp_path / "cv")
    assert result["epochsCompleted"] == 20
    assert result["bestEpoch"] == 3
    best = torch.load(result["bestCheckpointPath"], weights_only=True, map_location="cpu")
    last = torch.load(result["lastCheckpointPath"], weights_only=True, map_location="cpu")
    assert best["epoch"] == 2 and len(best["metricsHistory"]) == 3
    assert last["epoch"] == 19 and len(last["metricsHistory"]) == 20
    budget = epoch_budget([result["bestEpoch"]] * 5, 75)
    final = train_refit(refit_plan(plan, budget), tmp_path / "refit")
    assert final["epochsCompleted"] == final["bestEpoch"] == 3


@pytest.mark.parametrize("sparse_fallback", [False, True])
def test_worsening_validation_publishes_actual_final_fixed_budget_checkpoint(
    tmp_path, monkeypatch, sparse_fallback
):
    plan = support["tiny_plan"](tmp_path)
    plan["recipe"].update(maxEpochs=5, minEpochs=0, earlyStopping=True, fixedEpochBudget=3)
    if sparse_fallback:
        plan["recipe"].update(minValidationPositives=3, checkpointMetric="validation_auroc")
    prescribed_loss(monkeypatch, [0.3, 0.6, 0.9])
    result = train_fold(plan, tmp_path / "fixed")
    assert result["bestEpoch"] == result["epochsCompleted"] == 3
    assert result["bestCheckpointPath"] == result["lastCheckpointPath"]
    best = torch.load(tmp_path / "fixed/best.ckpt", weights_only=True, map_location="cpu")
    last = torch.load(result["lastCheckpointPath"], weights_only=True, map_location="cpu")
    assert best["epoch"] == 0 and last["epoch"] == 2
    assert [row["epoch"] for row in last["metricsHistory"]] == [0, 1, 2]
    assert any(
        not torch.equal(best["state_dict"][key], value) for key, value in last["state_dict"].items()
    )


@pytest.mark.parametrize("loss_type", ["ce", "bce"])
def test_legacy_weights_remain_frozen_for_assessment_and_reject_changed_training_resume(
    tmp_path, monkeypatch, loss_type
):
    plan = support["tiny_plan"](tmp_path)
    plan["recipe"].update(classWeighting="inverse_prevalence", lossType=loss_type)
    with monkeypatch.context() as legacy:
        legacy.setattr(MILDataModule, "training_class_weights", lambda self: [0.55, 5.5])
        legacy.setattr(MILDataModule, "training_class_weight_unit", lambda self: None)
        original = train_fold(plan, tmp_path / "legacy")
    # Assessment must keep the checkpoint's original objective in its receipt.
    resumed = train_fold(plan, tmp_path / "legacy", checkpoint_path=original["lastCheckpointPath"])
    assert resumed["assessmentOnlyResume"]
    assert resumed["resolvedClassWeights"] == original["resolvedClassWeights"] == [0.55, 5.5]
    assert resumed["classWeightingUnit"] is None
    assert resumed["metrics"] == original["metrics"]
    # A partial run cannot silently mix old and corrected objectives.
    (tmp_path / "legacy/fit-complete.json").unlink()
    plan["recipe"]["maxEpochs"] = 3
    with pytest.raises(ValueError, match="Checkpoint class weights differ"):
        train_fold(plan, tmp_path / "legacy", checkpoint_path=original["lastCheckpointPath"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
@pytest.mark.parametrize("precision", ["32-true", "16-mixed", "bf16-mixed"])
def test_gpu_mixed_precision_resume_preserves_weights_optimizer_and_refit(
    tmp_path, monkeypatch, precision
):
    if precision == "bf16-mixed" and not torch.cuda.is_bf16_supported():
        pytest.skip("GPU does not support BF16")
    plan = support["tiny_plan"](tmp_path)
    plan["device"] = "cuda"
    plan["recipe"].update(
        maxEpochs=3,
        precision=precision,
        lossType="bce",
        dropout=0.25,
        gradientCheckpointing=True,
        accumulateGradBatches=2,
        gradientClipNorm=1,
        lrScheduler="cosine",
        warmupEpochs=1,
        learningRate=1e-4,
        weightDecay=5e-3,
        samplingStrategy="patient_natural",
        classWeighting="inverse_prevalence",
        instanceDropout=0.2,
        featureNoiseStd=0.05,
    )
    full = train_fold(copy.deepcopy(plan), tmp_path / "full")
    original = MILTrainModule.training_step

    def interrupt(module, batch, batch_idx):
        if module.current_epoch == 1 and batch_idx == 1:
            raise SIGTERMException()
        return original(module, batch, batch_idx)

    monkeypatch.setattr(MILTrainModule, "training_step", interrupt)
    with pytest.raises(SIGTERMException):
        train_fold(copy.deepcopy(plan), tmp_path / "resumed")
    monkeypatch.setattr(MILTrainModule, "training_step", original)
    resumed = train_fold(plan, tmp_path / "resumed", checkpoint_path=tmp_path / "resumed/last.ckpt")
    left = torch.load(full["lastCheckpointPath"], weights_only=True, map_location="cpu")
    right = torch.load(resumed["lastCheckpointPath"], weights_only=True, map_location="cpu")
    assert left["metricsHistory"] == right["metricsHistory"]
    assert left["lr_schedulers"] == right["lr_schedulers"]
    assert left.get("MixedPrecision") == right.get("MixedPrecision")
    for name, tensor in left["state_dict"].items():
        assert torch.isfinite(tensor).all()
        torch.testing.assert_close(tensor, right["state_dict"][name], rtol=0, atol=0)
    for first, second in zip(left["optimizer_states"], right["optimizer_states"], strict=True):
        assert first["param_groups"] == second["param_groups"]
        for index, state in first["state"].items():
            for name, value in state.items():
                torch.testing.assert_close(value, second["state"][index][name], rtol=0, atol=0)
    assert full["metrics"] == resumed["metrics"]
    assert resumed["classWeightingUnit"] == "patient"
    final = train_refit(refit_plan(plan, 3), tmp_path / "refit")
    payload = torch.load(final["bestCheckpointPath"], weights_only=True, map_location="cpu")
    assert payload["epoch"] == 3 and len(payload["metricsHistory"]) == 3
    assert all(torch.isfinite(tensor).all() for tensor in payload["state_dict"].values())
