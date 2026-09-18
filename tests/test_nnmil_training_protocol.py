"""Actual nnMIL checkpoint selection, completed-fit replay, and interrupted fitting."""

# ruff: noqa: E402
import json
import runpy
from copy import deepcopy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

from lightning.pytorch.utilities.exceptions import SIGTERMException

from histopilot.datasets.mil import collate_mil
from histopilot.training.fold import train_fold
from histopilot.training.module import MILTrainModule

support = runpy.run_path(str(Path(__file__).with_name("test_mil_training.py")))
robustness = runpy.run_path(str(Path(__file__).with_name("test_mil_training_robustness.py")))


def nnmil_plan(tmp_path):
    plan = support["tiny_plan"](tmp_path)
    plan["recipe"].update(
        model="nnmil",
        attentionDim=2,
        maxEpochs=4,
        minEpochs=1,
        bagSizeMode="training_median",
        bagSizeFraction=0.5,
        nnmilBatchSampler="class_balanced",
        nnmilWindowSeed=23,
    )
    return plan


@pytest.mark.parametrize(
    "selection,epoch,score",
    [
        ("best_validation", 1, 0.2),
        ("latest", 4, 0.8),
    ],
)
def test_selected_checkpoint_and_best_epoch_are_distinct_evidence(
    tmp_path, monkeypatch, selection, epoch, score
):
    plan = nnmil_plan(tmp_path)
    plan["recipe"]["nnmilCheckpointSelection"] = selection
    robustness["prescribed_loss"](monkeypatch, [0.2, 0.5, 0.6, 0.8])
    result = train_fold(plan, tmp_path / "fit")
    assert result["bestEpoch"] == 1
    assert result["selectedEpoch"] == epoch
    assert result["epochsCompleted"] == 4
    assert result["checkpointSelection"] == selection
    assert result["bestValidationScore"] == pytest.approx(score)
    assert result["validationBestScore"] == pytest.approx(0.2)
    assert result["effectiveRecipe"]["bagSize"] == 2
    assert result["nnmilPlanning"]["trainingSlideCount"] == 4
    assert plan["recipe"]["bagSize"] == 3
    assert result["bestCheckpointPath"].endswith(
        "last.ckpt" if selection == "latest" else "best.ckpt"
    )
    selected = torch.load(result["bestCheckpointPath"], weights_only=True, map_location="cpu")
    assert selected["epoch"] + 1 == epoch
    for path in result["predictions"].values():
        rows = json.loads(Path(path).read_text())
        assert rows["checkpointPath"] == result["bestCheckpointPath"]
    replay = train_fold(plan, tmp_path / "fit", checkpoint_path=result["lastCheckpointPath"])
    assert replay["assessmentOnlyResume"]
    assert replay["bestEpoch"] == result["bestEpoch"]
    assert replay["selectedEpoch"] == result["selectedEpoch"]
    assert replay["metrics"] == result["metrics"]
    assert replay["nnmilPlanning"] == result["nnmilPlanning"]


def test_latest_resume_verifies_validation_best_artifact(tmp_path):
    plan = nnmil_plan(tmp_path)
    plan["recipe"].update(nnmilCheckpointSelection="latest", maxEpochs=1)
    result = train_fold(plan, tmp_path / "fit")
    Path(result["validationBestCheckpointPath"]).write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="checkpoint changed"):
        train_fold(plan, tmp_path / "fit", checkpoint_path=result["lastCheckpointPath"])


def test_fixed_budget_keeps_actual_best_and_selected_final_epochs(tmp_path, monkeypatch):
    plan = nnmil_plan(tmp_path)
    plan["recipe"].update(fixedEpochBudget=3, nnmilCheckpointSelection="best_validation")
    robustness["prescribed_loss"](monkeypatch, [0.2, 0.5, 0.8])
    result = train_fold(plan, tmp_path / "fit")
    assert result["bestEpoch"] == 1
    assert result["selectedEpoch"] == result["epochsCompleted"] == 3
    assert result["checkpointSelection"] == "final_epoch"
    assert result["bestCheckpointPath"] == result["lastCheckpointPath"]


@pytest.mark.parametrize("workers", [0, 1])
def test_interrupted_nnmil_replays_patch_views_optimizer_and_window_predictions(
    tmp_path, monkeypatch, workers
):
    plan = nnmil_plan(tmp_path)
    plan["resources"]["dataLoaderWorkers"] = workers
    plan["recipe"].update(
        maxEpochs=3,
        dropout=0.2,
        instanceDropout=0.1,
        featureNoiseStd=0.01,
        lrScheduler="cosine",
        warmupEpochs=1,
    )
    original = MILTrainModule.training_step

    def verify_views(module, batch, batch_idx):
        data = module.trainer.datamodule
        indices = list(data.train_batch_sampler)[batch_idx]
        assert {index[0] for index in indices} == {module.current_epoch}
        expected = collate_mil([data.train_dataset[index] for index in indices])
        assert batch["slideIds"] == expected["slideIds"]
        torch.testing.assert_close(batch["features"], expected["features"], rtol=0, atol=0)
        return original(module, batch, batch_idx)

    monkeypatch.setattr(MILTrainModule, "training_step", verify_views)
    full = train_fold(deepcopy(plan), tmp_path / "full")

    def interrupt(module, batch, batch_idx):
        if module.current_epoch == 1 and batch_idx == 1:
            raise SIGTERMException()
        return verify_views(module, batch, batch_idx)

    monkeypatch.setattr(MILTrainModule, "training_step", interrupt)
    with pytest.raises(SIGTERMException):
        train_fold(deepcopy(plan), tmp_path / "resumed")
    monkeypatch.setattr(MILTrainModule, "training_step", verify_views)
    resumed = train_fold(plan, tmp_path / "resumed", checkpoint_path=tmp_path / "resumed/last.ckpt")
    left = torch.load(full["lastCheckpointPath"], weights_only=True, map_location="cpu")
    right = torch.load(resumed["lastCheckpointPath"], weights_only=True, map_location="cpu")
    assert left["metricsHistory"] == right["metricsHistory"]
    assert left["lr_schedulers"] == right["lr_schedulers"]
    for name, tensor in left["state_dict"].items():
        torch.testing.assert_close(tensor, right["state_dict"][name], rtol=0, atol=0)
    for first, second in zip(left["optimizer_states"], right["optimizer_states"], strict=True):
        assert first["param_groups"] == second["param_groups"]
        for index, state in first["state"].items():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(value, second["state"][index][key], rtol=0, atol=0)
                else:
                    assert value == second["state"][index][key]
    assert full["metrics"] == resumed["metrics"]
    assert full["selectedEpoch"] == resumed["selectedEpoch"]
