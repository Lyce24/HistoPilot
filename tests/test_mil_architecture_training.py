"""Real fits of CLAM, TransMIL, DSMIL, DTFD-MIL and ACMIL through the fold worker.

Each architecture trains, reloads its selected checkpoint and predicts external slides
exactly as it scored its own assessment fold; its extra loss terms take the same
patient weights as the slide loss; an interrupted run resumes to the same model; and its
attention maps pass the interpretation worker's checks.
"""

import copy
import json
import runpy
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

from lightning.pytorch.utilities.exceptions import SIGTERMException  # noqa: E402

from histopilot.application.predictors import checkpoint_snapshot  # noqa: E402
from histopilot.models import catalog  # noqa: E402
from histopilot.storage.packed import stat_stamp  # noqa: E402
from histopilot.training.fold import train_fold  # noqa: E402
from histopilot.training.inference import evaluate  # noqa: E402
from histopilot.training.module import MILTrainModule, class_logits  # noqa: E402

support = runpy.run_path(str(Path(__file__).with_name("test_mil_training.py")))

NEW_MODELS = ("clam_sb", "clam_mb", "transmil", "dsmil", "dtfd", "acmil")
# Small widths for four-dimensional synthetic features.
SMALL = {
    "embedDim": 8,
    "transmilHeads": 2,
    "transmilLandmarks": 4,
    "clamAttentionDim": 4,
    "dsmilQueryDim": 4,
    "dtfdAttentionDim": 4,
    "acmilAttentionDim": 4,
}


def plan_for(tmp_path, name, classes=2, **recipe):
    plan = support["tiny_plan"](tmp_path, classes)
    plan["recipe"].update(model=name, **SMALL, **recipe)
    return plan


def inference_plan(training, checkpoint):
    data = copy.deepcopy(training["data"])
    data["memberships"] = [row for row in data["memberships"] if row["partition"] == "test"]
    ids = {row["slideId"] for row in data["memberships"]}
    data["featureFiles"] = {key: row for key, row in data["featureFiles"].items() if key in ids}
    for entry in data["featureFiles"].values():
        entry.update(stat_stamp(Path(entry["path"]).stat()))
    data["sourceStamps"] = {row["path"]: row for row in data["featureFiles"].values()}
    return {
        "kind": "evaluation",
        "runId": "architecture-evaluation",
        "method": "refit",
        "target": training["target"],
        "data": data,
        "resources": training["resources"],
        "device": "cpu",
        "checkpoints": [checkpoint],
        "inference": {
            "loadingPolicy": "per_slide",
            "batchSize": 2,
            "numWorkers": 0,
            "precision": "float32",
            "patientAggregation": "mean",
            "decisionThreshold": 0.5,
        },
    }


@pytest.mark.parametrize(
    "name,loss,classes",
    [(name, "ce", 3) for name in NEW_MODELS]
    + [(name, "bce", 2) for name in NEW_MODELS if name != "clam_mb"],
)
def test_architecture_fits_reloads_and_predicts_new_slides_as_it_scored_its_fold(
    tmp_path, name, loss, classes
):
    plan = plan_for(tmp_path, name, classes, lossType=loss, maxEpochs=2)
    result = train_fold(plan, tmp_path / "fit")
    assert result["state"] == "succeeded" and result["epochsCompleted"] == 2
    if catalog.auxiliary_loss(name):
        assert result["trainingObjective"].endswith(f"+{name}_auxiliary_losses")
    history = json.loads((tmp_path / "fit/history.json").read_text())
    assert all(np.isfinite(epoch["trainingLoss"]) for epoch in history)
    selected = MILTrainModule.load_from_checkpoint(result["bestCheckpointPath"], weights_only=True)
    assert type(selected.model).__module__ == f"histopilot.models.{name.split('_')[0]}"
    checkpoint = checkpoint_snapshot(result["bestCheckpointPath"], tmp_path / "fit")
    evaluate(inference_plan(plan, checkpoint), tmp_path / "external")
    predictions = json.loads((tmp_path / "external/predictions.json").read_text())["records"]
    assessment = json.loads(Path(result["predictions"]["assessment"]).read_text())["records"]
    by_slide = {row["slideId"]: row for row in assessment}
    assert {row["slideId"] for row in predictions} == set(by_slide)
    for row in predictions:
        assert row["probabilities"] == pytest.approx(by_slide[row["slideId"]]["probabilities"])
        assert len(row["probabilities"]) == classes


@pytest.mark.parametrize("name", [name for name in NEW_MODELS if catalog.auxiliary_loss(name)])
def test_auxiliary_losses_take_the_patient_loss_weights(tmp_path, name):
    plan = plan_for(tmp_path, name)
    module = MILTrainModule(4, plan["target"], plan["recipe"])
    module.log = lambda *_args, **_kwargs: None
    torch.manual_seed(8)
    batch = {
        "features": torch.randn(2, 5, 4),
        "mask": torch.tensor([[True] * 5, [True] * 3 + [False] * 2]),
        "labels": torch.tensor([0, 1]),
        "lossWeights": torch.tensor([1.5, 0.5]),
    }
    torch.manual_seed(9)
    output = module.model.training_output(
        batch["features"], batch["mask"], batch["labels"], module.loss
    )
    per_slide = (
        output["bagLossWeight"]
        * torch.nn.functional.cross_entropy(output["logits"], batch["labels"], reduction="none")
        + output["auxiliaryLoss"]
    )
    torch.manual_seed(9)
    actual = module.training_step(batch, 0)
    torch.testing.assert_close(actual, (per_slide * batch["lossWeights"]).mean())


@pytest.mark.parametrize("name", NEW_MODELS)
def test_cancelled_fit_resumes_to_the_uninterrupted_model(tmp_path, monkeypatch, name):
    # Dropout, ACMIL's masking and DTFD-MIL's random pseudo-bags all draw random numbers.
    plan = plan_for(tmp_path, name, dropout=0.25, maxEpochs=3)
    full = train_fold(copy.deepcopy(plan), tmp_path / "uninterrupted")
    original = MILTrainModule.training_step

    def interrupt(module, batch, batch_idx):
        if module.global_step >= 3:
            raise SIGTERMException()
        return original(module, batch, batch_idx)

    monkeypatch.setattr(MILTrainModule, "training_step", interrupt)
    output = tmp_path / "interrupted"
    with pytest.raises(SIGTERMException):
        train_fold(copy.deepcopy(plan), output)
    monkeypatch.setattr(MILTrainModule, "training_step", original)
    resumed = train_fold(plan, output, checkpoint_path=output / "last.ckpt")
    full_state = torch.load(full["lastCheckpointPath"], map_location="cpu", weights_only=True)
    resumed_state = torch.load(resumed["lastCheckpointPath"], map_location="cpu", weights_only=True)
    assert full_state["metricsHistory"] == resumed_state["metricsHistory"]
    for key, value in full_state["state_dict"].items():
        torch.testing.assert_close(value, resumed_state["state_dict"][key], rtol=0, atol=0)


@pytest.mark.parametrize("name", NEW_MODELS)
def test_attention_maps_come_from_the_predicting_forward_pass(tmp_path, name):
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("PIL.Image")
    from histopilot.training.attention import attention_note, interpret

    attention_support = runpy.run_path(str(Path(__file__).with_name("test_attention_execution.py")))
    plan = attention_support["attention_plan"](tmp_path)
    path = Path(plan["checkpoints"][0]["path"])
    payload = torch.load(path, weights_only=True)
    recipe = {**payload["hyper_parameters"]["recipe"], "model": name, **SMALL}
    model = MILTrainModule(4, plan["target"], recipe)
    payload["hyper_parameters"] = dict(model.hparams)
    payload["state_dict"] = model.state_dict()
    path = tmp_path / f"{name}.ckpt"
    torch.save(payload, path)
    plan["checkpoints"] = [checkpoint_snapshot(path, tmp_path)]
    result = interpret(plan, tmp_path / "attention")
    member = json.loads((tmp_path / "attention/slide-0-member-0.json").read_text())
    model.eval()
    with h5py.File(plan["slides"][0]["featurePath"]) as handle, torch.inference_mode():
        features = torch.tensor(handle["features"][:]).unsqueeze(0)
        output = model.prediction_output(features, return_attention=True)
    expected = class_logits(output["logits"], plan["target"]).softmax(-1)[0]
    assert member["probabilities"] == pytest.approx(expected.tolist())
    assert [row["weight"] for row in member["patches"]] == pytest.approx(
        output["attention"][0].tolist()
    )
    assert member["attentionNote"] == result["attentionNote"] == attention_note(name)
    if catalog.spec(name).attention_note:
        assert catalog.spec(name).attention_note in result["attentionNote"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("precision", ["16-mixed", "bf16-mixed"])
@pytest.mark.parametrize("name", NEW_MODELS)
def test_cuda_fit_with_mixed_precision_and_checkpointing(tmp_path, precision, name):
    if precision == "bf16-mixed" and not torch.cuda.is_bf16_supported():
        pytest.skip("GPU does not support bfloat16")
    plan = plan_for(
        tmp_path,
        name,
        batchSize=3,
        accumulateGradBatches=2,
        precision=precision,
        maxEpochs=1,
        gradientCheckpointing=True,
        dropout=0.1,
    )
    plan["device"] = "cuda"
    result = train_fold(plan, tmp_path / "fit")
    assert result["epochsCompleted"] == 1
    assert np.isfinite(result["metrics"]["validation"]["selected"]["loss"])
    state = torch.load(result["lastCheckpointPath"], map_location="cpu", weights_only=True)
    assert all(bool(torch.isfinite(tensor).all()) for tensor in state["state_dict"].values())
