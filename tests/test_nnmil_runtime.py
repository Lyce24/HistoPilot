"""Synthetic nnMIL fits, frozen inference and window attention agree."""

import copy
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

from histopilot.application.predictors import checkpoint_snapshot  # noqa: E402
from histopilot.storage.packed import _stamp  # noqa: E402
from histopilot.training.attention import interpret  # noqa: E402
from histopilot.training.fold import train_fold  # noqa: E402
from histopilot.training.inference import evaluate  # noqa: E402
from histopilot.training.module import MILTrainModule, class_logits  # noqa: E402
from histopilot.training.refit import train_refit  # noqa: E402

support = runpy.run_path(str(Path(__file__).with_name("test_mil_training.py")))


def inference_plan(training, checkpoint):
    data = copy.deepcopy(training["data"])
    data["memberships"] = [row for row in data["memberships"] if row["partition"] == "test"]
    ids = {row["slideId"] for row in data["memberships"]}
    data["featureFiles"] = {key: row for key, row in data["featureFiles"].items() if key in ids}
    for entry in data["featureFiles"].values():
        entry.update(_stamp(Path(entry["path"]).stat()))
    data["sourceStamps"] = {row["path"]: row for row in data["featureFiles"].values()}
    return {
        "kind": "evaluation",
        "runId": "nnmil-evaluation",
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
    "loss,aggregation",
    [
        ("ce", "mean_logits"),
        ("ce", "mean_probabilities"),
        ("bce", "mean_logits"),
        ("bce", "mean_probabilities"),
    ],
)
def test_nnmil_fits_reloads_and_evaluates_identical_fold_and_external_predictions(
    tmp_path, monkeypatch, loss, aggregation
):
    plan = support["tiny_plan"](tmp_path)
    plan["recipe"].update(
        model="nnmil",
        attentionDim=2,
        lossType=loss,
        nnmilWindowAggregation=aggregation,
        lrScheduler="cosine",
        lrScheduleInterval="step",
        warmupEpochs=1,
        weightDecayPolicy="weights_only",
        finalLrFraction=0,
        maxEpochs=3,
    )
    result = train_fold(plan, tmp_path / "fit")
    assert result["epochsCompleted"] == 3
    selected = MILTrainModule.load_from_checkpoint(result["bestCheckpointPath"], weights_only=True)
    selected.eval()
    checkpoint = checkpoint_snapshot(result["bestCheckpointPath"], tmp_path / "fit")
    external = inference_plan(plan, checkpoint)
    output_dir = tmp_path / "external"
    first = evaluate(external, output_dir)
    predictions = json.loads((output_dir / "predictions.json").read_text())["records"]
    assessment = json.loads(Path(result["predictions"]["assessment"]).read_text())["records"]
    by_slide = {row["slideId"]: row for row in assessment}
    for row in predictions:
        assert row["probabilities"] == pytest.approx(by_slide[row["slideId"]]["probabilities"])
        scores = row["windowUncertaintyByMember"][0]
        assert scores["windowCount"] == 3
        assert scores["meanWindowEntropy"] == pytest.approx(
            by_slide[row["slideId"]]["windowUncertainty"]["meanWindowEntropy"]
        )
        assert len(scores["probabilityVariance"]) == 2
    cache = json.loads((output_dir / "members/member-0.json").read_text())
    assert len(cache["windowUncertainty"]) == len(predictions)
    monkeypatch.setattr(
        MILTrainModule, "load_from_checkpoint", lambda *a, **k: pytest.fail("Use frozen cache")
    )
    assert evaluate(external, output_dir)["artifacts"] == first["artifacts"]


def test_nnmil_attention_uses_the_same_window_prediction_and_resumes_uncertainty(
    tmp_path, monkeypatch
):
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("PIL.Image")
    attention_support = runpy.run_path(str(Path(__file__).with_name("test_attention_execution.py")))
    plan = attention_support["attention_plan"](tmp_path)
    path = Path(plan["checkpoints"][0]["path"])
    payload = torch.load(path, weights_only=True)
    recipe = {
        **payload["hyper_parameters"]["recipe"],
        "model": "nnmil",
        "attentionDim": 2,
        "lossType": "bce",
        "nnmilWindowAggregation": "mean_probabilities",
    }
    model = MILTrainModule(4, plan["target"], recipe)
    payload["hyper_parameters"] = dict(model.hparams)
    payload["state_dict"] = model.state_dict()
    path = tmp_path / "nnmil.ckpt"
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
    assert member["windowUncertainty"]["windowCount"] == 3
    assert "feature windows" in member["attentionNote"]
    monkeypatch.setattr(
        MILTrainModule, "load_from_checkpoint", lambda *a, **k: pytest.fail("Use frozen cache")
    )
    assert interpret(plan, tmp_path / "attention") == result


def test_weights_only_decay_preserves_separate_learning_rates():
    module = MILTrainModule(
        7,
        support["target"](),
        {
            "model": "nnmil",
            "attentionDim": 3,
            "optimizer": "adamw",
            "learningRate": 1e-4,
            "weightDecay": 5e-3,
            "weightDecayPolicy": "weights_only",
            "headLearningRate": 2e-4,
            "aggregatorLearningRate": 5e-5,
        },
    )
    optimizer = module.configure_optimizers()
    head_ids = {id(parameter) for parameter in module.model.classifier.parameters()}
    observed = []
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            assert group["weight_decay"] == (5e-3 if parameter.ndim > 1 else 0)
            assert group["lr"] == (2e-4 if id(parameter) in head_ids else 5e-5)
            observed.append(id(parameter))
    assert len(observed) == len(set(observed)) == len(list(module.parameters()))


def test_step_cosine_warms_up_from_first_update_and_reaches_floor():
    module = MILTrainModule(
        4,
        support["target"](),
        {
            "model": "nnmil",
            "attentionDim": 2,
            "optimizer": "adamw",
            "learningRate": 1e-4,
            "weightDecay": 0,
            "lrScheduler": "cosine",
            "lrScheduleInterval": "step",
            "maxEpochs": 3,
            "warmupEpochs": 1,
            "finalLrFraction": 0,
        },
    )
    module._trainer = SimpleNamespace(estimated_stepping_batches=12, max_epochs=3)
    configured = module.configure_optimizers()
    optimizer, scheduler = configured["optimizer"], configured["lr_scheduler"]["scheduler"]
    assert configured["lr_scheduler"]["interval"] == "step"
    rates = []
    for _ in range(12):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert rates[:4] == pytest.approx([2.5e-5, 5e-5, 7.5e-5, 1e-4])
    assert rates[-1] == 0
    assert all(a >= b for a, b in zip(rates[4:], rates[5:]))
    # An explicit shorter fitting budget keeps the predeclared schedule horizon.
    module._trainer = SimpleNamespace(estimated_stepping_batches=8, max_epochs=2)
    short = module.configure_optimizers()
    short_scheduler = short["lr_scheduler"]["scheduler"]
    assert short_scheduler.lr_lambdas[0](7) > 0


def test_legacy_optimizer_defaults_keep_decay_on_biases():
    module = MILTrainModule(
        4,
        support["target"](),
        {
            "optimizer": "adamw",
            "learningRate": 1e-4,
            "weightDecay": 5e-3,
        },
    )
    optimizer = module.configure_optimizers()
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["weight_decay"] == 5e-3


@pytest.mark.parametrize("sampler", ["class_balanced", "auc_stratified"])
def test_nnmil_refit_resolves_final_pool_then_reloads_and_predicts(tmp_path, monkeypatch, sampler):
    original = support["tiny_plan"](tmp_path)
    for row in original["data"]["memberships"]:
        if row["partition"] != "train":
            entry = original["data"]["featureFiles"][row["slideId"]]
            features = np.tile(np.load(entry["path"]), (4, 1))
            np.save(entry["path"], features)
            entry["patchCount"] = len(features)
    plan = copy.deepcopy(original)
    plan["kind"] = "refit"
    plan["epochBudget"] = {"epochs": 1, "percentile": 50}
    plan["recipe"].update(
        model="nnmil",
        attentionDim=2,
        bagSizeMode="training_median",
        bagSizeFraction=0.5,
        nnmilBatchSampler=sampler,
        maxEpochs=1,
        minEpochs=1,
        earlyStopping=False,
    )
    for row in plan["data"]["memberships"]:
        row.update(partition="train", phase="refit")
    monkeypatch.setattr(
        MILTrainModule, "validation_step", lambda *a, **k: pytest.fail("Refit cannot validate")
    )
    result = train_refit(plan, tmp_path / "refit")
    assert result["epochsCompleted"] == 1
    assert not result["validationUsed"] and not result["testDataUsed"]
    assert result["nnmilPlanning"]["trainingSlideCount"] == 12
    assert result["nnmilPlanning"]["medianPatchCount"] == 20
    assert result["effectiveRecipe"]["bagSize"] == 10
    model = MILTrainModule.load_from_checkpoint(
        result["bestCheckpointPath"], weights_only=True
    ).eval()
    assert model.recipe["model"] == "nnmil" and model.recipe["bagSize"] == 10
    checkpoint = checkpoint_snapshot(result["bestCheckpointPath"], tmp_path / "refit")
    external = inference_plan(original, checkpoint)
    # Separate synthetic external identities; use exact source tensors to
    # compare published prediction with the checkpoint's whole-bag forward.
    for row in external["data"]["memberships"]:
        previous = row["slideId"]
        row.update(slideId=f"external-{previous}", patientId=f"external-{row['patientId']}")
        entry = external["data"]["featureFiles"].pop(previous)
        external["data"]["featureFiles"][row["slideId"]] = {**entry, "slideId": row["slideId"]}
    evaluated = evaluate(external, tmp_path / "evaluation")
    assert evaluated["slideCount"] == 4
    records = json.loads((tmp_path / "evaluation/predictions.json").read_text())["records"]
    with torch.inference_mode():
        for row in records:
            entry = external["data"]["featureFiles"][row["slideId"]]
            features = torch.tensor(np.load(entry["path"])).unsqueeze(0)
            assert len(features[0]) > model.recipe["bagSize"]
            expected = class_logits(model(features), model.target).softmax(-1)[0].tolist()
            assert row["probabilities"] == pytest.approx(expected)


@pytest.mark.parametrize("microbatch_sizes", [[2, 1, 2, 1], [2, 1, 2]])
@pytest.mark.parametrize("model_name", ["nnmil", "abmil", "mean_pool", "max_pool"])
def test_accumulated_mil_updates_equal_true_weighted_batch_means(microbatch_sizes, model_name):
    import lightning as L
    from torch.nn import functional as F
    from torch.utils.data import DataLoader

    torch.set_num_threads(1)
    torch.manual_seed(26)
    recipe = {
        "model": model_name,
        "embedDim": 4,
        "attentionDim": 2,
        "nnmilFeatureSampling": False,
        "dropout": 0,
        "optimizer": "sgd",
        "learningRate": 0.03,
        "weightDecay": 0,
        "batchSize": 2,
    }
    module = MILTrainModule(4, support["target"](), recipe)
    reference = copy.deepcopy(module.model)
    count = sum(microbatch_sizes)
    features = torch.randn(count, 3, 4)
    labels = torch.arange(count) % 2
    weights = torch.tensor([0.5, 1.5, 2, 0.3, 1.7, 0.9])[:count]
    batches, position = [], 0
    for size in microbatch_sizes:
        batches.append(
            {
                "features": features[position : position + size],
                "mask": torch.ones(size, 3, dtype=torch.bool),
                "labels": labels[position : position + size],
                "lossWeights": weights[position : position + size],
            }
        )
        position += size
    optimizer = torch.optim.SGD(reference.parameters(), lr=recipe["learningRate"])
    for index in range(0, len(batches), 2):
        full = {
            key: torch.cat([row[key] for row in batches[index : index + 2]]) for key in batches[0]
        }
        optimizer.zero_grad()
        losses = F.cross_entropy(
            reference(full["features"], full["mask"]), full["labels"], reduction="none"
        )
        (losses * full["lossWeights"]).mean().backward()
        optimizer.step()
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        accumulate_grad_batches=2,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        limit_val_batches=0,
    )
    trainer.fit(module, train_dataloaders=DataLoader(batches, batch_size=None))
    assert module._optimizer_presentation_count == 0
    assert trainer.global_step == (len(batches) + 1) // 2
    for actual, expected in zip(module.model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("precision", ["16-mixed", "bf16-mixed"])
@pytest.mark.parametrize("model_name", ["nnmil", "abmil", "mean_pool", "max_pool"])
def test_cuda_mil_fit_handles_amp_and_uneven_accumulated_batches(tmp_path, precision, model_name):
    if precision == "bf16-mixed" and not torch.cuda.is_bf16_supported():
        pytest.skip("GPU does not support bfloat16")
    plan = support["tiny_plan"](tmp_path)
    plan["device"] = "cuda"
    plan["recipe"].update(
        model=model_name,
        embedDim=8,
        attentionDim=2,
        batchSize=3,
        accumulateGradBatches=2,
        precision=precision,
        maxEpochs=1,
        gradientCheckpointing=True,
        dropout=0.1,
    )
    result = train_fold(plan, tmp_path / "fit")
    assert result["epochsCompleted"] == 1
    assert np.isfinite(result["metrics"]["validation"]["selected"]["loss"])
    state = torch.load(result["lastCheckpointPath"], map_location="cpu", weights_only=True)
    assert state["global_step"] == 1
    assert all(bool(torch.isfinite(tensor).all()) for tensor in state["state_dict"].values())
