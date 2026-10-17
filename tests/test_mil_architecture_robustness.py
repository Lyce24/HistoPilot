"""Stress and integration checks of CLAM, TransMIL, DSMIL, DTFD-MIL and ACMIL.

Beyond parity with the authors' code: tiny and oddly padded slides, extreme feature
scales and settings, identical patches, every training loss, clinical plus image inputs,
refits, pooled attention across architectures, and repeated fits that must be identical.
"""

import copy
import json
import runpy
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

from histopilot.application.predictors import checkpoint_snapshot  # noqa: E402
from histopilot.models import catalog, registry  # noqa: E402
from histopilot.schemas.development import TrainingRecipe  # noqa: E402
from histopilot.training.fold import train_fold  # noqa: E402
from histopilot.training.inference import evaluate  # noqa: E402
from histopilot.training.module import MILTrainModule, _BinaryLoss  # noqa: E402
from histopilot.training.refit import train_refit  # noqa: E402

HERE = Path(__file__).parent
fits = runpy.run_path(str(HERE / "test_mil_architecture_training.py"))
clinical = runpy.run_path(str(HERE / "test_clinical_model_execution.py"))
NEW_MODELS = ("clam_sb", "clam_mb", "transmil", "dsmil", "dtfd", "acmil")
SMALL = {**fits["SMALL"], "transmilHeads": 2, "transmilLandmarks": 4, "dropout": 0.25}


def build(name, classes=2, **options):
    torch.manual_seed(5)
    recipe = TrainingRecipe.model_validate({"model": name, **SMALL, **options}).model_dump()
    return registry.build(name, 6, classes, recipe)


def logit_counts(name):
    return (2, 4) if catalog.spec(name).requires_class_logits else (1, 2, 4)


def criterion(classes):
    return _BinaryLoss(1, None) if classes == 1 else torch.nn.CrossEntropyLoss(reduction="none")


def fit_loss(network, features, mask, labels, classes):
    loss_fn = criterion(classes)
    if hasattr(network, "training_output"):  # architectures with their own loss terms
        output = network.training_output(features, mask, labels, loss_fn)
        return output["bagLossWeight"] * loss_fn(output["logits"], labels) + output["auxiliaryLoss"]
    return loss_fn(network(features, mask), labels)


def assert_finite_gradients(network):
    for name, parameter in network.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize("name", NEW_MODELS)
def test_tiny_slides_fit_and_predict(name):
    for classes in logit_counts(name):
        for count in (1, 2, 3, 7, 8, 9, 15, 16, 17):
            network = build(name, classes)
            features = torch.randn(1, count, 6)
            network.eval()
            with torch.inference_mode():
                output = network(features, return_attention=True)
            assert torch.isfinite(output["logits"]).all() and output["logits"].shape == (1, classes)
            torch.testing.assert_close(output["attention"].sum(), torch.tensor(1.0))
            network.train()
            labels = torch.tensor([min(1, classes - 1) if classes > 1 else 1])
            loss = fit_loss(network, features, None, labels, classes)
            assert torch.isfinite(loss).all(), (name, classes, count)
            loss.sum().backward()
            assert_finite_gradients(network)


@pytest.mark.parametrize("name", NEW_MODELS)
def test_any_patch_mask_gives_each_slide_its_own_prediction(name):
    network = build(name, 3).eval()
    generator = torch.Generator().manual_seed(9)
    features = torch.randn(4, 30, 6, generator=generator)
    mask = torch.rand(4, 30, generator=generator) > 0.45
    mask[:, 0] = True  # every slide keeps a patch
    features[~mask] = float("nan")
    with torch.inference_mode():
        batch = network(features, mask, return_attention=True)
        for index in range(4):
            alone = network(features[index : index + 1, mask[index]], return_attention=True)
            torch.testing.assert_close(alone["logits"][0], batch["logits"][index])
            torch.testing.assert_close(
                alone["attention"][0], batch["attention"][index, mask[index]]
            )
    assert torch.count_nonzero(batch["attention"][~mask]) == 0


@pytest.mark.parametrize("scale", [1e-4, 1e3])
@pytest.mark.parametrize("name", NEW_MODELS)
def test_extreme_feature_scales_stay_finite(name, scale):
    network = build(name, 2)
    features = torch.randn(2, 25, 6) * scale
    labels = torch.tensor([0, 1])
    loss = fit_loss(network, features, None, labels, 2)
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    assert_finite_gradients(network)
    with torch.inference_mode():
        output = network.eval()(features, return_attention=True)
    assert torch.isfinite(output["logits"]).all() and torch.isfinite(output["attention"]).all()


@pytest.mark.parametrize("name", NEW_MODELS)
def test_identical_patches_spread_attention_evenly(name):
    network = build(name, 2).eval()
    features = torch.randn(1, 1, 6).expand(1, 12, 6).contiguous()
    with torch.inference_mode():
        output = network(features, return_attention=True)
    assert torch.isfinite(output["logits"]).all()
    torch.testing.assert_close(output["attention"].sum(), torch.tensor(1.0))
    if name != "transmil":
        # Ties everywhere (CLAM's top-k, DSMIL's critical patch, DTFD-MIL's four equal
        # pseudo-bags): every patch carries one twelfth. TransMIL's grid copies and the
        # position encoding's borders make otherwise identical tokens differ.
        torch.testing.assert_close(output["attention"][0], torch.full((12,), 1 / 12))


EXTREMES = {
    "clam_sb": [
        {"clamAttentionDim": 1, "clamInstanceSamples": 4096, "clamBagWeight": 0.0},
        {
            "clamSubtyping": True,
            "clamInstanceLoss": "ce",
            "dropout": 0.9,
            "clamGatedAttention": False,
        },
        {"clamRecipeAttentionDropout": True, "dropout": 0.9, "numFcLayers": 8},
    ],
    "clam_mb": [{"clamBagWeight": 1.0, "clamGatedAttention": False}, {"clamSubtyping": True}],
    "transmil": [
        {"transmilLayers": 12, "transmilLandmarks": 1, "transmilPinvIterations": 1},
        {
            "embedDim": 64,
            "transmilHeads": 64,
            "transmilPinvIterations": 32,
            "transmilDropout": 0.9,
            "transmilPositionEncoding": False,
            "transmilResidualConv": False,
        },
        {"transmilLayers": 1, "transmilLandmarks": 4096},
    ],
    "dsmil": [
        {"dsmilQueryDim": 1, "dsmilMaxLossWeight": 0.0, "dsmilAveragePrediction": True},
        {
            "dsmilProjection": True,
            "numFcLayers": 8,
            "dsmilPassingValue": True,
            "dsmilValueDropout": 0.9,
            "dsmilMaxLossWeight": 1.0,
            "dsmilNonlinearQuery": False,
        },
    ],
    "dtfd": [
        {"dtfdPseudoBags": 256, "dtfdDistill": "maxmins", "dtfdDistillPatches": 1024},
        {"dtfdTestSplits": 64, "dtfdResidualBlocks": 8, "dtfdDropout": 0.9, "dtfdAttentionDim": 1},
        {"dtfdDistill": "maxs", "dtfdPseudoBags": 1},
    ],
    "acmil": [
        {"acmilBranches": 64, "acmilMaskedPatches": 100000, "acmilMaskDrop": 0.99},
        {"acmilAttentionDim": 1, "acmilDropout": 0.9, "acmilBranches": 1, "acmilMaskedPatches": 0},
    ],
}


@pytest.mark.parametrize(
    "name,options",
    [(name, options) for name, cases in EXTREMES.items() for options in cases],
    ids=lambda value: value if isinstance(value, str) else "-".join(map(str, value.values())),
)
def test_settings_at_the_limits_of_their_ranges_fit_and_predict(name, options):
    for classes in logit_counts(name)[-2:]:
        network = build(name, classes, **options)
        features = torch.randn(3, 20, 6)
        mask = torch.ones(3, 20, dtype=torch.bool)
        mask[2, 5:] = False
        labels = torch.tensor([0, classes - 1, min(1, classes - 1)])
        loss = fit_loss(
            network, features, mask, labels if classes > 1 else torch.tensor([0, 1, 1]), classes
        )
        assert torch.isfinite(loss).all()
        loss.sum().backward()
        assert_finite_gradients(network)
        with torch.inference_mode():
            output = network.eval()(features, mask, return_attention=True)
        assert torch.isfinite(output["logits"]).all()
        torch.testing.assert_close(output["attention"].sum(1), torch.ones(3))


@pytest.mark.parametrize("loss", ["ce", "focal", "bce"])
@pytest.mark.parametrize("name", NEW_MODELS)
def test_every_training_loss_runs_through_the_module(tmp_path, name, loss):
    if loss == "bce" and catalog.spec(name).requires_class_logits:
        pytest.skip("CLAM-MB needs one logit per class")
    plan = fits["plan_for"](tmp_path, name)
    recipe = {
        **TrainingRecipe.model_validate(plan["recipe"]).model_dump(),
        "lossType": loss,
        **({"labelSmoothing": 0.1} if loss == "ce" else {}),
        **({"classWeights": [0.7, 1.6]} if loss != "bce" else {}),
    }
    module = MILTrainModule(4, plan["target"], TrainingRecipe.model_validate(recipe).model_dump())
    module.log = lambda *args, **kwargs: None
    batch = {
        "features": torch.randn(3, 9, 4),
        "mask": torch.tensor([[True] * 9, [True] * 4 + [False] * 5, [True] * 2 + [False] * 7]),
        "labels": torch.tensor([0, 1, 1]),
        "lossWeights": torch.tensor([1.0, 0.5, 1.5]),
    }
    value = module.training_step(batch, 0)
    assert torch.isfinite(value)
    value.backward()
    assert_finite_gradients(module)


@pytest.mark.parametrize("name", NEW_MODELS)
def test_clinical_and_image_inputs_fit_predict_externally_and_refit(tmp_path, name):
    plan = clinical["clinical_plan"](tmp_path, "multimodal", name)
    plan["recipe"].update(fits["SMALL"])
    fitted = train_fold(plan, tmp_path / "fit")
    loaded = MILTrainModule.load_from_checkpoint(
        fitted["bestCheckpointPath"], weights_only=True
    ).eval()
    values = [{"age": 40, "site": "fitting"}, {"age": None, "site": None}]
    with torch.inference_mode():
        image_only = loaded.model(torch.randn(2, 7, 4))
        combined = loaded.prediction_output(torch.randn(2, 7, 4), clinical=values)["logits"]
        attention = loaded.prediction_output(
            torch.randn(1, 7, 4), return_attention=True, clinical=values[:1]
        )
    assert torch.isfinite(image_only).all() and torch.isfinite(combined).all()
    torch.testing.assert_close(attention["attention"].sum(), torch.tensor(1.0))
    external = copy.deepcopy(plan)
    external["data"]["memberships"] = [
        r for r in plan["data"]["memberships"] if r["partition"] == "test"
    ]
    ids = {row["slideId"] for row in external["data"]["memberships"]}
    evaluation = fits["inference_plan"](
        external, checkpoint_snapshot(fitted["bestCheckpointPath"], tmp_path / "fit")
    )
    evaluation["data"]["featureFiles"] = {
        k: v for k, v in evaluation["data"]["featureFiles"].items() if k in ids
    }
    evaluation["data"].update(inputMode="multimodal", clinicalFields=clinical["FIELDS"])
    result = evaluate(evaluation, tmp_path / "external")
    assert result["slideCount"] == len(ids)
    refit = copy.deepcopy(plan)
    refit["epochBudget"] = {"epochs": 1}
    refit["recipe"].update(maxEpochs=1, minEpochs=1, earlyStopping=False)
    for row in refit["data"]["memberships"]:
        row.update(partition="train", phase="refit", pool="development")
    refitted = train_refit(refit, tmp_path / "refit")
    assert refitted["epochsCompleted"] == 1 and not refitted["validationUsed"]
    again = MILTrainModule.load_from_checkpoint(refitted["bestCheckpointPath"], weights_only=True)
    assert type(again.model) is type(loaded.model)


@pytest.mark.parametrize("name", NEW_MODELS)
def test_repeat_fits_are_bit_identical(tmp_path, name):
    plan = fits["plan_for"](tmp_path, name, dropout=0.25, maxEpochs=2)
    first = train_fold(copy.deepcopy(plan), tmp_path / "first")
    second = train_fold(copy.deepcopy(plan), tmp_path / "second")
    a = torch.load(first["lastCheckpointPath"], map_location="cpu", weights_only=True)
    b = torch.load(second["lastCheckpointPath"], map_location="cpu", weights_only=True)
    assert a["metricsHistory"] == b["metricsHistory"]
    for key, value in a["state_dict"].items():
        assert torch.equal(value, b["state_dict"][key]), key
    one = json.loads((tmp_path / "first/assessment-predictions.json").read_text())["records"]
    two = json.loads((tmp_path / "second/assessment-predictions.json").read_text())["records"]
    assert one == two


def test_pooled_attention_combines_maps_of_different_architectures(tmp_path):
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("PIL.Image")
    from histopilot.training.attention import interpret

    attention = runpy.run_path(str(HERE / "test_attention_execution.py"))
    plan = attention["attention_plan"](tmp_path)
    payload = torch.load(Path(plan["checkpoints"][0]["path"]), weights_only=True)
    models, maps = [], []
    for index, name in enumerate(("abmil", "transmil", "dtfd", "clam_mb")):
        recipe = {**payload["hyper_parameters"]["recipe"], "model": name, **fits["SMALL"]}
        module = MILTrainModule(4, plan["target"], recipe)
        saved = {
            **payload,
            "hyper_parameters": dict(module.hparams),
            "state_dict": module.state_dict(),
        }
        path = tmp_path / f"{name}.ckpt"
        torch.save(saved, path)
        checkpoint = checkpoint_snapshot(path, tmp_path)
        models.append(
            {
                "predictorId": f"p-{index}",
                "name": name,
                "method": "refit",
                "aggregation": "mean_probability",
                "checkpoints": [checkpoint],
            }
        )
        with h5py.File(plan["slides"][0]["featurePath"]) as handle, torch.inference_mode():
            features = torch.tensor(handle["features"][:]).unsqueeze(0)
            maps.append(
                module.eval()
                .prediction_output(features, return_attention=True)["attention"][0]
                .double()
                .numpy()
            )
    pooled = {
        **plan,
        "method": "pooled",
        "pooling": "median",
        "models": models,
        "checkpoints": [c for model in models for c in model["checkpoints"]],
    }
    result = interpret(pooled, tmp_path / "pooled")
    summary = result["slides"][0]
    assert [member["name"] for member in summary["members"]] == [
        "abmil",
        "transmil",
        "dtfd",
        "clam_mb",
    ]
    written = json.loads((tmp_path / "pooled" / summary["attentionArtifact"]).read_text())
    weights = np.asarray([patch["weight"] for patch in written["patches"]])
    # NumPy's median averages the two middle maps of four, as the pooling does.
    expected = np.median(np.stack([m / m.sum() for m in maps]), axis=0)
    np.testing.assert_allclose(weights, expected / expected.sum(), rtol=1e-6, atol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("name", NEW_MODELS)
def test_cuda_repeat_fits_are_bit_identical(tmp_path, name):
    plan = fits["plan_for"](tmp_path, name, dropout=0.25, maxEpochs=2, batchSize=3)
    plan["device"] = "cuda"
    first = train_fold(copy.deepcopy(plan), tmp_path / "first")
    second = train_fold(copy.deepcopy(plan), tmp_path / "second")
    a = torch.load(first["lastCheckpointPath"], map_location="cpu", weights_only=True)["state_dict"]
    b = torch.load(second["lastCheckpointPath"], map_location="cpu", weights_only=True)[
        "state_dict"
    ]
    for key, value in a.items():
        assert torch.equal(value, b[key]), key


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("name", NEW_MODELS)
def test_cuda_whole_slide_inference_on_twenty_thousand_patches(name):
    torch.manual_seed(0)
    recipe = TrainingRecipe.model_validate({"model": name}).model_dump()
    network = registry.build(name, 1024, 2, recipe).cuda().eval()
    features = torch.randn(1, 20000, 1024, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        output = network(features, return_attention=True)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 2**30
    assert torch.isfinite(output["logits"]).all()
    torch.testing.assert_close(output["attention"].sum().float().cpu(), torch.tensor(1.0))
    assert peak < 4, f"{name} used {peak:.2f} GiB"
    del network, features, output
    torch.cuda.empty_cache()
