"""OceanPath Mamba-2 MIL as a HistoPilot model: batching, ordering and fitting.

Its numerical parity with OceanPath is tests/test_oceanpath_parity.py, its settings are
checked in tests/test_model_options.py, and starting from pinned pretrained weights in
tests/test_model_checkpoints.py.
"""

import json
import runpy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

from torch.nn import functional as F  # noqa: E402

from histopilot.models import registry  # noqa: E402
from histopilot.models.mamba2 import ssd_scan  # noqa: E402
from histopilot.schemas.development import TrainingRecipe  # noqa: E402
from histopilot.training.module import MILTrainModule  # noqa: E402

MODEL = "oceanpath_mamba2"
# Small widths for four-dimensional synthetic features.
SMALL = {
    "mamba2EmbedDim": 8,
    "mamba2Layers": 2,
    "mamba2StateDim": 4,
    "mamba2HeadDim": 4,
    "mamba2AttentionDim": 4,
}


def recipe(**values):
    return TrainingRecipe.model_validate({"model": MODEL, **values}).model_dump()


def checkpointed(path, **values):
    return recipe(mamba2Initialization="checkpoint", mamba2Checkpoint=str(path), **values)


# -- settings ------------------------------------------------------------------------------


# -- the model -----------------------------------------------------------------------------


def build(**options):
    torch.manual_seed(3)
    return registry.build(MODEL, 6, 3, {**SMALL, "mamba2Dropout": 0.0, **options}).eval()


def bags(count=3, longest=40):
    generator = torch.Generator().manual_seed(5)
    features = torch.randn(count, longest, 6, generator=generator)
    cells = torch.stack([torch.randperm(64, generator=generator)[:longest] for _ in range(count)])
    coords = torch.stack([cells % 8 * 256, cells // 8 * 256], dim=-1)
    lengths = [longest, longest - 13, 2][:count]
    mask = torch.arange(longest)[None] < torch.tensor(lengths)[:, None]
    return features, coords, mask, lengths


@pytest.mark.parametrize("hilbert", [True, False])
def test_a_slide_predicts_the_same_alone_or_in_a_padded_batch(hilbert):
    model = build(mamba2Hilbert=hilbert)
    features, coords, mask, lengths = bags()
    with torch.no_grad():
        batched = model(features, mask, coords=coords, return_attention=True)
        for slide, length in enumerate(lengths):
            alone = model(
                features[slide : slide + 1, :length],
                coords=coords[slide : slide + 1, :length],
                return_attention=True,
            )
            torch.testing.assert_close(batched["logits"][slide], alone["logits"][0])
            torch.testing.assert_close(batched["attention"][slide, :length], alone["attention"][0])
            assert not batched["attention"][slide, length:].any()
            assert float(batched["attention"][slide].sum()) == pytest.approx(1.0)


def test_padding_anywhere_in_the_bag_leaves_the_real_patches_unchanged():
    model = build(mamba2Hilbert=False)
    features, _, _, _ = bags(1, 12)
    holes = torch.tensor([[True, False, True, True, False, True] + [True] * 6])
    with torch.no_grad():
        gapped = model(features, holes, return_attention=True)
        compact = model(features[:, holes[0]], return_attention=True)
    torch.testing.assert_close(gapped["logits"], compact["logits"])
    torch.testing.assert_close(gapped["attention"][0, holes[0]], compact["attention"][0])


def test_hilbert_ordering_makes_the_prediction_independent_of_file_order():
    features, coords, _, _ = bags(1, 30)
    shuffle = torch.randperm(30, generator=torch.Generator().manual_seed(9))
    ordered, plain = build(), build(mamba2Hilbert=False)
    with torch.no_grad():
        original = ordered(features, coords=coords, return_attention=True)
        shuffled = ordered(features[:, shuffle], coords=coords[:, shuffle], return_attention=True)
        torch.testing.assert_close(original["logits"], shuffled["logits"])
        # Attention comes back in the order the patches were given.
        torch.testing.assert_close(original["attention"][0, shuffle], shuffled["attention"][0])
        # Without the curve the scan reads patches in file order, so order matters.
        assert not torch.allclose(plain(features), plain(features[:, shuffle]))


def test_fitting_needs_no_operation_that_deterministic_cuda_refuses(monkeypatch):
    """Folds train with torch.use_deterministic_algorithms(True), under which CUDA refuses
    torch.cumsum; the scan must not call it."""

    def refused(*_args, **_kwargs):
        raise RuntimeError("torch.cumsum has no deterministic CUDA kernel")

    monkeypatch.setattr(torch, "cumsum", refused)
    monkeypatch.setattr(torch.Tensor, "cumsum", refused)
    model = build().train()
    features, coords, mask, _ = bags()
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        logits = model(features, mask, coords=coords)
        F.cross_entropy(logits, torch.tensor([0, 1, 2])).backward()
    finally:
        torch.use_deterministic_algorithms(previous)
    assert all(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.parametrize("chunk", [1, 7, 64, 256])
def test_the_chunked_scan_equals_the_sequential_recurrence(chunk):
    generator = torch.Generator().manual_seed(1)
    batch, length, heads, width, groups, state = 2, 150, 4, 3, 2, 5
    x = torch.randn(batch, length, heads, width, generator=generator, dtype=torch.float64)
    dt = F.softplus(torch.randn(batch, length, heads, generator=generator, dtype=torch.float64) - 3)
    A = -torch.rand(heads, generator=generator, dtype=torch.float64) * 4
    B, C = (
        torch.randn(batch, length, groups, state, generator=generator, dtype=torch.float64)
        for _ in range(2)
    )
    hidden = torch.zeros(batch, heads, width, state, dtype=torch.float64)
    expected = []
    for step in range(length):
        decay = torch.exp(dt[:, step] * A)[..., None, None]
        inputs = B[:, step].repeat_interleave(heads // groups, dim=1)[:, :, None]
        hidden = decay * hidden + (dt[:, step][..., None, None] * x[:, step][..., None]) * inputs
        reads = C[:, step].repeat_interleave(heads // groups, dim=1)
        expected.append(torch.einsum("bhpn,bhn->bhp", hidden, reads))
    torch.testing.assert_close(ssd_scan(x, dt, A, B, C, chunk), torch.stack(expected, 1))


TARGET = {
    "task": "multiclass_classification",
    "unit": "slide",
    "classes": ["a", "b", "c"],
}


@pytest.mark.parametrize("frozen", [False, True])
def test_the_aggregator_has_its_own_rate_or_none_when_frozen(frozen):
    values = checkpointed("/data/weights.pt", **SMALL, mamba2FreezeAggregator=frozen)
    module = MILTrainModule(6, TARGET, {**values, "learningRate": 1e-4, "weightDecay": 0.01})
    groups = module.configure_optimizers().param_groups
    head = {id(parameter) for parameter in module.model.classifier.parameters()}
    if frozen:
        assert [group["lr"] for group in groups] == [1e-4]
        assert {id(parameter) for parameter in groups[0]["params"]} == head
    else:
        assert [group["lr"] for group in groups] == [1e-5, 1e-4]
        assert {id(parameter) for parameter in groups[1]["params"]} == head


# -- fitting -------------------------------------------------------------------------------

support = runpy.run_path(str(Path(__file__).with_name("test_mil_training.py")))
architectures = runpy.run_path(str(Path(__file__).with_name("test_mil_architecture_training.py")))


def h5_plan(tmp_path, **values):
    """tiny_plan's slides as H5 features with patch coordinates."""
    h5py = pytest.importorskip("h5py")
    import numpy as np

    plan = support["tiny_plan"](tmp_path)
    for index, entry in enumerate(plan["data"]["featureFiles"].values()):
        features = np.load(entry["path"])
        cells = np.random.default_rng(index).permutation(16)[: len(features)]
        path = Path(entry["path"]).with_suffix(".h5")
        with h5py.File(path, "w") as handle:
            handle.create_dataset("features", data=features)
            handle.create_dataset("coords", data=np.stack([cells % 4, cells // 4], 1) * 256)
        entry.update(path=str(path), format="h5")
    plan["recipe"].update(model=MODEL, **SMALL, **values)
    return plan


def test_a_random_aggregator_fits_and_new_slides_predict_as_its_fold_scored(tmp_path):
    from histopilot.application.predictors import checkpoint_snapshot
    from histopilot.training.fold import train_fold
    from histopilot.training.inference import evaluate

    plan = h5_plan(tmp_path)
    result = train_fold(plan, tmp_path / "fit")
    assert result["state"] == "succeeded" and result["epochsCompleted"] == 2
    assert "initialCheckpoint" not in result
    selected = MILTrainModule.load_from_checkpoint(result["bestCheckpointPath"], weights_only=True)
    assert type(selected.model).__module__ == "histopilot.models.mamba2"
    checkpoint = checkpoint_snapshot(result["bestCheckpointPath"], tmp_path / "fit")
    evaluate(architectures["inference_plan"](plan, checkpoint), tmp_path / "external")
    predictions = json.loads((tmp_path / "external/predictions.json").read_text())["records"]
    assessment = json.loads(Path(result["predictions"]["assessment"]).read_text())["records"]
    by_slide = {row["slideId"]: row for row in assessment}
    assert {row["slideId"] for row in predictions} == set(by_slide)
    for row in predictions:
        assert row["probabilities"] == pytest.approx(by_slide[row["slideId"]]["probabilities"])


def test_features_without_coordinates_are_refused_while_hilbert_ordering_is_on(tmp_path):
    from histopilot.datasets.mil import MILDataError
    from histopilot.training.fold import train_fold

    plan = support["tiny_plan"](tmp_path)
    plan["recipe"].update(model=MODEL, **SMALL)
    with pytest.raises(MILDataError, match="carry none"):
        train_fold(plan, tmp_path / "fit")
    plan["recipe"]["mamba2Hilbert"] = False
    assert train_fold(plan, tmp_path / "plain")["state"] == "succeeded"


def test_attention_maps_come_from_the_coordinate_ordered_forward_pass(tmp_path):
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("PIL.Image")
    from histopilot.application.predictors import checkpoint_snapshot
    from histopilot.training.attention import interpret
    from histopilot.training.module import class_logits

    attention_support = runpy.run_path(str(Path(__file__).with_name("test_attention_execution.py")))
    plan = attention_support["attention_plan"](tmp_path)
    path = Path(plan["checkpoints"][0]["path"])
    payload = torch.load(path, weights_only=True)
    model = MILTrainModule(
        4, plan["target"], {**payload["hyper_parameters"]["recipe"], "model": MODEL, **SMALL}
    )
    payload["hyper_parameters"] = dict(model.hparams)
    payload["state_dict"] = model.state_dict()
    path = tmp_path / "mamba2.ckpt"
    torch.save(payload, path)
    plan["checkpoints"] = [checkpoint_snapshot(path, tmp_path)]
    interpret(plan, tmp_path / "attention")
    member = json.loads((tmp_path / "attention/slide-0-member-0.json").read_text())
    model.eval()
    with h5py.File(plan["slides"][0]["featurePath"]) as handle, torch.inference_mode():
        features = torch.tensor(handle["features"][:]).unsqueeze(0)
        coords = torch.tensor(handle["coords"][:]).unsqueeze(0)
        output = model.prediction_output(features, return_attention=True, coords=coords)
    expected = class_logits(output["logits"], plan["target"]).softmax(-1)[0]
    assert member["probabilities"] == pytest.approx(expected.tolist())
    assert [row["weight"] for row in member["patches"]] == pytest.approx(
        output["attention"][0].tolist()
    )
