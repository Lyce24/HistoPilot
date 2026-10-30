"""Pretrained weights a recipe starts from: pinned by SHA-256, verified, then loaded.

Review hashes the checkpoint inside the data roots; the frozen batch keeps the pin;
launching, every fit and every resume refuse a file that changed or went missing; and a
fresh fit loads it strictly, while saved models never need it again.
"""

import copy
import hashlib
import json
import runpy
from pathlib import Path

import pytest
from support.training import development_batch, training_service

from histopilot.schemas.development import DevelopmentBatchSpec, TrainingRecipe
from histopilot.storage import model_weights
from histopilot.storage.filesystem import LocalFilesystem
from histopilot.storage.project_lock import StorageError

MODEL = "oceanpath_mamba2"
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


@pytest.fixture
def weights(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    path = root / "aggregator.pt"
    path.write_bytes(b"pretrained v1")
    return LocalFilesystem((root,)), path


def test_a_checkpoint_is_pinned_only_as_a_regular_file_inside_the_data_roots(tmp_path, weights):
    filesystem, path = weights
    assert model_weights.pin(filesystem, str(path)) == {
        "path": str(path),
        "bytes": 13,
        "sha256": hashlib.sha256(b"pretrained v1").hexdigest(),
    }
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"x")
    link = path.parent / "link.pt"
    link.symlink_to(path)
    for candidate in (outside, link, path.parent / "missing.pt", path.parent):
        with pytest.raises(StorageError) as refused:
            model_weights.pin(filesystem, str(candidate))
        assert refused.value.code == "MODEL_CHECKPOINT_INVALID"


def test_a_changed_or_missing_checkpoint_is_refused_after_pinning(weights):
    filesystem, path = weights
    pins = model_weights.pin_recipes(filesystem, [checkpointed(path), recipe()])
    model_weights.verify(pins, filesystem)
    assert model_weights.read_pinned(pins, str(path))[1] == b"pretrained v1"
    path.write_bytes(b"pretrained v2")
    for check in (
        lambda: model_weights.verify(pins),
        lambda: model_weights.verify(pins, filesystem),
        lambda: model_weights.read_pinned(pins, str(path)),
    ):
        with pytest.raises(StorageError) as refused:
            check()
        assert refused.value.code == "MODEL_CHECKPOINT_CHANGED"
    path.unlink()
    with pytest.raises(StorageError, match="missing") as refused:
        model_weights.verify(pins)
    assert refused.value.code == "MODEL_CHECKPOINT_CHANGED"
    with pytest.raises(StorageError, match="was not pinned"):
        model_weights.read_pinned([], str(path))


def test_review_pins_the_checkpoint_freeze_keeps_it_and_launch_refuses_a_changed_file(
    tmp_path, monkeypatch, task_center
):
    development, spec, _ = development_batch(tmp_path)
    path = tmp_path / "weights" / "aggregator_weights_lejepa.pt"
    path.parent.mkdir()
    path.write_bytes(b"pretrained v1")
    values = spec.model_dump()
    common = {**values["recipe"], "learningRate": 1e-4, "maxEpochs": 2, "bagSize": 2}
    values.update(
        mode="explicit",
        trainingSeeds=[11],
        candidateSelection="all",
        comparison={"reference": 1},
        configurations=[
            {**common, "model": MODEL},
            {**common, **checkpointed(path), "learningRate": 1e-4, "maxEpochs": 2, "bagSize": 2},
        ],
    )
    batch = DevelopmentBatchSpec.model_validate(values)
    preview = development.preview(batch)
    assert preview["canFreeze"], preview["findings"]
    pin = {"path": str(path), "bytes": 13, "sha256": hashlib.sha256(b"pretrained v1").hexdigest()}
    assert preview["modelCheckpoints"] == [pin]
    # A file edited between review and freeze changes the reviewed design.
    path.write_bytes(b"pretrained v2")
    with pytest.raises(StorageError) as refused:
        development.freeze(batch, preview["previewHash"], "pinned-batch", {"tag": "Pinned"})
    assert refused.value.code == "STALE_PREVIEW"
    path.write_bytes(b"pretrained v1")
    frozen = development.freeze(batch, preview["previewHash"], "pinned-batch", {"tag": "Pinned"})
    assert frozen["manifest"]["modelCheckpoints"] == [pin]
    training = training_service(development.store, development.filesystem, task_center, monkeypatch)
    plan, _guard = training._prepare(frozen)
    assert plan["data"]["modelCheckpoints"] == [pin]
    path.write_bytes(b"pretrained v3")
    with pytest.raises(StorageError) as refused:
        training._prepare(frozen)
    assert refused.value.code == "MODEL_CHECKPOINT_CHANGED"
    path.unlink()
    with pytest.raises(StorageError) as refused:
        training._prepare(frozen)
    assert refused.value.code == "MODEL_CHECKPOINT_CHANGED"


def test_review_blocks_a_checkpoint_outside_the_data_roots(tmp_path):
    development, spec, _ = development_batch(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-elsewhere" / "aggregator.pt"
    outside.parent.mkdir()
    outside.write_bytes(b"weights")
    values = spec.model_dump()
    values.update(mode="single", trainingSeeds=[11])
    values["recipe"] = {**values["recipe"], **checkpointed(outside)}
    preview = development.preview(DevelopmentBatchSpec.model_validate(values))
    assert not preview["canFreeze"]
    assert [item["code"] for item in preview["findings"]] == ["MODEL_CHECKPOINT_INVALID"]
    assert preview["configurations"] == [] and "modelCheckpoints" not in preview


def build(torch, **options):
    from histopilot.models import registry

    torch.manual_seed(3)
    return registry.build(MODEL, 6, 3, {**SMALL, "mamba2Dropout": 0.0, **options}).eval()


def test_pretrained_weights_load_strictly_or_not_at_all(weights):
    torch = pytest.importorskip("torch")
    from histopilot.training.pretrained import load_pinned_weights

    filesystem, path = weights
    model = build(torch)
    state = {name: torch.randn_like(value) for name, value in model.aggregator.state_dict().items()}
    torch.save(state, path)
    pins = model_weights.pin_recipes(filesystem, [checkpointed(path, **SMALL)])
    pin = load_pinned_weights(model, checkpointed(path, **SMALL), pins)
    assert pin == pins[0]
    for name, value in model.aggregator.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
    wide = build(torch, mamba2EmbedDim=12)
    with pytest.raises(StorageError) as refused:
        load_pinned_weights(wide, checkpointed(path, **SMALL), pins)
    assert refused.value.code == "MODEL_CHECKPOINT_INCOMPATIBLE"
    torch.save({key: value for key, value in state.items() if "norm" not in key}, path)
    pins = model_weights.pin_recipes(filesystem, [checkpointed(path, **SMALL)])
    with pytest.raises(StorageError, match="Missing key"):
        load_pinned_weights(build(torch), checkpointed(path, **SMALL), pins)


def test_a_pinned_checkpoint_starts_the_fit_and_new_slides_predict_as_the_fold_scored(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("lightning")
    from histopilot.application.predictors import checkpoint_snapshot
    from histopilot.models import registry
    from histopilot.training.fold import train_fold
    from histopilot.training.inference import evaluate
    from histopilot.training.module import MILTrainModule

    tests = Path(__file__).parent
    models = runpy.run_path(str(tests / "test_oceanpath_mamba2.py"))
    architectures = runpy.run_path(str(tests / "test_mil_architecture_training.py"))

    pretrained = registry.build(MODEL, 4, 2, SMALL).aggregator.state_dict()
    path = tmp_path / "aggregator.pt"
    torch.save(pretrained, path)
    plan = models["h5_plan"](
        tmp_path,
        mamba2Initialization="checkpoint",
        mamba2Checkpoint=str(path),
        mamba2FreezeAggregator=True,
    )
    plan["data"]["modelCheckpoints"] = model_weights.pin_recipes(
        LocalFilesystem((tmp_path,)), [plan["recipe"]]
    )
    result = train_fold(copy.deepcopy(plan), tmp_path / "fit")
    assert result["state"] == "succeeded" and result["epochsCompleted"] == 2
    assert result["initialCheckpoint"] == plan["data"]["modelCheckpoints"][0]
    # Frozen, the aggregator still holds the pretrained weights; only the head trained.
    state = torch.load(result["bestCheckpointPath"], map_location="cpu", weights_only=True)
    for name, value in pretrained.items():
        torch.testing.assert_close(state["state_dict"][f"model.aggregator.{name}"], value)
    selected = MILTrainModule.load_from_checkpoint(result["bestCheckpointPath"], weights_only=True)
    assert type(selected.model).__module__ == "histopilot.models.mamba2"
    checkpoint = checkpoint_snapshot(result["bestCheckpointPath"], tmp_path / "fit")
    path.unlink()  # a saved model never needs the pretrained file again
    inference = architectures["inference_plan"](plan, checkpoint)
    del inference["data"]["modelCheckpoints"]  # predictor runs read features, not fit inputs
    evaluate(inference, tmp_path / "external")
    predictions = json.loads((tmp_path / "external/predictions.json").read_text())["records"]
    assessment = json.loads(Path(result["predictions"]["assessment"]).read_text())["records"]
    by_slide = {row["slideId"]: row for row in assessment}
    assert {row["slideId"] for row in predictions} == set(by_slide)
    for row in predictions:
        assert row["probabilities"] == pytest.approx(by_slide[row["slideId"]]["probabilities"])
    # A new fit from the same plan refuses the missing file instead of starting at random.
    with pytest.raises(StorageError) as refused:
        train_fold(copy.deepcopy(plan), tmp_path / "again")
    assert refused.value.code == "MODEL_CHECKPOINT_CHANGED"
