"""Development batch expansion preserves frozen splits and immutable run identity."""

import copy
import runpy
from pathlib import Path

import h5py
import pytest
from pydantic import ValidationError

from histopilot.application.development import DevelopmentService, development_plans, expand_recipes
from histopilot.application.feature_packs import FeaturePackService
from histopilot.application.protocols import ProtocolService
from histopilot.schemas.development import DevelopmentBatchSpec, TrainingRecipe
from histopilot.storage.filesystem import LocalFilesystem
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import ScientificStore

support = runpy.run_path(str(Path(__file__).with_name("test_evaluations.py")))


@pytest.fixture
def batch(tmp_path):
    folder = tmp_path / "project"
    folder.mkdir()
    store = ScientificStore(folder, "project-batches")
    rows = [
        {
            "slideId": f"s{i:02}",
            "patientId": f"p{i:02}",
            "attributes": {"label": str(i % 2), "cohort": "development"},
        }
        for i in range(30)
    ]
    dataset, rows = support["dataset"](store, rows=rows)
    bundle, _pack_id, source = support["bundle"](
        store, tmp_path, dataset, [row["slideId"] for row in rows], pack=True
    )
    protocol_spec = {
        "datasetId": dataset["id"],
        "target": support["TARGET"],
        "split": {
            "version": 4,
            "mode": "kfold",
            "folds": 5,
            "seeds": [42],
            "pools": {"trainSelection": "remaining"},
        },
    }
    draft = store.create_draft(
        "experiment", "Development protocol", {"type": "analysis-protocol", "spec": protocol_spec}
    )
    protocols = ProtocolService(store, LocalFilesystem((tmp_path,)))
    preview = protocols.preview(draft["id"], 1)
    assert preview["canFreeze"], preview["findings"]
    protocol = protocols.freeze(draft["id"], 1, preview["previewHash"], "protocol")
    spec = DevelopmentBatchSpec(
        experimentName="Optimizer tuning",
        batchName="Sweep v1",
        inputs={"protocolId": protocol["id"], "featureBundleId": bundle["id"]},
        mode="grid",
        grid={
            "learningRates": [0.0001, 0.0003, 0.001],
            "weightDecays": [0, 0.0001],
            "maxEpochs": [50, 100],
        },
        trainingSeeds=[10, 20, 30],
    )
    return DevelopmentService(store, LocalFilesystem((tmp_path,))), spec, source


def codes(preview):
    return {item["code"] for item in preview["findings"] if item["severity"] == "error"}


@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, "4096", 1000001])
def test_bag_size_rejects_invalid_limits_instead_of_interpreting_them_as_whole_bags(value):
    with pytest.raises(ValidationError):
        TrainingRecipe(bagSize=value)


def test_bag_size_whole_bag_is_explicit_null_and_default_remains_4096():
    assert TrainingRecipe().bagSize == 4096
    for value in (None, 1, 4096, 1000000):
        recipe = TrainingRecipe(bagSize=value)
        restored = TrainingRecipe.model_validate_json(recipe.model_dump_json())
        assert restored.bagSize == value
        assert "bagSize" in restored.model_dump()


@pytest.mark.parametrize(
    "changes",
    [
        {"precision": "16-true"},
        {"gradientClipNorm": -1},
        {"accumulateGradBatches": 0},
        {"accumulateGradBatches": 1.5},
        {"lrScheduler": "unknown"},
        {"finalLrFraction": 0},
        {"finalLrFraction": 1.1},
        {"warmupEpochs": 1},
        {"lrScheduler": "cosine", "maxEpochs": 2, "warmupEpochs": 2},
        {"minEpochs": 5, "maxEpochs": 4},
        {"earlyStoppingMinDelta": -1},
    ],
)
def test_advanced_recipe_rejects_invalid_optimization_and_epoch_controls(changes):
    with pytest.raises(ValidationError):
        TrainingRecipe(**changes)


def test_resolved_grid_recipes_validate_minimum_and_warmup_epochs(batch):
    service, spec, _source = batch
    raw = spec.model_dump()
    raw["recipe"].update(lrScheduler="cosine", warmupEpochs=10)
    raw["grid"]["maxEpochs"] = [10, 20]
    with pytest.raises(ValidationError, match="Warmup epochs"):
        DevelopmentBatchSpec.model_validate(raw)
    bypassed = spec.model_copy(
        update={
            "recipe": TrainingRecipe(lrScheduler="cosine", warmupEpochs=10),
            "grid": spec.grid.model_copy(update={"maxEpochs": [10, 20]}),
        }
    )
    with pytest.raises(StorageError) as error:
        service.preview(bypassed)
    assert error.value.code == "INVALID_TRAINING_RECIPE"


def test_whole_bag_freeze_preserves_original_bounded_batch_seeds_and_memberships(batch):
    service, spec, _source = batch
    original_preview = service.preview(spec)
    original = service.freeze(
        spec, original_preview["previewHash"], "bounded-bags", {"tag": "Bounded bags"}
    )
    whole = DevelopmentBatchSpec.model_validate(
        {**spec.model_dump(), "recipe": {**spec.recipe.model_dump(), "bagSize": None}}
    )
    preview = service.preview(whole)
    assert preview["canFreeze"], preview["findings"]
    assert preview["summary"] == original_preview["summary"]
    assert preview["splitPlans"] == original_preview["splitPlans"]
    assert {run["trainingSeed"] for run in preview["runs"]} == set(spec.trainingSeeds)
    assert all(item["recipe"]["bagSize"] is None for item in preview["configurations"])
    assert {item["id"] for item in preview["configurations"]}.isdisjoint(
        item["id"] for item in original_preview["configurations"]
    )
    frozen = service.freeze(whole, preview["previewHash"], "whole-bags", {"tag": "Whole bags"})
    assert frozen["manifest"]["spec"]["recipe"]["bagSize"] is None
    assert all(item["recipe"]["bagSize"] is None for item in frozen["manifest"]["configurations"])
    assert (
        service.freeze(whole, preview["previewHash"], "whole-bags", {"tag": "Whole bags"}) == frozen
    )
    assert service.store.get_configuration(original["id"]) == original
    assert all(item["recipe"]["bagSize"] == 4096 for item in original["manifest"]["configurations"])
    explicit = whole.model_copy(
        update={"mode": "explicit", "configurations": [spec.recipe, whole.recipe, whole.recipe]}
    )
    assert [recipe["bagSize"] for recipe in expand_recipes(explicit)] == [4096, None]


def test_grid_expands_twelve_configurations_three_training_seeds_five_frozen_plans(batch):
    service, spec, _source = batch
    protocol = service.store.get_configuration(spec.inputs.protocolId)
    before = copy.deepcopy(protocol)
    preview = service.preview(spec)
    assert preview["canFreeze"], preview["findings"]
    assert preview["summary"] == {
        "configurationCount": 12,
        "trainingSeedCount": 3,
        "splitPlanCount": 5,
        "runCount": 180,
    }
    assert len({run["id"] for run in preview["runs"]}) == 180
    assert {plan["seed"] for plan in preview["splitPlans"]} == {42}
    assert {run["trainingSeed"] for run in preview["runs"]} == {10, 20, 30}
    assert all(run["status"] == "planned" for run in preview["runs"])
    assert preview["executionImplemented"] is False
    assert service.store.get_configuration(spec.inputs.protocolId) == before
    changed_seeds = spec.model_copy(update={"trainingSeeds": [91, 92]})
    other = service.preview(changed_seeds)
    assert other["splitPlans"] == preview["splitPlans"]
    assert other["summary"]["runCount"] == 120
    assert service.store.get_configuration(spec.inputs.protocolId) == before


def test_explicit_rows_preserve_parameter_pairs_and_deduplicate_equal_recipes(batch):
    service, spec, _source = batch
    first = TrainingRecipe(learningRate=0.001, weightDecay=0, maxEpochs=50)
    second = TrainingRecipe(learningRate=0.0001, weightDecay=0.01, maxEpochs=100)
    spec = spec.model_copy(
        update={"mode": "explicit", "configurations": [first, second, first.model_copy()]}
    )
    preview = service.preview(spec)
    assert preview["canFreeze"], preview["findings"]
    assert preview["summary"]["configurationCount"] == 2
    assert preview["summary"]["runCount"] == 30
    assert [item["recipe"] for item in preview["configurations"]] == [
        first.model_dump(),
        second.model_dump(),
    ]


def test_identical_resolved_recipes_have_same_identity_across_setup_modes(batch):
    service, spec, _source = batch
    spec = spec.model_copy(update={"mode": "single"})
    single = service.preview(spec)
    explicit = service.preview(
        spec.model_copy(update={"mode": "explicit", "configurations": [spec.recipe.model_copy()]})
    )
    assert single["configurations"] == explicit["configurations"]
    assert single["runs"] == explicit["runs"]


@pytest.mark.parametrize("version", [1, 2, 3])
def test_legacy_protocols_cannot_create_new_development_batches(batch, version):
    service, spec, _source = batch
    manifest = service.store.get_configuration(spec.inputs.protocolId)["manifest"]
    manifest["spec"]["split"]["version"] = version
    old = service.store.publish_configuration(manifest=manifest, operation_id="legacy")
    spec = spec.model_copy(
        update={"inputs": spec.inputs.model_copy(update={"protocolId": old["id"]})}
    )
    preview = service.preview(spec)
    assert "LEGACY_DEVELOPMENT_PROTOCOL" in codes(preview)
    assert not preview["runs"]
    with pytest.raises(StorageError) as error:
        service.freeze(spec, preview["previewHash"], "blocked", {"tag": "Blocked"})
    assert error.value.code == "BATCH_PREFLIGHT_BLOCKED"


def test_nested_cv_search_dependency_is_blocked_without_generating_outer_scores(batch):
    service, spec, _source = batch
    manifest = service.store.get_configuration(spec.inputs.protocolId)["manifest"]
    manifest["spec"]["split"]["mode"] = "nested_kfold"
    nested = service.store.publish_configuration(manifest=manifest, operation_id="nested")
    spec = spec.model_copy(
        update={"inputs": spec.inputs.model_copy(update={"protocolId": nested["id"]})}
    )
    preview = service.preview(spec)
    assert "NESTED_SELECTION_REQUIRED" in codes(preview)
    assert not preview["runs"]
    assert not preview["canFreeze"]


@pytest.mark.parametrize("mode", ["monte_carlo", "leave_one_domain_out", "held_out"])
def test_nonexecutable_v4_splits_block_new_batches_without_changing_saved_protocols(batch, mode):
    service, spec, _source = batch
    manifest = service.store.get_configuration(spec.inputs.protocolId)["manifest"]
    manifest["spec"]["split"]["mode"] = mode
    protocol = service.store.publish_configuration(manifest=manifest, operation_id="unsupported")
    spec = spec.model_copy(
        update={"inputs": spec.inputs.model_copy(update={"protocolId": protocol["id"]})}
    )
    preview = service.preview(spec)
    assert "TRAINING_SPLIT_UNSUPPORTED" in codes(preview)
    assert not preview["canFreeze"]
    assert not preview["runs"]
    assert not preview["configurations"]
    with pytest.raises(StorageError) as error:
        service.freeze(spec, preview["previewHash"], "blocked", {"tag": "Blocked"})
    assert error.value.code == "BATCH_PREFLIGHT_BLOCKED"
    assert service.store.get_configuration(protocol["id"]) == protocol
    assert service.store.list_configurations("mil-batch") == []


def test_development_plan_expansion_omits_legacy_final_and_external_pool_rows():
    rows = [
        {
            "slideId": "development",
            "partition": "train",
            "seed": 42,
            "fold": 0,
            "phase": "evaluation",
        },
        {"slideId": "external", "partition": "test", "seed": 42, "fold": 0, "phase": "final"},
        {
            "slideId": "external",
            "partition": "test",
            "seed": 42,
            "fold": 0,
            "pool": "external_test",
        },
    ]
    result = development_plans({"memberships": rows})
    assert len(result) == 1
    assert result[0]["slideCount"] == 1
    assert result[0]["partitions"] == {"train": 1}


def test_freeze_is_idempotent_and_rejects_changed_spec_tag_or_note(batch):
    service, spec, _source = batch
    preview = service.preview(spec)
    label = {"tag": "AdamW exploration", "note": "Three training seeds."}
    frozen = service.freeze(spec, preview["previewHash"], "batch-freeze", label)
    assert service.freeze(spec, preview["previewHash"], "batch-freeze", label) == frozen
    assert len(service.list()["items"]) == 1
    assert frozen["manifest"]["summary"]["runCount"] == 180
    changed = spec.model_copy(update={"trainingSeeds": [100]})
    for changed_spec, changed_label in [
        (changed, label),
        (spec, {**label, "tag": "Other"}),
        (spec, {**label, "note": "Changed note"}),
    ]:
        with pytest.raises(StorageError) as error:
            service.freeze(changed_spec, preview["previewHash"], "batch-freeze", changed_label)
        assert error.value.code == "OPERATION_CONFLICT"


def test_freeze_replays_after_source_staleness_without_republishing(batch):
    service, spec, source = batch
    preview = service.preview(spec)
    label = {"tag": "Original batch"}
    original = service.freeze(spec, preview["previewHash"], "original-batch", label)
    with h5py.File(source / "s00.h5", "r+") as handle:
        handle["features"][0, 0] = 123
    assert service.freeze(spec, preview["previewHash"], "original-batch", label) == original
    assert "BUNDLE_STALE" in codes(service.preview(spec))


@pytest.mark.parametrize("changed", ["source", "pack", "evidence"])
def test_freshness_is_rechecked_inside_publication_transaction(batch, monkeypatch, changed):
    service, spec, source = batch
    preview = service.preview(spec)
    bundle = service.store.get_configuration(spec.inputs.featureBundleId)["manifest"]
    publish = service.store.publish_configuration

    def change_before_publish(*args, **kwargs):
        if changed == "source":
            with h5py.File(source / "s00.h5", "r+") as handle:
                handle["features"][0, 0] = 123
        elif changed == "pack":
            path = Path(bundle["packs"][0]["outputPath"]) / "features.bin"
            with path.open("r+b") as handle:
                handle.write(b"\x00\x00\x00\x00")
        else:
            packing = FeaturePackService(service.store, service.filesystem)
            path = packing.folder / bundle["feature"]["validation"]["jobId"] / "result.json"
            path.write_text(path.read_text() + "\n")
        return publish(*args, **kwargs)

    monkeypatch.setattr(service.store, "publish_configuration", change_before_publish)
    with pytest.raises(StorageError) as error:
        service.freeze(spec, preview["previewHash"], "raced", {"tag": "Raced batch"})
    assert error.value.code == "PREVIEW_STALE"
    assert service.store.configuration_publication("raced") is None


def test_changed_preview_intent_cannot_be_frozen(batch):
    service, spec, _source = batch
    preview = service.preview(spec)
    changed = spec.model_copy(update={"trainingSeeds": [777]})
    with pytest.raises(StorageError) as error:
        service.freeze(changed, preview["previewHash"], "wrong-preview", {"tag": "Changed"})
    assert error.value.code == "STALE_PREVIEW"


def test_batch_expansion_bounds_and_explicit_mode_contract():
    base = {
        "experimentName": "Search",
        "batchName": "Grid",
        "inputs": {"protocolId": "protocol", "featureBundleId": "bundle"},
    }
    spec = DevelopmentBatchSpec(
        **base,
        mode="grid",
        grid={
            "learningRates": [i / 10000 for i in range(1, 10)],
            "weightDecays": [i / 10000 for i in range(9)],
            "maxEpochs": list(range(1, 10)),
        },
    )
    with pytest.raises(StorageError) as error:
        expand_recipes(spec)
    assert error.value.code == "BATCH_TOO_LARGE"
    with pytest.raises(ValidationError):
        DevelopmentBatchSpec(**base, mode="explicit")
    with pytest.raises(ValidationError):
        DevelopmentBatchSpec(**base, trainingSeeds=[10, 10])
