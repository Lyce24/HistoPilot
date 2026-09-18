"""Frozen batches and completed receipts cannot bypass representation boundaries."""

import hashlib
import shutil
from pathlib import Path

import pytest
from test_evaluations import bundle
from test_predictor_registry import candidate, registry
from test_training_execution import execution, rewrite_batch

from histopilot.application.mil_inputs import MILInputService
from histopilot.schemas.mil import MILInputSpec
from histopilot.storage.project_lock import StorageError
from histopilot.workers.compute_archive import prepare_compute_archive
from histopilot.workers.training_process import compute_snapshot

__all__ = ["execution", "registry"]


def slide_bundle(service, tmp_path, protocol):
    dataset = service.store.get_dataset(protocol["manifest"]["datasetId"])
    ids = sorted({row["slideId"] for row in protocol["manifest"]["memberships"]})
    document, _, _ = bundle(service.store, tmp_path, dataset, ids,
                            name="slide-inventory", feature_kind="slide")
    return document


@pytest.mark.parametrize("kind,model", [("slide", "abmil"), ("patch", "slide_linear")])
def test_launch_rejects_incompatible_older_frozen_batch(execution, tmp_path, kind, model):
    service, original, executor, _ = execution
    inputs = dict(original["manifest"]["spec"]["inputs"])
    if kind == "slide":
        protocol = service.store.get_configuration(inputs["protocolId"])
        inputs["featureBundleId"] = slide_bundle(service, tmp_path, protocol)["id"]
    binding = MILInputService(service.store, service.filesystem).preview(
        MILInputSpec.model_validate(inputs)
    )
    assert binding["canPlan"], binding["findings"]

    def update(manifest):
        # Model the stored state of a batch from before launch validated the kind.
        manifest["spec"]["inputs"] = inputs
        manifest["spec"]["recipe"]["model"] = model
        manifest["resolvedInputs"] = binding
        for configuration in manifest["configurations"]:
            configuration["recipe"]["model"] = model

    malformed = rewrite_batch(service, original, update)
    with pytest.raises(StorageError) as error:
        service.launch(malformed["id"], "reject-wrong-representation")
    assert error.value.code == "TRAINING_FEATURE_KIND_MISMATCH"
    assert not executor.launches
    assert not (service.store.folder / "training" / malformed["id"] / "plan.json").exists()
    assert service.store.get_configuration(original["id"]) == original


@pytest.mark.parametrize("kind,model", [("slide", "abmil"), ("patch", "slide_mlp")])
def test_promotion_rejects_matching_receipts_for_incompatible_representation(
    registry, tmp_path, kind, model,
):
    service, _ = registry
    selected_bundle = None
    if kind == "slide":
        protocol = next(item for item in service.store.list_configurations("protocol")
                        if item["manifest"]["spec"]["split"].get("mode") == "kfold")
        selected_bundle = slide_bundle(service, tmp_path, protocol)["id"]
    selection, _, _ = candidate(service, model=model, feature_bundle_id=selected_bundle)
    reviewed = service.preview(selection)
    assert not reviewed["canFreeze"]
    assert reviewed["findings"][0]["code"] == "PREDICTOR_FEATURE_KIND_MISMATCH"
    assert not service.store.list_configurations("frozen-predictor")


def test_representation_helper_is_fingerprinted_before_first_worker_archive(tmp_path):
    from histopilot.workers import training_process

    source = Path(training_process.__file__).resolve().parents[1]
    code = compute_snapshot()
    helper = "domain/features.py"
    assert code["files"][helper] == hashlib.sha256((source / helper).read_bytes()).hexdigest()
    package = tmp_path / "source"
    package.mkdir()
    (package / "__init__.py").write_text("")
    for name in code["files"]:
        target = package / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    (package / helper).write_text("def representation_kind(manifest): return 'slide'\n")
    with pytest.raises(StorageError, match="domain/features.py") as error:
        prepare_compute_archive(tmp_path / "new-run", code, source_root=package)
    assert error.value.code == "TRAINING_RUNTIME_CHANGED"
    assert not (tmp_path / "new-run" / "compute").exists()
