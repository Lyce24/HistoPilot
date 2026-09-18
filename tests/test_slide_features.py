"""Attaching one embedding per slide from a slide encoder, and keeping it distinct.

A slide encoder reduces a whole slide to one vector, so these files carry no patch
grid. The attachment must accept them, must not accept a patch bag in their place,
and must leave the identity of records frozen before slide encoders untouched.
"""

import copy
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from histopilot.application.feature_bundles import FeatureBundleService
from histopilot.application.feature_packs import FeaturePackService
from histopilot.application.features import FeatureService
from histopilot.models import catalog
from histopilot.schemas.feature_bundles import FeatureBundleSpec
from histopilot.schemas.feature_packs import FeaturePackSpec
from histopilot.schemas.features import FeatureSpec
from histopilot.storage.filesystem import LocalFilesystem
from histopilot.storage.packed import PackedStoreError, PackingCancelled, validate_features
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import ScientificStore
from histopilot.workers.pack_features import run_job


@pytest.fixture
def attached(tmp_path):
    folder = tmp_path / "experiment"
    folder.mkdir()
    store = ScientificStore(folder, "project-slide-features")
    draft = store.create_draft("import", "dataset", {})
    frozen = store.publish_dataset(
        draft["id"],
        expected_revision=1,
        manifest={"kind": "dataset"},
        artifacts={
            "records.json": json.dumps(
                [{"slideId": "001.A", "patientId": "P1"}, {"slideId": "002", "patientId": "P2"}]
            ).encode()
        },
        operation_id="dataset",
    )
    root = tmp_path / "slide_features_titan"
    root.mkdir()
    service = FeatureService(store, LocalFilesystem((tmp_path,)))
    spec = FeatureSpec(datasetId=frozen["id"], path=str(root), featureKind="slide", layout="flat")
    return service, spec, root


def slide_embedding(path, dimensions=8, *, rows=None):
    """TRIDENT writes a bare vector or a single row; both are one slide embedding."""
    shape = (dimensions,) if rows is None else (rows, dimensions)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("features", data=np.ones(shape, dtype="float32"))


def patch_bag(path, dimensions=8):
    with h5py.File(path, "w") as handle:
        handle.create_dataset("features", data=np.ones((3, dimensions), dtype="float32"))
        handle.create_dataset("coords", data=np.ones((3, 2), dtype="int64"))


def test_a_bare_vector_and_a_single_row_both_attach_as_one_embedding(attached):
    service, spec, root = attached
    slide_embedding(root / "001.A.h5")
    slide_embedding(root / "002.h5", rows=1)
    preview = service.preview(spec)
    assert preview["summary"] == {
        "slideCount": 2,
        "scope": "dataset",
        "matchedSlides": 2,
        "missingSlides": 0,
        "orphanFiles": 0,
        "dimensions": 8,
        # One embedding per slide, whichever way the file stored it.
        "patchCount": 2,
    }
    assert [item["patchCount"] for item in preview["files"]] == [1, 1]
    assert {item["featureKind"] for item in preview["files"]} == {"slide"}
    frozen = service.freeze(spec, preview["previewHash"], "attach-slide")
    assert service.verify_binding(frozen) == []


def test_slide_attachment_records_its_kind_and_keeps_patch_records_unchanged(attached):
    service, spec, root = attached
    slide_embedding(root / "001.A.h5")
    slide_embedding(root / "002.h5")
    frozen = service.freeze(spec, service.preview(spec)["previewHash"], "attach-slide")
    assert frozen["manifest"]["spec"]["featureKind"] == "slide"
    # A patch selection serializes exactly as it did before this field existed.
    patch_spec = FeatureSpec(datasetId=spec.datasetId, path=str(root))
    assert "featureKind" not in patch_spec.model_dump(mode="json")


def test_a_patch_bag_cannot_be_attached_as_a_slide_embedding(attached):
    service, spec, root = attached
    patch_bag(root / "001.A.h5")
    patch_bag(root / "002.h5")
    preview = service.preview(spec)
    assert not preview["canFreeze"]
    assert any(
        "one nonempty floating-point vector" in item["message"] for item in preview["findings"]
    )
    with pytest.raises(StorageError) as error:
        service.freeze(spec, preview["previewHash"], "attach-slide")
    assert error.value.code == "FEATURES_INVALID"


def test_a_slide_embedding_cannot_be_attached_as_a_patch_bag(attached):
    service, spec, root = attached
    slide_embedding(root / "001.A.h5")
    slide_embedding(root / "002.h5")
    patch_request = FeatureSpec(datasetId=spec.datasetId, path=str(root), layout="flat")
    preview = service.preview(patch_request)
    assert not preview["canFreeze"]
    assert any("two-dimensional" in item["message"] for item in preview["findings"])


def test_an_empty_or_multi_row_matrix_is_not_one_slide_embedding(attached):
    service, spec, root = attached
    slide_embedding(root / "001.A.h5", rows=2)
    slide_embedding(root / "002.h5")
    preview = service.preview(spec)
    assert not preview["canFreeze"]
    assert any(
        "one nonempty floating-point vector" in item["message"] for item in preview["findings"]
    )


def test_discovery_finds_a_slide_features_folder_and_infers_its_encoder(attached):
    service, spec, root = attached
    slide_embedding(root / "001.A.h5")
    slide_embedding(root / "002.h5")
    # TRIDENT names slide output slide_features_<encoder>; automatic layout must
    # recognize it rather than looking for the patch features_<encoder> folder.
    discovered = FeatureSpec(datasetId=spec.datasetId, path=str(root), featureKind="slide")
    preview = service.preview(discovered)
    assert preview["canFreeze"]
    assert preview["layout"]["kind"] == "trident"
    assert preview["layout"]["encoderId"] == "titan"


def test_patch_discovery_does_not_adopt_a_slide_features_folder(tmp_path, attached):
    service, spec, root = attached
    slide_embedding(root / "001.A.h5")
    slide_embedding(root / "002.h5")
    # A patch attachment pointed at the parent must not descend into the slide
    # output and present those embeddings as a patch feature set.
    patch_request = FeatureSpec(datasetId=spec.datasetId, path=str(tmp_path))
    preview = service.preview(patch_request)
    assert preview["layout"]["kind"] == "flat"
    assert preview["summary"]["matchedSlides"] == 0


def test_a_slide_bundle_cannot_be_packed(attached, tmp_path):
    from histopilot.storage.pack_import import inspect_existing_pack
    from histopilot.storage.packed import build_pack

    service, spec, root = attached
    slide_embedding(root / "001.A.h5")
    slide_embedding(root / "002.h5")
    frozen = service.freeze(spec, service.preview(spec)["previewHash"], "attach-slide")
    # Packing exists to make large patch bags fast to read; one vector per slide
    # gains nothing from it, and the packed layout cannot describe it.
    with pytest.raises(PackedStoreError, match="one embedding per slide"):
        build_pack(frozen, tmp_path / "pack")
    with pytest.raises(PackedStoreError, match="one embedding per slide"):
        inspect_existing_pack(frozen, tmp_path / "pack")
    assert not (tmp_path / "pack").exists()


def freeze_slide_inventory(attached):
    service, spec, root = attached
    slide_embedding(root / "001.A.h5")
    slide_embedding(root / "002.h5", rows=1)
    return service.freeze(spec, service.preview(spec)["previewHash"], "slide-validation")


def test_full_validation_authenticates_both_shapes_without_coordinates(attached):
    frozen = freeze_slide_inventory(attached)
    events = []
    report = validate_features(frozen, chunk_bytes=8, progress=events.append)
    assert report["valid"] and report["tensorValidationComplete"]
    assert report["featureKind"] == "slide"
    assert report["slideCount"] == report["totalPatches"] == 2
    assert report["coordinateDtypes"] == []
    assert report["coordinateValidation"] == "not-applicable"
    assert all(event["unit"] == "slides" for event in events)
    assert events[-1]["stage"] == "complete"
    assert len(report["sourceStamps"]) == 2
    for item in report["files"]:
        assert "coordinateFile" not in item and "coordinateTensorSha256" not in item
        assert item["sha256"] == hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest()
        assert (
            item["featureTensorSha256"]
            == hashlib.sha256(np.ones(8, dtype="<f4").tobytes()).hexdigest()
        )
    assert report["files"][0]["featureTensorSha256"] == report["files"][1]["featureTensorSha256"]


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_full_validation_rejects_nonfinite_slide_values_beyond_header(attached, invalid):
    service, spec, root = attached
    slide_embedding(root / "001.A.h5")
    slide_embedding(root / "002.h5", rows=1)
    with h5py.File(root / "002.h5", "r+") as handle:
        handle["features"][0, -1] = invalid
    preview = service.preview(spec)
    assert preview["canFreeze"]  # Header-only evidence must not authorize a bundle.
    frozen = service.freeze(spec, preview["previewHash"], "nonfinite")
    with pytest.raises(PackedStoreError, match="NaN or infinity"):
        validate_features(frozen, chunk_bytes=8)


def test_slide_validation_identity_uses_values_and_not_container_shape(attached):
    service, spec, root = attached
    first = validate_features(freeze_slide_inventory(attached))
    # Rewrite both shapes without changing vectors: tensor identity stays stable.
    slide_embedding(root / "001.A.h5", rows=1)
    slide_embedding(root / "002.h5")
    second = service.freeze(spec, service.preview(spec)["previewHash"], "reshaped")
    assert validate_features(second)["sourceContentHash"] == first["sourceContentHash"]
    with h5py.File(root / "002.h5", "r+") as handle:
        handle["features"][-1] = 5
    third = service.freeze(spec, service.preview(spec)["previewHash"], "changed")
    assert validate_features(third)["sourceContentHash"] != first["sourceContentHash"]


def test_slide_validation_rejects_changed_source_during_container_checksum(attached, monkeypatch):
    from histopilot.storage import packed

    frozen = freeze_slide_inventory(attached)
    original = packed._stream_hash

    def mutate_after_hash(stream, *args):
        checksum = original(stream, *args)
        info = os.fstat(stream.fileno())
        os.utime(stream.fileno(), ns=(info.st_atime_ns, info.st_mtime_ns + 1000000))
        return checksum

    monkeypatch.setattr(packed, "_stream_hash", mutate_after_hash)
    with pytest.raises(PackedStoreError, match="changed since it was saved"):
        validate_features(frozen)


def test_slide_validation_rejects_header_and_kind_mismatch(attached):
    frozen = freeze_slide_inventory(attached)
    changed = copy.deepcopy(frozen)
    changed["manifest"]["files"][0]["dimensions"] = 4
    with pytest.raises(PackedStoreError, match="header changed"):
        validate_features(changed)
    changed = copy.deepcopy(frozen)
    changed["manifest"]["files"][0]["featureKind"] = "patch"
    with pytest.raises(PackedStoreError, match="inconsistent feature kinds"):
        validate_features(changed)


def test_slide_validation_checks_metadata_consistency(attached):
    service, spec, root = attached
    slide_embedding(root / "001.A.h5")
    slide_embedding(root / "002.h5")
    for index, name in enumerate(("001.A", "002")):
        with h5py.File(root / f"{name}.h5", "r+") as handle:
            handle["features"].attrs["checkpoint_sha256"] = str(index) * 64
    frozen = service.freeze(spec, service.preview(spec)["previewHash"], "metadata")
    with pytest.raises(PackedStoreError, match="conflicting checkpoint_sha256"):
        validate_features(frozen)


def test_slide_validation_rejects_symbolic_links_and_duplicate_source_files(attached):
    frozen = freeze_slide_inventory(attached)
    entries = frozen["manifest"]["files"]
    duplicate = copy.deepcopy(frozen)
    duplicate["manifest"]["files"][1] = {**entries[0], "slideId": "002"}
    with pytest.raises(PackedStoreError, match="same feature file"):
        validate_features(duplicate)
    original = Path(entries[0]["path"])
    target = original.with_suffix(".original")
    original.rename(target)
    original.symlink_to(target)
    with pytest.raises(PackedStoreError, match="symbolic links"):
        validate_features(frozen)


def test_slide_validation_cancels_between_vector_chunks(attached):
    frozen = freeze_slide_inventory(attached)
    calls = 0

    def cancelled():
        nonlocal calls
        calls += 1
        return calls == 3

    with pytest.raises(PackingCancelled):
        validate_features(frozen, cancelled=cancelled, chunk_bytes=8)
    assert calls == 3


class InlineValidationExecutor:
    def available(self):
        return True

    def running(self, session):
        return False

    def launch(self, session, runner, plan):
        run_job(plan)


def test_slide_inventory_full_worker_validation_and_frozen_bundle(attached):
    service, _, root = attached
    frozen = freeze_slide_inventory(attached)
    bundles = FeatureBundleService(service.store, service.filesystem)
    spec = FeatureBundleSpec(featureSetId=frozen["id"])
    assert not bundles.preview(spec)["canFreeze"]
    packs = FeaturePackService(service.store, service.filesystem, InlineValidationExecutor())
    validation = FeaturePackSpec(featureSetId=frozen["id"], action="validate")
    preview = packs.preview(validation)
    assert preview["canRun"], preview["findings"]
    job = packs.submit(validation, preview["previewHash"], "validate-slide-vectors")
    result = packs.get(job["id"])
    assert result["state"] == "succeeded", result
    assert result["result"]["artifact"] is None
    assert packs.validation_for(frozen["id"])["current"]
    ready = bundles.preview(spec)
    assert ready["canFreeze"], ready["findings"]
    bundle = bundles.freeze(
        spec, ready["previewHash"], "slide-bundle", version_label={"tag": "Slide vectors"}
    )
    assert bundle["manifest"]["feature"]["validation"]["tensorValidationComplete"]
    assert bundle["manifest"]["packs"] == []
    # A later external edit invalidates the evidence even after validation succeeds.
    with h5py.File(root / "002.h5", "r+") as handle:
        handle["features"][0, 0] = 9
    assert not bundles.preview(spec)["canFreeze"]


@pytest.mark.parametrize("action", ["pack", "attach"])
def test_slide_pack_requests_blocked_before_launch(attached, action):
    service, _, _ = attached
    frozen = freeze_slide_inventory(attached)
    packs = FeaturePackService(service.store, service.filesystem, InlineValidationExecutor())
    spec = FeaturePackSpec(featureSetId=frozen["id"], action=action)
    preview = packs.preview(spec)
    assert not preview["canRun"]
    assert any(item["code"] == "SLIDE_PACKING_UNSUPPORTED" for item in preview["findings"])


def test_the_catalog_separates_slide_probes_from_patch_architectures():
    assert catalog.feature_kind("slide_linear") == "slide"
    assert catalog.feature_kind("abmil") == "patch"
    # Nothing reads a slide embedding as a bag it can attend over.
    assert not catalog.supports_attention("slide_linear")
    assert not catalog.supports_attention("slide_mlp")
    assert catalog.SLIDE_MODELS == {"slide_linear", "slide_mlp"}
    assert "abmil" in catalog.PATCH_MODELS
