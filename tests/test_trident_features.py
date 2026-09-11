"""Native TRIDENT discovery retains coordinate and extraction provenance bindings."""

import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from histopilot.application.features import FeatureService
from histopilot.schemas.features import FeatureSpec
from histopilot.storage.filesystem import FilesystemError, LocalFilesystem
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import ScientificStore


@pytest.fixture
def trident(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    store = ScientificStore(project, "project-trident")
    draft = store.create_draft("import", "dataset", {})
    frozen = store.publish_dataset(
        draft["id"],
        expected_revision=1,
        manifest={"kind": "dataset"},
        artifacts={"records.json": json.dumps([{"slideId": "001.A"}]).encode()},
        operation_id="dataset",
    )
    job = tmp_path / "job"
    coords = job / "20x_256px_0px_overlap"
    features = coords / "features_uni_v1"
    features.mkdir(parents=True)
    (coords / "patches").mkdir()
    (job / "_config_segmentation.json").write_text(json.dumps({"seg_mag": 10}))
    (coords / "_config_coords.json").write_text(json.dumps({"patch_size": 256}))
    (coords / "_config_feats_uni_v1.json").write_text(json.dumps({"batch_limit": 64}))
    return (
        FeatureService(store, LocalFilesystem((tmp_path,))),
        FeatureSpec(datasetId=frozen["id"], path=str(job)),
        features,
    )


def write_features(path, *, embedded=True, encoder="uni_v1", name="001.A"):
    with h5py.File(path, "w") as handle:
        features = handle.create_dataset("features", data=np.ones((3, 4), dtype="float32"))
        features.attrs["encoder"] = encoder
        features.attrs["name"] = name
        if embedded:
            write_coords(handle, name)


def write_coords(handle, name="001.A", count=3):
    coords = handle.create_dataset("coords", data=np.arange(count * 2).reshape(count, 2))
    coords.attrs.update(
        {
            "name": name,
            "patch_size": 256,
            "patch_size_level0": 512,
            "level0_magnification": 40,
            "target_magnification": 20,
            "overlap": 0,
            "level0_width": 102400,
            "level0_height": 51200,
        }
    )


@pytest.mark.parametrize("selected", ["job", "coords", "encoder"])
def test_native_layout_auto_discovers_one_encoder_and_preserves_attributes(trident, selected):
    service, spec, features = trident
    write_features(features / "001.A.h5")
    with h5py.File(features.parent / "patches" / "001.A_patches.h5", "w") as handle:
        write_coords(handle)
    selected_path = {"job": features.parent.parent, "coords": features.parent, "encoder": features}
    spec = spec.model_copy(update={"path": str(selected_path[selected]), "recursive": True})
    preview = service.preview(spec)
    assert preview["canFreeze"]
    assert preview["summary"]["matchedSlides"] == 1
    assert preview["summary"]["orphanFiles"] == 0
    assert preview["layout"]["encoderId"] == "uni_v1"
    assert preview["layout"]["featureDirectory"] == str(features)
    item = preview["files"][0]
    assert item["coordinateSource"] == "embedded"
    assert item["coordinatePath"] == str(features / "001.A.h5")
    assert item["coordinateSpace"] == "level0_pixels"
    assert item["attributes"]["coords"]["patch_size_level0"] == 512
    assert item["attributes"]["features"]["encoder"] == "uni_v1"
    assert len(preview["provenance"]) == 3
    frozen = service.freeze(spec, preview["previewHash"], "attach")
    assert service.verify_binding(frozen) == []
    config = features.parent / "_config_coords.json"
    config.write_text(json.dumps({"patch_size": 512}))
    assert service.verify_binding(frozen)[0]["code"] == "FEATURE_SOURCE_CHANGED"
    with pytest.raises(StorageError, match="Preview again"):
        service.freeze(spec, preview["previewHash"], "stale")


def test_multiple_encoders_require_an_explicit_selection(trident):
    service, spec, features = trident
    write_features(features / "001.A.h5")
    other = features.parent / "features_virchow"
    other.mkdir()
    write_features(other / "001.A.h5", encoder="virchow")
    with pytest.raises(StorageError) as error:
        service.preview(spec)
    assert error.value.code == "FEATURE_LAYOUT_AMBIGUOUS"
    preview = service.preview(spec.model_copy(update={"encoderId": "virchow"}))
    assert preview["canFreeze"]
    assert preview["layout"]["featureDirectory"] == str(other)


def test_project_encoder_alias_preserves_spec_and_resolves_native_encoder(trident):
    service, spec, features = trident
    write_features(features / "001.A.h5")
    preview = service.preview(spec.model_copy(update={"encoderId": "uni"}))
    assert preview["canFreeze"]
    assert preview["spec"]["encoderId"] == "uni"
    assert preview["layout"]["encoderId"] == "uni_v1"


def test_legacy_encoder_alias_in_directory_and_attributes_is_accepted(trident):
    service, spec, features = trident
    legacy = features.with_name("features_uni")
    features.rename(legacy)
    write_features(legacy / "001.A.h5", encoder="uni")
    preview = service.preview(spec.model_copy(update={"encoderId": "uni"}))
    assert preview["canFreeze"]
    assert preview["layout"]["encoderId"] == "uni_v1"
    assert preview["files"][0]["attributes"]["features"]["encoder"] == "uni"


def test_encoder_folder_can_itself_be_the_only_permitted_root(trident):
    service, spec, features = trident
    write_features(features / "001.A.h5")
    service.filesystem = LocalFilesystem((features,))
    preview = service.preview(spec.model_copy(update={"path": str(features)}))
    assert preview["canFreeze"]
    assert preview["layout"]["coordinatesDirectory"] is None
    assert preview["provenance"] == []


def test_unmatched_encoder_has_actionable_layout_error(trident):
    service, spec, features = trident
    with pytest.raises(StorageError) as error:
        service.preview(spec.model_copy(update={"encoderId": "uni2"}))
    assert error.value.code == "FEATURE_LAYOUT_NOT_FOUND"


def test_same_encoder_at_multiple_patch_sizes_requires_specific_folder(trident):
    service, spec, features = trident
    (features.parent.parent / "10x_256px_0px_overlap" / "features_uni_v1").mkdir(parents=True)
    with pytest.raises(StorageError) as error:
        service.preview(spec.model_copy(update={"encoderId": "uni_v1"}))
    assert error.value.code == "FEATURE_LAYOUT_AMBIGUOUS"


def test_separate_patch_coordinates_are_bound_and_revalidated(trident):
    service, spec, features = trident
    write_features(features / "001.A.h5", embedded=False)
    coords = features.parent / "patches" / "001.A_patches.h5"
    with h5py.File(coords, "w") as handle:
        write_coords(handle)
    preview = service.preview(spec)
    assert preview["canFreeze"]
    assert preview["files"][0]["coordinateSource"] == "trident-patches"
    assert preview["files"][0]["coordinatePath"] == str(coords)
    frozen = service.freeze(spec, preview["previewHash"], "split-coords")
    assert service.verify_binding(frozen) == []
    original = coords.stat()
    replacement = coords.with_name("replacement.h5")
    with h5py.File(replacement, "w") as handle:
        write_coords(handle)
        handle["coords"][0, 0] = 999
    assert replacement.stat().st_size == original.st_size
    os.utime(replacement, ns=(original.st_atime_ns, original.st_mtime_ns))
    replacement.replace(coords)
    assert service.verify_binding(frozen)[0]["code"] == "FEATURE_SOURCE_CHANGED"
    with pytest.raises(StorageError) as error:
        service.freeze(spec, preview["previewHash"], "stale")
    assert error.value.code == "PREVIEW_STALE"


def test_explicit_coordinate_directory_works_for_flat_embeddings(trident, tmp_path):
    service, spec, features = trident
    flat = tmp_path / "flat"
    flat.mkdir()
    write_features(flat / "001.A.h5", embedded=False)
    coords_root = features.parent / "patches"
    with h5py.File(coords_root / "001.A_patches.h5", "w") as handle:
        write_coords(handle)
    preview = service.preview(
        spec.model_copy(
            update={
                "path": str(flat),
                "layout": "flat",
                "coordinatesPath": str(coords_root),
            }
        )
    )
    assert preview["canFreeze"]
    assert preview["layout"]["kind"] == "flat"
    assert preview["layout"]["encoderId"] == "uni_v1"
    assert preview["files"][0]["coordinateSource"] == "trident-patches"


@pytest.mark.parametrize("kind", ["mismatched_rows", "external", "soft", "symlink", "missing"])
def test_invalid_separate_coordinates_block_freezing(trident, kind):
    service, spec, features = trident
    write_features(features / "001.A.h5", embedded=False)
    coords = features.parent / "patches" / "001.A_patches.h5"
    if kind == "symlink":
        coords.symlink_to(features / "001.A.h5")
    elif kind != "missing":
        with h5py.File(coords, "w") as handle:
            if kind == "mismatched_rows":
                write_coords(handle, count=2)
            elif kind == "external":
                handle["coords"] = h5py.ExternalLink(str(features / "001.A.h5"), "features")
            else:
                write_coords(handle)
                handle.move("coords", "other")
                handle["coords"] = h5py.SoftLink("other")
    preview = service.preview(spec)
    assert not preview["canFreeze"]
    assert any(item["code"] == "INVALID_FEATURE_HEADER" for item in preview["findings"])


@pytest.mark.parametrize("field,value", [("encoder", "virchow"), ("name", "different-slide")])
def test_foreign_encoder_or_slide_attributes_block_freezing(trident, field, value):
    service, spec, features = trident
    write_features(features / "001.A.h5", **{field: value})
    preview = service.preview(spec)
    assert not preview["canFreeze"]
    assert any(item["code"] == "INVALID_FEATURE_HEADER" for item in preview["findings"])


def test_coordinate_and_provenance_symlinks_do_not_escape(trident, tmp_path):
    service, spec, features = trident
    write_features(features / "001.A.h5")
    link = tmp_path / "linked-coords"
    link.symlink_to(features.parent / "patches", target_is_directory=True)
    with pytest.raises(FilesystemError):
        service.preview(spec.model_copy(update={"coordinatesPath": str(link)}))
    config = features.parent / "_config_coords.json"
    original = config.with_name("original.json")
    config.rename(original)
    config.symlink_to(original)
    # A direct encoder import does not scan its sibling config directory; provenance
    # inspection must independently enforce the same no-link rule.
    preview = service.preview(spec.model_copy(update={"path": str(features)}))
    assert not preview["canFreeze"]
    assert any(item["code"] == "INVALID_FEATURE_PROVENANCE" for item in preview["findings"])


def test_flat_historical_binding_remains_valid(trident, tmp_path):
    service, spec, features = trident
    write_features(features / "001.A.h5")
    preview = service.preview(spec)
    frozen = service.freeze(spec, preview["previewHash"], "historical")
    for item in frozen["manifest"]["files"]:
        for key in ["attributes", "coordinatePath", "coordinateSource", "coordinateSpace"]:
            del item[key]
    assert service.verify_binding(frozen) == []


def test_explicit_coordinates_must_be_in_permitted_roots(trident, tmp_path):
    service, spec, features = trident
    service.filesystem = LocalFilesystem((Path(spec.path),))
    with pytest.raises(FilesystemError):
        service.preview(spec.model_copy(update={"coordinatesPath": str(tmp_path)}))
