"""Feature attachment verifies exact IDs and headers without trusting external HDF5 storage."""

import json
import os

import h5py
import numpy as np
import pytest

from histopilot.application.features import FeatureService
from histopilot.schemas.features import FeatureSpec
from histopilot.storage.filesystem import FilesystemError, LocalFilesystem
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import ScientificStore


@pytest.fixture
def attached(tmp_path):
    folder = tmp_path / "experiment"
    folder.mkdir()
    store = ScientificStore(folder, "project-features")
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
    root = tmp_path / "features"
    root.mkdir()
    service = FeatureService(store, LocalFilesystem((tmp_path,)))
    return service, FeatureSpec(datasetId=frozen["id"], path=str(root)), root


def hdf5(path, dimensions=4):
    with h5py.File(path, "w") as handle:
        handle.create_dataset("features", data=np.ones((3, dimensions), dtype="float32"))
        handle.create_dataset("coords", data=np.ones((3, 2), dtype="int64"))


def test_attach_reload_then_modified_file_blocks_validation(attached):
    service, spec, root = attached
    hdf5(root / "001.A.h5")
    hdf5(root / "002.h5")
    preview = service.preview(spec)
    assert preview["summary"] == {
        "slideCount": 2,
        "scope": "dataset",
        "matchedSlides": 2,
        "missingSlides": 0,
        "orphanFiles": 0,
        "dimensions": 4,
        "patchCount": 6,
    }
    frozen = service.freeze(spec, preview["previewHash"], "attach")
    assert service.freeze(spec, preview["previewHash"], "attach") == frozen
    assert service.verify_binding(frozen) == []
    assert service.store.get_configuration(frozen["id"]) == frozen
    hdf5(root / "002.h5", dimensions=8)
    assert service.verify_binding(frozen)[0]["code"] == "FEATURE_SOURCE_CHANGED"
    with pytest.raises(StorageError) as error:
        service.freeze(spec, preview["previewHash"], "stale")
    assert error.value.code == "PREVIEW_STALE"


def test_partial_coverage_is_explicit_and_exact_dotted_ids_are_preserved(attached):
    service, spec, root = attached
    hdf5(root / "001.A.h5")
    hdf5(root / "001.h5")
    preview = service.preview(spec)
    assert preview["canFreeze"]
    assert preview["files"][0]["slideId"] == "001.A"
    assert preview["summary"]["missingSlides"] == 1
    assert preview["summary"]["orphanFiles"] == 1


@pytest.mark.parametrize("kind", ["external", "soft", "virtual", "raw_external", "bad_coords"])
def test_bad_or_external_hdf5_storage_blocks_freeze(attached, kind):
    service, spec, root = attached
    external = root / "other.h5"
    hdf5(external)
    path = root / "001.A.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("coords", data=np.ones((3, 2), dtype="int64"))
        if kind == "external":
            handle["features"] = h5py.ExternalLink(str(external), "features")
        elif kind == "soft":
            handle["features"] = h5py.SoftLink("coords")
        elif kind == "virtual":
            layout = h5py.VirtualLayout(shape=(3, 4), dtype="float32")
            layout[:] = h5py.VirtualSource(str(external), "features", shape=(3, 4))
            handle.create_virtual_dataset("features", layout)
        elif kind == "raw_external":
            handle.create_dataset(
                "features", shape=(3, 4), dtype="float32", external=[("outside.bin", 0, 48)]
            )
        else:
            handle.create_dataset("features", shape=(4, 4), dtype="float32")
    preview = service.preview(spec)
    assert not preview["canFreeze"]
    assert any(row["code"] == "INVALID_FEATURE_HEADER" for row in preview["findings"])


def test_duplicate_ids_in_recursive_tree_block_and_symlinks_are_rejected(attached):
    service, spec, root = attached
    hdf5(root / "002.h5")
    (root / "nested").mkdir()
    hdf5(root / "nested" / "002.h5")
    spec = spec.model_copy(update={"recursive": True})
    assert any(row["code"] == "DUPLICATE_FEATURE_ID" for row in service.preview(spec)["findings"])
    (root / "linked.h5").symlink_to(root / "002.h5")
    with pytest.raises(FilesystemError):
        service.preview(spec)


def test_same_size_and_mtime_replacement_invalidates_frozen_feature_binding(attached):
    service, spec, root = attached
    path = root / "001.A.h5"
    hdf5(path)
    original = path.stat()
    candidate = service.preview(spec)
    frozen = service.freeze(spec, candidate["previewHash"], "attach-original")
    replacement = root / "replacement.h5"
    hdf5(replacement)
    with h5py.File(replacement, "r+") as handle:
        handle["features"][0, 0] = 2
    assert replacement.stat().st_size == original.st_size
    os.utime(replacement, ns=(original.st_atime_ns, original.st_mtime_ns))
    replacement.replace(path)
    assert path.stat().st_mtime_ns == original.st_mtime_ns
    assert service.verify_binding(frozen)[0]["code"] == "FEATURE_SOURCE_CHANGED"
    with pytest.raises(StorageError) as error:
        service.freeze(spec, candidate["previewHash"], "new-operation")
    assert error.value.code == "PREVIEW_STALE"


def test_distinct_slide_names_hardlinked_to_one_feature_file_are_rejected(attached):
    service, spec, root = attached
    hdf5(root / "001.A.h5")
    os.link(root / "001.A.h5", root / "002.h5")
    result = service.preview(spec)
    assert not result["canFreeze"]
    assert any(item["code"] == "ALIASED_FEATURE_FILES" for item in result["findings"])


def test_binding_preflight_reports_unvalidated_files_when_time_budget_expires(
    attached, monkeypatch
):
    service, spec, root = attached
    hdf5(root / "001.A.h5")
    result = service.preview(spec)
    frozen = service.freeze(spec, result["previewHash"], "timed")
    ticks = iter([0, 31])
    monkeypatch.setattr("histopilot.application.features.time.monotonic", lambda: next(ticks))
    assert service.verify_binding(frozen)[0]["code"] == "FEATURE_SCAN_LIMIT"


def test_completed_attachment_retry_keeps_original_snapshot_after_source_changes(attached):
    service, spec, root = attached
    hdf5(root / "001.A.h5")
    preview = service.preview(spec)
    frozen = service.freeze(spec, preview["previewHash"], "completed-attach")
    hdf5(root / "001.A.h5", dimensions=8)
    assert service.freeze(spec, preview["previewHash"], "completed-attach") == frozen
    assert service.verify_binding(frozen)[0]["code"] == "FEATURE_SOURCE_CHANGED"
    with pytest.raises(StorageError) as caught:
        service.freeze(
            spec.model_copy(update={"encoderId": "different"}),
            preview["previewHash"],
            "completed-attach",
        )
    assert caught.value.code == "OPERATION_CONFLICT"


def test_source_change_at_publication_boundary_cannot_publish_stale_headers(attached, monkeypatch):
    service, spec, root = attached
    hdf5(root / "001.A.h5")
    preview = service.preview(spec)
    publish = service.store.publish_configuration

    def replace_then_publish(*args, **kwargs):
        hdf5(root / "001.A.h5", dimensions=8)
        return publish(*args, **kwargs)

    monkeypatch.setattr(service.store, "publish_configuration", replace_then_publish)
    with pytest.raises(StorageError) as caught:
        service.freeze(spec, preview["previewHash"], "raced-attach")
    assert caught.value.code == "PREVIEW_STALE"
    assert service.store.configuration_publication("raced-attach") is None


def test_a_feature_set_without_a_dataset_covers_the_whole_slide_store(attached, tmp_path):
    service, spec, root = attached
    hdf5(root / "001.A.h5")
    hdf5(root / "002.h5")
    # An encoded store legitimately holds slides no single frozen cohort claims.
    hdf5(root / "encoded-but-not-in-this-dataset.h5")
    scoped = service.preview(spec)
    assert scoped["summary"]["scope"] == "dataset"
    assert (scoped["summary"]["matchedSlides"], scoped["summary"]["orphanFiles"]) == (2, 1)
    store_wide = service.preview(spec.model_copy(update={"datasetId": None}))
    assert store_wide["summary"]["scope"] == "store"
    assert (store_wide["summary"]["matchedSlides"], store_wide["summary"]["orphanFiles"]) == (3, 0)
    assert store_wide["summary"]["missingSlides"] == 0
    assert store_wide["summary"]["slideCount"] == 3
    frozen = service.freeze(
        spec.model_copy(update={"datasetId": None}), store_wide["previewHash"], "store-wide"
    )
    assert frozen["manifest"]["datasetId"] is None
    assert service.verify_binding(frozen) == []


def test_a_slide_list_narrows_the_feature_folder_before_any_dataset(attached, tmp_path):
    """Source, then filter: the list decides the initial set and a dataset narrows it."""
    service, spec, root = attached
    hdf5(root / "001.A.h5")
    hdf5(root / "002.h5")
    hdf5(root / "encoded-but-not-in-this-dataset.h5")
    listing = tmp_path / "slides.csv"
    listing.write_text(
        "wsi,mpp\ncohort/002.svs,0.25\ncohort/encoded-but-not-in-this-dataset.svs,0.5\n"
    )
    listed = spec.model_copy(update={"datasetId": None, "slideListPath": str(listing)})
    preview = service.preview(listed)
    assert preview["summary"]["scope"] == "list"
    assert preview["summary"]["matchedSlides"] == 2
    assert preview["slideList"]["listedCount"] == 2
    assert preview["slideList"]["declaresMpp"] is True
    # The same list, now narrowed by the cohort: only slides in both survive.
    both = service.preview(spec.model_copy(update={"slideListPath": str(listing)}))
    assert both["summary"]["scope"] == "dataset"
    assert [item["slideId"] for item in both["files"]] == ["002"]
    assert both["summary"]["orphanFiles"] == 2


def test_an_unreadable_slide_list_is_refused(attached, tmp_path):
    service, spec, root = attached
    hdf5(root / "001.A.h5")
    listing = tmp_path / "slides.csv"
    listing.write_text("slide\n001.A\n")
    with pytest.raises(StorageError) as error:
        service.preview(spec.model_copy(update={"slideListPath": str(listing)}))
    assert error.value.code == "INVALID_SLIDE_LIST"


def test_uploaded_slide_list_selects_features_without_slide_files_or_dataset(attached):
    import base64

    service, original, root = attached
    hdf5(root / "001.A.h5")
    hdf5(root / "002.h5")
    spec = FeatureSpec(
        path=original.path,
        slideList={
            "filename": "offline-slides.csv",
            "contentBase64": base64.b64encode(b"wsi,mpp\noffline/001.A.svs,0.25\n").decode(),
        },
    )
    preview = service.preview(spec)
    assert preview["canFreeze"], preview["findings"]
    assert preview["summary"]["scope"] == "list"
    assert [item["slideId"] for item in preview["files"]] == ["001.A"]
    assert preview["slideList"]["filename"] == "offline-slides.csv"
    frozen = service.freeze(spec, preview["previewHash"], "uploaded-features")
    assert frozen["manifest"]["datasetId"] is None
    assert frozen["manifest"]["spec"]["slideList"] == spec.slideList.model_dump()


def test_legacy_feature_preview_and_freeze_retry_keep_their_identity(attached):
    from histopilot.application.features import _hash

    service, spec, root = attached
    for name in ("001.A.h5", "002.h5"):
        hdf5(root / name)
    # The exact pre-upload schema: no slideList field, including no null placeholder.
    legacy_spec = {
        "slideListPath": None,
        "datasetId": spec.datasetId,
        "path": str(root),
        "encoderId": None,
        "fileSuffix": ".h5",
        "idSuffix": "",
        "recursive": False,
        "layout": "auto",
        "coordinatesPath": None,
        "sourceExtractionJobId": None,
    }
    request = FeatureSpec.model_validate({**legacy_spec, "slideList": None})
    assert request.model_dump(mode="json") == legacy_spec
    assert json.loads(request.model_dump_json()) == legacy_spec
    preview = service.preview(request)
    assert preview["spec"] == legacy_spec
    inventory = [
        {"path": item["path"], "size": item["sizeBytes"], "mtime": item["mtimeNs"]}
        for item in preview["files"]
    ]
    legacy_preview_hash = _hash(
        {
            **{key: value for key, value in preview.items() if key != "previewHash"},
            "spec": legacy_spec,
            "inventory": inventory,
        }
    )
    assert preview["previewHash"] == legacy_preview_hash
    legacy_manifest = {
        "kind": "feature",
        "schemaVersion": 1,
        "datasetId": spec.datasetId,
        **{key: value for key, value in preview.items() if key != "canFreeze"},
        "spec": legacy_spec,
    }
    saved = service.store.publish_configuration(
        manifest=legacy_manifest, operation_id="legacy-feature-freeze"
    )
    replay = service.freeze(request, legacy_preview_hash, "legacy-feature-freeze")
    assert replay["id"] == saved["id"]
    assert service.store.list_configurations("feature") == [saved]
