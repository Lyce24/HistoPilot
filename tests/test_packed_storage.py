"""Small tensor fixtures for preservation, identity, and publication integrity."""

import copy
import hashlib
import json
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from histopilot.storage.packed import (
    PackedFeatureStore,
    PackedStoreError,
    PackingCancelled,
    build_pack,
    validate_features,
    validate_pack,
)


def make_configuration(root, *, dtype="float32", sidecar=False, ids=("slide A.svs", "Ünicode.01")):
    root.mkdir(parents=True)
    files = []
    arrays = {}
    for number, slide_id in enumerate(ids):
        features = (np.arange((number + 3) * 4).reshape(number + 3, 4) / 7).astype(dtype)
        coords = np.arange((number + 3) * 2).reshape(number + 3, 2).astype(np.int64)
        path = root / f"opaque-{number}.hdf5"
        coords_path = root / f"coords-{number}.h5" if sidecar else path
        with h5py.File(path, "w") as handle:
            ds = handle.create_dataset("features", data=features)
            ds.attrs["encoder"] = "uni_v2"
            ds.attrs["name"] = slide_id
            if not sidecar:
                handle.create_dataset("coords", data=coords)
        if sidecar:
            with h5py.File(coords_path, "w") as handle:
                handle.create_dataset("coords", data=coords)
        files.append(
            {
                "slideId": slide_id,
                "path": str(path),
                "coordinatePath": str(coords_path),
                "patchCount": len(features),
                "dimensions": 4,
                "dtype": dtype,
            }
        )
        arrays[slide_id] = features, coords
    return {
        "id": "feature-original",
        "contentHash": "binding-hash",
        "manifest": {
            "kind": "feature",
            "files": files,
            "provenance": [],
            "layout": {"featureDirectory": str(root), "encoderId": "uni_v2"},
        },
    }, arrays


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("sidecar", [False, True])
def test_roundtrip_preserves_rows_precision_coordinates_and_ids(tmp_path, dtype, sidecar):
    configuration, arrays = make_configuration(tmp_path / "source", dtype=dtype, sidecar=sidecar)
    progress = []
    manifest = build_pack(
        configuration, tmp_path / "pack", chunk_bytes=64, progress=progress.append
    )
    assert manifest["outputDtype"] == dtype
    assert manifest["preservesSourcePrecision"] is True
    assert manifest["preservesOriginalContainers"] is False
    assert manifest["validation"]["tensorValidationComplete"] is True
    assert manifest["validation"]["provenanceComplete"] is False
    assert manifest["validation"]["files"][0]["coordinateDtype"] == "int64"
    assert progress[-1]["stage"] == "publishing"
    assert validate_pack(tmp_path / "pack") == manifest
    with PackedFeatureStore(tmp_path / "pack") as store:
        assert set(store.slide_ids) == set(arrays)
        for slide, (features, coords) in arrays.items():
            assert store.read_features(slide).dtype == np.dtype(dtype)
            np.testing.assert_array_equal(store.read_features(slide), features)
            np.testing.assert_array_equal(store.read_coords(slide), coords)
            np.testing.assert_array_equal(
                store.read_features(slide, [2, 0, 2]), features[[2, 0, 2]]
            )
            assert store.read_features(slide, []).shape == (0, 4)
            with pytest.raises(PackedStoreError, match="outside"):
                store.read_features(slide, [len(features)])


def test_validation_independent_of_packing_and_source_hashes(tmp_path):
    configuration, _ = make_configuration(tmp_path / "source", sidecar=True)
    report = validate_features(configuration, chunk_bytes=31)
    assert report["valid"] and report["tensorValidationComplete"]
    assert report["slideCount"] == 2 and report["totalPatches"] == 7
    for item in report["files"]:
        assert item["sha256"] == hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest()
        coordinate = item["coordinateFile"]
        assert (
            coordinate["sha256"]
            == hashlib.sha256(Path(coordinate["path"]).read_bytes()).hexdigest()
        )
    assert not (tmp_path / "pack").exists()


def test_pack_reopens_after_relocation_and_removing_original_sources(tmp_path):
    configuration, arrays = make_configuration(tmp_path / "source")
    original = build_pack(configuration, tmp_path / "pack")
    shutil.move(tmp_path / "pack", tmp_path / "moved")
    shutil.rmtree(tmp_path / "source")
    assert validate_pack(tmp_path / "moved")["materializationId"] == original["materializationId"]
    with PackedFeatureStore(tmp_path / "moved") as store:
        np.testing.assert_array_equal(store.read_features("slide A.svs"), arrays["slide A.svs"][0])


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_source_rejected_without_published_output(tmp_path, value):
    configuration, _ = make_configuration(tmp_path / "source")
    with h5py.File(configuration["manifest"]["files"][0]["path"], "r+") as handle:
        handle["features"][0, 0] = value
    with pytest.raises(PackedStoreError, match="NaN or infinity"):
        build_pack(configuration, tmp_path / "pack", chunk_bytes=32)
    assert not (tmp_path / "pack").exists()
    assert not list(tmp_path.glob(".pack.packing-*"))


@pytest.mark.parametrize("value", [-1, 2**31])
def test_coordinate_overflow_and_negative_values_rejected(tmp_path, value):
    configuration, _ = make_configuration(tmp_path / "source", sidecar=True)
    with h5py.File(configuration["manifest"]["files"][0]["coordinatePath"], "r+") as handle:
        handle["coords"][0, 0] = value
    with pytest.raises(PackedStoreError, match="nonnegative"):
        build_pack(configuration, tmp_path / "pack")
    assert not (tmp_path / "pack").exists()


def test_native_validation_supports_coordinates_larger_than_packing_format(tmp_path):
    configuration, _ = make_configuration(tmp_path / "source")
    with h5py.File(configuration["manifest"]["files"][0]["path"], "r+") as handle:
        handle["coords"][0, 0] = 2**31
    report = validate_features(configuration)
    assert report["coordinateValidation"] == "nonnegative-integer-xy"
    assert report["coordinateDtypes"] == ["int64"]
    with pytest.raises(PackedStoreError, match="int32"):
        build_pack(configuration, tmp_path / "pack")


def test_fractional_coordinate_dtype_rejected(tmp_path):
    configuration, _ = make_configuration(tmp_path / "source")
    with h5py.File(configuration["manifest"]["files"][0]["path"], "r+") as handle:
        count = len(handle["coords"])
        del handle["coords"]
        handle.create_dataset("coords", data=np.zeros((count, 2), dtype=float) + 0.5)
    with pytest.raises(PackedStoreError, match="integer XY"):
        validate_features(configuration)


def test_float16_conversion_is_explicit_recorded_and_rejects_overflow(tmp_path):
    configuration, _ = make_configuration(tmp_path / "source")
    preserved = build_pack(configuration, tmp_path / "preserved")
    reduced = build_pack(configuration, tmp_path / "reduced", dtype="float16")
    assert preserved["sourceContentHash"] == reduced["sourceContentHash"]
    assert preserved["materializationId"] != reduced["materializationId"]
    assert reduced["conversion"]["changedValues"] > 0
    assert reduced["conversion"]["maxAbsoluteError"] > 0
    assert reduced["preservesSourcePrecision"] is False
    with h5py.File(configuration["manifest"]["files"][0]["path"], "r+") as handle:
        handle["features"][0, 0] = 70000
    with pytest.raises(PackedStoreError, match="overflow float16"):
        build_pack(configuration, tmp_path / "overflow", dtype="float16")
    assert not (tmp_path / "overflow").exists()


def test_identity_excludes_location_stamps_tag_and_binding_but_includes_selection(tmp_path):
    configuration, _ = make_configuration(tmp_path / "source")
    original = build_pack(configuration, tmp_path / "pack")
    relocated = copy.deepcopy(configuration)
    shutil.copytree(tmp_path / "source", tmp_path / "other-source")
    relocated["id"] = "a-new-binding"
    relocated["contentHash"] = "a-new-binding-hash"
    relocated["versionLabel"] = {"tag": "future"}
    relocated["manifest"]["layout"]["featureDirectory"] = str(tmp_path / "other-source")
    for entry in relocated["manifest"]["files"]:
        for key in ("path", "coordinatePath"):
            entry[key] = str(tmp_path / "other-source" / Path(entry[key]).name)
    relocated["manifest"]["files"].reverse()
    copied = build_pack(relocated, tmp_path / "copy-pack")
    assert original["sourceContentHash"] == copied["sourceContentHash"]
    assert original["materializationId"] == copied["materializationId"]
    relocated["manifest"]["files"].pop()
    subset = build_pack(relocated, tmp_path / "subset")
    assert subset["sourceContentHash"] != original["sourceContentHash"]
    assert subset["materializationId"] != original["materializationId"]


@pytest.mark.parametrize(
    "filename", ["features.bin", "coords.bin", "index.parquet", "meta.json", "manifest.json"]
)
def test_same_size_corruption_detected(tmp_path, filename):
    configuration, _ = make_configuration(tmp_path / "source")
    build_pack(configuration, tmp_path / "pack")
    path = tmp_path / "pack" / filename
    raw = bytearray(path.read_bytes())
    raw[len(raw) // 2] ^= 1
    path.write_bytes(raw)
    with pytest.raises((PackedStoreError, ValueError)):
        validate_pack(tmp_path / "pack")


def test_source_mutation_during_scan_aborts_publication(tmp_path):
    configuration, _ = make_configuration(tmp_path / "source")
    mutated = False

    def progress(update):
        nonlocal mutated
        if mutated or update["stage"] != "packing":
            return
        mutated = True
        source = Path(configuration["manifest"]["files"][0]["path"])
        with source.open("ab") as stream:
            stream.write(b"mutation")

    with pytest.raises(PackedStoreError, match="changed"):
        build_pack(configuration, tmp_path / "pack", progress=progress, chunk_bytes=64)
    assert not (tmp_path / "pack").exists()


def test_frozen_source_stamp_and_provenance_change_are_rejected(tmp_path):
    configuration, _ = make_configuration(tmp_path / "source")
    source = Path(configuration["manifest"]["files"][0]["path"])
    configuration["manifest"]["files"][0]["sizeBytes"] = source.stat().st_size - 1
    with pytest.raises(PackedStoreError, match="changed"):
        validate_features(configuration)
    del configuration["manifest"]["files"][0]["sizeBytes"]
    provenance = tmp_path / "source" / "config.json"
    provenance.write_text('{"encoder":"uni_v2"}')
    configuration["manifest"]["provenance"] = [{"path": str(provenance), "sha256": "outdated"}]
    with pytest.raises(PackedStoreError, match="provenance changed"):
        build_pack(configuration, tmp_path / "pack")


def test_cancel_cleans_staging_and_preserves_sources(tmp_path):
    configuration, _ = make_configuration(tmp_path / "source")
    should_cancel = False

    def progress(update):
        nonlocal should_cancel
        should_cancel = update["stage"] == "packing"

    with pytest.raises(PackingCancelled):
        build_pack(
            configuration,
            tmp_path / "pack",
            progress=progress,
            cancelled=lambda: should_cancel,
            chunk_bytes=32,
        )
    assert not (tmp_path / "pack").exists()
    assert not list(tmp_path.glob(".pack.packing-*"))
    assert all(Path(entry["path"]).exists() for entry in configuration["manifest"]["files"])


@pytest.mark.parametrize("destination", ["source", "source/pack", "."])
def test_source_overlap_rejected(tmp_path, destination):
    configuration, _ = make_configuration(tmp_path / "source")
    with pytest.raises(PackedStoreError, match="overlap"):
        build_pack(configuration, tmp_path / destination)
    assert all(Path(entry["path"]).exists() for entry in configuration["manifest"]["files"])


def test_existing_empty_directory_allowed_nonempty_never_overwritten(tmp_path):
    configuration, _ = make_configuration(tmp_path / "source")
    destination = tmp_path / "pack"
    destination.mkdir()
    build_pack(configuration, destination)
    manifest_bytes = (destination / "manifest.json").read_bytes()
    with pytest.raises(PackedStoreError, match="new or empty"):
        build_pack(configuration, destination)
    assert (destination / "manifest.json").read_bytes() == manifest_bytes


def test_external_links_duplicate_id_and_mixed_geometry_rejected(tmp_path):
    configuration, _ = make_configuration(tmp_path / "source")
    duplicate = copy.deepcopy(configuration)
    duplicate["manifest"]["files"][1]["slideId"] = duplicate["manifest"]["files"][0]["slideId"]
    with pytest.raises(PackedStoreError, match="duplicate"):
        validate_features(duplicate)
    for index, entry in enumerate(configuration["manifest"]["files"]):
        with h5py.File(entry["path"], "r+") as handle:
            handle["coords"].attrs["patch_size"] = 256 + index
    with pytest.raises(PackedStoreError, match="conflicting patch_size"):
        validate_features(configuration)
    with h5py.File(configuration["manifest"]["files"][0]["path"], "r+") as handle:
        del handle["features"]
        handle["features"] = h5py.ExternalLink("other.h5", "features")
    with pytest.raises(PackedStoreError, match="embedded HDF5"):
        validate_features(configuration)


def test_manifest_cannot_redirect_payload_outside_pack(tmp_path):
    configuration, _ = make_configuration(tmp_path / "source")
    build_pack(configuration, tmp_path / "pack")
    manifest_path = tmp_path / "pack" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["features.bin"]["path"] = "../source/opaque-0.hdf5"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(PackedStoreError, match="manifest checksum"):
        validate_pack(tmp_path / "pack")
    os.symlink(tmp_path / "pack", tmp_path / "linked-pack")
    with pytest.raises(PackedStoreError, match="symbolic"):
        validate_pack(tmp_path / "linked-pack")


def test_big_endian_source_is_preserved_as_explicit_little_endian_payload(tmp_path):
    configuration, arrays = make_configuration(tmp_path / "source", dtype=">f4")
    manifest = build_pack(configuration, tmp_path / "pack")
    assert manifest["endianness"] == "little"
    assert manifest["outputDtype"] == "float32"
    with PackedFeatureStore(tmp_path / "pack") as store:
        for slide, (features, _) in arrays.items():
            np.testing.assert_array_equal(store.read_features(slide), features)


def test_disk_readback_corruption_fails_before_publication(tmp_path, monkeypatch):
    from histopilot.storage import packed

    configuration, _ = make_configuration(tmp_path / "source")
    original = packed._file_evidence

    def corrupt_before_hash(path, *args):
        if path.name == "features.bin":
            with path.open("r+b") as stream:
                stream.write(b"bad!")
        return original(path, *args)

    monkeypatch.setattr(packed, "_file_evidence", corrupt_before_hash)
    with pytest.raises(PackedStoreError, match="readback differs"):
        build_pack(configuration, tmp_path / "pack")
    assert not (tmp_path / "pack").exists()


def test_durable_publication_failure_rolls_back_output(tmp_path, monkeypatch):
    from histopilot.storage import packed

    configuration, _ = make_configuration(tmp_path / "source")
    original = packed._fsync_dir

    def fail_parent_sync(path):
        if path == tmp_path:
            raise OSError("Synthetic fsync failure")
        return original(path)

    monkeypatch.setattr(packed, "_fsync_dir", fail_parent_sync)
    with pytest.raises(OSError, match="fsync"):
        build_pack(configuration, tmp_path / "pack")
    assert not (tmp_path / "pack").exists()
    assert not list(tmp_path.glob(".pack.packing-*"))


def test_source_extraction_evidence_survives_pack(tmp_path):
    configuration, _ = make_configuration(tmp_path / "source")
    snapshot = {"jobId": "extract-job", "snapshotHash": "binding", "runtime": {"version": "1"}}
    configuration["manifest"]["sourceExtraction"] = snapshot
    manifest = build_pack(configuration, tmp_path / "pack")
    assert manifest["sourceExtraction"] == snapshot
    assert manifest["validation"]["sourceExtraction"] == snapshot
    assert manifest["validation"]["provenanceComplete"] is False


@pytest.mark.parametrize("chunk_bytes", [0, -1, 1.5])
def test_invalid_chunk_budget_is_rejected(tmp_path, chunk_bytes):
    configuration, _ = make_configuration(tmp_path / "source")
    with pytest.raises(PackedStoreError, match="positive integer"):
        build_pack(configuration, tmp_path / "pack", chunk_bytes=chunk_bytes)
    assert not (tmp_path / "pack").exists()


def test_source_file_checksums_are_distinct_from_label_and_locator_free_identity(tmp_path):
    configuration, _ = make_configuration(tmp_path / "source")
    source = configuration["manifest"]["files"][0]["path"]
    with h5py.File(source, "r+") as handle:
        handle.attrs["label"] = "old-label"
        handle.attrs["savetodir"] = "/original/directory"
    before = validate_features(configuration)
    with h5py.File(source, "r+") as handle:
        handle.attrs["label"] = "changed-label"
        handle.attrs["savetodir"] = "/relocated/directory"
    after = validate_features(configuration)
    assert before["files"][0]["sha256"] != after["files"][0]["sha256"]
    assert before["sourceContentHash"] == after["sourceContentHash"]
