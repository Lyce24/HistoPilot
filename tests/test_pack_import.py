"""An imported pack must match actual source tensors and remain unchanged."""

import copy
import json
import shutil

import h5py
import numpy as np
import pytest

from histopilot.storage.pack_import import (
    inspect_existing_pack,
    pack_file_stamps,
    verify_existing_pack,
)
from histopilot.storage.packed import PackedStoreError, PackingCancelled, build_pack


def configuration(root, *, number=2, dtype="float32", delta=0):
    root.mkdir()
    files = []
    for i in range(number):
        identity = f"slide.{i:03d}"
        path = root / f"{identity}.hdf5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("features", data=np.arange(12, dtype=dtype).reshape(3, 4) + delta)
            handle.create_dataset("coords", data=np.arange(6, dtype="int64").reshape(3, 2))
        files.append(
            {
                "slideId": identity,
                "path": str(path),
                "patchCount": 3,
                "dimensions": 4,
                "dtype": dtype,
            }
        )
    return {
        "id": "configuration-original",
        "contentHash": "original",
        "manifest": {
            "kind": "feature",
            "files": files,
            "provenance": [],
            "layout": {"featureDirectory": str(root)},
        },
    }


def test_relocated_pack_rebinds_current_feature_version_without_writing(tmp_path):
    original = configuration(tmp_path / "source")
    manifest = build_pack(original, tmp_path / "pack")
    shutil.move(tmp_path / "pack", tmp_path / "relocated")
    current = copy.deepcopy(original)
    current.update(id="configuration-current", contentHash="new-binding")
    path = tmp_path / "relocated"
    before = {item.name: item.read_bytes() for item in path.iterdir()}
    stamps = pack_file_stamps(path)
    inspection = inspect_existing_pack(current, path)
    assert inspection["matchesFeatures"]
    assert inspection["summary"]["expectedFeatureBytes"] == 96
    result = verify_existing_pack(current, path, expected_stamps=stamps, chunk_bytes=32)
    assert result["featureSetId"] == current["id"]
    assert result["sourceFeatureSetId"] == original["id"]
    assert result["materializationId"] == manifest["materializationId"]
    assert result["id"] != result["materializationId"]
    assert result["verification"] == "exact-source-values"
    assert result["packStamps"] == pack_file_stamps(path) == stamps
    assert before == {item.name: item.read_bytes() for item in path.iterdir()}


def test_counts_compare_every_slide_and_bound_warning_examples(tmp_path):
    source = configuration(tmp_path / "source", number=30)
    build_pack(source, tmp_path / "pack")
    changed = copy.deepcopy(source)
    for item in changed["manifest"]["files"][:29]:
        item["patchCount"] += 1
    result = inspect_existing_pack(changed, tmp_path / "pack")
    assert not result["matchesFeatures"]
    assert result["summary"]["mismatchedSlideCount"] == 29
    assert len(result["summary"]["mismatchedSlides"]) == 20
    assert result["summary"]["sourcePatchCount"] == 119
    assert result["summary"]["totalPatches"] == 90
    assert result["findings"][0]["severity"] == "warning"


def test_same_count_different_extraction_passes_headers_but_fails_values(tmp_path):
    source = configuration(tmp_path / "source")
    other = configuration(tmp_path / "other", delta=1)
    build_pack(other, tmp_path / "pack")
    assert inspect_existing_pack(source, tmp_path / "pack")["matchesFeatures"]
    with pytest.raises(PackedStoreError, match="feature values differ.*matching shape"):
        verify_existing_pack(source, tmp_path / "pack", chunk_bytes=32)


def test_same_count_different_coordinates_fail_full_comparison(tmp_path):
    source = configuration(tmp_path / "source")
    other = configuration(tmp_path / "other")
    with h5py.File(other["manifest"]["files"][0]["path"], "r+") as handle:
        handle["coords"][0, 0] = 99
    build_pack(other, tmp_path / "pack")
    with pytest.raises(PackedStoreError, match="coordinates differ"):
        verify_existing_pack(source, tmp_path / "pack", chunk_bytes=32)


def test_legacy_oceanpath_pack_verifies_without_being_rewritten(tmp_path):
    source = configuration(tmp_path / "source")
    path = tmp_path / "legacy"
    build_pack(source, path)
    (path / "manifest.json").unlink()
    (path / "checksums.json").unlink()
    before = {item.name: item.read_bytes() for item in path.iterdir()}
    result = verify_existing_pack(source, path, chunk_bytes=32)
    assert result["formatVariant"] == "oceanpath-legacy"
    assert result["verification"] == "exact-source-values"
    assert before == {item.name: item.read_bytes() for item in path.iterdir()}


@pytest.mark.parametrize("missing", ["manifest.json", "checksums.json"])
def test_partial_histopilot_evidence_cannot_be_downgraded_to_legacy(tmp_path, missing):
    source = configuration(tmp_path / "source")
    path = tmp_path / "pack"
    build_pack(source, path)
    (path / missing).unlink()
    with pytest.raises(PackedStoreError, match="require both manifest.json and checksums.json"):
        inspect_existing_pack(source, path)
    with pytest.raises(PackedStoreError, match="folder is incomplete"):
        verify_existing_pack(source, path)


def test_a_reduced_precision_pack_is_verified_against_the_cast_source(tmp_path):
    """Float16 storage of float32 features is a faithful pack at its own declared precision."""
    source = configuration(tmp_path / "source")
    path = tmp_path / "pack"
    build_pack(source, path, dtype="float16")
    result = inspect_existing_pack(source, path)
    assert result["matchesFeatures"], result["findings"]
    assert result["summary"]["precision"] == "reduced"
    assert (result["summary"]["sourceDtype"], result["summary"]["outputDtype"]) == (
        "float32",
        "float16",
    )
    artifact = verify_existing_pack(source, path)
    assert artifact["verification"] == "exact-cast-source-values"
    assert artifact["preservesSourcePrecision"] is False


def test_a_reduced_pack_still_fails_on_any_value_that_is_not_the_cast_source(tmp_path):
    source = configuration(tmp_path / "source")
    path = tmp_path / "pack"
    build_pack(source, path, dtype="float16")
    with (path / "features.bin").open("r+b") as stream:
        stream.write(np.array([123], dtype="<f2").tobytes())
    with pytest.raises(PackedStoreError, match="differ from the selected source"):
        verify_existing_pack(source, path)


def test_a_pack_claiming_higher_precision_than_its_source_is_refused(tmp_path):
    """Upcasting asserts detail the source never had, even when every value round-trips."""
    source = configuration(tmp_path / "source", dtype="float16")
    path = tmp_path / "pack"
    build_pack(source, path)
    meta = json.loads((path / "meta.json").read_text())
    assert meta["feat_dtype"] == "float16"
    # Widen the payload so its declared length stays consistent: a genuine float32 pack of
    # float16 sources, losing nothing, and still not a faithful representation of them.
    values = np.frombuffer((path / "features.bin").read_bytes(), dtype="<f2")
    (path / "features.bin").write_bytes(values.astype("<f4").tobytes())
    (path / "meta.json").write_text(json.dumps({**meta, "feat_dtype": "float32"}))
    (path / "manifest.json").unlink()
    (path / "checksums.json").unlink()
    result = inspect_existing_pack(source, path)
    assert not result["matchesFeatures"]
    assert any(item["code"] == "PACK_DTYPE_MISMATCH" for item in result["findings"])
    assert result["summary"]["precision"] == "exact"


def test_changed_pack_since_preview_is_rejected(tmp_path):
    source = configuration(tmp_path / "source")
    path = tmp_path / "pack"
    build_pack(source, path)
    before = pack_file_stamps(path)
    with (path / "features.bin").open("r+b") as stream:
        stream.write(np.array([123], dtype="<f4").tobytes())
    with pytest.raises(PackedStoreError, match="changed since.*preview"):
        verify_existing_pack(source, path, expected_stamps=before)


def test_truncated_payload_and_invalid_coordinate_metadata_are_rejected(tmp_path):
    source = configuration(tmp_path / "source")
    path = tmp_path / "pack"
    build_pack(source, path)
    with (path / "features.bin").open("r+b") as stream:
        stream.truncate(4)
    with pytest.raises(PackedStoreError, match="byte length"):
        inspect_existing_pack(source, path)
    meta = json.loads((path / "meta.json").read_text())
    meta["has_coords"] = "true"
    (path / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(PackedStoreError, match="coordinate pair"):
        inspect_existing_pack(source, path)


def test_cancellation_leaves_imported_pack_unchanged(tmp_path):
    source = configuration(tmp_path / "source")
    path = tmp_path / "pack"
    build_pack(source, path)
    stamps = pack_file_stamps(path)
    with pytest.raises(PackingCancelled):
        verify_existing_pack(source, path, cancelled=lambda: True)
    assert pack_file_stamps(path) == stamps


@pytest.mark.parametrize("payload", ["features", "coords"])
def test_legacy_pack_requires_every_row_to_match_not_only_headers(tmp_path, payload):
    source = configuration(tmp_path / "source")
    path = tmp_path / "legacy"
    build_pack(source, path)
    (path / "manifest.json").unlink()
    (path / "checksums.json").unlink()
    dtype = "<f4" if payload == "features" else "<i4"
    with (path / f"{payload}.bin").open("r+b") as stream:
        # Corrupt the final value, past the first slide and first comparison chunk.
        stream.seek(-4, 2)
        stream.write(np.array([123], dtype=dtype).tobytes())
    assert inspect_existing_pack(source, path)["matchesFeatures"]
    with pytest.raises(PackedStoreError, match="values differ|coordinates differ"):
        verify_existing_pack(source, path, chunk_bytes=16)


def test_slide_membership_mismatch_rejected_even_with_equal_totals(tmp_path):
    source = configuration(tmp_path / "source")
    path = tmp_path / "pack"
    build_pack(source, path)
    selected = copy.deepcopy(source)
    selected["manifest"]["files"][0]["slideId"] = "another-slide"
    result = inspect_existing_pack(selected, path)
    assert not result["matchesFeatures"]
    assert result["summary"]["sourcePatchCount"] == result["summary"]["totalPatches"]
    assert result["summary"]["missingSlides"] == ["another-slide"]
    assert result["summary"]["extraSlides"] == ["slide.000"]
    with pytest.raises(PackedStoreError, match="membership differs"):
        verify_existing_pack(selected, path)


@pytest.mark.parametrize("changed_input", ["source", "features.bin", "index.parquet"])
def test_inputs_changed_during_full_comparison_cannot_be_registered(tmp_path, changed_input):
    import os

    source = configuration(tmp_path / "source")
    path = tmp_path / "pack"
    build_pack(source, path)
    changed = False

    def progress(value):
        nonlocal changed
        if value["stage"] != "comparing-pack" or changed:
            return
        changed = True
        target = (
            tmp_path / "source" / "slide.000.hdf5"
            if changed_input == "source"
            else path / changed_input
        )
        stamp = target.stat()
        os.utime(target, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1))

    with pytest.raises(PackedStoreError, match="changed"):
        verify_existing_pack(source, path, progress=progress, chunk_bytes=16)
    assert changed


def test_pack_manifest_checksum_still_required_when_payload_matches_current_source(tmp_path):
    source = configuration(tmp_path / "source")
    path = tmp_path / "pack"
    build_pack(source, path)
    # Matching data alone cannot rehabilitate a pack whose manifest records old data.
    with h5py.File(source["manifest"]["files"][0]["path"], "r+") as handle:
        handle["features"][0, 0] = 123
    with (path / "features.bin").open("r+b") as stream:
        stream.write(np.array([123], dtype="<f4").tobytes())
    assert inspect_existing_pack(source, path)["matchesFeatures"]
    with pytest.raises(PackedStoreError, match="checksum differs from the pack manifest"):
        verify_existing_pack(source, path, chunk_bytes=16)
