"""Exact packed row offsets, geometry, encoder and immutable evidence for attention."""

import copy
import runpy
import shutil
from pathlib import Path

import numpy as np
import pytest

from histopilot.storage.attention_inputs import inspect_inputs
from histopilot.storage.pack_import import pack_file_stamps, verify_existing_pack
from histopilot.storage.packed import build_pack

_support = runpy.run_path(str(Path(__file__).with_name("test_packed_storage.py")))


def pack_fixture(tmp_path, *, dtype="float32", legacy=False):
    import h5py

    config, arrays = _support["make_configuration"](
        tmp_path / "source", dtype=dtype, ids=("case A", "nested.case-B")
    )
    for entry in config["manifest"]["files"]:
        entry["coordinateSpace"] = "level0"
        with h5py.File(entry["path"], "a") as handle:
            handle["coords"].attrs["patch_size_level0"] = 16
    path = tmp_path / "pack"
    manifest = build_pack(config, path)
    artifact = {**manifest, "outputPath": str(path), "packStamps": pack_file_stamps(path)}
    if legacy:
        (path / "manifest.json").unlink()
        (path / "checksums.json").unlink()
        artifact = verify_existing_pack(config, path)
    contract = {"encoderId": "uni_v2", "dimensions": 4, "dtype": dtype}
    selections = [
        {
            "sourceFormat": "packed",
            "slideId": "view-" + str(index),
            "slidePath": str(tmp_path / f"slide-{index}.png"),
            "packPath": str(path),
            "packSlideId": identity,
            "packEvidence": artifact,
            "coordinateSpace": "level0",
            "confirmRowAlignment": True,
        }
        for index, identity in enumerate(arrays)
    ]
    slide = {"width": 64, "height": 64, "levelDownsamples": [1.0]}
    return config, path, arrays, contract, selections, slide


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("legacy", [False, True])
def test_packed_attention_reads_exact_slide_offsets_without_original_hdf5(tmp_path, dtype, legacy):
    config, path, arrays, contract, selections, slide = pack_fixture(
        tmp_path, dtype=dtype, legacy=legacy
    )
    shutil.rmtree(Path(config["manifest"]["files"][0]["path"]).parent)
    for selection in reversed(selections):
        evidence, features, coords = inspect_inputs(selection, slide, contract, load=True)
        expected_features, expected_coords = arrays[selection["packSlideId"]]
        assert np.array_equal(features, expected_features.astype(np.float32))
        assert np.array_equal(coords, expected_coords)
        assert evidence["alignment"] == "packed_verified"
        assert evidence["patchWidthLevel0"] == evidence["patchHeightLevel0"] == 16
        assert len(evidence["packEvidence"]["validation"]["files"]) == 1
        assert set(evidence["packEvidence"]["payloadHashes"]) == {"features.bin", "coords.bin"}
        assert evidence["packedRow"]["offset"] == (0 if selection is selections[0] else 3)
        assert set(evidence["sourceStamps"]) == {
            str(path / name) for name in pack_file_stamps(path)
        }
        # The frozen compact evidence can recreate identical rows in a worker.
        repeated = inspect_inputs({**selection, **evidence}, slide, contract)
        assert repeated == evidence


def test_native_standalone_pack_authenticates_payloads_and_metadata(tmp_path):
    _, path, _, contract, selections, slide = pack_fixture(tmp_path)
    selection = {key: value for key, value in selections[0].items() if key != "packEvidence"}
    assert inspect_inputs(selection, slide, contract)["patchCount"] == 3
    with (path / "features.bin").open("r+b") as stream:
        stream.write(np.array([99.0], dtype="<f4").tobytes())
    with pytest.raises(ValueError, match="checksum"):
        inspect_inputs(selection, slide, contract)


@pytest.mark.parametrize("filename", ["features.bin", "coords.bin", "meta.json", "index.parquet"])
def test_changed_pack_files_fail_before_attention(tmp_path, filename):
    _, path, _, contract, selections, slide = pack_fixture(tmp_path)
    selection = {**selections[0], **inspect_inputs(selections[0], slide, contract)}
    file = path / filename
    with file.open("r+b") as stream:
        first = stream.read(1)
        stream.seek(0)
        stream.write(bytes([first[0] ^ 1]))
    with pytest.raises((ValueError, OSError)):
        inspect_inputs(selection, slide, contract)


def test_selected_tensor_hashes_reject_changed_features_or_reordered_coords(tmp_path):
    _, path, _, contract, selections, slide = pack_fixture(tmp_path)
    selection = {**selections[1], **inspect_inputs(selections[1], slide, contract)}
    coords = np.memmap(path / "coords.bin", dtype="<i4", mode="r+", shape=(7, 2))
    coords[3:7] = coords[3:7].copy()[::-1]
    coords.flush()
    coords._mmap.close()
    # Even a caller replacing only freshness stamps cannot replace tensor identity.
    selection["packEvidence"]["packStamps"] = pack_file_stamps(path)
    with pytest.raises(ValueError, match="coordinates differ"):
        inspect_inputs(selection, slide, contract)


@pytest.mark.parametrize(
    "change,match",
    [
        ("encoder", "encoder"),
        ("dimensions", "dimensions"),
        ("dtype", "dtype"),
        ("missing", "no exact row"),
        ("geometry", "geometry"),
        ("coord_bounds", "inside"),
        ("separate_coords", "coords.bin"),
    ],
)
def test_packed_contract_mismatches_fail_closed(tmp_path, change, match):
    _, _, _, contract, selections, slide = pack_fixture(tmp_path)
    selection = selections[0]
    if change == "encoder":
        contract["encoderId"] = "unrelated-encoder"
    elif change == "dimensions":
        contract["dimensions"] = 5
    elif change == "dtype":
        contract["dtype"] = "float16"
    elif change == "missing":
        selection["packSlideId"] = "not-in-pack"
    elif change == "geometry":
        selection.update(patchWidthLevel0=32.0, patchHeightLevel0=32.0)
    elif change == "coord_bounds":
        slide["width"] = 1
    else:
        selection["coordinatesPath"] = "/unused.h5"
    with pytest.raises(ValueError, match=match):
        inspect_inputs(selection, slide, contract)


def test_legacy_pack_requires_registered_encoder_and_source_verification(tmp_path):
    _, _, _, contract, selections, slide = pack_fixture(tmp_path, legacy=True)
    selection = copy.deepcopy(selections[0])
    selection.pop("packEvidence")
    with pytest.raises(ValueError, match="registered"):
        inspect_inputs(selection, slide, contract)
    selection = copy.deepcopy(selections[0])
    selection["packEvidence"]["validation"]["semanticIdentity"]["encoderId"] = "changed"
    with pytest.raises(ValueError, match="encoder"):
        inspect_inputs(selection, slide, contract)


def test_pack_geometry_can_be_provided_explicitly_and_time_budget_is_observed(tmp_path):
    _, _, _, contract, selections, slide = pack_fixture(tmp_path)
    with pytest.raises(ValueError, match="inspection budget"):
        inspect_inputs(selections[0], slide, contract, deadline=0)


def test_precision_reduced_native_pack_authenticates_actual_values(tmp_path):
    config, _, arrays, contract, selections, slide = pack_fixture(tmp_path)
    converted_path = tmp_path / "float16-pack"
    build_pack(config, converted_path, dtype="float16")
    selection = {key: value for key, value in selections[0].items() if key != "packEvidence"}
    selection["packPath"] = str(converted_path)
    contract["dtype"] = "float16"
    evidence, features, _ = inspect_inputs(selection, slide, contract, load=True)
    assert evidence["dtype"] == "float16"
    assert np.array_equal(
        features, arrays[selection["packSlideId"]][0].astype(np.float16).astype(np.float32)
    )


def test_caller_cannot_mutate_a_cached_native_manifest_via_returned_proof(tmp_path):
    _, _, _, contract, selections, slide = pack_fixture(tmp_path)
    selection = {key: value for key, value in selections[0].items() if key != "packEvidence"}
    evidence = inspect_inputs(selection, slide, contract)
    evidence["packEvidence"]["validation"]["files"][0]["patchCount"] = 999
    assert inspect_inputs(selection, slide, contract)["patchCount"] == 3
