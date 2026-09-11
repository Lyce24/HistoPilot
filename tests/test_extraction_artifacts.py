"""Completion requires native artifact contents, not merely nonempty output filenames."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from histopilot.application.extraction_artifacts import inspect_outputs


@pytest.fixture
def outputs(tmp_path):
    root = tmp_path / "outputs"
    coords = root / "20x_256px_0px_overlap"
    for folder in (
        root / "contours_geojson",
        root / "contours",
        coords / "patches",
        coords / "features_uni_v1",
    ):
        folder.mkdir(parents=True)
    return {
        "outputPath": str(root),
        "spec": {
            "options": {
                "task": "all",
                "mag": 20,
                "patch_size": 256,
                "overlap": 0,
                "patch_encoder": "uni_v1",
            }
        },
        "slides": [{"name": "001.A", "slideId": "001.A"}],
    }


def paths(job):
    root = Path(job["outputPath"])
    coords = root / "20x_256px_0px_overlap"
    return (
        root / "contours_geojson" / "001.A.geojson",
        coords / "patches" / "001.A_patches.h5",
        coords / "features_uni_v1" / "001.A.h5",
    )


def coord_dataset(handle, count=3):
    value = handle.create_dataset("coords", data=np.ones((count, 2), dtype="int64"))
    value.attrs.update(
        {
            "name": "001.A",
            "patch_size": 256,
            "patch_size_level0": 512,
            "target_magnification": 20,
            "level0_magnification": 40,
            "overlap": 0,
        }
    )


def valid_outputs(job, *, count=3):
    geojson, coords, features = paths(job)
    geojson.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]],
                        },
                        "properties": {},
                    }
                ],
            }
        )
    )
    with h5py.File(coords, "w") as handle:
        coord_dataset(handle, count)
    with h5py.File(features, "w") as handle:
        coord_dataset(handle, count)
        value = handle.create_dataset("features", data=np.ones((count, 4), dtype="float32"))
        value.attrs.update({"name": "001.A", "encoder": "uni_v1"})


def test_all_requires_native_outputs_for_every_stage(outputs):
    valid_outputs(outputs)
    result = inspect_outputs(outputs)
    assert result["completedSlides"] == 1
    assert result["missingSlides"] == 0
    assert result["unvalidatedSlides"] == 0
    assert result["inspectionComplete"]
    assert result["findings"] == []
    assert result["featurePath"] == str(paths(outputs)[2].parent)
    paths(outputs)[0].unlink()
    result = inspect_outputs(outputs)
    assert result["completedSlides"] == 0
    assert result["missingSlides"] == 1


@pytest.mark.parametrize("stage,index", [("seg", 0), ("coords", 1), ("feat", 2)])
def test_nonempty_corrupt_outputs_do_not_count_as_success(outputs, stage, index):
    valid_outputs(outputs)
    outputs["spec"]["options"]["task"] = stage
    paths(outputs)[index].write_bytes(b"nonempty but not a scientific artifact")
    result = inspect_outputs(outputs)
    assert result["completedSlides"] == 0
    assert result["findings"][0]["severity"] == "error"


@pytest.mark.parametrize("stage,index", [("seg", 0), ("coords", 1), ("feat", 2)])
def test_upstream_locks_invalidate_otherwise_valid_outputs(outputs, stage, index):
    valid_outputs(outputs)
    outputs["spec"]["options"]["task"] = stage
    path = paths(outputs)[index]
    if stage == "seg":
        path = Path(outputs["outputPath"]) / "contours" / "001.A.jpg"
    Path(f"{path}.lock").write_text("worker lock")
    assert inspect_outputs(outputs)["completedSlides"] == 0


@pytest.mark.parametrize("stage,index", [("seg", 0), ("coords", 1), ("feat", 2)])
def test_symlink_outputs_are_not_followed(outputs, stage, index):
    valid_outputs(outputs)
    outputs["spec"]["options"]["task"] = stage
    path = paths(outputs)[index]
    target = path.with_suffix(".original")
    path.rename(target)
    path.symlink_to(target)
    assert inspect_outputs(outputs)["completedSlides"] == 0


def test_empty_tissue_and_coords_are_flagged_but_valid_stage_outputs(outputs):
    valid_outputs(outputs, count=0)
    paths(outputs)[0].write_text(json.dumps({"type": "FeatureCollection", "features": []}))
    for stage, code in (("seg", "EMPTY_TISSUE"), ("coords", "EMPTY_COORDINATES")):
        outputs["spec"]["options"]["task"] = stage
        result = inspect_outputs(outputs)
        assert result["completedSlides"] == 1
        assert result["findings"][0]["code"] == code
        assert result["findings"][0]["severity"] == "warning"
    outputs["spec"]["options"]["task"] = "all"
    assert inspect_outputs(outputs)["completedSlides"] == 0


def test_hdf5_external_links_are_rejected_without_following_them(outputs):
    valid_outputs(outputs)
    outputs["spec"]["options"]["task"] = "coords"
    coords = paths(outputs)[1]
    with h5py.File(coords, "w") as handle:
        handle["coords"] = h5py.ExternalLink("/not-accessible.h5", "coords")
    result = inspect_outputs(outputs)
    assert result["completedSlides"] == 0
    assert "embedded" in result["findings"][0]["message"]


@pytest.mark.parametrize(
    "attribute,value",
    [("name", "foreign"), ("target_magnification", 10), ("patch_size", 512), ("overlap", 20)],
)
def test_native_geometry_and_identity_must_match_run(outputs, attribute, value):
    valid_outputs(outputs)
    with h5py.File(paths(outputs)[1], "r+") as handle:
        handle["coords"].attrs[attribute] = value
    assert inspect_outputs(outputs)["completedSlides"] == 0


def test_coordinate_and_feature_patch_counts_must_agree(outputs):
    valid_outputs(outputs)
    with h5py.File(paths(outputs)[1], "w") as handle:
        coord_dataset(handle, 4)
    assert inspect_outputs(outputs)["completedSlides"] == 0


@pytest.mark.parametrize("task", ["all", "feat"])
def test_copied_feature_coordinates_must_match_the_requested_patch_rows(outputs, task):
    valid_outputs(outputs)
    outputs["spec"]["options"]["task"] = task
    with h5py.File(paths(outputs)[1], "r+") as handle:
        handle["coords"][0, 0] = 123
    result = inspect_outputs(outputs)
    assert result["completedSlides"] == 0
    assert "coordinate" in result["findings"][0]["message"].lower()


def test_feature_only_completion_still_validates_its_upstream_coordinates(outputs):
    valid_outputs(outputs)
    outputs["spec"]["options"]["task"] = "feat"
    paths(outputs)[1].write_bytes(b"corrupt predecessor with intact copied feature coords")
    assert inspect_outputs(outputs)["completedSlides"] == 0


@pytest.mark.parametrize("source_kind", ["slide", "checkpoint"])
def test_inputs_changed_during_extraction_cannot_claim_verified_completion(
    outputs, source_kind, tmp_path
):
    valid_outputs(outputs)
    source = tmp_path / ("slide.svs" if source_kind == "slide" else "encoder.pt")
    source.write_bytes(b"original input")
    info = source.stat()
    snapshot = {
        "path": str(source),
        "sizeBytes": info.st_size,
        "mtimeNs": info.st_mtime_ns,
        "ctimeNs": info.st_ctime_ns,
        "deviceId": info.st_dev,
        "inode": info.st_ino,
    }
    if source_kind == "slide":
        snapshot["size"] = snapshot.pop("sizeBytes")
        outputs["slides"][0].update(snapshot)
    else:
        outputs["inputFiles"] = [snapshot]
    assert inspect_outputs(outputs)["completedSlides"] == 1
    source.write_bytes(b"different input used during extraction")
    result = inspect_outputs(outputs)
    assert not result["inspectionComplete"]
    assert result["completedSlides"] == 0
    assert result["findings"][0]["code"] == "EXTRACTION_INPUT_CHANGED"


@pytest.mark.parametrize("shape", [(4,), (1, 4), (2, 4)])
def test_pooled_slide_features_do_not_require_one_embedding_per_coordinate(outputs, shape):
    valid_outputs(outputs)
    outputs["spec"]["options"]["slide_encoder"] = "mean-uni_v1"
    folder = paths(outputs)[2].parent.with_name("slide_features_mean-uni_v1")
    folder.mkdir()
    with h5py.File(folder / "001.A.h5", "w") as handle:
        coord_dataset(handle)
        value = handle.create_dataset("features", data=np.ones(shape, dtype="float32"))
        value.attrs["name"] = "001.A"
    result = inspect_outputs(outputs)
    assert result["completedSlides"] == 1
    assert result["featurePath"] == str(folder)


def test_output_layout_cannot_escape_its_job_root(outputs, tmp_path):
    outputs["outputLayout"] = {"featuresDir": str(tmp_path)}
    result = inspect_outputs(outputs)
    assert result["completedSlides"] == 0
    assert result["findings"][0]["code"] == "INVALID_OUTPUT_LAYOUT"


def test_time_budget_reports_unvalidated_slides(outputs, monkeypatch):
    valid_outputs(outputs)
    ticks = iter([0, 31])
    monkeypatch.setattr(
        "histopilot.application.extraction_artifacts.time.monotonic", lambda: next(ticks)
    )
    result = inspect_outputs(outputs)
    assert result["completedSlides"] == 0
    assert result["missingSlides"] == 0
    assert result["unvalidatedSlides"] == 1
    assert not result["inspectionComplete"]
    assert result["findings"][0]["code"] == "ARTIFACT_SCAN_LIMIT"


def test_invalid_geojson_polygon_is_rejected(outputs):
    valid_outputs(outputs)
    paths(outputs)[0].write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[0, 0], [1, 0], [1, 1], [2, 0]]],
                        },
                        "properties": {},
                    }
                ],
            }
        )
    )
    assert inspect_outputs(outputs)["completedSlides"] == 0
