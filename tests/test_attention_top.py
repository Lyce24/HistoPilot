"""Whole-slide ranking and authenticated crops from the original frozen slide."""

import hashlib
import io
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from test_interpretation import save
from test_interpretation import study as study

from histopilot.storage.project_lock import StorageError
from histopilot.viewer.attention_arrays import attention_page, attention_top
from histopilot.workers.packing_process import write_json
from histopilot.workers.training_process import read_json

Image = pytest.importorskip("PIL.Image")


def receipt(path):
    content = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def array_case(tmp_path, weights):
    count = len(weights)
    values = np.column_stack(
        (np.arange(count) % 1000, np.arange(count) // 1000, weights, np.full(count, 0.5))
    ).astype("<f8")
    path = tmp_path / "attention.npy"
    np.save(path, values)
    return (
        path,
        receipt(path),
        {
            "patchCount": count,
            "width": 1000,
            "height": max(1, (count + 999) // 1000),
            "patchWidthLevel0": 1,
            "patchHeightLevel0": 1,
        },
    )


def complete(study, *, legacy=False, footprint=(100, 100), coords=None):
    service, selection, _ = study
    if coords is None:
        coords = [[0, 0], [100, 0], [290, 190]]
    # A coordinate gradient makes accidental thumbnail/attention-overlay crops
    # distinguishable from the real source pixels.
    pixels = np.zeros((200, 300, 3), dtype=np.uint8)
    pixels[:, :, 0] = np.arange(300, dtype=np.uint16) % 256
    pixels[:, :, 1] = np.arange(200, dtype=np.uint8)[:, None]
    pixels[:, :, 2] = 91
    Image.fromarray(pixels).save(selection.slides[0].slidePath)
    with h5py.File(selection.slides[0].featurePath, "a") as handle:
        handle["coords"][:] = coords
        del handle["coords"].attrs["patch_size_level0"]
        handle["coords"].attrs["patch_width_level0"] = footprint[0]
        handle["coords"].attrs["patch_height_level0"] = footprint[1]
    document, _ = save(study)
    identity = document["id"]
    service.launch(identity, "launch")
    folder = service.jobs.folder(identity)
    artifacts = {}
    weights = ([0.1, 0.3, 0.6], [0.6, 0.3, 0.1])
    for member, suffix in enumerate(("", "-member-0")):
        path = folder / f"slide-0{suffix}.{'json' if legacy else 'npy'}"
        if legacy:
            write_json(
                path,
                {
                    "slideId": "independent",
                    "patchCount": 3,
                    "probabilities": [0.6, 0.4],
                    "patches": [
                        {"index": i, "x": xy[0], "y": xy[1], "weight": weight, "percentile": 0.5}
                        for i, (xy, weight) in enumerate(zip(coords, weights[member], strict=True))
                    ],
                },
            )
        else:
            np.save(path, np.column_stack((coords, weights[member], [0.5] * 3)).astype("<f8"))
        artifacts[path.name] = receipt(path)
    result = {
        "runId": identity,
        "state": "succeeded",
        "artifacts": artifacts,
        "slides": [
            {
                "slideId": "independent",
                "patchCount": 3,
                "probabilities": [0.6, 0.4],
                "attentionArray": "slide-0.npy",
                "members": [
                    {
                        "index": 0,
                        "probabilities": [0.6, 0.4],
                        "attentionArray": "slide-0-member-0.npy",
                    }
                ],
            }
        ],
    }
    update_result(service, identity, result)
    return service, identity, document, pixels


def update_result(service, identity, result):
    folder = service.jobs.folder(identity)
    state = read_json(folder / "state.json")
    write_json(folder / "state.json", {**state, "status": "completed", "result": result})
    write_json(folder / "result.json", result)


def test_top_scans_beyond_display_limit_and_viewport_with_global_tie_order(tmp_path):
    weights = np.full(131080, 0.000001)
    indices = [17, 65535, 65536, 100001, 131079]
    weights[indices] = [0.01, 0.02, 0.02, 0.03, 0.03]
    path, expected, slide = array_case(tmp_path, weights)
    page = attention_page(path, expected, slide, offset=0, limit=10, region=(0, 0, 1, 1))
    assert [row["index"] for row in page["patches"]] == [0]
    top = attention_top(path, expected, slide, limit=10)
    assert [row["index"] for row in top["patches"]] == [
        100001,
        131079,
        65535,
        65536,
        17,
        0,
        1,
        2,
        3,
        4,
    ]
    assert [row["rank"] for row in top["patches"]] == list(range(1, 11))
    assert top["scope"] == "whole_slide" and top["total"] == len(weights) and top["returned"] == 10
    assert top["coordinateBounds"] == {"x": 0, "y": 0, "width": 1000, "height": 132}


@pytest.mark.parametrize("legacy", [False, True])
def test_coordinate_coverage_includes_low_attention_rows_and_clips_fractional_footprints(
    study, legacy
):
    service, identity, _, _ = complete(
        study,
        legacy=legacy,
        footprint=(31.5, 47.25),
        coords=[[20, 10], [290, 190], [100, 70]],
    )
    expected = {"x": 20, "y": 10, "width": 280, "height": 190}
    highest = service.top_attention(identity, "independent", limit=1)
    assert highest["patches"][0]["index"] == 2
    assert highest["coordinateBounds"] == expected
    highest["coordinateBounds"]["x"] = 999
    for member in ("mean", "0"):
        for count in (1, 20):
            assert (
                service.top_attention(identity, "independent", member=member, limit=count)[
                    "coordinateBounds"
                ]
                == expected
            )


@pytest.mark.parametrize("count", [1, 3, 65560])
def test_equal_weights_small_slides_and_top_twenty_are_deterministic(tmp_path, count):
    path, expected, slide = array_case(tmp_path, np.full(count, 1 / count))
    top = attention_top(path, expected, slide, limit=20)
    assert top["returned"] == min(count, 20)
    assert [row["index"] for row in top["patches"]] == list(range(min(count, 20)))


@pytest.mark.parametrize(
    "column,value", [(0, 0.5), (0, -1), (1, 99999), (2, np.nan), (2, 1.01), (3, -0.1)]
)
def test_ranking_rejects_invalid_values_even_outside_winning_rows(tmp_path, column, value):
    path, _, slide = array_case(tmp_path, np.full(70000, 1 / 70000))
    values = np.load(path)
    values[-1, column] = value
    np.save(path, values)
    with pytest.raises(StorageError, match="invalid"):
        attention_top(path, receipt(path), slide, limit=10)


def test_verified_hash_is_shared_between_top_and_index_reads_and_changes_fail(
    tmp_path, monkeypatch
):
    from histopilot.viewer import attention_arrays

    path, expected, slide = array_case(tmp_path, [0.1, 0.3, 0.6])
    attention_top(path, expected, slide, limit=10)
    hashing = attention_arrays.hashlib.sha256
    monkeypatch.setattr(
        attention_arrays.hashlib, "sha256", lambda: pytest.fail("Duplicate full hash")
    )
    assert attention_page(path, expected, slide, offset=2, limit=1)["patches"][0]["index"] == 2
    monkeypatch.setattr(attention_arrays.hashlib, "sha256", hashing)
    values = np.load(path)
    values[2, 2] = 0.7
    np.save(path, values)
    with pytest.raises(StorageError, match="checksum changed"):
        attention_top(path, expected, slide, limit=10)


@pytest.mark.parametrize("legacy", [False, True])
def test_service_ranks_members_globally_and_crops_exact_original_edge_pixels(study, legacy):
    service, identity, _, pixels = complete(study, legacy=legacy)
    top = service.top_attention(identity, "independent", limit=20)
    assert [row["index"] for row in top["patches"]] == [2, 1, 0]
    assert top["returned"] == top["patchCount"] == top["total"] == 3
    assert top["patchWidthLevel0"] == 100 and top["classOrder"] == ["yes", "no"]
    assert [
        row["index"]
        for row in service.top_attention(identity, "independent", member="0")["patches"]
    ] == [0, 1, 2]
    for patch_index, box in [(2, (290, 190, 300, 200)), (1, (100, 0, 200, 100))]:
        with Image.open(
            io.BytesIO(service.patch_image(identity, "independent", patch_index))
        ) as image:
            x, y, right, bottom = box
            assert np.array_equal(np.asarray(image), pixels[y:bottom, x:right])
    # The selected index can be any patch, regardless of top-list membership.
    assert service.top_attention(identity, "independent", limit=1)["patches"][0]["index"] == 2
    assert service.patch_image(identity, "independent", 0, member="0")


@pytest.mark.parametrize("legacy", [False, True])
def test_fractional_frozen_footprint_is_rendered_exactly_and_clipped(study, monkeypatch, legacy):
    service, identity, _, _ = complete(study, legacy=legacy, footprint=(100.5, 83.25))
    original = Image.Image.resize
    boxes = []

    def resize(self, size, resample=None, box=None, **kwargs):
        boxes.append((size, box))
        return original(self, size, resample=resample, box=box, **kwargs)

    monkeypatch.setattr(Image.Image, "resize", resize)
    service.patch_image(identity, "independent", 1)
    service.patch_image(identity, "independent", 2)
    assert boxes == [((100, 83), (0, 0, 100.5, 83.25)), ((10, 10), (0, 0, 10, 10))]


@pytest.mark.parametrize("legacy", [False, True])
def test_crop_rejects_changed_source_or_attention_and_unknown_indices(study, legacy):
    service, identity, document, _ = complete(study, legacy=legacy)
    with pytest.raises(StorageError) as missing:
        service.patch_image(identity, "independent", 3)
    assert missing.value.code == "INTERPRETATION_PATCH_NOT_FOUND"
    with pytest.raises(StorageError) as member:
        service.top_attention(identity, "independent", member="1")
    assert member.value.code == "INTERPRETATION_MEMBER_INVALID"
    source = document["manifest"]["slides"][0]["slidePath"]
    Image.new("RGB", (300, 200)).save(source)
    with pytest.raises(StorageError, match="changed"):
        service.patch_image(identity, "independent", 1)
    artifact = service.jobs.folder(identity) / ("slide-0.json" if legacy else "slide-0.npy")
    with artifact.open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(StorageError, match="changed"):
        service.top_attention(identity, "independent")
    with pytest.raises(StorageError, match="changed"):
        service.patch_image(identity, "independent", 1)


@pytest.mark.parametrize(
    "field,value", [("index", 0), ("x", 0.25), ("y", 200), ("weight", float("nan"))]
)
def test_legacy_indices_and_values_are_validated_even_with_matching_receipt(study, field, value):
    service, identity, _, _ = complete(study, legacy=True)
    path = service.jobs.folder(identity) / "slide-0.json"
    content = json.loads(path.read_bytes())
    content["patches"][2][field] = value
    path.write_text(json.dumps(content))
    result = service.execution(identity)["result"]
    result["artifacts"][path.name] = receipt(path)
    update_result(service, identity, result)
    with pytest.raises(StorageError, match="invalid"):
        service.top_attention(identity, "independent")
    with pytest.raises(StorageError, match="invalid"):
        service.patch_image(identity, "independent", 1)


def test_legacy_view_limit_preserves_raw_export(study, monkeypatch):
    import histopilot.application.interpretation as module

    service, identity, _, _ = complete(study, legacy=True)
    path = service.jobs.folder(identity) / "slide-0.json"
    monkeypatch.setattr(module, "MAX_LEGACY_VIEW_BYTES", path.stat().st_size)
    assert service.top_attention(identity, "independent")["returned"] == 3
    monkeypatch.setattr(module, "MAX_LEGACY_VIEW_BYTES", path.stat().st_size - 1)
    with pytest.raises(StorageError, match="Recompute attention") as error:
        service.top_attention(identity, "independent")
    assert error.value.code == "INTERPRETATION_LEGACY_VIEW_LIMIT"
    with pytest.raises(StorageError, match="Recompute attention"):
        service.patch_image(identity, "independent", 1)
    assert service.artifact(identity, "slide-0.json") == path.read_bytes()


def test_legacy_contact_sheet_reuses_compact_verified_rows_without_mutable_aliases(
    study, monkeypatch
):
    import histopilot.application.interpretation as module

    service, identity, _, _ = complete(study, legacy=True)
    path = service.jobs.folder(identity) / "slide-0.json"
    content = json.loads(path.read_bytes())
    content["patches"][2]["unrelated"] = {"nested": ["extra metadata"]}
    write_json(path, content)
    result = service.execution(identity)["result"]
    result["artifacts"][path.name] = receipt(path)
    update_result(service, identity, result)
    top = service.top_attention(identity, "independent")
    assert "unrelated" not in top["patches"][0]
    top["patches"][0]["x"] = -5
    top["probabilities"][0] = -5
    monkeypatch.setattr(
        service, "artifact", lambda *args, **kwargs: pytest.fail("Repeated legacy JSON decode")
    )
    repeated = service.top_attention(identity, "independent", limit=20)
    assert repeated["patches"][0]["x"] == 290
    assert repeated["probabilities"] == [0.6, 0.4]
    for patch in repeated["patches"]:
        assert service.patch_image(identity, "independent", patch["index"])
    entries = [item for key, item in module._LEGACY_RANKS.items() if identity in key[0]]
    assert entries and all(len(item["rows"]) <= 21 and len(item["top"]) <= 20 for item in entries)
    assert all(
        set(patch) == {"index", "x", "y", "weight", "percentile"}
        for item in entries
        for patch in item["rows"].values()
    )


def test_patch_crops_refuse_symlink_replacement_and_incomplete_jobs(study, tmp_path):
    service, identity, document, _ = complete(study)
    source = Path(document["manifest"]["slides"][0]["slidePath"])
    target = tmp_path / "replacement.png"
    source.rename(target)
    source.symlink_to(target)
    with pytest.raises(StorageError) as error:
        service.patch_image(identity, "independent", 1)
    assert error.value.code == "INTERPRETATION_PATH_INVALID"
    state_path = service.jobs.folder(identity) / "state.json"
    state = read_json(state_path)
    write_json(state_path, {**state, "status": "running"})
    with pytest.raises(StorageError) as error:
        service.top_attention(identity, "independent")
    assert error.value.code == "INTERPRETATION_NOT_COMPLETED"
