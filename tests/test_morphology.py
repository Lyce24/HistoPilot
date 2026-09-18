"""Exploration uses authentic feature rows and exact frozen slide identities."""

import io
import json

import h5py
import numpy as np
import pytest
from PIL import Image

from histopilot.application.feature_bundles import FeatureBundleService
from histopilot.application.feature_packs import FeaturePackService
from histopilot.application.features import FeatureService
from histopilot.application.morphology import MorphologyService, normalized, projection, sample_rows
from histopilot.schemas.feature_bundles import FeatureBundleSpec
from histopilot.schemas.feature_packs import FeaturePackSpec
from histopilot.schemas.features import FeatureSpec
from histopilot.schemas.morphology import MorphologyIndexRequest, MorphologyNeighborsRequest
from histopilot.storage.filesystem import LocalFilesystem
from histopilot.storage.packed import _stamp
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import ScientificStore
from histopilot.workers.pack_features import run_job


class FakeExecutor:
    def available(self):
        return True

    def running(self, session):
        return False

    def launch(self, session, runner, plan):
        pass


@pytest.fixture
def study(tmp_path, request):
    feature_kind = getattr(request, "param", "patch")
    root = tmp_path / "project"
    root.mkdir()
    store = ScientificStore(root, "morphology-test")
    images, sources = tmp_path / "images", tmp_path / "features"
    images.mkdir()
    sources.mkdir()
    records, inventory = [], []
    for name, vector, color in (
        ("a", [1.0, 0.0, 0.0], "red"),
        ("b", [0.9, 0.1, 0.0], "green"),
        ("c", [0.0, 1.0, 0.0], "blue"),
    ):
        path = images / f"{name}.png"
        Image.new("RGB", (64, 64), color).save(path)
        records.append(
            {
                "slideId": name,
                "patientId": f"p-{name}",
                "slidePath": str(path),
                "attributes": {"site": "A" if name != "c" else "B"},
            }
        )
        inventory.append({"slideId": name, "path": str(path), **_stamp(path.stat())})
        with h5py.File(sources / f"{name}.h5", "w") as handle:
            if feature_kind == "slide":
                handle.create_dataset("features", data=np.array(vector, dtype=np.float32))
                continue
            handle.create_dataset(
                "features", data=np.tile(np.array(vector, dtype=np.float32), (4, 1))
            )
            coords = handle.create_dataset(
                "coords", data=np.array([[0, 0], [20, 0], [0, 20], [20, 20]], dtype=np.int64)
            )
            coords.attrs["patch_size_level0"] = 16
    draft = store.create_draft("import", "slides", {})
    dataset = store.publish_dataset(
        draft["id"],
        expected_revision=1,
        manifest={"kind": "dataset"},
        artifacts={
            "records.json": json.dumps(records).encode(),
            "inventory.json": json.dumps(inventory).encode(),
        },
        operation_id="dataset",
    )
    filesystem = LocalFilesystem((tmp_path,))
    features = FeatureService(store, filesystem)
    spec = FeatureSpec(datasetId=dataset["id"], path=str(sources), featureKind=feature_kind)
    feature = features.freeze(
        spec, features.preview(spec)["previewHash"], "feature", version_label={"tag": "test"}
    )
    packs = FeaturePackService(store, filesystem, FakeExecutor())
    pack_spec = FeaturePackSpec(featureSetId=feature["id"], action="validate")
    preview = packs.preview(pack_spec)
    job = packs.submit(pack_spec, preview["previewHash"], "validate")
    result = run_job(packs.folder / job["id"] / "plan.json")
    assert result["state"] == "succeeded", result
    bundles = FeatureBundleService(store, filesystem)
    bundle_spec = FeatureBundleSpec(featureSetId=feature["id"])
    preview = bundles.preview(bundle_spec)
    bundle = bundles.freeze(
        bundle_spec, preview["previewHash"], "bundle", version_label={"tag": "bundle"}
    )
    request = MorphologyIndexRequest(
        datasetId=dataset["id"], featureBundleId=bundle["id"], patchesPerSlide=2
    )
    return MorphologyService(store, filesystem), request, sources, images


def test_projection_and_neighbors_use_real_cosine_not_projected_distance(study):
    service, request, _, _ = study
    result = service.build(request)
    assert result["indexedSlides"] == 3
    assert result["indexedPatches"] == 6
    assert [row["patchIndex"] for row in result["patches"] if row["slideId"] == "a"] == [0, 3]
    assert all(point["sampledPatches"] == 2 for point in result["points"])
    assert sum(result["explainedVariance"]) == pytest.approx(1.0)
    nearest = service.neighbors(MorphologyNeighborsRequest(indexId=result["indexId"], slideId="a"))
    assert [item["slideId"] for item in nearest["items"]] == ["b", "c"]
    assert nearest["items"][0]["similarity"] == pytest.approx(0.9 / np.sqrt(0.82), abs=1e-6)
    patches = service.neighbors(
        MorphologyNeighborsRequest(
            indexId=result["indexId"], slideId="a", mode="patch", patchIndex=3
        )
    )
    assert patches["scope"] == "sampled_patches"
    assert [row["slideId"] for row in patches["items"][:2]] == ["b", "b"]
    assert patches["items"][0]["patchIndex"] == 0
    assert service.build(request) == result


def test_projection_handles_single_and_identical_slides():
    xy, ratio = projection(np.ones((1, 4), dtype=np.float32))
    assert xy == [[0.0, 0.0]] and ratio == [0.0, 0.0]
    xy, ratio = projection(np.ones((3, 4), dtype=np.float32))
    assert np.isfinite(xy).all() and ratio == [0.0, 0.0]
    assert sample_rows(3, 8).tolist() == [0, 1, 2]
    assert normalized([[0.0, 0.0]]).tolist() == [[0.0, 0.0]]


def test_cached_index_rejects_changed_source(study):
    service, request, source, _ = study
    result = service.build(request)
    with h5py.File(source / "a.h5", "r+") as handle:
        handle["features"][0, 0] = 42
    with pytest.raises(StorageError, match="verification changed"):
        service.build(request)
    with pytest.raises(StorageError, match="verification changed"):
        service.neighbors(MorphologyNeighborsRequest(indexId=result["indexId"], slideId="a"))


def test_exact_geometry_patch_crop_and_no_invented_tissue(study):
    service, request, _, _ = study
    quality = service.quality(request.datasetId, "a", request.featureBundleId)
    assert quality["patchWidth"] == 16
    assert quality["patchCount"] == 4
    assert quality["coordinateBounds"] == {"x": 0, "y": 0, "width": 36.0, "height": 36.0}
    assert quality["tissueContours"] == [] and quality["artifactRemoval"] is None
    crop = service.patch_image(request.datasetId, "a", request.featureBundleId, 3)
    with Image.open(io.BytesIO(crop)) as image:
        assert image.size == (16, 16)
        assert image.getpixel((5, 5)) == (255, 0, 0)
    with pytest.raises(StorageError, match="outside"):
        service.patch_image(request.datasetId, "a", request.featureBundleId, 100)
    with pytest.raises(StorageError, match="not in"):
        service.image(request.datasetId, "../outside")


def test_changed_slide_cannot_be_visualized_again(study):
    service, request, _, images = study
    Image.new("RGB", (64, 64), "white").save(images / "a.png")
    with pytest.raises(StorageError) as caught:
        service.image(request.datasetId, "a")
    assert caught.value.code == "MORPHOLOGY_SLIDE_CHANGED"


def test_explicit_sampling_limits_and_unmatched_selection(study, monkeypatch):
    service, request, _, _ = study
    with pytest.raises(StorageError, match="Every selected"):
        service.build(request.model_copy(update={"slideIds": ["not-in-dataset"]}))
    monkeypatch.setattr("histopilot.application.morphology.MAX_INDEX_VALUES", 1)
    with pytest.raises(StorageError) as caught:
        service.build(request.model_copy(update={"patchesPerSlide": 3}))
    assert caught.value.code == "MORPHOLOGY_INDEX_LIMIT"


def test_index_project_scope_and_unsampled_query(study, tmp_path):
    service, request, _, _ = study
    result = service.build(request)
    with pytest.raises(StorageError, match="sampled patch"):
        service.neighbors(
            MorphologyNeighborsRequest(
                indexId=result["indexId"], slideId="a", mode="patch", patchIndex=1
            )
        )
    (tmp_path / "other").mkdir()
    other = MorphologyService(ScientificStore(tmp_path / "other", "other"), service.filesystem)
    with pytest.raises(StorageError) as caught:
        other.neighbors(MorphologyNeighborsRequest(indexId=result["indexId"], slideId="a"))
    assert caught.value.code == "MORPHOLOGY_INDEX_EXPIRED"


def test_reused_slide_names_from_another_dataset_cannot_bind_features(study):
    service, request, _, _ = study
    records = service.records(request.datasetId)
    draft = service.store.create_draft("import", "Other cohort", {})
    other = service.store.publish_dataset(
        draft["id"],
        expected_revision=1,
        manifest={"kind": "dataset", "name": "Other"},
        artifacts={"records.json": json.dumps(records).encode()},
        operation_id="other-dataset",
    )
    with pytest.raises(StorageError) as caught:
        service.build(request.model_copy(update={"datasetId": other["id"]}))
    assert caught.value.code == "MORPHOLOGY_DATASET_MISMATCH"
    with pytest.raises(StorageError) as caught:
        service.quality(other["id"], "a", request.featureBundleId)
    assert caught.value.code == "MORPHOLOGY_DATASET_MISMATCH"


def test_explicit_verified_store_scope_can_be_used_with_dataset(study):
    service, request, sources, _ = study
    features = FeatureService(service.store, service.filesystem)
    spec = FeatureSpec(path=str(sources))
    feature = features.freeze(
        spec,
        features.preview(spec)["previewHash"],
        "store-feature",
        version_label={"tag": "Store source"},
    )
    packing = FeaturePackService(service.store, service.filesystem, FakeExecutor())
    validate = FeaturePackSpec(featureSetId=feature["id"], action="validate")
    job = packing.submit(validate, packing.preview(validate)["previewHash"], "store-validate")
    assert run_job(packing.folder / job["id"] / "plan.json")["state"] == "succeeded"
    spec = FeatureBundleSpec(featureSetId=feature["id"])
    bundle = service.bundles.freeze(
        spec,
        service.bundles.preview(spec)["previewHash"],
        "store-bundle",
        version_label={"tag": "Store bundle"},
    )
    assert (
        service.build(request.model_copy(update={"featureBundleId": bundle["id"]}))["indexedSlides"]
        == 3
    )


def test_cached_public_response_is_an_independent_snapshot(study):
    service, request, _, _ = study
    result = service.build(request)
    result["points"][0]["slideId"] = "changed-client-object"
    result["patches"][0]["x"] = -999
    again = service.build(request)
    assert again["points"][0]["slideId"] == "a"
    assert again["patches"][0]["x"] == 0


def test_duplicate_frozen_slide_identity_is_rejected(study):
    service, request, _, _ = study
    rows = service.records(request.datasetId)
    draft = service.store.create_draft("import", "Invalid duplicate", {})
    dataset = service.store.publish_dataset(
        draft["id"],
        expected_revision=1,
        manifest={"kind": "dataset"},
        artifacts={"records.json": json.dumps([rows[0], rows[0]]).encode()},
        operation_id="duplicate-dataset",
    )
    with pytest.raises(StorageError) as caught:
        service.records(dataset["id"])
    assert caught.value.code == "REVIEW_DATASET_INVALID"


def test_contour_holes_preserved_and_invalid_geometry_rejected(study, tmp_path):
    service, _, _, _ = study
    identity = "extraction-test"
    folder = service.store.folder / "extractions" / identity
    folder.mkdir(parents=True)
    contours = tmp_path / "contours_geojson"
    contours.mkdir()
    (folder / "job.json").write_text(json.dumps({"outputLayout": {"geojsonDir": str(contours)}}))
    rings = [[[0, 0], [60, 0], [60, 60], [0, 0]], [[5, 5], [10, 5], [10, 10], [5, 5]]]
    path = contours / "a.geojson"
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [{"geometry": {"type": "Polygon", "coordinates": rings}}],
            }
        )
    )
    value, warnings = service._contours({"jobId": identity}, "a", {"width": 64, "height": 64})
    assert value == [rings]
    assert "not part of the frozen" in warnings[0]
    rings[0][0] = [-1, 0]
    path.write_text(json.dumps({"type": "Polygon", "coordinates": rings}))
    value, warnings = service._contours({"jobId": identity}, "a", {"width": 64, "height": 64})
    assert value == [] and "unavailable" in warnings[0]
