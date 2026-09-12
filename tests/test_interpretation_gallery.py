"""Folder search, frozen feature matching, thumbnails and per-slide execution retries."""

import copy
import io
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from pydantic import ValidationError
from test_interpretation import study as study

from histopilot.application.feature_bundles import FeatureBundleService
from histopilot.application.feature_packs import FeaturePackService
from histopilot.application.features import FeatureService
from histopilot.schemas.feature_bundles import FeatureBundleSpec
from histopilot.schemas.feature_packs import FeaturePackSpec
from histopilot.schemas.features import FeatureSpec
from histopilot.schemas.interpretation import (
    InterpretationGalleryQuery,
    InterpretationSelection,
    VisualizeInterpretation,
)
from histopilot.storage.project_lock import StorageError
from histopilot.workers.pack_features import run_job

Image = pytest.importorskip("PIL.Image")


class PackingExecutor:
    def available(self):
        return True

    def running(self, session):
        return False

    def launch(self, *args):
        pass


@pytest.fixture
def gallery(study, tmp_path):
    service, manual, executor = study
    store = service.store
    slides = tmp_path / "gallery"
    features = tmp_path / "gallery-features"
    slides.mkdir()
    features.mkdir()
    identities = ["001", "Tumour-02", "病例 03"]
    records = []
    for index, identity in enumerate(identities):
        slide_path = slides / f"{identity}.png"
        Image.new("RGB", (300, 200), color=(230, 100 + index * 30, 150)).save(slide_path)
        records.append(
            {"slideId": identity, "patientId": f"p{index}", "slidePath": str(slide_path)}
        )
        with h5py.File(features / f"{identity}_patch_features.h5", "w") as handle:
            vectors = handle.create_dataset(
                "features", data=np.arange(12, dtype="float32").reshape(3, 4) + index
            )
            vectors.attrs["encoder_id"] = "test-encoder"
            coords = handle.create_dataset(
                "coords", data=np.array([[0, 0], [100, 0], [100, 100]], dtype="int64")
            )
            coords.attrs["patch_size_level0"] = 100
    draft = store.create_draft("import", "Gallery", {})
    dataset = store.publish_dataset(
        draft["id"],
        expected_revision=1,
        manifest={"kind": "dataset", "name": "Gallery"},
        artifacts={"records.json": json.dumps(records).encode()},
        operation_id="gallery-data",
    )
    inventories = FeatureService(store, service.filesystem)
    spec = FeatureSpec(
        datasetId=dataset["id"],
        path=str(features),
        encoderId="test-encoder",
        idSuffix="_patch_features",
    )
    preview = inventories.preview(spec)
    assert preview["canFreeze"], preview
    inventory = inventories.freeze(spec, preview["previewHash"], "gallery-features")
    packing = FeaturePackService(store, service.filesystem, PackingExecutor())
    pack_spec = FeaturePackSpec(featureSetId=inventory["id"], action="pack", dtype="preserve")
    pack_preview = packing.preview(pack_spec)
    assert pack_preview["canRun"], pack_preview
    job = packing.submit(pack_spec, pack_preview["previewHash"], "gallery-pack")
    packed = run_job(packing.folder / job["id"] / "plan.json")
    assert packed["state"] == "succeeded", packed
    bundles = FeatureBundleService(store, service.filesystem)
    bundle_spec = FeatureBundleSpec(
        featureSetId=inventory["id"], packArtifactIds=[packed["artifact"]["id"]]
    )
    bundle_preview = bundles.preview(bundle_spec)
    assert bundle_preview["canFreeze"], bundle_preview
    bundle = bundles.freeze(
        bundle_spec,
        bundle_preview["previewHash"],
        "gallery-bundle",
        version_label={"tag": "All slide patches"},
    )
    source = {
        "slideFolder": str(slides),
        "featureBundleId": bundle["id"],
        "predictorId": manual.predictorId,
    }
    return service, source, executor, packed["artifact"]


def visualize_request(source, paths, operation="visualize-test", **extra):
    return VisualizeInterpretation(
        **{**source, "slidePaths": [str(path) for path in paths], "operationId": operation, **extra}
    )


def test_full_folder_search_precedes_pagination_and_unicode_casefold(gallery):
    service, source, _, _ = gallery
    first = service.gallery.query(InterpretationGalleryQuery(**source, limit=1))
    assert first["total"] == 3 and first["hasMore"]
    assert first["items"][0]["slideId"] == "001"
    assert first["items"][0]["available"]
    second = service.gallery.query(InterpretationGalleryQuery(**source, search="tUmOuR", limit=1))
    assert second["total"] == 1 and second["items"][0]["slideId"] == "Tumour-02"
    unicode = service.gallery.query(InterpretationGalleryQuery(**source, search="病例"))
    assert unicode["items"][0]["available"]
    assert service.gallery.query(InterpretationGalleryQuery(**source, search="none"))["total"] == 0
    assert service.gallery.query(InterpretationGalleryQuery(**source, offset=10))["items"] == []
    assert service.gallery.sources()["items"][0]["name"] == "All slide patches"


def test_duplicate_stems_missing_features_and_symlinks_are_visible_and_safe(gallery, tmp_path):
    service, source, _, _ = gallery
    folder = Path(source["slideFolder"])
    nested = folder / "nested"
    nested.mkdir()
    Image.new("RGB", (30, 20)).save(nested / "001.png")
    Image.new("RGB", (30, 20)).save(folder / "missing.png")
    (folder / "linked.png").symlink_to(folder / "Tumour-02.png")
    (folder / "external-folder").symlink_to(tmp_path.parent, target_is_directory=True)
    result = service.gallery.query(InterpretationGalleryQuery(**source))
    assert result["total"] == 5
    duplicate = [row for row in result["items"] if row["slideId"] == "001"]
    assert len(duplicate) == 2 and all(
        not row["available"] and "Ambiguous" in row["reason"] for row in duplicate
    )
    missing = next(row for row in result["items"] if row["slideId"] == "missing")
    assert not missing["available"] and "No matching" in missing["reason"]
    assert result["warnings"] == ["Skipped 2 symbolic links."]
    with pytest.raises(StorageError):
        service.gallery.query(
            InterpretationGalleryQuery(**{**source, "slideFolder": str(tmp_path.parent)})
        )
    with pytest.raises(StorageError):
        service.gallery_thumbnail(str(folder / "linked.png"))


def test_thumbnail_uses_real_slide_without_study_and_is_bounded(gallery):
    service, source, _, _ = gallery
    path = str(Path(source["slideFolder"]) / "001.png")
    with Image.open(io.BytesIO(service.gallery_thumbnail(path, max_size=128))) as thumbnail:
        assert thumbnail.size == (128, 85)
        assert thumbnail.getpixel((0, 0)) == (230, 100, 150)
    assert service.list()["items"] == []
    with pytest.raises(StorageError):
        service.gallery_thumbnail(path, max_size=100000)


@pytest.mark.parametrize("method", ["refit", "ensemble"])
def test_single_visualization_freezes_exact_inputs_and_reuses_inflight_model(gallery, method):
    service, source, executor, _ = gallery
    predictor = service.predictors.get(source["predictorId"])
    if method == "refit":
        from histopilot.application.predictors import checkpoint_snapshot

        refit_id = "configuration-" + "e" * 64
        root = service.store.folder / "compute-jobs" / refit_id
        root.mkdir(parents=True)
        checkpoint = root / "best.ckpt"
        checkpoint.write_bytes(b"refit-checkpoint-evidence")
        frozen = service.store.publish_configuration(
            manifest={
                **predictor["manifest"],
                "method": "refit",
                "refitId": refit_id,
                "checkpoints": [{**checkpoint_snapshot(checkpoint, root), "runId": "refit"}],
            },
            operation_id="refit-predictor",
        )
        source = {**source, "predictorId": frozen["id"]}
    path = Path(source["slideFolder"]) / "001.png"
    request = visualize_request(source, [path])
    first = service.visualize(request)
    assert first["items"][0]["status"] == "queued", first
    record = first["interpretations"][0]
    assert record["manifest"]["method"] == method
    assert record["manifest"]["featureBundleId"] == source["featureBundleId"]
    assert record["manifest"]["slides"][0]["coordinatesPath"].endswith("001_patch_features.h5")
    assert len(record["manifest"]["references"]) == 3
    again = service.visualize(request)
    assert again["items"][0]["interpretationId"] == record["id"] and again["items"][0]["reused"]
    assert service.visualize(visualize_request(source, [path], operation="new-click"))["items"][0][
        "reused"
    ]
    assert len(executor.calls) == 1
    with pytest.raises(StorageError, match="different selection"):
        service.visualize(
            visualize_request(source, [path], patchWidthLevel0=50, patchHeightLevel0=50)
        )


def test_batch_keeps_successes_when_one_slide_fails_and_retry_does_not_duplicate(gallery):
    service, source, executor, _ = gallery
    folder = Path(source["slideFolder"])
    Image.new("RGB", (30, 20)).save(folder / "missing.png")
    request = visualize_request(
        source, [folder / "001.png", folder / "missing.png", folder / "病例 03.png"]
    )
    response = service.visualize(request)
    assert [row["status"] for row in response["items"]] == ["queued", "error", "queued"], response
    assert response["items"][1]["error"]["code"] == "INTERPRETATION_SLIDE_UNAVAILABLE"
    assert len(response["interpretations"]) == 2
    assert len(executor.calls) == 2
    retried = service.visualize(request)
    assert [row.get("interpretationId") for row in retried["items"]] == [
        row.get("interpretationId") for row in response["items"]
    ]
    assert len(executor.calls) == 2


def test_launch_failure_retains_saved_study_and_resource_change_creates_new_one(
    gallery, monkeypatch
):
    service, source, executor, _ = gallery
    path = Path(source["slideFolder"]) / "001.png"
    original = service.jobs.runtime
    monkeypatch.setattr(service.jobs, "runtime", lambda: {"available": False})
    response = service.visualize(visualize_request(source, [path]))
    assert response["items"][0]["status"] == "error"
    old = response["items"][0]["interpretationId"]
    assert response["items"][0]["error"]["code"] == "TRAINING_RUNTIME_UNAVAILABLE"
    monkeypatch.setattr(service.jobs, "runtime", original)
    changed = service.visualize(
        visualize_request(
            source,
            [path],
            operation="new-resource-choice",
            resources={"gpuIds": [], "dataLoaderWorkers": 0, "ramGbPerRun": 4},
        )
    )
    assert changed["items"][0]["status"] == "queued"
    assert changed["items"][0]["interpretationId"] != old
    assert len(executor.calls) == 1


def test_wrong_bundle_pack_and_encoder_are_rejected_before_compute(gallery):
    service, source, executor, _ = gallery
    with pytest.raises(StorageError, match="included"):
        service.gallery.query(
            InterpretationGalleryQuery(**source, packArtifactId="pack-" + "f" * 64)
        )
    predictor = service.predictors.get(source["predictorId"])
    manifest = copy.deepcopy(predictor["manifest"])
    manifest["inputs"]["features"]["encoderId"] = "other-encoder"
    wrong = service.store.publish_configuration(manifest=manifest, operation_id="wrong-encoder")
    response = service.gallery.query(
        InterpretationGalleryQuery(**{**source, "predictorId": wrong["id"]})
    )
    assert all(not row["available"] and "encoder" in row["reason"] for row in response["items"])
    assert executor.calls == []


def test_scan_bound_fails_instead_of_returning_incomplete_matches(gallery, monkeypatch):
    import histopilot.application.interpretation_gallery as module

    service, source, _, _ = gallery
    monkeypatch.setattr(module, "MAX_ENTRIES", 1)
    with pytest.raises(StorageError) as caught:
        service.gallery.query(InterpretationGalleryQuery(**source))
    assert caught.value.code == "INTERPRETATION_SCAN_LIMIT"


def test_legacy_manual_study_plan_and_resume_keep_original_slide_shape(study):
    service, selection, executor = study
    original = service._prepare(selection)
    # Reproduce a pre-gallery saved manifest and selection, without new defaults.
    for key in ("featureBundleId", "packArtifactId", "slideFolder"):
        original.pop(key, None)
        original["selection"].pop(key, None)
    for row in original["selection"]["slides"]:
        for key in ("sourceFormat", "packPath", "packSlideId"):
            row.pop(key, None)
    from histopilot.application.feature_bundles import _hash

    original["previewHash"] = _hash(original)
    legacy = service.store.publish_configuration(manifest=original, operation_id="legacy-manual")
    assert service.launch(legacy["id"], "legacy-launch")["status"] == "queued"
    executor.sessions.clear()
    assert service.execution(legacy["id"])["status"] == "interrupted"
    assert service.launch(legacy["id"], "legacy-resume", resume=True)["attempt"] == 2
    assert "sourceFormat" not in service._execution_plan(legacy["id"])["slides"][0]


def test_bound_inputs_cannot_swap_to_unrelated_coordinates(gallery):
    service, source, _, _ = gallery
    resolved, folder, rows, _ = service.gallery.rows(InterpretationGalleryQuery(**source))
    selected = service.gallery.input(resolved, rows[0])
    other = service.gallery.input(resolved, rows[1])
    selected.coordinatesPath = other.coordinatesPath
    payload = InterpretationSelection(
        **source,
        name="Wrong coordinates",
        encoderId=resolved["contract"]["encoderId"],
        slides=[selected],
    )
    preview = service.preview(payload)
    assert not preview["canSave"]
    assert preview["findings"][0]["code"] == "INTERPRETATION_SOURCE_MISMATCH"


def test_schema_rejects_duplicate_batch_and_partial_geometry():
    source = {
        "slideFolder": "/slides",
        "featureBundleId": "configuration-" + "a" * 64,
        "predictorId": "configuration-" + "b" * 64,
    }
    with pytest.raises(ValidationError):
        visualize_request(source, ["/slides/a.png", "/slides/a.png"])
    with pytest.raises(ValidationError):
        visualize_request(source, ["/slides/a.png"], patchWidthLevel0=20)


def test_lost_response_after_fast_failure_does_not_resume_same_operation(gallery):
    from histopilot.workers.training_process import read_json, write_json

    service, source, executor, _ = gallery
    request = visualize_request(source, [Path(source["slideFolder"]) / "001.png"])
    first = service.visualize(request)
    identity = first["items"][0]["interpretationId"]
    state_path = service.jobs.folder(identity) / "state.json"
    state = read_json(state_path)
    write_json(state_path, {**state, "status": "failed", "error": "Fast worker failure"})
    executor.sessions.clear()
    repeated = service.visualize(request)
    assert repeated["items"][0]["status"] == "failed"
    assert repeated["items"][0]["interpretation"]["execution"]["attempt"] == 1
    assert len(executor.calls) == 1
    # A new explicit operation can resume the same scientific study.
    retry = service.visualize(
        visualize_request(source, request.slidePaths, operation="explicit-retry")
    )
    assert retry["items"][0]["status"] == "queued"
    assert retry["items"][0]["interpretation"]["execution"]["attempt"] == 2
    assert len(executor.calls) == 2


def test_uncertain_response_after_launch_replays_original_operation(gallery):
    from histopilot.application.feature_bundles import _hash
    from histopilot.workers.training_process import read_json, write_json

    service, source, executor, _ = gallery
    request = visualize_request(source, [Path(source["slideFolder"]) / "001.png"])
    first = service.visualize(request)
    identity = first["items"][0]["interpretationId"]
    receipt_path = (
        service.store.folder
        / "interpretation-requests"
        / _hash(request.operationId)
        / "result.json"
    )
    receipt = read_json(receipt_path)
    receipt["items"][0]["_launched"] = (
        False  # Simulate interruption after launch before acknowledgement.
    )
    write_json(receipt_path, receipt)
    executor.sessions.clear()
    assert service.execution(identity)["status"] == "interrupted"
    repeat = service.visualize(request)
    assert repeat["items"][0]["status"] == "interrupted"
    assert len(executor.calls) == 1


def test_batch_resolves_folder_once_and_uses_real_selected_pack(gallery, monkeypatch):
    service, source, executor, artifact = gallery
    source = {**source, "packArtifactId": artifact["id"]}
    calls = []
    original = service.gallery.scan

    def scan(value):
        calls.append(value)
        return original(value)

    monkeypatch.setattr(service.gallery, "scan", scan)
    request = visualize_request(
        source, [Path(source["slideFolder"]) / name for name in ["001.png", "Tumour-02.png"]]
    )
    response = service.visualize(request)
    assert [row["status"] for row in response["items"]] == ["queued", "queued"], response
    assert len(calls) == 1
    assert len(executor.calls) == 2
    for document in response["interpretations"]:
        row = document["manifest"]["slides"][0]
        assert row["sourceFormat"] == "packed"
        assert row["packPath"] == artifact["outputPath"]
        assert row["alignment"] == "packed_verified"
        assert row["packEvidence"]["id"] == artifact["id"]
        assert len(row["packEvidence"]["validation"]["files"]) == 1
        assert document["manifest"]["packArtifactId"] == artifact["id"]


def test_gallery_http_auth_search_thumbnail_single_and_batch(gallery, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from histopilot.api.app import create_app
    from histopilot.config import Settings

    service, source, executor, _ = gallery
    monkeypatch.setattr(
        "histopilot.api.interpretation.InterpretationService", lambda *args: service
    )
    settings = Settings(workspace=tmp_path / "api-registry", data_roots=(tmp_path,))
    app = create_app(settings)
    monkeypatch.setattr(app.state.projects, "scientific_store", lambda identity: service.store)
    with TestClient(app, base_url="http://127.0.0.1:8787") as client:
        base = "/api/v1/projects/test/interpretations"
        assert client.get(base + "/sources").status_code == 401
        token = client.get("/api/v1/session").json()["token"]
        client.headers["X-HistoPilot-Token"] = token
        assert client.get(base + "/sources").json()["items"][0]["slideCount"] == 3
        searched = client.post(base + "/gallery", json={**source, "search": "tumour", "limit": 1})
        assert searched.status_code == 200, searched.text
        assert searched.json()["total"] == 1
        thumbnail = client.get(
            base + "/gallery/thumbnail",
            params={"path": searched.json()["items"][0]["slidePath"], "max_size": 128},
        )
        assert thumbnail.status_code == 200 and thumbnail.headers["content-type"] == "image/png"
        assert (
            client.get(base + "/gallery/thumbnail", params={"path": "/etc/passwd"}).status_code
            == 403
        )
        assert (
            client.get(
                base + "/gallery/thumbnail",
                params={"path": searched.json()["items"][0]["slidePath"], "max_size": 4000},
            ).status_code
            == 422
        )
        first = visualize_request(source, [Path(source["slideFolder"]) / "001.png"])
        response = client.post(base + "/visualize", json=first.model_dump())
        assert response.status_code == 202 and response.json()["items"][0]["status"] == "queued", (
            response.text
        )
        batch = visualize_request(
            source,
            [Path(source["slideFolder"]) / "001.png", Path(source["slideFolder"]) / "病例 03.png"],
            operation="api-batch",
        )
        response = client.post(base + "/visualize", json=batch.model_dump())
        assert response.status_code == 202 and len(response.json()["items"]) == 2
        assert response.json()["items"][0]["reused"]
        assert len(executor.calls) == 2


def test_tensor_review_does_not_hold_project_lifecycle_lock(gallery, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    import histopilot.application.interpretation as module
    from histopilot.storage.lifecycle import lifecycle_guard

    service, source, executor, _ = gallery
    inspect = module.inspect_inputs
    checked = []

    def acquire_as_worker():
        with lifecycle_guard(service.store.folder, timeout=0.1):
            return True

    def inspect_while_worker_can_verify(*args, **kwargs):
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(acquire_as_worker).result(timeout=1)
        checked.append(bool(executor.calls))
        return inspect(*args, **kwargs)

    monkeypatch.setattr(module, "inspect_inputs", inspect_while_worker_can_verify)
    paths = [Path(source["slideFolder"]) / name for name in ["001.png", "Tumour-02.png"]]
    response = service.visualize(visualize_request(source, paths))
    assert [row["status"] for row in response["items"]] == ["queued", "queued"], response
    assert any(checked), "Verified lock access while an earlier slide's worker could start."
    assert len(checked) == 6  # Preview, publication check, execution check for each slide.


def test_batch_review_budget_retains_started_slides_and_marks_remaining(gallery, monkeypatch):
    import time
    from types import SimpleNamespace

    import histopilot.application.interpretation as module

    service, source, executor, _ = gallery
    offset = [0]
    monkeypatch.setattr(
        module, "time", SimpleNamespace(monotonic=lambda: time.monotonic() + offset[0])
    )
    launch = executor.launch

    def consume_budget(*args, **kwargs):
        launch(*args, **kwargs)
        offset[0] = 46

    monkeypatch.setattr(executor, "launch", consume_budget)
    paths = [Path(source["slideFolder"]) / name for name in ["001.png", "Tumour-02.png"]]
    response = service.visualize(visualize_request(source, paths))
    assert [row["status"] for row in response["items"]] == ["queued", "error"], response
    assert response["items"][1]["error"]["code"] == "INTERPRETATION_BATCH_LIMIT"
    assert len(executor.calls) == 1


def test_concurrent_distinct_operations_reuse_one_scientific_study(gallery, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    service, source, executor, _ = gallery
    barrier = Barrier(2)
    preview = service.preview

    def both_prepared(*args, **kwargs):
        result = preview(*args, **kwargs)
        barrier.wait(timeout=5)
        return result

    monkeypatch.setattr(service, "preview", both_prepared)
    path = Path(source["slideFolder"]) / "001.png"
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                service.visualize, visualize_request(source, [path], operation=f"parallel-{index}")
            )
            for index in range(2)
        ]
        responses = [future.result(timeout=15) for future in futures]
    assert all(response["items"][0]["status"] == "queued" for response in responses), responses
    assert (
        responses[0]["items"][0]["interpretationId"] == responses[1]["items"][0]["interpretationId"]
    )
    assert len(executor.calls) == 1


def test_unacknowledged_operation_reuses_study_launched_by_other_operation(gallery, monkeypatch):
    service, source, executor, _ = gallery
    original = service.jobs.runtime
    path = Path(source["slideFolder"]) / "001.png"
    first = visualize_request(source, [path], operation="original-failed-launch")
    monkeypatch.setattr(service.jobs, "runtime", lambda: {"available": False})
    assert service.visualize(first)["items"][0]["status"] == "error"
    monkeypatch.setattr(service.jobs, "runtime", original)
    second = service.visualize(visualize_request(source, [path], operation="second-launch"))
    assert second["items"][0]["status"] == "queued"
    repeated = service.visualize(first)
    assert repeated["items"][0]["status"] == "queued", repeated
    assert repeated["items"][0]["reused"]
    assert len(executor.calls) == 1
