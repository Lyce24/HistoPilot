"""Extraction preflight, durable jobs, recovery and native artifacts without GPU execution."""

import json
from pathlib import Path

import pytest

from histopilot.application.extractions import ExtractionService, _write
from histopilot.schemas.extractions import ExtractionSpec
from histopilot.storage.filesystem import LocalFilesystem
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import ScientificStore


class FakeExecutor:
    sessions: set[str]

    def __init__(self):
        self.sessions = set()
        self.launches = []

    def available(self):
        return True

    def launch(self, session, runner, plan):
        self.sessions.add(session)
        self.launches.append(json.loads(plan.read_text()))

    def running(self, session):
        return session in self.sessions

    def cancel(self, session):
        self.sessions.discard(session)


@pytest.fixture
def extraction(tmp_path, monkeypatch):
    folder = tmp_path / "experiment"
    folder.mkdir()
    roots = [tmp_path / "drive-d", tmp_path / "oceanpath-hot"]
    slides = []
    for number, root in enumerate(roots):
        root.mkdir()
        slide = root / f"slide.{number}.svs"
        slide.write_bytes(b"fixture slide")
        slides.append({"slideId": slide.stem, "patientId": f"p{number}", "slidePath": str(slide)})
    store = ScientificStore(folder, "project-extractions")
    draft = store.create_draft("import", "test", {})
    dataset = store.publish_dataset(
        draft["id"],
        expected_revision=1,
        manifest={"kind": "dataset"},
        artifacts={"records.json": json.dumps(slides).encode()},
        operation_id="dataset",
    )
    executor = FakeExecutor()
    service = ExtractionService(store, LocalFilesystem(tuple(roots)), executor)
    monkeypatch.setattr(
        "histopilot.adapters.trident.discover_runtime",
        lambda: {
            "available": True,
            "pythonPath": "/usr/bin/python3",
            "tridentRoot": str(tmp_path / "runtime"),
        },
    )
    # Command generation is separately tested against the real upstream option catalog.
    monkeypatch.setattr(
        "histopilot.adapters.trident.build_command",
        lambda options, **kwargs: [
            kwargs["python_path"],
            "run_batch_of_slides.py",
            "--custom_list_of_wsis",
            kwargs["custom_list_of_wsis"],
        ],
    )
    spec = ExtractionSpec(
        datasetId=dataset["id"], outputPath=str(folder / "trident"), options={"task": "seg"}
    )
    return service, spec, executor, slides


def submit(service, spec, operation="run"):
    preview = service.preview(spec)
    assert preview["canRun"], preview
    return service.submit(spec, preview["previewHash"], operation)


def test_lost_launch_acknowledgement_keeps_extraction_job_active(extraction, monkeypatch):
    service, spec, executor, _slides = extraction
    original = executor.launch

    def launch_then_timeout(*args, **kwargs):
        original(*args, **kwargs)
        raise TimeoutError("Lost acknowledgement")

    monkeypatch.setattr(executor, "launch", launch_then_timeout)
    preview = service.preview(spec)
    job = service.submit(spec, preview["previewHash"], "launch")
    assert job["state"] == "running"
    assert service.submit(spec, preview["previewHash"], "launch")["id"] == job["id"]
    assert service.cancel(job["id"])["state"] == "cancelling"
    assert len(executor.launches) == 1


def test_exact_multiple_roots_manifest_idempotency_logs_and_reopen(extraction):
    service, spec, executor, slides = extraction
    preview = service.preview(spec)
    assert preview["slideCount"] == 2
    assert not Path(spec.outputPath).exists()
    job = service.submit(spec, preview["previewHash"], "run")
    assert job["state"] == "running"
    assert len(executor.launches) == 1
    assert service.submit(spec, preview["previewHash"], "run")["id"] == job["id"]
    folder = service.folder / job["id"]
    assert (folder / "slides.csv").read_text().splitlines() == [
        "wsi",
        "drive-d/slide.0.svs",
        "oceanpath-hot/slide.1.svs",
    ]
    (folder / "worker.log").write_text("segmenting slide 1\n")
    reopened = ExtractionService(service.store, service.filesystem, executor)
    assert reopened.get(job["id"], logs=True)["logs"] == "segmenting slide 1\n"
    executor.sessions.clear()
    assert reopened.get(job["id"])["state"] == "interrupted"


def test_progress_reads_existing_worker_logs_on_list_and_detail(extraction, monkeypatch):
    service, spec, executor, slides = extraction
    job = submit(service, spec)
    folder = service.folder / job["id"]
    original = (folder / "job.json").read_bytes()
    log = (
        "\rSegmenting tissue:  50%|#####     | 1/2 "
        "[00:19<00:19, 19.05s/it, Segmenting <name=slide.1>]\x1b[A"
    )
    (folder / "worker.log").write_text(log)
    monkeypatch.setattr(
        "histopilot.application.extraction_artifacts.inspect_outputs",
        lambda *_: pytest.fail("Progress polling must not open scientific artifacts"),
    )
    detail = service.get(job["id"], logs=True)
    progress = detail["progress"]
    assert progress["stage"] == "segmentation"
    assert progress["completed"] == 1
    assert progress["total"] == 2
    assert progress["percent"] == 50
    assert progress["currentSlide"] == "slide.1"
    assert progress["etaSeconds"] == 19
    assert detail["logs"] == log
    listed = service.list()["jobs"][0]
    assert listed["progress"]["completed"] == 1
    assert "logs" not in listed
    assert "_logUpdatedAt" not in listed
    assert (folder / "job.json").read_bytes() == original


def test_progress_tail_is_bounded_and_terminal_runtime_stops(extraction):
    from histopilot.application.extractions import MAX_LOG_BYTES

    service, spec, executor, slides = extraction
    job = submit(service, spec)
    folder = service.folder / job["id"]
    with (folder / "worker.log").open("wb") as stream:
        stream.seek(MAX_LOG_BYTES * 4)
        stream.write(b"\rSegmenting tissue:  50%|#####| 1/2 [00:19<00:19, 19.05s/it]")
    _write(
        folder / "result.json",
        {
            "state": "cancelled",
            "exitCode": -15,
            "startedAt": "2026-09-10T06:20:18+00:00",
            "finishedAt": "2026-09-10T06:23:19+00:00",
        },
    )
    response = service.get(job["id"], logs=True)
    assert len(response["logs"].encode()) == MAX_LOG_BYTES
    assert response["progress"]["completed"] == 1
    assert response["progress"]["elapsedSeconds"] == 181
    assert response["progress"]["etaSeconds"] is None


@pytest.mark.parametrize("kind", ["symlink", "fifo", "unreadable"])
def test_unavailable_progress_does_not_block_status_or_cancellation(extraction, kind, monkeypatch):
    import os

    service, spec, executor, slides = extraction
    job = submit(service, spec)
    folder = service.folder / job["id"]
    if kind == "symlink":
        (folder / "worker.log").symlink_to(folder / "job.json")
    elif kind == "fifo":
        os.mkfifo(folder / "worker.log")
    else:

        def denied(_path):
            raise PermissionError("Read denied")

        monkeypatch.setattr("histopilot.application.extractions._log_tail", denied)
    response = service.get(job["id"], logs=True)
    assert response["state"] == "running"
    assert response["logs"] == ""
    assert any("cannot be read safely" in item for item in response["progress"]["warnings"])
    assert service.list()["jobs"][0]["state"] == "running"
    assert service.cancel(job["id"])["state"] == "cancelling"
    assert (folder / "cancelled").exists()


def test_skip_errors_exit_zero_cannot_claim_missing_outputs(extraction):
    service, spec, executor, slides = extraction
    job = submit(service, spec)
    _write(service.folder / job["id"] / "result.json", {"state": "succeeded", "exitCode": 0})
    _write(
        service.folder / job["id"] / "validation.json",
        {"jobId": job["id"], "completedSlides": 0, "missingSlides": 2},
    )
    result = service.get(job["id"])
    assert result["state"] == "failed"
    assert result["result"]["missingSlides"] == 2
    contours = Path(spec.outputPath) / "contours_geojson"
    contours.mkdir()
    for slide in slides:
        (contours / f"{slide['slideId']}.geojson").write_text(
            '{"type":"FeatureCollection","features":[]}'
        )
    _write(
        service.folder / job["id"] / "validation.json",
        {"jobId": job["id"], "completedSlides": 2, "missingSlides": 0},
    )
    assert service.get(job["id"])["state"] == "succeeded"


def test_cancel_retains_artifacts_and_resume_requires_matching_configuration(extraction):
    service, spec, executor, slides = extraction
    job = submit(service, spec)
    result = service.cancel(job["id"])
    assert result["state"] == "cancelling"
    assert not service.preview(spec)["canRun"]
    executor.sessions.clear()
    _write(service.folder / job["id"] / "result.json", {"state": "cancelled", "exitCode": -15})
    assert service.get(job["id"])["state"] == "cancelled"
    assert Path(spec.outputPath).is_dir()
    assert service.preview(spec)["canRun"]
    changed = spec.model_copy(update={"options": {"task": "seg", "patch_size": 512}})
    preview = service.preview(changed)
    assert not preview["canRun"]
    assert any(item["code"] == "OUTPUT_CONFIG_CHANGED" for item in preview["findings"])


def test_reject_changed_source_busy_output_and_operation_conflict(extraction):
    service, spec, executor, slides = extraction
    preview = service.preview(spec)
    Path(slides[0]["slidePath"]).write_bytes(b"changed slide")
    with pytest.raises(StorageError, match="changed"):
        service.submit(spec, preview["previewHash"], "stale")
    submit(service, spec)
    assert not service.preview(spec)["canRun"]
    altered = spec.model_copy(update={"outputPath": spec.outputPath + "-other"})
    with pytest.raises(StorageError) as error:
        service.submit(altered, preview["previewHash"], "run")
    assert error.value.code == "OPERATION_CONFLICT"


def test_blocks_unowned_nonempty_paths_traversal_cache_and_invalid_options(extraction, tmp_path):
    service, spec, executor, slides = extraction
    Path(spec.outputPath).mkdir()
    (Path(spec.outputPath) / "original.txt").write_text("keep")
    assert not service.preview(spec)["canRun"]
    for path in (
        "/outside/new-output",
        str(service.store.folder),
        str(service.store.folder / "datasets" / "new"),
    ):
        with pytest.raises(StorageError):
            service.preview(spec.model_copy(update={"outputPath": path}))
    for options in (
        {"surprise": True},
        {"wsi_cache": str(tmp_path)},
        {"coords_dir": "../../escape"},
    ):
        with pytest.raises(StorageError):
            service.preview(spec.model_copy(update={"options": options}))
    link = service.store.folder / "link"
    link.symlink_to(Path(spec.outputPath))
    with pytest.raises(StorageError):
        service.preview(spec.model_copy(update={"outputPath": str(link)}))


def test_runtime_unavailable_never_launches(extraction, monkeypatch):
    service, spec, executor, slides = extraction
    monkeypatch.setattr(
        "histopilot.adapters.trident.discover_runtime",
        lambda: {"available": False, "error": "Missing TRIDENT"},
    )
    preview = service.preview(spec)
    assert not preview["canRun"]
    with pytest.raises(StorageError) as error:
        service.submit(spec, preview["previewHash"], "blocked")
    assert error.value.code == "EXTRACTION_INVALID"
    assert not executor.launches


def test_custom_csv_subset_mpp_cache_and_stage_prerequisites(extraction):
    service, spec, executor, slides = extraction
    csv_path = service.store.folder / "selected.csv"
    csv_path.write_text("wsi,mpp\nslide.1.svs,0.5\n")
    selected = spec.model_copy(
        update={
            "options": {
                "task": "seg",
                "custom_list_of_wsis": str(csv_path),
                "wsi_cache": str(service.filesystem.roots[1] / "cache"),
            }
        }
    )
    preview = service.preview(selected)
    assert preview["slideCount"] == 1
    job = submit(service, selected)
    assert (service.folder / job["id"] / "slides.csv").read_text().splitlines() == [
        "wsi,mpp",
        "slide.1.svs,0.5",
    ]
    assert job["spec"]["options"]["wsi_cache"].endswith("/cache")
    csv_path.write_text("wsi,mpp\nnot-in-dataset.svs,0.5\n")
    with pytest.raises(StorageError) as error:
        service.preview(selected)
    assert error.value.code == "INVALID_SLIDE_LIST"
    for task in ("coords", "feat"):
        preview = service.preview(
            spec.model_copy(
                update={"outputPath": spec.outputPath + task, "options": {"task": task}}
            )
        )
        assert not preview["canRun"]
        assert any(item["code"] == "STAGE_INPUT_MISSING" for item in preview["findings"])


def test_result_polling_reads_saved_validation_without_reopening_artifacts(extraction, monkeypatch):
    service, spec, executor, slides = extraction
    job = submit(service, spec)
    _write(service.folder / job["id"] / "result.json", {"state": "succeeded", "exitCode": 0})
    assert service.get(job["id"])["state"] == "failed"  # No validation is not success.
    _write(
        service.folder / job["id"] / "validation.json",
        {"jobId": job["id"], "completedSlides": 2, "missingSlides": 0, "unvalidatedSlides": 0},
    )
    monkeypatch.setattr(
        "histopilot.application.extraction_artifacts.inspect_outputs",
        lambda *_: pytest.fail("HTTP status must not inspect tensors"),
    )
    assert service.get(job["id"])["state"] == "succeeded"


def test_modified_checkpoint_at_same_path_cannot_reuse_embeddings(extraction):
    service, spec, executor, slides = extraction
    checkpoint = service.store.folder / "model.pt"
    checkpoint.write_bytes(b"first checkpoint")
    spec = spec.model_copy(
        update={"options": {"task": "all", "patch_encoder_ckpt_path": str(checkpoint)}}
    )
    job = submit(service, spec)
    executor.sessions.clear()
    _write(service.folder / job["id"] / "result.json", {"state": "cancelled", "exitCode": -15})
    assert service.preview(spec)["canRun"]
    checkpoint.write_bytes(b"other checkpoint")
    preview = service.preview(spec)
    assert not preview["canRun"]
    assert any(item["code"] == "ENCODER_CONFIG_CHANGED" for item in preview["findings"])


@pytest.mark.parametrize(
    "change", ["missing_counts", "short_count", "boolean_count", "incomplete", "error_finding"]
)
def test_completion_requires_explicit_consistent_coverage(extraction, change):
    service, spec, executor, slides = extraction
    job = submit(service, spec)
    folder = service.folder / job["id"]
    _write(folder / "result.json", {"state": "succeeded", "exitCode": 0})
    coverage = {
        "jobId": job["id"],
        "completedSlides": 2,
        "missingSlides": 0,
        "unvalidatedSlides": 0,
    }
    if change == "missing_counts":
        coverage.pop("completedSlides")
    elif change == "short_count":
        coverage["completedSlides"] = 1
    elif change == "boolean_count":
        coverage["missingSlides"] = False
    elif change == "incomplete":
        coverage["inspectionComplete"] = False
    else:
        coverage["findings"] = [{"severity": "error", "code": "INVALID_EXTRACTION_ARTIFACT"}]
    _write(folder / "validation.json", coverage)
    assert service.get(job["id"])["state"] == "failed"


@pytest.mark.parametrize(
    "folder", ["packing/new-run", "configurations/new-run", "jobs/new-run", ".git/new-run"]
)
def test_extraction_cannot_write_inside_managed_metadata(extraction, folder):
    service, spec, executor, slides = extraction
    with pytest.raises(StorageError) as caught:
        service.preview(spec.model_copy(update={"outputPath": str(service.store.folder / folder)}))
    assert caught.value.code == "INVALID_OUTPUT"


def test_extraction_cannot_create_children_inside_an_existing_pack(extraction):
    service, spec, executor, slides = extraction
    pack = service.filesystem.roots[0] / "existing-pack"
    pack.mkdir()
    for name in ("features.bin", "coords.bin", "index.parquet", "meta.json"):
        (pack / name).write_bytes(b"fixture")
    with pytest.raises(StorageError) as caught:
        service.preview(spec.model_copy(update={"outputPath": str(pack / "new-extraction")}))
    assert caught.value.code == "OUTPUT_IMMUTABLE"


def test_authenticated_project_extraction_api_roundtrip(extraction, monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from histopilot.api import create_app
    from histopilot.config import Settings

    service, spec, executor, slides = extraction
    monkeypatch.setattr(
        "histopilot.application.extractions.TmuxExtractionExecutor", lambda: executor
    )
    settings = Settings(workspace=tmp_path / "registry", data_roots=(tmp_path,))
    app = create_app(settings)
    with TestClient(app, base_url="http://127.0.0.1:8787") as client:
        assert client.get("/api/v1/projects/unknown/extractions/catalog").status_code == 401
        client.headers["X-HistoPilot-Token"] = client.get("/api/v1/session").json()["token"]
        response = client.post(
            "/api/v1/projects",
            json={"name": "PFM API", "storagePath": str(tmp_path / "api-project")},
        )
        assert response.status_code == 201, response.text
        project = response.json()["id"]
        store = app.state.projects.scientific_store(project)
        draft = store.create_draft("import", "slides", {})
        dataset = store.publish_dataset(
            draft["id"],
            expected_revision=1,
            manifest={"kind": "dataset"},
            artifacts={"records.json": json.dumps(slides).encode()},
            operation_id="api-dataset",
        )
        base = f"/api/v1/projects/{project}/extractions"
        assert client.get(base + "/catalog").json()["runtime"]["available"]
        intent = {
            "datasetId": dataset["id"],
            "outputPath": str(store.folder / "trident"),
            "options": {"task": "seg"},
        }
        bad = client.post(
            base + "/preview", json={**intent, "options": {"arbitrary_python": "bad"}}
        )
        assert bad.status_code == 422
        preview = client.post(base + "/preview", json=intent)
        assert preview.status_code == 200, preview.text
        assert preview.json()["canRun"]
        request = {**intent, "previewHash": preview.json()["previewHash"], "operationId": "api-run"}
        started = client.post(base, json=request)
        assert started.status_code == 201, started.text
        identity = started.json()["id"]
        assert client.post(base, json=request).json()["id"] == identity
        assert len(client.get(base).json()["jobs"]) == 1
        job_folder = store.folder / "extractions" / identity
        (job_folder / "worker.log").write_text(
            "\rSegmenting tissue:  50%|#####| 1/2 [00:19<00:19, 19.05s/it]"
        )
        detail = client.get(f"{base}/{identity}").json()
        assert detail["state"] == "running"
        assert detail["progress"]["stage"] == "segmentation"
        assert detail["progress"]["percent"] == 50
        assert "Segmenting tissue" in detail["logs"]
        assert client.get(base).json()["jobs"][0]["progress"]["completed"] == 1
        assert client.post(f"{base}/{identity}/cancel", json={}).json()["state"] == "cancelling"
        assert client.get(f"{base}/invalid").status_code == 404


def test_generated_project_features_attach_without_external_data_root(tmp_path):
    import h5py
    import numpy as np
    from fastapi.testclient import TestClient

    from histopilot.api import create_app
    from histopilot.config import Settings

    settings = Settings(workspace=tmp_path / "workspace")
    app = create_app(settings)
    with TestClient(app, base_url="http://127.0.0.1:8787") as client:
        client.headers["X-HistoPilot-Token"] = client.get("/api/v1/session").json()["token"]
        project = client.post(
            "/api/v1/projects",
            json={"name": "Generated PFM", "storagePath": str(settings.workspace / "experiment")},
        ).json()["id"]
        store = app.state.projects.scientific_store(project)
        draft = store.create_draft("import", "slides", {})
        dataset = store.publish_dataset(
            draft["id"],
            expected_revision=1,
            manifest={"kind": "dataset"},
            artifacts={"records.json": json.dumps([{"slideId": "A", "patientId": "P1"}]).encode()},
            operation_id="generated-dataset",
        )
        output = store.folder / "trident" / "20x_256px_0px_overlap" / "features_uni_v1"
        output.mkdir(parents=True)
        with h5py.File(output / "A.h5", "w") as handle:
            handle.create_dataset("features", data=np.ones((2, 4), dtype="float32"))
            handle.create_dataset("coords", data=np.ones((2, 2), dtype="int64"))
        base = f"/api/v1/projects/{project}"
        spec = {"datasetId": dataset["id"], "path": str(output)}
        preview = client.post(base + "/features/preview", json=spec)
        assert preview.status_code == 200, preview.text
        assert preview.json()["canFreeze"]
        frozen = client.post(
            base + "/features/freeze",
            json={
                **spec,
                "previewHash": preview.json()["previewHash"],
                "operationId": "generated-features",
                "versionLabel": {"tag": "Generated UNI features"},
            },
        )
        assert frozen.status_code == 201, frozen.text
        rejected = client.post(
            base + "/features/preview", json={**spec, "path": str(settings.workspace)}
        )
        assert rejected.status_code == 403
