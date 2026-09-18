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
            "slideRoot": str(service.filesystem.roots[1]),
            "options": {
                "task": "seg",
                "custom_list_of_wsis": str(csv_path),
                "wsi_cache": str(service.filesystem.roots[1] / "cache"),
            },
        }
    )
    preview = service.preview(selected)
    assert preview["slideCount"] == 1
    assert preview["slideList"]["source"] == "list"
    job = submit(service, selected)
    assert (service.folder / job["id"] / "slides.csv").read_text().splitlines() == [
        "wsi,mpp",
        "slide.1.svs,0.5",
    ]
    assert job["spec"]["options"]["wsi_cache"].endswith("/cache")
    csv_path.write_text("wsi,mpp\nnot-on-disk.svs,0.5\n")
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


@pytest.fixture
def cohort_extraction(tmp_path, monkeypatch):
    """One slide root with per-cohort subfolders, as a multi-cohort study is imported."""
    folder = tmp_path / "experiment"
    folder.mkdir()
    root = tmp_path / "drive-d" / "slides" / "colon"
    slides = []
    for cohort, name in (("rih", "SL-1.svs"), ("rih", "SL-2.svs"), ("TCGA", "TCGA-A6.svs")):
        slide = root / cohort / name
        slide.parent.mkdir(parents=True, exist_ok=True)
        slide.write_bytes(b"fixture slide")
        slides.append({"slideId": slide.stem, "patientId": slide.stem, "slidePath": str(slide)})
    excluded = root / "SURGEN" / "SR386.tiff"
    excluded.parent.mkdir(parents=True, exist_ok=True)
    excluded.write_bytes(b"fixture slide")
    store = ScientificStore(folder, "project-cohort")
    draft = store.create_draft("import", "test", {})
    store.publish_dataset(
        draft["id"],
        expected_revision=1,
        manifest={"kind": "dataset", "provenance": {"mapping": {"slideRoot": str(root)}}},
        artifacts={"records.json": json.dumps(slides).encode()},
        operation_id="dataset",
    )
    executor = FakeExecutor()
    service = ExtractionService(store, LocalFilesystem((tmp_path / "drive-d",)), executor)
    monkeypatch.setattr(
        "histopilot.adapters.trident.discover_runtime",
        lambda: {
            "available": True,
            "pythonPath": "/usr/bin/python3",
            "tridentRoot": str(tmp_path / "runtime"),
        },
    )
    monkeypatch.setattr(
        "histopilot.adapters.trident.build_command",
        lambda options, **kwargs: [kwargs["wsi_dir"], kwargs["custom_list_of_wsis"]],
    )
    dataset = store.list_datasets()[0]
    spec = ExtractionSpec(
        datasetId=dataset["id"], outputPath=str(folder / "trident"), options={"task": "seg"}
    )
    return service, spec, root


def test_cohort_manifest_selects_nested_slides_and_pins_each_source_mpp(cohort_extraction):
    service, spec, root = cohort_extraction
    manifest = service.store.folder / "colon_ready_wsi_mpp.csv"
    # A study-wide manifest: extra columns, and rows for cohorts this version excludes.
    manifest.write_text(
        "wsi,mpp,cohort\n"
        "rih/SL-1.svs,0.5016,RIH\n"
        "TCGA/TCGA-A6.svs,0.252,TCGA\n"
        "SURGEN/SR386.tiff,0.25,SurGen\n"
    )
    selected = spec.model_copy(
        update={"options": {"task": "seg", "custom_list_of_wsis": str(manifest)}}
    )
    preview = service.preview(selected)
    assert preview["canRun"], preview["findings"]
    assert preview["slideCount"] == 2
    assert preview["slideList"] == {
        "source": "list",
        "listPath": str(manifest),
        "sha256": preview["customListSha256"],
        "root": str(root),
        "initialCount": 3,
        "selectedCount": 2,
        "declaresMpp": True,
        "datasetFiltered": True,
        "outside": ["SURGEN/SR386.tiff"],
        "outsideCount": 1,
        "outsideExamples": ["SURGEN/SR386.tiff"],
        "unlisted": ["SL-2"],
        "unlistedCount": 1,
        "unlistedExamples": ["SL-2"],
    }
    job = submit(service, selected)
    # TRIDENT names outputs from the file stem and reads wsi paths relative to --wsi_dir.
    assert (service.folder / job["id"] / "slides.csv").read_text().splitlines() == [
        "wsi,mpp",
        "rih/SL-1.svs,0.5016",
        "TCGA/TCGA-A6.svs,0.252",
    ]
    assert job["command"][0] == str(root)


def test_a_dataset_alone_is_a_slide_source_and_reports_itself_as_one(cohort_extraction):
    """With no list and no folder the dataset's own linked files are the selection."""
    service, spec, _root = cohort_extraction
    preview = service.preview(spec)
    assert preview["canRun"], preview["findings"]
    assert preview["slideCount"] == 3
    assert preview["slideList"]["source"] == "dataset"
    assert preview["slideList"]["initialCount"] == 3
    assert preview["slideList"]["declaresMpp"] is False
    assert preview["slideList"]["datasetFiltered"] is False


def test_a_slide_folder_extracts_without_any_dataset(cohort_extraction):
    """Encoding depends on slide files; a dataset only narrows what was already selected."""
    service, spec, root = cohort_extraction
    folder = spec.model_copy(update={"datasetId": None, "slideRoot": str(root)})
    preview = service.preview(folder)
    assert preview["canRun"], preview["findings"]
    # Every slide under the root, including the one no dataset version claims.
    assert preview["slideCount"] == 4
    assert preview["slideList"]["source"] == "folder"
    assert preview["slideList"]["datasetFiltered"] is False
    job = submit(service, folder)
    assert (service.folder / job["id"] / "slides.csv").read_text().splitlines() == [
        "wsi",
        "rih/SL-1.svs",
        "rih/SL-2.svs",
        "SURGEN/SR386.tiff",
        "TCGA/TCGA-A6.svs",
    ]


def test_the_same_folder_narrowed_by_a_dataset_selects_only_its_slides(cohort_extraction):
    service, spec, root = cohort_extraction
    both = spec.model_copy(update={"slideRoot": str(root)})
    preview = service.preview(both)
    assert preview["slideCount"] == 3
    assert preview["slideList"]["initialCount"] == 4
    assert preview["slideList"]["datasetFiltered"] is True
    assert preview["slideList"]["outsideExamples"] == ["SURGEN/SR386.tiff"]


def test_extraction_rejects_selected_physical_slide_aliases(cohort_extraction):
    service, spec, root = cohort_extraction
    alias = root / "copied-identity.svs"
    alias.hardlink_to(root / "rih" / "SL-1.svs")
    independent = spec.model_copy(update={"datasetId": None, "slideRoot": str(root)})
    reviewed = service.preview(independent)
    assert not reviewed["canRun"]
    assert any(row["code"] == "DUPLICATE_SLIDE_ALIAS" for row in reviewed["findings"])
    with pytest.raises(StorageError, match="preflight"):
        service.submit(independent, reviewed["previewHash"], "duplicate-source")
    assert service.executor.launches == []
    # A dataset restriction excludes the alias, so it still selects a valid source.
    selected = service.preview(spec.model_copy(update={"slideRoot": str(root)}))
    assert selected["canRun"], selected["findings"]
    assert selected["slideCount"] == 3


def test_a_slide_list_needs_the_folder_its_paths_are_relative_to(cohort_extraction):
    service, spec, _root = cohort_extraction
    listing = service.store.folder / "list.csv"
    listing.write_text("wsi\nrih/SL-1.svs\n")
    with pytest.raises(StorageError) as error:
        service.preview(
            spec.model_copy(
                update={
                    "datasetId": None,
                    "options": {"task": "seg", "custom_list_of_wsis": str(listing)},
                }
            )
        )
    assert error.value.code == "SLIDE_ROOT_REQUIRED"


def test_changing_a_declared_mpp_cannot_reuse_an_existing_output(cohort_extraction):
    service, spec, _root = cohort_extraction
    manifest = service.store.folder / "list.csv"
    manifest.write_text("wsi,mpp\nrih/SL-1.svs,0.5016\n")
    selected = spec.model_copy(
        update={"options": {"task": "seg", "custom_list_of_wsis": str(manifest)}}
    )
    submit(service, selected)
    manifest.write_text("wsi,mpp\nrih/SL-1.svs,0.25\n")
    preview = service.preview(selected)
    assert not preview["canRun"]
    assert any(item["code"] == "OUTPUT_CONFIG_CHANGED" for item in preview["findings"])


def test_a_partly_declared_mpp_column_never_reaches_trident(cohort_extraction):
    service, spec, _root = cohort_extraction
    manifest = service.store.folder / "list.csv"
    manifest.write_text("wsi,mpp\nrih/SL-1.svs,0.5016\nTCGA/TCGA-A6.svs,\n")
    selected = spec.model_copy(
        update={"options": {"task": "seg", "custom_list_of_wsis": str(manifest)}}
    )
    with pytest.raises(StorageError) as error:
        service.preview(selected)
    assert error.value.code == "INVALID_SLIDE_LIST"


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


def test_uploaded_slide_list_extracts_without_dataset_and_preserves_mpp(cohort_extraction):
    import base64

    service, original, root = cohort_extraction
    content = b"wsi,mpp\nrih/SL-1.svs,0.5016\nSURGEN/SR386.tiff,0.25\n"
    spec = ExtractionSpec(
        slideRoot=str(root),
        slideList={
            "filename": "selection.csv",
            "contentBase64": base64.b64encode(content).decode(),
        },
        outputPath=original.outputPath,
        options={"task": "seg"},
    )
    preview = service.preview(spec)
    assert preview["canRun"], preview["findings"]
    assert preview["slideCount"] == 2
    assert preview["spec"]["datasetId"] is None
    assert preview["spec"]["slideList"] == spec.slideList.model_dump()
    assert preview["slideList"]["filename"] == "selection.csv"
    assert preview["slideList"]["datasetFiltered"] is False
    # The normalized API preview must round-trip and produce an identical run request.
    reviewed_spec = ExtractionSpec.model_validate(preview["spec"])
    job = service.submit(reviewed_spec, preview["previewHash"], "uploaded-selection")
    assert (service.folder / job["id"] / "slides.csv").read_bytes() == content.replace(
        b"\n", b"\r\n"
    )


def test_uploaded_slide_list_can_be_narrowed_by_a_dataset(cohort_extraction):
    import base64

    service, original, root = cohort_extraction
    spec = ExtractionSpec(
        datasetId=original.datasetId,
        slideRoot=str(root),
        slideList={
            "filename": "selection.csv",
            "contentBase64": base64.b64encode(b"wsi\nrih/SL-1.svs\nSURGEN/SR386.tiff\n").decode(),
        },
        outputPath=original.outputPath,
        options={"task": "seg"},
    )
    preview = service.preview(spec)
    assert preview["canRun"]
    assert preview["slideCount"] == 1
    assert preview["slideList"]["outsideCount"] == 1
    assert preview["slideList"]["datasetFiltered"] is True


def test_uploaded_slide_list_rejects_invalid_encoding_and_conflicting_sources(cohort_extraction):
    from pydantic import ValidationError

    service, original, root = cohort_extraction
    spec = ExtractionSpec(
        slideRoot=str(root),
        slideList={"filename": "selection.csv", "contentBase64": "not-base64!"},
        outputPath=original.outputPath,
        options={"task": "seg"},
    )
    with pytest.raises(StorageError, match="not valid base64"):
        service.preview(spec)
    with pytest.raises(ValidationError, match="Choose one slide list"):
        ExtractionSpec.model_validate(
            {**spec.model_dump(), "options": {"custom_list_of_wsis": "/another.csv"}}
        )


def test_legacy_extraction_preview_and_retry_keep_their_hashes(extraction):
    from histopilot.application.extractions import _hash

    service, spec, executor, _slides = extraction
    legacy_spec = {
        "datasetId": spec.datasetId,
        "slideRoot": None,
        "recursive": True,
        "outputPath": spec.outputPath,
        "options": {"task": "seg"},
    }
    request = ExtractionSpec.model_validate({**legacy_spec, "slideList": None})
    assert request.model_dump(mode="json") == legacy_spec
    assert json.loads(request.model_dump_json()) == legacy_spec
    preview, slides = service._prepare(request)
    assert "slideList" not in preview["spec"]
    legacy_preview_hash = _hash(
        {
            **{key: value for key, value in preview.items() if key != "previewHash"},
            "spec": {key: value for key, value in preview["spec"].items() if key != "slideList"},
            "slides": slides,
        }
    )
    assert preview["previewHash"] == legacy_preview_hash
    job = service.submit(request, legacy_preview_hash, "legacy-extraction-submit")
    path = service.folder / job["id"] / "job.json"
    recorded = json.loads(path.read_text())
    assert recorded["requestHash"] == _hash(legacy_spec)
    # Replay persisted old metadata rather than relying on this version's serializer.
    recorded["requestHash"] = _hash(legacy_spec)
    recorded["spec"].pop("slideList", None)
    _write(path, recorded)
    replay = service.submit(request, legacy_preview_hash, "legacy-extraction-submit")
    assert replay["id"] == job["id"]
    assert len(executor.launches) == 1
