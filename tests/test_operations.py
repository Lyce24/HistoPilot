"""Archive integrity, restore isolation, explicit source changes and shared leases."""

import hashlib
import json
import os
import stat
import subprocess
import sys
import zipfile
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from histopilot.api import create_app
from histopilot.application.operations import (
    MANIFEST,
    StudyPortability,
    operations_inventory,
    relink_source,
    restore_archive,
    source_inventory,
    verify_archive,
)
from histopilot.application.portability_jobs import PortabilityJobs
from histopilot.archive_cli import launch_archive_recovery
from histopilot.config import Settings
from histopilot.schemas.operations import PortabilityRequest, RelinkSource
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import ScientificStore
from histopilot.workers import resource_reservation as reservations
from histopilot.workers.portability import run


@pytest.fixture
def project(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    settings = Settings(workspace=tmp_path / "workspace", data_roots=(data,))
    with TestClient(create_app(settings), base_url="http://127.0.0.1:8787") as client:
        client.headers["X-HistoPilot-Token"] = client.get("/api/v1/session").json()["token"]
        response = client.post(
            "/api/v1/projects",
            json={
                "name": "Portable study",
                "storagePath": str(settings.workspace / "study"),
                "slidePath": str(data),
            },
        )
        assert response.status_code == 201, response.text
        projects = client.app.state.projects
        store = projects.scientific_store(response.json()["id"])
        yield store, projects, client, data


def test_archive_roundtrip_preserves_scientific_data_and_reviews(project):
    store, projects, _, data = project
    draft = store.create_draft(kind="import", name="Import", payload={"source": "A"})
    reviews = store.folder / "slide-reviews"
    reviews.mkdir()
    (reviews / "case.json").write_text('{"decision":"accept","note":"Tumor present"}')
    external = data / "slide.svs"
    external.write_bytes(b"external source is not copied")
    archive = projects.database.workspace / "study.zip"
    result = StudyPortability(store, projects.storage).export(str(archive))
    assert result["verified"] and result["externalSourcePolicy"] == "references-only"
    with zipfile.ZipFile(archive) as zipped:
        assert "project/slide-reviews/case.json" in zipped.namelist()
        assert not any(name.endswith("slide.svs") for name in zipped.namelist())
        assert not any(name.endswith("-wal") for name in zipped.namelist())
    destination = projects.database.workspace / "restored"
    restored = restore_archive(str(archive), str(destination), projects.storage)
    assert restored["verified"] and restored["relocated"]
    assert ScientificStore(destination, store.project_id).get_draft(draft["id"]) == draft
    assert (destination / "slide-reviews/case.json").read_bytes() == (
        reviews / "case.json"
    ).read_bytes()
    assert external.read_bytes() == b"external source is not copied"


def rewrite_zip(archive, change):
    with zipfile.ZipFile(archive) as original:
        entries = {info.filename: original.read(info.filename) for info in original.infolist()}
    change(entries)
    with zipfile.ZipFile(archive, "w") as target:
        for name, data in entries.items():
            target.writestr(name, data)


def test_checksum_corruption_is_rejected_before_restore_publication(project):
    store, projects, _, _ = project
    archive = projects.database.workspace / "study.zip"
    StudyPortability(store, projects.storage).export(str(archive))
    rewrite_zip(
        archive,
        lambda entries: entries.__setitem__(
            "project/histopilot-project.json",
            entries["project/histopilot-project.json"].replace(b"Portable", b"Tampered"),
        ),
    )
    destination = projects.database.workspace / "restored"
    with pytest.raises(StorageError, match="Checksum"):
        restore_archive(str(archive), str(destination), projects.storage)
    assert not destination.exists()
    assert not list(destination.parent.glob(".histopilot-restore-*"))


@pytest.mark.parametrize(
    "unsafe", ["../outside", "/tmp/escape", "nested/../../escape", "a\\b", "a//b"]
)
def test_archive_member_path_traversal_rejected(project, unsafe):
    store, projects, _, _ = project
    archive = projects.database.workspace / "study.zip"
    StudyPortability(store, projects.storage).export(str(archive))

    def tamper(entries):
        manifest = json.loads(entries[MANIFEST])
        manifest["files"].append(
            {"path": unsafe, "size": 1, "sha256": hashlib.sha256(b"x").hexdigest()}
        )
        entries[MANIFEST] = json.dumps(manifest).encode()
        entries[f"project/{unsafe}"] = b"x"

    rewrite_zip(archive, tamper)
    with pytest.raises(StorageError, match="unsafe path"):
        verify_archive(archive)


def test_archive_rejects_duplicate_and_symlink_entries(project):
    store, projects, _, _ = project
    archive = projects.database.workspace / "study.zip"
    StudyPortability(store, projects.storage).export(str(archive))
    with zipfile.ZipFile(archive, "a") as zipped:
        info = zipfile.ZipInfo("project/alias")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        zipped.writestr(info, b"/tmp/elsewhere")
    with pytest.raises(StorageError, match="unexpected"):
        verify_archive(archive)


def test_export_refuses_links_overwrites_and_active_jobs(project, monkeypatch):
    store, projects, _, data = project
    (store.folder / "alias").symlink_to(data)
    archive = projects.database.workspace / "study.zip"
    with pytest.raises(StorageError, match="symbolic"):
        StudyPortability(store, projects.storage).export(str(archive))
    (store.folder / "alias").unlink()
    monkeypatch.setattr(
        "histopilot.application.operations.CleanupService._catalog",
        lambda _: {
            "items": [{"job": {"status": "running", "busy": True}}],
        },
    )
    with pytest.raises(StorageError, match="active jobs"):
        StudyPortability(store, projects.storage).export(str(archive))
    archive.write_bytes(b"keep")
    with pytest.raises(StorageError, match="already exists"):
        StudyPortability(store, projects.storage).export(str(archive))
    assert archive.read_bytes() == b"keep"


def test_restore_never_replaces_an_existing_folder(project):
    store, projects, _, _ = project
    archive = projects.database.workspace / "study.zip"
    StudyPortability(store, projects.storage).export(str(archive))
    destination = projects.database.workspace / "exists"
    destination.mkdir()
    with pytest.raises(StorageError, match="new folder"):
        restore_archive(str(archive), str(destination), projects.storage)
    assert destination.is_dir()


def test_relink_is_compare_and_swap_and_keeps_frozen_paths(project):
    store, projects, _, data = project
    original = source_inventory(store, projects.storage)["sources"][0]
    moved = data / "moved"
    moved.mkdir()
    spec = RelinkSource(
        sourceId=original["id"], expectedPath=original["path"], replacementPath=str(moved)
    )
    result = relink_source(projects, store.project_id, spec)
    assert result["project"]["sources"][0]["path"] == str(moved)
    with pytest.raises(StorageError, match="another tab"):
        relink_source(projects, store.project_id, spec)
    assert "frozen" in result["note"]


def test_missing_slide_references_are_detected_in_frozen_artifacts(project):
    store, projects, _, data = project
    draft = store.create_draft("import", "Data", {})
    missing = str(data / "unavailable-slide.svs")
    store.publish_dataset(
        draft["id"],
        expected_revision=1,
        manifest={"kind": "dataset", "name": "Data"},
        artifacts={"records.json": json.dumps([{"slidePath": missing}]).encode()},
        operation_id="source-inventory",
    )
    result = source_inventory(store, projects.storage)
    assert result["missingReferences"] == [{"path": missing, "reason": "missing"}]


def test_authenticated_operations_api_executes_archive_worker(project, monkeypatch):
    store, projects, client, _ = project
    monkeypatch.setattr(
        "histopilot.application.portability_jobs.PortabilityExecutor.available", lambda _: True
    )
    monkeypatch.setattr(
        "histopilot.application.portability_jobs.PortabilityExecutor.running",
        lambda _self, _session: False,
    )
    monkeypatch.setattr(
        "histopilot.application.portability_jobs.PortabilityExecutor.launch",
        lambda _self, _session, _runner, plan: run(plan),
    )
    prefix = f"/api/v1/projects/{store.project_id}/operations"
    assert client.get(prefix).status_code == 200
    archive = str(projects.database.workspace / "api-study.zip")
    response = client.post(
        prefix + "/archives",
        json={"action": "export", "archivePath": archive, "operationId": "api-export"},
    )
    assert response.status_code == 202, response.text
    job = response.json()
    assert job["status"] == "completed", job
    assert client.get(prefix + f"/archives/{job['id']}").json() == job
    assert client.get(prefix + "/archives").json()["jobs"] == [job]
    client.headers.pop("X-HistoPilot-Token")
    assert client.post(
        prefix + "/archives",
        json={"action": "export", "archivePath": archive, "operationId": "unauthorized"},
    ).status_code in {401, 403}


class InlineExecutor:
    def available(self):
        return True

    def running(self, _session):
        return False

    def launch(self, _session, _runner, plan):
        run(plan)


def test_durable_archive_submission_is_idempotent_and_isolated(project):
    store, projects, _, _ = project
    jobs = PortabilityJobs(projects, projects.storage, executor=InlineExecutor())
    request = PortabilityRequest(
        action="export",
        archivePath=str(projects.database.workspace / "study.zip"),
        operationId="test-archive",
    )
    first = jobs.submit(store.project_id, request)
    assert first["status"] == "completed", first
    assert first["result"]["verified"]
    assert jobs.submit(store.project_id, request) == first
    assert jobs.list(store.project_id)["jobs"] == [first]
    with pytest.raises(StorageError, match="different archive request"):
        jobs.submit(store.project_id, request.model_copy(update={"action": "verify"}))


def test_lost_launch_acknowledgement_does_not_clobber_completed_receipt(project):
    store, projects, _, _ = project

    class LostAcknowledgement(InlineExecutor):
        def launch(self, _session, _runner, plan):
            run(plan)
            raise RuntimeError("Response lost after worker completed")

    jobs = PortabilityJobs(projects, projects.storage, executor=LostAcknowledgement())
    request = PortabilityRequest(
        action="export",
        archivePath=str(projects.database.workspace / "study.zip"),
        operationId="lost-ack",
    )
    result = jobs.submit(store.project_id, request)
    assert result["status"] == "completed"
    assert result["result"]["verified"]
    assert result["error"] is None


def test_archive_cancel_retry_and_active_request_coalescing_survive_reconnect(project):
    store, projects, _, _ = project

    class QueuedExecutor(InlineExecutor):
        def __init__(self):
            self.sessions = set()
            self.calls = []

        def launch(self, session, _runner, plan):
            self.sessions.add(session)
            self.calls.append(plan)

        def running(self, session):
            return session in self.sessions

    executor = QueuedExecutor()
    jobs = PortabilityJobs(projects, projects.storage, executor=executor)
    request = PortabilityRequest(
        action="export",
        archivePath=str(projects.database.workspace / "study.zip"),
        operationId="first-request",
    )
    first = jobs.submit(store.project_id, request)
    duplicate = jobs.submit(
        store.project_id, request.model_copy(update={"operationId": "reconnected-request"})
    )
    assert first["id"] == duplicate["id"]
    assert len(executor.calls) == 1
    assert jobs.cancel(store.project_id, first["id"])["status"] == "cancelling"
    assert run(executor.calls[0])["status"] == "cancelled"
    assert not Path(request.archivePath).exists()
    executor.sessions.clear()
    jobs.executor = InlineExecutor()
    retried = jobs.retry(store.project_id, first["id"])
    assert retried["status"] == "completed"
    assert retried["attempts"] == 2


def test_export_recovers_publication_after_worker_receipt_was_lost(project):
    store, projects, _, _ = project
    archive = projects.database.workspace / "study.zip"
    service = StudyPortability(store, projects.storage)
    first = service.export(str(archive), operation_id="publication")
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    recovered = service.export(str(archive), operation_id="publication")
    assert recovered["recovered"]
    assert recovered["manifestSha256"] == first["manifestSha256"]
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == checksum
    with pytest.raises(StorageError, match="already exists"):
        service.export(str(archive), operation_id="different-operation")


def test_export_destination_race_preserves_other_file(project):
    store, projects, _, _ = project
    archive = projects.database.workspace / "study.zip"

    def publish_race(value):
        if value["stage"] == "publishing":
            archive.write_bytes(b"another operation owns this file")

    with pytest.raises(StorageError, match="another operation"):
        StudyPortability(store, projects.storage).export(str(archive), progress=publish_race)
    assert archive.read_bytes() == b"another operation owns this file"


def test_standalone_restore_works_when_original_project_is_gone(project):
    store, projects, _, _ = project
    archive = projects.database.workspace / "study.zip"
    StudyPortability(store, projects.storage).export(str(archive))
    store.folder.rename(store.folder.with_name("unregistered-original"))
    destination = projects.database.workspace / "restored"
    settings = Settings(workspace=projects.database.workspace, data_roots=projects.storage.roots)
    submitted = launch_archive_recovery(settings, archive, destination, executor=InlineExecutor())
    state = json.loads(Path(submitted["statePath"]).read_text())
    assert state["status"] == "completed", state
    assert projects.open(str(destination))["id"] == store.project_id


def test_unified_inventory_includes_host_capacity(project):
    store, projects, _, _ = project
    result = operations_inventory(store, projects.storage)
    assert result["projectId"] == store.project_id
    assert result["capacity"]["cpus"] > 0
    assert result["jobs"] == []


def test_preparation_reservations_hold_all_requested_gpus_atomically(tmp_path, monkeypatch):
    registry = tmp_path / "leases"
    registry.mkdir()
    job = tmp_path / "job"
    job.mkdir()
    observations = []
    attempts = 0

    @contextmanager
    def leases():
        nonlocal attempts
        attempts += 1
        active = [{"gpu": 1, "cpus": 2, "ramGb": 1, "runsPerGpu": 1}] if attempts == 1 else []
        yield registry, active

    def sleep(_seconds):
        observations.append(list(registry.glob("lease-*.json")))

    monkeypatch.setattr(reservations, "_leases", leases)
    monkeypatch.setattr(reservations, "_capacity", lambda: (32, 64))
    monkeypatch.setattr(reservations.time, "sleep", sleep)
    resources = reservations.preparation_resources("extraction", {"gpus": [0, 1], "max_workers": 0})
    with reservations.reserve_preparation(job, "extraction", resources, lambda: False):
        saved = [json.loads(path.read_text()) for path in registry.glob("lease-*.json")]
        assert sorted(row["gpu"] for row in saved) == [0, 1]
        assert sum(row["cpus"] for row in saved) == 2
        assert sum(row["ramGb"] for row in saved) == 8
        assert all(row["supervisor"]["pid"] == os.getpid() for row in saved)
    assert observations == [[]]
    assert not list(registry.glob("lease-*.json"))
    assert json.loads((job / "resources.json").read_text())["status"] == "released"


def test_preparation_wait_honors_cancellation_without_leaking_lease(tmp_path, monkeypatch):
    registry = tmp_path / "leases"
    registry.mkdir()
    job = tmp_path / "job"
    job.mkdir()
    cancelled = False

    @contextmanager
    def leases():
        yield registry, [{"gpu": None, "cpus": 8, "ramGb": 1, "runsPerGpu": 1}]

    def sleep(_seconds):
        nonlocal cancelled
        cancelled = True

    monkeypatch.setattr(reservations, "_leases", leases)
    monkeypatch.setattr(reservations, "_capacity", lambda: (8, 64))
    monkeypatch.setattr(reservations.time, "sleep", sleep)
    with pytest.raises(ValueError, match="Cancelled"):
        with reservations.reserve_preparation(
            job, "packing", reservations.preparation_resources("packing"), lambda: cancelled
        ):
            pytest.fail("A blocked job must not execute")
    assert not list(registry.glob("lease-*.json"))


def test_managed_extraction_supervisor_runs_both_phases_and_releases_resources(tmp_path):
    from histopilot.adapters.trident import runner

    plan = {
        "command": [sys.executable, "-c", "print('extraction complete')"],
        "validationCommand": [sys.executable, "-c", "print('validation complete')"],
        "resultPath": str(tmp_path / "result.json"),
        "processPath": str(tmp_path / "process.json"),
        "logPath": str(tmp_path / "worker.log"),
        "cancelPath": str(tmp_path / "cancelled"),
        "resources": {
            "gpuIds": [],
            "cpuThreadsPerRun": 1,
            "dataLoaderWorkers": 0,
            "ramGbPerRun": 0.01,
            "runsPerGpu": 1,
            "maxConcurrentRuns": 1,
        },
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    result = subprocess.run(
        [sys.executable, runner.__file__, str(path)], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path / "result.json").read_text())["state"] == "succeeded"
    assert json.loads((tmp_path / "resources.json").read_text())["status"] == "released"
    log = (tmp_path / "worker.log").read_text()
    assert "extraction complete" in log and "validation complete" in log
