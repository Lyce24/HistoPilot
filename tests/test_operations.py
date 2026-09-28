"""Archive integrity, restore isolation, explicit source changes and shared leases."""

import hashlib
import json
import stat
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from support.task_center import begin, conclude, task_environment
from support.workers import run_archive

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
from histopilot.taskcenter.adapters.archive import ArchiveAdapter
from histopilot.workers.packing_process import write_json
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


def test_authenticated_operations_api_executes_archive_worker(project, task_center):
    store, projects, client, _ = project
    prefix = f"/api/v1/projects/{store.project_id}/operations"
    assert client.get(prefix).status_code == 200
    archive = str(projects.database.workspace / "api-study.zip")
    response = client.post(
        prefix + "/archives",
        json={"action": "export", "archivePath": archive, "operationId": "api-export"},
    )
    assert response.status_code == 202, response.text
    queued = response.json()
    assert queued["status"] == "queued" and queued["executor"] == "task-center"
    run_archive(task_center.store, queued)
    job = client.get(prefix + f"/archives/{queued['id']}").json()
    assert job["status"] == "completed", job
    assert job["result"]["verified"]
    assert client.get(prefix + "/archives").json()["jobs"] == [job]
    client.headers.pop("X-HistoPilot-Token")
    assert client.post(
        prefix + "/archives",
        json={"action": "export", "archivePath": archive, "operationId": "unauthorized"},
    ).status_code in {401, 403}


def archive_jobs(projects, task_center):
    return PortabilityJobs(projects, projects.storage, task_center=task_center.client)


def test_durable_archive_submission_is_idempotent_and_isolated(project, task_center):
    store, projects, _, _ = project
    jobs = archive_jobs(projects, task_center)
    request = PortabilityRequest(
        action="export",
        archivePath=str(projects.database.workspace / "study.zip"),
        operationId="test-archive",
    )
    queued = jobs.submit(store.project_id, request)
    assert queued["status"] == "queued"
    run_archive(task_center.store, queued)
    first = jobs.get(store.project_id, queued["id"])
    assert first["status"] == "completed", first
    assert first["result"]["verified"]
    assert jobs.submit(store.project_id, request) == first
    assert jobs.list(store.project_id)["jobs"] == [first]
    assert [task["attempt"] for task in task_center.tasks(kind="archive")] == [1]
    with pytest.raises(StorageError, match="different archive request"):
        jobs.submit(store.project_id, request.model_copy(update={"action": "verify"}))


def test_a_retry_after_a_lost_conclusion_keeps_the_completed_receipt(project, task_center):
    store, projects, _, _ = project
    jobs = archive_jobs(projects, task_center)
    request = PortabilityRequest(
        action="export",
        archivePath=str(projects.database.workspace / "study.zip"),
        operationId="lost-conclusion",
    )
    job = jobs.submit(store.project_id, request)
    task = task_center.task(job["taskId"])
    assert begin(task_center.store, task["id"], ArchiveAdapter())
    with task_environment(task_center.task(task["id"])):
        assert run(Path(task["command"]["argv"][-1]))["status"] == "completed"
    # The worker completed, but the runner lost it before concluding the attempt.
    task_center.finish(task["id"], "interrupted", returncode=None, reason="lost")
    receipt = json.loads(Path(task["command"]["result"]).read_text())
    assert receipt["status"] == "completed" and receipt["error"] is None
    assert jobs.retry(store.project_id, job["id"])["status"] == "queued"
    run_archive(task_center.store, job)  # the retry finds the operation complete
    assert task_center.task(task["id"])["exit"]["reason"] == "already-complete"
    result = jobs.get(store.project_id, job["id"])
    assert result["status"] == "completed"
    assert result["result"] == receipt["result"] and result["result"]["verified"]
    assert result["error"] is None


def test_archive_cancel_retry_and_active_request_coalescing_survive_reconnect(
    project, task_center
):
    store, projects, _, _ = project
    jobs = archive_jobs(projects, task_center)
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
    assert len(task_center.tasks(kind="archive")) == 1
    adapter = ArchiveAdapter()
    # A queued operation is cancelled at once; a running worker stops at its next file.
    assert begin(task_center.store, first["taskId"], adapter)
    assert jobs.cancel(store.project_id, first["id"])["status"] == "cancelling"
    task = task_center.task(first["taskId"])
    with task_environment(task):
        assert run(Path(task["command"]["argv"][-1]))["status"] == "cancelled"
    assert conclude(task_center.store, task["id"], adapter, 1)["state"] == "cancelled"
    assert jobs.get(store.project_id, first["id"])["status"] == "cancelled"
    assert not Path(request.archivePath).exists()
    retried = jobs.retry(store.project_id, first["id"])
    assert retried["status"] == "queued"
    run_archive(task_center.store, retried)
    retried = jobs.get(store.project_id, first["id"])
    assert retried["status"] == "completed"
    # The retry is the operation task's second attempt.
    assert retried["task"]["attempt"] == 2


def test_an_archive_operation_from_before_the_task_center_is_read_only(project, task_center):
    store, projects, _, _ = project
    jobs = archive_jobs(projects, task_center)
    request = PortabilityRequest(
        action="export",
        archivePath=str(projects.database.workspace / "study.zip"),
        operationId="legacy-archive",
    )
    job = jobs.submit(store.project_id, request)
    folder = jobs._folder(job["id"])
    # Model an operation started in tmux: its record names no Task Center task.
    saved = json.loads((folder / "state.json").read_text())
    for key in ("executionMode", "taskId", "ownerKey"):
        saved.pop(key)
    saved.update(status="running", sessionName="histopilot-archive-0123456789abcdef0123")
    write_json(folder / "state.json", saved)
    view = jobs.get(store.project_id, job["id"])
    assert view["status"] == "interrupted" and "executor" not in view
    assert "Created before the Task Center" in view["error"]
    for action in (jobs.cancel, jobs.retry):
        with pytest.raises(StorageError) as refused:
            action(store.project_id, job["id"])
        assert refused.value.code == "CREATED_BEFORE_TASK_CENTER"
    assert json.loads((folder / "state.json").read_text()) == saved
    assert not (folder / "cancel.requested").exists()
    # A finished one shows exactly its saved record.
    write_json(folder / "state.json", {**saved, "status": "completed"})
    assert jobs.get(store.project_id, job["id"]) == {**saved, "status": "completed"}


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


def test_standalone_restore_works_when_original_project_is_gone(project, task_center):
    store, projects, _, _ = project
    archive = projects.database.workspace / "study.zip"
    StudyPortability(store, projects.storage).export(str(archive))
    store.folder.rename(store.folder.with_name("unregistered-original"))
    destination = projects.database.workspace / "restored"
    settings = Settings(workspace=projects.database.workspace, data_roots=projects.storage.roots)
    submitted = launch_archive_recovery(
        settings, archive, destination, task_center=task_center.client
    )
    # Queued in the Task Center like any archive operation, without a registered project.
    assert submitted["status"] == "queued" and submitted["executionMode"] == "task-center"
    task = task_center.task(submitted["taskId"])
    assert (task["kind"], task["labels"]["action"]) == ("archive", "restore")
    assert task_center.store.owner(task["ownerKey"])["kind"] == "archive"
    state = run_archive(task_center.store, submitted)
    assert state["status"] == "completed", state
    assert json.loads(Path(submitted["statePath"]).read_text())["status"] == "completed"
    assert projects.open(str(destination))["id"] == store.project_id


def test_recovery_commands_queue_in_the_task_center_and_wake_the_runner(project, monkeypatch):
    from typer.testing import CliRunner

    from histopilot.cli import app
    from histopilot.taskcenter import launcher
    from histopilot.taskcenter.client import default_client

    store, projects, _, data = project
    archive = data / "study.zip"
    StudyPortability(store, projects.storage).export(str(archive))
    woken = []
    monkeypatch.setattr(launcher, "ensure_runner", lambda: woken.append(1) or {"started": True})
    arguments = ["--workspace", str(projects.database.workspace), "--data-root", str(data)]
    result = CliRunner().invoke(app, ["verify-study", str(archive), *arguments])
    assert result.exit_code == 0, result.output
    assert "histopilot runner status" in result.output and "tmux attach" not in result.output
    assert woken == [1]
    [task] = default_client().store.list(kinds=("archive",), limit=None)
    assert task["state"] == "queued" and task["labels"]["action"] == "verify"
    missing = CliRunner().invoke(app, ["verify-study", str(data / "absent.zip"), *arguments])
    assert missing.exit_code != 0


def test_unified_inventory_includes_host_capacity(project):
    store, projects, _, _ = project
    result = operations_inventory(store, projects.storage)
    assert result["projectId"] == store.project_id
    assert result["capacity"]["cpus"] > 0
    assert result["jobs"] == []
