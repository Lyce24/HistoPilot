"""Workspace cleanup cannot race job starts or erase historical validation receipts."""

import hashlib
import runpy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from support.t1 import lose_batch, run_managed, task_ids

from histopilot.storage.lifecycle import LifecycleStore, lifecycle_guard
from histopilot.storage.project_lock import StorageError
from histopilot.workers.pack_features import run_job
from histopilot.workers.packing_process import write_json
from histopilot.workers.training_process import save_state

packing_support = runpy.run_path(str(Path(__file__).with_name("test_feature_packs.py")))
extraction_support = runpy.run_path(str(Path(__file__).with_name("test_extractions.py")))
training_support = runpy.run_path(str(Path(__file__).with_name("test_training_execution.py")))


def change(service, kind, identity, state):
    lifecycle = LifecycleStore(service.store.folder, service.store.project_id)
    revision = lifecycle.read()["revision"]
    operation = f"{kind}:{identity}:{state}:{revision}"
    lifecycle.apply(
        {f"{kind}:{identity}": state},
        operation,
        hashlib.sha256(operation.encode()).hexdigest(),
        revision,
    )


# The job modules' fixtures; each service queues its work in this test's Task Center.


@pytest.fixture
def packing(tmp_path, task_center):
    return packing_support["packs"].__wrapped__(tmp_path, task_center)


@pytest.fixture
def extraction(tmp_path, monkeypatch, task_center):
    return extraction_support["extractions"].__wrapped__(tmp_path, monkeypatch, task_center)


@pytest.fixture
def training(tmp_path, monkeypatch, task_center):
    return training_support["tc_execution"].__wrapped__(tmp_path, monkeypatch, task_center)


def complete(service, center, job, monkeypatch):
    """Run the packing worker as its task, then record the task as the runner would."""
    [task] = center.tasks(group=job["id"])
    result = run_managed(
        center, task, monkeypatch, lambda: run_job(service.folder / job["id"] / "plan.json")
    )
    center.finish(task["id"], "succeeded" if result["state"] == "succeeded" else "failed")
    return result


@pytest.mark.parametrize("kind", ["dataset", "project"])
def test_extraction_rejects_trashed_inputs_before_creating_job(extraction, task_center, kind):
    service, spec, _ = extraction
    before = task_ids(task_center)
    preview = service.preview(spec)
    identity = spec.datasetId if kind == "dataset" else service.store.project_id
    change(service, kind, identity, "trashed")
    with pytest.raises(StorageError, match="Trash"):
        service.submit(spec, preview["previewHash"], "blocked")
    assert task_ids(task_center) == before
    assert not service.folder.exists()
    assert not Path(spec.outputPath).exists()


@pytest.mark.parametrize("kind", ["configuration", "project"])
def test_packing_rejects_trashed_inputs_before_creating_job(packing, task_center, kind):
    service, spec, _ = packing
    before = task_ids(task_center)
    preview = service.preview(spec)
    identity = spec.featureSetId if kind == "configuration" else service.store.project_id
    change(service, kind, identity, "trashed")
    with pytest.raises(StorageError, match="Trash"):
        service.submit(spec, preview["previewHash"], "blocked")
    assert task_ids(task_center) == before
    assert not service.folder.exists()


@pytest.mark.parametrize("reference", ["batch", "protocol", "bundle", "dataset", "project"])
def test_training_rejects_trashed_records_before_creating_execution(
    training, task_center, reference
):
    service, batch, _ = training
    before = task_ids(task_center)
    inputs = batch["manifest"]["spec"]["inputs"]
    kind, identity = {
        "batch": ("configuration", batch["id"]),
        "protocol": ("configuration", inputs["protocolId"]),
        "bundle": ("configuration", inputs["featureBundleId"]),
        "dataset": ("dataset", batch["manifest"]["datasetId"]),
        "project": ("project", service.store.project_id),
    }[reference]
    change(service, kind, identity, "trashed")
    with pytest.raises(StorageError, match="Trash"):
        service.launch(batch["id"], "blocked")
    assert task_ids(task_center) == before
    assert not (service.store.folder / "training").exists()


@pytest.mark.parametrize("job_type", ["packing", "extraction", "training"])
def test_lifecycle_guard_covers_the_entire_job_start(job_type, request, monkeypatch):
    service, record, *_ = request.getfixturevalue(job_type)

    def attempt_cleanup():
        with lifecycle_guard(service.store.folder, timeout=0):
            pytest.fail("Cleanup acquired the guard during a job start")

    def start(*_args, **_kwargs):
        with ThreadPoolExecutor(max_workers=1) as pool:
            with pytest.raises(StorageError) as caught:
                pool.submit(attempt_cleanup).result(timeout=2)
        assert caught.value.code == "PROJECT_BUSY"
        return {"guarded": True}

    if job_type == "training":
        monkeypatch.setattr(service, "_launch", start)
        assert service.launch(record["id"], "test") == {"guarded": True}
    else:
        monkeypatch.setattr(service, "_submit", start)
        assert service.submit(record, "unused", "test") == {"guarded": True}


def test_archived_pack_receipt_still_resolves_but_trash_blocks_it(
    packing, task_center, monkeypatch
):
    service, spec, _ = packing
    job = packing_support["submit"](service, spec)
    result = complete(service, task_center, job, monkeypatch)
    artifact = result["artifact"]
    service.select(spec.featureSetId, artifact["id"])
    change(service, "packing", job["id"], "archived")
    assert service.list()["jobs"] == []
    assert service.list()["artifacts"] == []
    assert service.get(job["id"])["state"] == "succeeded"
    assert service.artifact(artifact["id"])["jobId"] == job["id"]
    assert service.validation_for(spec.featureSetId)["jobId"] == job["id"]
    assert service.list(include_inactive=True)["artifacts"][0]["id"] == artifact["id"]

    change(service, "packing", job["id"], "trashed")
    with pytest.raises(StorageError, match="Trash"):
        service.artifact(artifact["id"])
    with pytest.raises(StorageError, match="Trash"):
        service.get(job["id"])
    assert service.validation_for(spec.featureSetId) is None
    assert service.list()["selections"][spec.featureSetId] is None
    assert not service.selection_for(spec.featureSetId)["current"]
    assert service.get(job["id"], include_inactive=True)["state"] == "succeeded"
    assert Path(artifact["outputPath"]).exists()
    with pytest.raises(StorageError, match="Trash"):
        service.select(spec.featureSetId, artifact["id"])


def test_extraction_archive_hides_terminal_job_and_keeps_direct_history(extraction, task_center):
    service, spec, _ = extraction
    job = extraction_support["submit"](service, spec)
    task_center.finish(job["taskId"], "interrupted", returncode=None, reason="lost")
    change(service, "extraction", job["id"], "archived")
    assert service.list()["jobs"] == []
    assert service.get(job["id"])["state"] == "interrupted"
    assert service.list(include_inactive=True)["jobs"][0]["id"] == job["id"]
    change(service, "extraction", job["id"], "trashed")
    with pytest.raises(StorageError, match="Trash"):
        service.get(job["id"])
    assert service.get(job["id"], include_inactive=True)["state"] == "interrupted"


@pytest.mark.parametrize("job_type", ["packing", "extraction", "training"])
def test_active_jobs_remain_visible_and_cancellable_if_lifecycle_is_inconsistent(
    job_type, request, task_center
):
    service, record, *_ = request.getfixturevalue(job_type)
    if job_type == "training":
        service.launch(record["id"], "launch")
        change(service, "configuration", record["id"], "trashed")
        assert len(service.list()["items"]) == 1
        assert service.execution(record["id"], include_inactive=True)["status"] == "queued"
        assert service.cancel(record["id"], "cancel")["cancelRequested"]
    else:
        preview = service.preview(record)
        job = service.submit(record, preview["previewHash"], "launch")
        # The worker is running: a queued task would be cancelled at once instead.
        task_center.start(job["taskId"])
        change(service, job_type, job["id"], "trashed")
        assert service.list()["jobs"][0]["id"] == job["id"]
        assert service.cancel(job["id"])["state"] == "cancelling"


def test_trashed_job_idempotency_receipt_cannot_launch_replacement(extraction, task_center):
    service, spec, _ = extraction
    preview = service.preview(spec)
    job = service.submit(spec, preview["previewHash"], "once")
    task_center.finish(job["taskId"], "interrupted", returncode=None, reason="lost")
    queued = task_ids(task_center)
    change(service, "extraction", job["id"], "trashed")
    with pytest.raises(StorageError, match="Trash"):
        service.submit(spec, preview["previewHash"], "once")
    assert task_ids(task_center) == queued
    assert task_center.task(job["taskId"])["attempt"] == 1


@pytest.mark.parametrize("job_type", ["packing", "extraction"])
def test_terminal_job_with_worker_still_running_accepts_cancellation(
    job_type, request, task_center
):
    service, spec, _ = request.getfixturevalue(job_type)
    preview = service.preview(spec)
    job = service.submit(spec, preview["previewHash"], "launch")
    task_center.start(job["taskId"])
    task = task_center.task(job["taskId"])
    folder = service.folder / job["id"]
    write_json(
        folder / "result.json",
        {
            "jobId": job["id"],
            "taskId": task["id"],
            "taskAttempt": task["attempt"],
            "state": "failed",
        },
    )
    # Task Center: the task, not a receipt written before the worker exits, says whether
    # the job still runs.
    assert service.get(job["id"])["state"] == "running"
    assert service.cancel(job["id"])["state"] == "cancelling"
    assert (folder / "cancelled").exists()
    assert task_center.task(job["taskId"])["stopRequest"] == "cancel"


def test_terminal_training_with_live_orphan_accepts_cancellation(
    training, task_center, monkeypatch
):
    service, batch, _ = training
    state = service.launch(batch["id"], "launch")
    child = {"pid": 77777, "startTicks": 1234, "bootId": "test"}
    state["status"] = "failed"
    state["runs"][0]["process"] = child
    save_state(Path(state["outputPath"]), state)
    # Every task of the batch ended; one fold's worker still runs.
    lose_batch(task_center, batch["id"], state="failed")
    monkeypatch.setattr(
        "histopilot.application.training.process_alive", lambda value: value == child
    )
    signalled = []
    monkeypatch.setattr(
        "histopilot.application.training.os.killpg", lambda pid, sig: signalled.append(pid)
    )
    assert service.cancel(batch["id"], "cancel")["cancelRequested"]
    assert signalled == [child["pid"]]


def test_a_live_worker_of_a_batch_from_before_the_task_center_blocks_cleanup_but_not_cancel(
    training, task_center, monkeypatch
):
    from histopilot.application.lifecycle import CleanupService

    service, batch, _ = training
    state = service.launch(batch["id"], "launch")
    child = {"pid": 77777, "startTicks": 1234, "bootId": "test"}
    # A batch launched in tmux: no executor, and one fold's worker still runs.
    for key in ("executor", "taskGroup"):
        state.pop(key)
    state["runs"][0]["process"] = child
    save_state(Path(state["outputPath"]), state)
    monkeypatch.setattr(
        "histopilot.application.training.process_alive", lambda value: value == child
    )
    cleanup = CleanupService(service.store, service.filesystem, training=service)
    [item] = [row for row in cleanup.catalog()["items"] if row["id"] == batch["id"]]
    # Its files stay protected, but the Task Center cannot stop a worker it never ran.
    assert item["job"] == {"status": "running", "cancellable": False, "busy": True}
    with pytest.raises(StorageError) as refused:
        service.cancel(batch["id"], "cancel")
    assert refused.value.code == "CREATED_BEFORE_TASK_CENTER"


def test_artifact_can_use_an_older_retained_receipt_with_the_same_identity(
    packing, task_center, monkeypatch
):
    service, spec, _ = packing
    job = packing_support["submit"](service, spec)
    result = complete(service, task_center, job, monkeypatch)
    artifact = result["artifact"]
    attach_spec = spec.model_copy(
        update={"action": "attach", "existingPath": artifact["outputPath"]}
    )
    retained = packing_support["submit"](service, attach_spec, "verify-existing-first")
    retained_result = complete(service, task_center, retained, monkeypatch)
    later = packing_support["submit"](service, attach_spec, "verify-existing-again")
    later_result = complete(service, task_center, later, monkeypatch)
    artifact_id = retained_result["artifact"]["id"]
    assert later_result["artifact"]["id"] == artifact_id
    change(service, "packing", later["id"], "trashed")
    assert service.artifact(artifact_id)["jobId"] == retained["id"]
    with pytest.raises(StorageError, match="Trash"):
        service.artifact(later["id"])


@pytest.mark.parametrize(
    "path", ["histopilot-lifecycle.json", ".histopilot-lifecycle.lock", "training/output"]
)
def test_packing_cannot_claim_lifecycle_or_training_storage(packing, path):
    service, spec, _ = packing
    with pytest.raises(StorageError) as caught:
        service.preview(spec.model_copy(update={"outputPath": str(service.store.folder / path)}))
    assert caught.value.code == "INVALID_OUTPUT"
