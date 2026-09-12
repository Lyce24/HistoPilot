"""Workspace cleanup cannot race job starts or erase historical validation receipts."""

import hashlib
import runpy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from histopilot.storage.lifecycle import LifecycleStore, lifecycle_guard
from histopilot.storage.project_lock import StorageError
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


@pytest.fixture
def packing(tmp_path):
    return packing_support["packing"].__wrapped__(tmp_path)


@pytest.fixture
def extraction(tmp_path, monkeypatch):
    return extraction_support["extraction"].__wrapped__(tmp_path, monkeypatch)


@pytest.fixture
def training(tmp_path, monkeypatch):
    return training_support["execution"].__wrapped__(tmp_path, monkeypatch)


@pytest.mark.parametrize("kind", ["dataset", "project"])
def test_extraction_rejects_trashed_inputs_before_creating_job(extraction, kind):
    service, spec, executor, _ = extraction
    preview = service.preview(spec)
    identity = spec.datasetId if kind == "dataset" else service.store.project_id
    change(service, kind, identity, "trashed")
    with pytest.raises(StorageError, match="Trash"):
        service.submit(spec, preview["previewHash"], "blocked")
    assert not executor.launches
    assert not service.folder.exists()
    assert not Path(spec.outputPath).exists()


@pytest.mark.parametrize("kind", ["configuration", "project"])
def test_packing_rejects_trashed_inputs_before_creating_job(packing, kind):
    service, spec, executor, _ = packing
    preview = service.preview(spec)
    identity = spec.featureSetId if kind == "configuration" else service.store.project_id
    change(service, kind, identity, "trashed")
    with pytest.raises(StorageError, match="Trash"):
        service.submit(spec, preview["previewHash"], "blocked")
    assert not executor.launches
    assert not service.folder.exists()


@pytest.mark.parametrize("reference", ["batch", "protocol", "bundle", "dataset", "project"])
def test_training_rejects_trashed_records_before_creating_execution(training, reference):
    service, batch, executor, _ = training
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
    assert not executor.launches
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


def test_archived_pack_receipt_still_resolves_but_trash_blocks_it(packing):
    service, spec, executor, _ = packing
    job = packing_support["submit"](service, spec)
    result = packing_support["complete"](service, executor, job)
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


def test_extraction_archive_hides_terminal_job_and_keeps_direct_history(extraction):
    service, spec, executor, _ = extraction
    job = extraction_support["submit"](service, spec)
    executor.sessions.clear()
    change(service, "extraction", job["id"], "archived")
    assert service.list()["jobs"] == []
    assert service.get(job["id"])["state"] == "interrupted"
    assert service.list(include_inactive=True)["jobs"][0]["id"] == job["id"]
    change(service, "extraction", job["id"], "trashed")
    with pytest.raises(StorageError, match="Trash"):
        service.get(job["id"])
    assert service.get(job["id"], include_inactive=True)["state"] == "interrupted"


@pytest.mark.parametrize("job_type", ["packing", "extraction", "training"])
def test_active_jobs_remain_visible_and_cancellable_if_lifecycle_is_inconsistent(job_type, request):
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
        change(service, job_type, job["id"], "trashed")
        assert service.list()["jobs"][0]["id"] == job["id"]
        assert service.cancel(job["id"])["state"] == "cancelling"


def test_trashed_job_idempotency_receipt_cannot_launch_replacement(extraction):
    service, spec, executor, _ = extraction
    preview = service.preview(spec)
    job = service.submit(spec, preview["previewHash"], "once")
    executor.sessions.clear()
    change(service, "extraction", job["id"], "trashed")
    with pytest.raises(StorageError, match="Trash"):
        service.submit(spec, preview["previewHash"], "once")
    assert len(executor.launches) == 1


@pytest.mark.parametrize("job_type", ["packing", "extraction"])
def test_terminal_job_with_worker_still_running_accepts_cancellation(job_type, request):
    service, spec, _executor, _ = request.getfixturevalue(job_type)
    preview = service.preview(spec)
    job = service.submit(spec, preview["previewHash"], "launch")
    folder = service.folder / job["id"]
    write_json(folder / "result.json", {"jobId": job["id"], "state": "failed"})
    assert service.get(job["id"])["state"] == "failed"
    service.cancel(job["id"])
    assert (folder / "cancelled").exists()


def test_terminal_training_with_live_orphan_accepts_cancellation(training, monkeypatch):
    service, batch, executor, _ = training
    state = service.launch(batch["id"], "launch")
    child = {"pid": 77777, "startTicks": 1234, "bootId": "test"}
    state["status"] = "failed"
    state["runs"][0]["process"] = child
    save_state(Path(state["outputPath"]), state)
    executor.sessions.clear()
    monkeypatch.setattr(
        "histopilot.application.training.process_alive", lambda value: value == child
    )
    signalled = []
    monkeypatch.setattr(
        "histopilot.application.training.os.killpg", lambda pid, sig: signalled.append(pid)
    )
    assert service.cancel(batch["id"], "cancel")["cancelRequested"]
    assert signalled == [child["pid"]]


def test_artifact_can_use_an_older_retained_receipt_with_the_same_identity(packing):
    service, spec, executor, _ = packing
    job = packing_support["submit"](service, spec)
    result = packing_support["complete"](service, executor, job)
    artifact = result["artifact"]
    attach_spec = spec.model_copy(
        update={"action": "attach", "existingPath": artifact["outputPath"]}
    )
    retained = packing_support["submit"](service, attach_spec, "verify-existing-first")
    retained_result = packing_support["complete"](service, executor, retained)
    later = packing_support["submit"](service, attach_spec, "verify-existing-again")
    later_result = packing_support["complete"](service, executor, later)
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
    service, spec, _, _ = packing
    with pytest.raises(StorageError) as caught:
        service.preview(spec.model_copy(update={"outputPath": str(service.store.folder / path)}))
    assert caught.value.code == "INVALID_OUTPUT"
