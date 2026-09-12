"""Evaluation groups retain their jobs and dependencies through workspace cleanup."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from histopilot.application.lifecycle import CleanupService
from histopilot.schemas.bulk_evaluations import BulkEvaluationSelection
from histopilot.schemas.lifecycle import CancelCleanupJob, CleanupSelection
from histopilot.storage.project_lock import StorageError
from histopilot.workers.packing_process import write_json
from histopilot.workers.training_process import read_json

bulk_support = runpy.run_path(str(Path(__file__).with_name("test_bulk_evaluations.py")))
cleanup_support = runpy.run_path(str(Path(__file__).with_name("test_workspace_cleanup.py")))
job_support = runpy.run_path(str(Path(__file__).with_name("test_job_lifecycle.py")))
packing = job_support["packing"]
extraction = job_support["extraction"]


@pytest.fixture
def batch_cleanup(tmp_path, monkeypatch):
    bulk, predictor, cohort, executor = bulk_support["bulk"].__wrapped__(tmp_path, monkeypatch)
    cleanup = CleanupService(
        bulk.store,
        bulk.filesystem,
        training=SimpleNamespace(execution=lambda *args, **kwargs: None),
        compute=bulk.evaluations.jobs,
        evaluation_batches=bulk,
    )
    choice = BulkEvaluationSelection(cohortId=cohort["id"])
    request = bulk_support["run_request"](choice, bulk.preview(choice))
    return bulk, cleanup, predictor, cohort, executor, request


def key(identity):
    return "configuration:" + identity


def review(service, action, *identities):
    return service.preview(
        CleanupSelection(action=action, keys=[key(identity) for identity in identities])
    )


def finish(bulk, evaluation_id, status="cancelled"):
    folder = bulk.evaluations.jobs.folder(evaluation_id)
    state = read_json(folder / "state.json")
    result = {"state": "succeeded", "runId": evaluation_id} if status == "completed" else None
    if result:
        write_json(folder / "result.json", result)
    state.update(status=status, process=None, result=result)
    write_json(folder / "state.json", state)


def test_active_batch_blocks_its_own_and_source_cleanup_until_children_stop(batch_cleanup):
    bulk, cleanup, predictor, _, executor, request = batch_cleanup
    batch = bulk.run(request)
    child = batch["items"][0]["evaluationId"]
    for identity in (batch["id"], child, predictor["id"]):
        result = review(cleanup, "trash", identity)
        assert not result["canApply"]
        assert any(row["code"] == "JOBS_ACTIVE" for row in result["blockers"])
    cancel = CancelCleanupJob(key=key(batch["id"]), operationId="cancel-batch")
    cleanup.cancel(cancel)
    pending = review(cleanup, "archive", batch["id"])
    assert not pending["canApply"]
    parent = next(row for row in cleanup.catalog()["items"] if row["key"] == key(batch["id"]))
    assert parent["job"] == {"status": "cancelling", "cancellable": False, "busy": True}
    assert bulk.evaluations.jobs.status(child)["cancellationRequested"]
    finish(bulk, child)
    executor.sessions.clear()
    assert review(cleanup, "archive", batch["id"])["canApply"]
    before = (bulk.evaluations.jobs.folder(child) / "cancel.requested").read_bytes()
    cleanup.cancel(cancel)
    assert (bulk.evaluations.jobs.folder(child) / "cancel.requested").read_bytes() == before


def test_cleanup_cancel_stops_unsubmitted_members_and_prevents_later_launch(
    batch_cleanup, monkeypatch
):
    bulk, cleanup, _, _, executor, request = batch_cleanup
    submit = bulk._submit
    monkeypatch.setattr(bulk, "_submit", lambda batch: None)
    batch = bulk.run(request)
    assert batch["status"] == "queued" and batch["items"][0]["status"] == "planned"
    child = batch["items"][0]["evaluationId"]
    assert not review(cleanup, "trash", batch["id"])["canApply"]
    cleanup.cancel(CancelCleanupJob(key=key(batch["id"]), operationId="cancel-before-submit"))
    assert bulk.get(batch["id"])["status"] == "cancelled"
    monkeypatch.setattr(bulk, "_submit", submit)
    assert bulk.run(request)["status"] == "cancelled"
    assert executor.calls == []
    with pytest.raises(StorageError) as absent:
        bulk.store.get_configuration(child)
    assert absent.value.code == "CONFIGURATION_NOT_FOUND"
    assert review(cleanup, "trash", batch["id"])["canApply"]


def test_retained_batch_protects_child_results_and_restore_requires_them(batch_cleanup):
    bulk, cleanup, _, _, executor, request = batch_cleanup
    batch = bulk.run(request)
    child = batch["items"][0]["evaluationId"]
    finish(bulk, child, "completed")
    executor.sessions.clear()
    artifact = bulk.evaluations.jobs.folder(child) / "retained-results.txt"
    artifact.write_text("Keep completed evidence")
    cleanup_support["apply"](cleanup, "archive", key(batch["id"]))
    blocked = review(cleanup, "trash", child)
    assert not blocked["canApply"]
    assert key(batch["id"]) in blocked["requiredKeys"]
    cleanup_support["apply"](cleanup, "trash", key(batch["id"]), key(child))
    assert artifact.read_text() == "Keep completed evidence"
    restore = review(cleanup, "restore", batch["id"])
    assert not restore["canApply"] and key(child) in restore["requiredKeys"]
    cleanup_support["apply"](cleanup, "restore", key(batch["id"]), key(child))
    assert bulk.get(batch["id"])["status"] == "completed"
    assert bulk.store.get_configuration(child)["id"] == child


@pytest.mark.parametrize("job_type", ["packing", "extraction"])
@pytest.mark.parametrize(
    "reserved",
    [
        "predictor-builds",
        "evaluation-batches",
        "predictor-builds/operation/receipt.json",
        "evaluation-batches/batch/state.json",
    ],
)
def test_feature_jobs_cannot_overwrite_bulk_receipts(job_type, reserved, request):
    service, spec, _executor, _ = request.getfixturevalue(job_type)
    destination = service.store.folder / reserved
    destination.parent.mkdir(parents=True, exist_ok=True)
    before = destination.parent / "sentinel.txt"
    before.write_text("durable operation evidence")
    with pytest.raises(StorageError) as rejected:
        service.preview(spec.model_copy(update={"outputPath": str(destination)}))
    assert rejected.value.code == "INVALID_OUTPUT"
    assert before.read_text() == "durable operation evidence"
