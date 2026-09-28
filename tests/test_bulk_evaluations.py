"""Bulk evaluation snapshots preserve identities through partial submissions and retries."""

import copy
import runpy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest
from pydantic import ValidationError
from support.compute import compute_tasks
from support.predictors import candidate, freeze
from support.projects import lifecycle

from histopilot.application.bulk_evaluations import BulkEvaluationService
from histopilot.schemas.bulk_evaluations import BulkEvaluationSelection, RunBulkEvaluation
from histopilot.storage.io import read_json_bounded, write_json_atomic
from histopilot.storage.project_lock import StorageError

support = runpy.run_path(str(Path(__file__).with_name("test_evaluation_execution_service.py")))


@pytest.fixture
def bulk(tmp_path, monkeypatch, task_center):
    """Bulk evaluation whose members queue in this test's Task Center, as in production."""
    evaluations, _record, predictor, cohort = support["evaluation"].__wrapped__(
        tmp_path, monkeypatch, task_center
    )
    service = BulkEvaluationService(
        evaluations.store, evaluations.filesystem, evaluations=evaluations
    )
    return service, predictor, cohort


def run_request(selection, preview, operation="bulk-run"):
    return RunBulkEvaluation(
        **selection.model_dump(),
        reviewedPredictorIds=preview["reviewedPredictorIds"],
        previewHash=preview["previewHash"],
        operationId=operation,
    )


def another(service, name):
    selected, *_ = candidate(service.evaluations.predictors, name)
    return freeze(service.evaluations.predictors, selected)[0]


def test_all_scope_freezes_reviewed_ids_and_does_not_include_later_predictors(
    bulk, task_center
):
    service, first, cohort = bulk
    choice = BulkEvaluationSelection(cohortId=cohort["id"])
    preview = service.preview(choice)
    assert preview["eligibleCount"] == 1 and preview["reviewedPredictorIds"] == [first["id"]]
    later = another(service, "Later")
    result = service.run(run_request(choice, preview))
    assert result["status"] == "queued"
    assert [row["predictorId"] for row in result["items"]] == [first["id"]]
    assert (
        later["id"]
        not in service._record(result["id"])["manifest"]["request"]["reviewedPredictorIds"]
    )
    assert len(compute_tasks(task_center)) == 1


def test_mixed_compatibility_review_skips_only_explicitly_blocked_predictors(
    bulk, task_center
):
    service, valid, cohort = bulk
    invalid = another(service, "Changed weights")
    Path(invalid["manifest"]["checkpoints"][0]["path"]).write_bytes(b"changed")
    choice = BulkEvaluationSelection(cohortId=cohort["id"])
    preview = service.preview(choice)
    assert preview["eligibleCount"] == preview["blockedCount"] == 1
    assert next(row for row in preview["items"] if row["predictorId"] == invalid["id"])["findings"]
    result = service.run(run_request(choice, preview))
    assert {row["predictorId"]: row["status"] for row in result["items"]} == {
        valid["id"]: "queued",
        invalid["id"]: "skipped",
    }
    assert len(compute_tasks(task_center)) == 1


def test_changed_checkpoint_after_review_rejects_entire_unstarted_batch(bulk, task_center):
    service, predictor, cohort = bulk
    choice = BulkEvaluationSelection(cohortId=cohort["id"])
    preview = service.preview(choice)
    Path(predictor["manifest"]["checkpoints"][0]["path"]).write_bytes(b"changed after review")
    with pytest.raises(StorageError, match="changed"):
        service.run(run_request(choice, preview))
    assert service.list()["items"] == []
    assert not compute_tasks(task_center)


def test_partial_submission_retry_reuses_success_and_continues_missing_member(
    bulk, task_center, monkeypatch
):
    service, _, cohort = bulk
    another(service, "Second")
    choice = BulkEvaluationSelection(cohortId=cohort["id"])
    preview = service.preview(choice)
    request = run_request(choice, preview)
    original = service.evaluations.launch
    attempts = []

    def once_missing(identity, operation, **kwargs):
        attempts.append(identity)
        if len(attempts) == 2:
            raise ConnectionError("Interrupted before the second launch")
        return original(identity, operation, **kwargs)

    monkeypatch.setattr(service.evaluations, "launch", once_missing)
    first = service.run(request)
    assert first["counts"]["failed"] == 1 and len(compute_tasks(task_center)) == 1
    second = service.run(request)
    assert second["id"] == first["id"] and second["counts"]["queued"] == 2
    assert [task["attempt"] for task in compute_tasks(task_center)] == [1, 1]
    assert len({row["evaluationId"] for row in second["items"]}) == 2
    assert len(service.store.list_configurations("evaluation-batch")) == 1


def test_new_requests_create_distinct_records_but_exact_retries_do_not(bulk, task_center):
    service, _, cohort = bulk
    choice = BulkEvaluationSelection(cohortId=cohort["id"])
    preview = service.preview(choice)
    first = service.run(run_request(choice, preview, "first"))
    assert service.run(run_request(choice, preview, "first"))["id"] == first["id"]
    second = service.run(run_request(choice, preview, "second"))
    assert first["id"] != second["id"]
    assert first["items"][0]["evaluationId"] != second["items"][0]["evaluationId"]
    assert [task["attempt"] for task in compute_tasks(task_center)] == [1, 1]
    changed = run_request(choice, preview, "first").model_copy(update={"namePrefix": "Another"})
    with pytest.raises(StorageError, match="another evaluation batch"):
        service.run(changed)


def test_batch_cancel_stops_pending_submission_and_preserves_running_jobs(
    bulk, task_center, monkeypatch
):
    service, _, cohort = bulk
    another(service, "Second")
    choice = BulkEvaluationSelection(cohortId=cohort["id"])
    preview = service.preview(choice)
    request = run_request(choice, preview)
    original = service._submit
    monkeypatch.setattr(service, "_submit", lambda _batch: None)
    pending = service.run(request)
    assert pending["status"] == "queued" and pending["counts"]["planned"] == 2
    cancelled = service.cancel(pending["id"], "cancel")
    assert cancelled["status"] == "cancelled"
    monkeypatch.setattr(service, "_submit", original)
    assert service.run(request)["status"] == "cancelled"
    assert not compute_tasks(task_center)


def test_batch_manifest_retains_dependency_refs_and_selected_scope_is_explicit(bulk):
    service, predictor, cohort = bulk
    choice = BulkEvaluationSelection(
        cohortId=cohort["id"], scope="selected", predictorIds=[predictor["id"]]
    )
    preview = service.preview(choice)
    result = service.run(run_request(choice, preview))
    manifest = service._record(result["id"])["manifest"]
    assert manifest["cohort"] == {key: cohort[key] for key in ("id", "contentHash")}
    assert manifest["items"][0]["predictor"]["id"] == predictor["id"]
    assert manifest["items"][0]["evaluationId"] == result["items"][0]["evaluationId"]
    assert result["items"][0]["trainingSeed"] == predictor["manifest"]["trainingSeed"]
    assert service.list()["items"][0]["id"] == result["id"]
    with pytest.raises(ValidationError):
        BulkEvaluationSelection(cohortId=cohort["id"], scope="selected")
    with pytest.raises(ValidationError):
        BulkEvaluationSelection(cohortId=cohort["id"], predictorIds=[predictor["id"]])
    changed = copy.deepcopy(run_request(choice, preview))
    changed.reviewedPredictorIds = []
    with pytest.raises(StorageError, match="scope changed"):
        service.run(changed.model_copy(update={"operationId": "changed"}))


def test_deleted_explicit_selection_requires_removal_before_submission(bulk, task_center):
    service, valid, cohort = bulk
    deleted = another(service, "Deleted")
    lifecycle(service.store, deleted, "trashed")
    choice = BulkEvaluationSelection(
        cohortId=cohort["id"], scope="selected", predictorIds=[valid["id"], deleted["id"]]
    )
    preview = service.preview(choice)
    assert preview["eligibleCount"] == 1 and not preview["canRun"]
    assert (
        next(row for row in preview["items"] if row["predictorId"] == deleted["id"])["findings"][0][
            "code"
        ]
        == "PREDICTOR_INACTIVE"
    )
    with pytest.raises(StorageError, match="remove deleted"):
        service.run(run_request(choice, preview))
    assert not compute_tasks(task_center)


def test_group_cancellation_marks_existing_jobs_pending_until_workers_stop(
    bulk, task_center
):
    service, _, cohort = bulk
    choice = BulkEvaluationSelection(cohortId=cohort["id"])
    result = service.run(run_request(choice, service.preview(choice)))
    [task] = compute_tasks(task_center)
    # The member's worker has started: the Task Center cancels a queued member at once.
    task_center.start(task["id"])
    cancelled = service.cancel(result["id"], "cancel-bulk")
    assert cancelled["cancelRequested"] and cancelled["status"] == "queued"
    assert cancelled["items"][0]["execution"]["cancellationRequested"]
    assert len(compute_tasks(task_center)) == 1
    assert task_center.state(task["id"]) == "stopping"
    # The worker records the cancellation and exits; only then is the group cancelled.
    member = cancelled["items"][0]["evaluationId"]
    folder = service.evaluations.jobs.folder(member)
    write_json_atomic(folder / "state.json", {**read_json_bounded(folder / "state.json"), "status": "cancelled"})
    task_center.finish(task["id"], "cancelled")
    assert service.get(result["id"])["status"] == "cancelled"


def test_batch_routes_precede_the_individual_evaluation_getter():
    from histopilot.api.predictors import evaluation_run_router

    routes = evaluation_run_router(None, None).routes
    paths = [route.path for route in routes]
    base = "/api/v1/projects/{identity}/evaluation-runs"
    assert paths.index(base + "/bulk") < paths.index(base + "/{evaluation_id}")
    assert paths.index(base + "/bulk/{batch_id}") < paths.index(base + "/{evaluation_id}")


def test_concurrent_cancel_between_members_prevents_the_next_launch(
    bulk, task_center, monkeypatch
):
    service, _, cohort = bulk
    another(service, "Second")
    choice = BulkEvaluationSelection(cohortId=cohort["id"])
    request = run_request(choice, service.preview(choice))
    submitted, continue_submission = Event(), Event()
    original = service._submit_member
    batch_ids = []

    def pause_after_first(batch, member):
        result = original(batch, member)
        if not batch_ids:
            batch_ids.append(batch["id"])
            submitted.set()
            assert continue_submission.wait(5)
        return result

    monkeypatch.setattr(service, "_submit_member", pause_after_first)
    with ThreadPoolExecutor(max_workers=2) as pool:
        running = pool.submit(service.run, request)
        try:
            assert submitted.wait(5)
            # The first member's worker has started, so the cancel can only request its stop.
            task_center.start(compute_tasks(task_center)[0]["id"])
            cancelling = pool.submit(service.cancel, batch_ids[0], "concurrent-cancel")
            cancelled = cancelling.result(timeout=3)
            assert cancelled["cancelRequested"]
        finally:
            continue_submission.set()
        finished = running.result(timeout=5)
    assert len(compute_tasks(task_center)) == 1
    assert finished["counts"]["cancelled"] == 1
    assert finished["items"][0]["execution"]["cancellationRequested"]
