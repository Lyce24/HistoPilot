"""Automatic experiment predictors preserve source groups, frozen policy and retries."""

import copy

import pytest
from pydantic import ValidationError
from support.predictors import managed as managed
from support.predictors import registry as registry

from histopilot.application.experiment_predictors import source_items
from histopilot.application.model_experiments import ModelExperimentService
from histopilot.schemas.model_experiments import ExperimentPredictorPolicy
from histopilot.schemas.predictors import LaunchRefit
from histopilot.storage.io import content_hash, read_json_bounded, write_json_atomic
from histopilot.storage.project_lock import StorageError


def coordinators(center):
    """The coordinator tasks in ``center`` (fixtures also queue feature validation)."""
    return center.tasks(kind="predictor-coordinator")


def coordinator(center):
    """The id of the experiment's single coordinator task."""
    [task] = coordinators(center)
    return task["id"]


@pytest.mark.parametrize("method", ["skip", "ensemble"])
def test_non_refit_choices_have_no_epoch_policy(method):
    assert ExperimentPredictorPolicy(method=method).refitPercentile is None
    with pytest.raises(ValidationError):
        ExperimentPredictorPolicy(method=method, refitPercentile=50)


@pytest.mark.parametrize("method", ["refit", "both"])
@pytest.mark.parametrize("percentile", [None, 0, 101, True, "75", float("nan")])
def test_refit_requires_an_explicit_finite_percentile(method, percentile):
    with pytest.raises(ValidationError):
        ExperimentPredictorPolicy(method=method, refitPercentile=percentile)


def test_example_counts_seed_configuration_groups_once_per_method():
    configurations = [{"id": "candidate-" + content_hash(index), "number": index} for index in range(15)]
    splits = [{"id": str(fold), "seed": 42} for fold in range(5)]
    runs = [
        {
            "id": str((config["id"], seed, split["id"])),
            "candidateId": config["id"],
            "trainingSeed": seed,
            "splitPlanId": split["id"],
        }
        for config in configurations
        for seed in (1, 2, 3)
        for split in splits
    ]
    batches = [
        {
            "id": "batch",
            "manifest": {"configurations": configurations, "splitPlans": splits, "runs": runs},
        }
    ]
    items = source_items("experiment", batches, {"method": "both"})
    assert len(runs) == 225 and len(items) == 90
    assert sum(row["method"] == "ensemble" for row in items) == 45
    assert sum(row["method"] == "refit" for row in items) == 45
    assert {row["foldCount"] for row in items} == {5}
    assert len({row["key"] for row in items}) == 90
    assert source_items("experiment", batches, {"method": "skip"}) == []


def test_get_never_dispatches_and_cv_finished_waits_for_predictors(managed, task_center):
    service, identity, jobs, _ = managed
    models = ModelExperimentService(
        service.store, service.filesystem, training=service.training, predictor_execution=service
    )
    detail = models.get(identity)
    assert detail["stage"] == "running"
    assert detail["predictorExecution"]["status"] == "interrupted"
    assert jobs.launches == [] and coordinators(task_center) == []
    service.launch(identity, "start")
    for _ in range(3):
        assert models.get(identity)["predictorExecution"]["counts"]["completed"] == 0
    assert jobs.launches == []
    assert [task["attempt"] for task in coordinators(task_center)] == [1]


def test_both_automatically_publishes_ensembles_and_launches_every_ready_refit(
    managed, task_center
):
    service, identity, jobs, _ = managed
    launched = service.launch(identity, "start")
    assert launched["counts"]["total"] == 4
    replay = service.launch(identity, "start")
    # The replay also reports the queue the accepted coordinator now waits in.
    assert replay["waitingReason"] and replay["runnerAlive"] is False
    assert {**replay, "waitingReason": None, "runnerAlive": None} == launched
    assert [task["attempt"] for task in coordinators(task_center)] == [1]
    first = service.advance(identity)
    # Every ready refit is its own task; the Task Center, not the coordinator, admits them.
    assert first["counts"]["completed"] == 2 and first["counts"]["active"] == 2
    assert len(jobs.launches) == 2
    assert {
        item["epochBudget"]["epochs"] for item in first["items"] if item["method"] == "refit"
    } == {4}
    service.advance(identity)
    assert len(jobs.launches) == 2
    jobs.complete(jobs.launches[0][0])
    second = service.advance(identity)
    assert second["counts"]["completed"] == 3 and len(jobs.launches) == 2
    jobs.complete(jobs.launches[1][0])
    final = service.advance(identity)
    assert final["status"] == "completed" and final["counts"]["completed"] == 4
    assert len({row["predictorId"] for row in final["items"]}) == 4
    assert len(service.builds.predictors.list()["items"]) == 4
    assert service.advance(identity)["counts"]["completed"] == 4
    assert len(jobs.launches) == 2
    models = ModelExperimentService(
        service.store, service.filesystem, training=service.training, predictor_execution=service
    )
    assert models.get(identity)["stage"] == "finished"


def test_partial_folds_wait_without_publishing(managed):
    service, identity, jobs, selections = managed
    folder = service.store.folder / "training" / selections[0].batchId
    state = read_json_bounded(folder / "state.json")
    state["status"] = "running"
    write_json_atomic(folder / "state.json", state)
    service.launch(identity, "start")
    status = service.advance(identity)
    assert status["status"] == "waiting" and status["counts"]["waiting"] == 4
    assert service.store.list_configurations("frozen-predictor") == [] and jobs.launches == []


def test_worker_launch_failure_is_recoverable_without_replaying_cv(
    managed, task_center, monkeypatch
):
    service, identity, jobs, _ = managed
    enqueue = task_center.client.enqueue

    def refuse(*_args, **_kwargs):
        raise OSError("Synthetic launch failed")

    monkeypatch.setattr(task_center.client, "enqueue", refuse)
    state = service.launch(identity, "start")
    assert state["status"] == "interrupted" and state["retryable"]
    assert service.launch(identity, "start")["status"] == "interrupted"
    monkeypatch.setattr(task_center.client, "enqueue", enqueue)
    recovered = service.launch(identity, "retry", resume=True)
    # The refused submission left no task behind; the retry queues the coordinator.
    assert recovered["status"] == "queued"
    assert [task["attempt"] for task in coordinators(task_center)] == [1]
    assert jobs.launches == []


def test_lost_coordinator_acknowledgement_keeps_worker_ownership(
    managed, task_center, monkeypatch
):
    service, identity, _jobs, _ = managed
    enqueue = task_center.client.enqueue

    def enqueue_then_lose_the_reply(*args, **kwargs):
        enqueue(*args, **kwargs)
        raise OSError("Synthetic lost acknowledgement")

    monkeypatch.setattr(task_center.client, "enqueue", enqueue_then_lose_the_reply)
    state = service.launch(identity, "start")
    assert state["status"] == "queued" and not state["retryable"]
    assert service.launch(identity, "start")["status"] == "queued"
    assert [task["attempt"] for task in coordinators(task_center)] == [1]


def test_cancel_stops_active_refits_and_never_launches_them_again(managed, task_center):
    service, identity, jobs, _ = managed
    service.launch(identity, "start")
    service.advance(identity)
    result = service.cancel(identity, "cancel")
    assert result["status"] == "cancelled" and result["counts"]["completed"] == 2
    assert result["counts"]["cancelled"] == 2 and len(jobs.cancels) == 2
    # The coordinator had not started, so the Task Center drops it at once.
    assert task_center.state(coordinator(task_center)) == "cancelled"
    assert service.advance(identity)["status"] == "cancelled"
    assert len(jobs.launches) == 2


def test_a_busy_project_leaves_an_item_waiting_instead_of_failing_it(managed, monkeypatch):
    service, identity, jobs, _ = managed
    service.launch(identity, "start")
    original = service.builds.apply
    busy = []

    def contended(request):
        if not busy:
            busy.append(request)
            raise StorageError(
                "Another operation is writing this project. Retry after it finishes.",
                "PROJECT_BUSY",
            )
        return original(request)

    monkeypatch.setattr(service.builds, "apply", contended)
    first = service.advance(identity)
    assert busy and first["counts"]["failed"] == 0 and first["status"] != "attention"
    assert first["counts"]["waiting"] >= 1
    service.advance(identity)
    for launch in jobs.launches:
        jobs.complete(launch[0])
    final = service.advance(identity)
    assert final["status"] == "completed" and final["counts"]["completed"] == 4


def test_failed_refit_waits_for_explicit_resume(managed, task_center):
    service, identity, jobs, _ = managed
    service.launch(identity, "start")
    task_id = coordinator(task_center)
    service.advance(identity)
    first_refit = jobs.launches[0][0]
    jobs.states[first_refit]["status"] = "failed"
    jobs.states[first_refit]["error"] = "Synthetic training failure"
    state = service.advance(identity)
    assert state["counts"]["failed"] == 1
    assert sum(row[0] == first_refit for row in jobs.launches) == 1
    service.advance(identity)
    assert sum(row[0] == first_refit for row in jobs.launches) == 1
    second_refit = jobs.launches[-1][0]
    jobs.complete(second_refit)
    assert service.advance(identity)["status"] == "attention"
    # The coordinator exits after recording attention; its task ends failed.
    task_center.finish(task_id, "failed", returncode=0)
    service.launch(identity, "retry", resume=True)
    assert task_center.task(task_id)["attempt"] == 2
    service.advance(identity)
    assert sum(row[0] == first_refit for row in jobs.launches) == 2
    assert jobs.launches[-1][2]


def test_published_build_receipt_recovers_a_lost_reply(managed, task_center, monkeypatch):
    service, identity, _jobs, _ = managed
    service.launch(identity, "start")
    original = service.builds.apply
    failed = False

    def lost(request):
        nonlocal failed
        result = original(request)
        if not failed:
            failed = True
            raise OSError("Accepted publication; reply lost")
        return result

    monkeypatch.setattr(service.builds, "apply", lost)
    service.advance(identity)
    _plan, state = service._read(identity)
    failed_item = next(row for row in state["items"] if row["status"] == "failed")
    assert failed_item["buildRequest"]["previewHash"]
    # Simulate coordinator process loss; resume must reuse the persisted request
    # whose manifest action is 'create', not review a new conflicting 'reuse'.
    task_center.finish(coordinator(task_center), "interrupted", returncode=None, reason="lost")
    service.launch(identity, "retry", resume=True)
    assert task_center.task(coordinator(task_center))["attempt"] == 2
    service.advance(identity)
    _plan, state = service._read(identity)
    assert (
        next(row for row in state["items"] if row["key"] == failed_item["key"])["status"]
        != "failed"
    )
    assert len(service.store.list_configurations("frozen-predictor")) == 2


@pytest.mark.parametrize("policy", [None, {"method": "skip", "refitPercentile": None}])
def test_historical_and_skip_submissions_cannot_start_automatic_work(
    managed, task_center, policy
):
    service, identity, _jobs, _ = managed
    record = service.store.get_draft(identity)
    payload = copy.deepcopy(record["payload"])
    if policy:
        payload["submission"]["predictorPolicy"] = policy
    else:
        payload["submission"].pop("predictorPolicy")
    service.store.update_draft(
        identity, expected_revision=record["revision"], name=record["name"], payload=payload
    )
    with pytest.raises(StorageError):
        service.launch(identity, "start")
    assert coordinators(task_center) == [] and service.status(identity) is None
    models = ModelExperimentService(
        service.store, service.filesystem, training=service.training, predictor_execution=service
    )
    assert models.get(identity)["stage"] == "finished"


def test_manual_refit_cannot_change_submitted_percentile_or_resources(managed):
    service, identity, _jobs, selections = managed
    service.launch(identity, "start")
    status = service.advance(identity)
    selection = selections[0].model_copy(update={"method": "refit", "refitPercentile": 50.0})
    preview = service.builds.predictors.preview(selection)
    assert not preview["canFreeze"]
    assert preview["findings"][0]["code"] == "EXPERIMENT_PREDICTOR_POLICY_LOCKED"
    refit_id = next(row["recordId"] for row in status["items"] if row["method"] == "refit")
    with pytest.raises(StorageError) as error:
        service.refits.launch(
            refit_id, LaunchRefit(operationId="override", resources={"ramGbPerRun": 99})
        )
    assert error.value.code == "EXPERIMENT_CONFIGURATION_LOCKED"


@pytest.mark.parametrize("archived", [True, False])
def test_manual_refit_runs_the_submitted_archive_after_the_checkout_changes(
    managed, monkeypatch, archived
):
    from histopilot.workers import training_process
    from histopilot.workers.compute_archive import prepare_compute_archive

    service, identity, jobs, selections = managed
    submission = service.store.get_draft(identity)["payload"]["submission"]
    code = submission["executionContract"]["code"]
    batch = service.store.folder / "training" / selections[0].batchId
    if archived:
        prepare_compute_archive(batch, code)
    service.launch(identity, "start")
    service.advance(identity)
    refit_id = jobs.launches[0][0]
    jobs.states[refit_id]["status"] = "failed"
    monkeypatch.setattr(
        training_process, "compute_snapshot", lambda: {"sha256": "edited", "files": {}}
    )
    request = LaunchRefit(operationId="relaunch-after-edit")
    if not archived:
        # Without an archive there is nothing verified to run, so the old rule stands.
        with pytest.raises(StorageError) as error:
            service.refits.launch(refit_id, request, resume=True)
        assert error.value.code == "EXPERIMENT_RUNTIME_CHANGED"
        return
    service.refits.launch(refit_id, request, resume=True)
    assert jobs.task_options["pinned"] == (code, batch / "compute" / "histopilot")


def test_cleanup_catalog_tracks_active_coordinator(managed, task_center):
    from histopilot.application.lifecycle import CleanupService

    service, identity, jobs, _ = managed
    service.launch(identity, "start")
    task_id = coordinator(task_center)
    cleanup = CleanupService(
        service.store, service.filesystem, training=service.training, compute=jobs
    )

    def job():
        return next(row for row in cleanup.catalog()["items"] if row["id"] == identity)["job"]

    assert job()["busy"]  # queued
    # The coordinator is running; the cancel below would drop a queued one at once.
    task_center.start(task_id)
    experiment = job()
    assert experiment["busy"]
    assert not experiment["cancellable"]
    # A terminal receipt does not permit cleanup while the coordinator's task still
    # runs and could be finalizing its state.
    service.cancel(identity, "cancel-for-cleanup")
    assert service.status(identity)["status"] == "cancelled"
    assert task_center.state(task_id) == "running"
    assert job()["busy"]
    task_center.finish(task_id, "cancelled")
    assert not job()["busy"]


def test_cancel_can_finish_failed_start_without_starting_a_process(managed, task_center):
    service, identity, jobs, _ = managed
    assert service.status(identity) is None
    cancelled = service.cancel(identity, "stop-before-start")
    assert cancelled["status"] == "cancelled"
    assert cancelled["counts"]["total"] == cancelled["counts"]["cancelled"] == 4
    assert service.status(identity)["status"] == "cancelled"
    assert service.launch(identity, "try-resume", resume=True)["status"] == "cancelled"
    assert not jobs.launches and coordinators(task_center) == []


def test_attention_is_cancellable(managed):
    service, identity, jobs, _ = managed
    service.launch(identity, "start")
    service.advance(identity)
    first = jobs.launches[0][0]
    jobs.states[first].update(status="failed", error="Synthetic failure")
    service.advance(identity)
    second = jobs.launches[1][0]
    jobs.states[second].update(status="failed", error="Synthetic failure")
    attention = service.advance(identity)
    assert attention["status"] == "attention" and attention["cancellable"]
    cancelled = service.cancel(identity, "stop")
    assert cancelled["status"] == "cancelled" and cancelled["counts"]["cancelled"] == 2


def test_overall_status_includes_predictor_work(managed):
    service, identity, _jobs, _ = managed
    models = ModelExperimentService(
        service.store, service.filesystem, training=service.training, predictor_execution=service
    )
    # A coordinator that never started needs attention; it is not a failed experiment.
    assert models.get(identity)["status"] == "needs-attention"
    service.launch(identity, "start")
    assert models.get(identity)["status"] == "queued"
    service.advance(identity)
    assert models.get(identity)["status"] == "running"


def test_accepted_noop_resume_cannot_launch_a_later_attempt(managed, task_center):
    service, identity, _jobs, _ = managed
    service.launch(identity, "start")
    task_id = coordinator(task_center)
    assert service.launch(identity, "resume-while-active", resume=True)["status"] == "queued"
    task_center.finish(task_id, "interrupted", returncode=None, reason="lost")
    assert service.status(identity)["status"] == "interrupted"
    assert service.launch(identity, "resume-while-active", resume=True)["status"] == "interrupted"
    assert task_center.task(task_id)["attempt"] == 1
    service.launch(identity, "fresh-resume", resume=True)
    assert task_center.task(task_id)["attempt"] == 2


def test_old_cancel_retry_cannot_cancel_a_new_attempt(managed, task_center, monkeypatch):
    service, identity, _jobs, _ = managed
    service.launch(identity, "start")
    task_id = coordinator(task_center)
    original = service._cancel_items

    def fail_cancel(_plan, _state):
        raise StorageError("Synthetic cancellation transport failure", "CANCEL_FAILED", 409)

    monkeypatch.setattr(service, "_cancel_items", fail_cancel)
    with pytest.raises(StorageError):
        service.cancel(identity, "old-cancel")
    # A failed cancellation worker reports attention and exits. The user then
    # explicitly resumes a new attempt; retrying the old request must be inert.
    path = service.folder(identity) / "state.json"
    state = read_json_bounded(path)
    state["status"] = "attention"
    write_json_atomic(path, state)
    task_center.finish(task_id, "failed", returncode=0)
    monkeypatch.setattr(service, "_cancel_items", original)
    service.launch(identity, "fresh-resume", resume=True)
    assert task_center.task(task_id)["attempt"] == 2
    result = service.cancel(identity, "old-cancel")
    assert result["status"] == "queued"
    assert not (service.folder(identity) / "cancel.requested").exists()
    assert service.cancel(identity, "fresh-cancel")["status"] == "cancelled"


def test_predictor_actions_cannot_reuse_another_action_identity(managed):
    service, identity, _jobs, _ = managed
    service.launch(identity, "start")
    with pytest.raises(StorageError) as error:
        service.cancel(identity, "start")
    assert error.value.code == "OPERATION_CONFLICT"
    service.cancel(identity, "stop")
    with pytest.raises(StorageError) as error:
        service.launch(identity, "stop", resume=True)
    assert error.value.code == "OPERATION_CONFLICT"


def test_manual_refit_cannot_reopen_cancelled_experiment_predictor_work(managed):
    service, identity, jobs, _ = managed
    service.launch(identity, "start")
    service.advance(identity)
    refit_id = jobs.launches[0][0]
    service.cancel(identity, "finish-predictors")
    assert service.status(identity)["status"] == "cancelled"
    with pytest.raises(StorageError) as error:
        service.refits.launch(refit_id, LaunchRefit(operationId="manual-reopen"), resume=True)
    assert error.value.code == "EXPERIMENT_PREDICTORS_LOCKED"
    assert len(jobs.launches) == 2  # both refits launched together; neither reopened


def test_cancelled_intent_blocks_manual_creation_but_not_existing_evidence(managed):
    from histopilot.schemas.predictors import FreezePredictor

    service, identity, jobs, selections = managed
    service.launch(identity, "start")
    request = selections[0].model_copy(update={"method": "ensemble"})
    preview = service.builds.predictors.preview(request)
    assert preview["canFreeze"]
    service.cancel(identity, "close-before-build")
    assert not service.builds.predictors.preview(request)["canFreeze"]
    with pytest.raises(StorageError) as error:
        service.builds.predictors.freeze(
            FreezePredictor(
                **request.model_dump(), previewHash=preview["previewHash"], operationId="late"
            )
        )
    assert error.value.code == "EXPERIMENT_PREDICTORS_LOCKED"
    assert not jobs.launches


def test_cancelled_intent_blocks_late_refit_publication(managed):
    service, identity, jobs, _ = managed
    service.launch(identity, "start")
    service.advance(identity)
    refit_id = jobs.launches[0][0]
    jobs.complete(refit_id)
    retained = service.builds.predictors.list()["items"]
    assert len(retained) == 2
    service.cancel(identity, "stop-before-publication")
    with pytest.raises(StorageError) as error:
        service.refits.publish(refit_id, "late-publication")
    assert error.value.code == "EXPERIMENT_PREDICTORS_LOCKED"
    for predictor in retained:
        service.builds.predictors.verify_checkpoints(predictor)


def test_the_coordinator_loops_through_a_busy_project_then_yields_its_slot(tmp_path):
    from types import SimpleNamespace

    from histopilot.workers import experiment_predictors as worker

    busy = StorageError("Another operation is changing this workspace.", "PROJECT_BUSY")

    class Service:
        def __init__(self, answers):
            self.answers, self.calls = list(answers), 0

        def _read(self, _identity):
            return {}, {"status": "queued"}

        def advance(self, _identity):
            self.calls += 1
            answer = self.answers.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return {"status": answer}

    store, clock = SimpleNamespace(folder=tmp_path), [0.0]

    def run(service):
        return worker._coordinate(
            service,
            store,
            "experiment",
            tmp_path,
            sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
            clock=lambda: clock[0],
        )

    # Busy passes are retried, never recorded as attention.
    service = Service([busy, busy, "running", busy, "completed"])
    assert run(service) == 0 and service.calls == 5
    assert read_json_bounded(tmp_path / "state.json")["status"] == "queued"
    # A project that stays busy: the worker gives its slot back (run() exits 75) and leaves
    # the saved state as it was.
    with pytest.raises(StorageError) as caught:
        run(Service([busy] * 1000))
    assert caught.value.code == "PROJECT_BUSY"
    assert read_json_bounded(tmp_path / "state.json")["status"] == "queued"
    # A real failure still asks for attention.
    assert run(Service([StorageError("Refit stopped.", "EXPERIMENT_REFIT_STOPPED")])) == 0
    state = read_json_bounded(tmp_path / "state.json")
    assert (state["status"], state["error"]["code"]) == ("attention", "EXPERIMENT_REFIT_STOPPED")
