"""Automatic experiment predictors preserve source groups, frozen policy and retries."""

import copy
import runpy
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from histopilot.application.experiment_predictors import ExperimentPredictorService, source_items
from histopilot.application.feature_bundles import _hash
from histopilot.application.model_experiments import ModelExperimentService, execution_contract
from histopilot.application.refits import RefitService
from histopilot.schemas.model_experiments import ExperimentPredictorPolicy
from histopilot.schemas.predictors import LaunchRefit
from histopilot.storage.project_lock import StorageError
from histopilot.workers.packing_process import write_json
from histopilot.workers.training_process import compute_snapshot, read_json

support = runpy.run_path(str(Path(__file__).with_name("test_predictor_builds.py")))
registry = support["registry"]
FakeJobs = support["support"]["FakeJobs"]


class Executor:
    def __init__(self):
        self.live, self.launches, self.fail, self.lost = False, [], False, False

    def running(self, _session):
        return self.live

    def launch(self, *args, **kwargs):
        self.launches.append((args, kwargs))
        if self.fail:
            raise OSError("Synthetic launch failed")
        self.live = True
        if self.lost:
            raise OSError("Synthetic lost acknowledgement")


class Training:
    def __init__(self, folder):
        self.folder = folder

    def execution(self, identity, **_kwargs):
        return read_json(self.folder / "training" / identity / "state.json")


class Jobs(FakeJobs):
    def __init__(self, store, runtime):
        super().__init__(store)
        self.runtime = lambda: runtime
        self.launches, self.cancels = [], []

    def launch(self, identity, plan, operation_id, resume=False):
        self.launches.append((identity, operation_id, resume))
        return super().launch(identity, plan, operation_id, resume)

    def cancel(self, identity, operation_id=None):
        self.cancels.append(identity)
        self.states[identity] = {**self.states[identity], "status": "cancelled"}
        return self.states[identity]


@pytest.fixture
def integrated(registry):
    predictors, _cohort = registry
    selections, folder = support["two_seeds"](predictors)
    store = predictors.store
    identity = selections[0].experimentId
    runtime = {"available": True, "python": sys.executable, "versions": {"torch": "fixture"}}
    plan = read_json(folder / "plan.json")
    plan.update(runtime=runtime, code=compute_snapshot())
    state = read_json(folder / "state.json")
    state["planHash"] = _hash(plan)
    write_json(folder / "plan.json", plan)
    write_json(folder / "state.json", state)
    for run in plan["runs"]:
        path = folder / "runs" / run["id"] / "plan.json"
        run_plan = read_json(path)
        run_plan.update(runtime=runtime, code=plan["code"])
        write_json(path, run_plan)
    record = store.get_draft(identity)
    policy = {"method": "both", "refitPercentile": 75.0}
    submission = {
        "operationId": "submission",
        "expectedRevision": 1,
        "submittedAt": record["createdAt"],
        "status": "submitted",
        "error": None,
        "batchIds": [selections[0].batchId],
        "publications": [],
        "executionContract": execution_contract(plan),
        "predictorPolicy": policy,
        "experiment": {"name": record["name"]},
    }
    store.update_draft(
        identity,
        expected_revision=record["revision"],
        name=record["name"],
        payload={**record["payload"], "submission": submission, "predictorPolicy": policy},
    )
    executor = Executor()
    jobs = Jobs(store, runtime)
    service = ExperimentPredictorService(
        store,
        predictors.filesystem,
        executor=executor,
        training=Training(store.folder),
        refits=RefitService(store, predictors.filesystem, jobs=jobs),
        runtime=lambda: runtime,
    )
    return service, identity, jobs, executor, selections


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
    configurations = [{"id": "candidate-" + _hash(index), "number": index} for index in range(15)]
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


def test_get_never_dispatches_and_cv_finished_waits_for_predictors(integrated):
    service, identity, jobs, executor, _ = integrated
    models = ModelExperimentService(
        service.store, service.filesystem, training=service.training, predictor_execution=service
    )
    detail = models.get(identity)
    assert detail["stage"] == "running"
    assert detail["predictorExecution"]["status"] == "interrupted"
    assert jobs.launches == [] and executor.launches == []
    service.launch(identity, "start")
    for _ in range(3):
        assert models.get(identity)["predictorExecution"]["counts"]["completed"] == 0
    assert jobs.launches == []
    assert len(executor.launches) == 1


def test_both_automatically_publishes_ensembles_and_sequential_refits(integrated):
    service, identity, jobs, executor, _ = integrated
    launched = service.launch(identity, "start")
    assert launched["counts"]["total"] == 4
    assert service.launch(identity, "start") == launched
    assert len(executor.launches) == 1
    first = service.advance(identity)
    assert first["counts"]["completed"] == 2 and first["counts"]["active"] == 1
    assert len(jobs.launches) == 1
    assert {
        item["epochBudget"]["epochs"] for item in first["items"] if item["method"] == "refit"
    } == {4}
    service.advance(identity)
    assert len(jobs.launches) == 1
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


def test_partial_folds_wait_without_publishing(integrated):
    service, identity, jobs, _executor, selections = integrated
    folder = service.store.folder / "training" / selections[0].batchId
    state = read_json(folder / "state.json")
    state["status"] = "running"
    write_json(folder / "state.json", state)
    service.launch(identity, "start")
    status = service.advance(identity)
    assert status["status"] == "waiting" and status["counts"]["waiting"] == 4
    assert service.store.list_configurations("frozen-predictor") == [] and jobs.launches == []


def test_worker_launch_failure_is_recoverable_without_replaying_cv(integrated):
    service, identity, jobs, executor, _ = integrated
    executor.fail = True
    state = service.launch(identity, "start")
    assert state["status"] == "interrupted" and state["retryable"]
    assert service.launch(identity, "start")["status"] == "interrupted"
    executor.fail = False
    recovered = service.launch(identity, "retry", resume=True)
    assert recovered["status"] == "queued" and len(executor.launches) == 2
    assert jobs.launches == []


def test_lost_coordinator_acknowledgement_keeps_worker_ownership(integrated):
    service, identity, _jobs, executor, _ = integrated
    executor.lost = True
    state = service.launch(identity, "start")
    assert state["status"] == "queued" and not state["retryable"]
    assert service.launch(identity, "start")["status"] == "queued"
    assert len(executor.launches) == 1


def test_cancel_stops_active_refit_and_never_launches_remaining_items(integrated):
    service, identity, jobs, _executor, _ = integrated
    service.launch(identity, "start")
    service.advance(identity)
    result = service.cancel(identity, "cancel")
    assert result["status"] == "cancelled" and result["counts"]["completed"] == 2
    assert result["counts"]["cancelled"] == 2 and len(jobs.cancels) == 1
    assert service.advance(identity)["status"] == "cancelled"
    assert len(jobs.launches) == 1


def test_failed_refit_waits_for_explicit_resume(integrated):
    service, identity, jobs, executor, _ = integrated
    service.launch(identity, "start")
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
    executor.live = False
    service.launch(identity, "retry", resume=True)
    service.advance(identity)
    assert sum(row[0] == first_refit for row in jobs.launches) == 2
    assert jobs.launches[-1][2]


def test_published_build_receipt_recovers_a_lost_reply(integrated, monkeypatch):
    service, identity, _jobs, executor, _ = integrated
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
    executor.live = False
    service.launch(identity, "retry", resume=True)
    service.advance(identity)
    _plan, state = service._read(identity)
    assert (
        next(row for row in state["items"] if row["key"] == failed_item["key"])["status"]
        != "failed"
    )
    assert len(service.store.list_configurations("frozen-predictor")) == 2


@pytest.mark.parametrize("policy", [None, {"method": "skip", "refitPercentile": None}])
def test_historical_and_skip_submissions_cannot_start_automatic_work(integrated, policy):
    service, identity, _jobs, executor, _ = integrated
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
    assert executor.launches == [] and service.status(identity) is None
    models = ModelExperimentService(
        service.store, service.filesystem, training=service.training, predictor_execution=service
    )
    assert models.get(identity)["stage"] == "finished"


def test_manual_refit_cannot_change_submitted_percentile_or_resources(integrated):
    service, identity, _jobs, _executor, selections = integrated
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


def test_cleanup_catalog_tracks_active_coordinator(integrated, monkeypatch):
    from histopilot.application.experiment_predictors import TmuxExperimentExecutor
    from histopilot.application.lifecycle import CleanupService

    service, identity, jobs, executor, _ = integrated
    service.launch(identity, "start")
    monkeypatch.setattr(TmuxExperimentExecutor, "running", lambda _self, _session: executor.live)
    cleanup = CleanupService(
        service.store, service.filesystem, training=service.training, compute=jobs
    )
    experiment = next(row for row in cleanup.catalog()["items"] if row["id"] == identity)
    assert experiment["job"]["busy"]
    assert not experiment["job"]["cancellable"]
    # A terminal receipt does not permit cleanup while the owning tmux worker
    # still exists and could be finalizing its state.
    service.cancel(identity, "cancel-for-cleanup")
    assert service.status(identity)["status"] == "cancelled"
    retained = next(row for row in cleanup.catalog()["items"] if row["id"] == identity)
    assert retained["job"]["busy"]
    executor.live = False
    stopped = next(row for row in cleanup.catalog()["items"] if row["id"] == identity)
    assert not stopped["job"]["busy"]


def test_cancel_can_finish_failed_start_without_starting_a_process(integrated):
    service, identity, jobs, executor, _ = integrated
    assert service.status(identity) is None
    cancelled = service.cancel(identity, "stop-before-start")
    assert cancelled["status"] == "cancelled"
    assert cancelled["counts"]["total"] == cancelled["counts"]["cancelled"] == 4
    assert service.status(identity)["status"] == "cancelled"
    assert service.launch(identity, "try-resume", resume=True)["status"] == "cancelled"
    assert not jobs.launches and not executor.launches


def test_attention_is_cancellable(integrated):
    service, identity, jobs, _executor, _ = integrated
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


def test_overall_status_includes_predictor_work(integrated):
    service, identity, _jobs, _executor, _ = integrated
    models = ModelExperimentService(
        service.store, service.filesystem, training=service.training, predictor_execution=service
    )
    assert models.get(identity)["status"] == "interrupted"
    service.launch(identity, "start")
    assert models.get(identity)["status"] == "queued"
    service.advance(identity)
    assert models.get(identity)["status"] == "running"


def test_accepted_noop_resume_cannot_launch_a_later_attempt(integrated):
    service, identity, _jobs, executor, _ = integrated
    service.launch(identity, "start")
    assert service.launch(identity, "resume-while-active", resume=True)["status"] == "queued"
    executor.live = False
    assert service.status(identity)["status"] == "interrupted"
    assert service.launch(identity, "resume-while-active", resume=True)["status"] == "interrupted"
    assert len(executor.launches) == 1
    service.launch(identity, "fresh-resume", resume=True)
    assert len(executor.launches) == 2


def test_old_cancel_retry_cannot_cancel_a_new_attempt(integrated, monkeypatch):
    service, identity, _jobs, executor, _ = integrated
    service.launch(identity, "start")
    original = service._cancel_items

    def fail_cancel(_plan, _state):
        raise StorageError("Synthetic cancellation transport failure", "CANCEL_FAILED", 409)

    monkeypatch.setattr(service, "_cancel_items", fail_cancel)
    with pytest.raises(StorageError):
        service.cancel(identity, "old-cancel")
    # A failed cancellation worker reports attention and exits. The user then
    # explicitly resumes a new attempt; retrying the old request must be inert.
    path = service.folder(identity) / "state.json"
    state = read_json(path)
    state["status"] = "attention"
    write_json(path, state)
    executor.live = False
    monkeypatch.setattr(service, "_cancel_items", original)
    service.launch(identity, "fresh-resume", resume=True)
    result = service.cancel(identity, "old-cancel")
    assert result["status"] == "queued"
    assert not (service.folder(identity) / "cancel.requested").exists()
    assert service.cancel(identity, "fresh-cancel")["status"] == "cancelled"


def test_predictor_actions_cannot_reuse_another_action_identity(integrated):
    service, identity, _jobs, _executor, _ = integrated
    service.launch(identity, "start")
    with pytest.raises(StorageError) as error:
        service.cancel(identity, "start")
    assert error.value.code == "OPERATION_CONFLICT"
    service.cancel(identity, "stop")
    with pytest.raises(StorageError) as error:
        service.launch(identity, "stop", resume=True)
    assert error.value.code == "OPERATION_CONFLICT"


def test_manual_refit_cannot_reopen_cancelled_experiment_predictor_work(integrated):
    service, identity, jobs, _executor, _ = integrated
    service.launch(identity, "start")
    service.advance(identity)
    refit_id = jobs.launches[0][0]
    service.cancel(identity, "finish-predictors")
    assert service.status(identity)["status"] == "cancelled"
    with pytest.raises(StorageError) as error:
        service.refits.launch(refit_id, LaunchRefit(operationId="manual-reopen"), resume=True)
    assert error.value.code == "EXPERIMENT_PREDICTORS_LOCKED"
    assert len(jobs.launches) == 1


def test_cancelled_intent_blocks_manual_creation_but_not_existing_evidence(integrated):
    from histopilot.schemas.predictors import FreezePredictor

    service, identity, jobs, _executor, selections = integrated
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


def test_cancelled_intent_blocks_late_refit_publication(integrated):
    service, identity, jobs, _executor, _ = integrated
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
