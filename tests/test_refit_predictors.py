"""Refit provenance, epoch selection, and independent ensemble/refit identities."""

import copy
import runpy
import sys
from pathlib import Path

import pytest

from histopilot.application.compute_jobs import ComputeJobService
from histopilot.application.feature_bundles import _hash
from histopilot.application.refits import RefitService, epoch_budget
from histopilot.schemas.development import ResourcePolicy
from histopilot.schemas.predictors import FreezePredictor, LaunchRefit, PredictorSelection
from histopilot.storage.project_lock import StorageError
from histopilot.workers.packing_process import write_json

support = runpy.run_path(str(Path(__file__).with_name("test_predictor_registry.py")))
registry = support["registry"]
candidate = support["candidate"]


class FakeJobs:
    def __init__(self, store):
        self.store, self.states = store, {}

    def folder(self, identity):
        return self.store.folder / "compute-jobs" / identity

    def status(self, identity, **kwargs):
        return self.states.get(identity, {"status": "not_started"})

    def replay_launch(self, identity, operation_id, **kwargs):
        return None

    def launch(self, identity, plan, operation_id, resume=False):
        folder = self.folder(identity)
        folder.mkdir(parents=True, exist_ok=True)
        write_json(folder / "plan.json", plan)
        self.states[identity] = {"status": "running", "planHash": _hash(plan), "result": None}
        return self.states[identity]

    def complete(self, identity):
        folder = self.folder(identity)
        import json

        plan = json.loads((folder / "plan.json").read_text())
        checkpoint = folder / "final.ckpt"
        checkpoint.write_bytes(b"Synthetic final model weights")
        result = {
            "runId": identity,
            "state": "succeeded",
            "bestCheckpointPath": str(checkpoint),
            "epochsCompleted": plan["epochBudget"]["epochs"],
        }
        write_json(folder / "result.json", result)
        self.states[identity].update(status="completed", result=result)


def refit_candidate(service, *, epochs=(3, 10), percentile=50, legacy_history=False):
    # Historical loss-only histories must declare their historical monitor;
    # new recipes intentionally default to validation AUROC.
    selection, folder, state = candidate(
        service, refit_ready=True, checkpoint_metric="validation_loss" if legacy_history else None
    )
    for run, epoch in zip(state["runs"], epochs, strict=True):
        result = run["result"]
        result["epochsCompleted"] = 12
        if not legacy_history:
            result["bestEpoch"] = epoch
        else:
            write_json(
                folder / "runs" / run["id"] / "history.json",
                [
                    {"epoch": i, "validation": {"loss": 0.7 if i + 1 == epoch else 1.0}}
                    for i in range(12)
                ],
            )
        write_json(folder / "runs" / run["id"] / "result.json", result)
    write_json(folder / "state.json", state)
    return (
        PredictorSelection(
            **{**selection.model_dump(), "method": "refit", "refitPercentile": percentile}
        ),
        folder,
        state,
    )


def create(service, selection, jobs):
    refits = RefitService(service.store, service.filesystem, jobs)
    preview = service.preview(selection)
    assert preview["canFreeze"], preview
    request = FreezePredictor(
        **selection.model_dump(), previewHash=preview["previewHash"], operationId="create-refit"
    )
    return refits, refits.create(request), request


@pytest.mark.parametrize(
    ("epochs", "percentile", "expected"),
    [
        ([2, 3, 5, 8], 50, 4),
        ([2, 3, 5, 8], 75, 6),
        ([2, 3, 5, 8], 90, 8),
        ([7], 1, 7),
        ([1, 100], 100, 100),
    ],
)
def test_epoch_budget_linear_quantile_ceil(epochs, percentile, expected):
    assert epoch_budget(epochs, percentile) == expected


@pytest.mark.parametrize("legacy", [False, True])
def test_refit_uses_best_epochs_not_stopped_epochs_and_all_development_rows(registry, legacy):
    service, _ = registry
    selection, _, _ = refit_candidate(service, legacy_history=legacy)
    preview = service.preview(selection)
    assert preview["canFreeze"], preview
    manifest = preview["manifest"]
    assert manifest["epochBudget"]["epochs"] == 7
    assert sorted(row["bestEpoch"] for row in manifest["epochBudget"]["foldBestEpochs"]) == [3, 10]
    assert manifest["recipe"]["maxEpochs"] == manifest["recipe"]["minEpochs"] == 7
    assert manifest["recipe"]["earlyStopping"] is False
    rows = manifest["planTemplate"]["data"]["memberships"]
    protocol = service.store.get_configuration(manifest["inputs"]["protocol"]["id"])
    assert {row["slideId"] for row in rows} == {
        row["slideId"] for row in protocol["manifest"]["memberships"]
    }
    assert len(rows) == len({row["slideId"] for row in rows})
    assert all(row["partition"] == "train" and row["pool"] == "development" for row in rows)
    assert manifest["checkpoints"] == []


def test_refit_rejects_missing_best_epoch_instead_of_using_stopped_epochs(registry):
    service, _ = registry
    selection, _, _ = candidate(service)
    selection = selection.model_copy(update={"method": "refit"})
    preview = service.preview(selection)
    assert not preview["canFreeze"]
    assert preview["findings"][0]["code"] == "REFIT_BEST_EPOCH_UNAVAILABLE"


def test_ensemble_and_refit_are_independent_predictors_and_refit_publish_is_idempotent(registry):
    service, _ = registry
    selection, _, _ = refit_candidate(service)
    ensemble, _ = support["freeze"](service, selection.model_copy(update={"method": "ensemble"}))
    assert ensemble["manifest"]["method"] == "ensemble"
    choices = service.choices()["items"]
    assert choices[0]["eligibleMethods"] == ["refit"]
    jobs = FakeJobs(service.store)
    refits, record, request = create(service, selection, jobs)
    assert refits.create(request)["id"] == record["id"]
    assert len(service.list()["items"]) == 1
    with pytest.raises(StorageError, match="complete"):
        refits.publish(record["id"], "publish-refit")
    refits.launch(record["id"], LaunchRefit(operationId="launch-refit"))
    jobs.complete(record["id"])
    model = refits.publish(record["id"], "publish-refit")
    assert model["manifest"]["method"] == "refit"
    assert model["manifest"]["refitId"] == record["id"]
    assert len(model["manifest"]["checkpoints"]) == 1
    assert len(service.list()["items"]) == 2
    assert refits.publish(record["id"], "publish-refit")["id"] == model["id"]
    service.verify_checkpoints(model)
    assert not service.preview(selection)["canFreeze"]
    with pytest.raises(StorageError) as error:
        refits.publish(record["id"], "publish-other")
    assert error.value.code == "EXPERIMENT_ALREADY_FROZEN"


def test_refit_rejects_changed_source_after_review_or_training(registry):
    service, _ = registry
    selection, folder, state = refit_candidate(service)
    jobs = FakeJobs(service.store)
    refits, record, _ = create(service, selection, jobs)
    refits.launch(record["id"], LaunchRefit(operationId="launch-refit"))
    jobs.complete(record["id"])
    (folder / "runs" / state["runs"][0]["id"] / "best.ckpt").write_bytes(b"changed")
    with pytest.raises(StorageError) as error:
        refits.publish(record["id"], "publish-refit")
    assert error.value.code == "REFIT_EVIDENCE_CHANGED"


def test_accepted_refit_retry_skips_evidence_scan_but_rejects_changed_resources(registry, monkeypatch):
    service, _ = registry
    selection, _, _ = refit_candidate(service)
    job_support = runpy.run_path(str(Path(__file__).with_name("test_compute_jobs.py")))
    executor = job_support["Executor"]()
    jobs = ComputeJobService(service.store, executor=executor, runtime=lambda: {
        "available": True, "python": sys.executable, "versions": {},
        "cudaAvailable": False, "gpuCount": 0,
        "host": {"cpuCount": 8, "totalRamGb": 16}})
    refits, record, _ = create(service, selection, jobs)
    request = LaunchRefit(operationId="launch-refit", resources=ResourcePolicy(gpuIds=[]))
    first = refits.launch(record["id"], request)
    monkeypatch.setattr(refits, "_verify_sources", lambda *_: pytest.fail(
        "Accepted refit retry must not scan completed fold evidence"))
    assert refits.launch(record["id"], request)["planHash"] == first["planHash"]
    assert len(executor.calls) == 1
    with pytest.raises(StorageError) as caught:
        refits.launch(record["id"], LaunchRefit(operationId="launch-refit",
                      resources=ResourcePolicy(gpuIds=[], ramGbPerRun=2)))
    assert caught.value.code == "OPERATION_CONFLICT"


def test_refit_refuses_altered_completed_output_plan(registry):
    service, _ = registry
    selection, _, _ = refit_candidate(service)
    jobs = FakeJobs(service.store)
    refits, record, _ = create(service, selection, jobs)
    refits.launch(record["id"], LaunchRefit(operationId="launch-refit"))
    jobs.complete(record["id"])
    jobs.states[record["id"]]["result"] = copy.deepcopy(jobs.states[record["id"]]["result"])
    jobs.states[record["id"]]["result"]["epochsCompleted"] += 1
    with pytest.raises(StorageError) as error:
        refits.publish(record["id"], "publish-refit")
    assert error.value.code == "REFIT_PROVENANCE_CHANGED"


def test_legacy_epoch_selection_preserves_lightning_float32_first_tie(registry):
    service, _ = registry
    selection, folder, state = refit_candidate(service, epochs=(3, 3), legacy_history=True)
    for run in state["runs"]:
        history = [{"epoch": i, "validation": {"loss": 1.0}} for i in range(12)]
        history[1]["validation"]["loss"] = 0.7000000001
        history[2]["validation"]["loss"] = 0.7
        write_json(folder / "runs" / run["id"] / "history.json", history)
    preview = service.preview(selection)
    assert preview["canFreeze"], preview
    assert preview["manifest"]["epochBudget"]["epochs"] == 2


@pytest.mark.parametrize("legacy", [False, True])
def test_refit_evidence_survives_experiment_rename(registry, monkeypatch, legacy):
    """Renaming the owning experiment must not wedge a reviewed refit plan."""
    from histopilot.application import predictors as predictor_module
    from histopilot.application import refits as refit_module

    service, _ = registry
    selection, _, _ = refit_candidate(service)
    jobs = FakeJobs(service.store)
    # Reproduce both old hashes: the outer review hash and the name-dependent
    # provenance hash inside planTemplate. Neither is rewritten on upgrade.
    with monkeypatch.context() as old_release:
        if legacy:
            old_release.setattr(predictor_module, "evidence_hash", _hash)
            old_release.setattr(refit_module, "evidence_hash", _hash)
        refits, record, _ = create(service, selection, jobs)
    refits._verify_sources(record)
    experiment = service.store.get_draft(selection.experimentId)
    service.store.update_draft(
        experiment["id"],
        expected_revision=experiment["revision"],
        name=experiment["name"] + " (final)",
        payload=experiment["payload"],
    )
    refits.launch(record["id"], LaunchRefit(operationId="launch-refit"))
    jobs.complete(record["id"])
    model = refits.publish(record["id"], "publish-refit")
    # The published model keeps the plan-time name as provenance; the live name is display only.
    assert model["manifest"]["experiment"]["name"] == "Trial"
    assert model["manifest"]["refitId"] == record["id"]
    if not legacy:
        assert (
            predictor_module.evidence_hash(record["manifest"]) == record["manifest"]["previewHash"]
        )


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("change", ["checkpoint", "recipe", "membership", "stored_tamper"])
def test_refit_hash_compatibility_rejects_scientific_changes(registry, monkeypatch, legacy, change):
    from histopilot.application import predictors as predictor_module
    from histopilot.application import refits as refit_module

    service, _ = registry
    selection, _, _ = refit_candidate(service)
    jobs = FakeJobs(service.store)
    with monkeypatch.context() as old_release:
        if legacy:
            old_release.setattr(predictor_module, "evidence_hash", _hash)
            old_release.setattr(refit_module, "evidence_hash", _hash)
        refits, record, _ = create(service, selection, jobs)
    document = service.store.get_configuration(record["id"])
    refits._verify_sources(document)
    if change == "stored_tamper":
        document["manifest"]["recipe"]["maxEpochs"] += 1
    else:
        current = service._prepare(selection)
        if change == "checkpoint":
            current["sourceCheckpoints"][0]["sha256"] = "0" * 64
        elif change == "recipe":
            current["recipe"]["maxEpochs"] += 1
        else:
            current["planTemplate"]["data"]["memberships"][0]["label"] = "other-class"
        monkeypatch.setattr(refits.predictors, "_prepare", lambda selection: current)
    with pytest.raises(StorageError) as error:
        refits._verify_sources(document)
    assert error.value.code == "REFIT_EVIDENCE_CHANGED"
