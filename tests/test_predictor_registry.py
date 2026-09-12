"""Synthetic training receipts exercise promotion without launching training workers."""

import copy
import runpy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import pytest

from histopilot.application.development import development_plans
from histopilot.application.evaluation_runs import EvaluationRunService
from histopilot.application.feature_bundles import _hash
from histopilot.application.lifecycle import CleanupService
from histopilot.application.predictors import PredictorService
from histopilot.application.training import membership_plan_id
from histopilot.schemas.development import TrainingRecipe
from histopilot.schemas.lifecycle import CleanupSelection
from histopilot.schemas.predictors import (
    EvaluationRunSelection,
    FreezePredictor,
    PredictorSelection,
    SaveEvaluationRun,
)
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import ScientificStore
from histopilot.workers.packing_process import write_json
from histopilot.workers.train_batch import _run_plan
from histopilot.workers.training_process import process_identity

support = runpy.run_path(str(Path(__file__).with_name("test_evaluations.py")))


@pytest.fixture
def registry(tmp_path):
    folder = tmp_path / "project"
    folder.mkdir()
    store = ScientificStore(folder, "project-predictors")
    cohorts, spec, _source = support["setup"](store, tmp_path)
    protocol = store.get_configuration(spec["protocolId"])
    manifest = copy.deepcopy(protocol["manifest"])
    manifest["spec"]["split"].update(mode="kfold", seeds=[42], folds=2)
    manifest["memberships"] = [
        {**row, "seed": 42, "fold": fold} for fold in (0, 1) for row in manifest["memberships"]
    ]
    protocol = store.publish_configuration(manifest=manifest, operation_id="kfold-protocol")
    spec["protocolId"] = protocol["id"]
    draft = support["draft"](cohorts, spec)
    preview = cohorts.preview(draft["id"], 1)
    assert preview["canFreeze"], preview["findings"]
    cohort = cohorts.freeze(draft["id"], 1, preview["previewHash"], "test-cohort")
    predictor = PredictorService(store, cohorts.protocols.filesystem)
    return predictor, cohort


def candidate(service, name="Trial", *, legacy=False, refit_ready=False):
    store = service.store
    protocol = next(
        item
        for item in store.list_configurations("protocol")
        if item["manifest"]["spec"]["split"].get("mode") == "kfold"
    )
    if refit_ready:
        manifest = copy.deepcopy(protocol["manifest"])
        labels = manifest["spec"]["target"]["classes"]
        for row in manifest["memberships"]:
            row["label"] = labels[int(row["slideId"].removeprefix("s")) % len(labels)]
        protocol = store.publish_configuration(manifest=manifest, operation_id=uuid4().hex)
    bundle = store.list_configurations("feature-bundle")[0]
    feature = store.get_configuration(bundle["manifest"]["feature"]["id"])
    experiment = store.create_draft("experiment", name, {"type": "model-experiment"})
    inputs = {"protocolId": protocol["id"], "featureBundleId": bundle["id"]}
    recipe = TrainingRecipe().model_dump()
    candidate_id = "candidate-" + _hash(recipe)
    splits = development_plans(protocol["manifest"])
    runs = [
        {
            "id": "run-" + _hash(split),
            "candidateId": candidate_id,
            "trainingSeed": 11,
            "splitPlanId": split["id"],
            "status": "planned",
        }
        for split in splits
    ]
    spec = {"experimentName": name, "batchName": name + " batch", "inputs": inputs}
    if not legacy:
        spec.update(experimentId=experiment["id"], experimentRevision=1)
    manifest = {
        "kind": "mil-batch",
        "datasetId": protocol["manifest"]["datasetId"],
        "spec": spec,
        "configurations": [{"id": candidate_id, "number": 1, "recipe": recipe}],
        "splitPlans": splits,
        "runs": runs,
    }
    batch = store.publish_configuration(manifest=manifest, operation_id=uuid4().hex)
    folder = store.folder / "training" / batch["id"]
    plan = {
        "batchId": batch["id"],
        "batchContentHash": batch["contentHash"],
        "protocolContentHash": protocol["contentHash"],
        "featureBundleContentHash": bundle["contentHash"],
        "configurations": manifest["configurations"],
        "runs": runs,
        "splitPlans": splits,
        "target": protocol["manifest"]["spec"]["target"],
        "resources": {},
        "runtime": {"python": "/fixture/python", "versions": {"lightning": "fixture"}},
        "code": {"sha256": "fixture"},
        "memberships": {
            split["id"]: [
                row
                for row in protocol["manifest"]["memberships"]
                if membership_plan_id(row) == split["id"]
            ]
            for split in splits
        },
        "data": {
            "featureDim": feature["manifest"]["files"][0]["dimensions"],
            "featureFiles": {
                row["slideId"]: row
                for row in feature["manifest"]["files"]
                if row["slideId"] in {row["slideId"] for row in protocol["manifest"]["memberships"]}
            },
            "sourceStamps": {
                row["path"]: row
                for row in feature["manifest"]["files"]
                if row["slideId"] in {row["slideId"] for row in protocol["manifest"]["memberships"]}
            },
        },
    }
    states = []
    for run in runs:
        run_folder = folder / "runs" / run["id"]
        run_folder.mkdir(parents=True)
        checkpoint = run_folder / "best.ckpt"
        checkpoint.write_bytes(b"Synthetic checkpoint bytes; these tests do not unpickle weights.")
        result = {
            "runId": run["id"],
            "state": "succeeded",
            "bestCheckpointPath": str(checkpoint),
            "checkpointMetric": recipe["checkpointMetric"],
            "bestValidationScore": 0.7,
            "epochsCompleted": 2,
        }
        write_json(run_folder / "result.json", result)
        write_json(run_folder / "plan.json", _run_plan(plan, run, None))
        states.append({**run, "status": "completed", "result": result})
    state = {"batchId": batch["id"], "status": "completed", "planHash": _hash(plan), "runs": states}
    write_json(folder / "plan.json", plan)
    write_json(folder / "state.json", state)
    selection = PredictorSelection(
        experimentId=f"legacy-{batch['id']}" if legacy else experiment["id"],
        batchId=batch["id"],
        candidateId=candidate_id,
        trainingSeed=11,
        splitSeed=42,
        name=name + " predictor",
    )
    return selection, folder, state


def freeze(service, selection, operation=None):
    preview = service.preview(selection)
    assert preview["canFreeze"], preview
    request = FreezePredictor(
        **selection.model_dump(),
        previewHash=preview["previewHash"],
        operationId=operation or uuid4().hex,
    )
    return service.freeze(request), request


def lifecycle(store, document, state):
    store.lifecycle.apply(
        {f"configuration:{document['id']}": state},
        operation_id=uuid4().hex,
        request_hash=_hash({"id": document["id"], "state": state}),
        expected_revision=store.lifecycle.read()["revision"],
    )


def test_complete_candidate_freezes_exact_inputs_folds_hashes_and_immutable_retry(registry):
    service, _cohort = registry
    selection, folder, state = candidate(service)
    choices = service.choices()["items"]
    assert choices[0]["eligible"] and choices[0]["completedRuns"] == 2
    predictor, request = freeze(service, selection)
    manifest = predictor["manifest"]
    assert manifest["runIds"] == sorted(run["id"] for run in state["runs"])
    assert manifest["aggregation"] == "mean_probability"
    assert all(len(row["sha256"]) == 64 and row["bytes"] > 0 for row in manifest["checkpoints"])
    assert manifest["inputs"]["features"]["sourceContentHash"]
    assert service.freeze(request) == predictor
    # Response-loss retry returns the existing publication, never promotes new evidence.
    Path(manifest["checkpoints"][0]["path"]).write_bytes(b"changed")
    assert service.freeze(request) == predictor
    assert service.choices()["items"][0]["eligibleMethods"] == ["refit"]
    assert (folder / "state.json").exists()


def test_multiple_experiments_have_distinct_predictors_and_legacy_promotion(registry):
    service, _cohort = registry
    one, *_ = candidate(service, "One")
    two, *_ = candidate(service, "Two", legacy=True)
    first, _ = freeze(service, one)
    second, _ = freeze(service, two)
    assert first["id"] != second["id"]
    assert len(service.list()["items"]) == 2
    assert second["manifest"]["experiment"]["legacy"]


@pytest.mark.parametrize("state", ["archived", "trashed"])
def test_archive_and_trash_preserve_one_predictor_per_experiment(registry, state):
    service, _cohort = registry
    selection, *_ = candidate(service)
    predictor, _ = freeze(service, selection)
    lifecycle(service.store, predictor, state)
    assert not service.list()["items"]
    assert service.list(include_inactive=True)["items"][0]["lifecycleState"] == state
    preview = service.preview(selection)
    assert preview["findings"][0]["code"] == "EXPERIMENT_ALREADY_FROZEN"
    lifecycle(service.store, predictor, "active")
    assert service.get(predictor["id"])["id"] == predictor["id"]


def test_parallel_freeze_cannot_create_second_predictor(registry):
    service, _cohort = registry
    selection, *_ = candidate(service)
    preview = service.preview(selection)

    def submit(index):
        try:
            return service.freeze(
                FreezePredictor(
                    **selection.model_dump(),
                    previewHash=preview["previewHash"],
                    operationId=f"parallel-{index}",
                )
            )
        except StorageError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(submit, range(2)))
    assert sum(isinstance(item, dict) for item in outcomes) == 1
    assert len(service.list()["items"]) == 1


def test_incomplete_and_foreign_candidate_memberships_are_rejected(registry):
    service, _cohort = registry
    selection, folder, state = candidate(service)
    state["runs"][0]["status"] = "running"
    write_json(folder / "state.json", state)
    assert service.preview(selection)["findings"][0]["code"] == "PREDICTOR_RUN_INCOMPLETE"
    assert service.choices()["items"][0]["eligibleMethods"] == []
    other, *_ = candidate(service, "Other")
    foreign = selection.model_copy(update={"experimentId": other.experimentId})
    assert service.preview(foreign)["findings"][0]["code"] == "PREDICTOR_EXPERIMENT_MISMATCH"
    wrong_seed = selection.model_copy(update={"trainingSeed": 99})
    assert service.preview(wrong_seed)["findings"][0]["code"] == "PREDICTOR_CANDIDATE_MISSING"


def test_changed_checkpoint_or_receipt_invalidates_review(registry):
    service, _cohort = registry
    selection, _folder, state = candidate(service)
    preview = service.preview(selection)
    checkpoint = Path(state["runs"][0]["result"]["bestCheckpointPath"])
    checkpoint.write_bytes(b"new checkpoint")
    with pytest.raises(StorageError) as error:
        service.freeze(
            FreezePredictor(
                **selection.model_dump(), previewHash=preview["previewHash"], operationId="stale"
            )
        )
    assert error.value.code == "PREVIEW_STALE"
    receipt = state["runs"][0]["result"]
    receipt["bestValidationScore"] = 0.1
    write_json(checkpoint.parent / "result.json", receipt)
    assert service.preview(selection)["findings"][0]["code"] == "PREDICTOR_PROVENANCE_CHANGED"


def test_checkpoint_cannot_escape_run_directory(registry, tmp_path):
    service, _cohort = registry
    selection, folder, state = candidate(service)
    run = state["runs"][0]
    outside = tmp_path / "unrelated.ckpt"
    outside.write_bytes(b"unrelated")
    run["result"]["bestCheckpointPath"] = str(outside)
    write_json(folder / "runs" / run["id"] / "result.json", run["result"])
    write_json(folder / "state.json", state)
    assert service.preview(selection)["findings"][0]["code"] == "PREDICTOR_EVIDENCE_INVALID"


def test_operation_identity_is_bound_to_predictor_selection(registry):
    service, _cohort = registry
    selection, *_ = candidate(service)
    _predictor, request = freeze(service, selection)
    with pytest.raises(StorageError) as error:
        service.freeze(request.model_copy(update={"name": "Different predictor"}))
    assert error.value.code == "OPERATION_CONFLICT"


def test_evaluation_plan_binds_exact_predictor_cohort_and_reports_no_execution(registry):
    service, cohort = registry
    selection, *_ = candidate(service)
    predictor, _ = freeze(service, selection)
    evaluations = EvaluationRunService(service.store, service.filesystem)
    selection = EvaluationRunSelection(
        predictorId=predictor["id"], cohortId=cohort["id"], name="External validation"
    )
    preview = evaluations.preview(selection)
    assert preview["canSave"], preview
    request = SaveEvaluationRun(
        **selection.model_dump(), previewHash=preview["previewHash"], operationId="evaluation"
    )
    saved = evaluations.save(request)
    assert saved["manifest"]["predictorId"] == predictor["id"]
    assert saved["manifest"]["cohortId"] == cohort["id"]
    assert saved["manifest"]["status"] == "planned"
    assert saved["manifest"]["results"] is None
    assert saved["manifest"]["executionEnabled"]
    assert evaluations.save(request) == saved
    assert len(evaluations.list()["items"]) == 1
    lifecycle(service.store, predictor, "archived")
    assert evaluations.get(saved["id"])["id"] == saved["id"]
    assert evaluations.preview(selection)["canSave"]


def test_evaluation_rejects_mismatched_protocol_and_changed_checkpoints(registry, monkeypatch):
    service, cohort = registry
    selection, *_ = candidate(service)
    predictor, _ = freeze(service, selection)
    evaluations = EvaluationRunService(service.store, service.filesystem)
    selection = EvaluationRunSelection(
        predictorId=predictor["id"], cohortId=cohort["id"], name="Test"
    )
    altered = copy.deepcopy(cohort)
    altered["current"] = True
    altered["manifest"]["bindings"]["protocol"]["contentHash"] = "other"
    monkeypatch.setattr(evaluations.cohorts, "get", lambda _: altered)
    assert evaluations.preview(selection)["findings"][0]["code"] == "EVALUATION_PREDICTOR_MISMATCH"
    monkeypatch.undo()
    Path(predictor["manifest"]["checkpoints"][0]["path"]).write_bytes(b"changed weights")
    assert evaluations.preview(selection)["findings"][0]["code"] == "PREDICTOR_CHECKPOINT_CHANGED"


def test_evaluation_rejects_aggregation_drift_and_stale_cohort(registry, monkeypatch):
    service, cohort = registry
    selected, *_ = candidate(service)
    predictor, _ = freeze(service, selected)
    evaluations = EvaluationRunService(service.store, service.filesystem)
    selection = EvaluationRunSelection(
        predictorId=predictor["id"], cohortId=cohort["id"], name="Test"
    )
    altered = copy.deepcopy(cohort)
    altered["current"] = True
    altered["manifest"]["spec"]["inference"]["patientAggregation"] = "max"
    monkeypatch.setattr(evaluations.cohorts, "get", lambda _: altered)
    assert (
        evaluations.preview(selection)["findings"][0]["code"] == "EVALUATION_AGGREGATION_MISMATCH"
    )
    altered["current"] = False
    assert evaluations.preview(selection)["findings"][0]["code"] == "EVALUATION_COHORT_STALE"


def test_completed_receipt_with_live_or_unverifiable_process_cannot_be_promoted(
    registry, monkeypatch
):
    service, _cohort = registry
    selection, folder, state = candidate(service)
    state["runs"][0]["process"] = process_identity()
    write_json(folder / "state.json", state)
    assert service.preview(selection)["findings"][0]["code"] == "PREDICTOR_RUN_INCOMPLETE"

    def unknown(_identity):
        raise StorageError("Unreadable process", "CLEANUP_PROCESS_UNKNOWN")

    monkeypatch.setattr("histopilot.application.lifecycle._confirmed_live", unknown)
    assert service.preview(selection)["findings"][0]["code"] == "PREDICTOR_PROCESS_UNKNOWN"


def test_mutated_memberships_are_rejected_even_if_execution_hash_is_rewritten(registry):
    service, _cohort = registry
    selection, folder, state = candidate(service)
    import json

    plan = json.loads((folder / "plan.json").read_text())
    split_id = next(iter(plan["memberships"]))
    plan["memberships"][split_id][0]["slideId"] = "wrong-slide"
    state["planHash"] = _hash(plan)
    write_json(folder / "plan.json", plan)
    write_json(folder / "state.json", state)
    assert service.preview(selection)["findings"][0]["code"] == "PREDICTOR_PROVENANCE_CHANGED"


def test_malformed_receipts_fail_closed_with_review_finding(registry):
    service, _cohort = registry
    selection, folder, state = candidate(service)
    run = state["runs"][0]
    del run["result"]["bestCheckpointPath"]
    write_json(folder / "runs" / run["id"] / "result.json", run["result"])
    write_json(folder / "state.json", state)
    assert service.preview(selection)["findings"][0]["code"] == "PREDICTOR_EVIDENCE_INVALID"


def test_cleanup_follows_experiment_predictor_and_evaluation_dependencies(registry):
    service, cohort = registry
    selection, *_ = candidate(service)
    predictor, _ = freeze(service, selection)
    evaluations = EvaluationRunService(service.store, service.filesystem)
    selected = EvaluationRunSelection(
        predictorId=predictor["id"], cohortId=cohort["id"], name="Test"
    )
    preview = evaluations.preview(selected)
    saved = evaluations.save(
        SaveEvaluationRun(
            **selected.model_dump(), previewHash=preview["previewHash"], operationId="evaluate"
        )
    )
    cleanup = CleanupService(service.store, service.filesystem, "Fixture")
    review = cleanup.preview(
        CleanupSelection(action="trash", keys=[f"draft:{selection.experimentId}"])
    )
    assert not review["canApply"]
    assert {
        f"configuration:{selection.batchId}",
        f"configuration:{predictor['id']}",
        f"configuration:{saved['id']}",
    }.issubset(set(review["requiredKeys"]))
    predictor_review = cleanup.preview(
        CleanupSelection(action="trash", keys=[f"configuration:{predictor['id']}"])
    )
    assert f"configuration:{saved['id']}" in predictor_review["requiredKeys"]


def test_evaluation_operation_cannot_be_reused_for_other_predictor_or_name(registry):
    service, cohort = registry
    selected, *_ = candidate(service)
    predictor, _ = freeze(service, selected)
    evaluations = EvaluationRunService(service.store, service.filesystem)
    selection = EvaluationRunSelection(
        predictorId=predictor["id"], cohortId=cohort["id"], name="Test"
    )
    preview = evaluations.preview(selection)
    request = SaveEvaluationRun(
        **selection.model_dump(), previewHash=preview["previewHash"], operationId="save"
    )
    evaluations.save(request)
    with pytest.raises(StorageError) as error:
        evaluations.save(request.model_copy(update={"name": "Other"}))
    assert error.value.code == "OPERATION_CONFLICT"
