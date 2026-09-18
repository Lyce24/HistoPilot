"""Validation winners control publication and automatic predictor construction."""

import copy
import runpy
import sys
from pathlib import Path

import pytest

from histopilot.application.experiment_predictors import ExperimentPredictorService
from histopilot.application.feature_bundles import _hash
from histopilot.application.model_experiments import execution_contract
from histopilot.application.refits import RefitService
from histopilot.workers.packing_process import write_json
from histopilot.workers.train_batch import _run_plan
from histopilot.workers.training_process import compute_snapshot, read_json

support = runpy.run_path(str(Path(__file__).with_name("test_predictor_registry.py")))
coordinator_support = runpy.run_path(str(Path(__file__).with_name("test_experiment_predictors.py")))
registry = support["registry"]


@pytest.fixture
def selected_batch(registry):
    service, _ = registry
    source, old_folder, old_state = support["candidate"](service, refit_ready=True)
    manifest = copy.deepcopy(service.store.get_configuration(source.batchId)["manifest"])
    manifest["spec"].update(
        selectionMetric="validation_auroc", candidateSelection="best_validation"
    )
    first = manifest["configurations"][0]
    second_recipe = {**first["recipe"], "learningRate": 0.002}
    second = {"id": "candidate-" + _hash(second_recipe), "number": 2, "recipe": second_recipe}
    manifest["configurations"].append(second)
    old_runs = manifest["runs"]
    manifest["runs"] = [
        {**row, "candidateId": candidate["id"], "id": "run-" + _hash([row["id"], candidate["id"]])}
        for candidate in manifest["configurations"]
        for row in old_runs
    ]
    batch = service.store.publish_configuration(manifest=manifest, operation_id="selected-batch")
    runtime = {"available": True, "python": sys.executable, "versions": {"torch": "fixture"}}
    plan = {
        **read_json(old_folder / "plan.json"),
        "batchId": batch["id"],
        "batchContentHash": batch["contentHash"],
        "selectionMetric": "validation_auroc",
        "candidateSelection": "best_validation",
        "configurations": manifest["configurations"],
        "runs": manifest["runs"],
        "runtime": runtime,
        "code": compute_snapshot(),
    }
    folder = service.store.folder / "training" / batch["id"]
    states = []
    for run in plan["runs"]:
        run_folder = folder / "runs" / run["id"]
        run_folder.mkdir(parents=True)
        checkpoint = run_folder / "best.ckpt"
        checkpoint.write_bytes(b"Synthetic checkpoint for validation selection")
        result = copy.deepcopy(old_state["runs"][0]["result"])
        score = 0.7 if run["candidateId"] == first["id"] else 0.8
        result.update(
            runId=run["id"],
            bestCheckpointPath=str(checkpoint),
            bestValidationScore=score,
            metrics={
                "validation": {
                    "unit": "patient",
                    "patientAggregation": "mean_probabilities",
                    "patient": {"available": True, "auroc": score},
                },
                "assessment": {"patient": {"auroc": 1 - score}},
            },
        )
        write_json(run_folder / "plan.json", _run_plan(plan, run, None))
        write_json(run_folder / "result.json", result)
        states.append({**run, "status": "completed", "result": result})
    state = {"batchId": batch["id"], "status": "completed", "planHash": _hash(plan), "runs": states}
    write_json(folder / "plan.json", plan)
    write_json(folder / "state.json", state)
    loser = source.model_copy(update={"batchId": batch["id"]})
    winner = loser.model_copy(update={"candidateId": second["id"], "name": "Validation winner"})
    return service, winner, loser, folder, plan, state


def test_only_validation_winner_can_publish_and_choices_explain_selection(selected_batch):
    service, winner, loser, *_ = selected_batch
    denied = service.preview(loser)
    assert not denied["canFreeze"]
    assert denied["findings"][0]["code"] == "PREDICTOR_VALIDATION_SELECTION_REQUIRED"
    choices = {
        row["candidateId"]: row
        for row in service.choices()["items"]
        if row["batchId"] == winner.batchId
    }
    assert choices[winner.candidateId]["eligible"]
    assert not choices[loser.candidateId]["eligible"]
    predictor, _ = support["freeze"](service, winner)
    evidence = predictor["manifest"]["selectionEvidence"]
    assert evidence["selectedCandidateId"] == winner.candidateId
    assert evidence["metric"] == "validation_auroc"


def test_competing_configuration_receipts_are_verified(selected_batch):
    service, winner, loser, folder, _plan, state = selected_batch
    next(row for row in state["runs"] if row["candidateId"] == loser.candidateId)["result"][
        "metrics"
    ]["validation"]["patient"]["auroc"] = 0.1
    write_json(folder / "state.json", state)
    preview = service.preview(winner)
    assert not preview["canFreeze"]
    assert preview["findings"][0]["code"] == "PREDICTOR_SELECTION_CHANGED"


def test_automatic_construction_skips_losers_and_finishes_without_extra_refits(selected_batch):
    predictors, winner, loser, _folder, plan, _state = selected_batch
    store = predictors.store
    record = store.get_draft(winner.experimentId)
    policy = {"method": "ensemble", "refitPercentile": None}
    submission = {
        "operationId": "submission",
        "expectedRevision": 1,
        "submittedAt": record["createdAt"],
        "status": "submitted",
        "error": None,
        "batchIds": [winner.batchId],
        "publications": [],
        "executionContract": execution_contract(plan),
        "predictorPolicy": policy,
        "experiment": {"name": record["name"]},
    }
    store.update_draft(
        record["id"],
        expected_revision=record["revision"],
        name=record["name"],
        payload={**record["payload"], "submission": submission, "predictorPolicy": policy},
    )
    runtime = plan["runtime"]
    jobs = coordinator_support["Jobs"](store, runtime)
    executor = coordinator_support["Executor"]()
    current = ExperimentPredictorService(
        store,
        predictors.filesystem,
        executor=executor,
        training=coordinator_support["Training"](store.folder),
        refits=RefitService(store, predictors.filesystem, jobs=jobs),
        runtime=lambda: runtime,
    )
    current.launch(record["id"], "start-selected")
    result = current.advance(record["id"])
    assert result["status"] == "completed", result
    assert result["counts"]["completed"] == 1 and result["counts"]["skipped"] == 1
    assert {
        row["source"]["candidateId"] for row in result["items"] if row["status"] == "completed"
    } == {winner.candidateId}
    assert not jobs.launches
    assert current.advance(record["id"])["counts"] == result["counts"]
