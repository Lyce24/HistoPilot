"""Synthetic trained candidates, frozen predictors and the fakes that refit them.

``registry`` is a project with a frozen evaluation cohort and a 2-fold protocol;
``candidate`` writes a completed batch's receipts without training anything, and
``freeze`` publishes a predictor from it. ``managed`` queues an experiment's predictor
coordinator in this test's Task Center while refits stay on ``FakeJobs``.
"""

import copy
import json
import sys
from uuid import uuid4

import pytest

from histopilot.application.development import development_plans
from histopilot.application.experiment_predictors import ExperimentPredictorService
from histopilot.application.feature_bundles import _hash
from histopilot.application.model_experiments import execution_contract
from histopilot.application.predictors import PredictorService
from histopilot.application.refits import RefitService
from histopilot.application.training import membership_plan_id
from histopilot.schemas.development import TrainingRecipe
from histopilot.schemas.predictors import FreezePredictor, PredictorSelection
from histopilot.storage.scientific import ScientificStore
from histopilot.workers.packing_process import write_json
from histopilot.workers.train_batch import _run_plan
from histopilot.workers.training_process import compute_snapshot, read_json
from support.projects import draft, setup


@pytest.fixture
def registry(tmp_path):
    """A predictor service over a frozen test cohort of ``projects.setup``, 2-fold protocol."""
    folder = tmp_path / "project"
    folder.mkdir()
    store = ScientificStore(folder, "project-predictors")
    cohorts, spec, _source = setup(store, tmp_path)
    protocol = store.get_configuration(spec["protocolId"])
    manifest = copy.deepcopy(protocol["manifest"])
    manifest["spec"]["split"].update(mode="kfold", seeds=[42], folds=2)
    manifest["memberships"] = [
        {**row, "seed": 42, "fold": fold} for fold in (0, 1) for row in manifest["memberships"]
    ]
    protocol = store.publish_configuration(manifest=manifest, operation_id="kfold-protocol")
    spec["protocolId"] = protocol["id"]
    cohort_draft = draft(cohorts, spec)
    preview = cohorts.preview(cohort_draft["id"], 1)
    assert preview["canFreeze"], preview["findings"]
    cohort = cohorts.freeze(cohort_draft["id"], 1, preview["previewHash"], "test-cohort")
    predictor = PredictorService(store, cohorts.protocols.filesystem)
    return predictor, cohort


def candidate(
    service,
    name="Trial",
    *,
    legacy=False,
    refit_ready=False,
    checkpoint_metric=None,
    model="abmil",
    feature_bundle_id=None,
):
    """A completed one-seed batch over the k-fold protocol, from receipts alone.

    Returns the predictor selection, the batch's training folder and its saved state.
    """
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
    bundle = (
        store.get_configuration(feature_bundle_id)
        if feature_bundle_id
        else store.list_configurations("feature-bundle")[0]
    )
    feature = store.get_configuration(bundle["manifest"]["feature"]["id"])
    experiment = store.create_draft("experiment", name, {"type": "model-experiment"})
    inputs = {"protocolId": protocol["id"], "featureBundleId": bundle["id"]}
    recipe = TrainingRecipe(model=model).model_dump()
    if checkpoint_metric is not None:
        recipe["checkpointMetric"] = checkpoint_metric
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
        **(
            {"splitUnit": protocol["manifest"]["spec"]["splitUnit"]}
            if "splitUnit" in protocol["manifest"]["spec"]
            else {}
        ),
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
    """Publish the predictor ``selection`` names; returns it and the request."""
    preview = service.preview(selection)
    assert preview["canFreeze"], preview
    request = FreezePredictor(
        **selection.model_dump(),
        previewHash=preview["previewHash"],
        operationId=operation or uuid4().hex,
    )
    return service.freeze(request), request


# -- refits --------------------------------------------------------------------------------


class FakeJobs:
    """Compute jobs that record launches and complete only when a test says so."""

    def __init__(self, store):
        self.store, self.states = store, {}

    def folder(self, identity):
        return self.store.folder / "compute-jobs" / identity

    def status(self, identity, **kwargs):
        return self.states.get(identity, {"status": "not_started"})

    def replay_launch(self, identity, operation_id, **kwargs):
        return None

    def launch(self, identity, plan, operation_id, resume=False, **task):
        self.task_options = task  # Task Center owner/title; the fake never queues anything.
        folder = self.folder(identity)
        folder.mkdir(parents=True, exist_ok=True)
        write_json(folder / "plan.json", plan)
        self.states[identity] = {"status": "running", "planHash": _hash(plan), "result": None}
        return self.states[identity]

    def complete(self, identity):
        folder = self.folder(identity)
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
    """A ``candidate`` whose folds stopped at 12 epochs, best at ``epochs``; a refit selection."""
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
    """Create the refit ``selection`` names on ``jobs``; returns (service, record, request)."""
    refits = RefitService(service.store, service.filesystem, jobs)
    preview = service.preview(selection)
    assert preview["canFreeze"], preview
    request = FreezePredictor(
        **selection.model_dump(), previewHash=preview["previewHash"], operationId="create-refit"
    )
    return refits, refits.create(request), request


def two_seeds(service):
    """A refit-ready candidate trained again with seeds 11 and 22; one selection per seed."""
    source, original_folder, original_state = refit_candidate(service, epochs=(2, 4))
    original = service.store.get_configuration(source.batchId)
    manifest = copy.deepcopy(original["manifest"])
    originals = manifest["runs"]
    manifest["runs"] = [
        {**row, "id": "run-" + _hash([row["id"], seed]), "trainingSeed": seed}
        for seed in (11, 22)
        for row in originals
    ]
    batch = service.store.publish_configuration(manifest=manifest, operation_id=uuid4().hex)
    plan = {
        **read_json(original_folder / "plan.json"),
        "batchId": batch["id"],
        "batchContentHash": batch["contentHash"],
        "runs": manifest["runs"],
    }
    folder = service.store.folder / "training" / batch["id"]
    states = []
    for index, run in enumerate(manifest["runs"]):
        run_folder = folder / "runs" / run["id"]
        run_folder.mkdir(parents=True)
        checkpoint = run_folder / "best.ckpt"
        checkpoint.write_bytes(b"seed checkpoint")
        result = {
            **original_state["runs"][index % 2]["result"],
            "runId": run["id"],
            "bestCheckpointPath": str(checkpoint),
        }
        write_json(run_folder / "result.json", result)
        write_json(run_folder / "plan.json", _run_plan(plan, run, None))
        states.append({**run, "status": "completed", "result": result})
    write_json(folder / "plan.json", plan)
    write_json(
        folder / "state.json",
        {"batchId": batch["id"], "status": "completed", "planHash": _hash(plan), "runs": states},
    )
    return [
        source.model_copy(update={"batchId": batch["id"], "trainingSeed": seed})
        for seed in (11, 22)
    ], folder


# -- experiment predictors -----------------------------------------------------------------


class Training:
    """Training that reads each batch's saved state and never launches."""

    def __init__(self, folder):
        self.folder = folder

    def execution(self, identity, **_kwargs):
        return read_json(self.folder / "training" / identity / "state.json")


class Jobs(FakeJobs):
    """``FakeJobs`` that also cancel and record their launches and cancels."""

    def __init__(self, store, runtime):
        super().__init__(store)
        self.runtime = lambda: runtime
        self.launches, self.cancels = [], []

    def launch(self, identity, plan, operation_id, resume=False, **task):
        self.launches.append((identity, operation_id, resume))
        return super().launch(identity, plan, operation_id, resume, **task)

    def cancel(self, identity, operation_id=None):
        self.cancels.append(identity)
        self.states[identity] = {**self.states[identity], "status": "cancelled"}
        return self.states[identity]


def submitted(registry):
    """An experiment whose two-seed CV batch finished, submitted with both predictor methods."""
    predictors, _cohort = registry
    selections, folder = two_seeds(predictors)
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
    return predictors, identity, runtime, selections


@pytest.fixture
def managed(registry, task_center):
    """The coordinator queued in this test's Task Center, as in production.

    Tests call ``advance`` themselves, as the coordinator's worker would; refits stay fake.
    """
    predictors, identity, runtime, selections = submitted(registry)
    store = predictors.store
    jobs = Jobs(store, runtime)
    service = ExperimentPredictorService(
        store,
        predictors.filesystem,
        training=Training(store.folder),
        refits=RefitService(store, predictors.filesystem, jobs=jobs),
        runtime=lambda: runtime,
        task_center=task_center.client,
    )
    return service, identity, jobs, selections
