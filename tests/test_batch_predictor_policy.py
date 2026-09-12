"""Per-batch predictor recipes coexist without changing historical frozen meaning."""

import copy
import runpy
from pathlib import Path

import pytest

from histopilot.application.experiment_policy import policy_for_batch, submission_policies
from histopilot.application.feature_bundles import _hash
from histopilot.application.model_experiments import ModelExperimentService
from histopilot.schemas.development import DevelopmentBatchSpec, SearchGrid, TrainingRecipe
from histopilot.storage.project_lock import StorageError
from histopilot.workers.packing_process import write_json
from histopilot.workers.train_batch import _run_plan
from histopilot.workers.training_process import read_json

support = runpy.run_path(str(Path(__file__).with_name("test_experiment_predictors.py")))
registry = support["registry"]
integrated = support["integrated"]


def new_batch(service, original_id, name, policy):
    original = service.store.get_configuration(original_id)
    manifest = copy.deepcopy(original["manifest"])
    manifest["spec"].update(batchName=name, predictorPolicy=policy)
    batch = service.store.publish_configuration(manifest=manifest, operation_id=name)
    source = service.store.folder / "training" / original_id
    destination = service.store.folder / "training" / batch["id"]
    plan = read_json(source / "plan.json")
    plan.update(batchId=batch["id"], batchContentHash=batch["contentHash"])
    state = read_json(source / "state.json")
    state.update(batchId=batch["id"], planHash=_hash(plan))
    for run in state["runs"]:
        folder = destination / "runs" / run["id"]
        folder.mkdir(parents=True)
        checkpoint = folder / "best.ckpt"
        checkpoint.write_bytes(Path(run["result"]["bestCheckpointPath"]).read_bytes())
        run["result"]["bestCheckpointPath"] = str(checkpoint)
        write_json(folder / "result.json", run["result"])
        intent = next(row for row in plan["runs"] if row["id"] == run["id"])
        write_json(folder / "plan.json", _run_plan(plan, intent, None))
    write_json(destination / "plan.json", plan)
    write_json(destination / "state.json", state)
    return batch


@pytest.fixture
def mixed(integrated):
    service, identity, jobs, executor, selections = integrated
    policies = [
        {"method": "skip", "refitPercentile": None},
        {"method": "ensemble", "refitPercentile": None},
        {"method": "refit", "refitPercentile": 50.0},
        {"method": "both", "refitPercentile": 75.0},
    ]
    batches = [
        new_batch(service, selections[0].batchId, f"mixed-{index}", policy)
        for index, policy in enumerate(policies)
    ]
    record = service.store.get_draft(identity)
    payload = copy.deepcopy(record["payload"])
    payload["submission"].pop("predictorPolicy")
    payload["submission"]["batchIds"] = [row["id"] for row in batches]
    payload["submission"]["predictorPolicies"] = {
        batch["id"]: policy for batch, policy in zip(batches, policies, strict=True)
    }
    service.store.update_draft(
        identity, expected_revision=record["revision"], name=record["name"], payload=payload
    )
    return service, identity, jobs, executor, batches


def test_new_defaults_and_legacy_omissions_keep_separate_meanings():
    assert TrainingRecipe().maxEpochs == 40
    assert TrainingRecipe().patience == 8
    assert SearchGrid().maxEpochs == [40]
    legacy = TrainingRecipe.model_validate({}, context={"legacy": True})
    assert legacy.maxEpochs == 100 and legacy.patience == 15
    assert SearchGrid.model_validate({}, context={"legacy": True}).maxEpochs == [100]
    explicit = TrainingRecipe.model_validate(
        {"maxEpochs": 120, "patience": 19}, context={"legacy": True}
    )
    assert explicit.maxEpochs == 120 and explicit.patience == 19


def test_optional_policy_does_not_alter_legacy_batch_serialization(integrated):
    service, _identity, _jobs, _executor, selections = integrated
    spec = service.store.get_configuration(selections[0].batchId)["manifest"]["spec"]
    spec.pop("experimentId", None)
    spec.pop("experimentRevision", None)
    legacy = DevelopmentBatchSpec.model_validate(spec, context={"legacy": True})
    assert legacy.recipe.maxEpochs == 100 and legacy.recipe.patience == 15
    assert "predictorPolicy" not in legacy.model_dump()
    assert legacy.grid.maxEpochs == [100]
    new = DevelopmentBatchSpec.model_validate(spec)
    assert new.recipe.maxEpochs == 40 and new.recipe.patience == 8


def test_mixed_batches_have_exact_method_counts_and_individual_epoch_budgets(mixed):
    service, identity, jobs, _executor, batches = mixed
    started = service.launch(identity, "start")
    assert started["counts"]["total"] == 8
    assert started["counts"]["ensemble"] == started["counts"]["refit"] == 4
    assert batches[0]["id"] not in {row["source"]["batchId"] for row in started["items"]}
    plan, _state = service._read(identity)
    assert plan["version"] == 2 and "policy" not in plan
    assert {row["refitPercentile"] for row in plan["items"] if row["method"] == "refit"} == {
        50.0,
        75.0,
    }
    for _ in range(8):
        result = service.advance(identity)
        for item in result["items"]:
            if (
                item["method"] == "refit"
                and item["recordId"]
                and jobs.status(item["recordId"])["status"] == "running"
            ):
                jobs.complete(item["recordId"])
        if result["status"] == "completed":
            break
    assert result["status"] == "completed" and result["counts"]["completed"] == 8
    assert len(jobs.launches) == 4
    budgets = {
        (row["refitPercentile"], row["epochBudget"]["epochs"])
        for row in result["items"]
        if row["method"] == "refit"
    }
    assert budgets == {(50.0, 3), (75.0, 4)}
    models = ModelExperimentService(
        service.store, service.filesystem, training=service.training, predictor_execution=service
    )
    record = models.get(identity)
    assert record["predictorPolicy"] is None
    assert set(record["predictorPolicies"]) == {row["id"] for row in batches}
    assert record["stage"] == "finished"


def test_manual_build_cannot_use_another_batch_policy(mixed):
    service, identity, _jobs, _executor, batches = mixed
    from histopilot.schemas.predictors import PredictorSelection

    for batch, method, percentile in [
        (batches[0], "ensemble", 50),
        (batches[1], "refit", 50),
        (batches[2], "refit", 75),
        (batches[3], "refit", 50),
    ]:
        selection = PredictorSelection(
            experimentId=identity,
            batchId=batch["id"],
            candidateId=batch["manifest"]["configurations"][0]["id"],
            trainingSeed=11,
            splitSeed=42,
            name="Wrong policy",
            method=method,
            refitPercentile=percentile,
        )
        preview = service.builds.predictors.preview(selection)
        assert not preview["canFreeze"]
        assert preview["findings"][0]["code"] == "EXPERIMENT_PREDICTOR_POLICY_LOCKED"


def test_old_global_submissions_keep_version_one_plan_and_original_item_shape(integrated):
    service, identity, _jobs, _executor, selections = integrated
    service.launch(identity, "original-start")
    plan, _state = service._read(identity)
    assert plan["version"] == 1 and plan["policy"]["refitPercentile"] == 75
    assert "policies" not in plan and all("refitPercentile" not in row for row in plan["items"])
    submission = service.store.get_draft(identity)["payload"]["submission"]
    assert policy_for_batch(submission, selections[0].batchId) == plan["policy"]
    assert service.status(identity)["counts"]["total"] == 4


def test_incomplete_frozen_policy_map_fails_closed(mixed):
    service, identity, _jobs, _executor, batches = mixed
    submission = copy.deepcopy(service.store.get_draft(identity)["payload"]["submission"])
    submission["predictorPolicies"].pop(batches[0]["id"])
    with pytest.raises(StorageError) as error:
        submission_policies(submission)
    assert error.value.code == "EXPERIMENT_PREDICTOR_PLAN_CHANGED"


def test_new_submission_freezes_each_batch_choice_before_dispatch(tmp_path):
    stages = runpy.run_path(str(Path(__file__).with_name("test_experiment_stages.py")))
    service, _development, record, training = stages["experiment"].__wrapped__(tmp_path)
    from histopilot.application.experiment_predictors import ExperimentPredictorService
    from histopilot.schemas.model_experiments import UpdateModelExperiment

    policies = [
        {"method": "skip", "refitPercentile": None},
        {"method": "refit", "refitPercentile": 90.0},
    ]
    plans = [
        {**row, "spec": {**row["spec"], "predictorPolicy": policy}}
        for row, policy in zip(record["batchPlans"], policies, strict=True)
    ]
    record = service.update(
        record["id"],
        UpdateModelExperiment(
            name=record["name"], expectedRevision=record["revision"], batchPlans=plans
        ),
    )

    class Coordinator:
        public = staticmethod(ExperimentPredictorService.public)

        def launch(self, identity, operation):
            submission = service.store.get_draft(identity)["payload"]["submission"]
            assert len(training.launches) == 2
            assert {tuple(row.values()) for row in submission["predictorPolicies"].values()} == {
                tuple(row.values()) for row in policies
            }
            assert "predictorPolicy" not in submission

        def status(self, identity, summary=False):
            return self.public({"status": "waiting", "items": []})

    service.predictor_execution = Coordinator()
    submitted = stages["submit"](service, record)
    assert submitted["submission"]["status"] == "submitted"
    by_name = {"First": policies[0], "Second": policies[1]}
    assert len(submitted["predictorPolicies"]) == 2
    for batch in submitted["batches"]:
        expected = by_name[batch["manifest"]["spec"]["batchName"]]
        assert submitted["predictorPolicies"][batch["id"]] == expected
        assert batch["manifest"]["spec"]["predictorPolicy"] == expected


def test_copy_legacy_global_policy_materializes_editable_batch_choices(tmp_path):
    stages = runpy.run_path(str(Path(__file__).with_name("test_experiment_stages.py")))
    service, _development, record, _training = stages["experiment"].__wrapped__(tmp_path)
    from histopilot.schemas.model_experiments import CreateModelExperiment

    stored = service.store.get_draft(record["id"])
    payload = copy.deepcopy(stored["payload"])
    payload["predictorPolicy"] = {"method": "refit", "refitPercentile": 75.0}
    for plan in payload["batchPlans"]:
        plan["spec"].pop("predictorPolicy", None)
        plan["spec"]["recipe"].pop("maxEpochs", None)
        plan["spec"]["recipe"].pop("patience", None)
    service.store.update_draft(
        record["id"], expected_revision=stored["revision"], name=stored["name"], payload=payload
    )
    copied = service.create(
        CreateModelExperiment(
            name="Legacy template",
            operationId="copy-legacy-policy",
            sourceExperimentId=record["id"],
        )
    )
    assert copied["stage"] == "planning" and copied["predictorExecution"] is None
    assert all(
        plan["spec"]["predictorPolicy"] == payload["predictorPolicy"]
        for plan in copied["batchPlans"]
    )
    assert all(
        plan["spec"]["recipe"]["maxEpochs"] == 100 and plan["spec"]["recipe"]["patience"] == 15
        for plan in copied["batchPlans"]
    )


@pytest.mark.parametrize("changed", [None, {}, {"method": "refit", "refitPercentile": None}])
def test_corrupt_or_missing_policy_map_remains_visible_and_blocks_build(mixed, changed):
    from histopilot.schemas.predictors import PredictorSelection

    service, identity, _jobs, _executor, batches = mixed
    record = service.store.get_draft(identity)
    payload = copy.deepcopy(record["payload"])
    payload["submission"]["predictorPolicyVersion"] = 2
    if changed is None:
        payload["submission"].pop("predictorPolicies")
    elif changed == {}:
        payload["submission"]["predictorPolicies"] = {}
    else:
        payload["submission"]["predictorPolicies"][batches[1]["id"]] = changed
    service.store.update_draft(
        identity, expected_revision=record["revision"], name=record["name"], payload=payload
    )
    models = ModelExperimentService(
        service.store, service.filesystem, training=service.training, predictor_execution=service
    )
    presented = models.get(identity)
    assert presented["stage"] == "running" and presented["configurationLocked"]
    assert presented["predictorPolicies"] is None
    assert presented["predictorExecution"]["error"]["code"] == "EXPERIMENT_PREDICTOR_PLAN_CHANGED"
    source = batches[1]
    selection = PredictorSelection(
        experimentId=identity,
        batchId=source["id"],
        candidateId=source["manifest"]["configurations"][0]["id"],
        trainingSeed=11,
        splitSeed=42,
        name="Invalid intent",
        method="ensemble",
    )
    preview = service.builds.predictors.preview(selection)
    assert not preview["canFreeze"]
    assert preview["findings"][0]["code"] == "EXPERIMENT_PREDICTOR_PLAN_CHANGED"
    with pytest.raises(StorageError):
        service.launch(identity, "invalid-start")


def bulk_selection(service, identity, batch):
    from histopilot.schemas.predictors import PredictorBuildSelection

    return PredictorBuildSelection(
        selections=[
            {
                "experimentId": identity,
                "batchId": batch["id"],
                "candidateId": batch["manifest"]["configurations"][0]["id"],
                "trainingSeed": 11,
                "splitSeed": 42,
            }
        ],
        method="ensemble",
    )


def test_bulk_creation_cannot_reopen_cancelled_predictor_work(mixed):
    from histopilot.schemas.predictors import ApplyPredictorBuilds

    service, identity, _jobs, _executor, batches = mixed
    selection = bulk_selection(service, identity, batches[1])
    preview = service.builds.preview(selection)
    assert preview["canBuild"]
    reviewed = ApplyPredictorBuilds(
        **selection.model_dump(),
        previewHash=preview["previewHash"],
        operationId="reviewed-before-cancel",
    )
    service.cancel(identity, "cancel-all")
    blocked = service.builds.preview(selection)
    assert not blocked["canBuild"]
    assert blocked["items"][0]["findings"][0]["code"] == "EXPERIMENT_PREDICTORS_LOCKED"
    with pytest.raises(StorageError) as error:
        service.builds.apply(reviewed)
    assert error.value.code == "PREVIEW_STALE"
    assert service.store.list_configurations("frozen-predictor") == []


def test_completed_bulk_receipt_remains_replayable_after_cancellation(mixed):
    from histopilot.schemas.predictors import ApplyPredictorBuilds

    service, identity, _jobs, _executor, batches = mixed
    selection = bulk_selection(service, identity, batches[1])
    preview = service.builds.preview(selection)
    request = ApplyPredictorBuilds(
        **selection.model_dump(),
        previewHash=preview["previewHash"],
        operationId="accepted-before-cancel",
    )
    accepted = service.builds.apply(request)
    assert accepted["status"] == "completed"
    service.cancel(identity, "cancel-all")
    assert service.builds.apply(request) == accepted
    assert len(service.store.list_configurations("frozen-predictor")) == 1


def test_valid_but_changed_map_cannot_override_frozen_batch_before_first_launch(mixed):
    service, identity, _jobs, executor, batches = mixed
    record = service.store.get_draft(identity)
    payload = copy.deepcopy(record["payload"])
    payload["submission"]["predictorPolicies"][batches[1]["id"]] = {
        "method": "both",
        "refitPercentile": 90.0,
    }
    service.store.update_draft(
        identity, expected_revision=record["revision"], name=record["name"], payload=payload
    )
    with pytest.raises(StorageError) as error:
        service.launch(identity, "changed-map")
    assert error.value.code == "EXPERIMENT_PREDICTOR_PLAN_CHANGED"
    assert not executor.launches
    preview = service.builds.preview(bulk_selection(service, identity, batches[1]))
    assert not preview["canBuild"]
    assert preview["items"][0]["findings"][0]["code"] == "EXPERIMENT_PREDICTOR_PLAN_CHANGED"


def test_mapped_policy_can_inherit_from_an_old_frozen_batch_without_policy():
    from histopilot.application.experiment_policy import verify_batch_policy

    policy = {"method": "refit", "refitPercentile": 75.0}
    assert verify_batch_policy(policy, {}) == policy
    assert verify_batch_policy(policy, {"predictorPolicy": None}) == policy
