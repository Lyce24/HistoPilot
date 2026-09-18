"""An experiment submission locks one complete plan across retries and recovery."""

import copy
import runpy
from pathlib import Path

import pytest

from histopilot.application.lifecycle import CleanupService
from histopilot.application.model_experiments import ModelExperimentService
from histopilot.schemas.development import DevelopmentBatchSpec
from histopilot.schemas.lifecycle import CleanupSelection
from histopilot.schemas.model_experiments import (
    CreateModelExperiment,
    SubmitModelExperiment,
    UpdateModelExperiment,
)
from histopilot.storage.project_lock import StorageError

support = runpy.run_path(str(Path(__file__).with_name("test_development_batches.py")))


class Training:
    def __init__(self, store):
        self.store = store
        self.states, self.operations, self.launches = {}, {}, []
        self.fail_preflight = None
        self.fail_launch = None

    def execution(self, identity, **_kwargs):
        return copy.deepcopy(self.states.get(identity))

    def _prepare(self, batch):
        if batch["manifest"]["spec"]["batchName"] == self.fail_preflight:
            raise StorageError("Synthetic runtime failure", "TRAINING_RUNTIME_UNAVAILABLE")
        return {}, lambda: None

    def launch(self, identity, operation):
        owner = self.store.get_configuration(identity)["manifest"]["spec"]["experimentId"]
        assert self.store.get_draft(owner)["payload"]["submission"]["batchIds"]
        if operation in self.operations:
            return self.execution(identity)
        if len(self.launches) == self.fail_launch:
            raise StorageError("Synthetic launch failure", "TRAINING_LAUNCH_FAILED")
        self.operations[operation] = identity
        self.launches.append(identity)
        self.states[identity] = {"status": "running", "runs": [], "cancelRequested": False}
        return self.execution(identity)


@pytest.fixture
def experiment(tmp_path):
    development, spec, _ = support["batch"].__wrapped__(tmp_path)
    training = Training(development.store)
    service = ModelExperimentService(development.store, development.filesystem, training=training)
    record = service.create(
        CreateModelExperiment(
            name="Study",
            operationId="create",
            inputs=spec.inputs,
            predictorPolicy={"method": "skip", "refitPercentile": None},
        )
    )
    values = spec.model_dump()
    values["batchName"] = "First"
    second = {**values, "batchName": "Second"}
    second["recipe"] = {**values["recipe"], "learningRate": 0.002}
    record = service.update(
        record["id"],
        UpdateModelExperiment(
            name="Study",
            expectedRevision=1,
            batchPlans=[{"id": "first", "spec": values}, {"id": "second", "spec": second}],
        ),
    )
    return service, development, record, training


def submit(service, record, operation="submit"):
    return service.submit(
        record["id"],
        SubmitModelExperiment(expectedRevision=record["revision"], operationId=operation),
    )


def test_plans_edit_in_planning_without_publishing_and_inherit_current_inputs(experiment):
    service, _, record, _ = experiment
    assert record["stage"] == "planning" and not record["configurationLocked"]
    assert record["batches"] == []
    assert len(record["batchPlans"]) == 2
    updated = service.update(
        record["id"],
        UpdateModelExperiment(
            name="Renamed", expectedRevision=record["revision"], batchPlans=record["batchPlans"][:1]
        ),
    )
    assert updated["batchPlans"][0]["spec"]["experimentName"] == "Renamed"
    assert updated["batchPlans"][0]["spec"]["experimentRevision"] == updated["revision"]
    assert updated["batchPlans"][0]["spec"]["inputs"] == updated["inputs"]
    assert len(updated["batchPlans"]) == 1 and updated["batches"] == []


def test_all_preflight_checks_finish_before_any_configuration_is_locked(experiment):
    service, _, record, training = experiment
    training.fail_preflight = "Second"
    with pytest.raises(StorageError) as error:
        submit(service, record)
    assert error.value.code == "TRAINING_RUNTIME_UNAVAILABLE"
    current = service.get(record["id"])
    assert current["stage"] == "planning" and current["submission"] is None
    assert current["batches"] == [] and training.launches == []
    assert current["revision"] == record["revision"]


def test_submission_locks_inputs_batches_and_receipt_replay_never_duplicates(experiment):
    service, development, record, training = experiment
    submitted = submit(service, record)
    assert submitted["stage"] == "running" and submitted["configurationLocked"]
    assert submitted["submission"]["status"] == "submitted"
    assert len(submitted["batches"]) == len(training.launches) == 2
    assert not submitted["submission"]["retryable"]
    assert "publications" not in submitted["submission"]
    assert submit(service, record)["id"] == record["id"]
    assert len(training.launches) == 2
    for field in ({"inputs": None}, {"batchPlans": []}):
        with pytest.raises(StorageError) as error:
            service.update(
                record["id"],
                UpdateModelExperiment(
                    name="Study", expectedRevision=submitted["revision"], **field
                ),
            )
        assert error.value.code == "EXPERIMENT_CONFIGURATION_LOCKED"
    with pytest.raises(StorageError) as error:
        development.preview(
            DevelopmentBatchSpec.model_validate(submitted["batchPlans"][0]["spec"]).model_copy(
                update={"experimentRevision": submitted["revision"]}
            )
        )
    assert error.value.code == "EXPERIMENT_CONFIGURATION_LOCKED"
    metadata = service.update(
        record["id"],
        UpdateModelExperiment(
            name="Annotated",
            expectedRevision=submitted["revision"],
            notes="Training note",
            tags=["reviewed"],
        ),
    )
    assert metadata["notes"] == "Training note" and metadata["stage"] == "running"
    assert metadata["batchPlans"] == submitted["batchPlans"]
    with pytest.raises(StorageError) as conflict:
        submit(service, metadata, "different-submit")
    assert conflict.value.code == "EXPERIMENT_ALREADY_SUBMITTED"


def test_partial_launch_retry_preserves_batch_ids_and_first_live_job(experiment):
    service, _, record, training = experiment
    training.fail_launch = 1
    partial = submit(service, record)
    assert partial["stage"] == "running" and partial["configurationLocked"]
    assert partial["submission"]["status"] == "attention" and partial["submission"]["retryable"]
    assert partial["submission"]["error"]["code"] == "TRAINING_LAUNCH_FAILED"
    assert len(training.launches) == 1
    training.fail_launch = None
    recovered = submit(service, record)
    assert recovered["submission"]["batchIds"] == partial["submission"]["batchIds"]
    assert recovered["submission"]["status"] == "submitted"
    assert len(training.launches) == 2


def test_lost_reply_after_worker_acceptance_reuses_durable_operation(experiment, monkeypatch):
    service, _, record, training = experiment
    original = training.launch
    lost = False

    def launch(identity, operation):
        nonlocal lost
        result = original(identity, operation)
        if not lost:
            lost = True
            raise OSError("Lost local acknowledgement")
        return result

    monkeypatch.setattr(training, "launch", launch)
    partial = submit(service, record)
    assert partial["submission"]["status"] == "attention"
    final = submit(service, record)
    assert final["submission"]["status"] == "submitted"
    assert len(training.launches) == len(set(training.launches)) == 2


def test_completed_runs_keep_incomplete_submission_recoverable(experiment, monkeypatch):
    service, _, record, training = experiment
    original = training.launch

    def finish_before_reply(identity, operation):
        result = original(identity, operation)
        if len(training.launches) == 2:
            for execution in training.states.values():
                execution["status"] = "completed"
            raise OSError("Final launch was accepted, but its response was lost")
        return result

    monkeypatch.setattr(training, "launch", finish_before_reply)
    partial = submit(service, record)
    assert partial["submission"]["status"] == "attention"
    assert partial["stage"] == "running"
    assert partial["status"] == "failed"
    assert partial["submission"]["retryable"]
    recovered = submit(service, record)
    assert recovered["submission"]["status"] == "submitted"
    assert recovered["stage"] == "finished"
    assert not recovered["submission"]["retryable"]
    assert len(training.launches) == len(set(training.launches)) == 2


def test_partial_publication_retry_uses_saved_owner_after_metadata_edit(experiment, monkeypatch):
    service, _, record, training = experiment
    original = service.store.publish_configuration
    failed = False

    def publish(**kwargs):
        nonlocal failed
        result = original(**kwargs)
        if not failed:
            failed = True
            raise OSError("Lost publication reply")
        return result

    monkeypatch.setattr(service.store, "publish_configuration", publish)
    partial = submit(service, record)
    assert partial["configurationLocked"] and training.launches == []
    changed = service.update(
        record["id"],
        UpdateModelExperiment(
            name="Annotated after submit",
            notes="Keep original frozen note",
            expectedRevision=partial["revision"],
        ),
    )
    final = submit(service, changed)
    assert final["submission"]["status"] == "submitted"
    assert len(final["batches"]) == len(training.launches) == 2
    assert all(batch["manifest"]["experiment"]["name"] == "Study" for batch in final["batches"])


def test_finished_stage_blocks_resume_and_copy_reopens_only_recipes(experiment):
    service, _, record, training = experiment
    submitted = submit(service, record)
    for index, batch in enumerate(submitted["batches"]):
        training.states[batch["id"]]["status"] = "completed" if index == 0 else "cancelled"
    finished = service.get(record["id"])
    assert finished["stage"] == "finished" and finished["configurationLocked"]
    with pytest.raises(StorageError) as error:
        service.require_training_action(record["id"], submitted["batches"][1]["id"], resume=True)
    assert error.value.code == "EXPERIMENT_FINISHED"
    command = CreateModelExperiment(
        name="Copy", operationId="copy", sourceExperimentId=record["id"]
    )
    copied = service.create(command)
    assert copied["stage"] == "planning" and not copied["configurationLocked"]
    assert copied["revision"] == 1
    assert copied["submission"] is None and copied["batches"] == []
    assert copied["inputs"] == record["inputs"]
    assert len(copied["batchPlans"]) == 2
    assert service.create(command)["id"] == copied["id"]
    assert copied["id"] != record["id"]
    assert len(training.launches) == 2


def test_failed_interrupted_unknown_remain_recoverable_running(experiment):
    service, _, record, training = experiment
    submitted = submit(service, record)
    for status in ("failed", "interrupted", "unknown"):
        for batch in submitted["batches"]:
            training.states[batch["id"]]["status"] = status
        assert service.get(record["id"])["stage"] == "running"
        service.require_training_action(record["id"], submitted["batches"][0]["id"], resume=True)


def test_submitted_batch_cannot_be_hidden_to_reopen_experiment(experiment):
    service, _, record, training = experiment
    submitted = submit(service, record)
    for batch in submitted["batches"]:
        training.states[batch["id"]]["status"] = "completed"
    cleanup = CleanupService(service.store, service.filesystem, training=training)
    for action in ("archive", "trash", "restore"):
        preview = cleanup.preview(
            CleanupSelection(action=action, keys=[submitted["batches"][0]["key"]])
        )
        assert not preview["canApply"]
        assert any(
            item["code"] == "EXPERIMENT_CONFIGURATION_LOCKED" for item in preview["blockers"]
        )
    preview = cleanup.preview(
        CleanupSelection(
            action="archive",
            keys=[submitted["key"], *[batch["key"] for batch in submitted["batches"]]],
        )
    )
    assert preview["canApply"]


def test_copy_archived_and_reject_trashed_source_without_partial_record(experiment):
    service, _, record, _ = experiment
    lifecycle = service.store.lifecycle
    lifecycle.apply(
        {record["key"]: "archived"},
        operation_id="archive",
        request_hash="a" * 64,
        expected_revision=lifecycle.read()["revision"],
    )
    copied = service.create(
        CreateModelExperiment(
            name="From archived", operationId="copy", sourceExperimentId=record["id"]
        )
    )
    assert len(copied["batchPlans"]) == 2
    lifecycle.apply(
        {record["key"]: "trashed"},
        operation_id="trash",
        request_hash="b" * 64,
        expected_revision=lifecycle.read()["revision"],
    )
    before = len(service.list()["items"])
    with pytest.raises(StorageError) as error:
        service.create(
            CreateModelExperiment(
                name="No", operationId="copy-trash", sourceExperimentId=record["id"]
            )
        )
    assert error.value.code == "RECORD_TRASHED"
    assert len(service.list()["items"]) == before


def test_duplicate_batch_plan_rejected_while_configuration_remains_editable(experiment):
    service, _, record, _ = experiment
    first = record["batchPlans"][0]
    with pytest.raises(StorageError) as error:
        service.update(
            record["id"],
            UpdateModelExperiment(
                name=record["name"],
                expectedRevision=record["revision"],
                batchPlans=[first, {**first, "id": "same-content"}],
            ),
        )
    assert error.value.code == "EXPERIMENT_DUPLICATE_BATCH"
    assert service.get(record["id"])["stage"] == "planning"
    assert service.get(record["id"])["revision"] == record["revision"]


def test_historical_execution_keeps_configuration_locked_even_when_hidden(experiment):
    service, development, record, training = experiment
    spec = DevelopmentBatchSpec.model_validate(record["batchPlans"][0]["spec"])
    preview = development.preview(spec)
    batch = development.freeze(spec, preview["previewHash"], "legacy-batch", {"tag": "Old batch"})
    training.states[batch["id"]] = {"status": "failed", "runs": [], "cancelRequested": False}
    service.store.lifecycle.apply(
        {f"configuration:{batch['id']}": "archived"},
        operation_id="old-archive",
        request_hash="c" * 64,
        expected_revision=service.store.lifecycle.read()["revision"],
    )
    retained = service.get(record["id"])
    assert retained["submission"] is None
    assert retained["configurationLocked"] and retained["stage"] == "running"
    with pytest.raises(StorageError) as error:
        service.update(
            record["id"],
            UpdateModelExperiment(
                name=record["name"], expectedRevision=record["revision"], inputs=None
            ),
        )
    assert error.value.code == "EXPERIMENT_CONFIGURATION_LOCKED"


def test_copy_archived_inputs_remains_an_editable_snapshot(experiment):
    service, _, record, _ = experiment
    bundle_key = "configuration:" + record["inputs"]["featureBundleId"]
    service.store.lifecycle.apply(
        {record["key"]: "archived", bundle_key: "archived"},
        operation_id="archive-input",
        request_hash="d" * 64,
        expected_revision=service.store.lifecycle.read()["revision"],
    )
    copied = service.create(
        CreateModelExperiment(
            name="Copy archived", operationId="copy-archived-input", sourceExperimentId=record["id"]
        )
    )
    assert copied["stage"] == "planning" and copied["inputs"] == record["inputs"]
    assert len(copied["batchPlans"]) == 2


def test_old_frozen_batch_with_other_inputs_must_be_resolved_before_submission(experiment):
    service, development, record, training = experiment
    spec = DevelopmentBatchSpec.model_validate(record["batchPlans"][0]["spec"])
    preview = development.preview(spec)
    batch = development.freeze(
        spec, preview["previewHash"], "older-inputs", {"tag": "Before input change"}
    )
    changed_inputs = {**record["inputs"], "loadingPolicy": "native", "packArtifactId": None}
    assert changed_inputs != record["inputs"]
    updated = service.update(
        record["id"],
        UpdateModelExperiment(
            name=record["name"], expectedRevision=record["revision"], inputs=changed_inputs
        ),
    )
    with pytest.raises(StorageError) as error:
        submit(service, updated)
    assert error.value.code == "EXPERIMENT_BATCH_INPUTS_MISMATCH"
    assert training.launches == []
    current = service.get(record["id"])
    assert current["stage"] == "planning" and current["submission"] is None
    assert service.store.get_configuration(batch["id"])["manifest"] == batch["manifest"]


def test_preflight_environment_drift_does_not_lock_or_publish(experiment, monkeypatch):
    service, _, record, training = experiment

    def changing_prepare(batch):
        return {
            "code": {"sha256": batch["manifest"]["spec"]["batchName"]},
            "runtime": {"versions": {"torch": "same"}},
        }, lambda: None

    monkeypatch.setattr(training, "_prepare", changing_prepare)
    with pytest.raises(StorageError) as error:
        submit(service, record)
    assert error.value.code == "EXPERIMENT_RUNTIME_CHANGED"
    current = service.get(record["id"])
    assert current["stage"] == "planning" and not current["configurationLocked"]
    assert current["batches"] == [] and training.launches == []


def test_copy_merges_equivalent_saved_frozen_and_legacy_draft_recipes(experiment):
    service, development, record, training = experiment
    spec = DevelopmentBatchSpec.model_validate(record["batchPlans"][0]["spec"])
    preview = development.preview(spec)
    frozen = development.freeze(
        spec, preview["previewHash"], "saved-also-frozen", {"tag": "Same retained recipe"}
    )
    service.store.create_draft(
        "experiment",
        "Older saved form",
        {"type": "development-batch", "experimentId": record["id"], "spec": spec.model_dump()},
    )
    command = CreateModelExperiment(
        name="Copied settings", operationId="copy-equivalent", sourceExperimentId=record["id"]
    )
    copied = service.create(command)
    assert copied["stage"] == "planning" and not copied["configurationLocked"]
    assert copied["revision"] == 1
    assert len(copied["batchPlans"]) == 2
    assert copied["batches"] == [] and copied["drafts"] == []
    assert service.create(command)["id"] == copied["id"]
    assert service.store.get_configuration(frozen["id"])["manifest"] == frozen["manifest"]
    assert training.launches == []


@pytest.mark.parametrize("recover_existing_receipt", [False, True])
def test_submission_reuses_an_identical_frozen_batch_and_its_existing_label(
    experiment, monkeypatch, recover_existing_receipt
):
    service, development, record, training = experiment
    spec = DevelopmentBatchSpec.model_validate(record["batchPlans"][0]["spec"])
    preview = development.preview(spec)
    existing = development.freeze(
        spec, preview["previewHash"], "already-frozen", {"tag": "Reviewed baseline"}
    )
    if recover_existing_receipt:
        # Reproduce the old publication failure after its irreversible receipt
        # was already saved, then recover using the original submission identity.
        with monkeypatch.context() as old_behavior:
            old_behavior.setattr(
                "histopilot.application.model_experiments.matching_published_batch",
                lambda *_: None,
            )
            partial = submit(service, record)
        assert partial["submission"]["status"] == "attention"
        assert partial["submission"]["error"]["code"] == "VERSION_LABEL_MISMATCH"
        assert partial["configurationLocked"] and not training.launches
        service.update(
            record["id"],
            UpdateModelExperiment(
                name="Annotated while recovering", expectedRevision=partial["revision"]
            ),
        )
    submitted = submit(service, record)
    assert submitted["submission"]["status"] == "submitted", submitted["submission"]
    assert set(submitted["submission"]["batchIds"]) == set(training.launches)
    assert len(training.launches) == len(set(training.launches)) == 2
    assert existing["id"] in training.launches
    assert (
        service.store.get_configuration(existing["id"])["versionLabel"] == existing["versionLabel"]
    )
    assert submit(service, record)["submission"]["status"] == "submitted"
    assert len(training.launches) == 2


@pytest.mark.parametrize("state", ["archived", "trashed"])
def test_identical_inactive_batch_blocks_before_submission_locks(experiment, state):
    service, development, record, training = experiment
    spec = DevelopmentBatchSpec.model_validate(record["batchPlans"][0]["spec"])
    preview = development.preview(spec)
    existing = development.freeze(
        spec, preview["previewHash"], "inactive-batch", {"tag": "Earlier batch"}
    )
    lifecycle = service.store.lifecycle
    lifecycle.apply(
        {f"configuration:{existing['id']}": state},
        operation_id="hide-batch",
        request_hash="d" * 64,
        expected_revision=lifecycle.read()["revision"],
    )
    with pytest.raises(StorageError) as error:
        submit(service, record)
    assert error.value.code == "EXPERIMENT_BATCH_INACTIVE"
    assert "First" in str(error.value) and state in str(error.value)
    current = service.get(record["id"])
    assert current["stage"] == "planning"
    assert current["submission"] is None
    assert current["revision"] == record["revision"]
    assert not training.launches


def test_batch_predictor_policy_freezes_with_submission_and_copy_reopens_it(experiment):
    from histopilot.application.experiment_predictors import ExperimentPredictorService

    service, _, record, training = experiment
    record = service.update(
        record["id"],
        UpdateModelExperiment(
            name=record["name"],
            expectedRevision=record["revision"],
            batchPlans=[
                {
                    **row,
                    "spec": {
                        **row["spec"],
                        "predictorPolicy": {"method": "both", "refitPercentile": 75},
                    },
                }
                for row in record["batchPlans"]
            ],
        ),
    )

    class Predictors:
        public = staticmethod(ExperimentPredictorService.public)
        launches = []

        def launch(self, identity, operation):
            self.launches.append((identity, operation))

        def status(self, identity, summary=False):
            return self.public({"status": "waiting", "items": []})

    coordinator = Predictors()
    service.predictor_execution = coordinator
    submitted = submit(service, record)
    policy = {"method": "both", "refitPercentile": 75.0}
    assert submitted["predictorPolicy"] is None
    assert list(submitted["predictorPolicies"].values()) == [policy, policy]
    assert len(coordinator.launches) == 1
    for changed in ({"method": "skip"}, {"method": "both", "refitPercentile": 50}):
        with pytest.raises(StorageError) as error:
            service.update(
                record["id"],
                UpdateModelExperiment(
                    name=record["name"],
                    expectedRevision=submitted["revision"],
                    batchPlans=[
                        {**row, "spec": {**row["spec"], "predictorPolicy": changed}}
                        for row in submitted["batchPlans"]
                    ],
                ),
            )
        assert error.value.code == "EXPERIMENT_CONFIGURATION_LOCKED"
    copied = service.create(
        CreateModelExperiment(
            name="New comparison",
            operationId="copy-predictors",
            sourceExperimentId=record["id"],
        )
    )
    assert copied["stage"] == "planning"
    assert [row["spec"]["predictorPolicy"] for row in copied["batchPlans"]] == [policy, policy]
    updated = service.update(
        copied["id"],
        UpdateModelExperiment(
            name=copied["name"],
            expectedRevision=copied["revision"],
            batchPlans=[
                {**row, "spec": {**row["spec"], "predictorPolicy": {"method": "skip"}}}
                for row in copied["batchPlans"]
            ],
        ),
    )
    assert all(row["spec"]["predictorPolicy"]["method"] == "skip" for row in updated["batchPlans"])
    assert updated["predictorExecution"] is None and len(training.launches) == 2
