"""Experimental Setup freezes scientific intent; explicit execution alone starts compute."""

import copy
import runpy
from functools import partial
from pathlib import Path

import h5py
import pytest
from support import t2

from histopilot.application.development import DevelopmentService
from histopilot.application.model_experiments import ModelExperimentService
from histopilot.application.target_splits import TargetSplitService
from histopilot.schemas.development import DevelopmentBatchSpec
from histopilot.schemas.model_experiments import (
    ConfigureModelExperimentSetup,
    CreateModelExperiment,
    FreezeModelExperimentSetup,
    SubmitModelExperiment,
    UpdateModelExperiment,
)
from histopilot.storage.filesystem import LocalFilesystem
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import ScientificStore

support = runpy.run_path(str(Path(__file__).with_name("test_evaluations.py")))
stages = runpy.run_path(str(Path(__file__).with_name("test_experiment_stages.py")))


class Training(stages["Training"]):
    def __init__(self, store):
        super().__init__(store)
        self.preflight_calls = 0

    def _prepare(self, batch):
        self.preflight_calls += 1
        return super()._prepare(batch)


def build_setup(tmp_path, slide_unit, bundle, make_training):
    store = ScientificStore(tmp_path, "project-setup")
    rows = [
        {
            "slideId": f"slide-{index}",
            "patientId": (None if index % 7 == 0 else "shared")
            if slide_unit
            else f"patient-{index}",
            "attributes": {"label": str(index % 2), "cohort": "source"},
        }
        for index in range(40)
    ]
    dataset, _ = support["dataset"](store, rows=rows)
    filesystem = LocalFilesystem((tmp_path,))
    targets = TargetSplitService(store, filesystem)
    draft = store.create_draft(
        "experiment",
        "Train and test",
        {
            "type": "target-split",
            "spec": {
                "datasetId": dataset["id"],
                "target": {**support["TARGET"], "unit": "slide"}
                if slide_unit
                else support["TARGET"],
                **({"splitUnit": "slide"} if slide_unit else {}),
                "split": {"method": "random", "testFraction": 0.2, "seed": 42},
            },
        },
    )
    reviewed = targets.preview(draft["id"], 1)
    assert reviewed["canFreeze"], reviewed["findings"]
    target_split = targets.freeze(draft["id"], 1, reviewed["previewHash"], "target-split")
    training_ids = [
        row["slideId"]
        for row in target_split["manifest"]["memberships"]
        if row["partition"] == "train"
    ]
    features, _, source = bundle(store, tmp_path, dataset, training_ids)
    training = make_training(store, filesystem)
    service = ModelExperimentService(store, filesystem, training=training)
    record = service.create(
        CreateModelExperiment(
            name="Pipeline experiment",
            operationId="create",
            setupVersion=1,
            predictorPolicy={"method": "skip", "refitPercentile": None},
        )
    )
    request = ConfigureModelExperimentSetup(
        expectedRevision=record["revision"],
        datasetId=dataset["id"],
        targetSplitId=target_split["id"],
        featureBundleId=features["id"],
        trainingSplit={
            "version": 4,
            "mode": "kfold",
            "folds": 2,
            "seeds": [42],
            "pools": {"trainSelection": "remaining"},
        },
    )
    return service, record, request, target_split, source, training


@pytest.fixture
def setup(tmp_path, request):
    slide_unit = getattr(request, "param", "legacy") == "slide"
    return build_setup(tmp_path, slide_unit, support["bundle"], lambda store, _: Training(store))


@pytest.fixture
def managed_setup(tmp_path, request, task_center, monkeypatch):
    """``setup`` with Task Center features and training; ``training.prepared`` spies preflight."""
    slide_unit = getattr(request, "param", "legacy") == "slide"
    return build_setup(
        tmp_path,
        slide_unit,
        partial(t2.bundle, task_center),
        partial(t2.training_service, center=task_center, monkeypatch=monkeypatch),
    )


def configure(setup):
    service, record, request, *_ = setup
    record = service.setup_inputs(record["id"], request)
    plan = DevelopmentBatchSpec(
        experimentName=record["name"],
        batchName="Baseline",
        inputs=record["inputs"],
        trainingSeeds=[42],
        predictorPolicy={"method": "skip", "refitPercentile": None},
    )
    return service.update(
        record["id"],
        UpdateModelExperiment(
            name=record["name"],
            expectedRevision=record["revision"],
            batchPlans=[{"id": "baseline", "spec": plan}],
        ),
    )


def freeze(service, record, operation="freeze"):
    return service.freeze_setup(
        record["id"],
        FreezeModelExperimentSetup(
            expectedRevision=record["revision"],
            operationId=operation,
        ),
    )


def submit(service, record):
    return service.submit(
        record["id"],
        SubmitModelExperiment(
            expectedRevision=record["revision"],
            operationId="execute",
        ),
    )


@pytest.mark.parametrize("managed_setup", ["legacy", "slide"], indirect=True)
def test_setup_selects_training_population_and_freezes_without_compute(
    managed_setup, task_center
):
    service, original, _request, target_split, _source, training = managed_setup
    with pytest.raises(StorageError) as pending:
        submit(service, original)
    assert pending.value.code == "EXPERIMENT_SETUP_REQUIRED"
    record = configure(managed_setup)
    protocol = service.store.get_configuration(record["inputs"]["protocolId"])
    expected_unit = target_split["manifest"]["spec"].get("splitUnit", "patient")
    assert record["setupDesign"]["splitUnit"] == expected_unit
    assert protocol["manifest"]["spec"].get("splitUnit", "patient") == expected_unit
    assert {row["slideId"] for row in protocol["manifest"]["memberships"]} == {
        row["slideId"]
        for row in target_split["manifest"]["memberships"]
        if row["partition"] == "train"
    }
    with pytest.raises(StorageError) as draft:
        submit(service, record)
    assert draft.value.code == "EXPERIMENT_SETUP_REQUIRED"
    frozen = freeze(service, record)
    assert frozen["setupVersion"] == 1 and frozen["setupStatus"] == "frozen"
    assert frozen["status"] == "ready" and frozen["stage"] == "planning"
    assert frozen["configurationLocked"] and frozen["submission"] is None
    assert frozen["frozenSetup"]["manifest"]["kind"] == "experiment-setup"
    assert frozen["frozenSetup"]["manifest"]["inputSnapshot"]["protocol"]["id"] == protocol["id"]
    assert len(service.store.list_configurations("experiment-setup")) == 1
    assert service.store.list_configurations("mil-batch") == []
    assert t2.preflights(training) == 0 and t2.launched(task_center) == []
    assert freeze(service, record) == frozen
    with pytest.raises(StorageError) as replay:
        freeze(service, frozen, "different-freeze")
    assert replay.value.code == "EXPERIMENT_SETUP_FROZEN"


def test_frozen_setup_blocks_science_edits_but_metadata_does_not_change_execution(
    managed_setup, task_center
):
    service, _, request, _, _, training = managed_setup
    frozen = freeze(service, configure(managed_setup))
    snapshot = copy.deepcopy(frozen["frozenSetup"])
    for changes in (
        {"inputs": None},
        {"batchPlans": []},
        {"predictorPolicy": {"method": "ensemble", "refitPercentile": None}},
    ):
        with pytest.raises(StorageError) as blocked:
            service.update(
                frozen["id"],
                UpdateModelExperiment(
                    name=frozen["name"], expectedRevision=frozen["revision"], **changes
                ),
            )
        assert blocked.value.code == "EXPERIMENT_CONFIGURATION_LOCKED"
    with pytest.raises(StorageError) as inputs:
        service.setup_inputs(
            frozen["id"], request.model_copy(update={"expectedRevision": frozen["revision"]})
        )
    assert inputs.value.code == "EXPERIMENT_CONFIGURATION_LOCKED"
    with pytest.raises(StorageError) as batch:
        DevelopmentService(service.store, service.filesystem).preview(
            DevelopmentBatchSpec.model_validate(frozen["batchPlans"][0]["spec"]).model_copy(
                update={"experimentRevision": frozen["revision"]}
            )
        )
    assert batch.value.code == "EXPERIMENT_CONFIGURATION_LOCKED"
    annotated = service.update(
        frozen["id"],
        UpdateModelExperiment(
            name="Display annotation", expectedRevision=frozen["revision"], notes="Ready to execute"
        ),
    )
    assert annotated["batchPlans"] == frozen["batchPlans"]
    assert annotated["frozenSetup"] == snapshot
    submitted = submit(service, annotated)
    assert submitted["stage"] == "running" and submitted["submission"]["status"] == "submitted"
    assert t2.launched(task_center) == submitted["submission"]["batchIds"]
    assert t2.preflights(training) == 1
    assert (
        submitted["batches"][0]["manifest"]["spec"] == snapshot["manifest"]["batchPlans"][0]["spec"]
    )
    assert submitted["batches"][0]["manifest"]["experiment"]["name"] == frozen["name"]
    assert submit(service, annotated)["id"] == annotated["id"]
    assert len(t2.launched(task_center)) == 1
    assert [task["attempt"] for task in task_center.tasks(kind="mil-fold")] == [1, 1]


def test_copy_of_frozen_setup_is_editable_and_requires_its_own_freeze(managed_setup):
    service, *_ = managed_setup
    frozen = freeze(service, configure(managed_setup))
    copied = service.create(
        CreateModelExperiment(name="New setup", operationId="copy", sourceExperimentId=frozen["id"])
    )
    assert copied["setupVersion"] == 1 and copied["setupDesign"] == frozen["setupDesign"]
    assert copied["frozenSetupId"] is None and copied["setupStatus"] == "draft"
    assert not copied["configurationLocked"] and copied["submission"] is None
    assert copied["batchPlans"][0]["spec"]["experimentName"] == "New setup"
    with pytest.raises(StorageError) as pending:
        submit(service, copied)
    assert pending.value.code == "EXPERIMENT_SETUP_REQUIRED"
    assert freeze(service, copied, "copy-freeze")["frozenSetupId"] != frozen["frozenSetupId"]


def test_setup_input_changes_require_typed_endpoint_and_matching_dataset(managed_setup):
    service, record, request, *_ = managed_setup
    with pytest.raises(StorageError) as mismatch:
        service.setup_inputs(
            record["id"], request.model_copy(update={"datasetId": "dataset-" + "f" * 64})
        )
    assert mismatch.value.code == "EXPERIMENT_DATASET_MISMATCH"
    with pytest.raises(StorageError) as bypass:
        service.update(
            record["id"],
            UpdateModelExperiment(
                name=record["name"],
                expectedRevision=record["revision"],
                inputs={"protocolId": "unbound", "featureBundleId": request.featureBundleId},
            ),
        )
    assert bypass.value.code == "EXPERIMENT_SETUP_INPUTS_REQUIRED"
    assert service.get(record["id"])["revision"] == record["revision"]


@pytest.mark.parametrize("phase", ["configure", "freeze", "submit"])
def test_changed_feature_files_block_each_setup_execution_boundary(
    managed_setup, task_center, phase
):
    service, record, request, _, source, training = managed_setup
    if phase != "configure":
        record = configure(managed_setup)
    if phase == "submit":
        record = freeze(service, record)
    path = next(source.glob("*.h5"))
    with h5py.File(path, "r+") as handle:
        handle["features"][0, 0] = 123
    with pytest.raises(StorageError) as changed:
        if phase == "configure":
            service.setup_inputs(record["id"], request)
        elif phase == "freeze":
            freeze(service, record)
        else:
            submit(service, record)
    assert changed.value.code in {"EXPERIMENT_INPUTS_INVALID", "BATCH_PREFLIGHT_BLOCKED"}
    assert service.get(record["id"])["revision"] == record["revision"]
    assert t2.launched(task_center) == [] and t2.preflights(training) == 0
    assert service.get(record["id"])["submission"] is None


def test_freeze_checks_revision_and_requires_saved_batches(managed_setup):
    service, record, request, *_ = managed_setup
    configured = service.setup_inputs(record["id"], request)
    with pytest.raises(StorageError) as stale:
        freeze(service, record)
    assert stale.value.code == "REVISION_CONFLICT"
    with pytest.raises(StorageError) as missing:
        freeze(service, configured)
    assert missing.value.code == "EXPERIMENT_BATCHES_REQUIRED"
    assert service.store.list_configurations("experiment-setup") == []


def test_freeze_recovers_publication_before_draft_receipt_without_duplicate(
    managed_setup, monkeypatch
):
    service, *_ = managed_setup
    record = configure(managed_setup)
    original = service.store.update_draft

    def lost_receipt(*args, **kwargs):
        if kwargs["payload"].get("frozenSetupId"):
            raise OSError("Lost setup receipt")
        return original(*args, **kwargs)

    monkeypatch.setattr(service.store, "update_draft", lost_receipt)
    with pytest.raises(OSError, match="Lost setup receipt"):
        freeze(service, record)
    assert len(service.store.list_configurations("experiment-setup")) == 1
    monkeypatch.setattr(service.store, "update_draft", original)
    recovered = freeze(service, record)
    assert recovered["setupStatus"] == "frozen"
    assert len(service.store.list_configurations("experiment-setup")) == 1


def test_freeze_remains_available_when_execution_runtime_is_unavailable(
    managed_setup, task_center
):
    service, _, _, _, _, training = managed_setup
    record = configure(managed_setup)
    training.runtime.unavailable_after = 0
    frozen = freeze(service, record)
    assert t2.preflights(training) == 0
    with pytest.raises(StorageError) as unavailable:
        submit(service, frozen)
    assert unavailable.value.code == "TRAINING_RUNTIME_UNAVAILABLE"
    current = service.get(record["id"])
    assert current["status"] == "ready" and current["submission"] is None
    assert current["frozenSetupId"] == frozen["frozenSetupId"]
    assert t2.launched(task_center) == []


def test_freeze_publication_rechecks_feature_files_after_review(
    managed_setup, task_center, monkeypatch
):
    service, _, _, _, source, training = managed_setup
    record = configure(managed_setup)
    publish = service.store.publish_configuration

    def changed_during_publication(*args, **kwargs):
        if kwargs["manifest"]["kind"] == "experiment-setup":
            with h5py.File(next(source.glob("*.h5")), "r+") as handle:
                handle["features"][0, 0] = 456
        return publish(*args, **kwargs)

    monkeypatch.setattr(service.store, "publish_configuration", changed_during_publication)
    with pytest.raises(StorageError) as stale:
        freeze(service, record)
    assert stale.value.code == "PREVIEW_STALE"
    assert service.store.list_configurations("experiment-setup") == []
    assert not service.get(record["id"])["configurationLocked"]
    assert t2.launched(task_center) == []


def test_failed_execution_launch_retries_same_frozen_setup(
    managed_setup, task_center, monkeypatch
):
    service, *_ = managed_setup
    frozen = freeze(service, configure(managed_setup))

    def offline(*_args, **_kwargs):
        raise StorageError("store offline", "TASK_CENTER_UNAVAILABLE", 503)

    # The Task Center is unreachable for the first launch only.
    with monkeypatch.context() as patch:
        patch.setattr(task_center.client, "enqueue", offline)
        interrupted = submit(service, frozen)
    assert interrupted["submission"]["status"] == "attention"
    assert interrupted["submission"]["error"]["code"] == "TRAINING_LAUNCH_FAILED"
    assert interrupted["configurationLocked"]
    assert len(service.store.list_configurations("mil-batch")) == 1
    assert t2.launched(task_center) == []
    recovered = submit(service, frozen)
    assert recovered["submission"]["status"] == "submitted"
    assert recovered["frozenSetupId"] == frozen["frozenSetupId"]
    assert t2.launched(task_center) == recovered["submission"]["batchIds"]
    assert len(service.store.list_configurations("mil-batch")) == 1


def test_setup_routes_expose_freezing_separately_from_execution(
    managed_setup, task_center, monkeypatch
):
    from types import SimpleNamespace

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import histopilot.api.model_experiments as api

    service, record, request, _, _, training = managed_setup
    monkeypatch.setattr(api, "ModelExperimentService", lambda *_: service)
    app = FastAPI()
    app.include_router(
        api.model_experiments_router(
            SimpleNamespace(scientific_store=lambda _: service.store), service.filesystem
        )
    )
    prefix = f"/api/v1/projects/project-setup/model-experiments/{record['id']}"
    with TestClient(app) as client:
        configured = client.post(prefix + "/setup-inputs", json=request.model_dump())
        assert configured.status_code == 200, configured.text
        body = configured.json()
        assert body["setupVersion"] == 1 and body["setupStatus"] == "draft"
        plan = DevelopmentBatchSpec(
            experimentName=body["name"],
            batchName="API batch",
            inputs=body["inputs"],
            predictorPolicy={"method": "skip", "refitPercentile": None},
        )
        updated = client.patch(
            prefix,
            json={
                "name": body["name"],
                "expectedRevision": body["revision"],
                "batchPlans": [{"id": "api-plan", "spec": plan.model_dump()}],
            },
        )
        assert updated.status_code == 200, updated.text
        frozen = client.post(
            prefix + "/freeze-setup",
            json={"expectedRevision": updated.json()["revision"], "operationId": "api-freeze"},
        )
        assert frozen.status_code == 201, frozen.text
        assert frozen.json()["status"] == "ready"
        assert t2.launched(task_center) == [] and t2.preflights(training) == 0
        execution = client.post(
            prefix + "/submit",
            json={"expectedRevision": frozen.json()["revision"], "operationId": "api-execute"},
        )
        assert execution.status_code == 202, execution.text
        assert execution.json()["submission"]["status"] == "submitted"
        assert len(t2.launched(task_center)) == 1 and t2.preflights(training) == 1


def test_frozen_setup_cannot_be_archived_apart_from_its_experiment(managed_setup):
    from histopilot.application.lifecycle import CleanupService
    from histopilot.schemas.lifecycle import CleanupSelection

    service, *_ = managed_setup
    frozen = freeze(service, configure(managed_setup))
    cleanup = CleanupService(service.store, service.filesystem, training=service.training)
    review = cleanup.preview(
        CleanupSelection(action="archive", keys=[f"configuration:{frozen['frozenSetupId']}"])
    )
    assert not review["canApply"]
    assert "EXPERIMENT_CONFIGURATION_LOCKED" in {item["code"] for item in review["blockers"]}


def test_generic_draft_api_cannot_bypass_frozen_setup_lock(managed_setup, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from histopilot.api import create_app
    from histopilot.config import Settings

    service, *_ = managed_setup
    frozen = freeze(service, configure(managed_setup))
    app = create_app(Settings(workspace=tmp_path / "api-workspace", data_roots=(tmp_path,)))
    monkeypatch.setattr(app.state.projects, "scientific_store", lambda _: service.store)
    with TestClient(app, base_url="http://127.0.0.1:8787") as client:
        client.headers["X-HistoPilot-Token"] = client.get("/api/v1/session").json()["token"]
        base = "/api/v1/projects/project-setup/drafts"
        overwritten = client.patch(
            base + "/" + frozen["id"],
            json={
                "name": frozen["name"],
                "expectedRevision": frozen["revision"],
                "payload": {"type": "analysis-protocol", "spec": {}},
            },
        )
        assert overwritten.status_code == 409, overwritten.text
        assert overwritten.json()["code"] == "EXPERIMENT_TYPED_ENDPOINT_REQUIRED"
        added = client.post(
            base,
            json={
                "name": "Bypass",
                "kind": "experiment",
                "payload": {
                    "type": "development-batch",
                    "experimentId": frozen["id"],
                    "spec": {
                        "experimentId": frozen["id"],
                        "experimentRevision": frozen["revision"],
                    },
                },
            },
        )
        assert added.status_code == 409, added.text
        assert added.json()["code"] == "EXPERIMENT_CONFIGURATION_LOCKED"
    assert service.get(frozen["id"])["frozenSetupId"] == frozen["frozenSetupId"]


@pytest.mark.parametrize(
    "mode", ["monte_carlo", "leave_one_domain_out", "nested_kfold", "held_out"]
)
def test_setup_rejects_unexecutable_training_design_before_derivation(managed_setup, mode):
    service, record, request, *_ = managed_setup
    before = service.store.list_configurations("protocol")
    split = request.trainingSplit.model_copy(
        update={"mode": mode, "domainField": "cohort" if mode == "leave_one_domain_out" else None}
    )
    with pytest.raises(StorageError) as unsupported:
        service.setup_inputs(record["id"], request.model_copy(update={"trainingSplit": split}))
    assert unsupported.value.code == "TRAINING_SPLIT_UNSUPPORTED"
    assert service.store.list_configurations("protocol") == before
    assert service.get(record["id"])["revision"] == record["revision"]
    assert service.get(record["id"])["setupDesign"] is None


def test_unlabeled_testing_partition_does_not_change_training_setup_or_launch_compute(
    managed_setup, task_center
):
    service, original, request, source_split, source, training = managed_setup
    targets = TargetSplitService(service.store, service.filesystem)
    draft = service.store.create_draft(
        "experiment",
        "Reserved slides for inference",
        {
            "type": "target-split",
            "spec": {**source_split["manifest"]["spec"], "testTarget": None},
        },
    )
    preview = targets.preview(draft["id"], 1)
    assert preview["canFreeze"], preview["findings"]
    unlabeled = targets.freeze(draft["id"], 1, preview["previewHash"], "inference-target-split")
    previous_roles = {
        row["slideId"]: row["partition"] for row in source_split["manifest"]["memberships"]
    }
    assert {
        row["slideId"]: row["partition"] for row in unlabeled["manifest"]["memberships"]
    } == previous_roles
    record = configure(
        (
            service,
            original,
            request.model_copy(update={"targetSplitId": unlabeled["id"]}),
            unlabeled,
            source,
            training,
        )
    )
    frozen = freeze(service, record)
    protocol = service.store.get_configuration(frozen["inputs"]["protocolId"])
    expected_training = {slide for slide, role in previous_roles.items() if role == "train"}
    assert {row["slideId"] for row in protocol["manifest"]["memberships"]} == expected_training
    assert all(row["label"] is not None for row in protocol["manifest"]["memberships"])
    cohort = service.store.get_configuration(unlabeled["evaluationCohortId"])
    assert cohort["manifest"]["spec"]["purpose"] == "inference"
    assert cohort["manifest"]["target"] is None
    assert {row["slideId"] for row in cohort["manifest"]["memberships"]} == {
        slide for slide, role in previous_roles.items() if role == "test"
    }
    assert all(row["label"] is None for row in cohort["manifest"]["memberships"])
    assert frozen["setupStatus"] == "frozen" and frozen["submission"] is None
    assert t2.preflights(training) == 0 and t2.launched(task_center) == []
