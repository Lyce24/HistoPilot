"""Stable experiments own snapshots, survive cleanup and keep historical records readable."""

import copy
import runpy
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from histopilot.api import create_app
from histopilot.application.lifecycle import CleanupService
from histopilot.application.model_experiments import ModelExperimentService, aggregate_status
from histopilot.config import Settings
from histopilot.schemas.development import DevelopmentBatchSpec
from histopilot.schemas.lifecycle import ApplyCleanup, CleanupSelection
from histopilot.schemas.model_experiments import CreateModelExperiment, UpdateModelExperiment
from histopilot.storage.filesystem import LocalFilesystem
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import ScientificStore

support = runpy.run_path(str(Path(__file__).with_name("test_development_batches.py")))


class Training:
    def __init__(self):
        self.states = {}

    def execution(self, identity, **_kwargs):
        return copy.deepcopy(self.states.get(identity))


@pytest.fixture
def service(tmp_path):
    folder = tmp_path / "project"
    folder.mkdir()
    return ModelExperimentService(
        ScientificStore(folder, "project-experiments"),
        LocalFilesystem((tmp_path,)),
        training=Training(),
    )


@pytest.fixture
def prepared(tmp_path):
    development, spec, source = support["batch"].__wrapped__(tmp_path)
    service = ModelExperimentService(development.store, development.filesystem, training=Training())
    experiment = service.create(
        CreateModelExperiment(
            name=spec.experimentName, operationId="experiment", inputs=spec.inputs
        )
    )
    spec = DevelopmentBatchSpec.model_validate(
        {**spec.model_dump(), "experimentId": experiment["id"], "experimentRevision": 1}
    )
    return service, development, spec, experiment, source


def cleanup(service, action, *keys):
    lifecycle = CleanupService(service.store, service.filesystem, training=service.training)
    selection = CleanupSelection(action=action, keys=list(keys))
    preview = lifecycle.preview(selection)
    assert preview["canApply"], preview
    return lifecycle.apply(
        ApplyCleanup(
            **selection.model_dump(), previewHash=preview["previewHash"], operationId=uuid4().hex
        )
    )


def test_create_before_inputs_and_lost_response_replay_keep_one_stable_record(service):
    command = CreateModelExperiment(
        name="  First experiment  ", operationId="create", tags=["abmil"]
    )
    record = service.create(command)
    assert record["name"] == "First experiment"
    assert record["status"] == "created"
    assert record["inputs"] is None and record["inputSnapshot"] is None
    assert record["batches"] == [] and record["drafts"] == []
    assert record["key"] == f"draft:{record['id']}"
    assert service.create(command) == record
    assert len(service.list()["items"]) == 1
    with pytest.raises(StorageError, match="another experiment") as conflict:
        service.create(command.model_copy(update={"name": "Different"}))
    assert conflict.value.code == "OPERATION_CONFLICT"
    updated = service.update(
        record["id"],
        UpdateModelExperiment(name="Renamed", expectedRevision=1, notes="Results soon"),
    )
    assert updated["revision"] == 2
    assert service.create(command) == updated
    assert len(service.list()["items"]) == 1


def test_revision_conflicts_archive_trash_and_restore_do_not_destroy_history(service):
    record = service.create(CreateModelExperiment(name="Keep evidence", operationId="create"))
    service.update(record["id"], UpdateModelExperiment(name="Changed", expectedRevision=1))
    with pytest.raises(StorageError) as conflict:
        service.update(record["id"], UpdateModelExperiment(name="Lost update", expectedRevision=1))
    assert conflict.value.code == "REVISION_CONFLICT"
    cleanup(service, "archive", record["key"])
    assert service.list(state="active")["items"] == []
    assert service.get(record["id"])["state"] == "archived"
    with pytest.raises(StorageError) as archived:
        service.update(record["id"], UpdateModelExperiment(name="No", expectedRevision=2))
    assert archived.value.code == "EXPERIMENT_ARCHIVED"
    cleanup(service, "trash", record["key"])
    assert service.get(record["id"])["state"] == "trashed"
    with pytest.raises(StorageError) as trashed:
        service.update(record["id"], UpdateModelExperiment(name="No", expectedRevision=2))
    assert trashed.value.code == "RECORD_TRASHED"
    cleanup(service, "restore", record["key"])
    restored = service.get(record["id"])
    assert restored["id"] == record["id"] and restored["revision"] == 2
    assert restored["state"] == "active" and restored["name"] == "Changed"


def test_freeze_pins_owner_revision_and_exact_configuration_input_snapshots(prepared):
    service, development, spec, experiment, _source = prepared
    preview = development.preview(spec)
    assert preview["canFreeze"], preview["findings"]
    snapshot = preview["inputSnapshot"]
    assert snapshot["protocol"]["id"] == spec.inputs.protocolId
    assert snapshot["protocol"]["spec"]["split"]["version"] == 4
    assert snapshot["featureBundle"]["id"] == spec.inputs.featureBundleId
    assert snapshot["configurations"] == preview["configurations"]
    frozen = development.freeze(spec, preview["previewHash"], "batch", {"tag": "Batch"})
    assert frozen["manifest"]["experiment"]["id"] == experiment["id"]
    assert frozen["manifest"]["inputSnapshot"] == snapshot
    service.update(
        experiment["id"],
        UpdateModelExperiment(name="Later name", expectedRevision=1, tags=["reviewed"]),
    )
    # A publication retry returns its original snapshot even after the owner changes.
    assert development.freeze(spec, preview["previewHash"], "batch", {"tag": "Batch"}) == frozen
    assert service.store.get_configuration(frozen["id"])["manifest"]["experiment"]["revision"] == 1
    presented = service.get(experiment["id"])
    assert presented["name"] == "Later name" and presented["inputs"] == spec.inputs.model_dump()
    assert presented["batches"][0]["manifest"] == frozen["manifest"]
    assert frozen["manifest"]["executionImplemented"] is False
    assert presented["executionImplemented"] is True
    assert presented["batches"][0]["inputSnapshot"] == snapshot
    assert presented["status"] == "planned"


def test_metadata_updates_preserve_omitted_inputs_and_allow_explicit_clears(prepared):
    service, _development, spec, experiment, _source = prepared
    updated = service.update(
        experiment["id"],
        UpdateModelExperiment(
            name="Renamed", expectedRevision=1, notes="Keep this note", tags=["reviewed"]
        ),
    )
    assert updated["inputs"] == spec.inputs.model_dump()
    updated = service.update(
        experiment["id"], UpdateModelExperiment(name="Renamed again", expectedRevision=2)
    )
    assert updated["inputs"] == spec.inputs.model_dump()
    assert updated["notes"] == "Keep this note" and updated["tags"] == ["reviewed"]
    updated = service.update(
        experiment["id"],
        UpdateModelExperiment(name="Cleared", expectedRevision=3, inputs=None, notes="", tags=[]),
    )
    assert updated["inputs"] is None
    assert updated["notes"] == "" and updated["tags"] == []


def test_changed_or_archived_experiment_invalidates_new_batch_review(prepared):
    service, development, spec, experiment, _source = prepared
    preview = development.preview(spec)
    service.update(
        experiment["id"], UpdateModelExperiment(name=experiment["name"], expectedRevision=1)
    )
    with pytest.raises(StorageError) as changed:
        development.freeze(spec, preview["previewHash"], "stale", {"tag": "Stale"})
    assert changed.value.code == "REVISION_CONFLICT"
    assert service.store.list_configurations("mil-batch") == []
    cleanup(service, "archive", experiment["key"])
    with pytest.raises(StorageError) as archived:
        development.preview(spec.model_copy(update={"experimentRevision": 2}))
    assert archived.value.code == "EXPERIMENT_ARCHIVED"


def test_experiment_deletion_requires_batches_and_saved_plans_and_restore_parents(prepared):
    service, development, spec, experiment, _source = prepared
    preview = development.preview(spec)
    frozen = development.freeze(spec, preview["previewHash"], "batch", {"tag": "Batch"})
    draft = service.store.create_draft(
        "experiment",
        "Next batch",
        {"type": "development-batch", "experimentId": experiment["id"], "spec": spec.model_dump()},
    )
    lifecycle = CleanupService(service.store, service.filesystem, training=service.training)
    review = lifecycle.preview(CleanupSelection(action="trash", keys=[experiment["key"]]))
    assert not review["canApply"]
    keys = {f"configuration:{frozen['id']}", f"draft:{draft['id']}"}
    assert set(review["requiredKeys"]) == keys
    cleanup(service, "trash", experiment["key"], *keys)
    record = service.get(experiment["id"])
    assert record["state"] == "trashed"
    assert record["batches"][0]["state"] == "trashed"
    review = lifecycle.preview(
        CleanupSelection(action="restore", keys=[f"configuration:{frozen['id']}"])
    )
    assert not review["canApply"] and experiment["key"] in review["requiredKeys"]
    cleanup(service, "restore", experiment["key"], *keys)
    assert service.get(experiment["id"])["batches"][0]["id"] == frozen["id"]


def test_running_batch_protects_experiment_upstream(prepared):
    service, development, spec, experiment, _source = prepared
    preview = development.preview(spec)
    frozen = development.freeze(spec, preview["previewHash"], "batch", {"tag": "Batch"})
    service.training.states[frozen["id"]] = {
        "status": "running",
        "runs": [],
        "cancelRequested": False,
    }
    assert service.get(experiment["id"])["status"] == "running"
    lifecycle = CleanupService(service.store, service.filesystem, training=service.training)
    review = lifecycle.preview(CleanupSelection(action="archive", keys=[experiment["key"]]))
    assert not review["canApply"]
    assert any(item["code"] == "JOBS_ACTIVE" for item in review["blockers"])


def test_unreadable_execution_stays_visible_as_unknown_not_planned(prepared, monkeypatch):
    service, development, spec, experiment, _source = prepared
    preview = development.preview(spec)
    frozen = development.freeze(spec, preview["previewHash"], "batch", {"tag": "Batch"})

    def unreadable(*_args, **_kwargs):
        raise StorageError("Corrupt fixture", "TRAINING_STATE_INVALID")

    monkeypatch.setattr(service.training, "execution", unreadable)
    record = service.get(experiment["id"])
    assert record["status"] == "unknown"
    assert record["batches"][0]["id"] == frozen["id"]
    assert record["batches"][0]["executionError"]["code"] == "TRAINING_STATE_INVALID"


def test_registry_summary_is_bounded_and_detail_hydrates_only_selected_experiment(
    prepared, monkeypatch
):
    service, development, spec, experiment, _source = prepared
    preview = development.preview(spec)
    frozen = development.freeze(spec, preview["previewHash"], "batch", {"tag": "Batch"})
    other = service.create(CreateModelExperiment(name="Other", operationId="other"))
    reads = []

    def execution(identity, **kwargs):
        reads.append((identity, kwargs["include_progress"]))
        return None

    monkeypatch.setattr(service.training, "execution", execution)
    records = service.list(summary=True)["items"]
    listed = next(item for item in records if item["id"] == experiment["id"])
    assert listed["summary"] is True and listed["inputSnapshot"] is None
    compact = listed["batches"][0]
    assert compact["id"] == frozen["id"]
    assert set(compact["manifest"]) == {"kind", "version", "summary", "spec"}
    assert "recipe" not in compact["manifest"]["spec"]
    assert "execution" not in compact and "inputSnapshot" not in compact
    assert reads == [(frozen["id"], False)]
    reads.clear()
    assert service.get(other["id"])["batches"] == []
    assert reads == []
    detail = service.get(experiment["id"])
    assert detail["summary"] is False
    assert detail["batches"][0]["manifest"] == frozen["manifest"]
    assert reads == [(frozen["id"], True)]


def test_legacy_batches_and_incomplete_drafts_remain_visible_without_rewriting(prepared):
    service, development, spec, _experiment, _source = prepared
    legacy_spec = DevelopmentBatchSpec.model_validate(
        {
            key: value
            for key, value in spec.model_dump().items()
            if key not in {"experimentId", "experimentRevision"}
        }
    )
    assert "experimentId" not in legacy_spec.model_dump()
    preview = development.preview(legacy_spec)
    frozen = development.freeze(legacy_spec, preview["previewHash"], "legacy", {"tag": "Legacy"})
    old_inputs = service.store.create_draft(
        "experiment", "Earlier input plan", {"type": "mil-experiment", "spec": {}}
    )
    old_batch = service.store.create_draft(
        "experiment", "Earlier batch plan", {"type": "development-batch", "spec": "incomplete"}
    )
    items = {item["id"]: item for item in service.list()["items"]}
    for record in (frozen, old_inputs, old_batch):
        item = items[f"legacy-{record['id']}"]
        assert item["legacy"] is True
        with pytest.raises(StorageError) as error:
            service.update(item["id"], UpdateModelExperiment(name="Changed", expectedRevision=1))
        assert error.value.code == "LEGACY_EXPERIMENT_READ_ONLY"
    assert service.store.get_configuration(frozen["id"]) == frozen
    assert service.store.get_draft(old_inputs["id"]) == old_inputs
    assert service.store.get_draft(old_batch["id"]) == old_batch
    cleanup(service, "archive", f"configuration:{frozen['id']}")
    assert service.get(f"legacy-{frozen['id']}")["batches"][0]["state"] == "archived"


@pytest.mark.parametrize(
    "values", [{"tags": [" "]}, {"tags": ["same", "same"]}, {"name": "  "}, {"tags": ["x" * 81]}]
)
def test_experiment_values_validate_names_and_tags(values):
    with pytest.raises(ValidationError):
        CreateModelExperiment.model_validate(
            {"name": "Experiment", "operationId": "create", **values}
        )


@pytest.mark.parametrize("ownership", [{"experimentId": "draft-owner"}, {"experimentRevision": 1}])
def test_batch_owner_requires_the_revision_together(ownership):
    with pytest.raises(ValidationError):
        DevelopmentBatchSpec.model_validate(
            {
                "experimentName": "Experiment",
                "batchName": "Batch",
                "inputs": {"protocolId": "p", "featureBundleId": "b"},
                **ownership,
            }
        )


def test_aggregate_status_uses_actual_retained_work():
    def batch(status, state="active"):
        return {"status": status, "state": state}

    assert aggregate_status([]) == "created"
    assert aggregate_status([], True) == "planned"
    assert aggregate_status([batch("completed"), batch("planned")]) == "partial"
    assert aggregate_status([batch("completed"), batch("running")]) == "running"
    assert aggregate_status([batch("failed", "trashed"), batch("completed")]) == "completed"
    assert aggregate_status([batch("completed", "archived")]) == "completed"


def test_registry_api_project_isolation_auth_and_generic_route_bypass_are_protected(tmp_path):
    settings = Settings(workspace=tmp_path / "registry", data_roots=(tmp_path,))
    with TestClient(create_app(settings), base_url="http://127.0.0.1:8787") as client:
        client.headers["X-HistoPilot-Token"] = client.get("/api/v1/session").json()["token"]
        project = client.post(
            "/api/v1/projects", json={"name": "Development", "storagePath": str(tmp_path / "owned")}
        ).json()
        other = client.post(
            "/api/v1/projects", json={"name": "Other", "storagePath": str(tmp_path / "other")}
        ).json()
        base = f"/api/v1/projects/{project['id']}"
        prefix = base + "/model-experiments"
        command = {"name": "First", "operationId": "create"}
        token = client.headers.pop("X-HistoPilot-Token")
        assert client.post(prefix, json=command).status_code == 401
        client.headers["X-HistoPilot-Token"] = token
        response = client.post(prefix, json=command)
        assert response.status_code == 201, response.text
        record = response.json()
        assert record["status"] == "created"
        assert client.post(prefix, json=command).json()["id"] == record["id"]
        assert len(client.get(prefix).json()["items"]) == 1
        assert client.get(prefix + "?state=unsupported").status_code == 422
        assert (
            client.get(
                f"/api/v1/projects/{other['id']}/model-experiments/{record['id']}"
            ).status_code
            == 404
        )
        response = client.patch(
            prefix + "/" + record["id"], json={"name": "Renamed", "expectedRevision": 1}
        )
        assert response.status_code == 200, response.text
        assert response.json()["revision"] == 2
        assert (
            client.patch(
                prefix + "/" + record["id"], json={"name": "Stale", "expectedRevision": 1}
            ).status_code
            == 409
        )
        assert (
            client.post(
                base + "/drafts",
                json={
                    "kind": "experiment",
                    "name": "Bypass",
                    "payload": {"type": "model-experiment"},
                },
            ).status_code
            == 409
        )
        assert (
            client.patch(
                base + "/drafts/" + record["id"],
                json={
                    "name": "Bypass",
                    "expectedRevision": 2,
                    "payload": {"type": "mil-experiment"},
                },
            ).status_code
            == 409
        )
        owned = {
            "type": "development-batch",
            "experimentId": record["id"],
            "spec": {"experimentId": record["id"], "experimentRevision": 2},
        }
        saved = client.post(
            base + "/drafts", json={"kind": "experiment", "name": "Owned plan", "payload": owned}
        )
        assert saved.status_code == 201, saved.text
        for values in (
            {"type": "development-batch", "experimentId": record["id"]},
            {"type": "development-batch", "experimentId": [record["id"]], "experimentRevision": 2},
            {**owned, "experimentId": "another-owner"},
            {**owned, "experimentRevision": 1},
        ):
            response = client.post(
                base + "/drafts", json={"kind": "experiment", "name": "Invalid", "payload": values}
            )
            assert response.status_code in (409, 422), response.text
        for replacement in (
            {"type": "development-batch", "spec": {}},
            {"type": "development-batch", "experimentId": "another-owner", "experimentRevision": 2},
        ):
            response = client.patch(
                base + "/drafts/" + saved.json()["id"],
                json={
                    "name": "Invalid reassignment",
                    "expectedRevision": 1,
                    "payload": replacement,
                },
            )
            assert response.status_code == 409, response.text
        # Simulate an already accepted submission without starting any workers.
        store = ScientificStore(tmp_path / "owned", project["id"])
        current = store.get_draft(record["id"])
        locked = store.update_draft(
            record["id"],
            expected_revision=current["revision"],
            name=current["name"],
            payload={
                **current["payload"],
                "submission": {
                    "operationId": "accepted",
                    "expectedRevision": 2,
                    "submittedAt": "2026-09-12T12:00:00+00:00",
                    "status": "launching",
                    "batchIds": [],
                    "publications": [],
                    "error": None,
                },
            },
        )
        locked_payload = {
            **owned,
            "spec": {"experimentId": record["id"], "experimentRevision": locked["revision"]},
        }
        response = client.post(
            base + "/drafts",
            json={"kind": "experiment", "name": "Cannot add", "payload": locked_payload},
        )
        assert response.status_code == 409, response.text
        assert response.json()["code"] == "EXPERIMENT_CONFIGURATION_LOCKED"
        response = client.patch(
            base + "/drafts/" + saved.json()["id"],
            json={"name": "Cannot edit", "expectedRevision": 1, "payload": locked_payload},
        )
        assert response.status_code == 409, response.text
        assert response.json()["code"] == "EXPERIMENT_CONFIGURATION_LOCKED"
        response = client.patch(
            prefix + "/" + record["id"],
            json={
                "name": "Cannot change inputs",
                "expectedRevision": locked["revision"],
                "inputs": {"protocolId": "another", "featureBundleId": "another"},
            },
        )
        assert response.status_code == 409, response.text
        assert response.json()["code"] == "EXPERIMENT_CONFIGURATION_LOCKED"
        response = client.patch(
            base + "/drafts/" + record["id"],
            json={
                "name": "Cannot remove lock",
                "expectedRevision": locked["revision"],
                "payload": {"type": "experiment"},
            },
        )
        assert response.status_code == 409, response.text
