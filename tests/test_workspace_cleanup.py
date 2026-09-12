"""Cleanup preserves evidence, rejects races and requires explicit dependency scope."""

import copy
import json
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from histopilot.api import create_app
from histopilot.application.lifecycle import CleanupService, _confirmed_live
from histopilot.config import Settings
from histopilot.schemas.lifecycle import ApplyCleanup, CancelCleanupJob, CleanupSelection
from histopilot.storage.filesystem import LocalFilesystem
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import ScientificStore
from histopilot.workers.training_process import process_identity


class Jobs:
    def __init__(self, folder):
        self.folder = folder
        self.records = []
        self.states = {}
        self.cancelled = []

    def _jobs(self):
        return copy.deepcopy(self.records)

    def get(self, identity, **_kwargs):
        return {"state": self.states[identity]}

    def cancel(self, identity):
        self.cancelled.append(identity)
        self.states[identity] = "cancelling"
        folder = self.folder / identity
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "cancelled").touch()
        return self.get(identity)

    def add(self, identity, document, *, state="succeeded", result=None):
        folder = self.folder / identity
        folder.mkdir(parents=True, exist_ok=True)
        self.records.append({"id": identity, **document})
        self.states[identity] = state
        if result is not None:
            (folder / "result.json").write_text(json.dumps(result))


class Training:
    def __init__(self):
        self.states = {}

    def execution(self, identity, **_kwargs):
        return copy.deepcopy(self.states.get(identity))

    def cancel(self, identity, _operation):
        self.states[identity]["cancelRequested"] = True
        return self.execution(identity)


@pytest.fixture
def context(tmp_path):
    folder = tmp_path / "project"
    folder.mkdir()
    store = ScientificStore(folder, "project-test")
    store.initialize()
    packs, extractions, training = (
        Jobs(folder / "packing"),
        Jobs(folder / "extractions"),
        Training(),
    )
    service = CleanupService(
        store,
        LocalFilesystem((tmp_path,)),
        "Fixture",
        training=training,
        packs=packs,
        extractions=extractions,
    )
    return store, service


def dataset(store, name="Source", parent=None):
    draft = store.create_draft("import", name, {})
    return store.publish_dataset(
        draft["id"],
        expected_revision=1,
        manifest={"kind": "dataset", "name": name, **({"parentId": parent} if parent else {})},
        artifacts={"source.csv": b"slide_id,label\ns1,positive\n"},
        operation_id=uuid4().hex,
    )


def configuration(store, source, kind="protocol", **data):
    return store.publish_configuration(
        manifest={"kind": kind, "datasetId": source["id"], **data}, operation_id=uuid4().hex
    )


def key(record, resource_type="configuration"):
    return f"{resource_type}:{record['id']}"


def selection(action, *keys):
    return CleanupSelection(action=action, keys=list(keys))


def apply(service, action, *keys):
    picked = selection(action, *keys)
    preview = service.preview(picked)
    assert preview["canApply"], preview
    request = ApplyCleanup(
        **picked.model_dump(), previewHash=preview["previewHash"], operationId=uuid4().hex
    )
    return service.apply(request), request


def test_archive_preserves_saved_references_and_archived_consumers_protect_inputs(context):
    store, service = context
    source = dataset(store)
    target = configuration(store, source, spec={"name": "Target"})
    apply(service, "archive", key(target))
    assert store.list_configurations() == []
    assert store.get_configuration(target["id"]) == target
    report = service.preview(selection("trash", key(source, "dataset")))
    assert not report["canApply"]
    assert report["requiredKeys"] == [key(target)]
    apply(service, "archive", key(source, "dataset"))
    assert store.get_dataset(source["id"]) == source
    assert store.list_datasets() == []


def test_explicit_trash_and_restore_order_keep_original_ids_and_bytes(context):
    store, service = context
    source = dataset(store)
    target = configuration(store, source)
    keys = [key(source, "dataset"), key(target)]
    path = store.folder / "datasets" / source["id"] / "source.csv"
    original = path.read_bytes()
    apply(service, "trash", *keys)
    with pytest.raises(StorageError) as error:
        store.get_configuration(target["id"])
    assert error.value.code == "RECORD_TRASHED"
    assert path.read_bytes() == original
    restore = service.preview(selection("restore", key(target)))
    assert not restore["canApply"]
    assert restore["requiredKeys"] == [key(source, "dataset")]
    apply(service, "restore", *keys)
    assert store.get_configuration(target["id"]) == target
    assert store.get_dataset(source["id"]) == source


def test_graph_includes_revisions_saved_inputs_test_cohorts_and_validation_receipts(context):
    store, service = context
    source = dataset(store)
    revision = dataset(store, "Revision", source["id"])
    extraction_id = "extraction-" + "a" * 32
    packing_id = "packing-" + "b" * 32
    pack_id = "pack-" + "c" * 64
    service.extractions.add(extraction_id, {"spec": {"datasetId": revision["id"]}})
    feature = configuration(
        store, revision, "feature", spec={"sourceExtractionJobId": extraction_id}
    )
    service.packs.add(
        packing_id,
        {"featureSetId": feature["id"], "spec": {"action": "validate"}},
        result={"artifact": {"id": pack_id, "jobId": packing_id}},
    )
    bundle = configuration(
        store,
        revision,
        "feature-bundle",
        spec={"featureSetId": feature["id"], "packArtifactIds": []},
        feature={"validation": {"jobId": packing_id}},
    )
    target = configuration(store, source)
    cohort = configuration(
        store,
        revision,
        "evaluation-cohort",
        spec={
            "protocolId": target["id"],
            "developmentFeatureBundleId": bundle["id"],
            "inference": {"packArtifactId": pack_id},
        },
    )
    saved = store.create_draft(
        "experiment",
        "Saved MIL inputs",
        {
            "type": "mil-experiment",
            "spec": {"protocolId": target["id"], "featureBundleId": bundle["id"]},
        },
    )
    records = {item["key"]: item for item in service.catalog()["items"]}
    assert key(source, "dataset") in records[key(revision, "dataset")]["dependsOn"]
    assert f"extraction:{extraction_id}" in records[key(feature)]["dependsOn"]
    assert f"packing:{packing_id}" in records[key(bundle)]["dependsOn"]
    assert f"packing:{packing_id}" in records[key(cohort)]["dependsOn"]
    assert key(bundle) in records[key(saved, "draft")]["dependsOn"]
    report = service.preview(selection("trash", f"packing:{packing_id}"))
    assert set(report["requiredKeys"]) == {key(bundle), key(cohort), key(saved, "draft")}


def test_new_consumer_invalidates_review_and_later_restore_does_not_replay_trash(context):
    store, service = context
    source = dataset(store)
    picked = selection("trash", key(source, "dataset"))
    preview = service.preview(picked)
    request = ApplyCleanup(
        **picked.model_dump(), previewHash=preview["previewHash"], operationId="trash-dataset"
    )
    target = configuration(store, source)
    with pytest.raises(StorageError) as error:
        service.apply(request)
    assert error.value.code == "CLEANUP_PREVIEW_STALE"
    result, successful = apply(service, "trash", key(source, "dataset"), key(target))
    apply(service, "restore", key(source, "dataset"), key(target))
    assert service.apply(successful) == result
    assert store.get_dataset(source["id"]) == source
    assert len(service.catalog()["audit"]) == 2
    with pytest.raises(StorageError) as error:
        service.apply(successful.model_copy(update={"action": "archive"}))
    assert error.value.code == "OPERATION_CONFLICT"


def test_draft_edit_invalidates_review_even_without_changing_dependencies(context):
    store, service = context
    draft = store.create_draft("experiment", "Keep name", {"learningRate": 0.001})
    picked = selection("trash", key(draft, "draft"))
    preview = service.preview(picked)
    store.update_draft(
        draft["id"], expected_revision=1, name=draft["name"], payload={"learningRate": 0.002}
    )
    request = ApplyCleanup(
        **picked.model_dump(), previewHash=preview["previewHash"], operationId="stale-draft"
    )
    with pytest.raises(StorageError) as error:
        service.apply(request)
    assert error.value.code == "CLEANUP_PREVIEW_STALE"


def test_shared_pack_reference_does_not_require_restoring_an_obsolete_receipt(context):
    store, service = context
    source = dataset(store)
    pack_id = "pack-" + "1" * 64
    old_job, current_job = "packing-" + "2" * 32, "packing-" + "3" * 32
    for identity in (old_job, current_job):
        service.packs.add(
            identity,
            {"spec": {"action": "attach"}},
            result={
                "state": "succeeded",
                "jobId": identity,
                "artifact": {"id": pack_id, "jobId": identity},
            },
        )
    apply(service, "trash", f"packing:{old_job}")
    target = configuration(store, source, spec={"featurePackId": pack_id})
    apply(service, "trash", key(target))
    report = service.preview(selection("restore", key(target)))
    assert report["canApply"], report
    assert report["requiredKeys"] == []


def test_cancel_request_is_not_permission_to_hide_running_jobs(context):
    store, service = context
    source = dataset(store)
    batch = configuration(store, source, "mil-batch")
    service.training.states[batch["id"]] = {"status": "running", "runs": []}
    for selected in (key(batch), key(source, "dataset"), f"project:{store.project_id}"):
        report = service.preview(selection("archive", selected))
        assert report["blockers"][0]["code"] == "JOBS_ACTIVE"
    request = CancelCleanupJob(key=key(batch), operationId="cancel-batch")
    service.cancel(request)
    assert not service.preview(selection("trash", key(batch)))["canApply"]
    assert service.catalog()["audit"][0]["action"] == "cancel"
    assert service.catalog()["audit"][0]["targets"] == [key(batch)]
    service.training.states[batch["id"]]["status"] = "cancelled"
    apply(service, "trash", key(batch))
    assert service.cancel(request)["job"]["status"] == "cancelled"


def test_terminal_receipt_with_live_process_still_blocks_cleanup(context):
    store, service = context
    identity = "packing-" + "f" * 32
    service.packs.add(identity, {"spec": {"action": "validate"}})
    (service.packs.folder / identity / "process.json").write_text(json.dumps(process_identity()))
    report = service.preview(selection("trash", f"packing:{identity}"))
    assert not report["canApply"]
    assert report["blockers"][0]["code"] == "JOBS_ACTIVE"


def test_process_permission_failure_is_not_treated_as_stopped(monkeypatch):
    def denied(_path):
        raise PermissionError("fixture")

    monkeypatch.setattr(Path, "read_text", denied)
    with pytest.raises(StorageError) as error:
        _confirmed_live({"pid": 9876, "startTicks": 1, "bootId": "fixture"})
    assert error.value.code == "CLEANUP_PROCESS_UNKNOWN"


def test_project_trash_keeps_child_states_and_requires_project_restore_first(context):
    store, service = context
    source = dataset(store)
    project = f"project:{store.project_id}"
    apply(service, "archive", key(source, "dataset"))
    apply(service, "trash", project)
    catalog = service.catalog()
    assert catalog["projectState"] == "trashed"
    assert (
        next(item for item in catalog["items"] if item["key"] == key(source, "dataset"))["state"]
        == "archived"
    )
    assert not service.preview(selection("restore", key(source, "dataset")))["canApply"]
    apply(service, "restore", project)
    assert store.list_datasets() == []
    assert store.get_dataset(source["id"]) == source


def test_api_auth_project_isolation_reopen_and_trash_recovery(tmp_path):
    settings = Settings(workspace=tmp_path / "registry", data_roots=(tmp_path,))
    with TestClient(create_app(settings), base_url="http://127.0.0.1:8787") as client:
        client.headers["X-HistoPilot-Token"] = client.get("/api/v1/session").json()["token"]
        project = client.post(
            "/api/v1/projects", json={"name": "Cleanup", "storagePath": str(tmp_path / "owned")}
        ).json()
        prefix = f"/api/v1/projects/{project['id']}"
        draft = client.post(
            prefix + "/drafts", json={"kind": "experiment", "name": "Disposable", "payload": {}}
        ).json()
        target = {"action": "trash", "keys": [f"draft:{draft['id']}"]}
        preview = client.post(prefix + "/cleanup/preview", json=target)
        assert preview.status_code == 200, preview.text
        request = {
            **target,
            "previewHash": preview.json()["previewHash"],
            "operationId": "delete-record",
        }
        token = client.headers.pop("X-HistoPilot-Token")
        assert client.post(prefix + "/cleanup/apply", json=request).status_code == 401
        client.headers["X-HistoPilot-Token"] = token
        assert client.post(prefix + "/cleanup/apply", json=request).status_code == 200
        assert client.get(prefix + "/drafts").json()["drafts"] == []
        assert client.get(prefix + f"/drafts/{draft['id']}").status_code == 409
        target = {"action": "trash", "keys": [f"project:{project['id']}"]}
        preview = client.post(prefix + "/cleanup/preview", json=target).json()
        assert (
            client.post(
                prefix + "/cleanup/apply",
                json={
                    **target,
                    "previewHash": preview["previewHash"],
                    "operationId": "delete-project",
                },
            ).status_code
            == 200
        )
        reopened = client.post(
            "/api/v1/projects/open", json={"path": project["storagePath"]}
        ).json()
        assert reopened["lifecycleState"] == "trashed"
        response = client.get(prefix + "/workspace")
        assert response.status_code == 200, response.text
        assert response.json()["project"]["lifecycleState"] == "trashed"
        assert (
            client.post(
                prefix + "/drafts",
                json={"kind": "experiment", "name": "No resurrection", "payload": {}},
            ).status_code
            == 409
        )
        other = client.post(
            "/api/v1/projects", json={"name": "Other", "storagePath": str(tmp_path / "other")}
        ).json()
        review = client.post(
            f"/api/v1/projects/{other['id']}/cleanup/preview",
            json={"action": "trash", "keys": [f"draft:{draft['id']}"]},
        ).json()
        assert not review["canApply"]
        assert review["blockers"][0]["code"] == "RECORD_NOT_FOUND"
        assert (Path(project["storagePath"]) / "histopilot-project.json").exists()


@pytest.mark.parametrize("kind", ["predictor-refit", "model-evaluation", "model-interpretation"])
def test_compute_jobs_protect_dependencies_and_cancel_through_cleanup(context, kind):
    store, service = context
    source = dataset(store)
    record = configuration(store, source, kind=kind, name="Compute chain")

    class Compute:
        def __init__(self):
            self.current = {"status": "running", "process": None}
            self.cancelled = []

        def status(self, identity, **kwargs):
            assert identity == record["id"]
            return dict(self.current)

        def cancel(self, identity, operation):
            self.cancelled.append((identity, operation))
            self.current["cancellationRequested"] = True
            return dict(self.current)

    service.compute = compute = Compute()
    key = f"configuration:{record['id']}"
    source_key = f"dataset:{source['id']}"
    catalog = {item["key"]: item for item in service.catalog()["items"]}
    assert catalog[key]["job"]["busy"]
    assert catalog[key]["job"]["cancellable"]
    assert source_key in catalog[key]["dependsOn"]
    preview = service.preview(CleanupSelection(action="archive", keys=[source_key]))
    assert not preview["canApply"]
    service.cancel(CancelCleanupJob(key=key, operationId="cancel-compute"))
    service.cancel(CancelCleanupJob(key=key, operationId="cancel-compute"))
    assert compute.cancelled == [(record["id"], "cancel-compute")]
    compute.current = {"status": "cancelled", "process": None}
    assert service.preview(CleanupSelection(action="archive", keys=[key]))["canApply"]


@pytest.mark.parametrize("kind", ["predictor-refit", "model-interpretation"])
def test_unlaunched_compute_is_a_deletable_plan_not_an_unknown_busy_job(context, kind):
    store, service = context
    record = configuration(store, dataset(store), kind=kind, name="Not launched")
    key = f"configuration:{record['id']}"
    assert service.preview(CleanupSelection(action="trash", keys=[key]))["canApply"]


def test_clinical_and_attention_evidence_preserve_the_full_model_chain(context):
    store, service = context
    source = dataset(store)
    experiment = store.create_draft("experiment", "Attention source", {"type": "model-experiment"})
    predictor = configuration(store, source, kind="frozen-predictor", experimentId=experiment["id"])
    evaluation = configuration(store, source, kind="model-evaluation", predictorId=predictor["id"])
    clinical = configuration(store, source, kind="clinical-analysis", evaluationId=evaluation["id"])
    attention = configuration(
        store, source, kind="model-interpretation",
        predictorId=predictor["id"], evaluationId=evaluation["id"], clinicalAnalysisId=clinical["id"],
    )
    catalog = {row["key"]: row for row in service.catalog()["items"]}
    assert key(clinical) in catalog[key(attention)]["dependsOn"]
    assert key(evaluation) in catalog[key(clinical)]["dependsOn"]
    review = service.preview(selection("trash", key(predictor)))
    assert not review["canApply"]
    assert {key(evaluation), key(clinical), key(attention)} <= set(review["requiredKeys"])
    apply(service, "archive", key(clinical), key(attention))
    assert not service.preview(selection("trash", key(evaluation)))["canApply"]
    apply(service, "trash", key(evaluation), key(clinical), key(attention))
    restore = service.preview(selection("restore", key(attention)))
    assert not restore["canApply"]
    assert {key(evaluation), key(clinical)} <= set(restore["requiredKeys"])


def test_folder_attention_preserves_its_external_bundle_and_shared_pack(context):
    store, service = context
    training_data = dataset(store, "Training")
    external_data = dataset(store, "Slides to interpret")
    features = configuration(store, external_data, "feature")
    pack_id = "pack-" + "d" * 64
    packing_id = "packing-" + "e" * 32
    service.packs.add(
        packing_id, {"featureSetId": features["id"], "spec": {"action": "pack"}},
        result={"artifact": {"id": pack_id, "jobId": packing_id}},
    )
    bundle = configuration(
        store, external_data, "feature-bundle",
        spec={"featureSetId": features["id"], "packArtifactIds": [pack_id]},
        feature={"validation": {"jobId": packing_id}},
    )
    predictor = configuration(store, training_data, "frozen-predictor")
    study = configuration(
        store, training_data, "model-interpretation", predictorId=predictor["id"],
        featureBundleId=bundle["id"], packArtifactId=pack_id,
    )
    for resource in (key(bundle), f"packing:{packing_id}", key(external_data, "dataset")):
        review = service.preview(selection("trash", resource))
        assert not review["canApply"]
        assert key(study) in review["requiredKeys"]
    apply(service, "archive", key(study))
    assert not service.preview(selection("trash", key(bundle)))["canApply"]
