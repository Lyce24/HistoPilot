"""Durable feature jobs honor frozen inputs, immutable outputs and cancellation."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from histopilot.api.scientific import scientific_router
from histopilot.application.feature_packs import FeaturePackService
from histopilot.application.features import FeatureService
from histopilot.schemas.feature_packs import FeaturePackSpec
from histopilot.schemas.features import FeatureSpec
from histopilot.storage.filesystem import LocalFilesystem
from histopilot.storage.packed import build_pack
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import ScientificStore
from histopilot.workers.pack_features import run_job
from histopilot.workers.packing_process import output_lock, write_json


class FakeExecutor:
    def __init__(self):
        self.sessions = set()
        self.launches = []

    def available(self):
        return True

    def running(self, session):
        return session in self.sessions

    def launch(self, session, runner, plan):
        self.sessions.add(session)
        self.launches.append((runner, plan))


@pytest.fixture
def packing(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    store = ScientificStore(project, "project-packing")
    draft = store.create_draft("import", "fixture", {})
    dataset = store.publish_dataset(
        draft["id"],
        expected_revision=1,
        manifest={"kind": "dataset"},
        artifacts={
            "records.json": json.dumps(
                [{"slideId": "001.A", "patientId": "p1"}, {"slideId": "002", "patientId": "p2"}]
            ).encode()
        },
        operation_id="dataset",
    )
    source = tmp_path / "source"
    source.mkdir()
    for slide in ("001.A", "002"):
        with h5py.File(source / f"{slide}.h5", "w") as handle:
            handle.create_dataset("features", data=np.arange(12, dtype="float32").reshape(3, 4))
            handle.create_dataset("coords", data=np.arange(6, dtype="int64").reshape(3, 2))
    filesystem = LocalFilesystem((tmp_path,))
    feature_service = FeatureService(store, filesystem)
    feature_spec = FeatureSpec(datasetId=dataset["id"], path=str(source))
    preview = feature_service.preview(feature_spec)
    feature = feature_service.freeze(feature_spec, preview["previewHash"], "features")
    executor = FakeExecutor()
    service = FeaturePackService(store, filesystem, executor)
    return service, FeaturePackSpec(featureSetId=feature["id"]), executor, source


def submit(service, spec, operation="operation"):
    preview = service.preview(spec)
    assert preview["canRun"], preview["findings"]
    return service.submit(spec, preview["previewHash"], operation)


def complete(service, executor, job):
    result = run_job(service.folder / job["id"] / "plan.json")
    executor.sessions.discard(job["sessionName"])
    return result


def test_preview_is_stable_and_submission_is_idempotent(packing):
    service, spec, executor, source = packing
    preview = service.preview(spec)
    assert preview["previewHash"] == service.preview(spec)["previewHash"]
    assert preview["slideCount"] == 2
    assert preview["patchCount"] == 6
    assert preview["sourceDtype"] == preview["outputDtype"] == "float32"
    job = service.submit(spec, preview["previewHash"], "same")
    assert job["state"] == "running"
    assert service.submit(spec, preview["previewHash"], "same")["id"] == job["id"]
    assert len(executor.launches) == 1
    plan = json.loads((service.folder / job["id"] / "plan.json").read_text())
    assert len(plan["configuration"]["manifest"]["files"]) == 2
    assert not Path(job["outputPath"]).exists()
    with pytest.raises(StorageError) as caught:
        service.submit(spec.model_copy(update={"dtype": "float16"}), preview["previewHash"], "same")
    assert caught.value.code == "OPERATION_CONFLICT"


def test_changed_source_rejects_stale_preview(packing):
    service, spec, executor, source = packing
    preview = service.preview(spec)
    with h5py.File(source / "001.A.h5", "r+") as handle:
        handle["features"][0, 0] = 999
    with pytest.raises(StorageError) as caught:
        service.submit(spec, preview["previewHash"], "stale")
    assert caught.value.code == "PREVIEW_STALE"
    assert not service.preview(spec)["canRun"]
    assert not executor.launches


def test_outputs_reject_source_overlap_symlinks_reserved_and_occupied(packing, tmp_path):
    service, spec, executor, source = packing
    linked = tmp_path / "linked"
    linked.symlink_to(tmp_path, target_is_directory=True)
    for output in (
        source,
        source / "pack",
        source.parent,
        service.store.folder,
        service.store.folder / "datasets" / "pack",
        service.store.folder / "packing" / "pack",
        linked / "output",
        Path("/outside/pack"),
        tmp_path / "parent" / ".." / "pack",
    ):
        with pytest.raises(StorageError):
            service.preview(spec.model_copy(update={"outputPath": str(output)}))
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "original").write_text("keep")
    preview = service.preview(spec.model_copy(update={"outputPath": str(occupied)}))
    assert not preview["canRun"]
    assert (occupied / "original").read_text() == "keep"


def test_validate_without_pack_and_freshness_is_metadata_only(packing, monkeypatch):
    service, spec, executor, source = packing
    spec = spec.model_copy(update={"action": "validate"})
    job = submit(service, spec)
    assert job["outputPath"] is None
    assert complete(service, executor, job)["state"] == "succeeded"
    monkeypatch.setattr(
        FeatureService, "_header", lambda *args: pytest.fail("Status opened HDF5 headers")
    )
    report = service.validation_for(spec.featureSetId)
    assert report["current"] and report["valid"] and report["tensorValidationComplete"]
    assert (
        "files" not in report and "semanticIdentity" not in report and "sourceStamps" not in report
    )
    assert service.list()["artifacts"] == []
    path = source / "001.A.h5"
    info = path.stat()
    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 1))
    assert not service.validation_for(spec.featureSetId)["current"]


def test_pack_completion_supports_empty_folder_and_preserves_sources(packing, tmp_path):
    service, spec, executor, source = packing
    output = tmp_path / "empty-output"
    output.mkdir()
    spec = spec.model_copy(update={"outputPath": str(output)})
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    job = submit(service, spec)
    result = complete(service, executor, job)
    assert result["state"] == "succeeded", result.get("error")
    artifact = service.list()["artifacts"][0]
    assert artifact["featureSetId"] == spec.featureSetId
    assert artifact["outputDtype"] == "float32"
    assert service.artifact(artifact["id"])["slides"]
    assert (output / "manifest.json").exists()
    assert before == {path.name: path.read_bytes() for path in source.iterdir()}
    assert not service.preview(spec)["canRun"]
    assert service.get(job["id"], logs=True)["logs"]
    with pytest.raises(StorageError) as caught:
        service.preview(spec.model_copy(update={"outputPath": str(output / "nested")}))
    assert caught.value.code == "OUTPUT_IMMUTABLE"


def test_existing_pack_attach_verifies_then_persists_selection(packing, tmp_path):
    service, spec, executor, source = packing
    path = tmp_path / "existing"
    original = service.store.get_configuration(spec.featureSetId)
    build_pack({**original, "id": "another-feature-version"}, path)
    before = {item.name: item.read_bytes() for item in path.iterdir()}
    attach = spec.model_copy(update={"action": "attach", "existingPath": str(path)})
    preview = service.preview(attach)
    assert preview["canRun"] and preview["matchesFeatures"]
    assert preview["packInspection"]["expectedFeatureBytes"] == 96
    assert service.selection_for(spec.featureSetId)["artifactId"] is None
    job = submit(service, attach)
    result = complete(service, executor, job)
    assert result["state"] == "succeeded", result.get("error")
    artifact = result["artifact"]
    assert artifact["id"] != artifact["materializationId"]
    assert service.resolve_artifact(spec.featureSetId, artifact["id"])["current"]
    selected = service.select(spec.featureSetId, artifact["id"])
    assert selected["current"] and selected["artifactId"] == artifact["id"]
    reopened = FeaturePackService(service.store, service.filesystem, executor)
    assert reopened.selection_for(spec.featureSetId)["artifactId"] == artifact["id"]
    assert reopened.list()["selections"][spec.featureSetId] == artifact["id"]
    assert reopened.list()["artifacts"][0]["current"]
    assert before == {item.name: item.read_bytes() for item in path.iterdir()}
    assert reopened.select(spec.featureSetId, None)["artifact"] is None
    assert reopened.selection_for(spec.featureSetId)["artifactId"] is None


def test_existing_pack_count_mismatch_is_warning_and_cannot_start(packing, tmp_path):
    service, spec, executor, source = packing
    path = tmp_path / "existing"
    build_pack(service.store.get_configuration(spec.featureSetId), path)
    with h5py.File(source / "001.A.h5", "r+") as handle:
        del handle["features"]
        del handle["coords"]
        handle.create_dataset("features", data=np.zeros((4, 4), dtype="float32"))
        handle.create_dataset("coords", data=np.zeros((4, 2), dtype="int64"))
    feature_service = FeatureService(service.store, service.filesystem)
    original = service.store.get_configuration(spec.featureSetId)
    feature_spec = FeatureSpec.model_validate(original["manifest"]["spec"])
    review = feature_service.preview(feature_spec)
    current = feature_service.freeze(feature_spec, review["previewHash"], "new-feature-counts")
    attach = FeaturePackSpec(featureSetId=current["id"], action="attach", existingPath=str(path))
    preview = service.preview(attach)
    assert not preview["canRun"] and not preview["matchesFeatures"]
    assert preview["packInspection"]["mismatchedSlideCount"] == 1
    assert any(
        item["severity"] == "warning" and item["code"] == "PACK_PATCH_COUNT_MISMATCH"
        for item in preview["findings"]
    )
    with pytest.raises(StorageError, match="Resolve preview"):
        service.submit(attach, preview["previewHash"], "blocked")
    assert not executor.launches


def test_existing_pack_changed_after_submission_cannot_publish_or_select(packing, tmp_path):
    service, spec, executor, source = packing
    path = tmp_path / "existing"
    build_pack(service.store.get_configuration(spec.featureSetId), path)
    attach = spec.model_copy(update={"action": "attach", "existingPath": str(path)})
    job = submit(service, attach)
    with (path / "coords.bin").open("r+b") as stream:
        stream.write(np.array([123], dtype="<i4").tobytes())
    result = complete(service, executor, job)
    assert result["state"] == "failed"
    assert "changed since the attachment preview" in result["error"]
    assert result["artifact"] is None
    assert service.list()["artifacts"] == []
    assert service.selection_for(spec.featureSetId)["artifactId"] is None


@pytest.mark.parametrize("changed_input", ["source", "pack"])
def test_same_size_changes_with_restored_mtime_invalidate_selected_pack(packing, changed_input):
    service, spec, executor, source = packing
    job = submit(service, spec)
    artifact = complete(service, executor, job)["artifact"]
    assert service.select(spec.featureSetId, artifact["id"])["current"]
    path = (
        source / "001.A.h5"
        if changed_input == "source"
        else Path(artifact["outputPath"]) / "features.bin"
    )
    before = path.stat()
    if changed_input == "source":
        with h5py.File(path, "r+") as handle:
            handle["features"][0, 0] = 123
    else:
        with path.open("r+b") as stream:
            stream.write(np.array([123], dtype="<f4").tobytes())
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert path.stat().st_size == before.st_size
    assert path.stat().st_mtime_ns == before.st_mtime_ns
    selected = service.selection_for(spec.featureSetId)
    assert not selected["current"]
    expected_code = "FEATURE_SOURCE_CHANGED" if changed_input == "source" else "PACK_SOURCE_CHANGED"
    assert any(item["code"] == expected_code for item in selected["findings"])
    with pytest.raises(StorageError) as caught:
        service.select(spec.featureSetId, artifact["id"])
    assert caught.value.code == "PACK_NOT_CURRENT"


def test_selected_pack_becomes_stale_without_reading_tensors(packing, monkeypatch):
    service, spec, executor, source = packing
    job = submit(service, spec)
    artifact = complete(service, executor, job)["artifact"]
    assert service.select(spec.featureSetId, artifact["id"])["current"]
    monkeypatch.setattr(
        FeatureService, "_header", lambda *args: pytest.fail("Freshness opened HDF5")
    )
    path = Path(artifact["outputPath"]) / "features.bin"
    info = path.stat()
    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 1))
    selection = service.selection_for(spec.featureSetId)
    assert not selection["current"]
    assert selection["artifactId"] == artifact["id"]
    assert any(item["code"] == "PACK_SOURCE_CHANGED" for item in selection["findings"])
    with pytest.raises(StorageError) as caught:
        service.select(spec.featureSetId, artifact["id"])
    assert caught.value.code == "PACK_NOT_CURRENT"
    assert not service.list()["artifacts"][0]["current"]
    assert service.select(spec.featureSetId, None)["current"]


def test_older_pack_receipt_needs_verification_before_selection(packing):
    service, spec, executor, source = packing
    job = submit(service, spec)
    result = complete(service, executor, job)
    result["artifact"].pop("packStamps")
    write_json(service.folder / job["id"] / "result.json", result)
    resolved = service.resolve_artifact(spec.featureSetId, result["artifact"]["id"])
    assert not resolved["current"]
    assert resolved["findings"][0]["code"] == "PACK_VERIFICATION_REFRESH_REQUIRED"


def test_old_operation_retry_without_existing_path_field_is_idempotent(packing):
    import hashlib

    service, spec, executor, source = packing
    preview = service.preview(spec)
    job = service.submit(spec, preview["previewHash"], "old-operation")
    record_path = service.folder / job["id"] / "job.json"
    record = json.loads(record_path.read_text())
    record["spec"].pop("existingPath")
    old_request = spec.model_dump(mode="json", exclude={"existingPath"})
    record["requestHash"] = hashlib.sha256(
        json.dumps(old_request, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    write_json(record_path, record)
    assert service.submit(spec, preview["previewHash"], "old-operation")["id"] == job["id"]
    assert len(executor.launches) == 1


def test_list_reads_each_terminal_receipt_once_and_deduplicates(packing, monkeypatch):
    service, spec, executor, source = packing
    job = submit(service, spec)
    complete(service, executor, job)
    real_result = service._result
    calls = []

    def counted(record):
        calls.append(record["id"])
        return real_result(record)

    monkeypatch.setattr(service, "_result", counted)
    assert len(service.list()["artifacts"]) == 1
    assert calls == [job["id"]]


def test_reverified_folder_supersedes_old_receipt_without_changing_selection(packing):
    service, spec, executor, source = packing
    created_job = submit(service, spec)
    created_result = complete(service, executor, created_job)
    original = created_result["artifact"]
    service.select(spec.featureSetId, original["id"])
    original.pop("packStamps")
    write_json(service.folder / created_job["id"] / "result.json", created_result)

    attach = spec.model_copy(update={"action": "attach", "existingPath": original["outputPath"]})
    attached_job = submit(service, attach, "reverify-existing")
    verified = complete(service, executor, attached_job)["artifact"]
    assert verified["id"] != original["id"]
    assert verified["materializationId"] == original["materializationId"]
    listing = service.list()
    assert [item["id"] for item in listing["artifacts"]] == [verified["id"]]
    assert listing["artifacts"][0]["current"]
    assert len(listing["jobs"]) == 2
    assert service.artifact(original["id"])["id"] == original["id"]
    selection = service.selection_for(spec.featureSetId)
    assert selection["artifactId"] == original["id"]
    assert not selection["current"]
    assert service.select(spec.featureSetId, verified["id"])["current"]


def test_cancellation_is_cooperative_and_retry_uses_a_new_output(packing):
    service, spec, executor, source = packing
    job = submit(service, spec)
    assert service.cancel(job["id"])["state"] == "cancelling"
    assert job["sessionName"] in executor.sessions
    result = complete(service, executor, job)
    assert result["state"] == "cancelled"
    assert not Path(job["outputPath"]).exists()
    assert service.get(job["id"])["state"] == "cancelled"
    retry = submit(service, spec, "retry")
    assert retry["id"] != job["id"]
    assert retry["outputPath"] != job["outputPath"]


def test_interrupted_launch_and_worker_lock_cannot_be_stolen(packing):
    service, spec, executor, source = packing
    job = submit(service, spec)
    executor.sessions.clear()
    assert service.get(job["id"])["state"] == "interrupted"
    with output_lock(job["outputPath"]):
        with pytest.raises(StorageError) as caught:
            with output_lock(job["outputPath"]):
                pytest.fail("A second process lock was acquired")
        assert caught.value.code == "OUTPUT_BUSY"
    retry = submit(service, spec, "retry")
    assert retry["id"] != job["id"]


def test_another_project_cannot_claim_an_active_nested_output(packing, tmp_path):
    service, spec, executor, source = packing
    output = tmp_path / "shared-output"
    job = submit(service, spec.model_copy(update={"outputPath": str(output)}))
    other = tmp_path / "other-project"
    other.mkdir()
    other_store = ScientificStore(other, "other")
    configuration = service.store.get_configuration(spec.featureSetId)
    draft = other_store.create_draft("import", "fixture", {})
    other_store.publish_dataset(
        draft["id"],
        expected_revision=1,
        manifest={"kind": "dataset"},
        artifacts={
            "records.json": service.store.read_artifact(
                configuration["manifest"]["datasetId"], "records.json"
            )
        },
        operation_id="dataset",
    )
    other_feature = other_store.publish_configuration(
        manifest=configuration["manifest"], operation_id="feature"
    )
    other_service = FeaturePackService(other_store, service.filesystem, executor)
    preview = other_service.preview(
        FeaturePackSpec(featureSetId=other_feature["id"], outputPath=str(output / "nested"))
    )
    assert not preview["canRun"]
    assert any(finding["code"] == "OUTPUT_BUSY" for finding in preview["findings"])
    executor.sessions.discard(job["sessionName"])


def test_script_worker_runs_from_unrelated_cwd(packing, tmp_path):
    service, spec, executor, source = packing
    job = submit(service, spec.model_copy(update={"action": "validate"}))
    runner, plan = executor.launches[0]
    result = subprocess.run(
        [sys.executable, str(runner), str(plan)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    executor.sessions.clear()
    assert service.get(job["id"])["state"] == "succeeded"
    assert not (service.folder / job["id"] / "process.json").exists()


def test_corrupt_terminal_receipt_does_not_masquerade_as_success(packing):
    service, spec, executor, source = packing
    job = submit(service, spec)
    write_json(
        service.folder / job["id"] / "result.json", {"jobId": "different", "state": "succeeded"}
    )
    with pytest.raises(StorageError) as caught:
        service.get(job["id"])
    assert caught.value.code == "PACKING_CORRUPT"


def test_launch_failure_is_durable_and_retry_can_use_unchanged_empty_output(
    packing, monkeypatch, tmp_path
):
    service, spec, executor, source = packing
    spec = spec.model_copy(update={"outputPath": str(tmp_path / "launch-retry")})
    launch = executor.launch

    def fail(*args):
        raise RuntimeError("tmux failed")

    monkeypatch.setattr(executor, "launch", fail)
    job = submit(service, spec)
    assert job["state"] == "failed"
    assert "tmux failed" in job["error"]
    monkeypatch.setattr(executor, "launch", launch)
    retry = submit(service, spec, "retry")
    assert retry["state"] == "running"
    assert retry["outputPath"] == job["outputPath"]


def test_source_modified_after_submission_stops_worker_before_output(packing):
    service, spec, executor, source = packing
    job = submit(service, spec)
    with h5py.File(source / "001.A.h5", "r+") as handle:
        handle["features"][0, 0] = 999
    result = complete(service, executor, job)
    assert result["state"] == "failed"
    assert "sources changed" in result["error"]
    assert not Path(job["outputPath"]).exists()
    assert service.validation_for(spec.featureSetId) is None


@pytest.mark.parametrize(
    "field",
    [
        "jobId",
        "validationFeature",
        "validationValid",
        "artifactFeature",
        "artifactJob",
        "artifactPath",
        "artifactIdentity",
        "artifactValidation",
    ],
)
def test_artifact_and_job_reads_reject_inconsistent_completion_receipts(packing, field):
    service, spec, executor, source = packing
    job = submit(service, spec)
    result = complete(service, executor, job)
    assert result["state"] == "succeeded"
    artifact_id = result["artifact"]["id"]
    if field == "jobId":
        result["jobId"] = "packing-" + "0" * 32
    elif field == "validationFeature":
        result["validation"]["featureSetId"] = "configuration-other"
    elif field == "validationValid":
        result["validation"]["valid"] = False
    elif field == "artifactFeature":
        result["artifact"]["featureSetId"] = "configuration-other"
    elif field == "artifactJob":
        result["artifact"]["jobId"] = "packing-" + "0" * 32
    elif field == "artifactPath":
        result["artifact"]["outputPath"] += "-other"
    elif field == "artifactIdentity":
        result["artifact"]["materializationId"] = "pack-" + "0" * 64
    elif field == "artifactValidation":
        result["artifact"]["validation"] = {
            **result["validation"],
            "sourceContentHash": "different",
        }
    write_json(service.folder / job["id"] / "result.json", result)
    for read in (
        lambda: service.get(job["id"]),
        service.list,
        lambda: service.artifact(artifact_id),
    ):
        with pytest.raises(StorageError) as caught:
            read()
        assert caught.value.code == "PACKING_CORRUPT"


@pytest.mark.parametrize(
    "validation,complete,warning",
    [
        (None, False, "FULL_FEATURE_VALIDATION_PENDING"),
        (
            {
                "valid": True,
                "current": False,
                "tensorValidationComplete": True,
                "provenanceComplete": True,
            },
            False,
            "FULL_FEATURE_VALIDATION_PENDING",
        ),
        (
            {
                "valid": False,
                "current": True,
                "tensorValidationComplete": True,
                "provenanceComplete": True,
            },
            False,
            "FULL_FEATURE_VALIDATION_PENDING",
        ),
        (
            {
                "valid": True,
                "current": True,
                "tensorValidationComplete": True,
                "provenanceComplete": False,
            },
            True,
            "ENCODER_PROVENANCE_UNVERIFIED",
        ),
        (
            {
                "valid": True,
                "current": True,
                "tensorValidationComplete": True,
                "provenanceComplete": True,
            },
            True,
            None,
        ),
    ],
)
def test_preflight_reports_content_scope_and_only_unverified_provenance(
    packing, monkeypatch, validation, complete, warning
):
    service, spec, executor, source = packing
    configuration = service.store.get_configuration(spec.featureSetId)
    protocol = service.store.publish_configuration(
        manifest={
            "kind": "protocol",
            "datasetId": configuration["manifest"]["datasetId"],
            "spec": {"featureSetId": spec.featureSetId},
            "memberships": [{"slideId": "001.A"}, {"slideId": "002"}],
        },
        operation_id="protocol",
    )
    monkeypatch.setattr(FeaturePackService, "validation_for", lambda *args: validation)
    router = scientific_router(
        SimpleNamespace(scientific_store=lambda identity: service.store), service.filesystem
    )
    endpoint = next(
        route.endpoint
        for route in router.routes
        if route.path.endswith("/protocols/{configuration_id}/preflight")
    )
    report = endpoint("project", protocol["id"])
    assert report["tensorValidationComplete"] is complete
    assert report["scope"] == (
        "protocol-and-feature-contents" if complete else "protocol-and-feature-headers"
    )
    assert report["scientificReady"] is (warning is None)
    assert not report["executionReady"]
    codes = {item["code"] for item in report["findings"]}
    if warning:
        assert warning in codes
    else:
        assert not codes
