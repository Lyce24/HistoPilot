"""Bundles pin verified inputs without mutating inventories or loading preferences."""

import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from histopilot.api import create_app
from histopilot.application.feature_bundles import FeatureBundleService
from histopilot.application.feature_packs import FeaturePackService
from histopilot.application.features import FeatureService
from histopilot.config import Settings
from histopilot.schemas.feature_bundles import FeatureBundleSpec
from histopilot.schemas.feature_packs import FeaturePackSpec
from histopilot.schemas.features import FeatureSpec
from histopilot.storage.filesystem import LocalFilesystem
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import ScientificStore
from histopilot.workers.pack_features import run_job


class FakeExecutor:
    def available(self):
        return True

    def running(self, session):
        return False

    def launch(self, session, runner, plan):
        pass


def feature_fixture(store, root):
    draft = store.create_draft("import", "Slides", {})
    dataset = store.publish_dataset(
        draft["id"],
        expected_revision=1,
        manifest={"kind": "dataset"},
        artifacts={"records.json": json.dumps([{"slideId": "001", "patientId": "p1"}]).encode()},
        operation_id="dataset",
    )
    source = root / "features"
    source.mkdir()
    with h5py.File(source / "001.h5", "w") as handle:
        handle.create_dataset("features", data=np.arange(12, dtype="float32").reshape(3, 4))
        handle.create_dataset("coords", data=np.arange(6, dtype="int64").reshape(3, 2))
    filesystem = LocalFilesystem((root,))
    features = FeatureService(store, filesystem)
    spec = FeatureSpec(datasetId=dataset["id"], path=str(source))
    feature = features.freeze(
        spec, features.preview(spec)["previewHash"], "inventory", version_label={"tag": "Source"}
    )
    packs = FeaturePackService(store, filesystem, FakeExecutor())
    return FeatureBundleService(store, filesystem), packs, feature, source


@pytest.fixture
def bundle_setup(tmp_path):
    (tmp_path / "project").mkdir()
    return feature_fixture(ScientificStore(tmp_path / "project", "project-bundles"), tmp_path)


def verify(packs, feature, operation="validate", **options):
    spec = FeaturePackSpec(featureSetId=feature["id"], action="validate", **options)
    preview = packs.preview(spec)
    assert preview["canRun"], preview["findings"]
    job = packs.submit(spec, preview["previewHash"], operation)
    result = run_job(packs.folder / job["id"] / "plan.json")
    assert result["state"] == "succeeded", result
    return result


def make_pack(packs, feature, operation="pack", dtype="preserve"):
    spec = FeaturePackSpec(featureSetId=feature["id"], action="pack", dtype=dtype)
    preview = packs.preview(spec)
    job = packs.submit(spec, preview["previewHash"], operation)
    result = run_job(packs.folder / job["id"] / "plan.json")
    assert result["state"] == "succeeded", result
    return result["artifact"]


def freeze(bundles, spec, operation="bundle", tag="My features"):
    preview = bundles.preview(spec)
    assert preview["canFreeze"], preview["findings"]
    return bundles.freeze(spec, preview["previewHash"], operation, version_label={"tag": tag})


def test_pack_ids_are_canonical_and_bounded():
    first, second = "pack-" + "a" * 64, "pack-" + "b" * 64
    spec = FeatureBundleSpec(featureSetId="feature", packArtifactIds=[second, first, second])
    assert spec.packArtifactIds == [first, second]
    with pytest.raises(ValidationError):
        FeatureBundleSpec(featureSetId="feature", packArtifactIds=["job-unknown"])


def test_header_only_inventory_cannot_freeze_a_bundle(bundle_setup):
    bundles, packs, feature, source = bundle_setup
    spec = FeatureBundleSpec(featureSetId=feature["id"])
    preview = bundles.preview(spec)
    assert not preview["canFreeze"]
    assert preview["summary"]["patchCount"] == 3
    assert preview["feature"]["validation"] is None
    assert any(item["code"] == "FULL_FEATURE_VALIDATION_REQUIRED" for item in preview["findings"])
    with pytest.raises(StorageError) as caught:
        bundles.freeze(spec, preview["previewHash"], "blocked", version_label={"tag": "Blocked"})
    assert caught.value.code == "FEATURE_BUNDLE_INVALID"
    assert bundles.list() == {"items": []}


def test_extraction_lineage_change_at_bundle_publication_is_rejected(bundle_setup, monkeypatch):
    bundles, packs, original, source = bundle_setup
    identity = "extraction-" + "e" * 32
    folder = bundles.store.folder / "extractions" / identity
    folder.mkdir(parents=True)
    job = {
        "id": identity,
        "slideCount": 1,
        "spec": {"datasetId": original["manifest"]["datasetId"], "options": {"task": "all"}},
        "outputLayout": {"featuresDir": str(source), "featureKind": "patch"},
        "runtime": {"scriptSha256": "original-script"},
    }
    (folder / "job.json").write_text(json.dumps(job))
    (folder / "result.json").write_text(json.dumps({"state": "succeeded"}))
    (folder / "validation.json").write_text(
        json.dumps(
            {
                "jobId": identity,
                "completedSlides": 1,
                "missingSlides": 0,
                "unvalidatedSlides": 0,
            }
        )
    )
    feature_service = FeatureService(bundles.store, packs.outputs)
    feature_spec = FeatureSpec.model_validate(
        {**original["manifest"]["spec"], "sourceExtractionJobId": identity}
    )
    candidate = feature_service.preview(feature_spec)
    feature = feature_service.freeze(feature_spec, candidate["previewHash"], "lineage-feature")
    verify(packs, feature)
    spec = FeatureBundleSpec(featureSetId=feature["id"])
    preview = bundles.preview(spec)
    assert preview["canFreeze"], preview["findings"]
    publish = bundles.store.publish_configuration

    def change_lineage_then_publish(*args, **kwargs):
        job["runtime"]["scriptSha256"] = "different-script"
        (folder / "job.json").write_text(json.dumps(job))
        return publish(*args, **kwargs)

    monkeypatch.setattr(bundles.store, "publish_configuration", change_lineage_then_publish)
    with pytest.raises(StorageError) as caught:
        bundles.freeze(
            spec, preview["previewHash"], "raced-lineage", version_label={"tag": "Lineage"}
        )
    assert caught.value.code == "PREVIEW_STALE"
    assert bundles.store.configuration_publication("raced-lineage") is None


@pytest.mark.parametrize("dtype,dimensions", [("float16", 4), ("float32", 8)])
def test_preview_does_not_misrepresent_mixed_feature_arrays(bundle_setup, dtype, dimensions):
    bundles, packs, feature, source = bundle_setup
    second = source / "002.h5"
    with h5py.File(second, "w") as handle:
        handle.create_dataset("features", data=np.ones((3, dimensions), dtype=dtype))
        handle.create_dataset("coords", data=np.arange(6, dtype="int64").reshape(3, 2))
    other = bundles.store.publish_configuration(
        manifest={
            **feature["manifest"],
            "files": [
                *feature["manifest"]["files"],
                {"slideId": "002", "path": str(second), **FeatureService._header(second)},
            ],
        },
        operation_id="mixed-inventory",
    )
    preview = bundles.preview(FeatureBundleSpec(featureSetId=other["id"]))
    assert not preview["canFreeze"]
    assert preview["summary"]["dtype"] == ("float32" if dtype == "float32" else None)
    assert preview["summary"]["dimensions"] == (4 if dimensions == 4 else None)
    assert preview["summary"]["slideCount"] == 2


def test_pack_current_false_blocks_bundle_even_without_resolver_findings(bundle_setup, monkeypatch):
    bundles, packs, feature, source = bundle_setup
    artifact = make_pack(packs, feature)
    resolved = bundles.packing.resolve_artifact(feature["id"], artifact["id"])
    assert resolved["current"]
    monkeypatch.setattr(
        bundles.packing,
        "resolve_artifact",
        lambda *args: {**resolved, "current": False, "findings": []},
    )
    spec = FeatureBundleSpec(featureSetId=feature["id"], packArtifactIds=[artifact["id"]])
    preview = bundles.preview(spec)
    assert not preview["canFreeze"]
    assert any(item["code"] == "PACK_VERIFICATION_REQUIRED" for item in preview["findings"])


def test_features_alone_bundle_is_immutable_named_and_does_not_change_old_protocol(bundle_setup):
    bundles, packs, feature, source = bundle_setup
    store = bundles.store
    protocol = store.publish_configuration(
        manifest={
            "kind": "protocol",
            "datasetId": feature["manifest"]["datasetId"],
            "spec": {"featureSetId": feature["id"]},
        },
        operation_id="old-protocol",
    )
    verify(packs, feature)
    bundle = freeze(bundles, FeatureBundleSpec(featureSetId=feature["id"]), tag="Source")
    assert bundle["current"] and not bundle["findings"]
    assert bundle["manifest"]["kind"] == "feature-bundle"
    assert bundle["manifest"]["packs"] == []
    assert bundle["manifest"]["feature"]["contentHash"] == feature["contentHash"]
    assert bundle["manifest"]["feature"]["validation"]["tensorValidationComplete"]
    assert bundle["versionLabel"]["tag"] == "Source"
    assert store.get_configuration(feature["id"]) == feature
    assert store.get_configuration(protocol["id"]) == protocol
    assert len(store.list_configurations("feature")) == 1
    assert len(store.list_configurations("feature-bundle")) == 1
    label = store.set_version_label(
        "configuration", bundle["id"], tag="Renamed", note="Personal note", expected_revision=1
    )
    reread = bundles.get(bundle["id"])
    assert reread["versionLabel"] == label
    assert reread["contentHash"] == bundle["contentHash"]
    assert reread["manifest"] == bundle["manifest"]


def test_bundle_includes_multiple_verified_packs_without_loading_selection(
    bundle_setup, monkeypatch
):
    bundles, packs, feature, source = bundle_setup
    first = make_pack(packs, feature)
    second = make_pack(packs, feature, "float16-pack", "float16")
    monkeypatch.setattr(
        FeaturePackService, "select", lambda *args: pytest.fail("Changed preference")
    )
    monkeypatch.setattr(
        FeaturePackService, "selection_for", lambda *args: pytest.fail("Read preference")
    )
    spec = FeatureBundleSpec(
        featureSetId=feature["id"], packArtifactIds=[second["id"], first["id"], first["id"]]
    )
    bundle = freeze(bundles, spec)
    assert bundle["current"]
    assert bundle["manifest"]["summary"]["packCount"] == 2
    assert [item["id"] for item in bundle["manifest"]["packs"]] == spec.packArtifactIds
    assert {item["outputDtype"] for item in bundle["manifest"]["packs"]} == {"float16", "float32"}
    assert all(
        item["validation"]["tensorValidationComplete"] for item in bundle["manifest"]["packs"]
    )
    assert not (packs.folder / "selections").exists()


@pytest.mark.parametrize("changed", ["source", "pack"])
def test_changes_between_preview_and_freeze_are_blocked(bundle_setup, changed):
    bundles, packs, feature, source = bundle_setup
    artifact = make_pack(packs, feature)
    spec = FeatureBundleSpec(featureSetId=feature["id"], packArtifactIds=[artifact["id"]])
    preview = bundles.preview(spec)
    path = (
        source / "001.h5" if changed == "source" else Path(artifact["outputPath"]) / "features.bin"
    )
    stamp = path.stat()
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1))
    with pytest.raises(StorageError) as caught:
        bundles.freeze(spec, preview["previewHash"], "changed", version_label={"tag": "Changed"})
    assert caught.value.code == "PREVIEW_STALE"
    assert bundles.list()["items"] == []


@pytest.mark.parametrize("changed", ["source", "pack"])
def test_final_publication_guard_rechecks_external_files(bundle_setup, monkeypatch, changed):
    bundles, packs, feature, source = bundle_setup
    artifact = make_pack(packs, feature)
    spec = FeatureBundleSpec(featureSetId=feature["id"], packArtifactIds=[artifact["id"]])
    preview = bundles.preview(spec)
    publish = bundles.store.publish_configuration

    def change_then_publish(*args, **kwargs):
        path = (
            source / "001.h5"
            if changed == "source"
            else Path(artifact["outputPath"]) / "coords.bin"
        )
        stamp = path.stat()
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1))
        return publish(*args, **kwargs)

    monkeypatch.setattr(bundles.store, "publish_configuration", change_then_publish)
    with pytest.raises(StorageError) as caught:
        bundles.freeze(spec, preview["previewHash"], "race", version_label={"tag": "Race"})
    assert caught.value.code == "PREVIEW_STALE"
    assert bundles.store.configuration_publication("race") is None
    assert bundles.store.list_configurations("feature-bundle") == []


def test_retry_preserves_original_bundle_after_source_changes_and_rejects_changed_intent(
    bundle_setup,
):
    bundles, packs, feature, source = bundle_setup
    verify(packs, feature)
    spec = FeatureBundleSpec(featureSetId=feature["id"])
    preview = bundles.preview(spec)
    original = bundles.freeze(
        spec, preview["previewHash"], "retry", version_label={"tag": "Original"}
    )
    with h5py.File(source / "001.h5", "r+") as handle:
        handle["features"][0, 0] = 99
    replay = bundles.freeze(
        spec, preview["previewHash"], "retry", version_label={"tag": "Original"}
    )
    assert replay["id"] == original["id"]
    assert replay["manifest"] == original["manifest"]
    assert not replay["current"]
    assert len(bundles.list()["items"]) == 1
    with pytest.raises(StorageError) as caught:
        bundles.freeze(spec, preview["previewHash"], "retry", version_label={"tag": "Other"})
    assert caught.value.code == "OPERATION_CONFLICT"


def test_later_validation_jobs_and_feature_labels_do_not_rewrite_frozen_evidence(bundle_setup):
    bundles, packs, feature, source = bundle_setup
    verify(packs, feature)
    bundle = freeze(bundles, FeatureBundleSpec(featureSetId=feature["id"]))
    verify(packs, feature, "validate-again")
    bundles.store.set_version_label(
        "configuration", feature["id"], tag="New source name", note="", expected_revision=1
    )
    current = bundles.get(bundle["id"])
    assert current["current"]
    assert current["manifest"] == bundle["manifest"]
    assert current["contentHash"] == bundle["contentHash"]


def test_changed_pack_invalidates_bundle_without_changing_frozen_membership(bundle_setup):
    bundles, packs, feature, source = bundle_setup
    artifact = make_pack(packs, feature)
    spec = FeatureBundleSpec(featureSetId=feature["id"], packArtifactIds=[artifact["id"]])
    bundle = freeze(bundles, spec)
    with (Path(artifact["outputPath"]) / "features.bin").open("r+b") as stream:
        stream.write(np.array([99], dtype="float32").tobytes())
    current = bundles.get(bundle["id"])
    assert not current["current"]
    assert any(item["code"] == "PACK_SOURCE_CHANGED" for item in current["findings"])
    assert current["manifest"] == bundle["manifest"]
    assert current["contentHash"] == bundle["contentHash"]
    assert not bundles.list()["items"][0]["current"]


def test_pack_from_another_feature_inventory_is_not_included(bundle_setup):
    bundles, packs, feature, source = bundle_setup
    artifact = make_pack(packs, feature)
    other = bundles.store.publish_configuration(
        manifest={**feature["manifest"], "extraEvidence": "another inventory"},
        operation_id="other-inventory",
    )
    verify(packs, other, "verify-other")
    preview = bundles.preview(
        FeatureBundleSpec(featureSetId=other["id"], packArtifactIds=[artifact["id"]])
    )
    assert not preview["canFreeze"]
    assert any(item["code"] == "PACK_FEATURE_MISMATCH" for item in preview["findings"])


def test_bundle_api_freezes_lists_resolves_and_requires_named_intent(tmp_path):
    settings = Settings(workspace=tmp_path / "registry", data_roots=(tmp_path,))
    with TestClient(create_app(settings), base_url="http://127.0.0.1:8787") as client:
        client.headers["X-HistoPilot-Token"] = client.get("/api/v1/session").json()["token"]
        project = client.post(
            "/api/v1/projects", json={"name": "Bundles", "storagePath": str(tmp_path / "project")}
        ).json()
        store = client.app.state.projects.scientific_store(project["id"])
        bundles, packs, feature, source = feature_fixture(store, tmp_path)
        verify(packs, feature)
        base = f"/api/v1/projects/{project['id']}/feature-bundles"
        spec = {"featureSetId": feature["id"], "packArtifactIds": []}
        preview = client.post(base + "/preview", json=spec).json()
        assert preview["canFreeze"]
        request = {**spec, "previewHash": preview["previewHash"], "operationId": "bundle-api"}
        assert client.post(base + "/freeze", json=request).status_code == 422
        response = client.post(
            base + "/freeze", json={**request, "versionLabel": {"tag": "Verified source"}}
        )
        assert response.status_code == 201, response.text
        bundle = response.json()
        assert bundle["current"]
        assert client.get(base).json()["items"] == [bundle]
        assert client.get(base + "/" + bundle["id"]).json() == bundle
