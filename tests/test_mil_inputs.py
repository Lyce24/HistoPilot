"""MIL resolves read policy without rewriting frozen feature bundles or protocols."""

import copy
from types import SimpleNamespace
from unittest.mock import Mock

import h5py
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from histopilot.api import create_app
from histopilot.application.feature_bundles import FeatureBundleService
from histopilot.application.feature_packs import FeaturePackService
from histopilot.application.mil_inputs import MILInputService
from histopilot.application.protocols import pack_binding_snapshot
from histopilot.config import Settings
from histopilot.schemas.feature_bundles import FeatureBundleSpec
from histopilot.schemas.mil import MILInputSpec
from histopilot.storage.filesystem import LocalFilesystem
from histopilot.storage.project_lock import StorageError
from histopilot.workers.pack_features import run_job

PACK_A = "pack-" + "a" * 64
PACK_B = "pack-" + "b" * 64
FEATURE_ID = "configuration-features"
PROTOCOL_ID = "configuration-protocol"
BUNDLE_ID = "configuration-bundle"


def pack(identity=PACK_A, dtype="float32"):
    return {
        "id": identity,
        "materializationId": "materialization-" + identity,
        "featureSetId": FEATURE_ID,
        "outputPath": "/packs/" + identity,
        "outputDtype": dtype,
        "sourceContentHash": "verified-source-hash",
        "verification": "exact-source-values",
    }


def specification(**changes):
    return MILInputSpec(protocolId=PROTOCOL_ID, featureBundleId=BUNDLE_ID, **changes)


def codes(result):
    return {item["code"] for item in result["findings"] if item["severity"] == "error"}


@pytest.fixture
def inputs(monkeypatch):
    bundle = {
        "id": BUNDLE_ID,
        "current": True,
        "findings": [],
        "manifest": {
            "kind": "feature-bundle",
            "datasetId": "dataset-a",
            "spec": {"featureSetId": FEATURE_ID, "packArtifactIds": []},
            "summary": {"dtype": "float32"},
            "packs": [],
        },
    }
    protocol = {
        "id": PROTOCOL_ID,
        "manifest": {
            "kind": "protocol",
            "datasetId": "dataset-a",
            "spec": {},
            "memberships": [{"slideId": "slide-a"}, {"slideId": "slide-b"}],
        },
    }
    feature = {
        "id": FEATURE_ID,
        "manifest": {
            "kind": "feature",
            "datasetId": "dataset-a",
            "files": [{"slideId": "slide-a"}, {"slideId": "slide-b"}],
        },
    }
    documents = {PROTOCOL_ID: protocol, FEATURE_ID: feature}

    def read(identity):
        if identity not in documents:
            raise StorageError("Configuration is unavailable.", "CONFIGURATION_NOT_FOUND", 404)
        # Return the original objects so a preview mutation is observable in tests.
        return documents[identity]

    store = Mock(spec=["get_configuration"])
    store.get_configuration.side_effect = read
    bundles = Mock(spec=["get"])
    bundles.get.return_value = bundle
    packing = Mock(spec=["resolve_artifact"])
    packing.resolve_artifact.return_value = {
        "current": True,
        "findings": [],
        "artifact": pack(),
    }
    monkeypatch.setattr(
        "histopilot.application.mil_inputs.FeatureBundleService", lambda *_: bundles
    )
    monkeypatch.setattr("histopilot.application.mil_inputs.FeaturePackService", lambda *_: packing)
    return SimpleNamespace(
        service=MILInputService(store, object()),
        store=store,
        bundles=bundles,
        packing=packing,
        bundle=bundle,
        protocol=protocol,
        feature=feature,
        documents=documents,
    )


@pytest.mark.parametrize(
    "included,source_dtype,policy,explicit,expected_policy,expected_pack",
    [
        ([], "float32", "auto", None, "native", None),
        ([pack()], "float32", "auto", None, "mmap", PACK_A),
        ([pack()], ["float32"], "auto", None, "mmap", PACK_A),
        ([pack()], "float32", "native", None, "native", None),
        ([pack()], "float32", "mmap", PACK_A, "mmap", PACK_A),
        ([pack(), pack(PACK_B)], "float32", "auto", PACK_B, "mmap", PACK_B),
        ([pack(dtype="float16")], "float32", "auto", PACK_A, "mmap", PACK_A),
        ([pack(dtype="float16")], "float16", "auto", None, "mmap", PACK_A),
        ([pack(dtype="float16")], "float32", "native", None, "native", None),
    ],
)
def test_loading_intent_resolves_without_rewriting_inputs(
    inputs, included, source_dtype, policy, explicit, expected_policy, expected_pack
):
    inputs.bundle["manifest"]["packs"] = copy.deepcopy(included)
    inputs.bundle["manifest"]["summary"]["dtype"] = source_dtype
    spec = specification(loadingPolicy=policy, packArtifactId=explicit)
    before = copy.deepcopy((inputs.bundle, inputs.documents, spec.model_dump()))
    result = inputs.service.preview(spec)
    assert result["canPlan"], result["findings"]
    assert result["resolvedLoadingPolicy"] == expected_policy
    assert result["packArtifactId"] == expected_pack
    assert result["featureSetId"] == FEATURE_ID
    assert result["executionImplemented"] is False
    assert (inputs.bundle, inputs.documents, spec.model_dump()) == before
    inputs.bundles.get.assert_called_once_with(BUNDLE_ID)
    inputs.packing.resolve_artifact.assert_not_called()


@pytest.mark.parametrize(
    "included,source_dtype,policy,explicit,code",
    [
        ([pack(), pack(PACK_B)], "float32", "auto", None, "PACK_CHOICE_REQUIRED"),
        ([pack()], "float32", "mmap", None, "PACK_CHOICE_REQUIRED"),
        ([], "float32", "mmap", None, "PACK_CHOICE_REQUIRED"),
        ([], "float32", "auto", PACK_A, "PACK_NOT_IN_BUNDLE"),
        ([pack()], "float32", "mmap", PACK_B, "PACK_NOT_IN_BUNDLE"),
        ([pack(dtype="float16")], "float32", "auto", None, "PACK_PRECISION_CHOICE_REQUIRED"),
        ([pack()], ["float16", "float32"], "auto", None, "PACK_PRECISION_CHOICE_REQUIRED"),
        ([pack()], None, "auto", None, "PACK_PRECISION_CHOICE_REQUIRED"),
        ([pack()], [], "auto", None, "PACK_PRECISION_CHOICE_REQUIRED"),
    ],
)
def test_ambiguous_or_outside_bundle_choices_never_silently_fallback(
    inputs, included, source_dtype, policy, explicit, code
):
    inputs.bundle["manifest"]["packs"] = included
    inputs.bundle["manifest"]["summary"]["dtype"] = source_dtype
    result = inputs.service.preview(specification(loadingPolicy=policy, packArtifactId=explicit))
    assert not result["canPlan"]
    assert code in codes(result)
    assert result["resolvedLoadingPolicy"] is None
    assert result["packArtifactId"] is None


def test_stale_bundle_and_dataset_conflict_both_block_planning(inputs):
    inputs.bundle.update(
        current=False,
        findings=[{"severity": "error", "code": "PACK_BINDING_CHANGED", "message": "Changed."}],
    )
    inputs.protocol["manifest"]["datasetId"] = "dataset-other"
    result = inputs.service.preview(specification())
    assert not result["canPlan"]
    assert {"BUNDLE_STALE", "PACK_BINDING_CHANGED", "BUNDLE_DATASET_MISMATCH"} <= codes(result)


def test_protocol_pinned_feature_source_must_match_bundle(inputs):
    inputs.protocol["manifest"]["spec"]["featureSetId"] = "other-feature-version"
    result = inputs.service.preview(specification())
    assert not result["canPlan"]
    assert "BUNDLE_FEATURE_MISMATCH" in codes(result)


def test_every_unique_eligible_slide_requires_features_across_repeated_folds(inputs):
    inputs.protocol["manifest"]["memberships"] += [
        {"slideId": "missing", "fold": 0},
        {"slideId": "missing", "fold": 1},
    ]
    result = inputs.service.preview(specification())
    assert not result["canPlan"]
    missing = next(item for item in result["findings"] if item["code"] == "MISSING_FEATURES")
    assert "1 eligible protocol slides" in missing["message"]


def test_source_feature_inventory_missing_after_bundle_resolution_blocks_cleanly(inputs):
    del inputs.documents[FEATURE_ID]
    result = inputs.service.preview(specification())
    assert not result["canPlan"]
    assert "CONFIGURATION_NOT_FOUND" in codes(result)


@pytest.mark.parametrize("missing", ["bundle", "protocol"])
def test_missing_scientific_inputs_return_findings(inputs, missing):
    if missing == "bundle":
        inputs.bundles.get.side_effect = StorageError("Bundle missing.", "FEATURE_BUNDLE_NOT_FOUND")
    else:
        del inputs.documents[PROTOCOL_ID]
    result = inputs.service.preview(specification())
    assert not result["canPlan"]
    expected = "FEATURE_BUNDLE_NOT_FOUND" if missing == "bundle" else "CONFIGURATION_NOT_FOUND"
    assert expected in codes(result)


def test_non_protocol_configuration_is_rejected(inputs):
    inputs.protocol["manifest"]["kind"] = "feature"
    result = inputs.service.preview(specification())
    assert not result["canPlan"]
    assert "INVALID_PROTOCOL" in codes(result)


@pytest.mark.parametrize("policy,explicit", [("native", None), ("mmap", PACK_B)])
def test_legacy_protocol_cannot_change_its_pinned_pack(inputs, policy, explicit):
    inputs.bundle["manifest"]["packs"] = [pack(), pack(PACK_B)]
    inputs.protocol["manifest"]["spec"]["featurePackId"] = PACK_A
    inputs.protocol["manifest"]["featurePack"] = pack_binding_snapshot(pack())
    result = inputs.service.preview(specification(loadingPolicy=policy, packArtifactId=explicit))
    assert not result["canPlan"]
    assert "PROTOCOL_PACK_CONFLICT" in codes(result)
    inputs.packing.resolve_artifact.assert_not_called()


def test_matching_legacy_protocol_rechecks_the_frozen_representation(inputs):
    inputs.bundle["manifest"]["packs"] = [pack()]
    inputs.protocol["manifest"]["spec"]["featurePackId"] = PACK_A
    inputs.protocol["manifest"]["featurePack"] = pack_binding_snapshot(pack())
    result = inputs.service.preview(specification())
    assert result["canPlan"]
    assert result["packArtifactId"] == PACK_A
    inputs.packing.resolve_artifact.assert_called_once_with(FEATURE_ID, PACK_A)


@pytest.mark.parametrize("change", ["stale", "identity", "missing"])
def test_legacy_pack_staleness_or_changed_identity_blocks_planning(inputs, change):
    inputs.bundle["manifest"]["packs"] = [pack()]
    inputs.protocol["manifest"]["spec"]["featurePackId"] = PACK_A
    inputs.protocol["manifest"]["featurePack"] = pack_binding_snapshot(pack())
    if change == "missing":
        inputs.packing.resolve_artifact.side_effect = StorageError(
            "Pack missing.", "PACK_NOT_FOUND"
        )
    elif change == "stale":
        inputs.packing.resolve_artifact.return_value["current"] = False
    else:
        inputs.packing.resolve_artifact.return_value["artifact"]["materializationId"] = "replaced"
    result = inputs.service.preview(specification())
    assert not result["canPlan"]
    assert ("PACK_NOT_FOUND" if change == "missing" else "PROTOCOL_PACK_CHANGED") in codes(result)


@pytest.mark.parametrize(
    "changes",
    [
        {"loadingPolicy": "native", "packArtifactId": PACK_A},
        {"loadingPolicy": "ram"},
        {"packArtifactId": "folder/path"},
        {"protocolId": ""},
        {"featureBundleId": ""},
        {"featureBundleId": "x" * 129},
        {"launch": True},
        {"workers": 99},
        {"dtype": "float16"},
    ],
)
def test_mil_request_rejects_contradictory_or_unimplemented_intent(changes):
    with pytest.raises(ValidationError):
        MILInputSpec.model_validate(
            {"protocolId": PROTOCOL_ID, "featureBundleId": BUNDLE_ID, **changes}
        )


class InlineTestExecutor:
    def __init__(self):
        self.sessions = set()
        self.plans = []

    def available(self):
        return True

    def running(self, name):
        return name in self.sessions

    def launch(self, name, _runner, plan):
        self.sessions.add(name)
        self.plans.append(plan)


def test_mil_api_uses_immutable_bundles_and_does_not_change_old_preferences(tmp_path, monkeypatch):
    executor = InlineTestExecutor()
    monkeypatch.setattr(
        "histopilot.application.feature_packs.TmuxPackingExecutor", lambda: executor
    )
    settings = Settings(workspace=tmp_path / "registry", data_roots=(tmp_path,))
    app = create_app(settings)
    with TestClient(app, base_url="http://127.0.0.1:8787") as client:
        assert (
            client.post("/api/v1/projects/unknown/mil-experiments/preview", json={}).status_code
            == 401
        )
        client.headers["X-HistoPilot-Token"] = client.get("/api/v1/session").json()["token"]

        def post(path, body, status=200):
            response = client.post(path, json=body)
            assert response.status_code == status, response.text
            return response.json()

        project = post(
            "/api/v1/projects",
            {"name": "MIL bundle inputs", "storagePath": str(tmp_path / "project")},
            201,
        )
        base = f"/api/v1/projects/{project['id']}"
        store = app.state.projects.scientific_store(project["id"])
        draft = store.create_draft("import", "slides", {})
        dataset = store.publish_dataset(
            draft["id"],
            expected_revision=1,
            manifest={"kind": "dataset"},
            artifacts={"records.json": b'[{"slideId":"A","patientId":"P1"}]'},
            operation_id="dataset",
        )
        source = tmp_path / "source-features"
        source.mkdir()
        with h5py.File(source / "A.h5", "w") as handle:
            handle.create_dataset("features", data=np.arange(12, dtype="float32").reshape(3, 4))
            handle.create_dataset("coords", data=np.arange(6, dtype="int64").reshape(3, 2))
        feature_spec = {"datasetId": dataset["id"], "path": str(source)}
        preview = post(base + "/features/preview", feature_spec)
        feature = post(
            base + "/features/freeze",
            {
                **feature_spec,
                "previewHash": preview["previewHash"],
                "operationId": "features",
                "versionLabel": {"tag": "Source features"},
            },
            201,
        )
        pack_spec = {"featureSetId": feature["id"], "action": "pack", "dtype": "preserve"}
        preview = post(base + "/feature-packs/preview", pack_spec)
        post(
            base + "/feature-packs",
            {**pack_spec, "previewHash": preview["previewHash"], "operationId": "pack"},
            201,
        )
        completed = run_job(executor.plans[0])
        executor.sessions.clear()
        assert completed["state"] == "succeeded", completed
        artifact = completed["artifact"]
        filesystem = LocalFilesystem((tmp_path,))
        packing = FeaturePackService(store, filesystem)
        packing.select(feature["id"], artifact["id"])
        bundles = FeatureBundleService(store, filesystem)
        saved = []
        for name, identities in [("Features only", []), ("Features and pack", [artifact["id"]])]:
            spec = FeatureBundleSpec(featureSetId=feature["id"], packArtifactIds=identities)
            preview = bundles.preview(spec)
            assert preview["canFreeze"], preview["findings"]
            saved.append(
                bundles.freeze(spec, preview["previewHash"], name, version_label={"tag": name})
            )
        protocol = store.publish_configuration(
            manifest={
                "kind": "protocol",
                "datasetId": dataset["id"],
                "spec": {},
                "memberships": [{"slideId": "A"}],
            },
            operation_id="protocol",
        )
        before = [store.get_configuration(item["id"]) for item in [*saved, protocol, feature]]
        for bundle, expected_policy, expected_pack in [
            (saved[0], "native", None),
            (saved[1], "mmap", artifact["id"]),
        ]:
            result = post(
                base + "/mil-experiments/preview",
                {"protocolId": protocol["id"], "featureBundleId": bundle["id"]},
            )
            assert result["canPlan"], result["findings"]
            assert result["resolvedLoadingPolicy"] == expected_policy
            assert result["packArtifactId"] == expected_pack
            assert result["executionImplemented"] is False
        outside = post(
            base + "/mil-experiments/preview",
            {
                "protocolId": protocol["id"],
                "featureBundleId": saved[0]["id"],
                "loadingPolicy": "mmap",
                "packArtifactId": artifact["id"],
            },
        )
        assert not outside["canPlan"]
        assert "PACK_NOT_IN_BUNDLE" in codes(outside)
        assert packing.selection_for(feature["id"])["artifactId"] == artifact["id"]
        assert [
            store.get_configuration(item["id"]) for item in [*saved, protocol, feature]
        ] == before
        with h5py.File(source / "A.h5", "r+") as handle:
            handle["features"][0, 0] = 777
        stale = post(
            base + "/mil-experiments/preview",
            {"protocolId": protocol["id"], "featureBundleId": saved[1]["id"]},
        )
        assert not stale["canPlan"]
        assert "BUNDLE_STALE" in codes(stale)
        post(
            base + "/mil-experiments/preview",
            {"protocolId": protocol["id"], "featureBundleId": saved[0]["id"], "launch": True},
            422,
        )
