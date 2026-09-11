"""Browser contract: freeze features, validate/pack, inspect and reopen durable results."""

from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest
from fastapi.testclient import TestClient

from histopilot.api import create_app
from histopilot.application.protocols import pack_binding_snapshot
from histopilot.config import Settings
from histopilot.storage.packed import PackedFeatureStore, validate_pack
from histopilot.workers.pack_features import run_job


class FakeExecutor:
    def __init__(self):
        self.sessions = set()
        self.plans = []

    def available(self):
        return True

    def running(self, session):
        return session in self.sessions

    def launch(self, session, runner, plan):
        self.sessions.add(session)
        self.plans.append(plan)


def auth(client):
    client.headers["X-HistoPilot-Token"] = client.get("/api/v1/session").json()["token"]


def post(client, path, data, status=200):
    response = client.post(path, json=data)
    assert response.status_code == status, response.text
    return response.json()


@pytest.mark.parametrize("action", ["pack", "validate"])
def test_frozen_features_worker_receipt_preflight_and_reopen(tmp_path, monkeypatch, action):
    executor = FakeExecutor()
    monkeypatch.setattr(
        "histopilot.application.feature_packs.TmuxPackingExecutor", lambda: executor
    )
    settings = Settings(workspace=tmp_path / "registry", data_roots=(tmp_path,))
    app = create_app(settings)
    with TestClient(app, base_url="http://127.0.0.1:8787") as client:
        assert client.get("/api/v1/projects/unknown/feature-packs").status_code == 401
        auth(client)
        project = post(
            client,
            "/api/v1/projects",
            {
                "name": "Feature packing",
                "storagePath": str(tmp_path / "project"),
            },
            201,
        )
        base = f"/api/v1/projects/{project['id']}"
        store = app.state.projects.scientific_store(project["id"])
        draft = store.create_draft("import", "slides", {})
        dataset = store.publish_dataset(
            draft["id"],
            expected_revision=1,
            manifest={"kind": "dataset"},
            artifacts={"records.json": b'[{"slideId":"001.A","patientId":"P1"}]'},
            operation_id="dataset",
        )
        features, coordinates = store.folder / "features", store.folder / "coordinates"
        features.mkdir()
        coordinates.mkdir()
        feature_file = features / "001.A_embed.hdf5"
        expected = np.arange(12, dtype="float32").reshape(3, 4) / 10
        expected_coords = np.arange(6, dtype="int64").reshape(3, 2)
        with h5py.File(feature_file, "w") as handle:
            handle.create_dataset("features", data=expected)
        with h5py.File(coordinates / "001.A_patches.h5", "w") as handle:
            handle.create_dataset("coords", data=expected_coords)
        spec = {
            "datasetId": dataset["id"],
            "path": str(features),
            "layout": "flat",
            "fileSuffix": ".hdf5",
            "idSuffix": "_embed",
            "coordinatesPath": str(coordinates),
        }
        preview = post(client, base + "/features/preview", spec)
        assert preview["canFreeze"], preview
        frozen = post(
            client,
            base + "/features/freeze",
            {
                **spec,
                "previewHash": preview["previewHash"],
                "operationId": "features",
                "versionLabel": {"tag": "Reviewed features"},
            },
            201,
        )
        feature_id = frozen["id"]
        assert client.get(base + f"/features/{feature_id}/validation").json() is None
        pack_spec = {"featureSetId": feature_id, "action": action, "dtype": "preserve"}
        route = base + "/feature-packs"
        preview = post(client, route + "/preview", pack_spec)
        assert preview["canRun"], preview
        assert preview["slideCount"] == 1
        assert preview["sourceDtype"] == "float32"
        intent = {**pack_spec, "previewHash": preview["previewHash"], "operationId": "pack-op"}
        job = post(client, route, intent, 201)
        assert job["state"] == "running"
        assert post(client, route, intent, 201)["id"] == job["id"]
        assert len(executor.plans) == 1
        completed = run_job(executor.plans[0])
        executor.sessions.clear()
        assert completed["state"] == "succeeded", completed
        detail = client.get(route + f"/{job['id']}").json()
        assert detail["state"] == "succeeded"
        assert detail["logs"]
        validation = client.get(base + f"/features/{feature_id}/validation").json()
        assert validation["current"] and validation["tensorValidationComplete"]
        assert not validation["provenanceComplete"]
        assert len(client.get(route).json()["jobs"]) == 1
        protocol = store.publish_configuration(
            manifest={
                "kind": "protocol",
                "datasetId": dataset["id"],
                "spec": {"featureSetId": feature_id},
                "memberships": [{"slideId": "001.A"}],
            },
            operation_id="protocol",
        )
        preflight = client.get(base + f"/protocols/{protocol['id']}/preflight").json()
        assert preflight["tensorValidationComplete"]
        assert preflight["scope"] == "protocol-and-feature-contents"
        assert not preflight["executionReady"]
        assert not preflight["fullFeatureValidationComplete"]
        if action == "pack":
            artifact = completed["artifact"]
            destination = Path(artifact["outputPath"])
            reader = PackedFeatureStore(destination)
            np.testing.assert_array_equal(reader.read_features("001.A"), expected)
            np.testing.assert_array_equal(reader.read_coords("001.A"), expected_coords)
            assert client.get(route + f"/artifacts/{artifact['id']}").status_code == 200
            selection_url = base + f"/features/{feature_id}/pack-selection"
            assert client.get(selection_url).json()["artifactId"] is None
            selected = client.put(selection_url, json={"artifactId": artifact["id"]})
            assert selected.status_code == 200, selected.text
            assert selected.json()["current"]
            assert client.get(route).json()["selections"][feature_id] == artifact["id"]
            packed_protocol = store.publish_configuration(
                manifest={
                    **protocol["manifest"],
                    "spec": {"featureSetId": feature_id, "featurePackId": artifact["id"]},
                    "featurePack": pack_binding_snapshot(artifact),
                },
                operation_id="packed-protocol",
            )
            packed_preflight = client.get(
                base + f"/protocols/{packed_protocol['id']}/preflight"
            ).json()
            assert packed_preflight["featureSource"]["type"] == "pack"
            assert packed_preflight["tensorValidationComplete"]
            assert client.put(selection_url, json={"artifactId": None}).status_code == 200
            # Editing the preference must not alter a representation pinned in a protocol.
            assert (
                client.get(base + f"/protocols/{packed_protocol['id']}/preflight").json()[
                    "featurePackId"
                ]
                == artifact["id"]
            )
            attach_spec = {
                "featureSetId": feature_id,
                "action": "attach",
                "existingPath": str(destination),
            }
            attach_preview = post(client, route + "/preview", attach_spec)
            assert attach_preview["canRun"] and attach_preview["matchesFeatures"]
            source_manifest_bytes = (destination / "manifest.json").read_bytes()
            attached = post(
                client,
                route,
                {
                    **attach_spec,
                    "previewHash": attach_preview["previewHash"],
                    "operationId": "attach-existing",
                },
                201,
            )
            attach_result = run_job(executor.plans[-1])
            executor.sessions.clear()
            assert attach_result["state"] == "succeeded", attach_result
            imported = attach_result["artifact"]
            assert imported["materializationId"] == artifact["materializationId"]
            assert imported["featureSetId"] == feature_id
            assert (destination / "manifest.json").read_bytes() == source_manifest_bytes
            assert client.get(route + f"/{attached['id']}").json()["state"] == "succeeded"
            assert client.put(selection_url, json={"artifactId": imported["id"]}).status_code == 200
        else:
            assert not (store.folder / "feature-packs").exists()
            assert completed["artifact"] is None
        with h5py.File(feature_file, "r+") as handle:
            handle["features"][0, 0] = 99
        validation = client.get(base + f"/features/{feature_id}/validation").json()
        assert not validation["current"]
        assert any(item["code"] == "FEATURE_SOURCE_CHANGED" for item in validation["findings"])
        if action == "pack":
            # Historical packed bytes remain valid when the external source changes.
            assert validate_pack(destination)["id"] == artifact["id"]
            assert not client.get(selection_url).json()["current"]
            stale_pack = client.get(base + f"/protocols/{packed_protocol['id']}/preflight").json()
            assert any(item["severity"] == "error" for item in stale_pack["findings"])
    reopened_app = create_app(replace(settings, workspace=tmp_path / "another-registry"))
    with TestClient(reopened_app, base_url="http://127.0.0.1:8787") as client:
        auth(client)
        reopened = post(client, "/api/v1/projects/open", {"path": project["storagePath"]})
        assert reopened["id"] == project["id"]
        assert client.get(route + f"/{job['id']}").json()["state"] == "succeeded"
        assert client.get(base + f"/configurations/{feature_id}").json() == frozen
        if action == "pack":
            assert client.get(selection_url).json()["artifactId"] == imported["id"]


def test_pack_api_rejects_unvalidated_execution_fields(tmp_path):
    with TestClient(
        create_app(Settings(workspace=tmp_path)), base_url="http://127.0.0.1:8787"
    ) as client:
        auth(client)
        response = client.post(
            "/api/v1/projects/unknown/feature-packs/preview",
            json={
                "featureSetId": "configuration-1",
                "shellCommand": "unexpected",
            },
        )
        assert response.status_code == 422
