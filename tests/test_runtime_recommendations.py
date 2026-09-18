"""Draft runtime suggestions connect real frozen inputs without launching jobs."""

from copy import deepcopy

import pytest
from test_training_execution import execution

from histopilot.application import runtime_recommendations as advice
from histopilot.schemas.development import DevelopmentBatchSpec, ResourcePolicy
from histopilot.storage.project_lock import StorageError

__all__ = ["execution"]


def hardware():
    return {
        "cpuCount": 36,
        "totalRamGb": 188,
        "availableRamGb": 173,
        "bootId": "boot",
        "kernel": "Linux",
    }


def gpu():
    return {
        "index": 0,
        "uuid": "gpu-a",
        "name": "RTX A5000",
        "driverVersion": "596",
        "totalMemoryGb": 24,
        "freeMemoryGb": 19.3,
        "usedMemoryGb": 4.4,
        "utilizationPercent": 20,
    }


@pytest.fixture
def environment(monkeypatch):
    monkeypatch.setattr(
        advice,
        "training_runtime",
        lambda: {
            "available": True,
            "cudaAvailable": True,
            "gpuCount": 1,
            "python": "/compute/python",
            "versions": {"torch": "2.10"},
            "host": {**hardware(), "availableRamGb": 1},
        },
    )
    monkeypatch.setattr(advice, "host_snapshot", hardware)
    monkeypatch.setattr(advice, "gpu_snapshot", lambda: {"gpus": [gpu()]})
    monkeypatch.setattr(advice, "read_active_leases", lambda: [])
    monkeypatch.setattr(advice, "observed_runtime_runs", lambda store, workloads: [])


def spec_for(batch):
    values = deepcopy(batch["manifest"]["spec"])
    values["resources"]["gpuIds"] = [0]
    return DevelopmentBatchSpec.model_validate(values, context={"legacy": True})


def test_real_input_advice_is_read_only_and_uses_fresh_hardware(execution, environment):
    service, batch, executor, _ = execution
    spec = spec_for(batch)
    before = spec.model_dump()
    configurations = service.store.list_configurations()
    result = advice.runtime_recommendation(service.store, service.filesystem, spec)
    assert result["applicable"] and result["basis"] == "estimated"
    policy = ResourcePolicy.model_validate(result["resources"])
    assert policy.gpuIds == [0] and policy.runsPerGpu == 1
    assert result["hardware"]["availableRamGb"] == 173
    assert result["memory"]["observedRuns"] == 0
    assert result["generatedAt"].endswith("+00:00")
    assert spec.model_dump() == before
    assert service.store.list_configurations() == configurations
    assert not (service.store.folder / "training").exists()
    assert not executor.launches


def test_nnmil_advice_uses_actual_fitting_headers(execution, environment, monkeypatch):
    service, batch, executor, _ = execution
    values = spec_for(batch).model_dump()
    values["recipe"].update(model="nnmil", bagSizeMode="training_median", attentionDim=2)
    spec = DevelopmentBatchSpec.model_validate(values, context={"legacy": True})
    seen = []

    def observations(store, profiles):
        seen.extend(profiles)
        return []

    monkeypatch.setattr(advice, "observed_runtime_runs", observations)
    result = advice.runtime_recommendation(service.store, service.filesystem, spec)
    assert result["applicable"] and len(seen) == 1
    assert seen[0]["model"] == "nnmil"
    assert seen[0]["trainingPatches"] == 1  # Fixture fitting bags contain three patches.
    assert seen[0]["evaluationPatches"] == 3
    assert seen[0]["runtimeVersions"] == {"torch": "2.10"}
    assert not executor.launches


def test_blocked_inputs_return_findings_without_hardware_probe(execution, monkeypatch):
    service, batch, _, _ = execution
    monkeypatch.setattr(
        advice.DevelopmentService,
        "preview",
        lambda self, spec: {
            "canFreeze": False,
            "findings": [
                {"severity": "error", "code": "BUNDLE_STALE", "message": "Refresh features."}
            ],
        },
    )
    monkeypatch.setattr(advice, "training_runtime", lambda: pytest.fail("Unexpected probe"))
    result = advice.runtime_recommendation(service.store, service.filesystem, spec_for(batch))
    assert not result["applicable"] and result["resources"] is None
    assert result["findings"][0]["code"] == "BUNDLE_STALE"


def test_hardware_probe_failure_keeps_manual_configuration_available(
    execution, environment, monkeypatch
):
    service, batch, executor, _ = execution

    def unavailable():
        raise OSError("Host counters unavailable")

    monkeypatch.setattr(advice, "host_snapshot", unavailable)
    result = advice.runtime_recommendation(service.store, service.filesystem, spec_for(batch))
    assert not result["applicable"] and result["resources"] is None
    assert result["basis"] == "unavailable" and result["findings"]
    assert not executor.launches


def test_runtime_suggestion_api_authenticates_and_scopes_before_probing(
    execution,
    environment,
    monkeypatch,
    tmp_path,
):
    from fastapi.testclient import TestClient

    from histopilot.api import create_app
    from histopilot.config import Settings

    service, batch, executor, _ = execution

    def scoped_store(self, identity):
        if identity != "project-one":
            raise StorageError("Unknown project", "PROJECT_NOT_FOUND", 404)
        return service.store

    monkeypatch.setattr(
        "histopilot.application.project_workspace.ProjectWorkspace.scientific_store",
        scoped_store,
    )
    app = create_app(Settings(workspace=tmp_path / "workspace", data_roots=(tmp_path,)))
    endpoint = "/api/v1/projects/project-one/mil-experiments/batches/runtime-recommendation"
    payload = spec_for(batch).model_dump()
    with TestClient(app, base_url="http://127.0.0.1:8787") as client:
        assert client.post(endpoint, json=payload).status_code == 401
        client.headers["X-HistoPilot-Token"] = client.get("/api/v1/session").json()["token"]
        response = client.post(endpoint, json=payload)
        assert response.status_code == 200, response.text
        assert response.json()["applicable"]
        assert (
            client.post(endpoint.replace("project-one", "foreign"), json=payload).status_code == 404
        )
        bad = deepcopy(payload)
        bad["resources"]["gpuIds"] = [-1]
        assert client.post(endpoint, json=bad).status_code == 422
    assert not executor.launches
