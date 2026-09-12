"""Registered experiment chains use the project boundary and real service routes."""

from fastapi.testclient import TestClient

from histopilot.api.app import create_app
from histopilot.config import Settings


def test_predictor_and_evaluation_routes_are_scoped_authenticated_and_honest(tmp_path):
    settings = Settings(workspace=tmp_path / "registry", data_roots=(tmp_path,))
    with TestClient(create_app(settings), base_url="http://127.0.0.1:8787") as client:
        token = client.get("/api/v1/session").json()["token"]
        client.headers["X-HistoPilot-Token"] = token
        created = client.post(
            "/api/v1/projects", json={"name": "Chains", "storagePath": str(tmp_path / "chains")}
        )
        assert created.status_code == 201
        base = f"/api/v1/projects/{created.json()['id']}"
        for suffix in (
            "/predictors",
            "/predictors/choices",
            "/predictors/refits",
            "/evaluation-runs",
        ):
            response = client.get(base + suffix)
            assert response.status_code == 200, response.text
            assert response.json()["items"] == []
            assert response.json()["executionEnabled"] is True
        assert client.get(base + "/predictors?include_inactive=true").json()["items"] == []
        bulk = client.get(base + "/evaluation-runs/bulk")
        assert bulk.status_code == 200
        assert bulk.json()["items"] == []
        client.headers.pop("X-HistoPilot-Token")
        assert client.post(base + "/predictors/freeze", json={}).status_code == 401
        assert client.post(base + "/predictors/builds", json={}).status_code == 401
        assert client.post(base + "/evaluation-runs", json={}).status_code == 401
        assert client.post(base + "/evaluation-runs/bulk", json={}).status_code == 401
        assert client.post(base + "/evaluation-runs/bulk/missing/cancel", json={}).status_code == 401
        client.headers["X-HistoPilot-Token"] = token
        assert client.post(base + "/predictors/freeze", json={}).status_code == 422
        assert client.post(base + "/predictors/builds/preview", json={}).status_code == 422
        assert client.post(base + "/predictors/builds", json={}).status_code == 422
        assert client.get(base + "/predictors/builds/unknown-operation").status_code == 404
        assert client.post(base + "/evaluation-runs", json={}).status_code == 422
        assert client.post(base + "/evaluation-runs/bulk/preview", json={}).status_code == 422
        assert client.post(base + "/evaluation-runs/bulk", json={}).status_code == 422
        assert client.get("/api/v1/projects/unknown/predictors").status_code == 404
