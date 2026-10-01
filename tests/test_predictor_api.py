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
        # Declared before /predictors/{predictor_id}, so it is never read as a predictor ID.
        seeds = client.get(base + "/predictors/seed-ensembles")
        assert seeds.status_code == 200, seeds.text
        assert seeds.json() == {"items": []}
        # Subgroup performance is routed; an unknown run is a missing record, not a missing route.
        missing = client.post(
            base + "/evaluation-runs/configuration-" + "0" * 64 + "/performance/breakdown",
            json={"attribute": "site"},
        )
        assert missing.status_code == 404 and "Unknown API endpoint" not in missing.text
        # Reference standards are their own collection; scores and agreement belong to a run.
        assert client.get(base + "/reference-standards").json() == {"items": []}
        for suffix, body in (
            ("/scores", {}),
            ("/agreement", {"unit": "slide"}),
            ("/recalibration", {"unit": "slide"}),
        ):
            unknown = client.post(
                base + "/evaluation-runs/configuration-" + "0" * 64 + suffix, json=body
            )
            assert unknown.status_code == 404 and "Unknown API endpoint" not in unknown.text
        table = client.get(
            base
            + "/evaluation-runs/configuration-"
            + "0" * 64
            + "/reference-standards/configuration-"
            + "1" * 64
            + "/slide-predictions.csv"
        )
        assert table.status_code == 404 and "Unknown API endpoint" not in table.text
        bulk = client.get(base + "/evaluation-runs/bulk")
        assert bulk.status_code == 200
        assert bulk.json()["items"] == []
        client.headers.pop("X-HistoPilot-Token")
        assert client.post(base + "/predictors/freeze", json={}).status_code == 401
        assert client.post(base + "/predictors/seed-ensembles", json={}).status_code == 401
        assert (
            client.post(base + "/evaluation-runs/run/performance/breakdown", json={}).status_code
            == 401
        )
        assert client.post(base + "/predictors/builds", json={}).status_code == 401
        assert client.post(base + "/reference-standards", json={}).status_code == 401
        assert client.post(base + "/reference-standards/preview", json={}).status_code == 401
        assert client.post(base + "/evaluation-runs/run/scores", json={}).status_code == 401
        assert client.post(base + "/evaluation-runs", json={}).status_code == 401
        assert client.post(base + "/evaluation-runs/compare", json={}).status_code == 401
        assert client.post(base + "/evaluation-runs/bulk", json={}).status_code == 401
        assert (
            client.post(base + "/evaluation-runs/bulk/missing/cancel", json={}).status_code == 401
        )
        client.headers["X-HistoPilot-Token"] = token
        assert client.post(base + "/predictors/freeze", json={}).status_code == 422
        assert client.post(base + "/predictors/builds/preview", json={}).status_code == 422
        assert client.post(base + "/predictors/builds", json={}).status_code == 422
        assert client.get(base + "/predictors/builds/unknown-operation").status_code == 404
        assert client.post(base + "/evaluation-runs", json={}).status_code == 422
        assert client.post(base + "/evaluation-runs/compare", json={}).status_code == 422
        assert client.post(base + "/evaluation-runs/bulk/preview", json={}).status_code == 422
        assert client.post(base + "/evaluation-runs/bulk", json={}).status_code == 422
        assert client.get("/api/v1/projects/unknown/predictors").status_code == 404
