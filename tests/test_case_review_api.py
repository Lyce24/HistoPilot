from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient
from test_case_review import cases as cases

from histopilot.api.app import create_app
from histopilot.config import Settings


@pytest.fixture
def client(cases, tmp_path, monkeypatch):
    service = cases[0]
    app = create_app(Settings(workspace=tmp_path / "api-workspace", data_roots=(tmp_path,)))
    monkeypatch.setattr(app.state.projects, "scientific_store", lambda identity: service.store)
    monkeypatch.setattr("histopilot.api.case_review.CaseReviewService", lambda *args: service)
    with TestClient(app, base_url="http://127.0.0.1:8787") as current:
        yield current


def test_case_query_requires_auth_and_exports_verified_records(client, cases):
    service, evaluation, *_ = cases
    base = f"/api/v1/projects/review/evaluation-runs/{evaluation['id']}/cases"
    assert client.post(base + "/query", json={}).status_code == 401
    client.headers["X-HistoPilot-Token"] = client.get("/api/v1/session").json()["token"]
    response = client.post(base + "/query", json={"outcome": "false_negative"})
    assert response.status_code == 200, response.text
    assert response.json()["items"][0]["id"] == "p1"
    response = client.post(base + "/export", json={"outcome": "false_negative"})
    assert response.status_code == 200 and "text/csv" in response.headers["content-type"]
    assert "p1" in response.text and "Predictions_SHA256" in response.text
    assert client.post(base + "/query", json={"limit": 100000}).status_code == 422


def test_review_routes_preserve_edits_and_conflict_response(client, cases):
    _, evaluation, dataset, *_ = cases
    base = f"/api/v1/projects/review/datasets/{dataset['id']}/slide-reviews"
    client.headers["X-HistoPilot-Token"] = client.get("/api/v1/session").json()["token"]
    path = base + "/" + quote("s1", safe="")
    assert client.get(path).json()["revision"] == 0
    payload = {"expectedRevision": 0, "status": "review", "notes": "Inspect tumor", "evaluationId": evaluation["id"]}
    response = client.put(path, json=payload)
    assert response.status_code == 200, response.text
    assert response.json()["revision"] == 1
    assert client.put(path, json=payload).json()["revision"] == 1
    response = client.put(path, json={**payload, "notes": "stale note"})
    assert response.status_code == 409 and response.json()["code"] == "SLIDE_REVIEW_CONFLICT"
    assert client.get(path).json()["notes"] == "Inspect tumor"
    assert client.get(base).json()["items"][0]["slideId"] == "s1"
