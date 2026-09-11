"""Browser capability negotiation and version-label method support."""

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from histopilot.api import create_app
from histopilot.config import Settings

BASE = "http://127.0.0.1:8787"


def test_session_advertises_tagged_freeze_without_exposing_project_data(tmp_path):
    settings = Settings(workspace=tmp_path / "workspace")
    with TestClient(create_app(settings), base_url=BASE) as client:
        response = client.get("/api/v1/session")
        assert response.status_code == 200
        session = response.json()
        assert set(session) == {"token", "scientificCapabilities"}
        assert isinstance(session["token"], str) and session["token"]
        assert session["scientificCapabilities"] == {
            "versionLabels": True,
            "taggedFreeze": True,
        }
        assert response.headers["cache-control"] == "no-store"
        assert str(tmp_path) not in response.text
        assert client.get("/api/v1/projects").status_code == 401


@pytest.mark.parametrize("resource", ["datasets", "configurations"])
def test_approved_dev_origin_can_preflight_put_without_bypassing_session(tmp_path, resource):
    settings = Settings(workspace=tmp_path / "workspace")
    path = f"/api/v1/projects/probe/{resource}/probe/label"
    origin = "http://localhost:5173"
    headers = {
        "Origin": origin,
        "Access-Control-Request-Method": "PUT",
        "Access-Control-Request-Headers": "content-type,x-histopilot-token",
    }
    with TestClient(create_app(settings), base_url=BASE) as production:
        assert production.options(path, headers=headers).status_code == 403
    with TestClient(create_app(replace(settings, dev=True)), base_url=BASE) as development:
        response = development.options(path, headers=headers)
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == origin
        assert "PUT" in response.headers["access-control-allow-methods"].split(", ")
        assert development.put(path, headers={"Origin": origin}, json={}).status_code == 401
        assert development.options(
            path, headers={**headers, "Origin": "https://evil.example"}
        ).status_code == 403
