"""Every request gets an ID it is answered with, and one access line that names its route
template and never a secret."""

import json
import re

import pytest
from fastapi.testclient import TestClient
from support.cli import BASE_URL, Service
from support.logs import collect as collect

from histopilot.api import audit


@pytest.fixture
def service(tmp_path, monkeypatch):
    with Service(tmp_path, monkeypatch) as current:
        yield current


def test_requests_get_an_id_that_is_echoed_and_never_trusted_blindly(service):
    http = service.http
    chosen = http.get("/api/v1/health", headers={"X-Request-ID": "trace-1"})
    assert chosen.headers["x-request-id"] == "trace-1"
    for offered in ("two words", "x" * 129):
        replaced = http.get("/api/v1/health", headers={"X-Request-ID": offered})
        assert re.fullmatch(r"[0-9a-f]{32}", replaced.headers["x-request-id"])
    twice = http.get("/api/v1/health", headers=[("X-Request-ID", "a"), ("X-Request-ID", "b")])
    assert twice.headers["x-request-id"] not in ("a", "b")
    generated = {http.get("/api/v1/health").headers["x-request-id"] for _ in range(3)}
    assert len(generated) == 3
    # Refused before any route runs.
    refused = http.get(
        "/api/v1/projects", headers={"X-HistoPilot-Token": "wrong", "X-Request-ID": "trace-2"}
    )
    assert refused.status_code == 401 and refused.headers["x-request-id"] == "trace-2"


def test_an_unexpected_error_still_names_its_request(service):
    app = service.http.app

    def explode():
        raise RuntimeError("unexpected")

    app.add_api_route("/api/v1/explode", explode, methods=["POST"])
    http = TestClient(app, base_url=BASE_URL, raise_server_exceptions=False)
    failed = http.post(
        "/api/v1/explode",
        headers={
            "X-HistoPilot-Token": service.http.headers["X-HistoPilot-Token"],
            "X-Request-ID": "trace-3",
        },
    )
    assert failed.status_code == 500 and failed.json()["code"] == "INTERNAL_ERROR"
    assert failed.headers["x-request-id"] == "trace-3"


def test_access_lines_name_the_route_template_and_never_a_secret(service, collect):
    http = service.http
    lines = collect("histopilot.api.access")
    token = http.headers["X-HistoPilot-Token"]
    http.post(
        "/api/v1/projects/project-secret-1/drafts?login=query-secret-2",
        headers={
            "X-Request-ID": "trace-4",
            "Authorization": "Bearer header-secret-3",
            "Cookie": "histopilot-login-8787=cookie-secret-4",
        },
        json={"kind": "experiment", "name": "body-secret-5", "payload": {}},
    )
    http.get("/api/v1/projects", headers={"X-HistoPilot-Token": "hpt_agent-secret-6"})
    http.get("/api/v1/health", headers={"X-HistoPilot-Token": ""})
    http.get("/assets/app.js")
    written = json.dumps(lines)
    for secret in (token, "project-secret-1", "query-secret-2", "header-secret-3"):
        assert secret not in written
    for secret in ("cookie-secret-4", "body-secret-5", "agent-secret-6"):
        assert secret not in written
    draft = next(line for line in lines if line["request_id"] == "trace-4")
    assert draft["msg"] == f"POST /api/v1/projects/{{identity}}/drafts {draft['status']}"
    assert draft["route"] == "/api/v1/projects/{identity}/drafts"
    assert draft["method"] == "POST" and draft["auth"] == "session"
    assert isinstance(draft["duration_ms"], float) and draft["duration_ms"] >= 0
    # A route refused before routing, a request without a credential, a static file.
    assert [(line["route"], line["status"], line["auth"]) for line in lines[-3:]] == [
        ("/api/v1/projects", 401, "token"),
        ("/api/v1/health", 200, "none"),
        ("/{path:path}", 404, "session"),
    ]


def test_audit_lines_name_the_request(service):
    project = service.create_project("Shared")
    changed = service.http.put(
        f"/api/v1/projects/{project}/ai-exposure",
        json={"level": "full", "expectedLevel": "none"},
        headers={"X-Request-ID": "trace-5"},
    )
    assert changed.status_code == 200, changed.text
    [entry] = [
        row
        for row in audit.read(service.settings.workspace / "projects" / "Shared")
        if row["route"] == "/api/v1/projects/{identity}/ai-exposure"
    ]
    assert entry["requestId"] == "trace-5" and entry["actor"]["kind"] == "person"
