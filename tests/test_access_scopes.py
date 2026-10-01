"""Scoped agent tokens: one project, some route classes, redaction, parked commits, audit."""

import json
import sys

import pytest
from support import projects as fixtures
from support.cli import Service

from histopilot.api import audit
from histopilot.storage.scientific import ScientificStore
from histopilot.taskcenter.client import default_client

API = "/api/v1"
CANARY_SLIDE = "CANARY-SL-7Q2X"
CANARY_PATIENT = "CANARY-PT-9K4Z"


@pytest.fixture
def service(tmp_path, monkeypatch):
    with Service(tmp_path, monkeypatch) as current:
        yield current


def expose(service, project, level, expected="none"):
    response = service.http.put(
        f"{API}/projects/{project}/ai-exposure", json={"level": level, "expectedLevel": expected}
    )
    assert response.status_code == 200, response.text


def token_for(service, project, scopes=("read", "preview")):
    response = service.http.post(
        f"{API}/tokens", json={"projectId": project, "scopes": list(scopes), "name": "test agent"}
    )
    assert response.status_code == 201, response.text
    return response.json()


def as_agent(token):
    return {"X-HistoPilot-Token": token["token"]}


def code(response):
    return response.json().get("code")


def test_tokens_need_a_shared_project_and_report_their_access(service):
    project = service.create_project("Shared")
    refused = service.http.post(f"{API}/tokens", json={"projectId": project})
    assert refused.status_code == 409 and code(refused) == "TOKEN_EXPOSURE_REQUIRED"
    expose(service, project, "full")
    token = token_for(service, project, scopes=("read",))
    assert token["token"].startswith(f"hpt_{token['id']}_")
    access = service.http.get(f"{API}/access", headers=as_agent(token)).json()
    assert access == {
        "kind": "token",
        "tokenId": token["id"],
        "name": "test agent",
        "projectId": project,
        "projectName": "Shared",
        "scopes": ["read"],
        "exposure": "full",
        "expiresAt": token["expiresAt"],
    }
    listed = service.http.get(f"{API}/tokens").json()["tokens"]
    assert [row["id"] for row in listed] == [token["id"]] and "token" not in listed[0]


def test_a_token_reaches_its_own_project_within_its_scopes(service):
    own, other = service.create_project("Own"), service.create_project("Other")
    for project in (own, other):
        expose(service, project, "full")
    reader = as_agent(token_for(service, own, scopes=("read",)))
    assert (
        service.http.get(f"{API}/projects/{own}/model-experiments", headers=reader).status_code
        == 200
    )
    outside = service.http.get(f"{API}/projects/{other}/model-experiments", headers=reader)
    assert outside.status_code == 403 and code(outside) == "PROJECT_OUT_OF_SCOPE"
    draft = {"kind": "experiment", "name": "Draft", "payload": {"type": "target-split", "spec": {}}}
    preview = service.http.post(f"{API}/projects/{own}/drafts", json=draft, headers=reader)
    assert preview.status_code == 403 and code(preview) == "SCOPE_MISSING"
    for method, path in (
        ("PUT", f"{API}/projects/{own}/ai-exposure"),
        ("PUT", f"{API}/task-center/capacity"),
        ("GET", f"{API}/filesystem/roots"),
        ("GET", f"{API}/system"),
        ("GET", f"{API}/tokens"),
        ("GET", f"{API}/task-center/snapshot"),
        ("DELETE", f"{API}/projects/{own}"),
    ):
        refused = service.http.request(method, path, headers=reader, json={})
        assert refused.status_code == 403 and code(refused) == "SCOPE_MISSING", (method, path)
    listed = service.http.get(f"{API}/projects", headers=reader).json()
    assert [item["id"] for item in listed["projects"]] == [
        own
    ] and "defaultStoragePath" not in listed

    previewer = as_agent(token_for(service, own))
    assert (
        service.http.post(f"{API}/projects/{own}/drafts", json=draft, headers=previewer).status_code
        == 201
    )


def test_commits_from_agents_wait_for_a_person(service):
    project = service.create_project("Shared")
    expose(service, project, "full")
    agent = as_agent(token_for(service, project, scopes=("read", "preview", "commit")))
    change = {"config": {"seed": 7}}
    parked = service.http.patch(f"{API}/projects/{project}", json=change, headers=agent)
    assert parked.status_code == 202 and code(parked) == "CONFIRMATION_PENDING"
    workspace = service.http.get(f"{API}/projects/{project}/workspace").json()
    assert workspace["project"]["config"].get("seed") is None
    pending = service.http.get(f"{API}/agent-requests", headers=agent).json()["requests"]
    assert [(row["method"], row["state"], row["body"]) for row in pending] == [
        ("PATCH", "pending", change)
    ]
    request = pending[0]["id"]
    decisions = f"{API}/agent-requests/{request}"
    for step in ("claim", "resolve"):
        refused = service.http.post(
            f"{decisions}/{step}", json={"outcome": "approved"}, headers=agent
        )
        assert refused.status_code == 403 and code(refused) == "SCOPE_MISSING", step
    # A replay is recorded only for a claimed request, and a request is claimed once.
    early = service.http.post(f"{decisions}/resolve", json={"outcome": "approved"})
    assert early.status_code == 409 and code(early) == "AGENT_REQUEST_NOT_CLAIMED"
    assert service.http.post(f"{decisions}/claim").json()["state"] == "approving"
    second = service.http.post(f"{decisions}/claim")
    assert second.status_code == 409 and code(second) == "AGENT_REQUEST_SETTLED"
    # The person replays it with the session token, then records how it ended.
    assert service.http.patch(f"{API}/projects/{project}", json=change).status_code == 200
    resolved = service.http.post(
        f"{decisions}/resolve", json={"outcome": "approved", "status": 200}
    )
    assert resolved.json()["state"] == "approved"
    again = service.http.post(f"{decisions}/resolve", json={"outcome": "declined"})
    assert again.status_code == 409 and code(again) == "AGENT_REQUEST_SETTLED"
    # Each decision is in the project's audit log.
    folder = service.settings.workspace / "projects" / "Shared"
    routes = [line["route"] for line in audit.read(folder) if line["actor"]["kind"] == "person"]
    for step in ("claim", "resolve"):
        assert f"/api/v1/agent-requests/{{request_id}}/{step}" in routes, step


def test_a_declined_request_cannot_be_claimed(service):
    project = service.create_project("Shared")
    expose(service, project, "full")
    agent = as_agent(token_for(service, project, scopes=("read", "preview", "commit")))
    parked = service.http.patch(
        f"{API}/projects/{project}", json={"config": {"seed": 7}}, headers=agent
    )
    decisions = f"{API}/agent-requests/{parked.json()['requestId']}"
    assert (
        service.http.post(f"{decisions}/resolve", json={"outcome": "declined"}).json()["state"]
        == "declined"
    )
    claimed = service.http.post(f"{decisions}/claim")
    assert claimed.status_code == 409 and code(claimed) == "AGENT_REQUEST_SETTLED"


def test_lowering_exposure_and_revoking_act_on_the_next_request(service):
    project = service.create_project("Shared")
    expose(service, project, "metadata")
    token = token_for(service, project)
    agent = as_agent(token)
    assert (
        service.http.get(f"{API}/projects/{project}/model-experiments", headers=agent).status_code
        == 200
    )
    expose(service, project, "none", expected="metadata")
    closed = service.http.get(f"{API}/projects/{project}/model-experiments", headers=agent)
    assert closed.status_code == 403 and code(closed) == "EXPOSURE_NONE"
    expose(service, project, "metadata")
    assert service.http.post(f"{API}/tokens/{token['id']}/revoke").json()["state"] == "revoked"
    revoked = service.http.get(f"{API}/projects/{project}/model-experiments", headers=agent)
    assert revoked.status_code == 401 and code(revoked) == "TOKEN_INVALID"
    stale = service.http.put(
        f"{API}/projects/{project}/ai-exposure", json={"level": "full", "expectedLevel": "none"}
    )
    assert stale.status_code == 409 and code(stale) == "EXPOSURE_CONFLICT"


def test_metadata_projects_are_redacted_and_their_files_refused(service):
    project = service.create_project("Shared")
    store = ScientificStore(service.settings.workspace / "projects" / "Shared", project)
    rows = [
        {
            "slideId": CANARY_SLIDE,
            "patientId": CANARY_PATIENT,
            "attributes": {"label": "1", "site": "Hospital X"},
        }
    ]
    data, _ = fixtures.dataset(store, rows=rows)
    service.http.post(
        f"{API}/projects/{project}/model-experiments",
        json={
            "name": f"Study of {CANARY_SLIDE}",
            "notes": "Seen by Dr. Canary",
            "operationId": "c",
            "setupVersion": 1,
        },
    )
    expose(service, project, "metadata")
    agent = as_agent(token_for(service, project))
    for path in (
        f"{API}/projects/{project}/datasets/{data['id']}/records",
        f"{API}/projects/{project}/model-experiments",
        f"{API}/projects/{project}/workspace",
        f"{API}/projects",
    ):
        response = service.http.get(path, headers=agent)
        assert response.status_code == 200, (path, response.text)
        text = response.text
        for secret in (
            CANARY_SLIDE,
            CANARY_PATIENT,
            "Hospital X",
            "Dr. Canary",
            str(service.settings.workspace),
        ):
            assert secret not in text, (path, secret)
    records = service.http.get(
        f"{API}/projects/{project}/datasets/{data['id']}/records", headers=agent
    ).json()
    assert records["records"][0]["slideId"].startswith("S-")
    refused = service.http.get(f"{API}/projects/{project}/morphology/slides", headers=agent)
    assert refused.status_code == 403 and code(refused) == "EXPOSURE_METADATA"
    # The person's own view is untouched.
    assert (
        CANARY_SLIDE
        in service.http.get(f"{API}/projects/{project}/datasets/{data['id']}/records").text
    )


def enqueue(project_id, folder, identity):
    owner = {
        "kind": "experiment",
        "id": identity,
        "projectId": project_id,
        "projectFolder": str(folder),
        "title": identity,
    }
    task = {
        "id": identity,
        "kind": "mil-fold",
        "adapter": "generic",
        "title": identity,
        "group": {"kind": "mil-batch", "id": identity},
        "planOrder": 0,
        "request": {"lane": "cpu"},
        "command": {
            "argv": [sys.executable, "-c", "pass"],
            "cwd": str(folder),
            "log": str(folder / f"{identity}.log"),
        },
    }
    default_client().store.enqueue(owner, [task])


def test_other_projects_task_center_rows_are_invisible(service):
    own, other = service.create_project("Own"), service.create_project("Other")
    expose(service, own, "full")
    for project, name in ((own, "Own"), (other, "Other")):
        enqueue(project, service.settings.workspace / "projects" / name, f"task-{name.lower()}")
    agent = as_agent(token_for(service, own))
    listed = service.http.get(f"{API}/task-center/tasks", headers=agent).json()["tasks"]
    assert [row["id"] for row in listed] == ["task-own"]
    forced = service.http.get(
        f"{API}/task-center/tasks", params={"project": other}, headers=agent
    ).json()["tasks"]
    assert [row["id"] for row in forced] == ["task-own"]
    hidden = service.http.get(f"{API}/task-center/tasks/task-other", headers=agent)
    assert hidden.status_code == 404 and code(hidden) == "TASK_NOT_FOUND"
    summary = service.http.get(f"{API}/task-center/summary", headers=agent).json()
    assert summary["counts"] == {"queued": 1} and "workspace" not in summary
    owners = service.http.get(f"{API}/task-center/owners", headers=agent).json()["owners"]
    assert [row["projectId"] for row in owners] == [own]
    cancel = service.http.post(
        f"{API}/task-center/tasks/task-own/cancel", json={"operationId": "x"}, headers=agent
    )
    assert cancel.status_code == 403 and code(cancel) == "SCOPE_MISSING"


def test_agents_file_task_center_changes_only_for_their_own_work(service):
    own, other = service.create_project("Own"), service.create_project("Other")
    expose(service, own, "full")
    for project, name in ((own, "Own"), (other, "Other")):
        enqueue(project, service.settings.workspace / "projects" / name, f"task-{name.lower()}")
    agent = as_agent(token_for(service, own, scopes=("read", "preview", "commit")))
    # The person who approves acts with the full session, so the gate checks the owner first.
    foreign = service.http.post(
        f"{API}/task-center/tasks/task-other/cancel", json={"operationId": "x"}, headers=agent
    )
    assert foreign.status_code == 404 and code(foreign) == "TASK_NOT_FOUND"
    missing = service.http.post(
        f"{API}/task-center/owners/nothing-here/hold", json={"operationId": "y"}, headers=agent
    )
    assert missing.status_code == 404 and code(missing) == "TASK_OWNER_NOT_FOUND"
    parked = service.http.post(
        f"{API}/task-center/tasks/task-own/cancel", json={"operationId": "z"}, headers=agent
    )
    assert parked.status_code == 202 and code(parked) == "CONFIRMATION_PENDING"
    pending = service.http.get(f"{API}/agent-requests", headers=agent).json()["requests"]
    assert [row["path"] for row in pending] == [f"{API}/task-center/tasks/task-own/cancel"]


def test_the_planning_settings_cannot_carry_the_exposure_level(service):
    project = service.create_project("Private")
    smuggled = service.http.patch(
        f"{API}/projects/{project}", json={"config": {"aiExposure": "full"}}
    )
    assert smuggled.status_code == 422
    assert service.http.get(f"{API}/projects/{project}/ai-exposure").json()["level"] == "none"


def test_agent_requests_are_audited_without_their_bodies(service):
    project = service.create_project("Shared")
    expose(service, project, "full")
    agent = as_agent(token_for(service, project))
    service.http.get(f"{API}/projects/{project}/model-experiments", headers=agent)
    service.http.post(
        f"{API}/projects/{project}/drafts",
        json={"kind": "experiment", "name": "Secret name", "payload": {}},
        headers=agent,
    )
    entries = audit.read(service.settings.workspace / "projects" / "Shared")
    agents = [entry for entry in entries if entry["actor"]["kind"] == "agent"]
    assert {(entry["method"], entry["route"]) for entry in agents} >= {
        ("GET", "/api/v1/projects/{identity}/model-experiments"),
        ("POST", "/api/v1/projects/{identity}/drafts"),
    }
    assert "Secret name" not in json.dumps(entries)
    people = [entry for entry in entries if entry["actor"]["kind"] == "person"]
    assert any(entry["route"] == "/api/v1/projects/{identity}/ai-exposure" for entry in people)


def test_a_token_reads_files_only_in_its_project_and_sources(service):
    project = service.create_project("Shared")
    expose(service, project, "full")
    elsewhere = service.data / "private"
    elsewhere.mkdir()
    (elsewhere / "table.csv").write_text("slide_id,label\ns1,1\n")
    registered = service.data / "shared-sources"
    registered.mkdir()
    (registered / "table.csv").write_text("slide_id,label\ns1,1\n")
    added = service.http.post(
        f"{API}/projects/{project}/sources", json={"path": str(registered), "role": "data"}
    )
    assert added.status_code == 201, added.text
    agent = as_agent(token_for(service, project))
    inspect = f"{API}/projects/{project}/imports/inspect"
    outside = service.http.post(
        inspect, json={"source": {"path": str(elsewhere / "table.csv")}}, headers=agent
    )
    assert outside.status_code == 403, outside.text
    inside = service.http.post(
        inspect, json={"source": {"path": str(registered / "table.csv")}}, headers=agent
    )
    assert inside.status_code == 200, inside.text
    # The person's own session still reaches every data root.
    mine = service.http.post(inspect, json={"source": {"path": str(elsewhere / "table.csv")}})
    assert mine.status_code == 200, mine.text


def test_a_token_cannot_trade_itself_for_the_session(service):
    project = service.create_project("Shared")
    expose(service, project, "full")
    agent = as_agent(token_for(service, project))
    refused = service.http.get(f"{API}/session", headers=agent)
    assert refused.status_code == 403 and code(refused) == "SESSION_NOT_FOR_TOKENS"
    assert "token" not in refused.json()
    folder = service.settings.workspace / "projects" / "Shared"
    attempts = [line for line in audit.read(folder) if line["route"] == f"{API}/session"]
    assert [(line["actor"]["kind"], line["status"]) for line in attempts] == [("agent", 403)]
    # A request with no token still gets the session, as the browser does without sign-in.
    assert "token" in service.http.get(f"{API}/session", headers={"X-HistoPilot-Token": ""}).json()


def test_case_review_is_refused_on_metadata_projects(service):
    project = service.create_project("Shared")
    expose(service, project, "metadata")
    agent = as_agent(token_for(service, project))
    for path in ("cases/query", "cases/export"):
        refused = service.http.post(
            f"{API}/projects/{project}/evaluation-runs/run-x/{path}", json={}, headers=agent
        )
        assert refused.status_code == 403 and code(refused) == "EXPOSURE_METADATA", path
    expose(service, project, "full", expected="metadata")
    # At the full level the request reaches the service, which finds no such run.
    reached = service.http.post(
        f"{API}/projects/{project}/evaluation-runs/run-x/cases/query", json={}, headers=agent
    )
    assert reached.status_code != 403, reached.text


def test_a_token_reads_the_feature_folders_its_project_froze(service):
    project = service.create_project("Shared")
    folder = service.settings.workspace / "projects" / "Shared"
    store = ScientificStore(folder, project)
    data, rows = fixtures.dataset(store)
    # The features sit in a data root the project never registered as a source.
    frozen, _, source = fixtures.bundle(store, service.data, data, [row["slideId"] for row in rows])
    expose(service, project, "full")
    agent = as_agent(token_for(service, project))
    bundles = f"{API}/projects/{project}/feature-bundles"
    mine = {item["id"]: item["current"] for item in service.http.get(bundles).json()["items"]}
    theirs = {
        item["id"]: item["current"]
        for item in service.http.get(bundles, headers=agent).json()["items"]
    }
    assert mine == theirs == {frozen["id"]: True}
    (source / "notes.csv").write_text("slide_id,label\ns1,1\n")
    inspect = service.http.post(
        f"{API}/projects/{project}/imports/inspect",
        json={"source": {"path": str(source / "notes.csv")}},
        headers=agent,
    )
    assert inspect.status_code == 200, inspect.text


def test_frozen_bundles_lend_their_pack_folders_to_tokens(tmp_path):
    from histopilot.api.scopes import ScopeGate

    manifests = {
        "feature": [{"manifest": {"spec": {"path": str(tmp_path / "features")}}}],
        "feature-bundle": [{"manifest": {"packs": [{"outputPath": str(tmp_path / "pack")}]}}],
    }

    class Store:
        def list_configurations(self, kind=None):
            return manifests[kind]

    roots = ScopeGate(None, None, None)._roots({"sources": []}, tmp_path / "project", Store())
    assert set(roots) == {(tmp_path / name).resolve() for name in ("project", "features", "pack")}
