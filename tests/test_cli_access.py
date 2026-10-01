"""The person's side of agent access: exposure, tokens, and approving parked requests."""

import pytest
from support.cli import Service


@pytest.fixture
def service(tmp_path, monkeypatch):
    with Service(tmp_path, monkeypatch) as current:
        project = current.create_project("Shared")
        current.cli("use", project)
        yield current, project


def test_exposure_is_shown_and_raised_only_with_confirmation(service):
    cli, project = service
    shown = cli.cli("project", "exposure", "--json").envelope["data"]
    assert shown["level"] == "none" and "nothing" in shown["meaning"]
    pending = cli.cli("project", "exposure", "--set", "full", "--json")
    assert pending.code == 7
    preview = pending.envelope["data"]["preview"]
    assert (preview["projectId"], preview["from"], preview["to"]) == (project, "none", "full")
    assert "AI provider" in preview["meaning"] and len(preview["previewHash"]) == 64
    assert cli.cli("project", "exposure", "--set", "metadata", "--yes", "--json").code == 0
    assert cli.cli("project", "exposure", "--json").envelope["data"]["level"] == "metadata"
    assert cli.cli("project", "exposure", "--set", "public", "--json").code == 2


def test_tokens_are_created_listed_and_revoked(service):
    cli, _ = service
    refused = cli.cli("token", "create", "--yes", "--json")
    assert refused.code == 3 and refused.envelope["error"]["code"] == "TOKEN_EXPOSURE_REQUIRED"
    cli.cli("project", "exposure", "--set", "metadata", "--yes")
    created = cli.cli("token", "create", "--name", "desktop app", "--yes", "--json")
    assert created.code == 0, created.stdout
    token = created.envelope["data"]["result"]
    assert token["token"].startswith("hpt_") and token["scopes"] == ["read", "preview"]
    listed = cli.cli("token", "list", "--json").envelope["data"]
    assert [(row["id"], row["state"]) for row in listed] == [(token["id"], "active")]
    assert "hpt_" not in cli.cli("token", "list").stdout
    assert cli.cli("token", "revoke", token["id"], "--yes", "--json").code == 0
    assert cli.cli("token", "list", "--json").envelope["data"][0]["state"] == "revoked"
    assert cli.cli("token", "create", "--scope", "admin", "--yes", "--json").code == 2


def test_a_parked_agent_change_is_approved_by_a_person(service, monkeypatch):
    cli, project = service
    cli.cli("project", "exposure", "--set", "full", "--yes")
    token = cli.cli(
        "token",
        "create",
        "--scope",
        "read",
        "--scope",
        "preview",
        "--scope",
        "commit",
        "--yes",
        "--json",
    ).envelope["data"]["result"]["token"]

    # The agent's side: its change is parked, not applied.
    monkeypatch.setenv("HISTOPILOT_TOKEN", token)
    parked = cli.cli(
        "api", "PATCH", f"/projects/{project}", "-d", '{"config": {"seed": 11}}', "--yes", "--json"
    )
    # Nothing changed yet, so the command is unconfirmed (exit 7) and names the request.
    assert parked.code == 7 and parked.envelope["error"]["code"] == "CONFIRMATION_PENDING"
    assert parked.envelope["data"]["preview"]["request"] == f"PATCH /api/v1/projects/{project}"
    request = parked.envelope["data"]["request"]["requestId"]
    monkeypatch.delenv("HISTOPILOT_TOKEN")

    # The person's side.
    waiting = cli.cli("confirm", "list", "--json").envelope["data"]
    assert [row["id"] for row in waiting] == [request]
    assert cli.cli("confirm", "approve", request, "--json").code == 7
    approved = cli.cli("confirm", "approve", request, "--yes", "--json")
    assert approved.code == 0, approved.stdout
    config = cli.http.get(f"/api/v1/projects/{project}/workspace").json()["project"]["config"]
    assert config["seed"] == 11
    assert cli.cli("confirm", "list", "--json").envelope["data"] == []
    settled = cli.cli("confirm", "decline", request, "--yes", "--json")
    assert settled.code == 3 and settled.envelope["error"]["code"] == "AGENT_REQUEST_SETTLED"


def test_a_token_names_its_one_project_without_the_machine(service, monkeypatch):
    cli, project = service
    cli.cli("project", "exposure", "--set", "metadata", "--yes")
    token = cli.cli("token", "create", "--yes", "--json").envelope["data"]["result"]["token"]
    cli.cli("use", "--clear")
    monkeypatch.setenv("HISTOPILOT_TOKEN", token)
    status = cli.cli("status", "--json")
    assert status.code == 0, status.stdout
    data = status.envelope["data"]
    assert data["project"] == project and data["workspace"] is None
    assert (data["access"]["projectName"], data["access"]["exposure"]) == ("Shared", "metadata")
    assert "this project only" in cli.cli("status").stdout
    # With no saved project, commands use the token's own.
    assert cli.cli("experiment", "list", "--json").code == 0
    monkeypatch.delenv("HISTOPILOT_TOKEN")
    folder = cli.settings.workspace / "projects" / "Shared"
    from histopilot.api import audit

    assert not [line for line in audit.read(folder) if line["status"] == 403]
