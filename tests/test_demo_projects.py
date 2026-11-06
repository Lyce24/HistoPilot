"""A demo service can also serve packaged projects: real studies, read-only, through the full
application. For those projects it answers exactly the routes classed as reads, minus the few
that reach past the project's own records; it changes nothing, and it never opens the folders
it was given, only copies of them."""

import hashlib
import shutil

import pytest
from fastapi.testclient import TestClient
from test_demo_mode import (
    API,
    METHODS,
    WRITES,
    concrete,
    declared,
    public_client,
    refused,
)
from typer.testing import CliRunner

import histopilot.cli as cli
from histopilot.api import create_app
from histopilot.api.demo_mode import PROJECT_REFUSED
from histopilot.api.route_classes import READ, ROUTE_CLASSES
from histopilot.application.blca_demo import DEMO_ID
from histopilot.config import Settings


@pytest.fixture
def static(tmp_path):
    folder = tmp_path / "static"
    folder.mkdir()
    (folder / "index.html").write_text("<!doctype html><title>HistoPilot test</title>")
    return folder


@pytest.fixture
def study(tmp_path, static):
    """A project made by an ordinary local service, as a study's own folder would be."""
    workspace = tmp_path / "author"
    (workspace / "projects").mkdir(parents=True)
    settings = Settings(workspace=workspace, static_dir=static)
    with TestClient(create_app(settings), base_url="http://127.0.0.1:8787") as client:
        client.headers["X-HistoPilot-Token"] = client.get(f"{API}/session").json()["token"]
        created = client.post(
            f"{API}/projects",
            json={
                "name": "Lung study",
                "description": "LUAD vs LUSC on public data.",
                "storagePath": str(workspace / "projects" / "lung"),
            },
        )
        assert created.status_code == 201, created.text
    return created.json()["id"], workspace / "projects" / "lung"


def fingerprint(folder):
    return {
        str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


@pytest.fixture
def demo(tmp_path, static, study):
    identity, folder = study
    # What cli._serve_demo does: copy the packaged project into the throwaway workspace.
    workspace = tmp_path / "demo-workspace"
    copy = workspace / "demo-projects" / folder.name
    shutil.copytree(folder, copy)
    settings = Settings(
        workspace=workspace,
        static_dir=static,
        host="0.0.0.0",
        port=7860,
        demo=True,
        demo_projects=(copy,),
    )
    with public_client(settings) as client:
        client.headers["X-HistoPilot-Token"] = client.get(f"{API}/session").json()["token"]
        yield client, identity


def test_the_packaged_project_is_listed_without_a_path_and_reads_as_read_only(demo):
    client, identity = demo
    listed = client.get(f"{API}/projects").json()
    assert listed["demoOnly"] is True and listed["defaultStoragePath"] == ""
    assert [project["id"] for project in listed["projects"]] == [identity, DEMO_ID]
    packaged = listed["projects"][0]
    assert packaged["name"] == "Lung study" and packaged["storagePath"] == ""
    assert "sources" not in packaged
    workspace = client.get(f"{API}/projects/{identity}/workspace")
    assert workspace.status_code == 200
    assert workspace.json()["readOnly"] is True
    assert workspace.json()["project"]["id"] == identity


def test_a_packaged_project_answers_its_reads_alone(demo):
    client, identity = demo
    api = sorted({path for path, _ in declared(client.app.routes) if path.startswith("/api/")})
    reads = refused_reads = writes = 0
    for template in api:
        path = concrete(template, identity)
        for method in METHODS:
            lookup = "GET" if method in {"HEAD", "OPTIONS"} else method
            kind = ROUTE_CLASSES.get((lookup, template))
            response = client.request(method, path)
            if not template.startswith(f"{API}/projects/{{identity}}"):
                continue  # Routes outside the project keep the demo's own rules (test_demo_mode).
            if kind == READ and template in PROJECT_REFUSED:
                assert refused(response, "DEMO_MODE_ROUTE_UNAVAILABLE"), (method, path)
                refused_reads += 1
            elif kind == READ:
                # Reached the route: whatever it answers, the boundary did not refuse it.
                assert not (
                    response.status_code == 403
                    and response.request.method != "HEAD"
                    and response.json().get("code", "").startswith("DEMO_MODE")
                ), (method, path, response.text)
                reads += 1
            elif method in WRITES and kind is not None:
                assert refused(response, "DEMO_MODE_READ_ONLY"), (method, path, response.text)
                writes += 1
    assert reads > 100 and refused_reads >= len(PROJECT_REFUSED) and writes > 50
    # Another project's reads, and the machine's, stay refused.
    assert refused(client.get(f"{API}/projects/another/workspace"), "DEMO_MODE_ROUTE_UNAVAILABLE")
    for path in ("/filesystem/roots", "/system", "/tokens", "/task-center/snapshot", "/metrics"):
        assert refused(client.get(f"{API}{path}"), "DEMO_MODE_ROUTE_UNAVAILABLE"), path


def test_every_project_refusal_names_a_read_route():
    assert set(PROJECT_REFUSED) <= {
        template for (method, template), kind in ROUTE_CLASSES.items() if kind == READ
    }


def test_the_token_is_still_required_for_a_packaged_project(tmp_path, static, study):
    identity, folder = study
    copy = tmp_path / "demo-workspace" / "demo-projects" / folder.name
    shutil.copytree(folder, copy)
    settings = Settings(
        workspace=tmp_path / "demo-workspace", static_dir=static, demo=True, demo_projects=(copy,)
    )
    with public_client(settings) as client:
        answer = client.get(f"{API}/projects/{identity}/workspace")
        assert answer.status_code == 401
        assert answer.json()["code"] == "SESSION_TOKEN_REQUIRED"


def test_packaged_projects_need_demo_mode(tmp_path):
    with pytest.raises(ValueError, match="demo mode only"):
        Settings(workspace=tmp_path, demo_projects=(tmp_path / "study",))


@pytest.fixture
def served(monkeypatch, static):
    runs = []

    def run(app, **options):
        settings = app.state.settings
        runs.append(
            {
                "settings": settings,
                "copies": [folder.is_dir() for folder in settings.demo_projects],
            }
        )

    monkeypatch.setattr("uvicorn.run", run)
    monkeypatch.setattr(
        cli, "_ensure_task_center_runner", lambda: pytest.fail("No runner in a demo")
    )
    monkeypatch.setattr(cli, "Settings", lambda **values: Settings(static_dir=static, **values))
    return runs


def test_serve_demo_copies_the_packaged_projects_and_leaves_them_untouched(served, tmp_path, study):
    _, folder = study
    packaged = tmp_path / "packaged"
    shutil.copytree(folder, packaged / "lung")
    (packaged / "notes").mkdir()  # not a project: ignored
    before = fingerprint(packaged)
    result = CliRunner().invoke(
        cli.app, ["serve", "--demo", "--no-browser", "--demo-projects", str(packaged)]
    )
    assert result.exit_code == 0, result.output
    assert "and 1 packaged project to anyone" in result.output
    [run] = served
    settings = run["settings"]
    assert [folder.name for folder in settings.demo_projects] == ["lung"]
    assert settings.demo_projects[0].parent == settings.workspace / "demo-projects"
    assert run["copies"] == [True]
    assert fingerprint(packaged) == before
    assert not settings.workspace.exists()  # removed with the rest of the scratch folder


def test_serve_demo_refuses_a_folder_without_projects_and_needs_demo(served, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    result = CliRunner().invoke(
        cli.app, ["serve", "--demo", "--no-browser", "--demo-projects", str(empty)]
    )
    assert result.exit_code != 0
    assert "no HistoPilot project folders" in result.output
    alone = CliRunner().invoke(cli.app, ["serve", "--no-browser", "--demo-projects", str(empty)])
    assert alone.exit_code == 1
    assert "--demo-projects needs --demo" in alone.output
    assert served == []
