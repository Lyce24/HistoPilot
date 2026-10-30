"""A demo service (`histopilot serve --demo`) answers the synthetic demo's reads, to anyone,
and refuses everything else; a service started without it keeps the loopback boundary."""

import os
import re
import signal
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from typer.testing import CliRunner

import histopilot.cli as cli
from histopilot.api import create_app
from histopilot.api.demo_mode import DEMO_ROUTES
from histopilot.application.blca_demo import DEMO_ID, load_demo
from histopilot.config import Settings

API = "/api/v1"
# What a Hugging Face Space's proxy hands the container: the Space's own Host, a client
# address that is not loopback, and forwarding headers.
PUBLIC_URL = "https://someone-histopilot.hf.space"
PUBLIC_CLIENT = ("203.0.113.7", 41234)
FORWARDED = {
    "X-Forwarded-For": "198.51.100.23",
    "X-Forwarded-Proto": "https",
    "X-Forwarded-Host": "someone-histopilot.hf.space",
}
METHODS = ("GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS")
WRITES = ("POST", "PUT", "PATCH", "DELETE")
FRAMING = "frame-ancestors 'self' https://huggingface.co"


@pytest.fixture
def static(tmp_path):
    folder = tmp_path / "static"
    folder.mkdir()
    (folder / "index.html").write_text("<!doctype html><title>HistoPilot test</title>")
    return folder


@pytest.fixture
def demo_settings(tmp_path, static):
    return Settings(
        workspace=tmp_path / "workspace", static_dir=static, host="0.0.0.0", port=7860, demo=True
    )


def public_client(settings):
    return TestClient(
        create_app(settings), base_url=PUBLIC_URL, client=PUBLIC_CLIENT, headers=FORWARDED
    )


@pytest.fixture
def demo(demo_settings):
    with public_client(demo_settings) as client:
        client.headers["X-HistoPilot-Token"] = client.get(f"{API}/session").json()["token"]
        yield client


def concrete(template: str, identity: str = DEMO_ID) -> str:
    """A request path for a route: every parameter gets a sample value, and the project is
    the demo's own unless named, so the demo's other routes are tried too."""

    def value(match):
        return identity if match.group(1) == "identity" else f"sample-{match.group(1)}"

    return re.sub(r"\{([^}:]+)(?::[^}]+)?\}", value, template)


def declared(routes, prefix: str = ""):
    """(path, methods) of every route in app.routes, through routers FastAPI nests there."""
    for route in routes:
        nested = getattr(route, "original_router", None)
        if nested is not None:
            context = getattr(route, "include_context", None)
            yield from declared(nested.routes, prefix + (getattr(context, "prefix", "") or ""))
        elif isinstance(route, APIRoute):
            yield prefix + route.path, route.methods


def refused(response, code: str) -> bool:
    if response.status_code != 403:
        return False
    # A HEAD answer has no body to carry the code.
    return response.request.method == "HEAD" or response.json()["code"] == code


def test_every_route_outside_the_allowlist_is_refused_for_every_method(demo):
    routes = list(declared(demo.app.routes))
    api = sorted({path for path, _ in routes if path.startswith("/api/")})
    # The whole service is mounted and the boundary refuses it, so a route added later is
    # refused in demo mode until someone allowlists it on purpose.
    schema = {path for path in demo.app.openapi()["paths"] if path.startswith("/api/")}
    # The schema writes {slide_id:path} as {slide_id}.
    assert {re.sub(r"\{(\w+):\w+\}", r"{\1}", path) for path in api} == schema
    assert len(api) > 150
    assert ("/{path:path}", {"GET"}) in routes
    allowed = {concrete(path) for path in api} & DEMO_ROUTES.keys()
    assert allowed == DEMO_ROUTES.keys(), "Every allowlisted path names a real route"
    checked = 0
    for template in api:
        for path in {concrete(template), concrete(template, "another-project")}:
            for method in METHODS:
                response = demo.request(method, path)
                if method in WRITES:
                    assert refused(response, "DEMO_MODE_READ_ONLY"), (method, path, response.text)
                elif path not in DEMO_ROUTES:
                    assert refused(response, "DEMO_MODE_ROUTE_UNAVAILABLE"), (method, path)
                elif method != "GET":
                    # Allowlisted reads answer GET alone; the router refuses the rest.
                    assert response.status_code == 405, (method, path)
                checked += 1
    assert checked > 2000
    # Paths that are no route, or a route spelled another way, are refused before routing.
    for path in (
        "/api",
        f"{API}/does-not-exist",
        f"{API}/projects/",
        f"{API}//projects",
        f"{API}/projects/{DEMO_ID}/workspace/",
        f"{API}/projects/{DEMO_ID}/workspace%3F",
        "/api/v2/projects",
    ):
        assert refused(demo.get(path), "DEMO_MODE_ROUTE_UNAVAILABLE"), path
    # The UI's own files are read-only too.
    for method in WRITES:
        for path in ("/", "/index.html", "/assets/app.js"):
            assert refused(demo.request(method, path), "DEMO_MODE_READ_ONLY"), (method, path)


def test_the_demo_reads_work_through_a_proxy_from_anywhere(demo, demo_settings):
    page = demo.get("/")
    assert page.status_code == 200
    assert "HistoPilot test" in page.text
    assert page.headers["content-security-policy"] == FRAMING
    assert "x-frame-options" not in page.headers
    # The Space's page on huggingface.co loads the demo in a frame: a cross-site navigation.
    framed = demo.get("/", headers={"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Dest": "iframe"})
    assert framed.status_code == 200
    assert demo.get("/?project=blca-demo-v1").status_code == 200

    health = demo.get(f"{API}/health")
    assert health.status_code == 200 and health.json()["status"] == "ok"
    listed = demo.get(f"{API}/projects")
    assert listed.status_code == 200
    assert listed.headers["cache-control"] == "no-store"
    body = listed.json()
    assert body == {
        "projects": [body["projects"][0]],
        "defaultStoragePath": "",
        "demoOnly": True,
    }
    assert body["projects"][0]["id"] == DEMO_ID and body["projects"][0]["storagePath"] == ""
    workspace = demo.get(f"{API}/projects/{DEMO_ID}/workspace")
    assert workspace.status_code == 200
    assert workspace.json() == load_demo()
    for response in (listed, workspace):
        assert str(demo_settings.workspace.parent) not in response.text
        assert response.headers["content-security-policy"] == FRAMING

    # No request attribute is trusted, or needed: any Host, Origin or fetch context reads
    # the same public content, and no other origin is ever granted a cross-origin read.
    for headers in (
        {"Host": "10.0.0.5:7860"},
        {"Host": "localhost:7860"},
        {"Origin": "https://huggingface.co", "Sec-Fetch-Site": "cross-site"},
        {"Origin": "null"},
    ):
        answer = demo.get(f"{API}/projects", headers=headers)
        assert answer.status_code == 200, headers
        assert answer.json()["demoOnly"] is True
        assert "access-control-allow-origin" not in answer.headers
    preflight = demo.options(
        f"{API}/projects",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"},
    )
    assert preflight.status_code == 405
    assert "access-control-allow-origin" not in preflight.headers


def test_the_token_is_still_required_and_agent_tokens_reach_nothing(demo_settings):
    with public_client(demo_settings) as client:
        for path in (f"{API}/projects", f"{API}/projects/{DEMO_ID}/workspace"):
            missing = client.get(path)
            assert missing.status_code == 401
            assert missing.json()["code"] == "SESSION_TOKEN_REQUIRED"
            agent = client.get(path, headers={"X-HistoPilot-Token": "hpt_" + "x" * 40})
            assert agent.json()["code"] == "SESSION_TOKEN_REQUIRED"
        assert client.get(f"{API}/session").status_code == 200


def test_a_demo_service_writes_nothing_and_reads_no_machine_state(demo, demo_settings, tmp_path):
    for method, path, body in (
        ("POST", f"{API}/projects", {"name": "Mine", "storagePath": str(tmp_path / "mine")}),
        ("POST", f"{API}/projects/open", {"path": str(tmp_path)}),
        ("POST", f"{API}/filesystem/directories", {"parentPath": str(tmp_path), "name": "x"}),
        ("POST", f"{API}/tokens", {"name": "agent"}),
        ("PUT", f"{API}/projects/{DEMO_ID}/name", {"name": "Mine", "expectedName": "BLCA demo"}),
        ("POST", f"{API}/task-center/runner/start", None),
    ):
        assert refused(demo.request(method, path, json=body), "DEMO_MODE_READ_ONLY")
    for path in (
        "/filesystem/roots?purpose=storage",
        f"/filesystem/list?path={tmp_path}",
        "/system",
        "/system/compute",
        "/version",
        "/tokens",
        "/agent-requests",
        "/task-center/capacity",
        "/task-center/snapshot",
        f"/projects/{DEMO_ID}/storage",
        f"/projects/{DEMO_ID}/morphology/image",
        f"/projects/{DEMO_ID}/operations/sources",
    ):
        assert refused(demo.get(f"{API}{path}"), "DEMO_MODE_ROUTE_UNAVAILABLE"), path
    assert not (tmp_path / "mine").exists()
    assert not list(demo_settings.workspace.rglob("histopilot-project.json"))
    # This test's Task Center state folder (tests/conftest.py) was never even created.
    assert not Path(os.environ["HISTOPILOT_STATE_DIR"]).exists()


def test_demo_settings_accept_any_host_and_nothing_that_reaches_this_machine(tmp_path):
    assert Settings(host="0.0.0.0", demo=True).host == "0.0.0.0"
    with pytest.raises(ValueError, match="loopback"):
        Settings(host="0.0.0.0")
    with pytest.raises(ValueError, match="loopback"):
        Settings(host="0.0.0.0", demo=False)
    for options in ({"data_roots": (tmp_path,)}, {"login": True}, {"dev": True}):
        with pytest.raises(ValueError, match="only the synthetic demo"):
            Settings(demo=True, **options)
    with pytest.raises(ValueError, match="true or false"):
        Settings(demo="yes")
    with pytest.raises(ValueError, match="nonempty"):
        Settings(host=" ", demo=True)


def test_normal_mode_keeps_the_loopback_boundary(tmp_path, static):
    data = tmp_path / "data"
    data.mkdir()
    settings = Settings(workspace=tmp_path / "workspace", data_roots=(data,), static_dir=static)
    # Through a proxy, with a foreign Host, a normal service answers nothing.
    with public_client(settings) as client:
        for path in ("/", f"{API}/health", f"{API}/session", f"{API}/projects"):
            answer = client.get(path)
            assert answer.status_code == 400
            assert answer.json()["code"] == "HOST_REFUSED"
    with TestClient(create_app(settings), base_url="http://127.0.0.1:8787") as client:
        page = client.get("/")
        assert page.headers["x-frame-options"] == "DENY"
        assert "content-security-policy" not in page.headers
        assert client.get("/", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
        client.headers["X-HistoPilot-Token"] = client.get(f"{API}/session").json()["token"]
        listed = client.get(f"{API}/projects").json()
        assert "demoOnly" not in listed
        assert listed["defaultStoragePath"] == str(settings.workspace)
        assert client.get(f"{API}/filesystem/roots").status_code == 200
        assert client.get(f"{API}/version").status_code == 200
        (settings.workspace / "projects").mkdir(parents=True)
        created = client.post(
            f"{API}/projects",
            json={"name": "Study", "storagePath": str(settings.workspace / "projects" / "s")},
        )
        assert created.status_code == 201, created.text


@pytest.fixture
def served(monkeypatch, static):
    """`histopilot serve` up to uvicorn, recording what it would serve and how."""
    runs = []

    def run(app, **options):
        runs.append(
            {
                "app": app,
                "options": options,
                "state": os.environ.get("HISTOPILOT_STATE_DIR"),
                "workspace_exists": app.state.settings.workspace.is_dir(),
            }
        )

    def runner():
        pytest.fail("A demo service never starts the Task Center runner")

    monkeypatch.setattr("uvicorn.run", run)
    monkeypatch.setattr(cli, "_ensure_task_center_runner", runner)
    # The checkout under test may have no built UI; the demo reads its index from here.
    monkeypatch.setattr(cli, "Settings", lambda **values: Settings(static_dir=static, **values))
    return runs


@pytest.mark.parametrize("switch", ["flag", "environment"])
def test_serve_demo_uses_a_throwaway_workspace_and_no_runner(served, monkeypatch, tmp_path, switch):
    # The person's own configuration and state are never read.
    home = tmp_path / "home"
    (home / ".histopilot").mkdir(parents=True)
    (home / ".histopilot" / "config.toml").write_text('[storage]\ndata_roots = ["/"]\n')
    monkeypatch.setenv("HOME", str(home))
    before = os.environ["HISTOPILOT_STATE_DIR"]
    arguments = ["serve", "--host", "0.0.0.0", "--port", "7860", "--no-browser"]
    if switch == "flag":
        arguments.append("--demo")
    else:
        monkeypatch.setenv("HISTOPILOT_DEMO", "1")

    result = CliRunner().invoke(cli.app, arguments)

    assert result.exit_code == 0, result.output
    assert "public demo, read-only" in result.output
    [run] = served
    settings = run["app"].state.settings
    assert settings.demo is True
    assert (settings.host, settings.port) == ("0.0.0.0", 7860)
    assert settings.data_roots == () and settings.login is False
    assert run["options"] == {
        "host": "0.0.0.0",
        "port": 7860,
        "workers": 1,
        "log_config": None,  # records route through histopilot/observability
        "access_log": False,
    }
    scratch = settings.workspace.parent
    assert scratch.name.startswith("histopilot-demo-") and run["workspace_exists"]
    assert run["state"] == str(scratch / "state")
    # Both folders are gone once the service stops, and the state setting is restored.
    assert not scratch.exists()
    assert os.environ["HISTOPILOT_STATE_DIR"] == before


def test_serve_demo_removes_its_folders_when_stopped_by_sigterm(served, monkeypatch):
    seen = {}

    def run(app, **options):
        seen["scratch"] = app.state.settings.workspace.parent
        # Uvicorn re-raises SIGTERM to the previous handler once it has shut down.
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler), "SIGTERM would end the process before the folders go"
        handler(signal.SIGTERM, None)

    monkeypatch.setattr("uvicorn.run", run)
    before = signal.getsignal(signal.SIGTERM)
    result = CliRunner().invoke(cli.app, ["serve", "--demo", "--no-browser"])
    assert result.exit_code == 128 + signal.SIGTERM
    assert not seen["scratch"].exists()
    assert signal.getsignal(signal.SIGTERM) is before


@pytest.mark.parametrize(
    "option",
    [["--data-root", "."], ["--workspace", "."], ["--config", "x.toml"], ["--dev"], ["--login"]],
)
def test_serve_demo_refuses_options_that_reach_this_machine(served, option):
    result = CliRunner().invoke(cli.app, ["serve", "--demo", "--no-browser", *option])
    assert result.exit_code == 1
    assert "--demo serves only the synthetic demo and takes no " + option[0] in result.output
    assert served == []


def test_serve_without_demo_is_unchanged(served, monkeypatch, tmp_path, static):
    monkeypatch.setenv("HISTOPILOT_DEMO", "0")
    original = cli.load_settings
    monkeypatch.setattr(
        cli, "load_settings", lambda *a, **k: replace(original(*a, **k), static_dir=static)
    )
    config = tmp_path / "config.toml"
    config.write_text('[storage]\nworkspace = "state"\n')
    result = CliRunner().invoke(
        cli.app, ["serve", "--config", str(config), "--no-browser", "--no-runner"]
    )
    assert result.exit_code == 0, result.output
    [run] = served
    assert run["app"].state.settings.demo is False
    assert run["app"].state.settings.workspace == tmp_path / "state"
    assert "public demo" not in result.output
