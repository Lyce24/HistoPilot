"""Container mode: the service listens on a container's interfaces and trusts nothing new.

Docker forwards a port published on the host's loopback to the container's own address, so
the service must listen beyond 127.0.0.1 there, and requests arrive from the bridge rather
than from loopback. Trust never came from the peer address: the Host and Origin checks, the
session token and the optional sign-in decide, and container mode leaves each one as it was.
"""

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

import histopilot.cli as cli
from histopilot.api import create_app, login
from histopilot.cli import app
from histopilot.config import Settings, load_settings

BASE = "http://127.0.0.1:8787"
# What a request forwarded from a published port looks like: it comes from the bridge.
BRIDGE_PEER = ("172.18.0.1", 41000)


def test_only_container_mode_listens_beyond_loopback(tmp_path):
    assert Settings().listen_host == "127.0.0.1"
    assert Settings(host="localhost").listen_host == "localhost"
    inside = Settings(container=True)
    assert inside.listen_host == "0.0.0.0"
    # Browsers are still told to open the loopback address, and must name it as the Host.
    assert inside.host == "127.0.0.1"
    with pytest.raises(ValueError, match="loopback"):
        Settings(host="0.0.0.0", container=True)
    config = tmp_path / "config.toml"
    config.write_text("[server]\nport = 8787\n")
    assert load_settings(config).container is False
    assert load_settings(config, container=True).container is True
    # How the process is deployed is not a configuration file setting.
    config.write_text("[server]\ncontainer = true\n")
    with pytest.raises(ValueError, match="Unknown server/storage setting"):
        load_settings(config)


@pytest.fixture
def serve(tmp_path, monkeypatch):
    """Run `histopilot serve` up to the point uvicorn starts; return the bind and the output."""
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<!doctype html>")
    original = cli.load_settings
    monkeypatch.setattr(
        cli, "load_settings", lambda *a, **k: replace(original(*a, **k), static_dir=static)
    )
    calls = []
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: calls.append(kwargs))
    monkeypatch.delenv("HISTOPILOT_CONTAINER", raising=False)
    config = tmp_path / "config.toml"
    config.write_text('[storage]\nworkspace = "workspace"\n')

    def run(*options, env=None):
        calls.clear()
        arguments = ["serve", "--config", str(config), "--no-browser", "--no-runner", *options]
        result = CliRunner().invoke(app, arguments, env=env)
        assert result.exit_code == 0, result.output
        assert len(calls) == 1
        return calls[0]["host"], result.output

    return run


@pytest.mark.parametrize(
    "options,env,listen",
    [
        ((), {}, "127.0.0.1"),
        ((), {"HISTOPILOT_CONTAINER": "0"}, "127.0.0.1"),
        ((), {"HISTOPILOT_CONTAINER": "1"}, "0.0.0.0"),
        (("--container",), {}, "0.0.0.0"),
    ],
)
def test_serve_listens_on_every_interface_only_in_container_mode(serve, options, env, listen):
    host, output = serve(*options, env=env)
    assert host == listen
    assert "Browser    http://127.0.0.1:8787" in output
    assert ("Container  Listening on all" in output) is (listen == "0.0.0.0")


def test_container_mode_says_when_sign_in_is_off_or_no_container_is_found(
    serve, monkeypatch, tmp_path
):
    marker = tmp_path / ".dockerenv"
    marker.touch()
    monkeypatch.setattr(cli, "CONTAINER_MARKERS", (str(marker),))
    _, output = serve("--container")
    assert "Sign-in is off" in output
    assert "does not look like a container" not in output
    _, output = serve("--container", "--login")
    assert "Sign-in is off" not in output
    assert f"Browser    {BASE}/?login=" in output
    marker.unlink()
    _, output = serve("--container", "--login")
    assert "does not look like a container" in output


@pytest.fixture
def forwarded(tmp_path):
    """A container-mode service as a request forwarded from the published port reaches it."""
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<!doctype html>")

    def connect(**options):
        settings = Settings(
            workspace=tmp_path / "workspace", static_dir=static, container=True, **options
        )
        return TestClient(
            create_app(settings), base_url=BASE, client=BRIDGE_PEER, follow_redirects=False
        )

    return connect


def test_a_forwarded_request_must_still_name_the_loopback_host_and_port(forwarded):
    with forwarded() as http:
        assert http.get("/api/v1/health").status_code == 200
        # The peer address is not a credential either way: the same checks as on loopback.
        assert http.get("/api/v1/session").status_code == 200
        assert http.get("/api/v1/projects").status_code == 401
        # The container's own address or name, and a port other than the service's own,
        # are refused: publish the same port on both sides, on the host's loopback.
        for host in ("172.18.0.2:8787", "histopilot:8787", "0.0.0.0:8787", "localhost:9000"):
            refused = http.get("/api/v1/session", headers={"Host": host})
            assert refused.status_code == 400 and refused.json()["code"] == "HOST_REFUSED"
        refused = http.get("/api/v1/session", headers={"Origin": "http://172.18.0.2:8787"})
        assert refused.status_code == 403 and refused.json()["code"] == "ORIGIN_REFUSED"


def test_sign_in_keeps_the_session_from_anything_else_that_reaches_the_port(forwarded):
    with forwarded(login=True) as http:
        refused = http.get("/api/v1/session")
        assert refused.status_code == 401 and refused.json()["code"] == "LOGIN_REQUIRED"
        secret = login.read_secret(8787)
        signed = http.get("/api/v1/session", headers={"X-HistoPilot-Login": secret})
        assert signed.status_code == 200
