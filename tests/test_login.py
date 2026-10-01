"""Opt-in sign-in: /session answers only the printed link's cookie or the secret."""

import pytest
from fastapi.testclient import TestClient
from support.cli import InProcessTransport

from histopilot.api import create_app, login
from histopilot.client import Client
from histopilot.config import Settings, load_settings

BASE = "http://127.0.0.1:8787"


@pytest.fixture
def signed(tmp_path):
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<!doctype html>")
    settings = Settings(workspace=tmp_path / "workspace", static_dir=static, login=True)
    with TestClient(create_app(settings), base_url=BASE, follow_redirects=False) as http:
        yield http, login.read_secret(8787)


def test_without_login_the_session_answers_as_before(tmp_path):
    static = tmp_path / "static"
    static.mkdir()
    settings = Settings(workspace=tmp_path / "workspace", static_dir=static)
    with TestClient(create_app(settings), base_url=BASE) as http:
        assert http.get("/api/v1/session").status_code == 200
    assert login.read_secret(8787) is None


def test_the_session_needs_the_link_or_the_secret(signed):
    http, secret = signed
    refused = http.get("/api/v1/session")
    assert refused.status_code == 401 and refused.json()["code"] == "LOGIN_REQUIRED"
    assert http.get("/api/v1/session", headers={"X-HistoPilot-Login": secret}).status_code == 200
    assert http.get("/api/v1/health").status_code == 200


def test_opening_the_link_signs_the_browser_in(signed):
    http, secret = signed
    wrong = http.get("/?login=wrong")
    assert wrong.status_code == 303 and "set-cookie" not in wrong.headers
    opened = http.get(f"/?login={secret}")
    assert opened.status_code == 303 and opened.headers["location"] == "/"
    cookie = opened.headers["set-cookie"]
    assert cookie.startswith("histopilot-login-8787=") and "HttpOnly" in cookie
    assert "Path=/api/v1/session" in cookie and "SameSite=strict" in cookie
    http.cookies.set("histopilot-login-8787", login.cookie_value(secret), path="/api/v1/session")
    assert http.get("/api/v1/session").status_code == 200


def test_the_secret_is_private_and_the_client_signs_in_from_it(signed):
    http, secret = signed
    assert oct(login.secret_path(8787).stat().st_mode & 0o777) == "0o600"
    client = Client(BASE, transport=InProcessTransport(http))
    assert client.session()["token"]
    rotated = login.rotate_secret(8787)
    assert rotated != secret and login.read_secret(8787) == rotated


def test_login_is_read_from_the_server_settings(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text("[server]\nlogin = true\n")
    assert load_settings(config).login is True
    config.write_text('[server]\nlogin = "yes"\n')
    with pytest.raises(ValueError):
        load_settings(config)


def test_a_rotated_secret_takes_effect_without_a_restart(signed):
    http, secret = signed
    rotated = login.rotate_secret(8787)
    assert http.get("/api/v1/session", headers={"X-HistoPilot-Login": secret}).status_code == 401
    assert http.get("/api/v1/session", headers={"X-HistoPilot-Login": rotated}).status_code == 200
    assert "set-cookie" not in http.get(f"/?login={secret}").headers
    assert http.get(f"/?login={rotated}").headers["set-cookie"].startswith("histopilot-login-8787=")


def test_a_cookie_that_is_not_ascii_is_refused_not_an_error(signed):
    http, _ = signed
    refused = http.get(
        "/api/v1/session", headers={"cookie": "histopilot-login-8787=caf\u00e9".encode()}
    )
    assert refused.status_code == 401 and refused.json()["code"] == "LOGIN_REQUIRED"
