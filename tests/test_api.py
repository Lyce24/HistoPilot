"""Local service authority, persistence, and browser/filesystem boundary tests."""

import json
import sqlite3
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from histopilot.api import create_app
from histopilot.config import Settings

BASE_URL = "http://127.0.0.1:8787"
API = "/api/v1"


@pytest.fixture
def settings(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<!doctype html><title>HistoPilot test</title>")
    return Settings(workspace=tmp_path / "workspace", data_roots=(data,), static_dir=static)


def authenticated_client(settings):
    client = TestClient(create_app(settings), base_url=BASE_URL)
    return client


def authenticate(client):
    token = client.get(f"{API}/session").json()["token"]
    client.headers["X-HistoPilot-Token"] = token
    return token


@pytest.fixture
def client(settings):
    with authenticated_client(settings) as current:
        authenticate(current)
        yield current


def project(client, settings, name="Study"):
    # Project folders are created inside an existing storage folder.
    (settings.workspace / "projects").mkdir(parents=True, exist_ok=True)
    response = client.post(
        f"{API}/projects",
        json={"name": name, "storagePath": str(settings.workspace / "projects" / name)},
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_health_is_minimal_and_sensitive_routes_require_token(settings):
    with authenticated_client(settings) as client:
        health = client.get(f"{API}/health")
        assert health.status_code == 200
        assert set(health.json()) == {"status", "version", "executionEnabled"}
        assert health.headers["cache-control"] == "no-store"
        assert str(settings.workspace) not in health.text
        for route in ("projects", "filesystem/roots", "system", "system/compute"):
            assert client.get(f"{API}/{route}").status_code == 401
        assert client.post(f"{API}/projects", json={}).status_code == 401
        assert (
            client.post(
                f"{API}/filesystem/directories",
                json={"parentPath": str(settings.data_roots[0]), "name": "new"},
            ).status_code
            == 401
        )


def test_system_distinguishes_implemented_workers_from_trident_readiness(client, monkeypatch):
    monkeypatch.setattr("histopilot.api.app.discover_runtime", lambda: {"available": False})
    # tmux hosts the Task Center runner; /system reports it with a plain PATH lookup.
    monkeypatch.setattr(
        "histopilot.api.app.shutil.which", lambda name: "/usr/bin/tmux" if name == "tmux" else None
    )
    authenticate(client)
    response = client.get(f"{API}/system")
    assert response.status_code == 200
    report = response.json()
    assert report["workers"]["executionEnabled"] is False
    assert report["workers"]["nativeExecutionImplemented"] is True
    assert report["workers"]["tmuxAvailable"] is True
    assert "check runtime readiness" in report["workers"]["status"]
    assert "not connected" not in report["workers"]["status"]
    assert report["diagnostics"]["compute"]["scope"] == "control-service"
    # Other tests in this process may import Torch; the field reports what is loaded.
    assert report["control"] == {
        "process": "control-service",
        "torchImported": "torch" in sys.modules,
    }


def test_compute_telemetry_is_authenticated_and_separate_from_runtime_probes(client, monkeypatch):
    def unexpected_runtime_probe():
        pytest.fail("Live compute polling must not probe or import worker runtimes")

    monkeypatch.setattr("histopilot.api.app.discover_runtime", unexpected_runtime_probe)
    monkeypatch.setattr("histopilot.api.app.system_report", unexpected_runtime_probe)
    snapshot = {"sampledAt": "2026-09-12T00:00:00+00:00", "cpu": {"utilizationPercent": None}}
    monkeypatch.setattr("histopilot.api.app.ComputeSampler.snapshot", lambda _self: snapshot)
    response = client.get(f"{API}/system/compute")
    assert response.status_code == 200
    assert response.json() == snapshot
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize(
    "headers,status",
    [
        ({"Origin": "https://evil.example"}, 403),
        ({"Origin": "null"}, 403),
        ({"Origin": "http://localhost:9999"}, 403),
        ({"Host": "evil.example:8787"}, 400),
        ({"Host": "127.0.0.1.evil.example:8787"}, 400),
        ({"Host": "localhost:9999"}, 400),
        ({"Sec-Fetch-Site": "cross-site"}, 403),
        ({"Sec-Fetch-Site": "same-site"}, 403),
    ],
)
def test_untrusted_browser_cannot_acquire_session(client, headers, status):
    assert client.get(f"{API}/session", headers=headers).status_code == status
    assert client.get(f"{API}/filesystem/roots", headers=headers).status_code == status


def test_dev_origin_is_explicit_and_cors_is_not_wildcard(settings):
    origin = {"Origin": "http://localhost:5173"}
    with authenticated_client(settings) as production:
        assert production.get(f"{API}/session", headers=origin).status_code == 403
    with authenticated_client(replace(settings, dev=True)) as development:
        assert development.get(f"{API}/session", headers=origin).status_code == 200
        preflight = development.options(
            f"{API}/projects",
            headers={
                **origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type,x-histopilot-token",
            },
        )
        assert preflight.status_code == 200
        assert preflight.headers["access-control-allow-origin"] == origin["Origin"]
        assert development.get(f"{API}/projects", headers=origin).status_code == 401


def test_restart_preserves_records_rotates_token_and_keeps_original_files(settings):
    original = settings.data_roots[0] / "original.svs"
    original.write_bytes(b"synthetic file sentinel")
    with authenticated_client(settings) as first:
        old_token = authenticate(first)
        created = project(first, settings)
        source = first.post(
            f"{API}/projects/{created['id']}/sources",
            json={"path": str(settings.data_roots[0]), "role": "slides"},
        ).json()
        assert source["importStatus"] == "not-imported"
    with authenticated_client(settings) as second:
        assert (
            second.get(f"{API}/projects", headers={"X-HistoPilot-Token": old_token}).status_code
            == 401
        )
        authenticate(second)
        listed = second.get(f"{API}/projects").json()["projects"]
        assert created["id"] in {row["id"] for row in listed}
        state = second.get(f"{API}/projects/{created['id']}/workspace").json()
        assert state["sources"] == [source]
    assert original.read_bytes() == b"synthetic file sentinel"
    assert not (settings.workspace / "original.svs").exists()
    with sqlite3.connect(settings.workspace / "histopilot.db") as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("SELECT version FROM schema_version").fetchone()[0] == 1


def test_filesystem_is_root_confined_and_never_serves_file_contents(client, settings, tmp_path):
    root = settings.data_roots[0]
    inside = root / "slides"
    inside.mkdir()
    (inside / "slide.svs").write_bytes(b"not a real slide")
    outside = tmp_path / "private"
    outside.mkdir()
    (outside / "secret.txt").write_text("do not expose")
    (root / "escape").symlink_to(outside, target_is_directory=True)
    sources = f"{API}/projects/{project(client, settings)['id']}/sources"
    roots = client.get(f"{API}/filesystem/roots").json()["roots"]
    assert roots == [{"path": str(root), "name": "data"}]
    listing = client.get(f"{API}/filesystem/list", params={"path": str(root)}).json()
    assert listing["parent"] is None
    assert all(entry["name"] != "escape" for entry in listing["entries"])
    for path in (outside, root / "escape", root / ".." / "private"):
        assert client.get(f"{API}/filesystem/list", params={"path": str(path)}).status_code == 403
        assert client.post(sources, json={"path": str(path), "role": "slides"}).status_code == 403
    assert client.get(f"{API}/filesystem/list", params={"path": "."}).status_code == 400
    assert (
        client.post(sources, json={"path": str(inside / "slide.svs"), "role": "slides"}).status_code
        == 400
    )
    assert (
        client.get(f"{API}/filesystem/read", params={"path": str(inside / "slide.svs")}).status_code
        == 404
    )
    assert client.get(f"{API}/filesystem/list", params={"path": str(inside)}).json()[
        "parent"
    ] == str(root)


def test_directory_listing_is_bounded(client, settings):
    for index in range(205):
        (settings.data_roots[0] / f"slide-{index}.svs").touch()
    result = client.get(
        f"{API}/filesystem/list", params={"path": str(settings.data_roots[0])}
    ).json()
    assert result["truncated"] is True
    assert len(result["entries"]) == 200


def test_spa_fallback_does_not_swallow_api_or_expose_files(client, settings, tmp_path):
    assert client.get("/").status_code == 200
    assert client.get("/experiments").status_code == 200
    assert client.get("/api/v1/unknown").status_code == 404
    assert client.get("/assets/missing.js").status_code == 404
    private = tmp_path / "private.txt"
    private.write_text("private sentinel")
    (settings.static_dir / "escaped.js").symlink_to(private)
    assert client.get("/escaped.js").status_code == 404
    assert client.get("/histopilot.db").status_code == 404
    assert client.get("/%2e%2e/private.txt").status_code == 404


@pytest.mark.parametrize("path", ["/", "/index.html", "/experiments"])
def test_frontend_html_revalidates_and_serves_rebuilt_bundle(client, settings, path):
    original = client.get(path)
    assert original.status_code == 200
    assert original.headers["cache-control"] == "no-cache"

    rebuilt = '<!doctype html><script src="/assets/index-rebuilt.js"></script>'
    (settings.static_dir / "index.html").write_text(rebuilt)
    current = client.get(path, headers={"If-None-Match": original.headers["etag"]})
    assert current.status_code == 200
    assert current.headers["cache-control"] == "no-cache"
    assert current.text == rebuilt


def test_frontend_assets_keep_their_existing_cache_policy(client, settings):
    assets = settings.static_dir / "assets"
    assets.mkdir()
    for name in ("index-bundle.js", "index-bundle.css"):
        (assets / name).write_text("/* bundled asset */")
        response = client.get(f"/assets/{name}")
        assert response.status_code == 200
        assert "cache-control" not in response.headers


def test_control_service_import_does_not_load_ml_libraries():
    script = """
import json, sys
from histopilot.api import create_app
blocked = {'torch', 'trident', 'openslide', 'cucim', 'transformers', 'h5py', 'duckdb'}
print(json.dumps(sorted(blocked.intersection(sys.modules))))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == []
