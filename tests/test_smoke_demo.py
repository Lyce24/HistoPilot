"""scripts/smoke_demo.py, the check CI runs against the wheel's and the images' demo services,
passes against a real demo service and names the first thing a broken one gets wrong."""

import json
import runpy
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import uvicorn

from histopilot import __version__
from histopilot.api import create_app
from histopilot.config import Settings

ROOT = Path(__file__).resolve().parents[1]
smoke = runpy.run_path(str(ROOT / "scripts" / "smoke_demo.py"))


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def demo_url(tmp_path):
    """A real demo service, as `histopilot serve --demo` builds it, on a free loopback port."""
    static = tmp_path / "static"
    (static / "assets").mkdir(parents=True)
    (static / "index.html").write_text(
        '<!doctype html><title>HistoPilot</title><script type="module" src="/assets/index-abc123.js"></script>'
    )
    (static / "assets" / "index-abc123.js").write_text("console.log('HistoPilot');")
    port = free_port()
    settings = Settings(
        workspace=tmp_path / "workspace", static_dir=static, host="127.0.0.1", port=port, demo=True
    )
    server = uvicorn.Server(
        uvicorn.Config(create_app(settings), host="127.0.0.1", port=port, log_config=None)
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 20
    while not server.started:
        assert time.monotonic() < deadline, "the demo service did not start"
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)


def run(url, **options):
    lines = []
    code = smoke["run"](url, out=lines.append, **options)
    return code, lines


def test_a_demo_service_passes_every_check(demo_url):
    code, lines = run(demo_url, expected_version=__version__, wait=10)
    assert code == 0, lines
    assert [line.split(":")[0] for line in lines[:-1]] == [
        "ok      service answers",
        "ok      health",
        "ok      session",
        "ok      project list",
        "ok      demo workspace",
        "ok      browser UI",
        "ok      read-only boundary",
    ]
    assert lines[-1] == "Demo smoke test passed."


def test_a_different_version_fails_the_health_check(demo_url):
    code, lines = run(demo_url, expected_version="9.9.9", wait=10)
    assert code == 1
    assert lines[-1].startswith(f"FAILED  health: version {__version__!r}, expected '9.9.9'")


def test_a_service_that_never_answers_fails_after_the_wait():
    started = time.monotonic()
    code, lines = run(f"http://127.0.0.1:{free_port()}", wait=1)
    assert code == 1
    assert lines == [lines[0]] and lines[0].startswith("FAILED  service answers:")
    assert time.monotonic() - started < 10


class Writable(BaseHTTPRequestHandler):
    """A service that answers the demo's reads but accepts a change: what the boundary
    exists to prevent."""

    def log_message(self, *args):
        pass

    def reply(self, status, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        token = self.headers.get("X-HistoPilot-Token")
        if self.path == "/api/v1/health":
            self.reply(200, {"status": "ok", "version": "0.1.0"})
        elif self.path == "/api/v1/session":
            self.reply(200, {"token": "t" * 32})
        elif self.path.startswith("/api/") and not token:
            self.reply(401, {"code": "SESSION_TOKEN_REQUIRED"})
        elif self.path == "/api/v1/projects":
            self.reply(200, {"projects": [{"id": smoke["DEMO_ID"]}], "demoOnly": True})
        elif self.path.endswith("/workspace"):
            pipeline = {"synthetic": True, "readOnly": True, "records": [{"id": "r"}]}
            self.reply(200, {"demoPipeline": pipeline})
        elif self.path == "/":
            self.reply(200, b'<script type="module" src="/assets/a.js"></script>', "text/html")
        elif self.path == "/assets/a.js":
            self.reply(200, b"1", "text/javascript")
        else:
            self.reply(403, {"code": "DEMO_MODE_ROUTE_UNAVAILABLE"})

    def do_POST(self):  # noqa: N802
        self.reply(200, {"created": True})


def test_a_service_that_accepts_a_change_fails_the_boundary_check():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Writable)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        code, lines = run(f"http://127.0.0.1:{server.server_address[1]}", wait=5)
    finally:
        server.shutdown()
    assert code == 1
    assert lines[-2].startswith("ok      browser UI")
    assert (
        lines[-1]
        == "FAILED  read-only boundary: expected 403 DEMO_MODE_READ_ONLY, got 200 b'{\"created\": true}'"
    )
