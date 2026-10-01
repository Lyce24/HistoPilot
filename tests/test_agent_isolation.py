"""The agent isolation probe fails loudly when anything private is reachable."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from typer.testing import CliRunner

from histopilot.cli import app
from histopilot.doctor import isolation_report


class Health(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - the http.server hook name
        body = json.dumps({"status": "ok", "version": "test"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_arguments):
        return


def test_a_reachable_private_service_or_folder_fails_the_probe(tmp_path, monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), Health)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        monkeypatch.setenv("HISTOPILOT_TOKEN", "hpt_abc_secret")
        report = isolation_report([url], [str(tmp_path)])
    finally:
        server.shutdown()
    statuses = {item["name"]: item["status"] for item in report["checks"]}
    assert report["isolated"] is False
    assert statuses[f"private service {url}"] == "fail"
    assert statuses[f"private path {tmp_path}"] == "fail"
    assert statuses["agent token"] == "pass"


class RefusesHost(Health):
    def do_GET(self):  # noqa: N802 - the http.server hook name
        body = json.dumps({"detail": "Unrecognized local service Host.", "code": "HOST_REFUSED"})
        self.send_response(400)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body.encode())


def test_a_service_that_refuses_the_probes_host_still_fails_the_probe():
    # Reached as host.docker.internal or a LAN address, the service answers 400 to that Host;
    # an agent there could send the Host it expects instead.
    server = HTTPServer(("127.0.0.1", 0), RefusesHost)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        report = isolation_report([url])
    finally:
        server.shutdown()
    statuses = {item["name"]: item["status"] for item in report["checks"]}
    assert report["isolated"] is False and statuses[f"private service {url}"] == "fail"


def test_an_unreachable_service_passes_and_the_command_exits_by_the_verdict(tmp_path):
    missing = tmp_path / "private"
    passed = CliRunner().invoke(
        app,
        [
            "doctor",
            "--agent-isolation",
            "--json",
            "--private-url",
            "http://127.0.0.1:9",
            "--private-path",
            str(missing),
        ],
    )
    assert passed.exit_code == 0, passed.stdout
    assert json.loads(passed.stdout)["isolated"] is True
    failed = CliRunner().invoke(
        app, ["doctor", "--agent-isolation", "--private-path", str(tmp_path)]
    )
    assert failed.exit_code == 1 and "FAIL" in failed.stdout
