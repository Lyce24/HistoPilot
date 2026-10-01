"""Drive the CLI and its client against an in-process service on a temporary workspace.

The transport hands requests to FastAPI's test client, so every route, boundary check and
error body is the real one; only the socket is missing.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from histopilot.api import create_app
from histopilot.client.transport import Response
from histopilot.config import Settings

BASE_URL = "http://127.0.0.1:8787"


class InProcessTransport:
    def __init__(self, http: TestClient):
        self.http = http
        self.sent: list[tuple[str, str]] = []

    def send(self, method, path, *, headers, body, timeout):
        self.sent.append((method, path))
        response = self.http.request(method, path, headers=headers, content=body)
        headers = {name.lower(): value for name, value in response.headers.items()}
        return Response(response.status_code, response.content, headers)


@dataclass
class Outcome:
    code: int
    stdout: str
    stderr: str

    @property
    def envelope(self) -> dict:
        return json.loads(self.stdout)


class Service:
    """A real service app, reachable by the CLI through ``TRANSPORT_FACTORY``."""

    def __init__(self, root: Path, monkeypatch):
        data = root / "data"
        data.mkdir()
        static = root / "static"
        static.mkdir()
        (static / "index.html").write_text("<!doctype html><title>HistoPilot test</title>")
        self.settings = Settings(
            workspace=root / "workspace", data_roots=(data,), static_dir=static
        )
        self.data = data
        self.http = TestClient(create_app(self.settings), base_url=BASE_URL)
        self.transport = InProcessTransport(self.http)
        from histopilot.commands import common

        monkeypatch.setattr(common, "TRANSPORT_FACTORY", lambda _url: self.transport)

    def __enter__(self):
        self.http.__enter__()
        token = self.http.get("/api/v1/session").json()["token"]
        self.http.headers["X-HistoPilot-Token"] = token
        return self

    def __exit__(self, *exc):
        return self.http.__exit__(*exc)

    def create_project(self, name: str = "Study") -> str:
        (self.settings.workspace / "projects").mkdir(parents=True, exist_ok=True)
        response = self.http.post(
            "/api/v1/projects",
            json={"name": name, "storagePath": str(self.settings.workspace / "projects" / name)},
        )
        assert response.status_code == 201, response.text
        return response.json()["id"]

    def cli(self, *arguments: str, input: str | None = None) -> Outcome:
        from histopilot.cli import app

        result = CliRunner().invoke(app, list(arguments), input=input)
        if result.exception and not isinstance(result.exception, SystemExit):
            raise result.exception.with_traceback(result.exc_info[2])
        return Outcome(result.exit_code, result.stdout, result.stderr)


def python() -> str:
    return sys.executable
