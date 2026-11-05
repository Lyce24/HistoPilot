"""Smoke-test a running public demo (`histopilot serve --demo`) the way a visitor reaches it.

    python scripts/smoke_demo.py http://127.0.0.1:7860 [--expect-version 0.1.0] [--wait 90]

CI runs it against the service of the wheel it just built and of each image before the image
is published. It waits for the service to answer, then checks, in order:

1. ``/api/v1/health`` is ok and, with ``--expect-version``, reports that version;
2. ``/api/v1/session`` hands out a token, and without it ``/api/v1/projects`` is refused;
3. with it, ``/api/v1/projects`` lists the BLCA demo and says the service serves demos only;
4. the demo's workspace route returns the packaged synthetic, read-only fixture;
5. the browser UI loads: ``index.html`` and the first script it references;
6. a change is refused (403 ``DEMO_MODE_READ_ONLY``), and so is a read that would describe
   the machine (403 ``DEMO_MODE_ROUTE_UNAVAILABLE`` for ``/api/v1/filesystem/roots``).

Prints one line per check and exits 0 when all pass, or names the first failure and exits 1.
Standard library only, so it runs from any Python without installing HistoPilot.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

DEMO_ID = "blca-demo-v1"
TOKEN_HEADER = "X-HistoPilot-Token"
SCRIPT = re.compile(r"""<script[^>]+src=["']([^"']+\.js)["']""")


class SmokeFailure(Exception):
    """A check found the demo misbehaving."""


@dataclass
class Response:
    status: int
    body: bytes
    content_type: str

    def json(self) -> object:
        try:
            return json.loads(self.body)
        except ValueError as error:
            raise SmokeFailure(f"expected JSON, got {self.body[:120]!r}") from error


class Demo:
    def __init__(self, base: str, timeout: float = 30.0) -> None:
        self.base = base.rstrip("/")
        self.timeout = timeout
        self.token: str | None = None

    def request(self, method: str, path: str, *, token: bool = True) -> Response:
        headers = {TOKEN_HEADER: self.token} if token and self.token else {}
        request = urllib.request.Request(self.base + path, method=method, headers=headers)
        if method == "POST":
            request.data = b"{}"
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return Response(
                    response.status, response.read(), response.headers.get_content_type()
                )
        except urllib.error.HTTPError as error:
            return Response(error.code, error.read(), error.headers.get_content_type())

    def get(self, path: str, *, token: bool = True) -> Response:
        return self.request("GET", path, token=token)


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


def refusal(response: Response, status: int, code: str) -> None:
    body = response.json() if response.content_type == "application/json" else {}
    found = body.get("code") if isinstance(body, dict) else None
    expect(
        response.status == status and found == code,
        f"expected {status} {code}, got {response.status} {found or response.body[:120]!r}",
    )


def wait_until_up(demo: Demo, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    last = "no answer"
    while True:
        try:
            response = demo.get("/api/v1/health", token=False)
            if response.status == 200:
                return
            last = f"HTTP {response.status}"
        except (OSError, urllib.error.URLError) as error:
            last = str(getattr(error, "reason", error))
        if time.monotonic() >= deadline:
            raise SmokeFailure(
                f"{demo.base} did not answer /api/v1/health within {seconds:g} s ({last})"
            )
        time.sleep(0.5)


def check_health(demo: Demo, expected_version: str | None) -> str:
    body = demo.get("/api/v1/health", token=False).json()
    expect(isinstance(body, dict) and body.get("status") == "ok", f"health is not ok: {body}")
    version = body.get("version")
    if expected_version is not None:
        expect(version == expected_version, f"version {version!r}, expected {expected_version!r}")
    return f"version {version}"


def check_session(demo: Demo) -> str:
    body = demo.get("/api/v1/session", token=False).json()
    token = body.get("token") if isinstance(body, dict) else None
    expect(isinstance(token, str) and len(token) >= 16, f"no session token: {body}")
    demo.token = token
    refusal(demo.get("/api/v1/projects", token=False), 401, "SESSION_TOKEN_REQUIRED")
    return "token handed out; refused without it"


def check_projects(demo: Demo) -> str:
    response = demo.get("/api/v1/projects")
    expect(response.status == 200, f"projects answered {response.status}")
    body = response.json()
    expect(isinstance(body, dict), f"unexpected project list: {body}")
    ids = [project.get("id") for project in body.get("projects", [])]
    expect(DEMO_ID in ids, f"{DEMO_ID} missing from the project list {ids}")
    expect(body.get("demoOnly") is True, "the project list does not say the service is demo-only")
    return f"{len(ids)} project(s), demo-only"


def check_workspace(demo: Demo) -> str:
    response = demo.get(f"/api/v1/projects/{DEMO_ID}/workspace")
    expect(response.status == 200, f"the demo workspace answered {response.status}")
    body = response.json()
    pipeline = body.get("demoPipeline") if isinstance(body, dict) else None
    expect(isinstance(pipeline, dict), "the workspace has no demoPipeline")
    expect(
        pipeline.get("synthetic") is True and pipeline.get("readOnly") is True,
        "the demo fixture is not marked synthetic and read-only",
    )
    records = pipeline.get("records") or []
    expect(len(records) > 0, "the demo fixture has no records")
    return f"{len(records)} walkthrough records, {len(response.body) // 1024} KB"


def check_ui(demo: Demo) -> str:
    index = demo.get("/", token=False)
    expect(
        index.status == 200 and index.content_type == "text/html",
        f"index answered {index.status} {index.content_type}",
    )
    scripts = SCRIPT.findall(index.body.decode("utf-8", "replace"))
    expect(bool(scripts), "index.html references no script")
    asset = demo.get("/" + scripts[0].lstrip("/"), token=False)
    expect(asset.status == 200 and len(asset.body) > 0, f"{scripts[0]} answered {asset.status}")
    return f"index.html and {scripts[0]}"


def check_boundary(demo: Demo) -> str:
    refusal(demo.request("POST", "/api/v1/projects"), 403, "DEMO_MODE_READ_ONLY")
    refusal(demo.get("/api/v1/filesystem/roots"), 403, "DEMO_MODE_ROUTE_UNAVAILABLE")
    return "changes and machine reads refused"


def run(base: str, *, expected_version: str | None = None, wait: float = 90.0, out=print) -> int:
    demo = Demo(base)
    checks: list[tuple[str, Callable[[], str]]] = [
        ("service answers", lambda: (wait_until_up(demo, wait), demo.base)[1]),
        ("health", lambda: check_health(demo, expected_version)),
        ("session", lambda: check_session(demo)),
        ("project list", lambda: check_projects(demo)),
        ("demo workspace", lambda: check_workspace(demo)),
        ("browser UI", lambda: check_ui(demo)),
        ("read-only boundary", lambda: check_boundary(demo)),
    ]
    for name, check in checks:
        try:
            detail = check()
        except (SmokeFailure, OSError, urllib.error.URLError) as error:
            out(f"FAILED  {name}: {error}")
            return 1
        out(f"ok      {name}: {detail}")
    out("Demo smoke test passed.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("url", help="the demo's base URL, such as http://127.0.0.1:7860")
    parser.add_argument("--expect-version", help="fail unless /api/v1/health reports this version")
    parser.add_argument("--wait", type=float, default=90.0, help="seconds to wait for the service")
    args = parser.parse_args(argv)
    return run(args.url, expected_version=args.expect_version, wait=args.wait)


if __name__ == "__main__":
    sys.exit(main())
