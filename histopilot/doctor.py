"""Read distribution metadata without importing optional compute libraries."""

import platform
import sqlite3
from importlib.metadata import PackageNotFoundError, version


def system_report() -> dict:
    packages = {}
    for name in (
        "histopilot",
        "fastapi",
        "sqlalchemy",
        "torch",
        "trident",
        "openslide-python",
        "nvidia-ml-py",
    ):
        try:
            packages[name] = {"installed": True, "version": version(name)}
        except PackageNotFoundError:
            packages[name] = {"installed": False, "version": None}
    return {
        "python": platform.python_version(),
        "platform": platform.system(),
        "sqlite": sqlite3.sqlite_version,
        "packages": packages,
        "compute": {
            "enabled": False,
            "scope": "control-service",
            "cuda": "not probed",
            "gpus": "not probed",
        },
        "note": (
            "This report inspects control-service package metadata only. Isolated workers "
            "implement extraction, packing, ABMIL training, evaluation and attention. "
            "Check runtime readiness in each module; package presence here does not establish "
            "worker dependencies, GPU availability or checkpoint access."
        ),
    }


def isolation_report(private_urls=(), private_paths=(), *, timeout: float = 2.0) -> dict:
    """Checks to run from where an AI agent works: nothing private may be reachable.

    Only operating-system separation protects private data from an agent with a shell, so
    this probes from the agent's side: the private service must not answer, private
    folders must not be readable, and the Task Center state must be separate.
    """
    import json
    import os
    from urllib.error import HTTPError, URLError
    from urllib.request import ProxyHandler, build_opener

    from histopilot.api.route_classes import TOKEN_PREFIX

    opener = build_opener(ProxyHandler({}))
    checks = []

    def check(name, status, detail):
        checks.append({"name": name, "status": status, "detail": detail})

    urls = list(private_urls)
    for url in urls:
        try:
            with opener.open(url.rstrip("/") + "/api/v1/health", timeout=timeout) as response:
                answered = json.loads(response.read(4096)).get("status") == "ok"
        except HTTPError:
            # Any reply is an answer. The service refuses a Host it does not serve, but an
            # agent can send the Host it expects and take the session token anyway.
            answered = True
        except (URLError, OSError, ValueError, AttributeError):
            answered = False
        check(
            f"private service {url}",
            "fail" if answered else "pass",
            "A service answers here, so an agent could send the Host a HistoPilot service "
            "expects and take its token from /api/v1/session."
            if answered
            else "Nothing answers.",
        )
    if not urls:
        check(
            "private service",
            "skip",
            "Name the private service as seen from here with --private-url, for example "
            "http://host.docker.internal:8787.",
        )
    for path in private_paths:
        readable = os.path.exists(path) and os.access(path, os.R_OK)
        check(
            f"private path {path}",
            "fail" if readable else "pass",
            "Readable from here." if readable else "Not readable from here.",
        )
    state = os.environ.get("HISTOPILOT_STATE_DIR")
    check(
        "task center state",
        "pass" if state else "warn",
        f"Separate state directory {state}."
        if state
        else "Uses the default per-user state directory; set HISTOPILOT_STATE_DIR so this "
        "Task Center never lists private work.",
    )
    scoped = os.environ.get("HISTOPILOT_TOKEN", "").strip().startswith(TOKEN_PREFIX)
    check(
        "agent token",
        "pass" if scoped else "warn",
        "Agent tools use a scoped token."
        if scoped
        else "No scoped token in HISTOPILOT_TOKEN; agent tools refuse to start without one.",
    )
    return {
        "isolated": not any(item["status"] == "fail" for item in checks),
        "checks": checks,
        "note": "Passing checks lower risk; they cannot prove isolation. See docs/agents.md.",
    }
