"""Read-only JSON-lines HTTP/ASGI bridge. No server or listening socket is started.

The workspace must already contain the explicitly selected, registered project.
Application startup is deliberately not run: it would initialize global records.
Each input line is {id, method, url, headers?, body?}; response bodies are base64.
Keep logs and captured responses private because they contain research identifiers.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import re
import sys
import traceback
from pathlib import Path
from urllib.parse import unquote, urlsplit

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
PROTOCOL_STDOUT = sys.stdout


def permitted(method: str, url: str, project: str) -> bool:
    """Deny mutations and every other project's API before ASGI dispatch."""
    parsed = urlsplit(url)
    if parsed.scheme != "http" or parsed.netloc != "127.0.0.1:8787":
        return False
    path = unquote(parsed.path)
    if any(part in {".", ".."} for part in path.split("/")) or "%" in path:
        return False
    if not path.startswith("/api/"):
        return method in {"GET", "HEAD"}
    if path in {
        "/api/v1/health", "/api/v1/session", "/api/v1/system",
        "/api/v1/system/compute", "/api/v1/jobs",
    } or path.startswith("/api/v1/models/"):
        return method == "GET"
    prefix = f"/api/v1/projects/{project}"
    if not path.startswith(prefix + "/"):
        return False
    if method in {"GET", "HEAD"}:
        return True
    if method != "POST":
        return False
    suffix = path[len(prefix):]
    return bool(
        re.fullmatch(r"/datasets/[^/]+/query", suffix)
        or re.fullmatch(r"/evaluation-runs/[^/]+/cases/(query|export)", suffix)
        or suffix in {
            "/interpretations/gallery", "/interpretations/slide-inspection",
            "/morphology/neighbors", "/evaluation-runs/compare", "/protocols/explore",
        }
    )


def encoded_response(status: int, body: bytes, headers: list[tuple[str, str]]) -> dict:
    # httpx exposes decoded bodies; sending stale compression/length headers breaks Chromium.
    excluded = {"content-length", "content-encoding", "transfer-encoding", "connection"}
    return {
        "status": status,
        "headers": [{"name": key, "value": value} for key, value in headers if key.lower() not in excluded],
        "body": base64.b64encode(body).decode("ascii"),
    }


async def main(args: argparse.Namespace) -> None:
    sys.stdout = sys.stderr  # Library diagnostics cannot corrupt the JSON-lines protocol.
    import httpx

    from histopilot.api.app import create_app
    from histopilot.config import Settings

    if not re.fullmatch(r"project-[a-f0-9]{32}", args.project):
        raise ValueError("Expected one registered project ID")
    if not (args.workspace / "histopilot.db").is_file():
        raise ValueError("Workspace registry must already exist; this bridge never initializes it")
    settings = Settings(
        workspace=args.workspace,
        data_roots=tuple(args.data_root),
        static_dir=args.static_dir,
        host="127.0.0.1",
        port=8787,
    )
    app = create_app(settings)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    semaphore = asyncio.Semaphore(4)
    tasks: set[asyncio.Task] = set()

    def emit(value: dict) -> None:
        PROTOCOL_STDOUT.write(json.dumps(value, separators=(",", ":")) + "\n")
        PROTOCOL_STDOUT.flush()

    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8787", timeout=120) as client:
        async def handle(value: dict) -> None:
            identity = value.get("id")
            try:
                method, url = str(value.get("method", "GET")).upper(), str(value["url"])
                if not permitted(method, url, args.project):
                    response = encoded_response(403, b'{"detail":"Read-only browser bridge blocked this request","code":"BRIDGE_READ_ONLY"}', [("Content-Type", "application/json")])
                    response["blocked"] = True
                else:
                    async with semaphore:
                        headers = {key: val for key, val in value.get("headers", {}).items() if key.lower() not in {"host", "content-length", "accept-encoding", "connection"}}
                        result = await client.request(method, url, headers=headers, content=value.get("body"))
                        response = encoded_response(result.status_code, result.content, list(result.headers.multi_items()))
                emit({"id": identity, **response})
            except Exception as error:
                traceback.print_exc(file=sys.stderr)
                emit({"id": identity, "error": f"{type(error).__name__}: {error}"})

        index = args.static_dir / "index.html"
        emit({"ready": True, "project": args.project, "python": sys.executable, "staticIndexSha256": hashlib.sha256(index.read_bytes()).hexdigest() if index.is_file() else None})
        while line := await asyncio.to_thread(sys.stdin.readline):
            value = json.loads(line)
            task = asyncio.create_task(handle(value))
            tasks.add(task)
            task.add_done_callback(tasks.discard)
        if tasks:
            await asyncio.gather(*tasks)
    app.state.projects.database.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--workspace", type=Path, default=Path.home() / ".histopilot/workspace")
    parser.add_argument("--data-root", type=Path, action="append", default=[])
    parser.add_argument("--static-dir", type=Path, default=REPOSITORY / "histopilot/static")
    asyncio.run(main(parser.parse_args()))
