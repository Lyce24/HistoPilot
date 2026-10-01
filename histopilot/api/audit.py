"""The per-project log of what agents did: `<project>/audit/actions.jsonl`.

One line per request made with a scoped token, allowed or refused, and one per commit or
admin request from anyone. Lines hold the route and its outcome, never request bodies or
case identifiers. Provenance lives here and on receipts, never in manifests, because a
configuration's ID is the hash of its manifest.
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path

ROTATE_BYTES = 10 * 1024 * 1024
KEEP_ROTATED = 3
NAME = "actions.jsonl"


def audit_path(project_folder: Path) -> Path:
    return project_folder / "audit" / NAME


def append(project_folder: Path, entry: dict) -> None:
    """Append one line. Small O_APPEND writes stay whole even with concurrent writers."""
    path = audit_path(project_folder)
    path.parent.mkdir(mode=0o700, exist_ok=True)
    line = json.dumps(
        {"time": datetime.now(UTC).isoformat().replace("+00:00", "Z"), **entry},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    _rotate(path)
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        os.write(descriptor, (line + "\n").encode("utf-8"))
    finally:
        os.close(descriptor)


def read(project_folder: Path, *, limit: int = 200) -> list[dict]:
    """The newest ``limit`` lines, newest first."""
    path = audit_path(project_folder)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    entries = []
    for line in reversed(lines[-limit:]):
        try:
            entries.append(json.loads(line))
        except ValueError:
            continue
    return entries


def _rotate(path: Path) -> None:
    try:
        if path.stat().st_size < ROTATE_BYTES:
            return
    except FileNotFoundError:
        return
    for index in range(KEEP_ROTATED, 0, -1):
        older = path.with_name(f"{path.stem}.{index}{path.suffix}")
        newer = path if index == 1 else path.with_name(f"{path.stem}.{index - 1}{path.suffix}")
        if newer.exists():
            os.replace(newer, older)
