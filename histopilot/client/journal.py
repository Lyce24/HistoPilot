"""Operation IDs written down before a commit is sent (docs/cli-contract.md#confirmation).

Rerunning a command after a lost answer, a timeout or Ctrl-C finds its pending entry and
replays the same ID, which the service applies once. A refusal settles the entry at once;
an answer settles it when the command ends, unless the command stopped short (a timeout,
Ctrl-C, no confirmation or a dry run), so the rerun replays it.
"""

import hashlib
import json
import os
import time
from pathlib import Path
from uuid import uuid4

KEEP_SECONDS = 7 * 24 * 3600


def new_operation_id(prefix: str) -> str:
    """A fresh operation ID: the command's prefix and a random UUID."""
    return f"{prefix}:{uuid4()}"


def operation_key(url: str, method: str, path: str, body: dict | None) -> str:
    material = {key: value for key, value in (body or {}).items() if key != "operationId"}
    return json.dumps(
        [url, method.upper(), path, material], sort_keys=True, default=str, ensure_ascii=False
    )


class OperationJournal:
    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def _path(self, key: str) -> Path:
        return self.directory / f"{hashlib.sha256(key.encode('utf-8')).hexdigest()[:40]}.json"

    def pending(self, key: str) -> str | None:
        try:
            entry = json.loads(self._path(key).read_text(encoding="utf-8"))
            if time.time() - float(entry["createdAt"]) < KEEP_SECONDS:
                return str(entry["operationId"])
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return None

    def begin(self, key: str, prefix: str, request: str) -> str:
        existing = self.pending(key)
        if existing:
            return existing
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._prune()
        operation = new_operation_id(prefix)
        path = self._path(key)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        # The request line only: bodies can hold paths and names from private studies.
        entry = {"operationId": operation, "createdAt": time.time(), "request": request}
        temporary.write_text(json.dumps(entry), encoding="utf-8")
        os.replace(temporary, path)
        return operation

    def settle(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)

    def _prune(self) -> None:
        cutoff = time.time() - KEEP_SECONDS
        for path in self.directory.glob("*.json"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
            except OSError:
                continue
