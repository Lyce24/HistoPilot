"""Reversible workspace visibility, separate from immutable scientific storage.

A sidecar leaves the scientific database at schema v4 so pinned training workers
remain compatible. The guard serializes dependency checks, launches and visibility
changes. It is deliberately distinct from the scientific writer lock and must be
acquired first whenever both locks are needed.
"""

import copy
import json
import os
import re
import stat
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock, local
from uuid import uuid4

from histopilot.storage.project_lock import (
    StorageError,
    _reject_symlink_components,
    fsync_directory,
)

LIFECYCLE_FILE = "histopilot-lifecycle.json"
LOCK_FILE = ".histopilot-lifecycle.lock"
MAX_BYTES = 16 * 1024 * 1024
_STATES = frozenset({"active", "archived", "trashed"})
_TYPES = frozenset({"dataset", "configuration", "draft", "packing", "extraction", "project"})
_PACK_ID = re.compile(r"pack-[a-f0-9]{64}\Z")
_PACK_JOB_ID = re.compile(r"packing-[a-f0-9]{32}\Z")
_SHA256 = re.compile(r"[a-f0-9]{64}\Z")
_REGISTRY_LOCK = RLock()
_GUARDS: dict[tuple[int, str], tuple[RLock, local]] = {}


def _error(message: str, code: str = "LIFECYCLE_INVALID", status: int = 409) -> StorageError:
    return StorageError(message, code, status)


def _key(value: object) -> str:
    if not isinstance(value, str) or ":" not in value:
        raise ValueError("Invalid lifecycle record key")
    kind, identity = value.split(":", 1)
    if (
        kind not in _TYPES
        or not identity
        or len(identity) > 200
        or identity != identity.strip()
        or any(character in identity for character in "/\\\x00:")
        or any(ord(character) < 32 for character in identity)
    ):
        raise ValueError("Invalid lifecycle record key")
    return value


def _time(value: object) -> None:
    if (
        not isinstance(value, str)
        or datetime.fromisoformat(value.replace("Z", "+00:00")).tzinfo is None
    ):
        raise ValueError("Invalid lifecycle timestamp")


def _regular(path: Path, *, missing_ok: bool = False) -> os.stat_result | None:
    _reject_symlink_components(path)
    try:
        info = path.lstat()
    except FileNotFoundError:
        if missing_ok:
            return None
        raise _error("Workspace lifecycle metadata is missing.", "LIFECYCLE_CORRUPT") from None
    except OSError as error:
        raise _error(
            "Workspace lifecycle metadata cannot be read.", "STORAGE_READ_FAILED", 403
        ) from error
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise _error(
            "Lifecycle files must be regular files without aliases.", "STORAGE_UNSAFE_PATH", 403
        )
    return info


@contextmanager
def lifecycle_guard(folder: Path, *, timeout: float = 5) -> Iterator[None]:
    """Reentrant in one thread; coordinates all threads and POSIX processes."""
    folder = Path(folder).absolute()
    _reject_symlink_components(folder)
    if os.name != "posix":
        raise _error("Workspace lifecycle requires POSIX locking.", "STORAGE_UNSUPPORTED")
    import fcntl

    with _REGISTRY_LOCK:
        mutex, state = _GUARDS.setdefault((os.getpid(), str(folder)), (RLock(), local()))
    deadline = time.monotonic() + max(0, timeout)
    if not mutex.acquire(timeout=max(0, timeout)):
        raise _error(
            "Another operation is changing this workspace. Retry after it finishes.", "PROJECT_BUSY"
        )
    descriptor = None
    try:
        if getattr(state, "depth", 0):
            state.depth += 1
            try:
                yield
            finally:
                state.depth -= 1
            return
        path = folder / LOCK_FILE
        try:
            descriptor = os.open(
                path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600
            )
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise _error(
                    "The lifecycle lock must be a regular file without aliases.",
                    "STORAGE_UNSAFE_PATH",
                    403,
                )
            while True:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError as error:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise _error(
                            "Another operation is changing this workspace. Retry after it finishes.",
                            "PROJECT_BUSY",
                        ) from error
                    time.sleep(min(0.01, remaining))
            current = os.stat(path, follow_symlinks=False)
            if (info.st_dev, info.st_ino) != (current.st_dev, current.st_ino):
                raise _error("The lifecycle lock changed.", "STORAGE_UNSAFE_PATH", 403)
            os.fsync(descriptor)
            fsync_directory(folder)
        except OSError as error:
            raise _error(
                "The workspace lifecycle lock is unavailable.", "STORAGE_LOCK_FAILED", 403
            ) from error
        state.depth = 1
        try:
            yield
        finally:
            state.depth = 0
    finally:
        try:
            if descriptor is not None:
                os.close(descriptor)
        finally:
            mutex.release()


class LifecycleStore:
    def __init__(self, folder: Path, project_id: str):
        self.folder = Path(folder).absolute()
        self.project_id = project_id
        try:
            if not isinstance(project_id, str):
                raise ValueError("Invalid project identity")
            _key(f"project:{project_id}")
        except ValueError as error:
            raise _error("Supply a valid project identity.", "INVALID_INPUT", 422) from error
        self.path = self.folder / LIFECYCLE_FILE

    def _empty(self) -> dict:
        return {
            "version": 1,
            "projectId": self.project_id,
            "revision": 0,
            "records": {},
            "operations": {},
            "audit": [],
        }

    def read(self) -> dict:
        """Read an atomic snapshot; absent metadata means all records are active."""
        info = _regular(self.path, missing_ok=True)
        if info is None:
            return self._empty()
        descriptor = None
        try:
            descriptor = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or opened.st_nlink > 1:
                raise _error(
                    "Lifecycle metadata must be a regular file without aliases.",
                    "STORAGE_UNSAFE_PATH",
                    403,
                )
            if opened.st_size > MAX_BYTES:
                raise ValueError("Oversized lifecycle metadata")
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                content = handle.read(MAX_BYTES + 1)
            if len(content) > MAX_BYTES:
                raise ValueError("Oversized lifecycle metadata")
            document = json.loads(content)
            self._validate(document)
            return document
        except (ValueError, TypeError, KeyError, AttributeError, RecursionError) as error:
            raise _error(
                "Workspace lifecycle metadata is invalid; restore its last intact copy.",
                "LIFECYCLE_CORRUPT",
            ) from error
        except OSError as error:
            raise _error(
                "Workspace lifecycle metadata cannot be read.", "STORAGE_READ_FAILED", 403
            ) from error
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def _validate(self, document: dict) -> None:
        if (
            not isinstance(document, dict)
            or document.get("version") != 1
            or document.get("projectId") != self.project_id
            or type(document.get("revision")) is not int
            or not 0 <= document["revision"] < 2**63 - 1
            or not isinstance(document.get("records"), dict)
            or not isinstance(document.get("operations"), dict)
            or not isinstance(document.get("audit"), list)
            or len(document["audit"]) != document["revision"]
            or len(document["operations"]) != document["revision"]
        ):
            raise ValueError("Invalid lifecycle envelope")
        for key, item in document["records"].items():
            _key(key)
            if not isinstance(item, dict) or item.get("state") not in _STATES:
                raise ValueError("Invalid lifecycle state")
            _time(item["updatedAt"])
        seen = set()
        replayed_records = {}
        for revision, event in enumerate(document["audit"], start=1):
            if not isinstance(event, dict) or event.get("revision") != revision:
                raise ValueError("Invalid lifecycle audit")
            identity = event["operationId"]
            if not isinstance(identity, str) or not identity or identity in seen:
                raise ValueError("Invalid lifecycle operation")
            seen.add(identity)
            targets = event.get("targets", [])
            if not isinstance(targets, list) or len(set(targets)) != len(targets):
                raise ValueError("Invalid lifecycle targets")
            for target in targets:
                _key(target)
            _time(event["at"])
            if event.get("action") not in {None, "archive", "trash", "restore", "cancel"}:
                raise ValueError("Invalid lifecycle action")
            if not isinstance(event["changes"], dict) or (
                not event["changes"] and event.get("action") != "cancel"
            ):
                raise ValueError("Invalid lifecycle audit changes")
            for key, change in event["changes"].items():
                _key(key)
                if (
                    change["from"] not in _STATES
                    or change["to"] not in _STATES
                    or change["from"] != replayed_records.get(key, {}).get("state", "active")
                ):
                    raise ValueError("Invalid lifecycle transition")
                replayed_records[key] = {"state": change["to"], "updatedAt": event["at"]}
            receipt = document["operations"][identity]
            if (
                receipt["revision"] != revision
                or not _SHA256.fullmatch(receipt["requestHash"])
                or receipt["changes"]
                != {key: change["to"] for key, change in event["changes"].items()}
                or receipt["at"] != event["at"]
                or receipt.get("action") != event.get("action")
                or receipt.get("targets", []) != event.get("targets", [])
            ):
                raise ValueError("Invalid lifecycle receipt")
        if document["records"] != replayed_records:
            raise ValueError("Lifecycle state disagrees with its audit history")

    def assert_usable(self, refs: Iterable[str]) -> None:
        """Reject trashed references; archived references remain valid provenance."""
        document = self.read()
        keys = [f"project:{self.project_id}", *refs]
        for key in keys:
            try:
                _key(key)
            except ValueError as error:
                raise _error(
                    "Supply valid lifecycle reference keys.", "INVALID_INPUT", 422
                ) from error
            if document["records"].get(key, {}).get("state") == "trashed":
                raise _error(
                    f"{key} is in Trash. Restore it before using or changing it.", "RECORD_TRASHED"
                )

    def assert_document_usable(self, document: object, *, exclude_ids: Iterable[str] = ()) -> None:
        """Check exact reference strings recursively, without interpreting source paths."""
        lifecycle = self.read()
        excluded = set(exclude_ids)
        blocked = {
            value
            for key, item in lifecycle["records"].items()
            if item["state"] == "trashed" and key.split(":", 1)[1] not in excluded
            for value in (key, key.split(":", 1)[1])
        }
        self.assert_usable(())
        pack_ids = set()
        stack = [document]
        while stack:
            item = stack.pop()
            if isinstance(item, str) and item in blocked:
                raise _error(
                    f"Referenced record {item} is in Trash. Restore it before continuing.",
                    "RECORD_TRASHED",
                )
            if isinstance(item, str) and _PACK_ID.fullmatch(item):
                pack_ids.add(item)
            if isinstance(item, dict):
                stack.extend(item.keys())
                stack.extend(item.values())
            elif isinstance(item, (list, tuple)):
                stack.extend(item)
        self._assert_pack_aliases(pack_ids, lifecycle["records"])

    def _assert_pack_aliases(self, candidates: set[str], records: dict) -> None:
        """A receipt may be referenced by its artifact ID instead of its job ID.

        Only inspect local receipts when a pack alias is actually present. A new
        independent verification of the same artifact can still authorize it.
        Original materialization IDs and external feature files are never removed
        or interpreted as aliases of one particular verification receipt.
        """
        if not candidates:
            return
        trashed = {
            key.split(":", 1)[1]
            for key, item in records.items()
            if key.startswith("packing:") and item["state"] == "trashed"
        }
        if not trashed:
            return
        folder = self.folder / "packing"
        total = 0

        def artifact_id(identity: str) -> str | None:
            nonlocal total
            if not _PACK_JOB_ID.fullmatch(identity):
                return None
            path = folder / identity / "result.json"
            info = _regular(path, missing_ok=True)
            if info is None:
                return None
            total += info.st_size
            if info.st_size > 64 * 1024 * 1024 or total > 256 * 1024 * 1024:
                raise _error(
                    "Packing receipt metadata exceeds cleanup reference-check bounds.",
                    "LIFECYCLE_LIMIT",
                    413,
                )
            descriptor = None
            try:
                descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                opened = os.fstat(descriptor)
                if not stat.S_ISREG(opened.st_mode) or opened.st_nlink > 1:
                    raise _error(
                        "Packing receipts must be regular files without aliases.",
                        "STORAGE_UNSAFE_PATH",
                        403,
                    )
                with os.fdopen(descriptor, "rb", closefd=False) as handle:
                    content = handle.read(64 * 1024 * 1024 + 1)
                if len(content) > 64 * 1024 * 1024:
                    raise ValueError("Oversized receipt")
                result = json.loads(content)
                if not isinstance(result, dict) or result.get("jobId") != identity:
                    raise ValueError("Packing receipt identity mismatch")
                artifact = result.get("artifact")
                if result.get("state") != "succeeded" or artifact is None:
                    return None
                if (
                    not isinstance(artifact, dict)
                    or artifact.get("jobId") != identity
                    or not isinstance(artifact.get("id"), str)
                    or not _PACK_ID.fullmatch(artifact["id"])
                ):
                    raise ValueError("Invalid packing artifact identity")
                return artifact["id"]
            except (ValueError, TypeError, RecursionError) as error:
                raise _error(
                    "Packing receipt metadata cannot establish lifecycle references.",
                    "LIFECYCLE_CORRUPT",
                ) from error
            except OSError as error:
                raise _error(
                    "Packing receipt metadata cannot be read.", "STORAGE_READ_FAILED", 403
                ) from error
            finally:
                if descriptor is not None:
                    os.close(descriptor)

        blocked = {
            alias for identity in sorted(trashed) if (alias := artifact_id(identity)) in candidates
        }
        if not blocked:
            return
        _reject_symlink_components(folder)
        paths = sorted(folder.glob("packing-*/result.json"))
        if len(paths) > 10000:
            raise _error(
                "Too many packing receipts for cleanup reference validation.",
                "LIFECYCLE_LIMIT",
                413,
            )
        for path in paths:
            identity = path.parent.name
            if identity not in trashed:
                blocked.discard(artifact_id(identity))
                if not blocked:
                    return
        raise _error(
            f"Referenced feature pack {sorted(blocked)[0]} belongs to a record in Trash. Restore its packing record before continuing.",
            "RECORD_TRASHED",
        )

    def apply(
        self,
        changes: dict[str, str],
        operation_id: str,
        request_hash: str,
        expected_revision: int,
        *,
        action: str | None = None,
        targets: list[str] | None = None,
    ) -> dict:
        """Atomically apply one reviewed set of visibility changes and retain its receipt.

        Exact operation retries are accepted even after later changes. They return
        the latest snapshot and never replay old mutations over newer decisions.
        Dependency and job-state checks belong to the application and must share
        lifecycle_guard with this call.
        """
        try:
            if targets is not None:
                if not isinstance(targets, list) or len(set(targets)) != len(targets):
                    raise ValueError("Invalid lifecycle targets")
                for target in targets:
                    _key(target)
            if action not in {None, "archive", "trash", "restore", "cancel"}:
                raise ValueError("Invalid lifecycle action")
            if not isinstance(changes, dict) or (not changes and action != "cancel"):
                raise ValueError("No lifecycle changes")
            for key, state in changes.items():
                _key(key)
                if state not in _STATES:
                    raise ValueError("Invalid lifecycle state")
            if (
                not isinstance(operation_id, str)
                or not operation_id.strip()
                or len(operation_id) > 200
                or any(ord(character) < 32 for character in operation_id)
            ):
                raise ValueError("Invalid lifecycle operation ID")
            if not isinstance(request_hash, str) or not _SHA256.fullmatch(request_hash):
                raise ValueError("Invalid lifecycle request hash")
            if type(expected_revision) is not int or not 0 <= expected_revision < 2**63 - 1:
                raise ValueError("Invalid lifecycle revision")
        except (ValueError, TypeError) as error:
            raise _error(
                "Supply valid lifecycle changes, operation identity and revision.",
                "INVALID_LIFECYCLE_CHANGE",
                422,
            ) from error
        with lifecycle_guard(self.folder):
            document = self.read()
            previous = document["operations"].get(operation_id)
            if previous is not None:
                if (
                    previous["requestHash"] != request_hash
                    or previous["changes"] != changes
                    or previous.get("action") != action
                    or previous.get("targets", []) != (targets or [])
                ):
                    raise _error(
                        "This cleanup operation ID was already used for a different request.",
                        "OPERATION_CONFLICT",
                    )
                return document
            if document["revision"] != expected_revision:
                raise _error(
                    "Workspace records changed. Review cleanup again before applying it.",
                    "LIFECYCLE_REVISION_CONFLICT",
                )
            if document["revision"] >= 2**63 - 2:
                raise _error("The lifecycle revision limit has been reached.", "LIFECYCLE_LIMIT")
            updated = copy.deepcopy(document)
            now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
            updated["revision"] += 1
            event = {
                "operationId": operation_id,
                "revision": updated["revision"],
                "at": now,
                "changes": {},
            }
            if action is not None:
                event["action"] = action
            if targets is not None:
                event["targets"] = list(targets)
            for key, state in sorted(changes.items()):
                event["changes"][key] = {
                    "from": document["records"].get(key, {}).get("state", "active"),
                    "to": state,
                }
                updated["records"][key] = {"state": state, "updatedAt": now}
            updated["audit"].append(event)
            updated["operations"][operation_id] = {
                "requestHash": request_hash,
                "revision": updated["revision"],
                "changes": dict(sorted(changes.items())),
                "at": now,
            }
            if action is not None:
                updated["operations"][operation_id]["action"] = action
            if targets is not None:
                updated["operations"][operation_id]["targets"] = list(targets)
            encoded = json.dumps(
                updated, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
            ).encode("utf-8")
            if len(encoded) > MAX_BYTES:
                raise _error(
                    "Workspace cleanup history reached its supported metadata size.",
                    "LIFECYCLE_LIMIT",
                    413,
                )
            self._write(encoded)
            return updated

    def _write(self, content: bytes) -> None:
        _regular(self.path, missing_ok=True)
        temporary = self.folder / f".histopilot-lifecycle-{uuid4().hex}.tmp"
        created = False
        try:
            descriptor = os.open(
                temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
            )
            created = True
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            _regular(self.path, missing_ok=True)
            os.replace(temporary, self.path)
            created = False
            fsync_directory(self.folder)
        except OSError as error:
            raise _error(
                "Workspace cleanup could not be saved. Reload before retrying.",
                "STORAGE_WRITE_FAILED",
                403,
            ) from error
        finally:
            if created:
                temporary.unlink(missing_ok=True)
