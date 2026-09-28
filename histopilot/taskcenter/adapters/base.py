"""Adapter contract between the runner and the records a task belongs to."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from histopilot.storage.project_lock import StorageError, reject_symlink_components

if TYPE_CHECKING:
    from histopilot.taskcenter.store import TaskStore

MAX_PROGRESS_BYTES = 1024 * 1024
LOG_TAIL_BYTES = 16 * 1024
# The busy contract: a worker that cannot get its project or output lock within a short
# wait exits with EX_TEMPFAIL, or records one of these error codes. Nothing was changed,
# so the runner requeues the task with a backoff instead of failing it.
BUSY_EXIT = 75
BUSY_CODES = frozenset({"PROJECT_BUSY", "OUTPUT_BUSY"})


class AdapterError(Exception):
    """A hook failed. Transient failures are retried later; fatal ones fail the task.

    ``retry_after`` (seconds) marks an answer that is simply not ready yet, such as a probe
    still running: the runner asks again after that pause instead of backing off.
    """

    def __init__(self, message, *, transient=True, fatal: bool | None = None, retry_after=None):
        super().__init__(message)
        self.transient = (not fatal) if fatal is not None else bool(transient)
        self.retry_after = retry_after

    @property
    def fatal(self) -> bool:
        return not self.transient


@dataclass
class RunnerContext:
    store: TaskStore
    now: Callable[[], str]
    settings: dict
    host: dict
    log: Callable[[str], None]
    cache: dict = field(default_factory=dict)  # fresh every tick
    shared: dict = field(default_factory=dict)  # lives as long as the runner


def read_json_file(
    path: str | Path | None, *, limit=MAX_PROGRESS_BYTES
) -> tuple[dict, float] | None:
    """A bounded JSON object and its mtime, or None when missing or unreadable."""
    if not path:
        return None
    path = Path(path)
    try:
        reject_symlink_components(path)
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as stream:
            mtime = os.fstat(stream.fileno()).st_mtime
            content = stream.read(limit + 1)
        if len(content) > limit:
            return None
        value = json.loads(content)
    except (OSError, ValueError, StorageError):
        return None
    return (value, mtime) if isinstance(value, dict) else None


def bind_cancel(path: str | Path, task: dict) -> None:
    """Name ``task``'s attempt in the cancel marker at ``path`` if it names none.

    A stage writes a marker without attempts when the Task Center could not say which
    attempt runs. Such a marker reads as current for every attempt, so each later retry
    would be cancelled before it starts. Once it has cancelled an attempt, it is bound to
    that attempt and a retry reads it as an earlier attempt's cancel.
    """
    path = Path(path)
    try:
        reject_symlink_components(path)
        if not path.is_file():
            return
        content = path.read_bytes()[:65536]
    except (OSError, StorageError):
        return
    try:
        marker = json.loads(content)
    except ValueError:
        marker = {}  # a legacy plain-text marker
    marker = marker if isinstance(marker, dict) else {}
    attempts = marker.get("attempts") if isinstance(marker.get("attempts"), dict) else {}
    if type(attempts.get(task["id"])) is int:
        return
    marker["attempts"] = {**attempts, task["id"]: task["attempt"]}
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(marker))
        os.replace(temporary, path)
    except OSError:  # still unbound: the attempt a retry cancels binds it
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def log_tail(path, limit=LOG_TAIL_BYTES) -> str:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            stream.seek(max(0, size - limit))
            return stream.read(limit).decode(errors="replace")
    except (OSError, TypeError, ValueError):
        return ""


def logged_busy(path, markers) -> bool:
    """Fallback for workers pinned before the busy contract: did the attempt end busy?

    Such workers exit 1 with the lock error as the last line of their log. The log is
    appended across attempts, so only its last line belongs to this attempt; an older
    busy exit must never turn a later crash into an endless requeue.
    """
    lines = [line for line in log_tail(path).splitlines() if line.strip()] if path else []
    return bool(lines) and any(marker in lines[-1] for marker in markers)


class Adapter:
    """Default behaviour for opaque commands; subclasses own their record's files."""

    def prepare(self, task: dict, ctx: RunnerContext) -> dict | None:
        return None

    def on_started(self, task: dict, identity: dict, gpu: int | None, ctx: RunnerContext) -> None:
        return None

    def progress(self, task: dict, ctx: RunnerContext) -> dict | None:
        loaded = read_json_file((task.get("command") or {}).get("progress"))
        if loaded is None:
            return None
        value, mtime = loaded
        return {**value, "updatedAt": datetime.fromtimestamp(mtime, UTC).isoformat()}

    def on_exit(self, task: dict, exit: dict, ctx: RunnerContext) -> dict:
        stop = exit.get("stopReason")
        returncode = exit.get("returncode")
        if stop == "pause":
            return outcome("requeue", "paused")
        if stop == "cancel":
            return outcome("cancelled", "cancelled")
        if exit.get("lost"):
            return outcome("interrupted", "lost")
        if returncode is None:
            # An adopted process's exit status cannot be observed; its outcome is unknown.
            return outcome("interrupted", "interrupted")
        if returncode == 0:
            return outcome("succeeded", "ok")
        if returncode == BUSY_EXIT:
            return outcome("requeue", "busy")
        detail = (
            f"stopped by signal {-returncode}"
            if returncode < 0
            else f"exited with code {returncode}"
        )
        return outcome("failed", "error", f"The task {detail}.")

    def can_requeue(self, task: dict, ctx: RunnerContext) -> bool:
        return False

    def on_requeue(self, task: dict, ctx: RunnerContext) -> None:
        return None


def outcome(
    state: str,
    exit_reason: str,
    error: str | None = None,
    *,
    measurement: dict | None = None,
) -> dict:
    return {
        "state": state,
        "exitReason": exit_reason,
        "error": error,
        "measurement": measurement,
    }
