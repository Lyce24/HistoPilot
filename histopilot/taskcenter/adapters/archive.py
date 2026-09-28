"""Adapter for study archive operations (``workers/portability``: export, verify, restore).

``state.json`` is the record and the outcome; the worker stamps it with its task attempt.
An export that finds its project busy records ``queued`` and exits 75; the task is then
requeued with the runner's busy backoff until the project is idle. A retry resets a
finished-but-unsuccessful record to ``queued`` first, because the worker returns a
terminal record unchanged.
"""

from __future__ import annotations

from pathlib import Path

from histopilot.storage.project_lock import StorageError, writer_lock
from histopilot.taskcenter.adapters.base import (
    Adapter,
    AdapterError,
    bind_cancel,
    outcome,
    read_json_file,
)
from histopilot.taskcenter.model import parse_iso

STATE_BYTES = 64 * 1024 * 1024
BUSY_EXIT = 75
RETRYABLE = frozenset({"failed", "cancelled", "interrupted"})


def _folder(task: dict) -> Path:
    return Path(task["adapterData"]["portabilityFolder"])


def _cancelled(task: dict, error: str | None = None) -> dict:
    """A cancelled outcome; an attempt-less cancel marker is bound to this attempt."""
    bind_cancel(_folder(task) / "cancel.requested", task)
    return outcome("cancelled", "cancelled", error)


def _state(task: dict) -> dict:
    path = _folder(task) / "state.json"
    loaded = read_json_file(path, limit=STATE_BYTES)
    if loaded is None:
        raise AdapterError(
            "The archive record cannot be read."
            if path.exists()
            else "The archive record no longer exists.",
            fatal=True,
        )
    state = loaded[0]
    if state.get("id") != task["adapterData"].get("jobId"):
        raise AdapterError("The archive record belongs to another operation.", fatal=True)
    return state


def _mine(state: dict, task: dict) -> bool:
    return state.get("taskId") == task["id"] and state.get("taskAttempt") == task["attempt"]


def _current_cancel(task: dict) -> bool:
    """A cancel request for this attempt; one naming an earlier attempt is stale (retried)."""
    loaded = read_json_file(_folder(task) / "cancel.requested", limit=65536)
    if loaded is None:
        return (_folder(task) / "cancel.requested").exists()
    attempt = (loaded[0].get("attempts") or {}).get(task["id"])
    return not (type(attempt) is int and attempt < task["attempt"])


class ArchiveAdapter(Adapter):
    def progress(self, task, ctx):
        value = super().progress(task, ctx)
        if value is None:
            return None
        started, updated = parse_iso(task.get("startedAt")), parse_iso(value.get("updatedAt"))
        return None if started and updated and updated < started else value

    def prepare(self, task, ctx):
        folder = _folder(task)
        state = _state(task)
        if state.get("status") == "completed":
            return {"skip": outcome("succeeded", "already-complete")}
        if _current_cancel(task):
            return {"skip": _cancelled(task, "Cancelled before start.")}
        if state.get("status") in RETRYABLE or (folder / "cancel.requested").exists():
            self._reset(task)
        return None

    @staticmethod
    def _reset(task: dict) -> None:
        """Make a finished record runnable again for this (retried) attempt."""
        folder = _folder(task)
        try:
            with writer_lock(folder.parent, timeout=5):
                state = _state(task)
                if state.get("status") == "completed":
                    return
                (folder / "cancel.requested").unlink(missing_ok=True)
                (folder / "progress.json").unlink(missing_ok=True)
                from histopilot.storage.io import write_json_atomic

                state.update(status="queued", error=None, result=None)
                write_json_atomic(folder / "state.json", state)
        except StorageError as error:
            raise AdapterError(str(error), fatal=error.code != "PROJECT_BUSY") from error
        except OSError as error:
            raise AdapterError(f"Cannot reset the archive record: {error}") from error

    def on_exit(self, task, exit, ctx):
        stop = exit.get("stopReason")
        if stop == "cancel":
            return _cancelled(task)
        if stop == "pause":
            return outcome("requeue", "paused")
        try:
            state = _state(task)
        except AdapterError as error:
            return outcome("failed", "error", str(error))
        status = state.get("status")
        busy = exit.get("returncode") == BUSY_EXIT and status in {"queued", "starting", "running"}
        if busy or (_mine(state, task) and status == "queued"):
            return outcome(
                "requeue",
                "busy",
                state.get("waitingReason") or "Waiting for the project to become idle.",
            )
        if not _mine(state, task):
            if _current_cancel(task):
                return _cancelled(task, "Cancelled from the Operations stage.")
            return outcome(
                "interrupted",
                "lost" if exit.get("lost") else "interrupted",
                "The archive worker stopped before recording an outcome; retry to run it again.",
            )
        if status == "completed":
            return outcome("succeeded", "ok")
        if status == "cancelled":
            return _cancelled(task, state.get("error"))
        if status == "interrupted":
            return outcome("interrupted", "interrupted", state.get("error"))
        return outcome("failed", "error", state.get("error") or "The archive operation failed.")

    def can_requeue(self, task, ctx):
        try:
            state = _state(task)
        except AdapterError:
            return False
        return state.get("status") != "completed" and not _current_cancel(task)

    def on_requeue(self, task, ctx):
        state = _state(task)
        if state.get("status") in RETRYABLE:
            self._reset(task)
