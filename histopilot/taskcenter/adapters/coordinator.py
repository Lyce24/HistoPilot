"""Adapter for the experiment predictor coordinator (one long-lived, low-cost task).

The coordinator records its own outcome in ``experiment-predictors/<hash>/state.json``;
the adapter reads it, and writes it only to requeue an attempt that found the project busy.
Resuming after a real ``attention`` stays an explicit user action.

A busy project is never a failure. Coordinators pinned before the busy contract record it
as ``attention`` with error code ``PROJECT_BUSY`` and exit 0, or mark the item they were
publishing failed with that code, or exit 1 with the lock error as their last log line.
Newer coordinators keep looping while the project is busy and exit 75 after a while.
Every form is requeued with the runner's busy backoff.
"""

from pathlib import Path

from histopilot.storage.lifecycle import lifecycle_guard
from histopilot.storage.project_lock import StorageError
from histopilot.taskcenter.adapters.base import (
    BUSY_CODES,
    BUSY_EXIT,
    Adapter,
    AdapterError,
    logged_busy,
    outcome,
)
from histopilot.workers.packing_process import write_json
from histopilot.workers.training_process import now, read_json

FINISHED = frozenset({"completed", "cancelled", "attention"})
# Last log lines of a coordinator pinned before the busy contract that stopped on a lock:
# the lifecycle guard, the project writer lock and a second coordinator's output lock.
BUSY_MARKERS = (
    "Another operation is changing this workspace",
    "Another operation is writing this project",
    "already using this output",
)
# The requeue hook waits this long for the project lock, then retries on a later tick.
LOCK_WAIT_SECONDS = 1.0


def _code(error) -> str | None:
    return error.get("code") if isinstance(error, dict) else None


def busy_attention(state: dict) -> bool:
    """Whether an ``attention`` state only records a busy project, not a real failure."""
    if state.get("status") != "attention":
        return False
    if _code(state.get("error")) in BUSY_CODES:
        return True
    failed = [item for item in state.get("items") or [] if item.get("status") == "failed"]
    return bool(failed) and all(_code(item.get("error")) in BUSY_CODES for item in failed)


class CoordinatorAdapter(Adapter):
    @staticmethod
    def _folder(task) -> Path:
        return Path(task["adapterData"]["coordinatorFolder"])

    def _state(self, task) -> dict | None:
        try:
            return read_json(self._folder(task) / "state.json")
        except (OSError, ValueError):
            return None

    def prepare(self, task, ctx):
        state = self._state(task) or {}
        if state.get("taskId") not in (None, task["id"]):
            return {"skip": outcome("cancelled", "cancelled", "Another launch owns this work.")}
        if state.get("status") == "completed":
            return {"skip": outcome("succeeded", "already-complete")}
        if state.get("status") == "cancelled":
            return {"skip": outcome("cancelled", "cancelled")}
        return None

    def on_exit(self, task, exit, ctx):
        state = self._state(task)
        if state is None:
            return outcome("failed", "error", "The predictor coordination state is unreadable.")
        status = state.get("status")
        stop = exit.get("stopReason")
        cancelled = stop == "cancel" or (self._folder(task) / "cancel.requested").exists()
        if status == "completed":
            return outcome("succeeded", "ok")
        if status == "cancelled":
            return outcome("cancelled", "cancelled")
        if status == "attention" and not busy_attention(state):
            error = state.get("error") or {}
            message = error.get("message") if isinstance(error, dict) else str(error)
            return outcome("failed", "error", message or "Predictor creation needs attention.")
        if stop == "pause":
            return outcome("requeue", "paused")
        if cancelled:
            return outcome("cancelled", "cancelled")
        if status == "attention" or self._busy_exit(task, exit):
            return outcome("requeue", "busy")
        return outcome(
            "interrupted",
            "lost" if exit.get("lost") else "interrupted",
            "Predictor coordination stopped before finishing.",
        )

    @staticmethod
    def _busy_exit(task, exit) -> bool:
        code = exit.get("returncode")
        if code == BUSY_EXIT:
            return True
        return bool(code) and logged_busy((task.get("command") or {}).get("log"), BUSY_MARKERS)

    def can_requeue(self, task, ctx):
        state = self._state(task)
        return (
            state is not None
            and state.get("taskId") in (None, task["id"])
            and (state.get("status") not in FINISHED or busy_attention(state))
            and not (self._folder(task) / "cancel.requested").exists()
        )

    def on_requeue(self, task, ctx):
        """Reopen what a busy project stopped, so the next attempt continues it.

        Coordinators pinned before the busy contract skip items marked failed and stop at
        ``attention``; items that failed only on a busy lock changed nothing and go back to
        waiting. Finished items are kept.
        """
        folder = self._folder(task)
        try:
            project = task.get("projectFolder") or read_json(folder / "plan.json")["projectFolder"]
            with lifecycle_guard(Path(project), timeout=LOCK_WAIT_SECONDS):
                state = read_json(folder / "state.json")
                if state.get("taskId") not in (None, task["id"]):
                    raise AdapterError("Another launch owns this work.", fatal=True)
                if (folder / "cancel.requested").exists():
                    raise AdapterError(
                        "Cancellation was requested; predictor creation will not run again.",
                        fatal=True,
                    )
                changed = False
                for item in state.get("items") or []:
                    if item.get("status") == "failed" and _code(item.get("error")) in BUSY_CODES:
                        item.update(status="waiting", error=None)
                        changed = True
                failed = any(item.get("status") == "failed" for item in state.get("items") or [])
                if state.get("status") == "attention" and not failed:
                    # Only a busy lock stopped it; a real failure keeps its attention.
                    state.update(status="queued", error=None)
                    changed = True
                if changed:
                    state["updatedAt"] = now()
                    write_json(folder / "state.json", state)
        except StorageError as error:
            # Another request holds the project: retry on a later tick.
            raise AdapterError(str(error), fatal=error.code not in BUSY_CODES) from error
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise AdapterError(f"Cannot requeue predictor coordination: {error}") from error
