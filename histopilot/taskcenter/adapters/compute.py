"""Adapter for managed compute workers: refits, evaluations and interpretations.

A worker exits 0 for every outcome it handles, so ``state.json`` (not the exit code) is
the outcome; the one exception is the busy contract's exit 75, which leaves the record
untouched. The adapter only writes that file to re-queue an attempt, under the compute
folder's writer lock; ``result.json`` stays the worker's evidence of success.
"""

import hashlib
import json
from pathlib import Path

from histopilot.storage.project_lock import StorageError, writer_lock
from histopilot.taskcenter.adapters.base import (
    BUSY_CODES,
    BUSY_EXIT,
    Adapter,
    AdapterError,
    logged_busy,
    outcome,
)
from histopilot.taskcenter.model import parse_iso
from histopilot.workers.packing_process import write_json
from histopilot.workers.training_process import classify_training_failure, read_json

REQUEUEABLE = frozenset({"queued", "running", "interrupted"})
# A worker records an out-of-memory or lost-device run as failed; the runner may still
# retry it (with more VRAM, or once the device answers again), so such a record requeues.
RETRIED_FAILURES = {"out_of_memory": "oom", "cuda_device_failure": "cuda_failure"}
# Last log line of a worker pinned before the busy contract whose output was busy.
BUSY_MARKERS = ("OUTPUT_BUSY", "already using this output")
FOREIGN = "The compute record now belongs to another launch."
# The requeue hook waits this long for the job folder's lock, then retries on a later tick.
LOCK_WAIT_SECONDS = 1.0


def plan_hash(value) -> str:
    # Same canonical hash as application.feature_bundles._hash, without its imports.
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def failure_reason(error: str | None) -> str:
    """The Task Center exit reason of a worker's recorded failure message."""
    return RETRIED_FAILURES.get(classify_training_failure(error or ""), "error")


class ComputeJobAdapter(Adapter):
    @staticmethod
    def _folder(task) -> Path:
        return Path(task["adapterData"]["computeFolder"])

    def _state(self, task) -> dict | None:
        """The job's saved state; None when it is gone. Corruption is a hard failure."""
        path = self._folder(task) / "state.json"
        if not path.exists():
            return None
        try:
            return read_json(path)
        except (OSError, ValueError) as error:
            raise AdapterError(f"The compute state cannot be read: {error}", fatal=True) from error

    @staticmethod
    def _worker_started(state: dict, task: dict) -> bool:
        """Whether this attempt's worker recorded itself (it writes its identity first)."""
        recorded = state.get("process") or {}
        spawned = task.get("process") or {}
        return bool(recorded) and recorded.get("pid") == spawned.get("pid")

    def progress(self, task, ctx):
        # progress.json survives across attempts; an older snapshot is not this attempt's.
        value = super().progress(task, ctx)
        if value is None:
            return None
        started, updated = parse_iso(task.get("startedAt")), parse_iso(value.get("updatedAt"))
        return None if started and updated and updated < started else value

    def prepare(self, task, ctx):
        # Dispatch-time fence: never start a worker for a record this task no longer owns.
        state = self._state(task)
        if state is None:
            raise AdapterError("The compute record no longer exists.", fatal=True)
        if state.get("taskId") != task["id"]:
            return {"skip": outcome("cancelled", "cancelled", FOREIGN)}
        if state.get("status") == "completed":
            return {"skip": outcome("succeeded", "already-complete")}
        return None

    def on_exit(self, task, exit, ctx):
        try:
            state = self._state(task)
        except AdapterError as error:
            return outcome("failed", "error", str(error))
        if state is None:
            return outcome("failed", "error", "The compute record's state is missing.")
        folder = self._folder(task)
        stop = exit.get("stopReason")
        status = state.get("status")
        lost = "lost" if exit.get("lost") else "interrupted"
        if state.get("taskId") != task["id"]:
            # The worker's fence refused a record another launch owns or finished.
            if status == "completed":
                return outcome("succeeded", "already-complete")
            return outcome("cancelled", "cancelled", FOREIGN)
        if status == "completed":
            return outcome("succeeded", "ok", measurement=self._measurement(state))
        if status == "failed":
            error = state.get("error") or "The compute job failed."
            return outcome("failed", failure_reason(error), error)
        if status == "cancelled":
            return outcome("cancelled", "cancelled", state.get("error"))
        if status == "interrupted":
            if stop == "pause":
                return outcome("requeue", "paused")
            return outcome("interrupted", lost, state.get("error"))
        # Still queued or running: the worker stopped before recording an outcome.
        if self._busy_exit(state, task, exit):
            return outcome("requeue", "busy", "Another worker or request holds this compute job.")
        if stop == "cancel" or (folder / "cancel.requested").exists():
            return outcome(
                "cancelled",
                "cancelled",
                "The compute worker stopped after cancellation was requested.",
            )
        if stop == "pause":
            return outcome("requeue", "paused")
        return outcome(
            "interrupted", lost, "The compute worker stopped before recording an outcome."
        )

    def _busy_exit(self, state, task, exit) -> bool:
        """The busy contract's exit 75, or the busy last log line of an older pinned worker
        that exited 1 before recording itself."""
        code = exit.get("returncode")
        if code == BUSY_EXIT:
            return True
        return (
            code == 1
            and not self._worker_started(state, task)
            and logged_busy(task["command"].get("log"), BUSY_MARKERS)
        )

    @staticmethod
    def _requeueable(state: dict, task: dict) -> bool:
        """A resumable record, or one the runner retries after an OOM or a lost device."""
        status = state.get("status")
        if status in REQUEUEABLE:
            return True
        reason = (task.get("exit") or {}).get("reason")
        return (
            status == "failed"
            and reason in RETRIED_FAILURES.values()
            and failure_reason(state.get("error")) == reason
        )

    @staticmethod
    def _measurement(state: dict) -> dict | None:
        result = state.get("result") or {}
        measurement = {}
        if type(result.get("epochsCompleted")) is int and result["epochsCompleted"] > 0:
            measurement["epochs"] = result["epochsCompleted"]
        peak = result.get("cudaPeakReservedBytes")
        if type(peak) is int and peak > 0:
            measurement["peakVramGb"] = peak / 2**30
        return measurement or None

    def can_requeue(self, task, ctx):
        folder = self._folder(task)
        try:
            state = read_json(folder / "state.json")
            plan = read_json(folder / "plan.json")
        except (OSError, ValueError):
            return False
        return (
            state.get("taskId") == task["id"]
            and state.get("planHash") == plan_hash(plan)
            and self._requeueable(state, task)
            and not (folder / "cancel.requested").exists()
        )

    def on_requeue(self, task, ctx):
        folder = self._folder(task)
        try:
            with writer_lock(folder, timeout=LOCK_WAIT_SECONDS):
                state = read_json(folder / "state.json")
                if state.get("taskId") != task["id"] or not self._requeueable(state, task):
                    raise AdapterError(FOREIGN, fatal=True)
                if (folder / "cancel.requested").exists():
                    raise AdapterError(
                        "Cancellation was requested; this job will not run again.", fatal=True
                    )
                state.pop("waitingReason", None)
                at = ctx.now()
                state.update(
                    status="queued",
                    attempt=int(state.get("attempt") or 1) + 1,
                    taskAttempt=int(task["attempt"]) + 1,
                    autoResumedAt=at,
                    error=None,
                    result=None,
                    updatedAt=at,
                )
                write_json(folder / "state.json", state)
        except StorageError as error:
            # Another writer (a launch or cancel) holds the folder: retry on a later tick.
            raise AdapterError(str(error), fatal=error.code not in BUSY_CODES) from error
        except OSError as error:
            raise AdapterError(f"Cannot re-queue the compute job: {error}") from error
