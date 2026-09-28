"""Adapter for feature validation, pack attachment and packing (``workers/pack_features``).

Jobs of one feature source share an ``exclusiveKey``, so the runner never runs two of
them together. A pack's destination is guarded by the worker's process-held output lock
(a busy output exits 75 and is requeued) and, before dispatch, by the live packing tasks
of every project. ``result.json`` is the outcome; it names the task attempt that wrote
it. A retried attempt first moves the previous attempt's unsuccessful receipt aside and
removes staging folders a killed pack left next to its destination.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from histopilot.storage.project_lock import StorageError, _reject_symlink_components
from histopilot.taskcenter.adapters.base import (
    Adapter,
    AdapterError,
    bind_cancel,
    outcome,
    read_json_file,
)
from histopilot.taskcenter.model import ACTIVE, parse_iso

RECEIPT_BYTES = 64 * 1024 * 1024
BUSY_EXIT = 75
MIN_FREE_BYTES = 256 * 1024**2
GONE = "The feature job record no longer exists."


def _folder(task: dict) -> Path:
    return Path(task["adapterData"]["packingFolder"])


def _cancelled(task: dict, error: str | None = None) -> dict:
    """A cancelled outcome; an attempt-less cancel marker is bound to this attempt."""
    bind_cancel(_folder(task) / "cancelled", task)
    return outcome("cancelled", "cancelled", error)


def _job(task: dict) -> dict:
    path = _folder(task) / "job.json"
    loaded = read_json_file(path, limit=RECEIPT_BYTES)
    if loaded is None:
        raise AdapterError(
            "The feature job record cannot be read." if path.exists() else GONE, fatal=True
        )
    job = loaded[0]
    if job.get("id") != task["adapterData"].get("jobId"):
        raise AdapterError("The feature job record belongs to another job.", fatal=True)
    return job


def _cancel_marker(task: dict) -> dict | None:
    path = _folder(task) / "cancelled"
    try:
        _reject_symlink_components(path)
        if not path.is_file():
            return None
        value = json.loads(path.read_bytes()[:65536])
    except (OSError, StorageError):
        return None
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _current_cancel(task: dict) -> bool:
    marker = _cancel_marker(task)
    if marker is None:
        return False
    attempt = (marker.get("attempts") or {}).get(task["id"])
    return not (type(attempt) is int and attempt < task["attempt"])


def overlaps(first: str, second: str) -> bool:
    first_path, second_path = Path(first), Path(second)
    return first_path.is_relative_to(second_path) or second_path.is_relative_to(first_path)


def reap_staging(destination: Path) -> int:
    """Remove ``.<name>.packing-*`` folders a killed pack left beside ``destination``.

    Only while holding the destination's output lock, so a live writer is never touched.
    The name is matched literally (not as a glob): ``pack[1]`` must never reach
    ``.pack1.packing-*``, and ``pack`` never ``.pack.packing-x.packing-*`` (the staging
    folder of a destination named ``pack.packing-x``), whose locks this call does not hold.
    """
    from histopilot.workers.packing_process import output_lock

    parent = destination.parent
    if not parent.is_dir():
        return 0
    prefix = f".{destination.name}.packing-"
    removed = 0
    try:
        with output_lock(destination):
            for path in parent.iterdir():
                name = path.name
                if not name.startswith(prefix) or "." in name[len(prefix) :]:
                    continue
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
                    removed += 1
    except StorageError:
        return 0  # a live worker holds the output
    return removed


class PackingAdapter(Adapter):
    def progress(self, task, ctx):
        value = super().progress(task, ctx)
        if value is None:
            return None
        started, updated = parse_iso(task.get("startedAt")), parse_iso(value.get("updatedAt"))
        return None if started and updated and updated < started else value

    def _receipt(self, task: dict) -> dict | None:
        loaded = read_json_file(_folder(task) / "result.json", limit=RECEIPT_BYTES)
        if loaded is None:
            return None
        value = loaded[0]
        if value.get("taskId") != task["id"] or value.get("taskAttempt") != task["attempt"]:
            return None
        return value

    def prepare(self, task, ctx):
        job = _job(task)
        folder = _folder(task)
        if _current_cancel(task):
            return {"skip": _cancelled(task, "Cancelled before start.")}
        marker = folder / "cancelled"
        if marker.exists():
            marker.unlink(missing_ok=True)  # an earlier attempt's cancel; retried since
        receipt_path = folder / "result.json"
        if receipt_path.exists():
            loaded = read_json_file(receipt_path, limit=RECEIPT_BYTES)
            previous = loaded[0] if loaded else {}
            if previous.get("state") == "succeeded" and previous.get("jobId") == job["id"]:
                return {"skip": outcome("succeeded", "already-complete")}
            # The worker returns an existing receipt unchanged; a retry needs a fresh one.
            aside = folder / f"result-attempt-{previous.get('taskAttempt') or 'legacy'}.json"
            try:
                os.replace(receipt_path, aside)
            except OSError as error:
                raise AdapterError(f"Cannot set the previous receipt aside: {error}") from error
        output = task["adapterData"].get("outputPath")
        if output:
            self._check_output(task, ctx, output)
            if task["attempt"] > 1:
                reap_staging(Path(output))
        return None

    @staticmethod
    def _check_output(task: dict, ctx, output: str) -> None:
        key = "packing.active"
        if key not in ctx.cache:
            try:
                ctx.cache[key] = ctx.store.list(states=ACTIVE, kinds=("packing",), limit=None)
            except StorageError as error:
                raise AdapterError(f"The Task Center store is unavailable: {error}") from error
        for other in ctx.cache[key]:
            theirs = (other.get("adapterData") or {}).get("outputPath")
            if other["id"] != task["id"] and theirs and overlaps(theirs, output):
                raise AdapterError("Waiting for another feature job using this output folder")
        estimated = int(task["adapterData"].get("estimatedBytes") or 0)
        ancestor = Path(output)
        while not ancestor.exists() and ancestor != ancestor.parent:
            ancestor = ancestor.parent
        try:
            free = shutil.disk_usage(ancestor).free
        except OSError:
            return
        if free < max(estimated, MIN_FREE_BYTES):
            raise AdapterError(
                f"Waiting for disk space: {ancestor} has {free / 1024**3:.1f} GiB free; "
                f"this pack needs about {max(estimated, MIN_FREE_BYTES) / 1024**3:.1f} GiB."
            )

    def on_exit(self, task, exit, ctx):
        stop = exit.get("stopReason")
        if stop == "cancel":
            return _cancelled(task)
        if stop == "pause":
            return outcome("requeue", "paused")
        receipt = self._receipt(task)
        if receipt is None:
            if exit.get("returncode") == BUSY_EXIT:
                return outcome("requeue", "busy", "Another worker is using this output.")
            if _current_cancel(task):
                return _cancelled(task, "Cancelled from the Features stage.")
            return outcome(
                "interrupted",
                "lost" if exit.get("lost") else "interrupted",
                "The feature worker stopped before recording an outcome; retry to run it again.",
            )
        state = receipt.get("state")
        if state == "succeeded":
            return outcome("succeeded", "ok")
        if state == "cancelled":
            if _current_cancel(task):
                return _cancelled(task, receipt.get("error"))
            return outcome(
                "interrupted",
                "interrupted",
                receipt.get("error") or "The feature worker was stopped without a cancel request.",
            )
        return outcome("failed", "error", receipt.get("error") or "The feature job failed.")

    def can_requeue(self, task, ctx):
        try:
            _job(task)
        except AdapterError:
            return False
        return not _current_cancel(task)

    def on_requeue(self, task, ctx):
        output = task["adapterData"].get("outputPath")
        if output:
            reap_staging(Path(output))
