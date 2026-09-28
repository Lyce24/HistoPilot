"""Adapters for TRIDENT extraction (GPU lane) and its artifact validation (CPU lane).

The extraction task runs ``adapters/trident/runner.py`` in managed mode: no lease of its
own, TRIDENT inside the task's process group, and a ``result.json`` receipt that names
its task attempt. The validation task runs ``workers/verify_extraction`` after it and
writes ``validation.json`` the same way. Both adapters trust these receipts, never the
exit code alone, and never a worker's own "cancelled" unless a cancel was requested:
a signal from the host (a closed terminal, a shutdown) is an interruption, which resumes.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from histopilot.storage.project_lock import StorageError, reject_symlink_components
from histopilot.taskcenter.adapters.base import (
    Adapter,
    AdapterError,
    bind_cancel,
    outcome,
    read_json_file,
)
from histopilot.taskcenter.model import ACTIVE, TERMINAL, awaiting_requeue, parse_iso

RECEIPT_BYTES = 8 * 1024 * 1024
LOG_TAIL_BYTES = 256 * 1024
OOM_MARKERS = ("cuda out of memory", "outofmemoryerror", "cuda error: out of memory")
# Dispatch waits while the output volume cannot hold this much of the estimated output.
MIN_FREE_BYTES = 2 * 1024**3
ATTEMPT_MARKER = "Starting TRIDENT worker"
GONE = "The extraction record no longer exists."


def _folder(task: dict) -> Path:
    return Path(task["adapterData"]["extractionFolder"])


def _cancelled(task: dict, error: str | None = None) -> dict:
    """A cancelled outcome; an attempt-less cancel marker is bound to this attempt."""
    bind_cancel(_folder(task) / "cancelled", task)
    return outcome("cancelled", "cancelled", error)


def overlaps(first: str, second: str) -> bool:
    first_path, second_path = Path(first), Path(second)
    return first_path.is_relative_to(second_path) or second_path.is_relative_to(first_path)


def _job(task: dict) -> dict:
    """The frozen job record; a missing or corrupt one ends the task."""
    path = _folder(task) / "job.json"
    loaded = read_json_file(path, limit=RECEIPT_BYTES)
    if loaded is None:
        if path.exists():
            raise AdapterError("The extraction record cannot be read.", fatal=True)
        raise AdapterError(GONE, fatal=True)
    job = loaded[0]
    if job.get("id") != task["adapterData"].get("jobId"):
        raise AdapterError("The extraction record belongs to another job.", fatal=True)
    return job


def _receipt(task: dict, name: str) -> dict | None:
    """``name`` beside the job, only when this attempt of this task wrote it."""
    loaded = read_json_file(_folder(task) / name, limit=RECEIPT_BYTES)
    if loaded is None:
        return None
    value = loaded[0]
    if value.get("taskId") != task["id"] or value.get("taskAttempt") != task["attempt"]:
        return None
    return value


def _cancel_marker(task: dict) -> dict | None:
    """The cancel request of this job, or None. Legacy markers are plain text."""
    path = _folder(task) / "cancelled"
    try:
        reject_symlink_components(path)
        if not path.is_file():
            return None
        content = path.read_bytes()[:65536]
    except (OSError, StorageError):
        return None
    try:
        value = json.loads(content)
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _stale_cancel(task: dict, marker: dict) -> bool:
    """A cancel recorded for an earlier attempt; the task was retried since."""
    attempt = (marker.get("attempts") or {}).get(task["id"])
    return type(attempt) is int and attempt < task["attempt"]


def _clear_stale_cancel(task: dict) -> bool:
    """Remove an earlier attempt's cancel marker; True when a current cancel remains."""
    marker = _cancel_marker(task)
    if marker is None:
        return False
    if _stale_cancel(task, marker):
        try:
            (_folder(task) / "cancelled").unlink(missing_ok=True)
        except OSError as error:
            raise AdapterError(f"Cannot clear an earlier cancel request: {error}") from error
        return False
    return True


def _attempt_log(task: dict) -> str:
    """This attempt's part of the shared job log (after its last start marker)."""
    path = (task.get("command") or {}).get("log")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            stream.seek(max(0, size - LOG_TAIL_BYTES))
            text = stream.read(LOG_TAIL_BYTES).decode(errors="replace")
    except (OSError, TypeError, ValueError):
        return ""
    index = text.rfind(ATTEMPT_MARKER)
    return text[index:] if index >= 0 else text


class _JobAdapter(Adapter):
    def progress(self, task, ctx):
        # Progress files survive across attempts; an older snapshot is not this attempt's.
        value = super().progress(task, ctx)
        if value is None:
            return None
        started, updated = parse_iso(task.get("startedAt")), parse_iso(value.get("updatedAt"))
        return None if started and updated and updated < started else value

    def _stopped(self, task: dict, exit: dict, lost_error: str) -> dict:
        """No receipt from this attempt: classify by what the Task Center knows."""
        if exit.get("stopReason") == "cancel":
            return _cancelled(task)
        if exit.get("stopReason") == "pause":
            return outcome("requeue", "paused")
        marker = _cancel_marker(task)
        if marker is not None and not _stale_cancel(task, marker):
            return _cancelled(task, "Cancelled from the Features stage.")
        return outcome("interrupted", "lost" if exit.get("lost") else "interrupted", lost_error)

    def can_requeue(self, task, ctx):
        try:
            _job(task)
        except AdapterError:
            return False
        marker = _cancel_marker(task)
        return marker is None or _stale_cancel(task, marker)


class ExtractionAdapter(_JobAdapter):
    def prepare(self, task, ctx):
        job = _job(task)
        if _clear_stale_cancel(task):
            return {"skip": _cancelled(task, "Cancelled before start.")}
        self._check_overlap(task, job, ctx)
        self._rearm_validation(task, ctx)
        self._check_space(task, job)
        return self._refine(task, ctx)

    @staticmethod
    def _check_overlap(task: dict, job: dict, ctx) -> None:
        """Wait while another extraction writes a nested or enclosing output folder.

        The exclusive key serializes identical folders only; a retry or an automatic
        resume never goes through the stage's own overlap checks, and this attempt's
        dead-lock sweep would walk the other job's live tree.
        """
        output = job.get("outputPath") or task["adapterData"].get("outputPath")
        if not output:
            return
        try:
            others = ctx.store.list(states=ACTIVE, kinds=("extraction",), limit=None)
        except StorageError as error:
            raise AdapterError(f"The Task Center store is unavailable: {error}") from error
        for other in others:
            theirs = (other.get("adapterData") or {}).get("outputPath")
            if other["id"] != task["id"] and theirs and overlaps(theirs, output):
                raise AdapterError(
                    "Waiting for another extraction using an overlapping output folder"
                )

    @staticmethod
    def _rearm_validation(task: dict, ctx) -> None:
        """A new extraction attempt invalidates an earlier validation outcome."""
        validation_id = task["adapterData"].get("validationTaskId")
        if not validation_id:
            return
        try:
            validation = ctx.store.get(validation_id)
            if (
                validation is not None
                and validation["state"] in TERMINAL
                and not awaiting_requeue(validation)
            ):
                ctx.store.requeue(
                    [validation_id], reason="extraction-retry", include_succeeded=True
                )
        except StorageError as error:
            raise AdapterError(f"The Task Center store is unavailable: {error}") from error

    @staticmethod
    def _check_space(task: dict, job: dict) -> None:
        """Dispatch-time check of the preview's disk estimate (free space may have gone)."""
        estimated = int(task["adapterData"].get("estimatedOutputBytes") or 0)
        output = Path(job.get("outputPath") or task["adapterData"].get("outputPath") or "/")
        ancestor = output
        while not ancestor.exists() and ancestor != ancestor.parent:
            ancestor = ancestor.parent
        try:
            free = shutil.disk_usage(ancestor).free
        except OSError:
            return
        needed = min(estimated, MIN_FREE_BYTES) if estimated else MIN_FREE_BYTES
        if free < needed:
            raise AdapterError(
                f"Waiting for disk space: {ancestor} has {free / 1024**3:.1f} GiB free; "
                f"this extraction needs at least {needed / 1024**3:.1f} GiB to start "
                f"(about {estimated / 1024**3:.0f} GiB in total)."
            )

    @staticmethod
    def _refine(task: dict, ctx) -> dict | None:
        """Replace the default VRAM/RAM guess with measured peaks of the same workload."""
        request = task["request"]
        key = request.get("workloadKey")
        if not key or task["adapterData"].get("oomRetries"):
            return None  # an OOM backoff's raised request is deliberate evidence; keep it
        cache_key = ("extraction.measurements", key)
        if cache_key not in ctx.cache:
            try:
                ctx.cache[cache_key] = ctx.store.measurements(
                    workload_key=key, kinds=("extraction",), limit=20
                )
            except StorageError:
                ctx.cache[cache_key] = []
        rows = [row for row in ctx.cache[cache_key] if row.get("exitReason") == "ok"]
        from histopilot.taskcenter.estimator import (
            RAM_MARGIN_GB,
            VRAM_MARGIN_FACTOR,
            VRAM_MARGIN_GB,
        )

        changes = {}
        peaks = [row["peakVramGb"] for row in rows if row.get("peakVramGb")]
        if request["lane"] == "gpu" and peaks:
            vram = round(max(peaks) * VRAM_MARGIN_FACTOR + VRAM_MARGIN_GB, 2)
            if abs(vram - float(request.get("vramGb") or 0.0)) > 0.05:
                changes["vramGb"] = vram
        rams = [row["peakPrivateRamGb"] for row in rows if row.get("peakPrivateRamGb")]
        if rams:
            ram = round(max(rams) + RAM_MARGIN_GB, 2)
            if abs(ram - float(request.get("ramGb") or 0.0)) > 0.05:
                changes["ramGb"] = ram
        return {"request": {**request, **changes}} if changes else None

    def on_exit(self, task, exit, ctx):
        stop = exit.get("stopReason")
        if stop == "cancel":
            return _cancelled(task)
        if stop == "pause":
            return outcome("requeue", "paused")
        receipt = _receipt(task, "result.json")
        if receipt is None:
            return self._stopped(
                task, exit, "TRIDENT stopped before recording an outcome; retry to resume."
            )
        state = receipt.get("state")
        measurement = None
        peak = receipt.get("cudaPeakReservedBytes")
        if type(peak) is int and peak > 0:
            measurement = {"peakVramGb": peak / 2**30}
        if state == "succeeded":
            return outcome("succeeded", "ok", measurement=measurement)
        if state == "cancelled" and receipt.get("cancelRequested"):
            marker = _cancel_marker(task)
            if marker is not None and not _stale_cancel(task, marker):
                return _cancelled(task, "Cancelled from the Features stage.")
        if state in {"cancelled", "interrupted"}:
            return outcome(
                "interrupted",
                "interrupted",
                receipt.get("error") or "TRIDENT was stopped without a cancel request.",
            )
        error = receipt.get("error") or "TRIDENT failed; see the worker log."
        text = _attempt_log(task).lower()
        if any(marker in text for marker in OOM_MARKERS):
            return outcome(
                "failed",
                "oom",
                f"TRIDENT ran out of GPU memory. {error}",
                measurement=measurement,
            )
        return outcome("failed", "error", error, measurement=measurement)


class ExtractionValidationAdapter(_JobAdapter):
    def prepare(self, task, ctx):
        _job(task)
        if _clear_stale_cancel(task):
            return {"skip": _cancelled(task, "Cancelled before start.")}
        return None

    def on_exit(self, task, exit, ctx):
        stop = exit.get("stopReason")
        if stop == "cancel":
            return _cancelled(task)
        if stop == "pause":
            return outcome("requeue", "paused")
        report = _receipt(task, "validation.json")
        if report is None:
            return self._stopped(
                task, exit, "Validation stopped before writing its report; retry to rerun it."
            )
        try:
            expected = int(_job(task).get("slideCount") or 0)
        except AdapterError as error:
            return outcome("failed", "error", str(error))
        if report.get("complete") is True:
            return outcome("succeeded", "ok")
        missing = report.get("missingSlides", 0)
        unvalidated = report.get("unvalidatedSlides", 0)
        return outcome(
            "failed",
            "error",
            f"Artifact validation does not confirm all {expected} slides: {missing} slides lack "
            f"valid outputs; {unvalidated} remain unvalidated. Inspect the worker log and "
            "artifact findings, then retry the extraction to resume.",
        )
