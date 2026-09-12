"""Bounded, read-only resource observations recorded by a batch's own worker."""

import json
import math
import os
import stat
from collections import deque
from datetime import datetime

from histopilot.storage.project_lock import StorageError, _reject_symlink_components
from histopilot.storage.scientific import ScientificStore

MAX_RESOURCE_BYTES = 4 * 1024**2
DISPLAY_RESOURCE_ROWS = 360


def _finite(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Resource counters must be finite.")
    return number


def _number(value, *, maximum=None):
    if value is not None and (
        type(value) not in {int, float}
        or not math.isfinite(value)
        or value < 0
        or (maximum is not None and value > maximum)
    ):
        raise ValueError("Invalid resource counter.")
    return value


def _sample(content: bytes, run_ids: set[str]) -> dict:
    value = json.loads(content, parse_float=_finite, parse_constant=_finite)
    if not isinstance(value, dict) or not isinstance(value.get("at"), str):
        raise ValueError("Invalid resource observation.")
    if datetime.fromisoformat(value["at"]).tzinfo is None:
        raise ValueError("Resource observations need a timezone.")
    host = value.get("host")
    if not isinstance(host, dict):
        raise ValueError("Invalid host counters.")
    if type(host.get("cpuCount")) is not int or host["cpuCount"] < 1:
        raise ValueError("Invalid CPU count.")
    normalized_host = {"cpuCount": host["cpuCount"]}
    for key in ("totalRamGb", "availableRamGb"):
        normalized_host[key] = _number(host.get(key))
        if normalized_host[key] is None:
            raise ValueError("Missing host memory counter.")
    if (
        normalized_host["totalRamGb"] <= 0
        or normalized_host["availableRamGb"] > normalized_host["totalRamGb"]
    ):
        raise ValueError("Invalid host memory capacity.")
    if "cpuUtilizationPercent" in host:
        normalized_host["cpuUtilizationPercent"] = _number(
            host["cpuUtilizationPercent"], maximum=100
        )
    for key in ("bootId", "kernel"):
        if not isinstance(host.get(key), str):
            raise ValueError("Invalid host identity.")
        normalized_host[key] = host[key]
    gpus, runs = value.get("gpus"), value.get("runs")
    if not isinstance(gpus, list) or not isinstance(runs, list):
        raise ValueError("Invalid resource devices or runs.")
    normalized_gpus = []
    for gpu in gpus:
        if not isinstance(gpu, dict) or type(gpu.get("index")) is not int or gpu["index"] < 0:
            raise ValueError("Invalid GPU index.")
        normalized_gpu = {"index": gpu["index"]}
        for key in ("name", "uuid", "driverVersion"):
            if not isinstance(gpu.get(key), str):
                raise ValueError("Invalid GPU identity.")
            normalized_gpu[key] = gpu[key]
        for key in ("totalMemoryGb", "usedMemoryGb", "freeMemoryGb"):
            normalized_gpu[key] = _number(gpu.get(key))
        normalized_gpu["utilizationPercent"] = _number(gpu.get("utilizationPercent"), maximum=100)
        normalized_gpus.append(normalized_gpu)
    normalized_runs = []
    for run in runs:
        if (
            not isinstance(run, dict)
            or not isinstance(run.get("runId"), str)
            or run["runId"] not in run_ids
            or type(run.get("pid")) is not int
            or run["pid"] <= 1
        ):
            raise ValueError("Invalid observed run identity.")
        rss = _number(run.get("rssGb"))
        if rss is None:
            raise ValueError("Missing run memory counter.")
        normalized_runs.append({"runId": run["runId"], "pid": run["pid"], "rssGb": rss})
    result = {
        "at": value["at"],
        "host": normalized_host,
        "gpus": normalized_gpus,
        "runs": normalized_runs,
    }
    if "gpuProbeError" in value:
        if not isinstance(value["gpuProbeError"], str):
            raise ValueError("Invalid GPU probe message.")
        result["gpuProbeError"] = value["gpuProbeError"][:2000]
    return result


def _tail(path) -> tuple[bytes, bool]:
    """Snapshot a regular file's tail without following aliases or reading a growing log."""
    ScientificStore._regular(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("Resource history must be a regular file without aliases.")
        start = max(0, info.st_size - MAX_RESOURCE_BYTES)
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            skip_partial = False
            if start:
                stream.seek(start - 1)
                skip_partial = stream.read(1) != b"\n"
            content = stream.read(info.st_size - start)
        if len(content) != info.st_size - start:
            raise ValueError("Resource history changed while reading.")
        if skip_partial:
            content = content.partition(b"\n")[2]
        return content, start > 0
    finally:
        os.close(descriptor)


def training_resources(store, batch_id: str) -> dict:
    """Authorize the frozen batch first. Never probe hardware or change worker state.

    totalRows counts valid complete observations in the bounded read window. If
    earlier bytes are omitted, totalRowsIsLowerBound explicitly marks that count.
    """
    if (
        not batch_id
        or batch_id in {".", ".."}
        or any(character in batch_id for character in ("/", "\\", "\0"))
    ):
        raise StorageError("The frozen batch identity is invalid.", "TRAINING_BATCH_INVALID", 409)
    batch = store.get_configuration(batch_id)
    manifest = batch["manifest"]
    if manifest.get("kind") != "mil-batch":
        raise StorageError("Choose a frozen development batch.", "INVALID_BATCH", 422)
    run_ids = {row["id"] for row in manifest["runs"]}
    response = {"batchId": batch_id, "rows": [], "totalRows": 0, "truncated": False}
    path = store.folder / "training" / batch_id / "telemetry.jsonl"
    try:
        _reject_symlink_components(path)
        if not path.exists():
            return response
        content, byte_truncated = _tail(path)
        rows = deque(maxlen=DISPLAY_RESOURCE_ROWS)
        total, skipped = 0, 0
        incomplete = bool(content and not content.endswith(b"\n"))
        # A final line becomes durable only when its newline has been written.
        for line in content.split(b"\n")[:-1]:
            if not line.strip():
                continue
            try:
                rows.append(_sample(line, run_ids))
                total += 1
            except (ValueError, TypeError, OverflowError, RecursionError):
                skipped += 1
        result = {
            **response,
            "rows": list(rows),
            "totalRows": total,
            "truncated": byte_truncated or total > DISPLAY_RESOURCE_ROWS,
        }
        warnings = []
        if byte_truncated:
            result["totalRowsIsLowerBound"] = True
            warnings.append(
                "Only the recent resource history was read; the sample count excludes older file data."
            )
        if skipped:
            warnings.append(f"Skipped {skipped} invalid resource observations.")
        if incomplete:
            warnings.append("The latest incomplete resource observation is not shown yet.")
        if warnings:
            result["warning"] = " ".join(warnings)
        return result
    except (OSError, ValueError, TypeError, OverflowError, RecursionError):
        return {
            **response,
            "warning": "Resource history is unavailable because its file is invalid or cannot be read safely. Run status and saved results remain available.",
        }
