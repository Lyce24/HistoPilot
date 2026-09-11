"""Best-effort progress from a bounded TRIDENT log tail; no filesystem or ML imports.

Outer TRIDENT progress bars count slides visited, including skips and failures.
Only the artifact verifier establishes usable output coverage. Concurrent GPU
workers and cached batches have independent counters, never an overall fraction.
"""

import math
import re
from datetime import datetime, timezone

MAX_LOG_CHARACTERS = 65536
MAX_LOG_LINES = 4096
_LABELS = {
    "preparing": "Preparing",
    "segmentation": "Tissue segmentation",
    "coordinates": "Patch coordinates",
    "patch_features": "Patch features",
    "slide_features": "Slide features",
    "validation": "Output validation",
}
_PREFIXES = (
    ("Segmenting tissue", "segmentation"),
    ("Saving tissue coordinates", "coordinates"),
    ("Extracting patch features from coords", "patch_features"),
    ("Extracting slide features using", "slide_features"),
)
_TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_COUNTER = re.compile(r"\|\s*(\d{1,12})\s*/\s*(\d{1,12})\s*\[([^\]]*)")
_TIME = re.compile(r"^(?:(\d+)d\s*)?(\d+):(\d{2})(?::(\d{2}))?$")
_RATE = re.compile(r"(?:^|,)\s*(\d+(?:\.\d+)?)\s*(it/s|s/it)(?:,|$)")
_VALIDATION = re.compile(r"\[validation\] Inspected (\d{1,12})/(\d{1,12}) slides:")
_STAMPED = re.compile(
    r"\[([^\[\]]+)\]\s*(Starting TRIDENT worker|Starting Artifact validation worker|(?:TRIDENT|Worker|Artifact validation) (?:succeeded|failed|cancelled))"
)


def _number(value):
    return (
        float(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        else None
    )


def _timestamp(value):
    number = _number(value)
    if number is not None:
        return number
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)  # noqa: UP017 - supports Python 3.10
    try:
        return parsed.timestamp()
    except (ValueError, OverflowError, OSError):
        return None


def _duration(value):
    match = _TIME.fullmatch(value.strip())
    if match is None:
        return None
    days, first, second, third = match.groups()
    if any(len(value) > 12 for value in (days, first) if value):
        return None
    if int(second) >= 60 or (third is not None and int(third) >= 60):
        return None
    if third is None:
        return int(days or 0) * 86400 + int(first) * 60 + int(second)
    return int(days or 0) * 86400 + int(first) * 3600 + int(second) * 60 + int(third)


def _pipeline(options):
    task = options.get("task", "all")
    stages = ["preparing"]
    if task in {"seg", "all"}:
        stages.append("segmentation")
    if task in {"coords", "all"}:
        stages.append("coordinates")
    if task in {"feat", "all"}:
        stages.append("patch_features")
        if options.get("slide_encoder"):
            stages.append("slide_features")
    stages.append("validation")
    return stages


def _current_slide(stage, text):
    patterns = {
        "segmentation": (r"Segmenting <name=([^>]+)>",),
        "coordinates": (r"Generating patch coords for (.+?)(?:\]|$)",),
        "patch_features": (r"Extracting features from (.+?)(?:\]|$)",),
        "slide_features": (r"Extracting slide features for (.+?)(?:\]|$)",),
    }
    for pattern in patterns.get(stage, ()):
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()[:256]
    return None


def _bar(line):
    stage = next((stage for prefix, stage in _PREFIXES if line.startswith(prefix)), None)
    if stage is None:
        return None
    match = _COUNTER.search(line)
    if match is None:
        return {
            "stage": stage,
            "completed": None,
            "total": None,
            "elapsed": None,
            "eta": None,
            "rate": None,
            "slide": _current_slide(stage, line),
        }
    completed, total = int(match[1]), int(match[2])
    if total <= 0 or completed > total:
        completed = total = None
    timing = match[3]
    clock = timing.split(",", 1)[0]
    elapsed, _, remaining = clock.partition("<")
    rate_match = _RATE.search(timing)
    rate = None
    if rate_match:
        value = float(rate_match[1])
        if value > 0 and math.isfinite(value):
            rate = value if rate_match[2] == "it/s" else 1 / value
            if not math.isfinite(rate):
                rate = None
    return {
        "stage": stage,
        "completed": completed,
        "total": total,
        "elapsed": _duration(elapsed),
        "eta": _duration(remaining),
        "rate": rate,
        "slide": _current_slide(stage, line),
    }


def build_progress(job, logs, now=None):
    """Return UI progress using only supplied job metadata and the latest 64 KiB.

    elapsedSeconds is whole-job wall time; stageElapsedSeconds and etaSeconds
    belong to the latest stage/batch. Counts from extraction bars mean processed
    slides, whereas terminal coverage means slides with validated outputs.
    `_logUpdatedAt`, when supplied by the service, suppresses stale live ETAs.
    """
    options = job.get("spec", {}).get("options", {})
    result = job.get("result") or {}
    state = job.get("state", "starting")
    terminal = state in _TERMINAL
    pipeline = _pipeline(options)
    log = _ANSI.sub("", str(logs or "")[-MAX_LOG_CHARACTERS:])
    lines = re.split(r"[\r\n]", log)[-MAX_LOG_LINES:]
    clock = _timestamp(now)
    if clock is None:
        clock = datetime.now(timezone.utc).timestamp()  # noqa: UP017 - Python 3.10
    starts, finishes = [], []
    validation_start = None
    for match in _STAMPED.finditer(log):
        stamp = _timestamp(match[1])
        if stamp is None:
            continue
        if match[2] == "Starting TRIDENT worker":
            starts.append(stamp)
        elif match[2] == "Starting Artifact validation worker":
            validation_start = stamp
        else:
            finishes.append(stamp)
    started = _timestamp(result.get("startedAt"))
    if started is None:
        started = starts[0] if starts else _timestamp(job.get("createdAt"))
    ended = _timestamp(result.get("finishedAt")) if terminal else None
    if terminal and ended is None:
        ended = _timestamp(job.get("_logUpdatedAt")) if state == "interrupted" else None
        if ended is None:
            ended = finishes[-1] if finishes else _timestamp(job.get("_logUpdatedAt"))
        if ended is None:
            ended = _timestamp(job.get("updatedAt"))
    elapsed = (
        max(0.0, (ended if ended is not None else clock) - started) if started is not None else None
    )

    gpus = options.get("gpus") or [options.get("gpu", 0)]
    effective_gpus = len(set(gpu for gpu in gpus if gpu >= 0)) + gpus.count(-1)
    batched = bool(options.get("wsi_cache")) or effective_gpus > 1
    stage = "preparing"
    current = None
    seen = set()
    finished_stages = set()
    backend_finished = False
    loading = False
    batch_number = None
    validation_counts = None
    for raw in lines:
        line = raw.strip()
        batch = re.search(r"\[CONSUMER\] Processing batch (\d{1,12}):", line)
        if batch:
            batched = True
            batch_number = int(batch[1])
            current = None
            stage = "preparing"
            loading = False
        if "Starting Artifact validation worker" in line:
            stage, current, loading = "validation", None, False
            seen.add(stage)
            backend_finished = True
        if re.search(r"(?:TRIDENT|Worker) succeeded \(exit 0\)", line):
            backend_finished = True
        validation = _VALIDATION.search(line)
        if validation:
            done, total = int(validation[1]), int(validation[2])
            if 0 <= done <= total and total > 0:
                stage = "validation"
                current = None
                validation_counts = (done, total)
                seen.add(stage)
                backend_finished = True
        parsed = _bar(line)
        if parsed:
            if parsed["stage"] not in pipeline:
                continue
            stage, current, loading = parsed["stage"], parsed, False
            seen.add(stage)
            if parsed["completed"] is not None and parsed["completed"] == parsed["total"]:
                finished_stages.add(stage)
        # Downloads are preparation evidence only, never slide progress.
        if re.search(r"(?:Fetching \d+ files:|Downloading|Loading checkpoint shards:)", line):
            if stage == "preparing" or (
                current
                and current["completed"] is not None
                and current["completed"] == current["total"]
            ):
                loading = True
                if current and stage in pipeline:
                    index = pipeline.index(stage)
                    if index + 1 < len(pipeline) - 1:
                        stage = pipeline[index + 1]
                        current = None
                        seen.add(stage)

    if backend_finished and stage != "validation":
        stage, current = "validation", None
        seen.add(stage)
    completed = current["completed"] if current else None
    total = current["total"] if current else None
    stage_elapsed = current["elapsed"] if current else None
    eta = current["eta"] if current else None
    rate = current["rate"] if current else None
    current_slide = current["slide"] if current else None
    scope = (
        "batch"
        if batched and stage not in {"preparing", "validation"}
        else "stage"
        if stage != "preparing"
        else None
    )
    warnings = []
    label = _LABELS[stage]
    detail = "Waiting for TRIDENT to report progress."
    if loading:
        label = "Loading model files"
        detail = (
            f"Preparing {_LABELS[stage].lower()}."
            if stage != "preparing"
            else "Preparing model dependencies and checkpoints."
        )
    elif stage == "validation":
        if validation_counts:
            completed, total = validation_counts
        stage_elapsed = (
            max(0.0, (ended or clock) - validation_start) if validation_start is not None else None
        )
        eta = rate = None
        detail = (
            f"{completed} of {total} slides inspected for usable outputs."
            if completed is not None
            else "Checking generated files and slide coverage."
        )
    elif completed is not None:
        detail = f"{completed} of {total} slides processed in this {'worker batch' if scope == 'batch' else 'stage'}."
        if options.get("skip_errors"):
            warnings.append(
                "Processed counts include skipped or failed slides; output validation checks usable files."
            )
    if scope == "batch":
        warnings.append(
            "Workers and cached batches report separate counts; this is the latest batch, not overall progress."
        )
        if batch_number is not None:
            detail += f" Cache batch {batch_number}."

    coverage = _number(result.get("completedSlides"))
    if terminal and coverage is not None:
        stage = "validation"
        scope = "stage"
        completed = max(0, int(coverage))
        known_total = _number(job.get("slideCount"))
        total = int(known_total) if known_total is not None and known_total >= completed else None
        label = _LABELS[stage]
        detail = (
            f"{completed} of {total} slides have validated outputs."
            if total is not None
            else f"{completed} slides have validated outputs."
        )
        current_slide = None
        backend_finished = True
        stage_elapsed = (
            max(0.0, (ended or clock) - validation_start) if validation_start is not None else None
        )
        rate = None
    if state == "succeeded":
        stage, scope, label = "validation", "stage", "Extraction complete"
        if coverage is None:
            known_total = _number(job.get("slideCount"))
            completed = total = int(known_total) if known_total is not None else None
        detail = (
            f"{completed} slides have validated outputs."
            if completed is not None
            else "Output validation completed."
        )
    elif state in {"failed", "cancelled", "interrupted"}:
        label = f"{'Cancelled' if state == 'cancelled' else 'Interrupted' if state == 'interrupted' else 'Failed'} during {_LABELS[stage].lower()}"
    elif state == "cancelling":
        label = "Stopping extraction"
        detail = "Waiting for the current worker to stop safely. " + detail

    if terminal or state == "cancelling" or (completed is not None and completed == total):
        eta = None
    log_updated = _timestamp(job.get("_logUpdatedAt"))
    stale_after = max(60.0, 2 / rate) if rate else 60.0
    if eta is not None and log_updated is not None and clock - log_updated > stale_after:
        eta = None
        warnings.append("ETA is waiting for a fresh progress update.")
    if eta is not None and (completed is None or completed <= 0):
        eta = None
    percent = (
        round(100 * completed / total, 2)
        if completed is not None and total is not None and total > 0
        else None
    )
    statuses = []
    for item in pipeline:
        status = "pending"
        if state == "succeeded":
            status = "complete"
        elif item == stage:
            status = "stopped" if terminal else "active"
        elif item == "preparing" and stage != "preparing":
            status = "complete"
        elif backend_finished and item != "validation":
            status = "complete"
        elif batched and item in seen:
            status = "stopped" if terminal else "active"
        elif not batched and (
            item in finished_stages or pipeline.index(item) < pipeline.index(stage)
        ):
            status = "complete"
        statuses.append({"id": item, "label": _LABELS[item], "status": status})
    return {
        "stage": stage,
        "stages": statuses,
        "label": label,
        "detail": detail,
        "completed": completed,
        "total": total,
        "unit": "slides",
        "percent": percent,
        "currentSlide": current_slide,
        "elapsedSeconds": elapsed,
        "stageElapsedSeconds": stage_elapsed,
        "etaSeconds": eta,
        "ratePerSecond": rate,
        "scope": scope,
        "warnings": warnings,
    }
