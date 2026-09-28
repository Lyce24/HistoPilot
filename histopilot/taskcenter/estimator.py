"""Per-task resource estimates and the parallel GPU task suggestion (Task Center Phase 0).

Evidence is read, never written: completed runs in project training folders
(``plan.json``, ``state.json``, ``telemetry.jsonl``) and Task Center measurement rows.
Unreadable or invalid files are skipped and counted, never raised. There is no code
snapshot gate: a deliberate ``KEY_VERSION`` bump replaces it. Units are binary GiB
named ``Gb``, like the runtime API.
"""

import hashlib
import json
import math
import os
import stat
import threading
from bisect import bisect_left
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path
from statistics import median

from histopilot.storage.project_lock import _reject_symlink_components

KEY_VERSION = 2
SUGGESTION_VERSION = 2
# Only fields that change tensor shapes or allocator behaviour. Bag sizes are compared
# per fold by scaling, so nearby folds and other cohorts share evidence.
MEMORY_FIELDS = (
    "model",
    "batchSize",
    "evalBatchSize",
    "precision",
    "embedDim",
    "attentionDim",
    "numFcLayers",
    "gatedAttention",
    "gradientCheckpointing",
    "nnmilFeatureSampling",
    "nnmilWindowStrideDivisor",
    "bagCurriculum",
)
MAX_PARALLEL = 16
# Without any throughput evidence a new workload starts no wider than this.
UNMEASURED_PARALLEL = 4
MEASURED_SCALE_LIMIT = 2.0
# Observations whose scale factor is within 10% of the nearest one are "nearest".
NEAREST_TOLERANCE = 1.1
VRAM_MARGIN_FACTOR = 1.15
VRAM_MARGIN_GB = 0.3
RAM_MARGIN_GB = 0.5
# Measured private RAM uses this percentile of a workload's peaks, not their maximum.
RAM_PERCENTILE = 0.9
THROUGHPUT_GAIN = 0.05
BUSY_UTILIZATION = 80.0
DEFAULT_TASK = {"vramGb": 3.0, "ramGb": 6.0, "cpuCores": 3.0}
DEFAULT_THREADS = 2
DEFAULT_WORKERS = 2
DEFAULT_CPU_RESERVE = 2

MAX_JSON_BYTES = 64 * 1024**2
MAX_TELEMETRY_BYTES = 64 * 1024**2
MAX_DIRECTORY_ENTRIES = 4096
_CACHE_ENTRIES = 128
_GIB = 1024**3
_INVALID = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    IndexError,
    AttributeError,
    OverflowError,
    RecursionError,
    ZeroDivisionError,
)


def _finite(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _positive(value) -> bool:
    return _finite(value) and value > 0


def _count(value) -> bool:
    return type(value) is int and value >= 1


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _reject_constant(value):
    raise ValueError(f"Non-finite number {value} in evidence.")


# ---------------------------------------------------------------------------
# Workloads


def workload_key(workload: dict) -> str:
    """Identity of a memory class: memory fields, feature dimension and loading policy."""
    identity = {
        "version": KEY_VERSION,
        **{field: workload.get(field) for field in MEMORY_FIELDS},
        "featureDimension": workload.get("featureDimension"),
        "loadingPolicy": workload.get("loadingPolicy"),
    }
    return hashlib.sha256(_canonical(identity).encode()).hexdigest()[:24]


def _recipe(recipe: dict) -> dict:
    from histopilot.schemas.development import TrainingRecipe

    parsed = TrainingRecipe.model_validate(recipe, context={"legacy": True})
    # Frozen legacy recipes omit newer defaults; identity needs their effective values.
    values = {key: getattr(parsed, key) for key in TrainingRecipe.model_fields}
    values["model"] = values["model"].lower()
    values["evalBatchSize"] = values["evalBatchSize"] or values["batchSize"]
    return values


def fold_workloads(recipe: dict, rows: list[dict], files: dict, loading_policy: str) -> dict:
    """Bound one fold's padded training and evaluation inputs from frozen feature headers.

    Assessment bag sizes only budget memory; they never set the fitting-only automatic
    patch cap. Raises ValueError for incomplete or ambiguous folds instead of guessing.
    """
    from histopilot.schemas.nnmil import resolve_nnmil_recipe

    values = _recipe(recipe)
    rows = [
        row
        for row in rows
        if row.get("partition") in {"train", "val", "test"}
        and row.get("phase") != "final"
        and row.get("pool") != "external_test"
    ]
    counts = {role: [] for role in ("train", "val", "test")}
    dimensions, seen = set(), set()
    for row in rows:
        identity = row["slideId"]
        if identity in seen:
            raise ValueError("A slide occurs more than once in a runtime planning fold.")
        seen.add(identity)
        entry = files.get(identity)
        if not isinstance(entry, dict):
            raise ValueError("Runtime planning requires feature headers for every eligible slide.")
        count, dimension = entry.get("patchCount"), entry.get("dimensions")
        if not _count(count) or not _count(dimension):
            raise ValueError(
                "Runtime planning requires positive patch counts and feature dimensions."
            )
        counts[row["partition"]].append(count)
        dimensions.add(dimension)
    if not counts["train"] or not (counts["val"] or counts["test"]) or len(dimensions) != 1:
        raise ValueError(
            "Runtime planning needs fitting/evaluation slides with compatible features."
        )
    effective, _ = resolve_nnmil_recipe(values, rows, files)
    cap = effective["bagCurriculumEnd"] if effective["bagCurriculum"] else effective["bagSize"]
    training = min(max(counts["train"]), cap) if cap else max(counts["train"])
    evaluation = max(counts["val"] + counts["test"])
    if effective["evalBagSize"]:
        evaluation = min(evaluation, effective["evalBagSize"])
    workload = {
        **{field: values[field] for field in MEMORY_FIELDS},
        "featureDimension": next(iter(dimensions)),
        "trainingPatches": training,
        "evaluationPatches": evaluation,
        "sourcePatches": max(counts["train"] + counts["val"] + counts["test"]),
        "loadingPolicy": loading_policy,
    }
    return {"key": workload_key(workload), **workload}


def gpu_estimate(workload: dict, *, evaluation_only: bool = False) -> float:
    """Unmeasured VRAM formula: full-dimensional bags, live activations and optimizer state.

    Even AMP keeps the source bag and stable pooling in float32. nnMIL evaluates its
    feature windows sequentially, so their number is not a memory multiplier.
    """
    dim, embed, attention = (
        workload["featureDimension"],
        workload["embedDim"],
        workload["attentionDim"],
    )
    activation_bytes = 4 if workload["precision"] == "32-true" else 2
    layers = workload["numFcLayers"]
    if workload["model"] == "nnmil":
        train_width, eval_width = 8 * attention, 4 * attention
        parameters = 2 * dim * attention
    elif workload["model"] == "abmil":
        train_width, eval_width = 4 * embed * layers + 8 * attention, 2 * embed + 4 * attention
        parameters = dim * embed + embed * embed * layers + 2 * embed * attention
    else:
        train_width, eval_width = 4 * embed * layers, 2 * embed
        parameters = dim * embed + embed * embed * layers
    train = (
        workload["batchSize"]
        * workload["trainingPatches"]
        * (8 * dim + activation_bytes * train_width)
    )
    evaluation = (
        workload["evalBatchSize"]
        * workload["evaluationPatches"]
        * (8 * dim + activation_bytes * eval_width)
    )
    # Adam state, gradients, allocator fragmentation, context and kernels.
    largest = evaluation if evaluation_only else max(train, evaluation)
    return 1.0 + 1.25 * (largest + parameters * 20) / _GIB


def ram_formula(workload: dict, workers: int = DEFAULT_WORKERS) -> float:
    """Unmeasured private RAM per task: interpreter plus two persistent loader pools."""
    patches = workload["trainingPatches"] + workload["evaluationPatches"]
    buffers = (2 * workers + 2) * patches * workload["featureDimension"] * 4 / _GIB
    return max(3.0, 2.5 + buffers)


# ---------------------------------------------------------------------------
# Evidence


class Observations(list):
    """Observation rows plus ``diagnostics`` about what the scan skipped."""

    def __init__(self, rows=(), diagnostics=None):
        super().__init__(rows)
        self.diagnostics = diagnostics or {"batches": 0, "used": 0, "skipped": 0, "errors": []}


_cache: OrderedDict = OrderedDict()
_cache_lock = threading.Lock()


def _stamp(path: Path):
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _read_bytes(path: Path, budget: list[int], maximum: int) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"{path.name} is not a regular file.")
        if info.st_size > maximum or info.st_size > budget[0]:
            raise ValueError(f"{path.name} exceeds the evidence read budget.")
        budget[0] -= info.st_size
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            # A growing telemetry file is read up to its size at open; the tail is skipped.
            return handle.read(info.st_size)
    finally:
        os.close(descriptor)


def _read_json(path: Path, budget: list[int], maximum: int = MAX_JSON_BYTES) -> dict:
    value = json.loads(_read_bytes(path, budget, maximum), parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object.")
    return value


def _instant(value) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


def _fold_task_id(batch_folder: Path, run_id: str) -> str:
    # Same formula as taskcenter.ids.fold_task_id, so a managed fold's measurement row
    # merges with its training observation instead of counting the run twice.
    digest = hashlib.sha256("\0".join(("mil-fold", str(batch_folder), run_id)).encode())
    return "task-" + digest.hexdigest()[:32]


def _telemetry(path: Path, budget: list[int], batch_id: str, indices: set[int]) -> dict | None:
    """Per-run private RAM from host-used deltas and GPU utilisation per live-run level."""
    samples = []
    for line in _read_bytes(path, budget, MAX_TELEMETRY_BYTES).splitlines():
        try:
            row = json.loads(line, parse_constant=_reject_constant)
            host, runs = row["host"], row.get("runs") or []
            used = host["totalRamGb"] - host["availableRamGb"]
            if not isinstance(runs, list) or not _finite(used):
                continue
            utilization = [
                gpu["utilizationPercent"]
                for gpu in row.get("gpus") or []
                if isinstance(gpu, dict)
                and _finite(gpu.get("utilizationPercent"))
                and (not indices or gpu.get("index") in indices)
            ]
            samples.append((len(runs), used, utilization))
        except _INVALID:
            continue
    first = next((index for index, sample in enumerate(samples) if sample[0] >= 1), None)
    if first is None:
        return None
    idle = [used for runs, used, _ in samples[:first] if runs == 0]
    baseline = idle[-1] if idle else min(used for _, used, _ in samples)
    per_run = [(used - baseline) / runs for runs, used, _ in samples if runs >= 1]
    levels: dict[str, list[float]] = {}
    for runs, _, utilization in samples:
        if runs >= 1 and utilization:
            levels.setdefault(str(runs), []).append(sum(utilization) / len(utilization))
    return {
        "batchId": batch_id,
        "ramPerRunGb": max(0.0, median(per_run)),
        "ramSamples": len(per_run),
        "gpuUtilization": {
            level: {"meanPercent": sum(values) / len(values), "samples": len(values)}
            for level, values in levels.items()
        },
    }


def _batch_observations(folder: Path, budget: list[int]) -> tuple[list[dict], int, list[str]]:
    plan = _read_json(folder / "plan.json", budget)
    state = _read_json(folder / "state.json", budget)
    batch_id = plan.get("batchId") or folder.name
    if state.get("batchId", batch_id) != batch_id:
        raise ValueError("state.json belongs to another batch.")
    recipes = {row["id"]: row["recipe"] for row in plan["configurations"]}
    planned = {row["id"]: row for row in plan["runs"]}
    memberships = plan.get("memberships") or {}
    data = plan.get("data") or {}
    files, policy = data.get("featureFiles") or {}, data.get("loadingPolicy")
    devices = {
        row.get("index"): row
        for row in (state.get("provenance") or {}).get("gpus") or []
        if isinstance(row, dict)
    }
    workloads, rows, skipped, errors = {}, [], 0, []
    for current in state.get("runs") or []:
        try:
            if not isinstance(current, dict) or current.get("status") != "completed":
                continue
            result, gpu = current.get("result"), current.get("gpu")
            if not isinstance(result, dict) or type(gpu) is not int:
                continue
            peak = result.get("cudaPeakReservedBytes")
            if not _positive(peak):
                continue
            run = planned[current["id"]]
            identity = (run["candidateId"], run["splitPlanId"])
            if identity not in workloads:
                recipe = result.get("effectiveRecipe") or recipes[run["candidateId"]]
                workloads[identity] = fold_workloads(
                    recipe, memberships[run["splitPlanId"]], files, policy
                )
            workload = workloads[identity]
            started, finished = current.get("startedAt"), current.get("finishedAt")
            start, end = _instant(started), _instant(finished)
            wall = end - start if start is not None and end is not None and end > start else None
            epochs = (
                result.get("epochsCompleted") if _count(result.get("epochsCompleted")) else None
            )
            attempt = current.get("attempt", 1)
            device = devices.get(gpu) or {}
            rows.append(
                {
                    "source": "training",
                    "batchId": batch_id,
                    "runId": current["id"],
                    "taskId": _fold_task_id(folder, current["id"]),
                    "workloadKey": workload["key"],
                    "workload": workload,
                    "gpuIndex": gpu,
                    "gpuUuid": device.get("uuid"),
                    "gpuName": device.get("name"),
                    "peakVramGb": peak / _GIB,
                    "peakPrivateRamGb": None,
                    "meanCpuCores": None,
                    "epochs": epochs,
                    "startedAt": started,
                    "finishedAt": finished,
                    "wallSeconds": wall,
                    "secondsPerEpoch": wall / epochs if wall and epochs else None,
                    "resumed": bool(
                        result.get("resumedFrom")
                        or result.get("assessmentOnlyResume")
                        or (type(attempt) is int and attempt > 1)
                    ),
                    "meanConcurrency": None,
                    "ok": True,
                    "telemetry": None,
                }
            )
        except _INVALID as error:
            skipped += 1
            run_id = current.get("id") if isinstance(current, dict) else None
            errors.append(f"{batch_id}/{run_id}: {error}")
    if not rows:
        return rows, skipped, errors
    telemetry = None
    try:
        indices = {row["gpuIndex"] for row in rows}
        telemetry = _telemetry(folder / "telemetry.jsonl", budget, batch_id, indices)
    except FileNotFoundError:
        pass
    except _INVALID as error:
        skipped += 1
        errors.append(f"{batch_id}/telemetry.jsonl: {error}")
    for row in rows:
        row["telemetry"] = telemetry
    return rows, skipped, errors


def observations_from_training(
    project_folder: Path, *, max_batches: int = 32, max_bytes: int = 256 * 2**20
) -> Observations:
    """Completed GPU runs of the most recent batches in ``<project>/training``.

    Reads only batch-level ``plan.json``, ``state.json`` and ``telemetry.jsonl``; never
    ``runs/<id>/plan.json``. Results are cached per batch by file stamps, so repeated
    calls (every runner tick, every draft edit) only stat the files.
    """
    diagnostics = {"batches": 0, "used": 0, "skipped": 0, "errors": []}
    result = Observations(diagnostics=diagnostics)
    try:
        # Normalized, not resolved: managed storage has no symlinks, so this equals the
        # resolved folder that Task Center ids are derived from.
        training = Path(os.path.abspath(project_folder)) / "training"
        _reject_symlink_components(training)
        entries = []
        with os.scandir(training) as scan:
            for index, entry in enumerate(scan):
                if index >= MAX_DIRECTORY_ENTRIES:
                    break
                if entry.is_dir(follow_symlinks=False):
                    entries.append((entry.stat(follow_symlinks=False).st_mtime_ns, entry.name))
    except FileNotFoundError:
        return result
    except _INVALID as error:
        diagnostics["skipped"] += 1
        diagnostics["errors"].append(str(error))
        return result
    budget = [max_bytes]
    for _, name in sorted(entries, reverse=True)[:max_batches]:
        folder = training / name
        diagnostics["batches"] += 1
        stamp = tuple(
            _stamp(folder / file) for file in ("plan.json", "state.json", "telemetry.jsonl")
        )
        with _cache_lock:
            cached = _cache.get(str(folder))
            hit = cached is not None and cached[0] == stamp
            if hit:
                _cache.move_to_end(str(folder))
        if hit:
            rows, skipped, errors = cached[1]
        else:
            try:
                rows, skipped, errors = _batch_observations(folder, budget)
            except FileNotFoundError:
                continue
            except _INVALID as error:
                diagnostics["skipped"] += 1
                diagnostics["errors"].append(f"{name}: {error}")
                continue
            with _cache_lock:
                _cache[str(folder)] = (stamp, (rows, skipped, errors))
                _cache.move_to_end(str(folder))
                while len(_cache) > _CACHE_ENTRIES:
                    _cache.popitem(last=False)
        diagnostics["used"] += 1 if rows else 0
        diagnostics["skipped"] += skipped
        diagnostics["errors"].extend(errors)
        # Copies: annotate_concurrency writes meanConcurrency into the rows it receives.
        result.extend(dict(row) for row in rows)
    diagnostics["errors"] = diagnostics["errors"][:20]
    return result


def observations_from_measurements(rows) -> list[dict]:
    """Observations from ``TaskStore.measurements()`` rows (duck-typed camelCase dicts)."""
    observations = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        workload = row.get("workload")
        if isinstance(workload, str):
            try:
                workload = json.loads(workload, parse_constant=_reject_constant)
            except ValueError:
                workload = None
        if not isinstance(workload, dict):
            workload = None
        key = row.get("workloadKey") or (workload_key(workload) if workload else None)
        attempt = row.get("attempt")
        observations.append(
            {
                "source": "measurement",
                "batchId": None,
                "runId": None,
                "taskId": row.get("taskId"),
                "kind": row.get("kind"),
                "workloadKey": key if isinstance(key, str) else None,
                "workload": workload,
                "gpuIndex": row.get("gpuIndex") if type(row.get("gpuIndex")) is int else None,
                "gpuUuid": row.get("gpuUuid") or None,
                "gpuName": row.get("gpuName") or None,
                **{
                    name: row.get(name) if _positive(row.get(name)) else None
                    for name in (
                        "peakVramGb",
                        "peakPrivateRamGb",
                        "meanCpuCores",
                        "wallSeconds",
                        "secondsPerEpoch",
                        "meanConcurrency",
                    )
                },
                "epochs": row.get("epochs") if _count(row.get("epochs")) else None,
                "startedAt": row.get("startedAt"),
                "finishedAt": row.get("finishedAt"),
                "resumed": type(attempt) is int and attempt > 1,
                "ok": row.get("exitReason") in (None, "ok"),
                "telemetry": None,
            }
        )
    return observations


def merge_observations(training: list[dict], measured: list[dict]) -> list[dict]:
    """Fold a managed fold's measurement into its training observation (one run, not two)."""
    merged = list(training)
    by_task = {row["taskId"]: row for row in training if row.get("taskId")}
    for row in measured:
        target = by_task.get(row.get("taskId")) if row.get("ok") else None
        if target is None:
            merged.append(row)
            continue
        for name in ("peakPrivateRamGb", "meanCpuCores", "meanConcurrency"):
            if target.get(name) is None and row.get(name) is not None:
                target[name] = row[name]
    return merged


def annotate_concurrency(observations) -> None:
    """Set meanConcurrency to the time-weighted count of runs sharing the GPU (itself included).

    Intervals from every observation count, across batches, projects and measurements.
    A value the runner measured directly is kept.
    """
    devices: dict[str, list] = {}
    for row in observations:
        if not isinstance(row, dict):
            continue
        device = row.get("gpuUuid") or (
            f"index:{row['gpuIndex']}" if type(row.get("gpuIndex")) is int else None
        )
        start, end = _instant(row.get("startedAt")), _instant(row.get("finishedAt"))
        if device and start is not None and end is not None and end > start:
            devices.setdefault(device, []).append((start, end, row))
    for intervals in devices.values():
        intervals.sort(key=lambda item: item[0])
        starts = [item[0] for item in intervals]
        longest = max(end - start for start, end, _ in intervals)
        for start, end, row in intervals:
            if row.get("meanConcurrency") is not None:
                continue
            first = bisect_left(starts, start - longest)
            last = bisect_left(starts, end)
            overlap = sum(
                max(0.0, min(end, other_end) - max(start, other_start))
                for other_start, other_end, _ in intervals[first:last]
            )
            row["meanConcurrency"] = round(overlap / (end - start), 3)


def gather_observations(
    project_folders, *, measurements=None, max_batches: int = 32, max_bytes: int = 256 * 2**20
) -> Observations:
    """Training evidence of several projects plus measurement rows, merged and annotated."""
    training = Observations()
    seen = set()
    for folder in project_folders:
        key = os.path.abspath(folder)
        if key in seen:
            continue
        seen.add(key)
        rows = observations_from_training(folder, max_batches=max_batches, max_bytes=max_bytes)
        training.extend(rows)
        for name in ("batches", "used", "skipped"):
            training.diagnostics[name] += rows.diagnostics[name]
        training.diagnostics["errors"].extend(rows.diagnostics["errors"])
    merged = Observations(
        merge_observations(training, observations_from_measurements(measurements)),
        training.diagnostics,
    )
    annotate_concurrency(merged)
    return merged


# ---------------------------------------------------------------------------
# Estimates


def _usable(row) -> bool:
    return isinstance(row, dict) and row.get("ok", True) and isinstance(row.get("workloadKey"), str)


def _scale(request: dict, observed) -> float | None:
    if not isinstance(observed, dict) or not all(
        _count(values.get(name))
        for values in (request, observed)
        for name in ("trainingPatches", "evaluationPatches")
    ):
        return None
    return max(
        1.0,
        request["trainingPatches"] / observed["trainingPatches"],
        request["evaluationPatches"] / observed["evaluationPatches"],
    )


def _vram(workload: dict, same: list[dict], gpu_name) -> tuple[float, str, list[float]]:
    candidates = []
    for row in same:
        scale = _scale(workload, row.get("workload"))
        if _positive(row.get("peakVramGb")) and scale is not None:
            candidates.append((row, scale))
    if gpu_name:
        named = [item for item in candidates if item[0].get("gpuName") == gpu_name]
        candidates = named or candidates
    peaks = [row["peakVramGb"] for row, _ in candidates]
    near = [item for item in candidates if item[1] <= MEASURED_SCALE_LIMIT]
    if near:
        # Upper-bound (same or larger) folds first: scaling a smaller fold's peak by its
        # bag ratio would overstate a size that was actually measured.
        nearest = min(scale for _, scale in near) * NEAREST_TOLERANCE
        peak = max(row["peakVramGb"] * scale for row, scale in near if scale <= nearest)
        return peak * VRAM_MARGIN_FACTOR + VRAM_MARGIN_GB, "measured", peaks
    try:
        formula = gpu_estimate(workload)
    except _INVALID:
        return DEFAULT_TASK["vramGb"], "default", peaks
    ratios = []
    for row, _ in candidates:
        try:
            ratios.append(row["peakVramGb"] / gpu_estimate(row["workload"]))
        except _INVALID:
            continue
    if ratios:
        calibrated = formula * median(ratios)
        return calibrated * VRAM_MARGIN_FACTOR + VRAM_MARGIN_GB, "calibrated", peaks
    return formula, "formula", peaks


def _percentile(values: list[float], fraction: float) -> float:
    """Linear interpolation between closest ranks (NumPy's default method)."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _ram(workload: dict, same: list[dict], workers: int) -> tuple[float, str]:
    measured = [
        (row["peakPrivateRamGb"], _scale(workload, row.get("workload")))
        for row in same
        if _positive(row.get("peakPrivateRamGb"))
    ]
    scales = [scale for _, scale in measured if scale is not None]
    peaks = [peak for peak, _ in measured]
    if scales:
        # Loader buffers grow with bag size and a workload's evidence spans cohorts, so, as
        # for VRAM, only the folds nearest in size (or larger) speak for this one.
        nearest = min(scales) * NEAREST_TOLERANCE
        peaks = [peak for peak, scale in measured if scale is not None and scale <= nearest]
    if peaks:
        # Not the maximum: one inflated reading (a fold that alone mapped a feature pack, a
        # transient spike) would otherwise set every later request of the workload. Folds of
        # one workload and size stay well within twice their median, so with three or more
        # readings any reading above that is set aside; the 90th percentile leans high.
        if len(peaks) >= 3:
            typical = median(peaks)
            peaks = [peak for peak in peaks if peak <= 2 * typical]
        return _percentile(peaks, RAM_PERCENTILE) + RAM_MARGIN_GB, "measured"
    batches = {}
    for row in same:
        telemetry = row.get("telemetry")
        if isinstance(telemetry, dict) and _positive(telemetry.get("ramPerRunGb")):
            batches[telemetry.get("batchId")] = telemetry["ramPerRunGb"]
    if batches:
        return median(batches.values()) + RAM_MARGIN_GB, "telemetry"
    try:
        return ram_formula(workload, workers) + RAM_MARGIN_GB, "formula"
    except _INVALID:
        return DEFAULT_TASK["ramGb"], "default"


def estimate(
    workload: dict,
    observations: list[dict],
    host: dict | None = None,
    *,
    gpu_name: str | None = None,
    cpu_threads: int = DEFAULT_THREADS,
    data_workers: int = DEFAULT_WORKERS,
) -> dict:
    """VRAM, private RAM and CPU cores one task of ``workload`` needs, with their basis."""
    if gpu_name is None and isinstance(host, dict):
        names = {gpu.get("name") for gpu in host.get("gpus") or [] if isinstance(gpu, dict)}
        gpu_name = next(iter(names)) if len(names) == 1 else None
    key = workload.get("key") or workload_key(workload)
    same = [row for row in observations or [] if _usable(row) and row["workloadKey"] == key]
    vram, vram_source, peaks = _vram(workload, same, gpu_name)
    ram, ram_source = _ram(workload, same, data_workers)
    cores = [row["meanCpuCores"] for row in same if _positive(row.get("meanCpuCores"))]
    cpu = median(cores) if cores else cpu_threads + 0.5 * data_workers
    return {
        "vramGb": round(vram, 2),
        "ramGb": round(ram, 2),
        "cpuCores": round(cpu, 2),
        "basis": "measured" if vram_source == "measured" else "estimated",
        "evidence": len(peaks),
        "sources": {
            "vram": vram_source,
            "ram": ram_source,
            "cpu": "measured" if cores else "default",
        },
        "peakVramGb": [round(min(peaks), 2), round(max(peaks), 2)] if peaks else None,
    }


# ---------------------------------------------------------------------------
# Suggestion


def _settings(settings) -> dict:
    settings = settings if isinstance(settings, dict) else {}
    reserves = settings.get("reserves") if isinstance(settings.get("reserves"), dict) else {}
    defaults = settings.get("defaults") if isinstance(settings.get("defaults"), dict) else {}
    return {
        "cpuReserve": reserves.get("cpuThreads")
        if _finite(reserves.get("cpuThreads"))
        else DEFAULT_CPU_RESERVE,
        "ramReserve": reserves.get("ramGb") if _finite(reserves.get("ramGb")) else None,
        "vramReserve": reserves.get("vramGb") if _finite(reserves.get("vramGb")) else None,
        "threads": defaults.get("cpuThreadsPerRun")
        if _count(defaults.get("cpuThreadsPerRun"))
        else DEFAULT_THREADS,
        "workers": defaults.get("dataLoaderWorkers")
        if type(defaults.get("dataLoaderWorkers")) is int and defaults["dataLoaderWorkers"] >= 0
        else DEFAULT_WORKERS,
    }


def _dominant(keys, observations) -> str | None:
    counts: dict[str, int] = {}
    for row in observations:
        counts[row["workloadKey"]] = counts.get(row["workloadKey"], 0) + 1
    ranked = [key for key in keys if counts.get(key)] or list(counts)
    return max(ranked, key=lambda key: counts.get(key, 0), default=None)


def _running_private_gb(running) -> float:
    total = 0.0
    for task in running or []:
        if not isinstance(task, dict):
            continue
        resources = task.get("resources") if isinstance(task.get("resources"), dict) else task
        value = resources.get("privateRamGb")
        total += value if _positive(value) else 0.0
    return total


def _fit(points) -> tuple[float, float]:
    weight = sum(runs for _, _, runs in points)
    mean_x = sum(x * runs for x, _, runs in points) / weight
    mean_y = sum(y * runs for _, y, runs in points) / weight
    sxx = sum(runs * (x - mean_x) ** 2 for x, _, runs in points)
    slope = sum(runs * (x - mean_x) * (y - mean_y) for x, y, runs in points) / sxx
    return mean_y - slope * mean_x, slope


def _level(value: float) -> str:
    return f"{value:.1f}".removesuffix(".0")


def _knee(intercept: float, slope: float) -> int:
    """Smallest concurrency where one more task adds < 5% throughput (T(c) = c / s(c))."""
    for level in range(1, MAX_PARALLEL):
        now, after = intercept + slope * level, intercept + slope * (level + 1)
        if now <= 0 or after <= 0:
            return level
        if ((level + 1) / after) / (level / now) - 1 < THROUGHPUT_GAIN:
            return level
    return MAX_PARALLEL


def _throughput(observations: list[dict], key: str | None, gpu_names: set) -> dict:
    rows = [
        row
        for row in observations
        if row["workloadKey"] == key
        and not row.get("resumed")
        and _positive(row.get("secondsPerEpoch"))
        and _positive(row.get("meanConcurrency"))
    ]
    named = [row for row in rows if row.get("gpuName") in gpu_names]
    rows = named or rows
    empty = {"observed": [], "trend": "unknown", "limit": None, "workloadKey": key}
    if not rows:
        return {**empty, "note": "No completed runs with epoch timings yet."}
    buckets: dict[float, list[dict]] = {}
    for row in rows:
        buckets.setdefault(round(row["meanConcurrency"] * 2) / 2, []).append(row)
    points = sorted(
        (
            (
                sum(row["meanConcurrency"] for row in members) / len(members),
                median(row["secondsPerEpoch"] for row in members),
                len(members),
                bucket,
            )
            for bucket, members in buckets.items()
        ),
        reverse=True,
    )
    observed = [
        {"concurrency": round(x, 1), "secondsPerEpoch": round(y, 1), "runs": runs}
        for x, y, runs, _ in points
    ]
    highest, lowest = points[0], points[-1]
    level = max(1, math.floor(highest[3]))
    result = {**empty, "observed": observed}
    if len(points) >= 2 and highest[3] - lowest[3] >= 1.0:
        best = _knee(*_fit([(x, y, runs) for x, y, runs, _ in points]))
        slower = highest[1] / lowest[1] - 1
        comparison = (
            f"At {_level(highest[0])} runs each epoch was {abs(slower):.0%} "
            f"{'slower' if slower >= 0 else 'faster'} than at {_level(lowest[0])} runs"
        )
        if best > highest[3]:
            limit = min(best, level + 1)
            return {
                **result,
                "trend": "rising",
                "limit": limit,
                "note": f"{comparison}, so throughput was still rising; try {limit}.",
            }
        trend = "falling" if best < level else "flat"
        return {
            **result,
            "trend": trend,
            "limit": best,
            "note": f"{comparison}, so more than {best} runs would add little throughput.",
        }
    samples = 0
    busy = 0.0
    batches = set()
    for row in buckets[highest[3]]:
        telemetry = row.get("telemetry")
        if not isinstance(telemetry, dict) or telemetry.get("batchId") in batches:
            continue
        batches.add(telemetry.get("batchId"))
        entry = (telemetry.get("gpuUtilization") or {}).get(str(level))
        if isinstance(entry, dict) and _count(entry.get("samples")):
            samples += entry["samples"]
            busy += entry["meanPercent"] * entry["samples"]
    if not samples:
        return {
            **result,
            "limit": level,
            "note": f"Only {level} parallel runs were measured and GPU utilisation is unknown.",
        }
    utilization = busy / samples
    if utilization < BUSY_UTILIZATION:
        return {
            **result,
            "trend": "rising",
            "limit": level + 1,
            "note": f"Only {level} parallel runs were measured and the GPU was "
            f"{utilization:.0f}% busy; try {level + 1}.",
        }
    return {
        **result,
        "trend": "flat",
        "limit": level,
        "note": f"At {level} parallel runs the GPU was already {utilization:.0f}% busy.",
    }


def _floor(value) -> int:
    return max(0, math.floor(value + 1e-9))


_BINDING_ORDER = ("throughput", "default", "gpu_memory", "ram", "cpu", "run_count")
_LIMIT_NAMES = {
    "gpu_memory": "GPU memory",
    "ram": "RAM",
    "cpu": "CPU",
    "throughput": "measured throughput",
    "run_count": "the number of runs",
}


def suggest(
    workloads: list[dict],
    observations: list[dict],
    host: dict,
    settings: dict | None = None,
    *,
    running: list[dict] | None = None,
    run_count: int | None = None,
) -> dict:
    """Parallel GPU tasks per GPU, the limit that binds it and plain-language reasons."""
    config = _settings(settings)
    host = host if isinstance(host, dict) else {}
    usable = [row for row in observations or [] if _usable(row)]
    gpus = sorted(
        (
            gpu
            for gpu in host.get("gpus") or []
            if isinstance(gpu, dict)
            and type(gpu.get("index")) is int
            and _positive(gpu.get("totalMemoryGb"))
        ),
        key=lambda gpu: gpu["index"],
    )
    requested = []
    for workload in workloads or []:
        if isinstance(workload, dict):
            requested.append(
                workload if workload.get("key") else {**workload, "key": workload_key(workload)}
            )
    keys = list(dict.fromkeys(row["key"] for row in requested))
    dominant = _dominant(keys, usable)
    targets = requested
    if not targets and dominant is not None:
        seen = {}
        for row in usable:
            workload = row.get("workload")
            if row["workloadKey"] == dominant and isinstance(workload, dict):
                seen.setdefault(_canonical(workload), {**workload, "key": dominant})
        targets = list(seen.values())
    names = sorted({gpu.get("name") for gpu in gpus}, key=str) or [None]
    per_name, estimates = {}, []
    for name in names:
        if targets:
            rows = [
                estimate(
                    target,
                    usable,
                    gpu_name=name,
                    cpu_threads=config["threads"],
                    data_workers=config["workers"],
                )
                for target in targets
            ]
            estimates.extend(rows)
            per_name[name] = {
                field: max(row[field] for row in rows) for field in ("vramGb", "ramGb", "cpuCores")
            }
        else:
            per_name[name] = dict(DEFAULT_TASK)
    per_task = {
        field: max(values[field] for values in per_name.values())
        for field in ("vramGb", "ramGb", "cpuCores")
    }
    if not targets:
        basis = "hardware_only"
    elif all(row["basis"] == "measured" for row in estimates):
        basis = "measured"
    else:
        basis = "estimated"

    shares = max(1, len(gpus))
    gpu_limits = {}
    for gpu in gpus:
        reserve = config["vramReserve"]
        if reserve is None:
            reserve = max(1.0, 0.05 * gpu["totalMemoryGb"])
        vram = per_name.get(gpu.get("name"), per_task)["vramGb"]
        gpu_limits[str(gpu["index"])] = _floor((gpu["totalMemoryGb"] - reserve) / vram)
    ram_limit = cpu_limit = None
    if _positive(host.get("totalRamGb")) and _finite(host.get("availableRamGb")):
        reserve = config["ramReserve"]
        if reserve is None:
            reserve = min(16.0, max(2.0, 0.05 * host["totalRamGb"]))
        budget = host["availableRamGb"] + _running_private_gb(running) - reserve
        ram_limit = _floor(budget / per_task["ramGb"]) // shares
    # Admission reserves every compute thread and loader worker of a task, whatever it
    # measurably uses, so measured cores are shown but do not set this limit.
    threads = config["threads"] + config["workers"]
    if _count(host.get("cpuCount")):
        cpu_limit = _floor((host["cpuCount"] - config["cpuReserve"]) / threads) // shares
    run_limit = math.ceil(run_count / shares) if _count(run_count) else None
    throughput = (
        _throughput(usable, dominant, {gpu.get("name") for gpu in gpus})
        if dominant is not None
        else {
            "observed": [],
            "trend": "unknown",
            "limit": None,
            "workloadKey": None,
            "note": "No completed runs with epoch timings yet.",
        }
    )
    default = MAX_PARALLEL if throughput["limit"] is not None else UNMEASURED_PARALLEL
    shared = {
        "throughput": throughput["limit"],
        "default": default,
        "ram": ram_limit,
        "cpu": cpu_limit,
        "run_count": run_limit,
    }

    def resolve(gpu_memory):
        values = {**shared, "gpu_memory": gpu_memory}
        known = [(values[name], name) for name in _BINDING_ORDER if values[name] is not None]
        value, name = min(known, key=lambda item: item[0])
        return max(1, min(value, MAX_PARALLEL)), name

    per_gpu = {index: resolve(limit)[0] for index, limit in gpu_limits.items()}
    gpu_memory = min(gpu_limits.values()) if gpu_limits else None
    parallel, binding = resolve(gpu_memory)
    limits = {
        "gpuMemory": gpu_memory,
        "ram": ram_limit,
        "cpu": cpu_limit,
        "throughput": throughput["limit"],
        "runCount": run_limit,
    }

    matching = [row for row in usable if row["workloadKey"] in (keys or [dominant])]
    runs = usable
    finished = [row["finishedAt"] for row in runs if isinstance(row.get("finishedAt"), str)]
    evidence = {
        "runs": len(runs),
        "batches": len({row["batchId"] for row in runs if row.get("batchId")}),
        "latest": max(finished, key=lambda value: _instant(value) or 0.0) if finished else None,
    }
    explanation, findings = [], []
    peaks = [row["peakVramGb"] for row in matching if _positive(row.get("peakVramGb"))]
    if basis == "hardware_only":
        explanation.append(
            "No workload or measurements yet: assumed "
            f"{DEFAULT_TASK['vramGb']:g} GiB GPU memory and {DEFAULT_TASK['ramGb']:g} GiB RAM "
            "per task."
        )
        findings.append(
            {
                "code": "CAPACITY_HARDWARE_ONLY",
                "severity": "info",
                "message": "This is a hardware-based starting point; it improves once runs finish.",
            }
        )
    elif peaks:
        subset = f" ({len(matching)} with this workload)" if len(matching) != len(runs) else ""
        explanation.append(
            f"Measured {len(runs)} run{'s' if len(runs) != 1 else ''} on this machine{subset}: "
            f"{min(peaks):.1f}–{max(peaks):.1f} GiB GPU memory and about "
            f"{per_task['ramGb']:.1f} GiB RAM per run."
        )
    else:
        explanation.append(
            "No completed run of this workload was measured here: estimated "
            f"{per_task['vramGb']:.1f} GiB GPU memory and {per_task['ramGb']:.1f} GiB RAM per "
            "run from the workload size."
        )
    if basis == "estimated":
        findings.append(
            {
                "code": "CAPACITY_UNMEASURED",
                "severity": "warning",
                "message": "Some requested folds have no nearby measured run on this machine, so "
                "their GPU memory is estimated from the workload size.",
            }
        )
    if throughput["limit"] is not None:
        explanation.append(throughput["note"])
        if keys and throughput["workloadKey"] not in keys:
            findings.append(
                {
                    "code": "CAPACITY_THROUGHPUT_OTHER_WORKLOAD",
                    "severity": "info",
                    "message": "Throughput was measured on a different workload on this machine.",
                }
            )
    else:
        explanation.append(
            f"Without epoch timings the suggestion starts at no more than {UNMEASURED_PARALLEL}."
        )
    stated = [
        (label, value)
        for label, value in (("GPU memory", gpu_memory), ("RAM", ram_limit), ("CPU", cpu_limit))
        if value is not None
    ]
    if stated:
        parts = [f"{label} {value}" for label, value in stated[1:]]
        explanation.append(
            f"{stated[0][0]} would allow {', '.join([str(stated[0][1]), *parts])}"
            f"{' per GPU' if shares > 1 else ''}."
        )
    if cpu_limit is not None:
        loading = f" and {config['workers']} for loading data" if config["workers"] else ""
        explanation.append(
            f"Each task reserves {threads} CPU thread{'s' if threads != 1 else ''}: "
            f"{config['threads']} for computing{loading}."
        )
    reason = _LIMIT_NAMES.get(binding)
    explanation.append(
        f"Suggested {parallel} parallel GPU task{'s' if parallel != 1 else ''}"
        + (f", limited by {reason}." if reason else ".")
    )
    if not gpus:
        findings.append(
            {
                "code": "CAPACITY_NO_GPU",
                "severity": "warning",
                "message": "No GPU with known memory was detected.",
            }
        )
    exhausted = [
        label
        for label, value in (("GPU memory", gpu_memory), ("RAM", ram_limit), ("CPU", cpu_limit))
        if value == 0
    ]
    if exhausted:
        findings.append(
            {
                "code": "CAPACITY_EXHAUSTED",
                "severity": "warning",
                "message": f"{' and '.join(exhausted)} cannot fit one more task after reserves.",
            }
        )
    return {
        "version": SUGGESTION_VERSION,
        "basis": basis,
        "parallelGpuTasks": parallel,
        "perGpu": per_gpu,
        "binding": binding,
        "limits": limits,
        "perTask": {
            **{field: round(value, 2) for field, value in per_task.items()},
            "cpuThreads": threads,
        },
        "throughput": {
            "observed": throughput["observed"],
            "trend": throughput["trend"],
            "note": throughput["note"],
            "workloadKey": throughput["workloadKey"],
        },
        "evidence": evidence,
        "explanation": explanation,
        "findings": findings,
    }
