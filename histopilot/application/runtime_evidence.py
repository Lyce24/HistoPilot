"""Bounded, read-only runtime observations; never initialize CUDA or alter jobs."""

import json
import math
import os
import stat
import tempfile
from datetime import datetime
from pathlib import Path

from histopilot.application.development import _plan_metadata
from histopilot.application.feature_bundles import _hash
from histopilot.schemas.nnmil import resolve_nnmil_plan
from histopilot.schemas.training_controls import sampling_memberships
from histopilot.storage.project_lock import StorageError, _reject_symlink_components
from histopilot.storage.scientific import ScientificStore
from histopilot.workers.training_process import compute_snapshot, owned_processes

MAX_DIRECTORY_ENTRIES = 1024
MAX_BATCHES = 16
MAX_RUNS = 128
MAX_JSON_BYTES = 16 * 1024**2
MAX_TOTAL_BYTES = 128 * 1024**2
MAX_LEASES = 256
_INVALID = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    IndexError,
    OverflowError,
    RecursionError,
    StorageError,
)


def _finite(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Non-finite observation.")
    return number


def _identity(value):
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 256
        or value in {".", ".."}
        or any(c in value for c in ("/", "\\", "\0"))
    ):
        raise ValueError("Invalid observation identity.")
    return value


def _positive(value):
    return type(value) in {int, float} and math.isfinite(value) and value > 0


def _read(path, budget, maximum=MAX_JSON_BYTES, *, with_stat=False):
    _reject_symlink_components(path)
    info = ScientificStore._regular(path)
    if info.st_size > maximum or info.st_size > budget[0]:
        raise ValueError("Observation read budget exceeded.")
    budget[0] -= info.st_size
    raw = ScientificStore._read_file(path, min(maximum, info.st_size))
    after = ScientificStore._regular(path)
    stamp_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if len(raw) != info.st_size or any(
        getattr(info, key) != getattr(after, key) for key in stamp_fields
    ):
        raise ValueError("Observation changed during read.")
    value = json.loads(raw, parse_float=_finite, parse_constant=_finite)
    if not isinstance(value, dict):
        raise ValueError("Observation must be an object.")
    return (value, info) if with_stat else value


def _recent_batches(folder):
    _reject_symlink_components(folder)
    rows = []
    with os.scandir(folder) as entries:
        for index, entry in enumerate(entries):
            if index >= MAX_DIRECTORY_ENTRIES:
                break
            if entry.is_dir(follow_symlinks=False):
                rows.append((entry.stat(follow_symlinks=False).st_mtime_ns, entry.name))
    return [name for _, name in sorted(rows, reverse=True)[:MAX_BATCHES]]


def _batch_evidence(store, batch_id, folder, budget, code):
    batch = store.get_configuration(batch_id)
    manifest = batch["manifest"]
    if manifest.get("kind") != "mil-batch":
        raise ValueError("Not a training batch.")
    inputs = manifest["spec"]["inputs"]
    protocol = store.get_configuration(inputs["protocolId"])
    bundle = store.get_configuration(inputs["featureBundleId"])
    target = protocol["manifest"]["spec"]["target"]
    plan, state = _read(folder / "plan.json", budget), _read(folder / "state.json", budget)
    if (
        plan.get("batchId") != batch_id
        or state.get("batchId") != batch_id
        or plan.get("batchContentHash") != batch["contentHash"]
        or state.get("planHash") != _hash(plan)
        or plan.get("protocolId") != inputs["protocolId"]
        or plan.get("featureBundleId") != inputs["featureBundleId"]
        or plan.get("protocolContentHash") != protocol["contentHash"]
        or plan.get("featureBundleContentHash") != bundle["contentHash"]
        or plan.get("target") != target
        or plan.get("code") != code
        or any(plan.get(key) != manifest[key] for key in ("configurations", "splitPlans", "runs"))
    ):
        raise ValueError("Batch provenance differs from frozen inputs or current code.")
    states = {row["id"]: row for row in state["runs"]}
    if len(states) != len(state["runs"]) or set(states) != {r["id"] for r in manifest["runs"]}:
        raise ValueError("Run identities differ from the batch.")
    devices = state["provenance"]["gpus"]
    gpus = {row["index"]: row["uuid"] for row in devices}
    if (
        len(gpus) != len(devices)
        or any(type(index) is not int or not 0 <= index <= 127 for index in gpus)
        or any(not isinstance(v, str) or not v for v in gpus.values())
    ):
        raise ValueError("GPU provenance is missing or ambiguous.")
    return manifest, protocol["manifest"], plan, states, gpus


def _run_evidence(manifest, protocol, plan, current, run, folder, budget):
    if current.get("status") not in {"running", "completed"}:
        raise ValueError("Only active or successful runs establish capacity.")
    if any(
        current.get(key) != run[key] for key in ("id", "candidateId", "splitPlanId", "trainingSeed")
    ):
        raise ValueError("Run state differs from its frozen identity.")
    candidate = next(row for row in manifest["configurations"] if row["id"] == run["candidateId"])
    split = next(row for row in manifest["splitPlans"] if row["id"] == run["splitPlanId"])
    recipe = candidate["recipe"]
    members = [
        row for row in protocol["memberships"] if _hash(_plan_metadata(row)) == run["splitPlanId"]
    ]
    expected_data = {
        **plan["data"],
        "memberships": sampling_memberships(members, recipe, plan["data"].get("cohortValues", {})),
    }
    run_plan = _read(folder / "plan.json", budget)
    resolved = resolve_nnmil_plan({"recipe": recipe, "data": expected_data})
    if (
        run_plan.get("runId") != run["id"]
        or run_plan.get("batchId") != plan["batchId"]
        or run_plan.get("batchContentHash") != plan["batchContentHash"]
        or run_plan.get("candidateId") != run["candidateId"]
        or run_plan.get("trainingSeed") != run["trainingSeed"]
        or run_plan.get("splitPlan") != split
        or run_plan.get("recipe") != recipe
        or run_plan.get("target") != plan["target"]
        or run_plan.get("data") != expected_data
        or run_plan.get("code") != plan["code"]
        or run_plan.get("runtime") != plan["runtime"]
        or run_plan.get("resources") != plan["resources"]
        or run_plan.get("device") != "cuda"
        or plan["memberships"].get(run["splitPlanId"]) != members
        or any(
            run_plan.get(key) != resolved.get(key) for key in ("effectiveRecipe", "nnmilPlanning")
        )
    ):
        raise ValueError("Run provenance differs from its frozen execution.")
    if current["status"] == "completed":
        evidence = _read(folder / "result.json", budget)
        if (
            evidence != current.get("result")
            or evidence.get("state") != "succeeded"
            or evidence.get("runId") != run["id"]
            or any(
                evidence.get(key) != resolved.get(key)
                for key in ("effectiveRecipe", "nnmilPlanning")
            )
        ):
            raise ValueError("Result is not the successful frozen run receipt.")
        epoch, stage = evidence.get("epochsCompleted"), "assessment"
    else:
        if not owned_processes(current.get("process"), current.get("processGroupId")):
            raise ValueError("Running observation has no verified live process.")
        started = datetime.fromisoformat(current["startedAt"])
        if started.tzinfo is None or started.utcoffset() is None:
            raise ValueError("Live GPU evidence needs a timezone-aware attempt start.")
        evidence, progress_info = _read(folder / "progress.json", budget, 1024**2, with_stat=True)
        if progress_info.st_mtime_ns < int(started.timestamp() * 1_000_000_000):
            raise ValueError("Progress predates the current worker attempt.")
        epoch, stage = evidence.get("epoch"), "fit"
    peak = evidence.get("cudaPeakReservedBytes")
    allocated = evidence.get("cudaPeakAllocatedBytes")
    if (
        type(epoch) is not int
        or not 1 <= epoch <= recipe["maxEpochs"]
        or not _positive(peak)
        or not _positive(allocated)
        or allocated > peak
    ):
        raise ValueError("GPU evidence needs a completed epoch and positive finite peaks.")
    if recipe.get("bagCurriculum") and epoch <= recipe.get("bagCurriculumWarmupEpochs", 5):
        # The dataset uses zero-based epoch / warmup. A peak before epoch
        # warmup+1 has never covered the final patch cap reported by the planner.
        raise ValueError("The full bag curriculum has not completed an epoch.")
    return run_plan, peak / 1024**3, epoch, stage


def observed_runtime_runs(store, workloads: list[dict]) -> list[dict]:
    """Return compatible same-project GPU peaks without trusting artifact paths."""
    from histopilot.application.runtime_workload import workload_for_recipe

    requested = {row["key"]: row for row in workloads}
    if not requested:
        return []
    budget, observations = [MAX_TOTAL_BYTES], []
    try:
        batches = _recent_batches(store.folder / "training")
        code = compute_snapshot()
    except _INVALID:
        return []
    for batch_id in batches:
        try:
            folder = store.folder / "training" / _identity(batch_id)
            manifest, protocol, plan, states, gpus = _batch_evidence(
                store, batch_id, folder, budget, code
            )
            compatible = [
                row
                for row in requested.values()
                if row.get("protocolHash") == plan["protocolContentHash"]
                and row.get("featureBundleHash") == plan["featureBundleContentHash"]
                and (
                    "runtimeVersions" not in row
                    or row["runtimeVersions"] == plan["runtime"].get("versions")
                )
                and (
                    "runtimePython" not in row
                    or row["runtimePython"] == plan["runtime"].get("python")
                )
            ]
            if not compatible:
                continue
            keys = {row["key"] for row in compatible}
            for run in manifest["runs"][:MAX_RUNS]:
                try:
                    current = states[run["id"]]
                    run_folder = folder / "runs" / _identity(run["id"])
                    run_plan, peak, epoch, stage = _run_evidence(
                        manifest, protocol, plan, current, run, run_folder, budget
                    )
                    # During fitting only validation has executed. Do not label
                    # its peak as having covered a larger unseen assessment bag.
                    observed_rows = [
                        row
                        for row in run_plan["data"]["memberships"]
                        if stage == "assessment" or row["partition"] != "test"
                    ]
                    work = workload_for_recipe(
                        run_plan.get("effectiveRecipe", run_plan["recipe"]),
                        observed_rows,
                        run_plan["data"]["featureFiles"],
                        plan["protocolContentHash"],
                        plan["featureBundleContentHash"],
                        run_plan["data"]["loadingPolicy"],
                        len(plan["target"]["classes"]),
                    )
                    gpu = current.get("gpu")
                    if (
                        work["key"] not in keys
                        or type(gpu) is not int
                        or gpu not in gpus
                        or gpu not in run_plan["resources"]["gpuIds"]
                    ):
                        continue
                    observations.append(
                        {
                            "key": work["key"],
                            "gpuUuid": gpus[gpu],
                            "trainingPatches": work["trainingPatches"],
                            "evaluationPatches": work["evaluationPatches"],
                            "peakReservedGpuGb": peak,
                            "stage": stage,
                            "epoch": epoch,
                            "batchId": batch_id,
                            "runId": run["id"],
                        }
                    )
                except (*_INVALID, StopIteration):
                    continue
        except (*_INVALID, StopIteration):
            continue
    return observations


def read_active_leases() -> list[dict] | None:
    """Inspect existing leases only. None means occupancy could not be verified."""
    folder = Path(tempfile.gettempdir()) / f"histopilot-training-{os.getuid()}"
    try:
        _reject_symlink_components(folder)
        try:
            info = folder.lstat()
        except FileNotFoundError:
            return []
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o022:
            return None
        active, budget = [], [MAX_LEASES * 16384]
        with os.scandir(folder) as entries:
            for index, entry in enumerate(entries):
                if index >= MAX_DIRECTORY_ENTRIES:
                    return None
                if not entry.name.startswith("lease-") or not entry.name.endswith(".json"):
                    continue
                if len(active) >= MAX_LEASES:
                    return None
                lease = _read(folder / entry.name, budget, 16384)
                identity = lease["process"]
                if (
                    not isinstance(identity, dict)
                    or type(identity.get("pid")) is not int
                    or identity["pid"] <= 1
                    or type(identity.get("startTicks")) is not int
                    or identity["startTicks"] <= 0
                    or not isinstance(identity.get("bootId"), str)
                    or not identity["bootId"]
                    or entry.name != f"lease-{identity['pid']}.json"
                    or type(lease.get("cpus")) is not int
                    or lease["cpus"] < 1
                    or not _positive(lease.get("ramGb"))
                    or type(lease.get("runsPerGpu")) is not int
                    or not 1 <= lease["runsPerGpu"] <= 16
                    or (
                        lease.get("gpu") is not None
                        and (type(lease["gpu"]) is not int or not 0 <= lease["gpu"] <= 127)
                    )
                    or lease.get("processGroupId", identity["pid"]) != identity["pid"]
                ):
                    return None
                if owned_processes(identity, lease.get("processGroupId", identity["pid"])):
                    result = {
                        key: lease[key] for key in ("process", "gpu", "cpus", "ramGb", "runsPerGpu")
                    }
                    for key in ("batchId", "runId"):
                        if key in lease:
                            result[key] = _identity(lease[key])
                    # Match scheduler RAM accounting: resident pages are already
                    # excluded from MemAvailable. Unknown RSS reserves the full budget.
                    result["rssGb"] = 0.0
                    try:
                        lines = Path(f"/proc/{identity['pid']}/status").read_text().splitlines()
                        result["rssGb"] = next(
                            int(line.split()[1]) / 1024**2
                            for line in lines
                            if line.startswith("VmRSS:")
                        )
                    except (OSError, ValueError, StopIteration):
                        pass
                    active.append(result)
        return active
    except _INVALID:
        return None
