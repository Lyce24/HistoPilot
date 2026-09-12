"""Durable batch metadata and a tmux boundary shared by API and training workers."""

import csv
import hashlib
import json
import math
import os
import platform
import shlex
import shutil
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from histopilot.storage.project_lock import StorageError, _reject_symlink_components
from histopilot.storage.scientific import ScientificStore
from histopilot.workers.packing_process import write_json

ACTIVE = {"queued", "running"}
TERMINAL = {"completed", "failed", "cancelled", "interrupted"}


def host_snapshot() -> dict:
    """Read host capacity without importing or initializing the CUDA runtime."""
    memory = {
        line.split(":")[0]: int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
    }
    return {
        "cpuCount": len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count() or 1,
        "totalRamGb": memory["MemTotal"] / 1024**3,
        "availableRamGb": memory["MemAvailable"] / 1024**3,
        "bootId": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "kernel": platform.release(),
    }


def gpu_snapshot() -> dict:
    """Best-effort driver telemetry. N/A fields stay unknown, never become zero."""
    executable = shutil.which("nvidia-smi")
    wsl_executable = Path("/usr/lib/wsl/lib/nvidia-smi")
    if executable is None and wsl_executable.is_file():
        executable = str(wsl_executable)
    if executable is None:
        return {"gpus": [], "gpuProbeError": "nvidia-smi is unavailable."}
    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        )

        def number(value, divisor=1):
            try:
                parsed = float(value) / divisor
                return parsed if math.isfinite(parsed) else None
            except ValueError:
                return None

        gpus = []
        for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
            if len(row) != 8:
                raise ValueError("Unexpected NVIDIA telemetry columns.")
            index, uuid, name, driver, total, used, free, utilization = row
            gpus.append(
                {
                    "index": int(index),
                    "uuid": uuid,
                    "name": name,
                    "driverVersion": driver,
                    "totalMemoryGb": number(total, 1024),
                    "usedMemoryGb": number(used, 1024),
                    "freeMemoryGb": number(free, 1024),
                    "utilizationPercent": number(utilization),
                }
            )
        return {"gpus": gpus}
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        detail = getattr(error, "stderr", None) or str(error)
        return {"gpus": [], "gpuProbeError": str(detail).strip()[:2000]}


def cpu_slots_per_run(resources: dict) -> int:
    # The persistent training pool overlaps the validation worker pool.
    return resources["cpuThreadsPerRun"] + 2 * resources["dataLoaderWorkers"]


def resource_plan(resources: dict, host: dict, run_count: int) -> dict:
    cpu_slots = cpu_slots_per_run(resources)
    cpu_limit = host["cpuCount"] // cpu_slots
    ram_limit = int(host["availableRamGb"] // resources["ramGbPerRun"])
    gpu_limit = len(resources["gpuIds"]) * resources["runsPerGpu"] or None
    return {
        "requestedConcurrency": resources["maxConcurrentRuns"],
        "effectiveConcurrency": min(
            resources["maxConcurrentRuns"], cpu_limit, ram_limit, gpu_limit or run_count, run_count
        ),
        "cpuSlotsPerRun": cpu_slots,
        "cpuLimit": cpu_limit,
        "ramLimit": ram_limit,
        "gpuSlotLimit": gpu_limit,
        "note": "Capacity ceiling before other HistoPilot leases. RAM is a requested reservation; GPU slots do not guarantee that bags fit in VRAM.",
    }


def append_event(path: Path, event: dict) -> None:
    """Flush each observation so an abrupt host loss leaves useful evidence."""
    _reject_symlink_components(path)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def classify_training_failure(message: str) -> str:
    value = message.lower()
    if "out of memory" in value or "outofmemoryerror" in value:
        return "out_of_memory"
    if any(
        token in value
        for token in (
            "cuda unknown error",
            "cuda error: unknown error",
            "cudaerrorunknown",
            "device has been lost",
            "device is lost",
            "gpu has fallen off",
            "driver shutting down",
            "cuda driver version is insufficient",
            "driver/library version mismatch",
            "cuda-capable device(s) is/are busy or unavailable",
            "cuda error: initialization error",
            "cuda initialization error",
            "cuda error: an illegal memory access",
            "cuda error: device-side assert",
        )
    ):
        return "cuda_device_failure"
    return "training_error"


def process_tree_rss(pid: int) -> float:
    """Observed RSS sum including loader children; shared pages can be counted twice."""
    pending, visited, resident = [pid], set(), 0
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        try:
            folder = Path(f"/proc/{current}")
            for line in (folder / "status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    resident += int(line.split()[1]) * 1024
            for children in (folder / "task").glob("*/children"):
                pending.extend(int(value) for value in children.read_text().split())
        except (OSError, ValueError):
            continue  # Processes may finish during an observation.
    return resident / 1024**3


class ResourceTelemetry:
    """Bounded-rate host/driver observations, independent of model and CUDA imports."""

    interval_seconds = 15

    def __init__(self, folder: Path):
        self.folder = folder
        self.last_observed = -float("inf")

    def record(self, state: dict, *, force=False) -> dict | None:
        current = time.monotonic()
        if not force and current - self.last_observed < self.interval_seconds:
            return None
        self.last_observed = current
        observation = {"at": now(), "host": host_snapshot(), **gpu_snapshot(), "runs": []}
        for run in state["runs"]:
            identity = run.get("process")
            if run["status"] == "running" and process_alive(identity):
                observation["runs"].append(
                    {
                        "runId": run["id"],
                        "pid": identity["pid"],
                        "rssGb": process_tree_rss(identity["pid"]),
                    }
                )
        telemetry = state.setdefault("telemetry", {})
        peak = telemetry.setdefault(
            "peak",
            {
                "hostUsedRamGb": 0,
                "runRssGb": {},
                "gpuUsedMemoryGb": {},
            },
        )
        host = observation["host"]
        peak["hostUsedRamGb"] = max(
            peak["hostUsedRamGb"], host["totalRamGb"] - host["availableRamGb"]
        )
        for run in observation["runs"]:
            peak["runRssGb"][run["runId"]] = max(
                peak["runRssGb"].get(run["runId"], 0), run["rssGb"]
            )
        for gpu in observation["gpus"]:
            if gpu["usedMemoryGb"] is not None:
                key = str(gpu["index"])
                peak["gpuUsedMemoryGb"][key] = max(
                    peak["gpuUsedMemoryGb"].get(key, 0), gpu["usedMemoryGb"]
                )
        path = self.folder / "telemetry.jsonl"
        telemetry.update(path=str(path), intervalSeconds=self.interval_seconds, latest=observation)
        append_event(path, observation)
        return observation


def device_health_failure(provenance: dict, observation: dict, gpu_ids: list[int]) -> str | None:
    if not gpu_ids:
        return None
    if classify_training_failure(observation.get("gpuProbeError", "")) == "cuda_device_failure":
        return "The GPU driver probe reported a device/driver failure."
    if observation.get("gpuProbeError"):
        return None  # Missing telemetry alone does not prove GPU loss.
    prior = {gpu["index"]: gpu for gpu in provenance.get("gpus", [])}
    current = {gpu["index"]: gpu for gpu in observation.get("gpus", [])}
    for gpu_id in gpu_ids:
        if gpu_id not in prior:
            continue
        if gpu_id not in current:
            return f"GPU {gpu_id} disappeared from the driver inventory."
        if any(prior[gpu_id][key] != current[gpu_id][key] for key in ("uuid", "driverVersion")):
            return f"GPU {gpu_id} or its driver changed while the batch was running."
    return None


def compute_snapshot() -> dict:
    root = Path(__file__).resolve().parents[1]
    paths = [
        *(root / "models").glob("*.py"),
        *(root / "datasets").glob("*.py"),
        *(root / "training").glob("*.py"),
        root / "workers" / "train_batch.py",
        root / "workers" / "training_process.py",
        root / "workers" / "compute_archive.py",
        root / "workers" / "compute_job.py",
        root / "workers" / "experiment_predictors.py",
        root / "application" / "experiment_predictors.py",
        root / "application" / "predictor_builds.py",
        root / "application" / "predictors.py",
        root / "application" / "refits.py",
        root / "schemas" / "model_experiments.py",
        root / "schemas" / "predictors.py",
        root / "storage" / "attention_inputs.py",
        root / "storage" / "attention_packs.py",
        root / "storage" / "packed.py",
        root / "storage" / "pack_import.py",
    ]
    files = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }
    return {
        "sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        "files": files,
    }


def now() -> str:
    return datetime.now(UTC).isoformat()


def read_json(path: Path) -> dict:
    content = ScientificStore._read_file(path, 64 * 1024 * 1024)

    def finite_number(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Metadata numbers must be finite.")
        return number

    try:
        value = json.loads(content, parse_float=finite_number, parse_constant=finite_number)
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object.")
        # The decoder may accept nesting that later exhausts FastAPI's recursive
        # response encoder. Inspect iteratively, keeping only one iterator per
        # level so wide metadata arrays do not create another large work list.
        pending = [(iter(value.values()), 1)]
        while pending:
            children, depth = pending[-1]
            try:
                child = next(children)
            except StopIteration:
                pending.pop()
                continue
            if isinstance(child, (dict, list)):
                if depth >= 64:
                    raise ValueError("Metadata exceeds its nesting limit.")
                pending.append(
                    (iter(child.values() if isinstance(child, dict) else child), depth + 1)
                )
        return value
    except (ValueError, UnicodeError, RecursionError) as error:
        raise StorageError("Invalid training metadata.", "TRAINING_STATE_INVALID") from error


def read_progress(path: Path) -> tuple[dict | None, str | None]:
    """Optional telemetry must not hide a job's durable state or prevent cancellation."""
    try:
        _reject_symlink_components(path)
        if not path.exists():
            return None, None
        value = read_json(path)
        if not value:
            return None, None  # An empty snapshot has no reported progress yet.
        _validate_progress(value)
        return value, None
    except (OSError, ValueError, OverflowError):
        return None, (
            "Progress details are unavailable because the progress file is invalid or cannot be read safely. "
            "Job status and cancellation remain available."
        )


def _validate_progress(value: dict) -> None:
    """Validate displayed fields while permitting future, unrecognized telemetry."""
    counters = {
        "epoch",
        "maxEpochs",
        "globalStep",
        "step",
        "completedModels",
        "totalModels",
        "completedPairs",
        "totalPairs",
        "completedSlides",
        "totalSlides",
        "slideCount",
        "cudaPeakAllocatedBytes",
        "cudaPeakReservedBytes",
    }
    for key in counters & value.keys():
        if type(value[key]) is not int or value[key] < 0:
            raise ValueError(f"Invalid progress counter: {key}")
    for key in {"completed", "total"} & value.keys():
        if value[key] is not None and (type(value[key]) is not int or value[key] < 0):
            raise ValueError(f"Invalid progress counter: {key}")
    for key in {"trainingLoss", "learningRate", "percent"} & value.keys():
        number = value[key]
        if number is None and key != "learningRate":
            continue
        if type(number) not in {int, float} or not math.isfinite(number):
            raise ValueError(f"Invalid progress number: {key}")
    for key in {"stage", "unit", "updatedAt", "currentSlide"} & value.keys():
        if value[key] is None and key == "currentSlide":
            continue
        if not isinstance(value[key], str):
            raise ValueError(f"Invalid progress text: {key}")
    validation = value.get("validation")
    if validation is None:
        return  # Refits have no validation partition.
    if not isinstance(validation, dict):
        raise ValueError("Invalid progress validation metrics.")
    for key in {
        "loss",
        "accuracy",
        "auroc",
        "auprc",
        "balancedAccuracy",
        "macroF1",
    } & validation.keys():
        number = validation[key]
        if number is not None and (type(number) not in {int, float} or not math.isfinite(number)):
            raise ValueError(f"Invalid validation metric: {key}")
    if "count" in validation and (type(validation["count"]) is not int or validation["count"] < 0):
        raise ValueError("Invalid validation count.")
    if "available" in validation and type(validation["available"]) is not bool:
        raise ValueError("Invalid validation availability.")
    if "reason" in validation and not isinstance(validation["reason"], str):
        raise ValueError("Invalid validation reason.")
    if "missingClasses" in validation and (
        not isinstance(validation["missingClasses"], list)
        or any(not isinstance(item, str) for item in validation["missingClasses"])
    ):
        raise ValueError("Invalid validation class labels.")


def process_identity(pid: int | None = None) -> dict:
    pid = pid or os.getpid()
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return {
        "pid": pid,
        "startTicks": int(fields[19]),
        "bootId": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    }


def process_alive(value: dict | None) -> bool:
    if not value or type(value.get("pid")) is not int or value["pid"] <= 1:
        return False
    try:
        fields = Path(f"/proc/{value['pid']}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[0] != "Z" and process_identity(value["pid"]) == value
    except (OSError, ValueError, IndexError):
        return False


def counts(runs: list[dict]) -> dict:
    observed = Counter(run["status"] for run in runs)
    return {
        "total": len(runs),
        **{
            key: observed[key]
            for key in ("queued", "running", "completed", "failed", "cancelled", "interrupted")
        },
    }


def save_state(folder: Path, state: dict) -> None:
    state["updatedAt"] = now()
    state["runCounts"] = counts(state["runs"])
    write_json(folder / "state.json", state)


class TmuxTrainingExecutor:
    def available(self):
        return shutil.which("tmux") is not None

    def running(self, session: str) -> bool:
        if not self.available():
            return False
        result = subprocess.run(
            ["tmux", "has-session", "-t", f"={session}"], capture_output=True, timeout=10
        )
        if result.returncode and b"Operation not permitted" in result.stderr:
            raise RuntimeError("Cannot inspect training tmux sessions: permission denied.")
        return result.returncode == 0

    def launch(self, session: str, python: str, plan: Path, log: Path, *, package_root: Path):
        subprocess.run(["tmux", "ls"], capture_output=True, timeout=10)
        if self.running(session):
            raise StorageError("This training session already exists.", "TRAINING_ACTIVE")
        _reject_symlink_components(log)
        command = "cd " + shlex.quote(str(package_root)) + " && "
        command += shlex.join(
            [
                "env",
                "PYTHONDONTWRITEBYTECODE=1",
                python,
                "-u",
                "-m",
                "histopilot.workers.train_batch",
                str(plan),
            ]
        )
        command += " >> " + shlex.quote(str(log)) + " 2>&1"
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", session, command],
            capture_output=True,
            check=True,
            timeout=15,
        )
