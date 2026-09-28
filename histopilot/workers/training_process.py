"""Durable batch metadata, host probes and worker process ownership for training."""

import csv
import hashlib
import json
import math
import os
import platform
import shutil
import signal
import subprocess
import time
from collections import Counter
from pathlib import Path

from histopilot.storage.io import read_json_bounded, utc_now, write_json_atomic
from histopilot.storage.project_lock import StorageError, reject_symlink_components

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


def append_event(path: Path, event: dict) -> None:
    """Flush each observation so an abrupt host loss leaves useful evidence."""
    reject_symlink_components(path)
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


def compute_snapshot() -> dict:
    root = Path(__file__).resolve().parents[1]
    paths = [
        root / "scoring.py",
        root / "statistics.py",
        # The inference worker freezes its label-free summary with this module.
        root / "inference_summary.py",
        root / "clinical_features.py",
        root / "domain" / "features.py",
        root / "application" / "clinical_inputs.py",
        root / "candidate_selection.py",
        # The training worker scores its predictions with cv_summary.point_metrics.
        root / "cv_summary.py",
        *(root / "models").glob("*.py"),
        *(root / "datasets").glob("*.py"),
        *(root / "training").glob("*.py"),
        root / "workers" / "train_batch.py",
        root / "workers" / "managed_fold.py",
        root / "workers" / "managed_collect.py",
        root / "workers" / "training_process.py",
        root / "workers" / "compute_archive.py",
        root / "workers" / "compute_job.py",
        root / "workers" / "experiment_predictors.py",
        root / "application" / "experiment_predictors.py",
        root / "application" / "experiment_policy.py",
        root / "application" / "predictor_builds.py",
        root / "application" / "predictors.py",
        root / "application" / "refits.py",
        root / "schemas" / "model_experiments.py",
        root / "schemas" / "development.py",
        root / "schemas" / "analysis.py",
        root / "schemas" / "training_controls.py",
        root / "schemas" / "nnmil.py",
        root / "schemas" / "predictor_policy.py",
        root / "schemas" / "predictors.py",
        root / "storage" / "attention_inputs.py",
        root / "storage" / "attention_packs.py",
        root / "storage" / "packed.py",
        root / "storage" / "pack_import.py",
        # Content hashes, timestamps and the JSON files the modules above read and write.
        root / "storage" / "io.py",
    ]
    files = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }
    return {
        "sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        "files": files,
    }


def read_progress(path: Path) -> tuple[dict | None, str | None]:
    """Optional telemetry must not hide a job's durable state or prevent cancellation."""
    try:
        reject_symlink_components(path)
        if not path.exists():
            return None, None
        value = read_json_bounded(path)
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


def confirmed_process_alive(value: dict | None) -> bool:
    """Distinguish a stopped worker from unreadable process ownership evidence."""
    if value is None:
        return False
    if (
        not isinstance(value, dict)
        or type(value.get("pid")) is not int
        or value["pid"] <= 1
        or type(value.get("startTicks")) is not int
        or not isinstance(value.get("bootId"), str)
    ):
        raise StorageError("Worker process identity is invalid.", "TRAINING_PROCESS_UNKNOWN")
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError as error:
        raise StorageError(
            "Cannot verify the workstation boot identity.", "TRAINING_PROCESS_UNKNOWN"
        ) from error
    if value["bootId"] != boot:
        return False
    try:
        fields = Path(f"/proc/{value['pid']}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[0] not in {"Z", "X"} and int(fields[19]) == value["startTicks"]
    except (FileNotFoundError, ProcessLookupError):
        return False
    except (OSError, ValueError, IndexError) as error:
        raise StorageError(
            "Cannot confirm whether a worker stopped.", "TRAINING_PROCESS_UNKNOWN"
        ) from error


def owned_processes(value: dict | None, group: int | None = None, *, descendants=False) -> list:
    """Identify live members of an owned, private Linux process session.

    A group's numeric ID remains reserved while any member survives its leader.
    Boot ID, leader start time, session, group, and UID prevent attaching to a
    recycled PID. A successful leader probe is sufficient for status/leases;
    cleanup explicitly asks to enumerate descendants as well.
    """
    alive = confirmed_process_alive(value)
    result = [value] if alive else []
    if value is None or group != value["pid"] or (alive and not descendants):
        return result
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        if value["bootId"] != boot:
            return result
        try:
            leader = Path(f"/proc/{group}/stat").read_text().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            leader = None
        if leader is not None and (
            int(leader[19]) != value["startTicks"]
            or int(leader[2]) != group
            or int(leader[3]) != group
        ):
            return result
        try:
            os.killpg(group, 0)
        except ProcessLookupError:
            return result  # Avoid scanning /proc for every old completed fold.
        for path in Path("/proc").iterdir():
            if not path.name.isdecimal() or int(path.name) == group:
                continue
            try:
                if path.stat().st_uid != os.getuid():
                    continue
                fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
                if (
                    fields[0] not in {"Z", "X"}
                    and int(fields[2]) == group
                    and int(fields[3]) == group
                    and int(fields[19]) >= value["startTicks"]
                ):
                    result.append(
                        {"pid": int(path.name), "startTicks": int(fields[19]), "bootId": boot}
                    )
            except (FileNotFoundError, ProcessLookupError):
                continue
        return result
    except (OSError, ValueError, IndexError) as error:
        raise StorageError(
            "Cannot confirm whether worker descendants stopped.", "TRAINING_PROCESS_UNKNOWN"
        ) from error


def stop_owned_processes(value: dict, *, exclude_pid=None, grace_seconds=5) -> None:
    """Drain a private worker session, including children of an exited leader."""
    started = time.monotonic()
    signalled = set()
    while True:
        members = [
            item
            for item in owned_processes(value, value["pid"], descendants=True)
            if item["pid"] != exclude_pid
        ]
        if not members:
            return
        elapsed = time.monotonic() - started
        if elapsed > grace_seconds + 5:
            raise StorageError(
                "Worker descendants have not stopped; their reservation is retained.",
                "TRAINING_CLEANUP_FAILED",
            )
        signum = signal.SIGKILL if elapsed >= grace_seconds else signal.SIGTERM
        for member in members:
            key = (member["pid"], member["startTicks"], signum)
            if key not in signalled and confirmed_process_alive(member):
                try:
                    os.kill(member["pid"], signum)
                except ProcessLookupError:
                    pass
                signalled.add(key)
        time.sleep(0.05)


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
    state["updatedAt"] = utc_now()
    state["runCounts"] = counts(state["runs"])
    write_json_atomic(folder / "state.json", state)
