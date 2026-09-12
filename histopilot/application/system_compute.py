"""Read host resource counters without loading compute runtimes or starting workers."""

import csv
import math
import os
import platform
import shutil
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

_GPU_FIELDS = (
    "index",
    "name",
    "uuid",
    "driver_version",
    "utilization.gpu",
    "memory.total",
    "memory.used",
    "memory.free",
    "temperature.gpu",
    "power.draw",
    "power.limit",
)
_GPU_BASE_FIELDS = _GPU_FIELDS[:8]


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _number(value: str | None) -> float | None:
    try:
        number = float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    return number if number is not None and math.isfinite(number) and number >= 0 else None


def _percent(used: float | None, total: float | None) -> float | None:
    if used is None or total is None or total <= 0:
        return None
    return round(min(100.0, max(0.0, used / total * 100)), 1)


def parse_cpu_times(text: str) -> tuple[int, int] | None:
    """Linux guest counters are already included in user/nice: do not count twice."""
    for line in text.splitlines():
        fields = line.split()
        if not fields or fields[0] != "cpu":
            continue
        try:
            counters = [int(value) for value in fields[1:9]]
        except ValueError:
            return None
        if len(counters) < 4 or any(value < 0 for value in counters):
            return None
        return sum(counters), counters[3] + (counters[4] if len(counters) > 4 else 0)
    return None


def parse_cpu_info(text: str) -> tuple[str | None, int | None]:
    model = None
    cores = set()
    for block in text.split("\n\n"):
        fields = dict(
            (key.strip(), value.strip())
            for line in block.splitlines()
            if ":" in line
            for key, value in [line.split(":", 1)]
        )
        model = model or fields.get("model name") or fields.get("Hardware")
        if fields.get("physical id") is not None and fields.get("core id") is not None:
            cores.add((fields["physical id"], fields["core id"]))
    return model, len(cores) or None


def parse_memory(text: str) -> dict:
    readings = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        value = _number(fields[1])
        if value is not None and (len(fields) == 2 or fields[2] == "kB"):
            readings[fields[0].rstrip(":")] = int(value * (1024 if len(fields) > 2 else 1))
    total, available = readings.get("MemTotal"), readings.get("MemAvailable")
    total = total if total and total > 0 else None
    available = available if total and available is not None and available <= total else None
    used = total - available if total is not None and available is not None else None
    swap_total, swap_free = readings.get("SwapTotal"), readings.get("SwapFree")
    swap_used = (
        swap_total - swap_free
        if swap_total is not None and swap_free is not None and swap_free <= swap_total
        else None
    )
    return {
        "totalBytes": total,
        "usedBytes": used,
        "availableBytes": available,
        "utilizationPercent": _percent(used, total),
        "swapTotalBytes": swap_total,
        "swapUsedBytes": swap_used,
        "status": "available" if total is not None else "unavailable",
        "message": (
            None if used is not None else "Memory counters are not available on this host."
        ),
    }


def parse_gpu_csv(output: str, fields: tuple[str, ...] = _GPU_FIELDS) -> list[dict]:
    """nvidia-smi emits MiB; unsupported NVML metrics remain explicitly unknown."""
    devices = []
    for row in csv.reader(output.splitlines(), skipinitialspace=True):
        if not row:
            continue
        if len(row) != len(fields):
            raise ValueError("Unexpected GPU counter format")
        values = dict(zip(fields, (value.strip() for value in row), strict=True))
        index = _number(values.get("index"))
        if index is None or not index.is_integer() or not values.get("name"):
            raise ValueError("Invalid GPU identity")

        def metric(key):
            return _number(values.get(key))

        def mib(key):
            value = metric(key)
            return int(value * 1024 * 1024) if value is not None else None

        def label(key):
            value = values.get(key)
            return value if value and value not in {"N/A", "[N/A]", "[Not Supported]"} else None

        total, used = mib("memory.total"), mib("memory.used")
        utilization = metric("utilization.gpu")
        devices.append(
            {
                "index": int(index),
                "name": values["name"],
                "uuid": label("uuid"),
                "driverVersion": label("driver_version"),
                "utilizationPercent": utilization
                if utilization is None or utilization <= 100
                else None,
                "memoryTotalBytes": total,
                "memoryUsedBytes": used,
                "memoryFreeBytes": mib("memory.free"),
                "memoryUtilizationPercent": _percent(used, total),
                "temperatureCelsius": metric("temperature.gpu"),
                "powerWatts": metric("power.draw"),
                "powerLimitWatts": metric("power.limit"),
            }
        )
    return devices


def read_gpus() -> dict:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        wsl = Path("/usr/lib/wsl/lib/nvidia-smi")
        executable = str(wsl) if wsl.is_file() else None
    if executable is None:
        return {
            "status": "unavailable",
            "devices": [],
            "message": "GPU telemetry is unavailable: nvidia-smi was not found. GPU presence is unknown.",
        }
    try:
        for fields in (_GPU_FIELDS, _GPU_BASE_FIELDS):
            result = subprocess.run(
                [executable, f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=1.5,
                check=False,
            )
            if result.returncode == 0:
                devices = parse_gpu_csv(result.stdout, fields)
                if not devices:
                    raise ValueError("No GPU counters were returned")
                return {"status": "available", "devices": devices, "message": None}
            error = (result.stdout + result.stderr).lower()
            if "no devices were found" in error:
                return {"status": "available", "devices": [], "message": None}
            # Some older/WSL drivers reject an optional field rather than returning N/A.
            if "not a valid field" not in error and "invalid field" not in error:
                break
        message = "The NVIDIA driver did not provide GPU counters. GPU availability is unknown."
    except subprocess.TimeoutExpired:
        message = "GPU telemetry timed out. Other system readings remain available."
    except (OSError, ValueError, csv.Error):
        message = "GPU counters could not be read. GPU availability is unknown."
    return {"status": "error", "devices": [], "message": message}


def _disk(path: Path, role: str) -> dict:
    try:
        usage = shutil.disk_usage(path)
        values = (usage.total, usage.used, usage.free)
    except OSError:
        values = (None, None, None)
    total, used, free = values
    return {
        "path": str(path),
        "role": role,
        "totalBytes": total,
        "usedBytes": used,
        "freeBytes": free,
        "utilizationPercent": _percent(used, total),
        "status": "available" if total is not None else "unavailable",
        "message": None if total is not None else "Storage volume is not accessible.",
    }


class ComputeSampler:
    """One bounded probe at a time, shared by browser polling in this service process.

    First CPU utilization is unknown; subsequent readings use counter differences.
    Linux/WSL reports describe the host visible to the server, not worker attribution.
    """

    def __init__(self, workspace: Path, data_roots: tuple[Path, ...], *, cache_seconds=2.0):
        self.paths = [(workspace, "workspace")]
        self.paths.extend((path, "data") for path in data_roots if path != workspace)
        self.cache_seconds = cache_seconds
        self._lock = threading.Lock()
        self._snapshot = None
        self._cached_at = 0.0
        self._cpu_previous = None
        self._cpu_sampled_at = None

    def snapshot(self) -> dict:
        with self._lock:
            now = time.monotonic()
            if self._snapshot is not None and now - self._cached_at < self.cache_seconds:
                return self._snapshot
            self._snapshot = self._collect(now)
            self._cached_at = time.monotonic()
            return self._snapshot

    def _collect(self, now: float) -> dict:
        proc = Path("/proc")
        counters = parse_cpu_times(_read(proc / "stat"))
        utilization, interval = None, None
        if counters is not None and self._cpu_previous is not None:
            total, idle = (new - old for new, old in zip(counters, self._cpu_previous, strict=True))
            if total > 0 and 0 <= idle <= total:
                utilization = _percent(total - idle, total)
                interval = round(now - self._cpu_sampled_at, 3)
        self._cpu_previous, self._cpu_sampled_at = counters, now
        model, physical = parse_cpu_info(_read(proc / "cpuinfo"))
        try:
            available = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            available = None
        try:
            load = list(os.getloadavg())
        except (AttributeError, OSError):
            load = None
        uptime = _read(proc / "uptime").split()
        return {
            "sampledAt": datetime.now(UTC).isoformat(),
            "sampleIntervalSeconds": interval,
            "host": {
                "hostname": platform.node(),
                "platform": platform.system(),
                "release": platform.release(),
                "uptimeSeconds": _number(uptime[0]) if uptime else None,
            },
            "cpu": {
                "model": model or platform.processor() or None,
                "logicalCores": os.cpu_count(),
                "physicalCores": physical,
                "availableCores": available,
                "utilizationPercent": utilization,
                "loadAverage": load,
                "status": "available" if counters is not None else "unavailable",
                "message": None
                if counters is not None
                else "CPU utilization counters are unavailable.",
            },
            "memory": parse_memory(_read(proc / "meminfo")),
            "gpu": read_gpus(),
            "disks": [_disk(path, role) for path, role in self.paths],
        }
