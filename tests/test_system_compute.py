"""Host telemetry handles unavailable counters without inventing healthy readings."""

import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from histopilot.application import system_compute as compute


def test_cpu_times_excludes_guest_and_includes_io_wait_as_idle():
    assert compute.parse_cpu_times("cpu 100 10 20 200 30 4 6 5 25 2\ncpu0 1 2 3 4") == (375, 230)
    assert compute.parse_cpu_times("cpu 10 2 3 40") == (55, 40)
    for value in ("", "cpu0 1 2 3 4", "cpu 1 -2 3 4", "cpu 1 2 3", "cpu 1 x 3 4"):
        assert compute.parse_cpu_times(value) is None


def test_cpu_physical_cores_count_socket_and_core_not_hyperthreads():
    text = "\n\n".join(
        f"processor: {index}\nmodel name: Test CPU\nphysical id: {socket}\ncore id: {core}"
        for index, (socket, core) in enumerate(((0, 0), (0, 0), (0, 1), (1, 0)))
    )
    assert compute.parse_cpu_info(text) == ("Test CPU", 3)
    assert compute.parse_cpu_info("Hardware: Example ARM\nprocessor: 0") == ("Example ARM", None)
    assert compute.parse_cpu_info("") == (None, None)


def test_ram_uses_available_memory_and_preserves_zero_swap():
    result = compute.parse_memory(
        "MemTotal: 1000 kB\nMemFree: 100 kB\nMemAvailable: 600 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB"
    )
    assert result["totalBytes"] == 1000 * 1024
    assert result["usedBytes"] == 400 * 1024
    assert result["availableBytes"] == 600 * 1024
    assert result["utilizationPercent"] == 40
    assert result["swapTotalBytes"] == result["swapUsedBytes"] == 0
    assert result["status"] == "available"
    unknown = compute.parse_memory("")
    assert unknown["status"] == "unavailable"
    assert unknown["usedBytes"] is unknown["swapUsedBytes"] is None
    partial = compute.parse_memory("MemTotal: 1000 kB\nMemFree: 500 kB\nSwapTotal: 20 kB")
    assert partial["usedBytes"] is partial["availableBytes"] is partial["swapUsedBytes"] is None
    invalid = compute.parse_memory("MemTotal: nan kB\nMemAvailable: inf kB\nSwapFree: -1 kB")
    assert invalid["totalBytes"] is invalid["availableBytes"] is None


def test_gpu_csv_supports_multiple_gpus_and_unavailable_optional_wsl_fields():
    rows = compute.parse_gpu_csv(
        '0, "NVIDIA GPU, Test", GPU-a, 580.0, 75, 10240, 2560, 7680, 64, 128.5, 250\n'
        "1, NVIDIA Another GPU, GPU-b, 580.0, [N/A], 24576, 0, 24576, "
        "[Not Supported], N/A, [N/A]\n"
    )
    assert [row["index"] for row in rows] == [0, 1]
    assert rows[0]["name"] == "NVIDIA GPU, Test"
    assert rows[0]["memoryTotalBytes"] == 10240 * 1024**2
    assert rows[0]["memoryUtilizationPercent"] == 25
    assert rows[0]["utilizationPercent"] == 75
    assert rows[0]["temperatureCelsius"] == 64
    assert rows[0]["powerWatts"] == 128.5
    assert rows[1]["memoryUsedBytes"] == 0
    assert rows[1]["utilizationPercent"] is rows[1]["temperatureCelsius"] is None
    assert rows[1]["powerWatts"] is rows[1]["powerLimitWatts"] is None


def test_gpu_unavailable_is_not_reported_as_no_hardware(monkeypatch):
    monkeypatch.setattr(compute.shutil, "which", lambda _name: None)
    monkeypatch.setattr(Path, "is_file", lambda _path: False)
    result = compute.read_gpus()
    assert result["status"] == "unavailable"
    assert result["devices"] == []
    assert "unknown" in result["message"]


@pytest.mark.parametrize("failure", ["timeout", "driver", "malformed", "empty"])
def test_gpu_failure_is_bounded_and_does_not_claim_no_gpus(monkeypatch, failure):
    monkeypatch.setattr(compute.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")

    def run(command, **kwargs):
        assert command[0] == "/usr/bin/nvidia-smi"
        assert kwargs["timeout"] == 1.5
        assert not kwargs.get("shell")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return SimpleNamespace(
            returncode=1 if failure == "driver" else 0,
            stdout="bad,csv" if failure == "malformed" else "",
            stderr="driver inaccessible",
        )

    monkeypatch.setattr(compute.subprocess, "run", run)
    result = compute.read_gpus()
    assert result["status"] == "error"
    assert result["devices"] == []
    assert result["message"]


def test_gpu_retries_only_unsupported_fields_and_supports_wsl_executable(monkeypatch):
    monkeypatch.setattr(compute.shutil, "which", lambda _name: None)
    monkeypatch.setattr(Path, "is_file", lambda path: str(path) == "/usr/lib/wsl/lib/nvidia-smi")
    calls = []

    def run(command, **_kwargs):
        calls.append(command)
        if len(calls) == 1:
            return SimpleNamespace(
                returncode=2, stdout="", stderr='Field "power.draw" is not a valid field'
            )
        return SimpleNamespace(
            returncode=0, stdout="0, Test GPU, GPU-a, 555.0, 90, 1000, 750, 250", stderr=""
        )

    monkeypatch.setattr(compute.subprocess, "run", run)
    result = compute.read_gpus()
    assert result["status"] == "available"
    assert len(calls) == 2
    assert calls[0][0] == "/usr/lib/wsl/lib/nvidia-smi"
    assert "power.draw" not in calls[1][1]
    assert result["devices"][0]["powerWatts"] is None
    assert result["devices"][0]["memoryUtilizationPercent"] == 75


def test_explicit_nvidia_no_devices_can_be_reported_as_empty(monkeypatch):
    monkeypatch.setattr(compute.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        compute.subprocess,
        "run",
        lambda *_a, **_kw: SimpleNamespace(
            returncode=6,
            stdout="No devices were found",
            stderr="",
        ),
    )
    assert compute.read_gpus() == {"status": "available", "devices": [], "message": None}


def test_sampler_cpu_delta_reset_cache_and_missing_volume(monkeypatch, tmp_path):
    files = {
        "stat": "cpu 100 0 0 900 0 0 0 0",
        "cpuinfo": "model name: Test CPU\nphysical id: 0\ncore id: 0",
        "meminfo": "MemTotal: 1000 kB\nMemAvailable: 500 kB",
        "uptime": "1234.5 100.2",
    }
    monkeypatch.setattr(compute, "_read", lambda path: files.get(path.name, ""))
    monkeypatch.setattr(
        compute, "read_gpus", lambda: {"status": "unavailable", "devices": [], "message": "unknown"}
    )
    sampler = compute.ComputeSampler(tmp_path, (tmp_path, tmp_path / "missing"))
    first = sampler.snapshot()
    assert first["cpu"]["utilizationPercent"] is first["sampleIntervalSeconds"] is None
    assert first["host"]["uptimeSeconds"] == 1234.5
    assert len(first["disks"]) == 2
    assert first["disks"][0]["status"] == "available"
    assert first["disks"][1]["status"] == "unavailable"
    assert first["disks"][1]["totalBytes"] is None
    files["stat"] = "cpu 160 0 0 940 0 0 0 0"
    assert sampler.snapshot() is first
    sampler._cached_at -= 3
    second = sampler.snapshot()
    assert second["cpu"]["utilizationPercent"] == 60
    assert second["sampleIntervalSeconds"] is not None
    # Counter resets/restarts must not become 0% healthy readings.
    files["stat"] = "cpu 1 0 0 1"
    sampler._cached_at -= 3
    assert sampler.snapshot()["cpu"]["utilizationPercent"] is None


def test_concurrent_clients_share_one_probe(monkeypatch, tmp_path):
    sampler = compute.ComputeSampler(tmp_path, ())
    entered, release = threading.Event(), threading.Event()
    calls = []

    def collect(_now):
        calls.append(True)
        entered.set()
        assert release.wait(2)
        return {"sampledAt": "test"}

    monkeypatch.setattr(sampler, "_collect", collect)
    with ThreadPoolExecutor(max_workers=3) as executor:
        first = executor.submit(sampler.snapshot)
        assert entered.wait(2)
        second, third = executor.submit(sampler.snapshot), executor.submit(sampler.snapshot)
        release.set()
        assert first.result() == second.result() == third.result() == {"sampledAt": "test"}
    assert len(calls) == 1
