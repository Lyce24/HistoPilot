"""GPU failures and capacity observations are explicit, bounded, and auditable."""

import json
import subprocess
from types import SimpleNamespace

import pytest

from histopilot.workers import training_process as process


def host():
    return {
        "cpuCount": 32,
        "totalRamGb": 256,
        "availableRamGb": 220,
        "bootId": "boot-a",
        "kernel": "test",
    }


def gpu(driver="560", used=3):
    return {
        "index": 0,
        "uuid": "gpu-a",
        "name": "Test GPU",
        "driverVersion": driver,
        "totalMemoryGb": 24,
        "usedMemoryGb": used,
        "freeMemoryGb": 24 - used,
        "utilizationPercent": 80,
    }


def test_six_requested_runs_are_limited_by_gpu_slots_and_overlapping_worker_pools():
    resources = {
        "cpuThreadsPerRun": 2,
        "dataLoaderWorkers": 2,
        "ramGbPerRun": 8,
        "gpuIds": [0],
        "runsPerGpu": 1,
        "maxConcurrentRuns": 6,
    }
    value = process.resource_plan(resources, host(), 20)
    assert value["cpuSlotsPerRun"] == 6
    assert value["cpuLimit"] == 5
    assert value["gpuSlotLimit"] == value["effectiveConcurrency"] == 1
    assert (
        process.resource_plan({**resources, "runsPerGpu": 6}, host(), 20)["effectiveConcurrency"]
        == 5
    )
    assert process.resource_plan({**resources, "gpuIds": []}, host(), 20)["gpuSlotLimit"] is None
    assert (
        process.resource_plan({**resources, "gpuIds": []}, host(), 20)["effectiveConcurrency"] == 5
    )


@pytest.mark.parametrize(
    "message, expected",
    [
        (
            "CUDA unknown error - this may be due to an incorrectly set up environment",
            "cuda_device_failure",
        ),
        ("CUDA error: an illegal memory access was encountered", "cuda_device_failure"),
        ("Failed to initialize NVML: Driver/library version mismatch", "cuda_device_failure"),
        ("CUDA out of memory. Tried to allocate 12 GiB", "out_of_memory"),
        ("OutOfMemoryError", "out_of_memory"),
        ("Cannot read feature file", "training_error"),
    ],
)
def test_failure_classification_does_not_confuse_oom_with_device_loss(message, expected):
    assert process.classify_training_failure(message) == expected


def test_gpu_probe_preserves_unknown_fields_and_uses_short_timeout(monkeypatch):
    calls = []
    monkeypatch.setattr(process.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")

    def probe(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout='0, GPU-a, "GPU, large", 560.35, 24576, 1024, [N/A], N/A\n')

    monkeypatch.setattr(process.subprocess, "run", probe)
    value = process.gpu_snapshot()
    assert value["gpus"][0]["name"] == "GPU, large"
    assert value["gpus"][0]["totalMemoryGb"] == 24
    assert value["gpus"][0]["usedMemoryGb"] == 1
    assert value["gpus"][0]["freeMemoryGb"] is None
    assert value["gpus"][0]["utilizationPercent"] is None
    assert calls[0][1]["timeout"] == 3


def test_gpu_probe_failure_remains_a_diagnostic_not_fabricated_zero_memory(monkeypatch):
    monkeypatch.setattr(process.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")

    def unavailable(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("nvidia-smi", 3)

    monkeypatch.setattr(process.subprocess, "run", unavailable)
    value = process.gpu_snapshot()
    assert value["gpus"] == []
    assert "timed out" in value["gpuProbeError"]


def test_host_and_gpu_telemetry_is_rate_limited_durable_and_accumulates_sampled_peaks(
    tmp_path, monkeypatch
):
    clock = [0]
    observations = [gpu(used=3), gpu(used=7), gpu(used=2)]
    monkeypatch.setattr(process.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(process, "host_snapshot", host)
    monkeypatch.setattr(process, "gpu_snapshot", lambda: {"gpus": [observations.pop(0)]})
    monkeypatch.setattr(process, "process_alive", lambda _identity: True)
    monkeypatch.setattr(process, "process_tree_rss", lambda _pid: 4)
    state = {"runs": [{"id": "run-a", "status": "running", "process": {"pid": 99}}]}
    telemetry = process.ResourceTelemetry(tmp_path)
    telemetry.record(state)
    clock[0] = 14
    assert telemetry.record(state) is None
    clock[0] = 15
    telemetry.record(state)
    clock[0] = 30
    telemetry.record(state)
    rows = [json.loads(row) for row in (tmp_path / "telemetry.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    assert rows[0]["host"]["bootId"] == "boot-a"
    assert state["telemetry"]["peak"] == {
        "hostUsedRamGb": 36,
        "runRssGb": {"run-a": 4},
        "gpuUsedMemoryGb": {"0": 7},
    }
    assert state["telemetry"]["latest"]["gpus"][0]["usedMemoryGb"] == 2


def test_cpu_utilization_uses_nonblocking_tick_deltas_and_preserves_unknowns(tmp_path, monkeypatch):
    counters = iter(
        [(100, 40), (200, 60), (200, 60), (10, 4), (110, 4), None, (210, 10), (310, 110)]
    )
    monkeypatch.setattr(process, "cpu_times", lambda: next(counters))
    telemetry = process.ResourceTelemetry(tmp_path)
    assert [telemetry.cpu_utilization() for _ in range(8)] == [
        None,
        80.0,
        None,
        None,
        100.0,
        None,
        None,
        0.0,
    ]


@pytest.mark.parametrize(
    "content, expected",
    [
        ("cpu 20 5 10 50 15 0 0 0 10 4\ncpu0 1 2 3 4\n", (100, 65)),
        ("cpu 20 5 10 50\n", (85, 50)),
        ("cpu0 20 5 10 50\n", None),
        ("cpu 1 2\n", None),
        ("cpu 1 2 invalid 4\n", None),
        ("cpu 1 2 -1 4\n", None),
    ],
)
def test_cpu_ticks_do_not_double_count_guest_time(tmp_path, monkeypatch, content, expected):
    from pathlib import Path

    path = tmp_path / "stat"
    path.write_text(content)
    monkeypatch.setattr(process, "Path", lambda _value: Path(path))
    assert process.cpu_times() == expected


def test_missing_cpu_counters_do_not_prevent_other_telemetry(tmp_path, monkeypatch):
    original_path = process.Path
    missing_stat = tmp_path / "missing-stat"
    monkeypatch.setattr(
        process,
        "Path",
        lambda value: missing_stat if value == "/proc/stat" else original_path(value),
    )
    monkeypatch.setattr(process, "host_snapshot", host)
    monkeypatch.setattr(process, "gpu_snapshot", lambda: {"gpus": [gpu()]})
    telemetry = process.ResourceTelemetry(tmp_path)
    state = {"runs": []}
    observation = telemetry.record(state)
    assert observation["host"]["cpuUtilizationPercent"] is None
    assert observation["host"]["totalRamGb"] == 256
    assert observation["gpus"][0]["utilizationPercent"] == 80
    assert json.loads((tmp_path / "telemetry.jsonl").read_text()) == observation


def test_device_health_changes_only_react_to_evidence_for_selected_gpus():
    baseline = {"gpus": [gpu()]}
    assert "driver changed" in process.device_health_failure(baseline, {"gpus": [gpu("561")]}, [0])
    assert "disappeared" in process.device_health_failure(baseline, {"gpus": []}, [0])
    assert (
        process.device_health_failure(
            baseline, {"gpus": [], "gpuProbeError": "nvidia-smi timeout"}, [0]
        )
        is None
    )
    assert process.device_health_failure(baseline, {"gpus": [gpu("561")]}, []) is None


def test_training_executor_runs_the_archived_package_with_bytecode_disabled(tmp_path, monkeypatch):
    import shlex

    calls = []
    monkeypatch.setattr(process.shutil, "which", lambda _name: "/usr/bin/tmux")

    def invoke(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1 if command[1] == "has-session" else 0, stderr=b"")

    monkeypatch.setattr(process.subprocess, "run", invoke)
    archive = tmp_path / "worker code"
    process.TmuxTrainingExecutor().launch(
        "hp-train-test",
        "/usr/bin/python",
        tmp_path / "plan.json",
        tmp_path / "batch.log",
        package_root=archive,
    )
    command = shlex.split(calls[-1][-1])
    assert command[:3] == ["cd", str(archive), "&&"]
    assert "PYTHONDONTWRITEBYTECODE=1" in command
    assert command[command.index("-m") + 1] == "histopilot.workers.train_batch"
