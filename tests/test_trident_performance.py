"""Worker budgets match available CPUs and the actual TRIDENT device fanout."""

import pytest

from histopilot.adapters.trident import performance
from histopilot.workers.resource_reservation import preparation_resources


@pytest.mark.parametrize("cpus, options, expected", [
    (48, {}, 8),
    (48, {"gpus": [0, 1]}, 4),
    (48, {"gpus": [0, 0]}, 8),
    (48, {"gpus": [-1, -1]}, 4),
    (48, {"gpus": [0, -1]}, 4),
    (16, {}, 7),
    (8, {}, 3),
    (4, {"gpus": [0, 1]}, 1),
    (1, {}, 1),
    (48, {"max_workers": 0}, 0),
    (4, {"max_workers": 12}, 12),
])
def test_automatic_budget_and_explicit_overrides(monkeypatch, cpus, options, expected):
    monkeypatch.setattr(performance, "usable_cpu_count", lambda: cpus)
    assert performance.resolve_max_workers(options) == expected
    resources = preparation_resources("extraction", options)
    assert resources["dataLoaderWorkers"] == expected * performance.execution_device_count(options)


def test_usable_cpu_count_honors_affinity(monkeypatch):
    monkeypatch.setattr(performance.os, "sched_getaffinity", lambda _pid: {1, 3, 5}, raising=False)
    monkeypatch.setattr(performance.os, "cpu_count", lambda: 48)
    assert performance.usable_cpu_count() == 3


def test_usable_cpu_count_falls_back_when_affinity_unavailable(monkeypatch):
    def unavailable(_pid):
        raise OSError("not supported")

    monkeypatch.setattr(performance.os, "sched_getaffinity", unavailable, raising=False)
    monkeypatch.setattr(performance.os, "cpu_count", lambda: 4)
    assert performance.usable_cpu_count() == 4
    monkeypatch.delattr(performance.os, "sched_getaffinity")
    monkeypatch.setattr(performance.os, "cpu_count", lambda: None)
    assert performance.usable_cpu_count() == 1


def test_cpu_only_parallel_workers_are_reserved_individually(monkeypatch):
    monkeypatch.setattr(performance, "usable_cpu_count", lambda: 48)
    resources = preparation_resources("extraction", {"gpus": [-1, -1]})
    assert resources["gpuIds"] == []
    assert resources["cpuThreadsPerRun"] == 2
    assert resources["dataLoaderWorkers"] == 8


@pytest.mark.parametrize("cpus, devices", [(3, 1), (8, 1), (8, 2), (16, 1), (36, 2)])
def test_automatic_worker_budget_fits_shared_scheduler(monkeypatch, cpus, devices):
    from histopilot.workers.training_process import cpu_slots_per_run

    monkeypatch.setattr(performance, "usable_cpu_count", lambda: cpus)
    resources = preparation_resources("extraction", {"gpus": list(range(devices))})
    assert cpu_slots_per_run(resources) <= cpus
