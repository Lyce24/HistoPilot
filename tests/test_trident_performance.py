"""Worker budgets match available CPUs and the actual TRIDENT device fanout."""

import pytest
from test_extractions import extractions as extractions  # pytest fixture

from histopilot.adapters.trident import performance


def task_request(extractions, task_center, options):
    """The preview of ``options`` and what its extraction task asks the Task Center for.

    The request is None when the preview blocks the run.
    """
    service, spec, _slides = extractions
    spec = spec.model_copy(update={"options": {"task": "seg", **options}})
    preview = service.preview(spec)
    if not preview["canRun"]:
        return preview, None
    job = service.submit(spec, preview["previewHash"], "worker-budget")
    return preview, task_center.task(job["taskId"])["request"]


# ``requested`` is the task's data workers: the Task Center runs an extraction on one GPU,
# so only CPU devices fan the resolved budget out within one task.
BUDGETS = [
    (48, {}, 8, 8),
    (48, {"gpus": [0, 1]}, 4, 4),
    (48, {"gpus": [0, 0]}, 8, 8),
    (48, {"gpus": [-1, -1]}, 4, 8),
    (48, {"gpus": [0, -1]}, 4, 4),
    (16, {}, 7, 7),
    (8, {}, 3, 3),
    (4, {"gpus": [0, 1]}, 1, 1),
    (1, {}, 1, 1),
    (48, {"max_workers": 0}, 0, None),
    (4, {"max_workers": 12}, 12, 12),
]


@pytest.mark.parametrize("cpus, options, expected, requested", BUDGETS)
def test_automatic_budget_and_explicit_overrides(
    extractions, task_center, monkeypatch, cpus, options, expected, requested
):
    monkeypatch.setattr(performance, "usable_cpu_count", lambda: cpus)
    assert performance.resolve_max_workers(options) == expected
    preview, request = task_request(extractions, task_center, options)
    if requested is None:
        # TRIDENT's CSV loader needs a worker: zero workers never reach the Task Center.
        assert "TRIDENT_WORKERS_ZERO" in {item["code"] for item in preview["findings"]}
        assert request is None
    else:
        assert request["dataWorkers"] == requested


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


def test_cpu_only_parallel_workers_are_reserved_individually(
    extractions, task_center, monkeypatch
):
    monkeypatch.setattr(performance, "usable_cpu_count", lambda: 48)
    _preview, request = task_request(extractions, task_center, {"gpus": [-1, -1]})
    assert request["lane"] == "cpu"
    assert request["cpuThreads"] == 2
    assert request["dataWorkers"] == 8


@pytest.mark.parametrize("cpus, devices", [(3, 1), (8, 1), (8, 2), (16, 1), (36, 2)])
def test_automatic_worker_budget_fits_shared_scheduler(
    extractions, task_center, monkeypatch, cpus, devices
):
    monkeypatch.setattr(performance, "usable_cpu_count", lambda: cpus)
    _preview, request = task_request(extractions, task_center, {"gpus": list(range(devices))})
    # The runner admits a task beside others while its threads fit the host's CPUs.
    assert request["cpuThreads"] + request["dataWorkers"] <= cpus
