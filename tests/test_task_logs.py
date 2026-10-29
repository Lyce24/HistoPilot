"""Each step of a task's life is logged with its ID, and the request that queued or retried
a task reaches its worker."""

import sys

from support.cli import Service
from support.logs import collect as collect

from histopilot.observability import logs
from histopilot.taskcenter import procs


def owner(tmp_path, **overrides):
    return {
        "kind": "experiment",
        "id": "experiment-1",
        "projectId": "project-1",
        "projectFolder": str(tmp_path / "project"),
        "title": "Experiment",
        **overrides,
    }


def spec(tmp_path, identity, code=0):
    return {
        "id": identity,
        "kind": "generic",
        "adapter": "generic",
        "title": identity,
        "request": {"lane": "cpu", "cpuThreads": 1, "ramGb": 0.1},
        "command": {
            "argv": [sys.executable, "-c", f"import sys; sys.exit({code})"],
            "cwd": str(tmp_path),
            "log": str(tmp_path / "tasks" / f"{identity}.log"),
        },
    }


def test_work_queued_by_a_request_remembers_it_and_logs_with_it(task_center, tmp_path, collect):
    lines = collect("histopilot.taskcenter")
    with logs.bound(request="request-5"):
        task_center.client.enqueue(owner(tmp_path), [spec(tmp_path, "t1")])
    assert task_center.store.requested_by("t1") == "request-5"
    [queued] = lines
    assert queued["msg"] == "task queued" and queued["event"] == "queued"
    assert (queued["task_id"], queued["request_id"]) == ("t1", "request-5")
    assert queued["adapter"] == "generic" and queued["state"] == "queued"
    assert queued["attempt"] == 1 and queued["owner"] == "experiment"
    task_center.finish("t1", "failed", returncode=1, reason="error")
    task_center.store.requeue(["t1"], reason="retry")  # as the runner's own: no request
    assert task_center.store.requested_by("t1") == "request-5"
    with logs.bound(request="request-6"):
        task_center.finish("t1", "failed", returncode=1, reason="error")
        task_center.store.requeue(["t1"], reason="retry")
    assert task_center.store.requested_by("t1") == "request-6"
    assert [(line["event"], line["attempt"], line["request_id"]) for line in lines[1:]] == [
        ("queued", 2, None),
        ("queued", 3, "request-6"),
    ]
    task_center.store.cancel_pending(["t1"])
    assert lines[-1]["msg"] == "task cancelled" and lines[-1]["state"] == "cancelled"


def test_a_retry_through_the_api_hands_its_request_id_to_the_task(
    task_center, tmp_path, monkeypatch
):
    with Service(tmp_path, monkeypatch) as service:
        project = service.create_project("Study")
        folder = service.settings.workspace / "projects" / "Study"
        work = owner(tmp_path, projectId=project, projectFolder=str(folder))
        task_center.client.enqueue(work, [spec(tmp_path, "t2")])
        task_center.finish("t2", "failed", returncode=1, reason="error")
        retried = service.http.post(
            "/api/v1/task-center/tasks/t2/retry",
            json={"operationId": "retry-1"},
            headers={"X-Request-ID": "trace-7"},
        )
    assert retried.status_code == 200, retried.text
    assert task_center.state("t2") == "queued"
    assert task_center.store.requested_by("t2") == "trace-7"


def test_the_runner_logs_each_step_of_a_task_with_its_id(task_center, tmp_path, collect):
    lines = collect("histopilot.taskcenter")
    spawned = {}

    def spawner(command, **options):
        spawned[options["task"]["id"]] = command
        return procs.spawn(command, **options)

    with logs.bound(request="request-9"):
        task_center.client.enqueue(
            owner(tmp_path), [spec(tmp_path, "works"), spec(tmp_path, "breaks", code=3)]
        )
    task_center.client.enqueue(owner(tmp_path, id="experiment-2"), [spec(tmp_path, "plain")])
    runner = task_center.runner(spawner=spawner)
    task_center.tick_until(
        runner,
        lambda: {task["state"] for task in task_center.tasks()} <= {"succeeded", "failed"},
    )
    assert spawned["works"]["env"][logs.REQUEST_ENV] == "request-9"
    assert spawned["breaks"]["env"][logs.REQUEST_ENV] == "request-9"
    assert logs.REQUEST_ENV not in spawned["plain"]["env"]
    assert "runner started" in [line["msg"] for line in lines]
    steps = [(line.get("event"), line.get("task_id"), line.get("state")) for line in lines]
    for task_id, outcome in (("works", "succeeded"), ("breaks", "failed")):
        assert (
            steps.index(("queued", task_id, "queued"))
            < steps.index(("started", task_id, None))
            < steps.index(("finished", task_id, outcome))
        )
    finished = {line["task_id"]: line for line in lines if line.get("event") == "finished"}
    assert finished["works"]["msg"] == "task succeeded" and finished["works"]["exit_code"] == 0
    assert finished["works"]["level"] == "INFO" and finished["works"]["duration_s"] >= 0
    failed = finished["breaks"]
    assert failed["msg"] == "task failed" and failed["level"] == "WARNING"
    assert failed["exit_code"] == 3 and failed["reason"] == "error"
    assert failed["adapter"] == "generic" and failed["attempt"] == 1
    started = next(line for line in lines if line.get("event") == "started")
    assert started["pid"] > 1 and started["lane"] == "cpu"
