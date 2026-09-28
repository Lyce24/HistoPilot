"""Task Center read models for the stage pages: rollup, snapshot, grouped history, task detail."""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from histopilot.api import create_app
from histopilot.config import Settings
from histopilot.taskcenter import default_client, launcher
from histopilot.taskcenter.model import utc_now_iso
from histopilot.taskcenter.service import explain_failure

BASE = "http://127.0.0.1:8787"
API = "/api/v1/task-center"
HOST = {
    "cpuCount": 16,
    "physicalCpuCount": 8,
    "totalRamGb": 64.0,
    "availableRamGb": 48.0,
    "gpus": [
        {
            "index": 0,
            "uuid": "GPU-test",
            "name": "Test GPU",
            "totalMemoryGb": 24.0,
            "usedMemoryGb": 2.0,
            "freeMemoryGb": 22.0,
            "utilizationPercent": 12.0,
        }
    ],
}


@pytest.fixture
def api(tmp_path, monkeypatch):
    folder = tmp_path / "tmp"
    folder.mkdir(mode=0o700)
    monkeypatch.setattr(tempfile, "tempdir", str(folder))
    data = tmp_path / "data"
    data.mkdir()
    settings = Settings(workspace=tmp_path / "workspace", data_roots=(data,))
    with TestClient(create_app(settings), base_url=BASE) as client:
        client.headers["X-HistoPilot-Token"] = client.get("/api/v1/session").json()["token"]
        client.app.state.task_center.host_probe = lambda: json.loads(json.dumps(HOST))
        yield client


@pytest.fixture
def runner_alive(monkeypatch):
    monkeypatch.setattr(launcher, "runner_alive", lambda _lock: True)


def register(client, name="Study"):
    workspace = client.app.state.settings.workspace
    response = client.post(
        "/api/v1/projects",
        json={"name": name, "storagePath": str(workspace / name.lower().replace(" ", "-"))},
    )
    assert response.status_code == 201, response.text
    project = response.json()
    return project["id"], project["storagePath"]


def owner(project, *, kind="experiment", identity="experiment-1", title="Experiment one"):
    project_id, folder = project
    return {
        "kind": kind,
        "id": identity,
        "projectId": project_id,
        "projectFolder": folder,
        "title": title,
    }


def task(project, identity, *, kind="mil-fold", group=("mil-batch", "batch-a"), order=0, **extra):
    folder = Path(project[1])
    request = extra.pop("request", {"lane": "gpu", "vramGb": 2.0, "cpuThreads": 2})
    return {
        "id": identity,
        "kind": kind,
        "adapter": "generic",
        "title": f"Task {identity}",
        "group": {"kind": group[0], "id": group[1]},
        "planOrder": order,
        "request": request,
        "command": {
            "argv": [sys.executable, "-c", "pass"],
            "cwd": str(folder),
            "env": {"PYTHONUNBUFFERED": "1", "SECRET_TOKEN": "hidden"},
            "log": str(folder / "logs" / f"{identity}.log"),
        },
        **extra,
    }


def enqueue(owner_spec, tasks):
    return default_client().store.enqueue(owner_spec, tasks)


def start(identity, gpu=0):
    assert default_client().store.transition(
        identity,
        from_states=("queued",),
        to_state="running",
        started_at=utc_now_iso(),
        gpu=gpu,
        process={"pid": 1, "startTicks": 1, "bootId": "none"},
        detail={"pid": 1, "gpu": gpu},
    )


def finish(identity, state, **exit_record):
    assert default_client().store.transition(
        identity,
        from_states=("queued", "running", "blocked"),
        to_state=state,
        finished_at=utc_now_iso(),
        exit={"reason": "error", **exit_record},
        detail={"exitReason": exit_record.get("reason", "error")},
    )


def rollup(client, **params):
    response = client.get(f"{API}/rollup", params=params)
    assert response.status_code == 200, response.text
    return response.json()


def test_rollup_by_owner_reports_state_progress_queue_place_and_deep_link(api, runner_alive):
    project = register(api)
    enqueue(owner(project), [task(project, f"fold-{index}", order=index) for index in range(3)])
    key = default_client().store.get("fold-0")["ownerKey"]
    queued = rollup(api, owner=key)
    assert queued["state"] == "queued" and queued["queuePosition"] == 1
    assert queued["progress"] == {"completed": 0, "total": 3}
    assert queued["position"] == 1 and queued["ownerKey"] == key
    assert queued["href"] == f"#task-center?owner={key}&project={project[0]}"
    assert queued["projectName"] == "Study" and queued["title"] == "Experiment one"

    start("fold-0")
    running = rollup(api, ownerKind="experiment", ownerId="experiment-1", project=project[0])
    assert running["state"] == "running" and running["active"] == 1 and running["pending"] == 2
    assert running["current"]["taskId"] == "fold-0"
    assert running["eta"]["seconds"] > 0

    default_client().store.hold_owner(key, True)
    finish("fold-0", "succeeded", reason="ok")
    held = rollup(api, owner=key)
    assert held["state"] == "held" and held["held"] is True
    assert "released" in held["waitingReason"]


def test_rollup_needs_attention_names_the_failure_and_links_to_the_task(api, runner_alive):
    project = register(api)
    enqueue(owner(project), [task(project, "fold-1"), task(project, "fold-2", order=1)])
    start("fold-1")
    finish("fold-1", "succeeded", reason="ok")
    finish(
        "fold-2",
        "failed",
        error="Another operation is changing this workspace. Retry after it finishes.",
        returncode=0,
    )
    result = rollup(api, ownerKind="experiment", ownerId="experiment-1")
    assert result["state"] == "attention"
    assert result["progress"] == {"completed": 1, "total": 2}
    failure = result["lastFailure"]
    assert failure["taskId"] == "fold-2" and failure["retry"] == "safe"
    assert failure["message"] == "The project was busy"
    assert "task=fold-2" in result["href"]
    assert result["finishedAt"] is not None


def test_rollup_runner_stopped_cancelled_completed_and_not_started(api):
    project = register(api)
    enqueue(owner(project), [task(project, "fold-1")])
    assert rollup(api, ownerKind="experiment", ownerId="experiment-1")["state"] == "runner-stopped"
    default_client().cancel_task("fold-1")
    cancelled = rollup(api, ownerKind="experiment", ownerId="experiment-1")
    assert cancelled["state"] == "cancelled" and cancelled["live"] == 0
    empty = rollup(api, ownerKind="experiment", ownerId="missing", project=project[0])
    assert empty["state"] == "not-started" and empty["counts"]["queued"] == 0
    enqueue(owner(project, identity="experiment-2"), [task(project, "fold-9")])
    start("fold-9")
    finish("fold-9", "succeeded", reason="ok")
    done = rollup(api, ownerKind="experiment", ownerId="experiment-2")
    assert done["state"] == "completed" and done["startedAt"] and done["finishedAt"]


def test_rollup_ignores_superseded_progress_collections_and_says_what_a_retry_resumes(
    api, runner_alive
):
    project = register(api)
    progress = {"final": False, "batchId": "batch-a"}
    enqueue(
        owner(project),
        [
            task(project, "fold-1"),
            task(project, "collect-p", kind="mil-collect", order=1, adapterData=progress),
            task(
                project,
                "collect-f",
                kind="mil-collect",
                order=2,
                adapterData={"final": True, "batchId": "batch-a"},
            ),
        ],
    )
    key = default_client().store.get("fold-1")["ownerKey"]
    for identity, state in (("fold-1", "succeeded"), ("collect-p", "failed")):
        start(identity)
        finish(identity, state, reason="ok" if state == "succeeded" else "error")
    running = rollup(api, owner=key)
    assert running["state"] == "queued" and running["counts"]["failed"] == 0
    start("collect-f")
    finish("collect-f", "succeeded", reason="ok")
    done = rollup(api, owner=key)
    assert done["state"] == "completed" and done["lastFailure"] is None
    assert done["progress"] == {"completed": 2, "total": 2} and done["retryable"] is False
    # The Task Center still lists the failed collection, but an owner retry resumes nothing.
    assert api.get(f"{API}/owners/{key}").json()["actions"]["retry"] is False
    assert rollup(api)["recentFailures"] == 1

    enqueue(owner(project, identity="experiment-2"), [task(project, "fold-2")])
    finish("fold-2", "failed", reason="oom")
    failed = rollup(api, ownerKind="experiment", ownerId="experiment-2")
    assert failed["state"] == "attention" and failed["retryable"] is True
    assert rollup(api, project=project[0])["retryable"] is None


def test_rollup_counts_only_retained_batches_but_always_live_work(api, runner_alive):
    project = register(api)
    enqueue(
        owner(project),
        [
            task(project, "kept-1"),
            task(project, "trashed-1", group=("mil-batch", "batch-old"), order=1),
            task(project, "trashed-2", group=("mil-batch", "batch-old"), order=2),
        ],
    )
    start("kept-1")
    finish("kept-1", "succeeded", reason="ok")
    finish("trashed-1", "failed", reason="oom")
    everything = rollup(api, ownerKind="experiment", ownerId="experiment-1")
    assert everything["state"] == "queued"
    default_client().cancel_task("trashed-2")
    assert rollup(api, ownerKind="experiment", ownerId="experiment-1")["state"] == "attention"
    kept = rollup(api, ownerKind="experiment", ownerId="experiment-1", batchIds="batch-a")
    assert kept["state"] == "completed" and kept["progress"] == {"completed": 1, "total": 1}
    assert kept["lastFailure"] is None and kept["retryable"] is False
    assert kept["scope"]["batchIds"] == "batch-a"

    enqueue(owner(project), [task(project, "late", group=("mil-batch", "batch-old"), order=3)])
    live = rollup(api, ownerKind="experiment", ownerId="experiment-1", batchIds="batch-a")
    assert live["state"] == "queued" and live["counts"]["queued"] == 1


def test_rollup_by_record_matches_compute_labels_and_record_owners(api, runner_alive):
    project = register(api)
    refit = task(
        project,
        "refit-task",
        kind="compute-job",
        group=("refit", "refit-1"),
        labels={"recordId": "refit-1", "computeKind": "refit", "experimentId": "experiment-1"},
    )
    enqueue(owner(project), [refit])
    enqueue(
        owner(project, kind="extraction", identity="job-7", title="UNI extraction"),
        [
            task(
                project,
                "extract-7",
                kind="extraction",
                group=("extraction", "job-7"),
                labels={"recordKind": "extraction", "recordId": "job-7"},
            )
        ],
    )
    by_label = rollup(api, recordKind="predictor-refit", recordId="refit-1", project=project[0])
    assert by_label["counts"]["queued"] == 1 and by_label["scope"]["recordId"] == "refit-1"
    assert by_label["href"].endswith("task=refit-task") or "task=refit-task" in by_label["href"]
    phase3 = rollup(api, recordKind="extraction", recordId="job-7")
    assert phase3["counts"]["queued"] == 1 and phase3["title"] == "UNI extraction"
    by_owner = rollup(api, ownerKind="extraction", ownerId="job-7")
    assert by_owner["ownerKey"] == phase3["ownerKey"]
    unrelated = rollup(api, recordKind="evaluation", recordId="refit-1")
    assert unrelated["state"] == "not-started"
    several = rollup(api, recordKind="refit", recordIds="missing,refit-1")
    assert several["counts"]["queued"] == 1 and several["scope"]["recordIds"] == "missing,refit-1"
    every_refit = rollup(api, recordKind="refit", project=project[0])
    assert every_refit["counts"]["queued"] == 1 and every_refit["progress"] == {
        "completed": 0,
        "total": 1,
    }
    invalid = api.get(f"{API}/rollup", params={"recordId": "refit-1"})
    assert invalid.status_code == 422
    assert by_label["retryable"] is False
    finish("refit-task", "failed", reason="oom")
    stopped = rollup(api, recordKind="refit", recordId="refit-1", project=project[0])
    assert stopped["state"] == "attention" and stopped["retryable"] is True


def test_machine_and_project_rollups_count_only_recent_failures(api, runner_alive):
    project = register(api)
    other = register(api, "Other")
    enqueue(owner(project), [task(project, "fold-1"), task(project, "fold-2", order=1)])
    enqueue(owner(other, identity="experiment-9"), [task(other, "fold-9")])
    start("fold-1")
    finish("fold-2", "failed", reason="oom")
    machine = rollup(api)
    assert machine["state"] == "running" and machine["progress"] is None
    assert (
        machine["recentFailures"] == 1 and machine["lastFailure"]["message"] == "Out of GPU memory"
    )
    assert machine["counts"]["queued"] == 1 and machine["href"] == "#task-center"
    scoped = rollup(api, project=other[0])
    assert scoped["counts"]["queued"] == 1 and scoped["recentFailures"] == 0
    assert scoped["href"] == f"#task-center?project={other[0]}"
    kinds = rollup(api, project=project[0], kinds="compute-job")
    assert kinds["state"] == "not-started"


def test_snapshot_returns_summary_running_and_live_owners_in_one_read(api, runner_alive):
    project = register(api)
    enqueue(owner(project), [task(project, "fold-1"), task(project, "fold-2", order=1)])
    enqueue(owner(project, identity="done"), [task(project, "old")])
    finish("old", "succeeded", reason="ok")
    start("fold-1")
    snapshot = api.get(f"{API}/snapshot").json()
    assert snapshot["summary"]["counts"]["running"] == 1
    assert "recentFailures" in snapshot["summary"]
    assert [item["id"] for item in snapshot["running"]] == ["fold-1"]
    assert snapshot["running"][0]["owner"]["projectName"] == "Study"
    assert [item["id"] for item in snapshot["owners"]] == ["experiment-1"]
    assert snapshot["owners"][0]["etaSeconds"] > 0 and snapshot["pendingCount"] == 1


def test_history_groups_finished_tasks_by_owner_with_filters_and_paging(api):
    project = register(api)
    other = register(api, "Other")
    enqueue(owner(project), [task(project, f"fold-{index}", order=index) for index in range(3)])
    enqueue(owner(other, identity="experiment-9"), [task(other, "fold-9")])
    finish("fold-0", "succeeded", reason="ok")
    finish("fold-1", "failed", reason="oom")
    finish("fold-9", "succeeded", reason="ok")
    everything = api.get(f"{API}/history").json()
    assert everything["total"] == 2
    first = next(group for group in everything["groups"] if group["owner"]["id"] == "experiment-1")
    assert first["finished"] == {
        "total": 2,
        "succeeded": 1,
        "failed": 1,
        "cancelled": 0,
        "interrupted": 0,
    }
    assert first["lastFailure"]["taskId"] == "fold-1"
    assert first["owner"]["projectName"] == "Study"
    scoped = api.get(f"{API}/history", params={"project": other[0]}).json()
    assert [group["owner"]["id"] for group in scoped["groups"]] == ["experiment-9"]
    failed = api.get(f"{API}/history", params={"state": "failed"}).json()
    assert [group["finished"]["total"] for group in failed["groups"]] == [1]
    page = api.get(f"{API}/history", params={"limit": 1, "offset": 1}).json()
    assert len(page["groups"]) == 1 and page["total"] == 2
    tasks = api.get(
        f"{API}/tasks", params={"state": "history", "limit": 1, "offset": 0, "project": project[0]}
    ).json()
    assert len(tasks["tasks"]) == 1 and tasks["hasMore"] is True


def test_task_detail_has_command_attempts_dependents_failure_and_full_log(api):
    project = register(api)
    fold = task(project, "fold-1")
    collect = task(
        project,
        "collect",
        kind="mil-collect",
        dependsOn=[{"task": "fold-1", "condition": "terminal"}],
    )
    enqueue(owner(project), [fold, collect])
    log = Path(project[1]) / "logs" / "fold-1.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        "line one\n"
        + "x" * 40_000
        + "\nTraceback (most recent call last):\nBlockingIOError: busy\n"
    )
    start("fold-1")
    finish("fold-1", "failed", error="Traceback (most recent call last):\nBlockingIOError: busy")
    detail = api.get(f"{API}/tasks/fold-1").json()
    assert detail["command"]["argv"][-1] == "pass"
    assert detail["command"]["env"] == {"PYTHONUNBUFFERED": "1"}
    assert detail["attempts"][0]["state"] == "failed" and detail["attempts"][0]["startedAt"]
    assert detail["dependents"] == [
        {"task": "collect", "title": "Task collect", "state": "blocked"}
    ]
    assert detail["failure"]["title"] == "The project was busy"
    assert detail["failure"]["retry"] == "safe"
    assert detail["logTruncated"] is True and detail["logSize"] > 40_000
    full = api.get(f"{API}/tasks/fold-1/log")
    assert full.status_code == 200 and full.text.startswith("line one")
    assert "inline" in full.headers["content-disposition"]
    download = api.get(f"{API}/tasks/fold-1/log", params={"download": True})
    assert "attachment" in download.headers["content-disposition"]
    blocked = api.get(f"{API}/tasks/collect").json()
    assert blocked["taskTitles"]["fold-1"] == "Task fold-1"
    missing = api.get(f"{API}/tasks/collect/log")
    assert missing.status_code == 404


def test_log_download_refuses_symlinks(api, tmp_path):
    project = register(api)
    enqueue(owner(project), [task(project, "fold-1")])
    log = Path(project[1]) / "logs" / "fold-1.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    secret = tmp_path / "secret.txt"
    secret.write_text("secret")
    os.symlink(secret, log)
    assert api.get(f"{API}/tasks/fold-1/log").status_code == 404


def test_back_links_for_experiment_refits_and_inference_batches(api):
    project = register(api)
    refit = task(
        project,
        "refit-task",
        kind="compute-job",
        group=("refit", "refit-1"),
        labels={"recordId": "refit-1", "computeKind": "refit", "experimentId": "experiment-1"},
        adapterData={"kind": "refit", "recordId": "refit-1"},
    )
    enqueue(owner(project), [refit])
    member = task(
        project,
        "member",
        kind="compute-job",
        group=("evaluation", "evaluation-1"),
        labels={"recordId": "evaluation-1", "computeKind": "evaluation", "purpose": "inference"},
        adapterData={"kind": "evaluation", "recordId": "evaluation-1"},
    )
    submit = task(project, "submit", kind="bulk-submit", group=("evaluation-batch", "bulk-1"))
    submit["labels"] = {"batchId": "bulk-1"}
    enqueue(owner(project, kind="evaluation-batch", identity="bulk-1"), [member, submit])
    tasks = {item["id"]: item for item in api.get(f"{API}/tasks").json()["tasks"]}
    assert tasks["refit-task"]["link"].endswith("#experiments?experiment=experiment-1&tab=runs")
    assert tasks["submit"]["link"].endswith("#inference?batch=bulk-1")
    owners = api.get(f"{API}/owners").json()["owners"]
    bulk = next(item for item in owners if item["kind"] == "evaluation-batch")
    assert bulk["link"].endswith("#inference?batch=bulk-1")


@pytest.mark.parametrize(
    ("state", "exit_record", "title", "retry"),
    [
        ("failed", {"reason": "oom"}, "Out of GPU memory", "safe"),
        ("failed", {"reason": "error", "returncode": 75}, "The project was busy", "safe"),
        ("failed", {"reason": "cuda_failure"}, "GPU device failure", "check"),
        ("interrupted", {"reason": "lost", "lost": True}, "Process lost", "safe"),
        ("cancelled", {"reason": "cancelled"}, "Cancelled", "safe"),
        (
            "failed",
            {"reason": "error", "error": "ModuleNotFoundError: No module named 'torch'"},
            "A Python package is missing",
            "after-fix",
        ),
        ("failed", {"reason": "error", "returncode": 3}, "The task failed", "unknown"),
    ],
)
def test_failures_are_explained_in_plain_language(state, exit_record, title, retry):
    explained = explain_failure({"state": state, "exit": exit_record, "error": None})
    assert explained["title"] == title and explained["retry"] == retry
    assert explain_failure({"state": "succeeded", "exit": {"reason": "ok"}}) is None
