"""Task Center HTTP surface: read models, capacity, routed actions and workspace scoping."""

import hashlib
import inspect
import json
import os
import runpy
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from histopilot.api import create_app
from histopilot.config import Settings
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import ScientificStore
from histopilot.taskcenter import default_client, leases
from histopilot.taskcenter.model import LIVE, utc_now_iso
from histopilot.taskcenter.runner import code_hash
from histopilot.taskcenter.service import derived_operation_id
from histopilot.workers.training_process import process_identity

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
def registry(tmp_path, monkeypatch):
    folder = tmp_path / "tmp"
    folder.mkdir(mode=0o700)
    monkeypatch.setattr(tempfile, "tempdir", str(folder))
    return folder / f"histopilot-training-{os.getuid()}"


@pytest.fixture
def api(tmp_path, registry):
    data = tmp_path / "data"
    data.mkdir()
    settings = Settings(workspace=tmp_path / "workspace", data_roots=(data,))
    with TestClient(create_app(settings), base_url=BASE) as client:
        client.headers["X-HistoPilot-Token"] = client.get("/api/v1/session").json()["token"]
        client.app.state.task_center.host_probe = lambda: json.loads(json.dumps(HOST))
        yield client


def register(client, name="Study"):
    workspace = client.app.state.settings.workspace
    response = client.post(
        "/api/v1/projects",
        json={"name": name, "storagePath": str(workspace / name.lower().replace(" ", "-"))},
    )
    assert response.status_code == 201, response.text
    project = response.json()
    return project["id"], project["storagePath"]


def owner(project, *, kind="experiment", identity="experiment-1", title="Experiment one", **extra):
    project_id, folder = project
    return {
        "kind": kind,
        "id": identity,
        "projectId": project_id,
        "projectFolder": folder,
        "title": title,
        **extra,
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
            "log": str(folder / "logs" / f"{identity}.log"),
        },
        **extra,
    }


def enqueue(owner_spec, tasks):
    return default_client().store.enqueue(owner_spec, tasks)


def start(identity, gpu=0):
    store = default_client().store
    assert store.transition(
        identity,
        from_states=("queued",),
        to_state="running",
        started_at=utc_now_iso(),
        gpu=gpu,
        process={"pid": 1, "startTicks": 1, "bootId": "none"},
    )


def finish(identity, state):
    store = default_client().store
    assert store.transition(
        identity, from_states=("queued", "running"), to_state=state, exit={"reason": "error"}
    )


def action(client, path, operation, **body):
    return client.post(f"{API}{path}", json={"operationId": operation, **body})


def test_summary_reports_runner_capacity_counts_eta_and_foreign_leases(api, registry, tmp_path):
    project = register(api)
    enqueue(owner(project), [task(project, "fold-1"), task(project, "fold-2", order=1)])
    start("fold-1")
    registry.mkdir(mode=0o700)
    me = process_identity()
    lease = {"process": me, "processGroupId": me["pid"], "gpu": 0, "cpus": 4, "ramGb": 8.0}
    (registry / f"lease-{me['pid']}.json").write_text(
        json.dumps({**lease, "runsPerGpu": 4, "kind": "extraction"})
    )
    store = default_client().store
    store.write_runner(
        pid=me["pid"],
        start_ticks=me["startTicks"],
        boot_id=me["bootId"],
        code_hash=code_hash(),
        protocol=1,
        heartbeat_at=utc_now_iso(),
        state="running",
    )
    response = api.get(f"{API}/summary")
    assert response.status_code == 200, response.text
    summary = response.json()
    assert summary["runner"] == {
        "alive": False,
        "heartbeatAt": summary["runner"]["heartbeatAt"],
        "startedAt": None,
        "pid": me["pid"],
        "state": "stopped",
        "message": None,
        "codeHash": code_hash(),
        "codeCurrent": True,
        "autostart": False,
    }
    assert summary["paused"] is False
    [gpu] = summary["capacity"]["gpus"]
    assert gpu["index"] == 0 and gpu["name"] == "Test GPU"
    assert (gpu["slots"], gpu["usedSlots"], gpu["foreignSlots"]) == (4, 2, 1)
    assert gpu["committedMemoryGb"] == 2.0 and gpu["totalMemoryGb"] == 24.0
    assert gpu["utilizationPercent"] == 12.0
    assert summary["capacity"]["cpu"] == {
        "logical": 16,
        "physical": 8,
        "committedThreads": 2 + 4,
        "reserveThreads": 2,
        "cpuTaskSlots": 4,
        "usedCpuTasks": 0,
    }
    assert summary["capacity"]["ram"]["totalGb"] == 64.0
    assert summary["capacity"]["ram"]["reserveGb"] == pytest.approx(3.2)
    assert summary["counts"]["queued"] == 1 and summary["counts"]["running"] == 1
    # One running and one queued GPU task with no measurements: 2 x 600 s over 4 slots.
    assert summary["eta"]["basis"] == "estimated"
    assert 290 <= summary["eta"]["seconds"] <= 300
    assert summary["foreignLeases"] == [
        {"kind": "extraction", "gpu": 0, "cpus": 4, "ramGb": 8.0, "runsPerGpu": 4}
    ]
    assert summary["workspace"] == str(api.app.state.settings.workspace)
    store.write_runner(code_hash="stale")
    assert api.get(f"{API}/summary").json()["runner"]["codeCurrent"] is False


def test_summary_prefers_a_fresh_runner_sample_to_probing_the_host(api):
    def unexpected():
        raise AssertionError("A fresh runner sample must not be re-probed.")

    api.app.state.task_center.host_probe = unexpected
    sample_host = {key: value for key, value in HOST.items() if key != "gpus"}
    gpus = [{**HOST["gpus"][0], "index": 1, "name": "Sampled GPU"}]
    default_client().store.write_runner(
        sample={"host": {**sample_host, "cpuCount": 32}, "gpus": gpus, "at": utc_now_iso()}
    )
    summary = api.get(f"{API}/summary").json()
    assert summary["capacity"]["cpu"]["logical"] == 32
    assert [gpu["name"] for gpu in summary["capacity"]["gpus"]] == ["Sampled GPU"]
    assert api.get(f"{API}/capacity").json()["effective"]["gpuSlots"] == {"1": 4}


def test_stop_and_hold_pauses_training_but_not_collections_or_services(api):
    project = register(api)
    result = enqueue(
        owner(project),
        [
            task(project, "fold"),
            task(
                project,
                "collect",
                kind="mil-collect",
                order=1,
                request={"lane": "cpu", "cpuThreads": 1},
            ),
            task(
                project,
                "coordinator",
                kind="predictor-coordinator",
                group=("predictor-coordinator", "experiment-1"),
                request={"lane": "cpu", "cpuThreads": 1, "service": True},
            ),
        ],
    )
    for identity in ("fold", "collect", "coordinator"):
        start(identity, gpu=0 if identity == "fold" else None)
    stopped = action(api, f"/owners/{result['owner']['key']}/stop", "stop-1").json()
    assert stopped["held"] is True and stopped["actions"]["release"] is True
    store = default_client().store
    assert [store.get(name)["state"] for name in ("fold", "collect", "coordinator")] == [
        "stopping",
        "running",
        "running",
    ]


def test_polled_reads_take_no_project_locks_probes_or_lease_deletions(
    api, registry, monkeypatch, tmp_path
):
    project = register(api)
    key = enqueue(owner(project), [task(project, "fold-1")])["owner"]["key"]
    registry.mkdir(mode=0o700)
    dead = {"pid": 999_999, "startTicks": 1, "bootId": "an-earlier-boot"}
    stale = registry / "lease-999999.json"
    stale.write_text(json.dumps({"process": dead, "gpu": 0, "cpus": 1, "ramGb": 1}))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("A polled Task Center read touched project state or runtimes.")

    # Patch attributes looked up at call time only: a module imported while a module-level
    # function is replaced would keep the replacement after the test.
    monkeypatch.setattr(api.app.state.projects, "scientific_store", forbidden)
    monkeypatch.setattr(ScientificStore, "initialize", forbidden)
    monkeypatch.setattr(subprocess.Popen, "__init__", forbidden)
    monkeypatch.setattr(leases, "remove_task_lease", forbidden)
    for path in (
        "/summary",
        "/tasks",
        "/tasks?state=history",
        "/tasks?state=live&limit=5",
        "/tasks/fold-1",
        "/owners",
        "/owners?scope=all",
        f"/owners/{key}",
        "/capacity",
    ):
        response = api.get(f"{API}{path}")
        assert response.status_code == 200, (path, response.text)
    assert stale.exists()


def test_tasks_filters_positions_links_and_detail_with_log_tail(api):
    project = register(api)
    labels = {"experimentId": "experiment-1", "batchId": "batch-a"}
    enqueue(
        owner(project),
        [
            task(project, "fold-1", labels=labels),
            task(project, "fold-2", order=1, labels=labels),
            task(
                project,
                "collect",
                kind="mil-collect",
                order=2,
                priority="interactive",
                request={"lane": "cpu", "cpuThreads": 1},
                dependsOn=[{"task": "fold-1", "condition": "terminal"}],
            ),
        ],
    )
    enqueue(
        owner(project, kind="model-evaluation", identity="configuration-e", title="Evaluation"),
        [
            task(
                project,
                "evaluation",
                kind="compute-job",
                group=("evaluation", "configuration-e"),
                labels={"purpose": "inference"},
                adapterData={"recordId": "configuration-e", "kind": "evaluation"},
            )
        ],
    )
    start("fold-1")
    finish("fold-2", "failed")
    tasks = api.get(f"{API}/tasks").json()["tasks"]
    # The interactive collection sorts first; the queue position counts pending tasks only.
    assert [item["id"] for item in tasks] == ["collect", "fold-1", "fold-2", "evaluation"]
    by_id = {item["id"]: item for item in tasks}
    fold = by_id["fold-1"]
    assert fold["state"] == "running" and fold["lane"] == "gpu" and fold["gpu"] == 0
    assert fold["owner"]["sameWorkspace"] is True and fold["owner"]["kind"] == "experiment"
    assert fold["owner"]["projectId"] == project[0]
    assert fold["group"] == {"kind": "mil-batch", "id": "batch-a"}
    assert fold["link"] == (
        f"?project={project[0]}#experiments?experiment=experiment-1&tab=runs&batch=batch-a"
    )
    assert fold["actions"] == {"cancel": True, "retry": False}
    assert fold["request"]["vramGb"] == 2.0 and fold["exit"] is None
    assert fold["logPath"].endswith("logs/fold-1.log")
    # Its batch still runs fold-1, and a batch resumes only as a whole.
    assert by_id["fold-2"]["actions"] == {"cancel": False, "retry": False}
    assert by_id["fold-2"]["exit"]["reason"] == "error"
    assert by_id["collect"]["state"] == "blocked" and by_id["collect"]["queuePosition"] == 1
    assert by_id["evaluation"]["queuePosition"] == 2 and fold["queuePosition"] is None
    assert by_id["evaluation"]["link"] == (
        f"?project={project[0]}#inference?evaluation=configuration-e"
    )
    live = api.get(f"{API}/tasks", params={"state": "live"}).json()["tasks"]
    assert {item["id"] for item in live} == {"fold-1", "collect", "evaluation"}
    history = api.get(f"{API}/tasks", params={"state": "history"}).json()["tasks"]
    assert [item["id"] for item in history] == ["fold-2"]
    both = api.get(f"{API}/tasks", params={"state": "running,failed"}).json()["tasks"]
    assert {item["id"] for item in both} == {"fold-1", "fold-2"}
    owner_key = fold["owner"]["key"]
    owned = api.get(f"{API}/tasks", params={"owner": owner_key}).json()["tasks"]
    assert {item["id"] for item in owned} == {"fold-1", "fold-2", "collect"}
    kinds = api.get(f"{API}/tasks", params={"kind": "compute-job,mil-collect"}).json()["tasks"]
    assert {item["id"] for item in kinds} == {"collect", "evaluation"}
    by_project = api.get(f"{API}/tasks", params={"project": project[0], "limit": 2}).json()
    assert len(by_project["tasks"]) == 2
    assert api.get(f"{API}/tasks", params={"project": "project-unknown"}).json() == {"tasks": []}
    invalid = api.get(f"{API}/tasks", params={"state": "sleeping"})
    assert invalid.status_code == 422 and invalid.json()["code"] == "TASK_CENTER_FILTER_INVALID"
    assert api.get(f"{API}/tasks", params={"limit": 0}).status_code == 422

    log = Path(project[1]) / "logs" / "fold-1.log"
    log.parent.mkdir(exist_ok=True)
    log.write_bytes(b"x" * 20_000 + "é end of log\n".encode())
    detail = api.get(f"{API}/tasks/fold-1").json()
    assert detail["id"] == "fold-1" and detail["logTruncated"] is True
    assert len(detail["logTail"].encode()) <= 16 * 1024 and detail["logTail"].endswith(
        "é end of log\n"
    )
    assert [event["toState"] for event in detail["events"]] == ["queued", "running"]
    blocked = api.get(f"{API}/tasks/collect").json()
    assert blocked["dependencies"] == [
        {"task": "fold-1", "condition": "terminal", "state": "running"}
    ]
    assert blocked["logTail"] is None and blocked["logTruncated"] is False
    missing = api.get(f"{API}/tasks/task-missing")
    assert missing.status_code == 404 and missing.json()["code"] == "TASK_NOT_FOUND"


def test_owners_are_ordered_with_positions_counts_links_and_actions(api):
    project = register(api)
    enqueue(owner(project), [task(project, "a-1"), task(project, "a-2", order=1)])
    enqueue(
        owner(project, kind="mil-batch", identity="batch-b", title="Batch B"),
        [task(project, "b-1", group=("mil-batch", "batch-b"))],
    )
    enqueue(
        owner(project, kind="evaluation-batch", identity="bulk-c", title="Bulk C"),
        [task(project, "c-1", kind="bulk-submit", group=("evaluation-batch", "bulk-c"))],
    )
    start("a-1")
    finish("c-1", "failed")
    owners = api.get(f"{API}/owners").json()["owners"]
    assert [item["id"] for item in owners] == ["experiment-1", "batch-b"]
    first, second = owners
    assert (first["position"], second["position"]) == (1, 2)
    assert first["counts"]["running"] == 1 and first["counts"]["queued"] == 1
    assert first["lanes"] == {"gpu": 2, "cpu": 0}
    assert first["sameWorkspace"] is True and first["held"] is False
    assert (
        first["link"]
        == f"?project={project[0]}#experiments?experiment=experiment-1&tab=runs&batch=batch-a"
    )
    assert second["link"] == (
        f"?project={project[0]}#experiments?experiment=legacy-batch-b&tab=runs&batch=batch-b"
    )
    assert first["actions"] == {
        "hold": True,
        "release": False,
        "stop": True,
        "cancel": True,
        "retry": False,
        "moveUp": False,
        "moveDown": True,
    }
    assert second["actions"]["moveUp"] is True and second["actions"]["stop"] is False
    # a-1 runs (600 s left), a-2 starts after 600/4 s, b-1 after 1200/4 s; 600 s each.
    assert first["etaSeconds"] in (749, 750) and second["etaSeconds"] in (899, 900)
    everything = api.get(f"{API}/owners", params={"scope": "all"}).json()["owners"]
    assert [item["id"] for item in everything] == ["experiment-1", "batch-b", "bulk-c"]
    bulk = everything[2]
    assert bulk["position"] is None
    assert bulk["link"] == f"?project={project[0]}#evaluation?batch=bulk-c"
    assert bulk["actions"]["retry"] is True and bulk["actions"]["cancel"] is False
    assert api.get(f"{API}/owners", params={"scope": "some"}).status_code == 422


def test_capacity_get_put_validation_idempotency_and_conflict(api):
    initial = api.get(f"{API}/capacity")
    assert initial.status_code == 200, initial.text
    body = initial.json()
    assert body["settings"]["defaultGpuSlots"] == 4 and body["settings"]["gpuSlots"] == {}
    assert body["effective"]["gpuSlots"] == {"0": 4} and body["effective"]["cpuTaskSlots"] == 4
    suggestion = body["suggestion"]
    assert suggestion["version"] == 2 and suggestion["basis"] == "hardware_only"
    assert suggestion["parallelGpuTasks"] >= 1 and suggestion["explanation"]

    updated = api.put(f"{API}/capacity", json={"operationId": "capacity-1", "parallelGpuTasks": 5})
    assert updated.status_code == 200, updated.text
    assert updated.json()["settings"]["defaultGpuSlots"] == 5
    assert updated.json()["settings"]["gpuSlots"] == {"0": 5}
    assert updated.json()["effective"]["gpuSlots"] == {"0": 5}
    replay = api.put(f"{API}/capacity", json={"operationId": "capacity-1", "parallelGpuTasks": 5})
    assert replay.status_code == 200 and replay.json()["settings"] == updated.json()["settings"]
    conflict = api.put(f"{API}/capacity", json={"operationId": "capacity-1", "parallelGpuTasks": 3})
    assert conflict.status_code == 409 and conflict.json()["code"] == "OPERATION_CONFLICT"

    changed = api.put(
        f"{API}/capacity",
        json={
            "operationId": "capacity-2",
            "gpuSlots": {"0": 2},
            "cpuTaskSlots": 3,
            "paused": True,
            "autoResume": False,
            "defaults": {"dataLoaderWorkers": 0},
        },
    ).json()
    assert changed["settings"]["gpuSlots"] == {"0": 2}
    assert changed["settings"]["defaultGpuSlots"] == 5
    assert changed["settings"]["cpuTaskSlots"] == 3 and changed["effective"]["cpuTaskSlots"] == 3
    assert changed["settings"]["paused"] is True and changed["settings"]["autoResume"] is False
    assert changed["settings"]["defaults"] == {"cpuThreadsPerRun": 2, "dataLoaderWorkers": 0}
    assert api.get(f"{API}/summary").json()["paused"] is True
    for invalid in (
        {"parallelGpuTasks": 0},
        {"parallelGpuTasks": 17},
        {"parallelGpuTasks": "5"},
        {"parallelGpuTasks": True},
        {"gpuSlots": {"gpu0": 2}},
        {"gpuSlots": {"0": 20}},
        {"cpuTaskSlots": 129},
        {"paused": "yes"},
        {"defaults": {"cpuThreadsPerRun": 0}},
        {"defaults": {"dataLoaderWorkers": 17}},
        {"reserves": {"cpuThreads": 1}},
    ):
        response = api.put(f"{API}/capacity", json={"operationId": "capacity-bad", **invalid})
        assert response.status_code == 422, (invalid, response.text)
    assert api.put(f"{API}/capacity", json={"parallelGpuTasks": 2}).status_code == 422
    assert api.get(f"{API}/capacity").json()["settings"]["gpuSlots"] == {"0": 2}


def test_capacity_null_cpu_task_slots_restores_the_automatic_value(api):
    changed = api.put(f"{API}/capacity", json={"operationId": "slots-1", "cpuTaskSlots": 3})
    assert changed.json()["effective"]["cpuTaskSlots"] == 3
    reset = api.put(f"{API}/capacity", json={"operationId": "slots-2", "cpuTaskSlots": None})
    assert reset.status_code == 200, reset.text
    assert reset.json()["settings"]["cpuTaskSlots"] is None
    assert reset.json()["effective"]["cpuTaskSlots"] == 4


def test_hold_release_move_and_stop_steer_a_store_only_owner(api):
    project = register(api)
    first = enqueue(owner(project), [task(project, "a-1"), task(project, "a-2", order=1)])
    second = enqueue(
        owner(project, kind="generic-owner", identity="other", title="Other"),
        [task(project, "b-1", kind="generic-task", group=("generic", "other"))],
    )
    key, other = first["owner"]["key"], second["owner"]["key"]
    start("a-1")

    held = action(api, f"/owners/{key}/hold", "hold-1")
    assert held.status_code == 200, held.text
    assert held.json()["held"] is True and held.json()["actions"]["release"] is True
    released = action(api, f"/owners/{key}/release", "release-1").json()
    assert released["held"] is False
    # A replayed hold is acknowledged without holding the owner again.
    assert action(api, f"/owners/{key}/hold", "hold-1").json()["held"] is False
    conflict = action(api, f"/owners/{key}/release", "hold-1")
    assert conflict.status_code == 409 and conflict.json()["code"] == "OPERATION_CONFLICT"

    moved = action(api, f"/owners/{other}/move", "move-1", position="up").json()
    assert moved["position"] == 1 and moved["actions"]["moveUp"] is False
    owners = api.get(f"{API}/owners").json()["owners"]
    assert [item["key"] for item in owners] == [other, key]
    assert action(api, f"/owners/{key}/move", "move-2").status_code == 422
    assert action(api, f"/owners/{key}/move", "move-3", position="sideways").status_code == 422
    assert action(api, f"/owners/{key}/pause", "pause-1").status_code == 422
    missing = action(api, "/owners/owner-missing/hold", "hold-missing")
    assert missing.status_code == 404 and missing.json()["code"] == "TASK_OWNER_NOT_FOUND"

    stopped = action(api, f"/owners/{key}/stop", "stop-1").json()
    assert stopped["held"] is True
    store = default_client().store
    running = store.get("a-1")
    assert running["state"] == "stopping" and running["stopRequest"] == "pause"
    assert store.get("a-2")["state"] == "queued"
    assert not (Path(project[1]) / "cancel.json").exists()


def test_cancel_and_retry_of_generic_tasks_go_through_the_store(api):
    project = register(api)
    enqueue(
        owner(project, kind="generic-owner", identity="jobs", title="Jobs"),
        [
            task(project, "queued", kind="generic-task", group=("generic", "jobs")),
            task(project, "running", kind="generic-task", group=("generic", "jobs"), order=1),
        ],
    )
    start("running")
    cancelled = action(api, "/tasks/queued/cancel", "cancel-queued")
    assert cancelled.status_code == 200, cancelled.text
    assert cancelled.json()["state"] == "cancelled"
    assert cancelled.json()["actions"] == {"cancel": False, "retry": True}
    stopping = action(api, "/tasks/running/cancel", "cancel-running").json()
    assert stopping["state"] == "stopping" and stopping["stopRequest"] == "cancel"
    assert stopping["actions"]["cancel"] is False
    retried = action(api, "/tasks/queued/retry", "retry-queued").json()
    assert retried["state"] == "queued" and retried["attempt"] == 2
    # Replays never add attempts.
    assert action(api, "/tasks/queued/retry", "retry-queued").json()["attempt"] == 2
    assert action(api, "/tasks/queued/stop", "stop-queued").status_code == 422
    missing = action(api, "/tasks/missing/cancel", "cancel-missing")
    assert missing.status_code == 404 and missing.json()["code"] == "TASK_NOT_FOUND"


class Recorder:
    """Fake application services that record the routed calls and mimic their queue effects."""

    def __init__(self):
        self.calls = []

    def services(self):
        recorder, client = self, default_client()

        class Training:
            def __init__(self, store, _filesystem):
                self.folder = str(store.folder)

            def cancel(self, batch, operation):
                recorder.calls.append(("training.cancel", batch, operation, self.folder))
                # The batch marker the real service writes before any stop request.
                marker = Path(self.folder) / "training" / batch / "cancel.json"
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text("{}")
                return client.cancel_group(
                    "mil-batch", batch, self.folder, exclude_kinds=("mil-collect",)
                )

            def cancel_runs(self, batch, runs, operation):
                recorder.calls.append(("training.cancel_runs", batch, runs, operation))
                for item in client.store.list(
                    group=("mil-batch", batch), project_folder=self.folder, limit=None
                ):
                    if item["adapterData"].get("runId") in runs:
                        client.cancel_task(item["id"])

            def launch(self, batch, operation, *, resume=False):
                recorder.calls.append(("training.launch", batch, operation, resume))
                return client.requeue_group(
                    "mil-batch",
                    batch,
                    self.folder,
                    kinds=("mil-fold", "mil-collect"),
                    reason="resume",
                )

        class Predictors:
            def __init__(self, _store, _filesystem):
                pass

            def cancel(self, experiment, operation):
                recorder.calls.append(("predictors.cancel", experiment, operation))

            def launch(self, experiment, operation, *, resume=False):
                recorder.calls.append(("predictors.launch", experiment, operation, resume))

        class Compute:
            def __init__(self, _store, _filesystem):
                pass

            def cancel(self, record, operation=None):
                recorder.calls.append(("compute.cancel", record, operation))

        class Launcher:
            def __init__(self, name):
                self.name = name

            def __call__(self, _store, _filesystem):
                return self

            def launch(self, record, request, *, resume=False):
                recorder.calls.append((f"{self.name}.launch", record, request, resume))

        class Bulk:
            def __init__(self, _store, _filesystem):
                pass

            def cancel(self, batch, operation):
                recorder.calls.append(("bulk.cancel", batch, operation))

        return {
            "training": Training,
            "predictors": Predictors,
            "compute": Compute,
            "refits": Launcher("refits"),
            "evaluations": Launcher("evaluations"),
            "interpretations": Launcher("interpretations"),
            "bulk": Bulk,
        }


def test_experiment_cancel_and_retry_route_through_the_owning_services(api):
    project = register(api)
    recorder = Recorder()
    api.app.state.task_center.services = recorder.services()
    result = enqueue(
        owner(project),
        [
            task(project, "a-1"),
            task(project, "a-2", order=1),
            task(
                project,
                "a-collect",
                kind="mil-collect",
                order=2,
                request={"lane": "cpu", "cpuThreads": 1},
                dependsOn=[
                    {"task": "a-1", "condition": "terminal"},
                    {"task": "a-2", "condition": "terminal"},
                ],
            ),
            task(project, "b-1", group=("mil-batch", "batch-b")),
            task(
                project,
                "coordinator",
                kind="predictor-coordinator",
                group=("experiment", "experiment-1"),
                request={"lane": "cpu", "cpuThreads": 1, "service": True},
            ),
            task(
                project,
                "refit",
                kind="compute-job",
                group=("refit", "configuration-r"),
                adapterData={"recordId": "configuration-r", "kind": "refit"},
            ),
        ],
    )
    key = result["owner"]["key"]
    finish("b-1", "succeeded")
    start("a-1")

    response = action(api, f"/owners/{key}/cancel", "cancel-1")
    assert response.status_code == 200, response.text
    derived = lambda target: derived_operation_id("cancel", "cancel-1", target)  # noqa: E731
    assert (
        derived("batch-a")
        == "task-center-cancel-" + hashlib.sha256(b"cancel-1batch-a").hexdigest()[:32]
    )
    assert recorder.calls == [
        ("predictors.cancel", "experiment-1", derived("experiment-1")),
        ("training.cancel", "batch-a", derived("batch-a"), project[1]),
        ("compute.cancel", "configuration-r", derived("configuration-r")),
    ]
    store = default_client().store
    assert store.get("a-1")["state"] == "stopping" and store.get("a-2")["state"] == "cancelled"
    # The final collection still records the batch outcome; the pending refit never started.
    assert store.get("a-collect")["state"] == "blocked"
    assert store.get("refit")["state"] == "cancelled"
    assert store.get("b-1")["state"] == "succeeded"
    assert action(api, f"/owners/{key}/cancel", "cancel-1").status_code == 200
    assert len(recorder.calls) == 3

    # Retry waits while a fold of the batch is still stopping.
    busy = action(api, f"/owners/{key}/retry", "retry-busy")
    assert busy.status_code == 409 and busy.json()["code"] == "TASK_RETRY_BUSY"
    assert store.transition("a-1", from_states=("stopping",), to_state="cancelled")
    assert store.promote_ready() == ["a-collect"]
    finish("a-collect", "succeeded")
    finish("coordinator", "failed")
    recorder.calls.clear()
    retried = action(api, f"/owners/{key}/retry", "retry-1")
    assert retried.status_code == 200, retried.text
    retry = lambda target: derived_operation_id("retry", "retry-1", target)  # noqa: E731
    assert recorder.calls == [
        ("training.launch", "batch-a", retry("batch-a"), True),
        ("predictors.launch", "experiment-1", retry("experiment-1"), True),
    ]
    assert store.get("a-1")["state"] == "queued" and store.get("a-1")["attempt"] == 2
    assert store.get("a-collect")["state"] == "blocked"
    assert store.get("refit")["state"] == "cancelled"  # the coordinator relaunches its refits


def test_owner_retry_resumes_a_separate_refit_after_the_coordinator_completed(api):
    project = register(api)
    recorder = Recorder()
    api.app.state.task_center.services = recorder.services()
    result = enqueue(
        owner(project),
        [
            task(
                project,
                "coordinator",
                kind="predictor-coordinator",
                group=("experiment", "experiment-1"),
                request={"lane": "cpu", "cpuThreads": 1, "service": True},
            ),
            task(
                project,
                "refit",
                kind="compute-job",
                group=("refit", "configuration-r"),
                adapterData={"recordId": "configuration-r", "kind": "refit"},
            ),
        ],
    )
    finish("coordinator", "succeeded")
    finish("refit", "failed")
    response = action(api, f"/owners/{result['owner']['key']}/retry", "retry-1")
    assert response.status_code == 200, response.text
    [(name, record, request, resume)] = recorder.calls
    assert (name, record, resume) == ("refits.launch", "configuration-r", True)
    assert request.operationId == derived_operation_id("retry", "retry-1", "configuration-r")


def test_task_actions_route_compute_coordinator_and_bulk_work(api):
    project = register(api)
    recorder = Recorder()
    api.app.state.task_center.services = recorder.services()
    for kind, record in (
        ("refit", "configuration-r"),
        ("evaluation", "configuration-e"),
        ("interpretation", "configuration-i"),
    ):
        enqueue(
            owner(project, kind=kind, identity=record, title=kind),
            [
                task(
                    project,
                    kind,
                    kind="compute-job",
                    group=(kind, record),
                    adapterData={"recordId": record, "kind": kind},
                )
            ],
        )
    bulk = enqueue(
        owner(project, kind="evaluation-batch", identity="bulk-1", title="Bulk"),
        [
            task(project, "submit", kind="bulk-submit", group=("evaluation-batch", "bulk-1")),
            task(
                project,
                "member",
                kind="compute-job",
                group=("evaluation", "configuration-m"),
                adapterData={"recordId": "configuration-m", "kind": "evaluation"},
            ),
        ],
    )
    enqueue(
        owner(project, identity="experiment-2"),
        [task(project, "coordinator", kind="predictor-coordinator", group=("x", "y"))],
    )
    cancelled = action(api, "/tasks/refit/cancel", "cancel-refit").json()
    assert cancelled["state"] == "cancelled"
    assert recorder.calls[-1] == (
        "compute.cancel",
        "configuration-r",
        derived_operation_id("cancel", "cancel-refit", "configuration-r"),
    )
    retried = action(api, "/tasks/refit/retry", "retry-refit")
    assert retried.status_code == 200, retried.text
    name, record, request, resume = recorder.calls[-1]
    assert (name, record, resume) == ("refits.launch", "configuration-r", True)
    assert request.operationId == derived_operation_id("retry", "retry-refit", "configuration-r")
    for kind in ("evaluation", "interpretation"):
        finish(kind, "interrupted")
        assert action(api, f"/tasks/{kind}/retry", f"retry-{kind}").status_code == 200
        record = f"configuration-{kind[0]}"
        assert recorder.calls[-1] == (
            f"{kind}s.launch",
            record,
            derived_operation_id("retry", f"retry-{kind}", record),
            True,
        )
    finish("coordinator", "interrupted")
    assert action(api, "/tasks/coordinator/retry", "retry-coordinator").status_code == 200
    assert recorder.calls[-1] == (
        "predictors.launch",
        "experiment-2",
        derived_operation_id("retry", "retry-coordinator", "experiment-2"),
        True,
    )
    response = action(api, f"/owners/{bulk['owner']['key']}/cancel", "cancel-bulk")
    assert response.status_code == 200, response.text
    assert recorder.calls[-1] == (
        "bulk.cancel",
        "bulk-1",
        derived_operation_id("cancel", "cancel-bulk", "bulk-1"),
    )
    store = default_client().store
    assert (
        store.get("submit")["state"] == "cancelled" and store.get("member")["state"] == "cancelled"
    )
    requeued = action(api, "/tasks/submit/retry", "retry-submit").json()
    assert requeued["state"] == "queued" and requeued["attempt"] == 2


def collect(project, identity, folds, *, batch="batch-a", final=True):
    """A batch's results collection, waiting on its folds like the real final one."""
    return task(
        project,
        identity,
        kind="mil-collect",
        group=("mil-batch", batch),
        order=100,
        priority="interactive",
        request={"lane": "cpu", "cpuThreads": 1},
        adapterData={"final": final, "batchFolder": str(Path(project[1]) / "training" / batch)},
        dependsOn=[{"task": fold, "condition": "terminal"} for fold in folds],
    )


def test_a_single_fold_cancel_is_recorded_by_its_batch(api):
    project = register(api)
    recorder = Recorder()
    api.app.state.task_center.services = recorder.services()
    folds = [
        task(project, f"fold-{index}", order=index, adapterData={"runId": f"run-{index}"})
        for index in range(3)
    ]
    enqueue(owner(project), [*folds, collect(project, "final", ["fold-0", "fold-1", "fold-2"])])
    start("fold-0")
    # A queued fold: the batch marks the run cancelled before the runner could start it.
    cancelled = action(api, "/tasks/fold-1/cancel", "cancel-one")
    assert cancelled.status_code == 200, cancelled.text
    assert cancelled.json()["state"] == "cancelled"
    derived = derived_operation_id("cancel", "cancel-one", "batch-a")
    assert recorder.calls == [("training.cancel_runs", "batch-a", ["run-1"], derived)]
    # A running fold is stopped the same way, never through the batch-wide cancel.
    stopping = action(api, "/tasks/fold-0/cancel", "cancel-running").json()
    assert stopping["state"] == "stopping" and stopping["stopRequest"] == "cancel"
    assert recorder.calls[-1][:3] == ("training.cancel_runs", "batch-a", ["run-0"])
    assert not any(call[0] == "training.cancel" for call in recorder.calls)
    store = default_client().store
    assert store.get("fold-2")["state"] == "queued"
    # A collection, or a fold whose run cannot be resolved, is cancelled in the store.
    enqueue(owner(project), [task(project, "orphan", group=("other", "x"), order=5)])
    before = len(recorder.calls)
    assert action(api, "/tasks/orphan/cancel", "cancel-orphan").json()["state"] == "cancelled"
    assert action(api, "/tasks/final/cancel", "cancel-final").json()["state"] == "cancelled"
    assert len(recorder.calls) == before


def test_cancelling_a_held_owner_lets_its_collections_conclude(api):
    project = register(api)
    recorder = Recorder()
    api.app.state.task_center.services = recorder.services()
    result = enqueue(
        owner(project),
        [
            task(project, "a-1"),
            task(project, "a-2", order=1),
            collect(project, "a-final", ["a-1", "a-2"]),
        ],
    )
    key = result["owner"]["key"]
    assert action(api, f"/owners/{key}/hold", "hold").status_code == 200
    response = action(api, f"/owners/{key}/cancel", "cancel")
    assert response.status_code == 200, response.text
    store = default_client().store
    assert store.promote_ready() == ["a-final"]
    view = api.get(f"{API}/owners/{key}").json()
    # The collection records the cancelled batch; a hold would keep it queued forever.
    assert view["held"] is False and view["counts"]["queued"] == 1
    assert view["actions"]["cancel"] is False and view["actions"]["release"] is False
    # A waiting collection does not make the batch busy: Retry resumes it at once.
    assert view["actions"]["retry"] is True
    recorder.calls.clear()
    retried = action(api, f"/owners/{key}/retry", "retry")
    assert retried.status_code == 200, retried.text
    assert recorder.calls == [
        ("training.launch", "batch-a", derived_operation_id("retry", "retry", "batch-a"), True)
    ]
    # A fold stopping for the cancel is already being cancelled: nothing is left to cancel.
    running = enqueue(
        owner(project, identity="experiment-3"),
        [
            task(project, "c-1", group=("mil-batch", "batch-c")),
            collect(project, "c-final", ["c-1"], batch="batch-c"),
        ],
    )
    start("c-1")
    running_key = running["owner"]["key"]
    assert action(api, f"/owners/{running_key}/cancel", "cancel-3").status_code == 200
    assert store.get("c-1")["state"] == "stopping"
    assert api.get(f"{API}/owners/{running_key}").json()["actions"]["cancel"] is False
    # A failed cancel keeps the hold, so nothing it missed starts.
    other = enqueue(
        owner(project, identity="experiment-2"),
        [task(project, "b-1", group=("mil-batch", "batch-b"))],
    )
    other_key = other["owner"]["key"]
    action(api, f"/owners/{other_key}/hold", "hold-2")

    class Refusing:
        def __init__(self, *_):
            pass

        def cancel(self, batch, operation):
            raise StorageError("Busy.", "PROJECT_BUSY", 409)

    api.app.state.task_center.services = {"training": Refusing}
    assert action(api, f"/owners/{other_key}/cancel", "cancel-2").status_code == 409
    assert store.owner(other_key)["held"] is True


def test_retry_is_offered_only_when_the_batch_can_resume(api):
    project = register(api)
    recorder = Recorder()
    api.app.state.task_center.services = recorder.services()
    result = enqueue(
        owner(project),
        [
            task(project, "a-1"),
            task(project, "a-2", order=1),
            collect(project, "a-final", ["a-1", "a-2"]),
            task(project, "b-1", group=("mil-batch", "batch-b"), order=2),
            collect(project, "b-final", ["b-1"], batch="batch-b"),
        ],
    )
    key = result["owner"]["key"]
    start("a-1")
    finish("a-2", "failed")
    start("b-1")
    owners = {item["key"]: item for item in api.get(f"{API}/owners").json()["owners"]}
    assert owners[key]["actions"]["retry"] is False
    assert api.get(f"{API}/tasks/a-2").json()["actions"]["retry"] is False
    busy = action(api, f"/owners/{key}/retry", "retry-busy")
    assert busy.status_code == 409 and busy.json()["code"] == "TASK_RETRY_BUSY"

    # Another batch that failed and is idle can resume, so the owner offers Retry.
    finish("b-1", "failed")
    store = default_client().store
    store.promote_ready()
    finish("b-final", "succeeded")
    assert api.get(f"{API}/owners/{key}").json()["actions"]["retry"] is True
    assert api.get(f"{API}/tasks/b-1").json()["actions"]["retry"] is True
    assert api.get(f"{API}/tasks/a-2").json()["actions"]["retry"] is False

    # A collection that is writing results keeps its batch busy too.
    assert store.transition("a-1", from_states=("running",), to_state="succeeded")
    store.promote_ready()
    start("a-final", gpu=None)
    assert api.get(f"{API}/tasks/a-2").json()["actions"]["retry"] is False
    assert store.transition("a-final", from_states=("running",), to_state="succeeded")
    assert api.get(f"{API}/tasks/a-2").json()["actions"]["retry"] is True

    # Work outside a busy batch is still retried while the batch runs.
    solo = enqueue(
        owner(project, identity="experiment-2"),
        [
            task(project, "c-1", group=("mil-batch", "batch-c")),
            task(
                project,
                "refit",
                kind="compute-job",
                group=("refit", "configuration-r"),
                adapterData={"recordId": "configuration-r", "kind": "refit"},
            ),
        ],
    )
    start("c-1")
    finish("refit", "failed")
    assert api.get(f"{API}/owners/{solo['owner']['key']}").json()["actions"]["retry"] is True


def test_owner_retry_leaves_out_batches_the_experiment_no_longer_resumes(api):
    """Failed folds of a trashed or unsubmitted batch are never resumed (the training
    service refuses them, which used to fail the whole retry), and a retry that would
    resume nothing else is not offered, by the owner or by its rollup."""
    project = register(api)
    recorder = Recorder()
    api.app.state.task_center.services = recorder.services()
    science = api.app.state.projects.scientific_store(project[0])
    draft = science.create_draft(
        "experiment",
        "Trashed batch",
        {"submission": {"status": "submitted", "batchIds": ["batch-a", "batch-b"]}},
    )
    result = enqueue(
        owner(project, identity=draft["id"]),
        [
            task(project, "a-1"),
            collect(project, "a-final", ["a-1"]),
            task(project, "b-1", group=("mil-batch", "batch-b"), order=1),
            collect(project, "b-final", ["b-1"], batch="batch-b"),
            task(project, "c-1", group=("mil-batch", "batch-c"), order=2),  # never submitted
            collect(project, "c-final", ["c-1"], batch="batch-c"),
        ],
    )
    key = result["owner"]["key"]
    store = default_client().store
    for fold in ("a-1", "b-1", "c-1"):
        finish(fold, "failed")
    store.promote_ready()
    for final in ("a-final", "b-final", "c-final"):
        finish(final, "succeeded")
    science.lifecycle.apply(
        {"configuration:batch-a": "trashed"}, "trash-a", hashlib.sha256(b"trash-a").hexdigest(), 0
    )
    retried = action(api, f"/owners/{key}/retry", "retry-1")
    assert retried.status_code == 200, retried.text
    assert [call[:2] for call in recorder.calls] == [("training.launch", "batch-b")]
    assert store.get("a-1")["state"] == store.get("c-1")["state"] == "failed"

    # An owner whose only failure belongs to a trashed batch offers no retry.
    trashed = enqueue(
        owner(project, identity="experiment-2"),
        [
            task(project, "d-1", group=("mil-batch", "batch-a")),
            collect(project, "d-final", ["d-1"]),
        ],
    )
    finish("d-1", "failed")
    store.promote_ready()
    finish("d-final", "succeeded")
    other = trashed["owner"]["key"]
    assert api.get(f"{API}/owners/{other}").json()["actions"]["retry"] is False
    owners = {item["key"]: item for item in api.get(f"{API}/owners?scope=all").json()["owners"]}
    assert owners[other]["actions"]["retry"] is False
    rollup = api.get(f"{API}/rollup", params={"owner": other}).json()
    assert rollup["retryable"] is False


def test_retry_waits_for_the_final_collection_of_a_batch_that_was_not_cancelled(api):
    project = register(api)
    recorder = Recorder()
    api.app.state.task_center.services = recorder.services()
    result = enqueue(
        owner(project),
        [task(project, "a-1"), task(project, "a-2", order=1), collect(project, "a-final", [])],
    )
    key = result["owner"]["key"]
    # Both folds ended while the owner was held (or the runner was stopped): the final
    # collection has yet to record the batch, which the training service still runs.
    action(api, f"/owners/{key}/hold", "hold")
    start("a-1")
    finish("a-1", "failed")
    finish("a-2", "succeeded")
    assert api.get(f"{API}/tasks/a-1").json()["actions"]["retry"] is False
    view = api.get(f"{API}/owners/{key}").json()
    assert view["actions"]["retry"] is False and view["actions"]["cancel"] is False
    assert view["actions"]["release"] is True
    busy = action(api, f"/owners/{key}/retry", "retry-busy")
    assert busy.status_code == 409 and busy.json()["code"] == "TASK_RETRY_BUSY"
    assert recorder.calls == []
    # Once it recorded the outcome the batch resumes.
    start("a-final", gpu=None)
    assert default_client().store.transition(
        "a-final", from_states=("running",), to_state="succeeded"
    )
    assert api.get(f"{API}/tasks/a-1").json()["actions"]["retry"] is True
    assert action(api, f"/owners/{key}/retry", "retry").status_code == 200
    assert recorder.calls[-1][:2] == ("training.launch", "batch-a")


def test_a_running_collection_alone_offers_no_owner_cancel(api):
    project = register(api)
    result = enqueue(owner(project), [task(project, "a-1"), collect(project, "a-final", ["a-1"])])
    finish("a-1", "succeeded")
    store = default_client().store
    assert store.promote_ready() == ["a-final"]
    start("a-final", gpu=None)
    # An owner-wide cancel leaves collections to record the batch: it would change nothing.
    view = api.get(f"{API}/owners/{result['owner']['key']}").json()
    assert view["counts"]["running"] == 1 and view["actions"]["cancel"] is False


def conclude_lost(identity):
    """A fold the runner found lost after a reboot, with its auto-resume still undecided."""
    store = default_client().store
    start(identity)
    outcome = store.conclude(
        identity,
        to_state="interrupted",
        stop_request=None,
        exit={"reason": "lost", "lost": True},
        follow_up={"hook": "requeue_intent", "reason": "auto-resume", "since": utc_now_iso()},
    )
    assert outcome == "concluded"


def test_a_fold_awaiting_its_auto_resume_is_live_work(api):
    project = register(api)
    recorder = Recorder()
    api.app.state.task_center.services = recorder.services()
    folds = [
        task(project, f"fold-{index}", order=index, adapterData={"runId": f"run-{index}"})
        for index in range(2)
    ]
    result = enqueue(owner(project), [*folds, collect(project, "final", ["fold-0", "fold-1"])])
    key = result["owner"]["key"]
    finish("fold-1", "failed")
    conclude_lost("fold-0")
    # The runner may still resume it (its runtime probe failed after the reboot): the
    # batch is not ready for a retry, and a cancel stops the resume.
    fold = api.get(f"{API}/tasks/fold-0").json()
    assert fold["state"] == "interrupted"
    assert fold["actions"] == {"cancel": True, "retry": False}
    assert api.get(f"{API}/tasks/fold-1").json()["actions"]["retry"] is False
    view = api.get(f"{API}/owners/{key}").json()
    listed = api.get(f"{API}/owners").json()["owners"]
    assert view["position"] == 1 and [item["key"] for item in listed] == [key]
    assert view["actions"]["retry"] is False and view["actions"]["cancel"] is True
    busy = action(api, f"/owners/{key}/retry", "retry")
    assert busy.status_code == 409 and busy.json()["code"] == "TASK_RETRY_BUSY"
    cancelled = action(api, "/tasks/fold-0/cancel", "cancel-one")
    assert cancelled.status_code == 200, cancelled.text
    assert recorder.calls == [
        (
            "training.cancel_runs",
            "batch-a",
            ["run-0"],
            derived_operation_id("cancel", "cancel-one", "batch-a"),
        )
    ]
    # The owner-wide cancel reaches it through the batch as well.
    store = default_client().store
    assert store.get("fold-0")["stopRequest"] == "cancel"
    other = enqueue(
        owner(project, identity="experiment-2"),
        [task(project, "b-1", group=("mil-batch", "batch-b"))],
    )
    conclude_lost("b-1")
    # Its only work awaits the resume: it keeps its place and can be held or cancelled.
    alone = api.get(f"{API}/owners/{other['owner']['key']}").json()
    assert alone["position"] == 2 and alone["actions"]["hold"] is True
    assert alone["actions"]["cancel"] is True and alone["actions"]["retry"] is False
    assert action(api, f"/owners/{other['owner']['key']}/cancel", "cancel-b").status_code == 200
    assert recorder.calls[-1][:2] == ("training.cancel", "batch-b")
    assert store.get("b-1")["stopRequest"] == "cancel"
    assert api.get(f"{API}/owners/{other['owner']['key']}").json()["actions"]["cancel"] is False


def test_stop_is_offered_only_for_tasks_a_pause_would_stop(api):
    project = register(api)
    result = enqueue(
        owner(project),
        [
            task(project, "fold", adapterData={"runId": "run-1"}),
            collect(project, "progress", [], final=False),
            task(
                project,
                "coordinator",
                kind="predictor-coordinator",
                group=("experiment", "experiment-1"),
                request={"lane": "cpu", "cpuThreads": 1, "service": True},
            ),
        ],
    )
    key = result["owner"]["key"]
    start("progress", gpu=None)
    start("coordinator", gpu=None)
    view = api.get(f"{API}/owners/{key}").json()
    assert view["counts"]["running"] == 2 and view["actions"]["stop"] is False
    start("fold")
    assert api.get(f"{API}/owners/{key}").json()["actions"]["stop"] is True


def test_live_tasks_list_running_work_before_queued_and_blocked(api):
    project = register(api)
    # Two experiments with three batches each; every batch has an interactive final
    # collection that waits on its folds.
    for experiment in ("e1", "e2"):
        for index in range(3):
            batch = f"{experiment}-b{index}"
            folds = [
                task(project, f"{batch}-fold{fold}", group=("mil-batch", batch), order=fold)
                for fold in range(5)
            ]
            enqueue(
                owner(project, identity=experiment, title=f"Experiment {experiment}"),
                [*folds, collect(project, f"{batch}-final", [f["id"] for f in folds], batch=batch)],
            )
    for fold in range(4):
        start(f"e1-b0-fold{fold}")
    rows = api.get(f"{API}/tasks", params={"state": "live", "limit": 5}).json()["tasks"]
    assert [row["id"] for row in rows] == [
        "e1-b0-fold0",
        "e1-b0-fold1",
        "e1-b0-fold2",
        "e1-b0-fold3",
        "e1-b0-fold4",
    ]
    everything = api.get(f"{API}/tasks", params={"state": "live"}).json()["tasks"]
    assert [row["state"] for row in everything] == ["running"] * 4 + ["queued"] * 26 + [
        "blocked"
    ] * 6
    # Other filters keep the queue order, interactive work first.
    queued = api.get(f"{API}/tasks", params={"state": "queued,blocked", "limit": 1}).json()
    assert [row["id"] for row in queued["tasks"]] == ["e1-b0-final"]


def test_one_owner_can_be_read_without_listing_every_owner(api):
    project = register(api)
    finished = enqueue(
        owner(project, identity="done"), [task(project, "done-1", group=("mil-batch", "d"))]
    )
    finish("done-1", "failed")
    live = enqueue(
        owner(project, identity="live"), [task(project, "live-1", group=("mil-batch", "l"))]
    )
    everything = {item["key"]: item for item in api.get(f"{API}/owners?scope=all").json()["owners"]}
    for key in (finished["owner"]["key"], live["owner"]["key"]):
        response = api.get(f"{API}/owners/{key}")
        assert response.status_code == 200, response.text
        assert response.json() == everything[key]
    assert api.get(f"{API}/owners/{finished['owner']['key']}").json()["position"] is None
    assert api.get(f"{API}/owners/{live['owner']['key']}").json()["position"] == 1
    missing = api.get(f"{API}/owners/owner-missing")
    assert missing.status_code == 404 and missing.json()["code"] == "TASK_OWNER_NOT_FOUND"


def test_routed_calls_bind_to_the_real_application_services(api):
    """The fakes above mirror call shapes; this binds every routed call to the real methods."""
    from histopilot.taskcenter.service import _default_service

    project = register(api)
    center = api.app.state.task_center
    bound = []

    class Checked:
        def __init__(self, name, store, filesystem):
            self.name, self.real = name, _default_service(name, store, filesystem)

        def __getattr__(self, method):
            real = getattr(self.real, method)

            def call(*args, **kwargs):
                inspect.signature(real).bind(*args, **kwargs)
                bound.append((self.name, method))

            return call

    center.services = {
        name: (lambda store, filesystem, name=name: Checked(name, store, filesystem))
        for name in (
            "training",
            "predictors",
            "compute",
            "refits",
            "evaluations",
            "interpretations",
            "bulk",
        )
    }
    experiment = enqueue(
        owner(project),
        [
            task(project, "fold", labels={"batchId": "batch-a"}),
            task(project, "fold-run", order=1, adapterData={"runId": "run-1"}),
            task(
                project,
                "coordinator",
                kind="predictor-coordinator",
                group=("experiment", "experiment-1"),
                request={"lane": "cpu", "cpuThreads": 1, "service": True},
            ),
            task(
                project,
                "refit",
                kind="compute-job",
                group=("refit", "configuration-r"),
                adapterData={"recordId": "configuration-r", "kind": "refit"},
            ),
        ],
    )
    bulk = enqueue(
        owner(project, kind="evaluation-batch", identity="bulk-1", title="Bulk"),
        [task(project, "submit", kind="bulk-submit", group=("evaluation-batch", "bulk-1"))],
    )
    for kind in ("evaluation", "interpretation"):
        record = f"configuration-{kind[0]}"
        enqueue(
            owner(project, kind=f"model-{kind}", identity=record, title=kind),
            [
                task(
                    project,
                    kind,
                    kind="compute-job",
                    group=(kind, record),
                    adapterData={"recordId": record, "kind": kind},
                )
            ],
        )
    assert action(api, "/tasks/fold-run/cancel", "c-0").status_code == 200
    assert action(api, f"/owners/{experiment['owner']['key']}/cancel", "c-1").status_code == 200
    assert action(api, f"/owners/{bulk['owner']['key']}/cancel", "c-2").status_code == 200
    for identity in ("fold", "coordinator", "refit", "evaluation", "interpretation"):
        store = default_client().store
        if store.get(identity)["state"] not in ("failed", "cancelled", "interrupted"):
            finish(identity, "interrupted")
        response = action(api, f"/tasks/{identity}/retry", f"r-{identity}")
        assert response.status_code == 200, (identity, response.text)
    assert set(bound) == {
        ("training", "cancel"),
        ("training", "cancel_runs"),
        ("training", "launch"),
        ("predictors", "cancel"),
        ("predictors", "launch"),
        ("compute", "cancel"),
        ("refits", "launch"),
        ("evaluations", "launch"),
        ("interpretations", "launch"),
        ("bulk", "cancel"),
    }


def test_stop_also_pauses_a_task_the_runner_admits_during_the_hold(api, monkeypatch):
    project = register(api)
    result = enqueue(owner(project), [task(project, "a-1"), task(project, "a-2", order=1)])
    start("a-1")
    store = default_client().store
    original = type(store).request_stop

    def racing(self, task_ids, reason):
        stopping = original(self, task_ids, reason)
        if self.get("a-2")["state"] == "queued":
            start("a-2")  # admitted from a candidate list read before the hold committed
        return stopping

    monkeypatch.setattr(type(store), "request_stop", racing)
    response = action(api, f"/owners/{result['owner']['key']}/stop", "stop-race")
    assert response.status_code == 200, response.text
    assert [store.get(name)["stopRequest"] for name in ("a-1", "a-2")] == ["pause", "pause"]


def test_a_duplicate_request_in_flight_waits_for_the_first_receipt(api, monkeypatch):
    import threading

    project = register(api)
    for identity in ("a", "b", "c"):
        enqueue(
            owner(project, identity=identity),
            [task(project, f"{identity}-1", group=("mil-batch", identity))],
        )
    keys = [item["key"] for item in api.get(f"{API}/owners").json()["owners"]]
    store = default_client().store
    original = type(store).move_owner

    def slow(self, key, position):
        moved = original(self, key, position)
        time.sleep(0.3)  # a client retry arrives before the first receipt is recorded
        return moved

    monkeypatch.setattr(type(store), "move_owner", slow)
    responses = []

    def move():
        responses.append(action(api, f"/owners/{keys[2]}/move", "move-once", position="up"))

    threads = [threading.Thread(target=move) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert [response.status_code for response in responses] == [200, 200]
    order = [item["key"] for item in api.get(f"{API}/owners").json()["owners"]]
    assert order == [keys[0], keys[2], keys[1]]


def test_routed_file_errors_do_not_blame_the_store_or_skip_other_targets(api):
    project = register(api)
    calls = []

    class Predictors:
        def __init__(self, *_):
            pass

        def cancel(self, experiment, operation):
            raise PermissionError("coordinator folder is read only")

    class Training:
        def __init__(self, *_):
            pass

        def cancel(self, batch, operation):
            calls.append(("training.cancel", batch))

    api.app.state.task_center.services = {"predictors": Predictors, "training": Training}
    result = enqueue(
        owner(project),
        [
            task(project, "fold"),
            task(
                project,
                "coordinator",
                kind="predictor-coordinator",
                group=("experiment", "experiment-1"),
                request={"lane": "cpu", "cpuThreads": 1, "service": True},
            ),
        ],
    )
    response = action(api, f"/owners/{result['owner']['key']}/cancel", "cancel-1")
    assert response.status_code == 503
    assert response.json()["code"] == "TASK_ACTION_FAILED"
    assert "read only" in response.json()["detail"]
    assert calls == [("training.cancel", "batch-a")]


def test_summary_reports_an_unsafe_lease_registry_without_failing(api, registry):
    registry.mkdir(mode=0o700)
    registry.chmod(0o777)
    response = api.get(f"{API}/summary")
    assert response.status_code == 200, response.text
    summary = response.json()
    assert summary["foreignLeases"] == [] and "unsafe" in summary["leaseError"]


def test_capacity_suggestion_degrades_when_evidence_cannot_be_read(api, monkeypatch):
    from histopilot.taskcenter import estimator

    def unreadable(*_args, **_kwargs):
        raise RuntimeError("training folder vanished")

    monkeypatch.setattr(estimator, "gather_observations", unreadable)
    response = api.get(f"{API}/capacity")
    assert response.status_code == 200, response.text
    suggestion = response.json()["suggestion"]
    assert suggestion["basis"] == "hardware_only" and suggestion["parallelGpuTasks"] >= 1
    assert [item["code"] for item in suggestion["findings"]][-1] == "CAPACITY_EVIDENCE_UNAVAILABLE"


def test_tasks_filter_by_absolute_project_folder(api):
    first, second = register(api), register(api, "Other study")
    enqueue(owner(first), [task(first, "first-1")])
    enqueue(owner(second, identity="experiment-2"), [task(second, "second-1")])
    for project, expected in ((first, ["first-1"]), (second, ["second-1"])):
        listed = api.get(f"{API}/tasks", params={"project": project[1]}).json()["tasks"]
        assert [item["id"] for item in listed] == expected
        listed = api.get(f"{API}/tasks", params={"project": project[0]}).json()["tasks"]
        assert [item["id"] for item in listed] == expected


def test_eta_uses_measured_durations_of_the_same_workload(api):
    project = register(api)
    store = default_client().store
    for seconds in (80.0, 100.0, 400.0):
        store.record_measurement(
            {
                "taskId": "earlier",
                "kind": "mil-fold",
                "lane": "gpu",
                "workloadKey": "workload-a",
                "wallSeconds": seconds,
                "exitReason": "ok",
            }
        )
    store.record_measurement(
        {"taskId": "failed", "workloadKey": "workload-a", "wallSeconds": 5.0, "exitReason": "oom"}
    )
    request = {"lane": "gpu", "vramGb": 2.0, "workloadKey": "workload-a"}
    enqueue(
        owner(project), [task(project, f"fold-{i}", order=i, request=request) for i in range(4)]
    )
    eta = api.get(f"{API}/summary").json()["eta"]
    # Median of successful runs (100 s) for four queued folds over four GPU slots.
    assert eta == {"seconds": 100, "basis": "measured"}
    [listed] = api.get(f"{API}/owners").json()["owners"]
    assert listed["etaSeconds"] == 175  # the last fold starts after 300 s / 4 slots


def test_owner_links_follow_the_purpose_of_their_tasks(api):
    project = register(api)
    record = "configuration-" + "e" * 64
    result = enqueue(
        owner(project, kind="model-evaluation", identity=record, title="Inference"),
        [
            task(
                project,
                "inference",
                kind="compute-job",
                group=("evaluation", record),
                labels={"purpose": "inference"},
                adapterData={"recordId": record, "kind": "evaluation"},
            )
        ],
    )
    [listed] = api.get(f"{API}/owners").json()["owners"]
    assert listed["key"] == result["owner"]["key"]
    assert listed["link"] == f"?project={project[0]}#inference?evaluation={record}"


def test_summary_survives_an_unreadable_checkout_hash(api, monkeypatch):
    from histopilot.taskcenter import runner

    default_client().store.write_runner(code_hash="recorded", protocol=1, state="running")

    def unreadable():
        raise FileNotFoundError("runner.py")

    monkeypatch.setattr(runner, "code_hash", unreadable)
    response = api.get(f"{API}/summary")
    assert response.status_code == 200, response.text
    assert response.json()["runner"]["codeCurrent"] is True


def test_restart_waits_for_the_old_runner_session_before_starting(api, monkeypatch):
    import shutil

    from histopilot.taskcenter import launcher

    center = api.app.state.task_center
    monkeypatch.setenv("HISTOPILOT_TASK_CENTER_AUTOSTART", "1")
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    # The old runner released its lock but its tmux session lingers for two probes.
    sessions = iter([True, True, False])
    monkeypatch.setattr(center, "_runner_session_exists", lambda: next(sessions, False))
    events = []
    monkeypatch.setattr(
        launcher, "stop_runner", lambda **_: events.append("stop") or {"stopped": True}
    )

    def ensure():
        events.append(("ensure", next(sessions, "gone")))
        return {"started": True, "session": "hp-runner"}

    monkeypatch.setattr(launcher, "ensure_runner", ensure)
    restarted = action(api, "/runner/restart", "restart-1")
    assert restarted.status_code == 200, restarted.text
    assert events == ["stop", ("ensure", "gone")]
    assert restarted.json()["result"] == {
        "started": True,
        "session": "hp-runner",
        "stopped": {"stopped": True},
    }

    # A runner that was not running is started at once, without waiting.
    events.clear()
    monkeypatch.setattr(
        launcher, "stop_runner", lambda **_: events.append("stop") or {"stopped": False}
    )
    monkeypatch.setattr(
        center,
        "_runner_session_exists",
        lambda: pytest.fail("A runner that never stopped needs no wait."),
    )
    assert action(api, "/runner/restart", "restart-2").status_code == 200
    assert events == ["stop", ("ensure", "gone")]


def test_a_restart_outlasting_the_wait_starts_the_runner_once_the_old_one_exits(api, monkeypatch):
    import shutil
    import threading

    from histopilot.taskcenter import launcher, service

    center = api.app.state.task_center
    monkeypatch.setenv("HISTOPILOT_TASK_CENTER_AUTOSTART", "1")
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(service, "RUNNER_EXIT_SECONDS", 0.2)
    monkeypatch.setattr(launcher, "stop_runner", lambda **_: {"stopped": True, "exited": False})
    # The old runner is inside a long step: its session outlives the request's wait.
    finishing = threading.Event()
    monkeypatch.setattr(center, "_runner_session_exists", lambda: not finishing.is_set())
    started = []
    monkeypatch.setattr(launcher, "ensure_runner", lambda: started.append(1) or {"started": True})
    response = action(api, "/runner/restart", "restart-slow")
    assert response.status_code == 200, response.text
    assert response.json()["result"] == {
        "started": False,
        "pending": True,
        "note": "The old runner is finishing its current step; it restarts when that ends.",
        "stopped": {"stopped": True, "exited": False},
    }
    # Starting now would find the old runner alive and leave none once it exits.
    assert started == []
    # A second restart while one is pending does not start another waiter.
    thread = center._restart_thread
    action(api, "/runner/restart", "restart-again")
    assert center._restart_thread is thread
    finishing.set()
    thread.join(timeout=10)
    assert not thread.is_alive() and started == [1]
    assert action(api, "/runner/restart", "restart-slow").json()["result"]["pending"] is True
    assert started == [1]


def test_code_current_covers_every_module_the_runner_loaded(api, tmp_path):
    module = tmp_path / "training_process.py"
    module.write_text("VERSION = 1\n")
    files = [str(module), str(tmp_path / "removed.py")]
    store = default_client().store
    store.write_runner(code_files=files, code_hash=code_hash(files), protocol=1, state="running")
    assert api.get(f"{API}/summary").json()["runner"]["codeCurrent"] is True
    # A module outside the Task Center package changed under the running runner.
    module.write_text("VERSION = 22\n")
    assert api.get(f"{API}/summary").json()["runner"]["codeCurrent"] is False
    # Runners that predate the module record are compared on the package hash.
    store.write_runner(code_files=None, code_hash=code_hash())
    assert api.get(f"{API}/summary").json()["runner"]["codeCurrent"] is True


def test_owners_of_other_workspaces_are_visible_but_not_actionable(api, tmp_path):
    foreign = ("project-foreign", str(tmp_path / "elsewhere" / "study"))
    result = enqueue(owner(foreign), [task(foreign, "remote")])
    key = result["owner"]["key"]
    [listed] = api.get(f"{API}/owners").json()["owners"]
    assert listed["sameWorkspace"] is False and listed["link"] is None
    assert not any(listed["actions"].values())
    [remote] = api.get(f"{API}/tasks").json()["tasks"]
    assert remote["owner"]["sameWorkspace"] is False and remote["link"] is None
    assert remote["actions"] == {"cancel": False, "retry": False}
    for index, path in enumerate(
        (f"/owners/{key}/hold", f"/owners/{key}/cancel", "/tasks/remote/cancel")
    ):
        response = action(api, path, f"foreign-{index}")
        assert response.status_code == 409, response.text
        assert response.json()["code"] == "TASK_OWNER_OTHER_WORKSPACE"
    assert default_client().store.get("remote")["state"] == "queued"


def test_runner_start_and_restart_respect_disabled_autostart(api, monkeypatch):
    from histopilot.taskcenter import launcher

    def forbidden(**_kwargs):
        raise AssertionError("A disabled runner must not be stopped or started.")

    monkeypatch.setattr(launcher, "stop_runner", forbidden)
    started = action(api, "/runner/start", "runner-1")
    assert started.status_code == 200, started.text
    assert started.json()["result"] == {"started": False, "reason": "disabled"}
    assert started.json()["runner"]["alive"] is False
    assert started.json()["runner"]["autostart"] is False
    restarted = action(api, "/runner/restart", "runner-2").json()
    assert restarted["result"] == {"started": False, "reason": "disabled"}
    conflict = action(api, "/runner/restart", "runner-1")
    assert conflict.status_code == 409 and conflict.json()["code"] == "OPERATION_CONFLICT"

    monkeypatch.setenv("HISTOPILOT_TASK_CENTER_AUTOSTART", "1")
    calls = []
    monkeypatch.setattr(
        launcher, "ensure_runner", lambda: calls.append("ensure") or {"started": True}
    )
    enabled = action(api, "/runner/start", "runner-3").json()
    assert enabled["result"] == {"started": True} and calls == ["ensure"]
    assert enabled["runner"]["autostart"] is True
    assert action(api, "/runner/start", "runner-3").json()["result"] == {"started": True}
    assert calls == ["ensure"]


def test_an_unreadable_store_is_reported_as_unavailable(api):
    store = default_client().store
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_bytes(b"this is not a database" * 100)
    response = api.get(f"{API}/summary")
    assert response.status_code == 503
    assert response.json()["code"] == "TASK_CENTER_UNAVAILABLE"


def test_operations_inventory_reads_leases_without_pruning(api, registry):
    project_id, _folder = register(api)
    registry.mkdir(mode=0o700)
    me = process_identity()
    live = registry / f"lease-{me['pid']}.json"
    live.write_text(
        json.dumps(
            {
                "process": me,
                "processGroupId": me["pid"],
                "gpu": None,
                "cpus": 2,
                "ramGb": 1.5,
                "runsPerGpu": 1,
                "batchId": "batch-x",
                "runId": "run-x",
            }
        )
    )
    dead = registry / "lease-999999.json"
    dead.write_text(
        json.dumps(
            {
                "process": {"pid": 999_999, "startTicks": 1, "bootId": "gone"},
                "gpu": 0,
                "cpus": 1,
                "ramGb": 1,
                "runsPerGpu": 1,
            }
        )
    )
    response = api.get(f"/api/v1/projects/{project_id}/operations")
    assert response.status_code == 200, response.text
    assert response.json()["reservations"] == [
        {
            "batchId": "batch-x",
            "runId": "run-x",
            "kind": None,
            "cpus": 2,
            "ramGb": 1.5,
            "gpu": None,
            "runsPerGpu": 1,
        }
    ]
    assert dead.exists() and live.exists()


def _task_center_training():
    from histopilot.application.training import TrainingService

    return "execution_mode" in inspect.signature(TrainingService.__init__).parameters


def _launched_batch(api, tmp_path, monkeypatch):
    """A real task-center batch of five folds, launched in a project of this workspace."""
    from histopilot.application.development import DevelopmentService
    from histopilot.application.protocols import ProtocolService
    from histopilot.application.training import TrainingService
    from histopilot.schemas.development import DevelopmentBatchSpec
    from histopilot.storage.filesystem import LocalFilesystem

    support = runpy.run_path(str(Path(__file__).with_name("test_evaluations.py")))
    monkeypatch.setattr("histopilot.application.training.gpu_snapshot", lambda: {"gpus": []})
    monkeypatch.setattr("histopilot.workers.training_process.gpu_snapshot", lambda: {"gpus": []})
    project_id, _folder = register(api)
    store = api.app.state.projects.scientific_store(project_id)
    rows = [
        {
            "slideId": f"s{i:02}",
            "patientId": f"p{i:02}",
            "attributes": {"label": str(i % 2), "cohort": "development"},
        }
        for i in range(30)
    ]
    dataset, rows = support["dataset"](store, rows=rows)
    bundle, _pack, _source = support["bundle"](
        store, tmp_path, dataset, [row["slideId"] for row in rows], pack=True
    )
    filesystem = LocalFilesystem((tmp_path,))
    draft = store.create_draft(
        "experiment",
        "Protocol",
        {
            "type": "analysis-protocol",
            "spec": {
                "datasetId": dataset["id"],
                "target": support["TARGET"],
                "split": {
                    "version": 4,
                    "mode": "kfold",
                    "folds": 5,
                    "seeds": [42],
                    "pools": {"trainSelection": "remaining"},
                },
            },
        },
    )
    protocols = ProtocolService(store, filesystem)
    preview = protocols.preview(draft["id"], 1)
    assert preview["canFreeze"], preview["findings"]
    protocol = protocols.freeze(draft["id"], 1, preview["previewHash"], "protocol")
    development = DevelopmentService(store, filesystem)
    spec = DevelopmentBatchSpec.model_validate(
        {
            "experimentName": "Cancel test",
            "batchName": "Batch",
            "inputs": {"protocolId": protocol["id"], "featureBundleId": bundle["id"]},
            "mode": "single",
            "trainingSeeds": [11],
            "recipe": {"maxEpochs": 1, "bagSize": 2, "batchSize": 2},
        }
    )
    preview = development.preview(spec)
    frozen = development.freeze(spec, preview["previewHash"], "batch", {"tag": "Batch"})

    def runtime():
        return {
            "available": True,
            "python": sys.executable,
            "versions": {},
            "cudaAvailable": False,
            "gpuCount": 0,
            "findings": [],
        }

    training = TrainingService(store, filesystem, runtime=runtime, execution_mode="task-center")
    training.launch(frozen["id"], "launch-1")
    return store, filesystem, frozen, runtime


@pytest.mark.skipif(not _task_center_training(), reason="TrainingService has no Task Center mode")
def test_real_training_batch_cancel_goes_through_training_service(api, tmp_path, monkeypatch):
    """A task-center batch launched in a registered project is cancelled by owner action."""
    from histopilot.application.training import TrainingService

    store, filesystem, frozen, runtime = _launched_batch(api, tmp_path, monkeypatch)
    [owner_row] = api.get(f"{API}/owners").json()["owners"]
    assert owner_row["counts"]["queued"] >= 5
    response = action(api, f"/owners/{owner_row['key']}/cancel", "cancel-real")
    assert response.status_code == 200, response.text
    tasks = default_client().store.list(owner_key=owner_row["key"], limit=None)
    folds = [item for item in tasks if item["kind"] == "mil-fold"]
    assert folds and all(item["state"] == "cancelled" for item in folds)
    assert (store.folder / "training" / frozen["id"] / "cancel.json").exists()
    with pytest.raises(StorageError) as error:
        api.app.state.task_center.owner_action(owner_row["key"], "hold", "cancel-real")
    assert error.value.code == "OPERATION_CONFLICT"

    # The runner's final collection records the cancelled batch; an owner retry then
    # resumes it through the real TrainingService.
    from histopilot.taskcenter import capacity
    from histopilot.taskcenter.runner import Runner

    messages = []
    runner = Runner(
        default_client().store,
        host_probe=lambda: capacity.host(gpu_probe=lambda: {"gpus": []}),
        sample_interval=1.0,
        host_interval=1.0,
        log=messages.append,
    )
    runner.start(lock=False)
    deadline = time.monotonic() + 600
    while any(
        item["state"] in LIVE
        for item in default_client().store.list(owner_key=owner_row["key"], limit=None)
    ):
        assert time.monotonic() < deadline, messages[-20:]
        runner.tick()
        time.sleep(0.2)
    training = TrainingService(store, filesystem, runtime=runtime, execution_mode="task-center")
    assert training.execution(frozen["id"])["status"] == "cancelled"
    # The features of this fixture live outside the app's data roots.
    api.app.state.task_center.services = {
        "training": lambda project_store, _files: TrainingService(
            project_store, filesystem, runtime=runtime, execution_mode="task-center"
        )
    }
    retried = action(api, f"/owners/{owner_row['key']}/retry", "retry-real")
    assert retried.status_code == 200, retried.text
    tasks = default_client().store.list(owner_key=owner_row["key"], limit=None)
    folds = [item for item in tasks if item["kind"] == "mil-fold"]
    assert {(item["state"], item["attempt"]) for item in folds} == {("queued", 2)}
    [final] = [item for item in tasks if item["kind"] == "mil-collect"]
    assert final["state"] == "blocked"
    assert not (store.folder / "training" / frozen["id"] / "cancel.json").exists()
    assert training.execution(frozen["id"])["status"] == "queued"


def _single_run_cancel():
    from histopilot.application.training import TrainingService

    return hasattr(TrainingService, "cancel_runs")


@pytest.mark.skipif(not _single_run_cancel(), reason="TrainingService cannot cancel single runs")
def test_real_single_fold_cancel_is_recorded_as_cancelled_by_its_batch(api, tmp_path, monkeypatch):
    from histopilot.application.training import TrainingService
    from histopilot.workers.training_process import read_json

    store, filesystem, frozen, runtime = _launched_batch(api, tmp_path, monkeypatch)
    tasks = default_client().store.list(kinds=("mil-fold",), limit=None)
    target = tasks[-1]
    response = action(api, f"/tasks/{target['id']}/cancel", "cancel-one")
    assert response.status_code == 200, response.text
    assert response.json()["state"] == "cancelled"
    folder = store.folder / "training" / frozen["id"]
    run_id = target["adapterData"]["runId"]
    [run] = [run for run in read_json(folder / "state.json")["runs"] if run["id"] == run_id]
    assert (run["status"], run["error"]) == ("cancelled", "Cancelled before start.")
    # The batch itself continues: no batch cancel marker, the other folds stay queued.
    assert not (folder / "cancel.json").exists()
    others = [item for item in tasks if item["id"] != target["id"]]
    assert all(default_client().store.get(item["id"])["state"] == "queued" for item in others)
    training = TrainingService(store, filesystem, runtime=runtime, execution_mode="task-center")
    assert training.execution(frozen["id"])["runCounts"]["cancelled"] == 1


@pytest.mark.skipif(not _task_center_training(), reason="TrainingService has no Task Center mode")
def test_real_retry_is_offered_exactly_when_the_training_service_resumes(
    api, tmp_path, monkeypatch
):
    from histopilot.application.training import TrainingService

    store, filesystem, frozen, runtime = _launched_batch(api, tmp_path, monkeypatch)
    api.app.state.task_center.services = {
        "training": lambda project_store, _files: TrainingService(
            project_store, filesystem, runtime=runtime, execution_mode="task-center"
        )
    }
    [owner_row] = api.get(f"{API}/owners").json()["owners"]
    key = owner_row["key"]
    tasks = default_client().store
    folds = tasks.list(owner_key=key, kinds=("mil-fold",), limit=None)
    # Every fold ends while the owner is held: the final collection waits for the release.
    assert action(api, f"/owners/{key}/hold", "hold").status_code == 200
    for index, fold in enumerate(folds):
        start(fold["id"], gpu=None)
        finish(fold["id"], "failed" if index == 0 else "succeeded")
    assert tasks.promote_ready()
    assert api.get(f"{API}/tasks/{folds[0]['id']}").json()["actions"]["retry"] is False
    assert api.get(f"{API}/owners/{key}").json()["actions"]["retry"] is False
    busy = action(api, f"/owners/{key}/retry", "retry-busy")
    assert busy.status_code == 409 and busy.json()["code"] == "TASK_RETRY_BUSY"
    # The training service agrees: the batch still runs until that collection records it.
    with pytest.raises(StorageError) as refused:
        TrainingService(store, filesystem, runtime=runtime, execution_mode="task-center").launch(
            frozen["id"], "resume-now", resume=True
        )
    assert refused.value.code == "TRAINING_ACTIVE"

    # Once the collection recorded the batch, both agree it can resume.
    [final] = tasks.list(owner_key=key, kinds=("mil-collect",), limit=None)
    start(final["id"], gpu=None)
    assert tasks.transition(final["id"], from_states=("running",), to_state="succeeded")
    assert api.get(f"{API}/owners/{key}").json()["actions"]["retry"] is True
    retried = action(api, f"/owners/{key}/retry", "retry")
    assert retried.status_code == 200, retried.text
    assert tasks.get(folds[0]["id"])["state"] == "queued"


@pytest.mark.skipif(not _task_center_training(), reason="TrainingService has no Task Center mode")
def test_real_cancelled_batch_resumes_while_its_final_collection_waits(api, tmp_path, monkeypatch):
    from histopilot.application.training import TrainingService

    store, filesystem, frozen, runtime = _launched_batch(api, tmp_path, monkeypatch)
    api.app.state.task_center.services = {
        "training": lambda project_store, _files: TrainingService(
            project_store, filesystem, runtime=runtime, execution_mode="task-center"
        )
    }
    [owner_row] = api.get(f"{API}/owners").json()["owners"]
    key = owner_row["key"]
    assert action(api, f"/owners/{key}/hold", "hold").status_code == 200
    assert action(api, f"/owners/{key}/cancel", "cancel").status_code == 200
    assert (store.folder / "training" / frozen["id"] / "cancel.json").exists()
    tasks = default_client().store
    [final] = tasks.promote_ready()
    assert tasks.get(final)["kind"] == "mil-collect"
    # The cancel marker says the waiting collection only records the cancel.
    view = api.get(f"{API}/owners/{key}").json()
    assert view["held"] is False and view["actions"]["retry"] is True
    retried = action(api, f"/owners/{key}/retry", "retry")
    assert retried.status_code == 200, retried.text
    folds = tasks.list(owner_key=key, kinds=("mil-fold",), limit=None)
    assert {(item["state"], item["attempt"]) for item in folds} == {("queued", 2)}


@pytest.mark.skipif(not _single_run_cancel(), reason="TrainingService cannot cancel single runs")
def test_real_cancel_of_a_fold_awaiting_its_auto_resume(api, tmp_path, monkeypatch):
    from histopilot.application.training import TrainingService
    from histopilot.workers.training_process import read_json

    store, filesystem, frozen, runtime = _launched_batch(api, tmp_path, monkeypatch)
    api.app.state.task_center.services = {
        "training": lambda project_store, _files: TrainingService(
            project_store, filesystem, runtime=runtime, execution_mode="task-center"
        )
    }
    [target, *_others] = default_client().store.list(kinds=("mil-fold",), limit=None)
    conclude_lost(target["id"])
    view = api.get(f"{API}/tasks/{target['id']}").json()
    assert view["actions"] == {"cancel": True, "retry": False}
    retry = action(api, f"/owners/{view['owner']['key']}/retry", "retry")
    assert retry.status_code == 409 and retry.json()["code"] == "TASK_RETRY_BUSY"
    response = action(api, f"/tasks/{target['id']}/cancel", "cancel-one")
    assert response.status_code == 200, response.text
    # The runner cancels it instead of resuming it, and the batch records a cancel.
    assert default_client().store.get(target["id"])["stopRequest"] == "cancel"
    assert response.json()["actions"]["cancel"] is False
    folder = store.folder / "training" / frozen["id"]
    run_id = target["adapterData"]["runId"]
    [run] = [run for run in read_json(folder / "state.json")["runs"] if run["id"] == run_id]
    assert (run["status"], run["error"]) == ("cancelled", "Cancelled before start.")
