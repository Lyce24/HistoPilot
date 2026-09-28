"""Task Center Phase 2: managed compute jobs, the predictor coordinator and bulk submission.

Everything runs against the per-test Task Center store (tests/conftest.py) with an
in-process runner; no tmux session is ever started or queried.
"""

import os
import runpy
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from histopilot.application import compute_jobs as compute_module
from histopilot.application.compute_jobs import ComputeJobService
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import ScientificStore
from histopilot.taskcenter import TaskStore, ids, leases, procs
from histopilot.taskcenter.adapters.base import RunnerContext
from histopilot.taskcenter.adapters.compute import ComputeJobAdapter, plan_hash
from histopilot.taskcenter.adapters.coordinator import CoordinatorAdapter
from histopilot.taskcenter.client import default_client
from histopilot.taskcenter.model import TERMINAL, normalize_request, utc_now_iso
from histopilot.taskcenter.runner import Runner
from histopilot.workers import compute_job as compute_worker
from histopilot.workers.packing_process import output_lock, write_json
from histopilot.workers.training_process import process_identity, read_json

HERE = Path(__file__).parent
UID = os.getuid()


def fake_host():
    return {
        "cpuCount": 8,
        "totalRamGb": 64.0,
        "availableRamGb": 60.0,
        "bootId": "test",
        "kernel": "test",
        "physicalCpuCount": 4,
        "gpus": [],
    }


def runtime():
    return {
        "available": True,
        "python": sys.executable,
        "versions": {},
        "cudaAvailable": False,
        "gpuCount": 0,
        "host": {"cpuCount": 8, "totalRamGb": 16},
    }


class LegacyExecutor:
    """Stands in for tmux: records launches, reports sessions it started as running."""

    def __init__(self):
        self.sessions, self.calls = set(), []

    def running(self, session):
        return session in self.sessions

    def launch(self, session, python, plan, log, *, package_root):
        self.sessions.add(session)
        self.calls.append((session, python, plan, log, package_root))


class Center:
    """The per-test Task Center (the one default_client() resolves) plus runners."""

    def __init__(self):
        self.client = default_client()
        self.store: TaskStore = self.client.store
        self.runners = []
        self.logs = []

    def runner(self):
        runner = Runner(
            self.store,
            host_probe=fake_host,
            sample_interval=0.0,
            host_interval=0.0,
            log=self.logs.append,
        )
        self.runners.append(runner)
        runner.start()
        return runner

    def state(self, task_id):
        return self.store.get(task_id)["state"]

    def tick_until(self, runner, predicate, timeout=30.0):
        deadline = time.monotonic() + timeout
        while True:
            runner.tick()
            if predicate():
                return
            if time.monotonic() > deadline:
                states = {task["id"]: task["state"] for task in self.store.list(limit=None)}
                raise AssertionError(f"Condition not reached; tasks: {states}; log: {self.logs}")
            time.sleep(0.02)

    def context(self):
        return RunnerContext(
            store=self.store,
            now=utc_now_iso,
            settings=self.store.settings(),
            host=fake_host(),
            log=self.logs.append,
        )

    def cleanup(self):
        for runner in self.runners:
            runner.close()
        for task in self.store.list(limit=None):
            if task["process"]:
                procs.kill_group(task["process"])
        for runner in self.runners:
            for child in runner._procs.values():
                try:
                    child.kill()
                    child.wait(timeout=5)
                except (OSError, subprocess.SubprocessError):
                    pass


@pytest.fixture
def center():
    value = Center()
    yield value
    value.cleanup()


@pytest.fixture
def job(tmp_path, center):
    folder = tmp_path / "project"
    folder.mkdir()
    store = ScientificStore(folder, "compute-project")
    draft = store.create_draft("import", "Data", {})
    dataset = store.publish_dataset(
        draft["id"],
        expected_revision=1,
        manifest={"name": "Data"},
        artifacts={"slides.json": b'[{"slideId":"slide-1"}]'},
        operation_id="dataset",
    )
    record = store.publish_configuration(
        manifest={"kind": "model-evaluation", "name": "Test", "datasetId": dataset["id"]},
        operation_id="record",
    )
    service = ComputeJobService(
        store, runtime=runtime, execution_mode="task-center", task_center=center.client
    )
    service.legacy_executor = LegacyExecutor()  # never probe real tmux
    plan = {
        "kind": "evaluation",
        "resources": {
            "gpuIds": [],
            "cpuThreadsPerRun": 1,
            "dataLoaderWorkers": 0,
            "ramGbPerRun": 0.01,
            "maxConcurrentRuns": 1,
            "runsPerGpu": 1,
        },
        "data": {"sourceStamps": {}},
    }
    return service, record["id"], plan


def worker(folder, task_id, **env):
    return subprocess.run(
        [sys.executable, "-m", "histopilot.workers.compute_job", str(folder / "plan.json")],
        cwd=folder / "compute",
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "HISTOPILOT_TASK_MANAGED": "1",
            "HISTOPILOT_TASK_ID": task_id,
            **env,
        },
        capture_output=True,
        text=True,
        timeout=120,
    )


# -- ComputeJobService with the Task Center executor ------------------------------------------


def test_managed_launch_queues_one_task_and_status_reports_its_queue(job, center):
    service, identity, plan = job
    state = service.launch(identity, plan, "launch")
    folder = service.folder(identity)
    task_id = ids.compute_task_id(str(folder))
    assert (state["executor"], state["taskId"], state["taskAttempt"]) == ("task-center", task_id, 1)
    assert state["status"] == "queued" and state["sessionName"].startswith("tc-evaluation-")
    task = center.store.get(task_id)
    assert (task["state"], task["kind"], task["adapter"]) == (
        "queued",
        "compute-job",
        "compute-job",
    )
    assert task["sessionName"] == state["sessionName"]
    assert task["group"] == {"kind": "evaluation", "id": identity}
    assert task["command"]["argv"][1:] == [
        "-u",
        "-m",
        "histopilot.workers.compute_job",
        str(folder / "plan.json"),
    ]
    assert task["command"]["cwd"] == str(folder / "compute")
    assert (folder / "compute" / "histopilot" / "workers" / "compute_job.py").is_file()
    assert task["command"]["env"]["HISTOPILOT_TASK_MANAGED"] == "1"
    assert task["command"]["log"] == str(folder / "worker.log")
    assert task["adapterData"] == {
        "computeFolder": str(folder),
        "recordId": identity,
        "kind": "evaluation",
    }
    request = task["request"]
    assert (request["lane"], request["cpuThreads"], request["dataWorkers"]) == ("cpu", 1, 0)
    assert (request["ramGb"], request["vramGb"]) == (6.0, 0.0)
    owner = center.store.owner(task["ownerKey"])
    assert (owner["kind"], owner["id"], owner["title"]) == ("model-evaluation", identity, "Test")
    assert owner["projectFolder"] == str(service.store.folder)
    assert task["title"] == "Evaluation · Test"

    status = service.status(identity)
    assert (status["status"], status["executor"]) == ("queued", "task-center")
    assert status["task"]["state"] == "queued" and status["waitingReason"] is None
    center.store.set_waiting_reasons({task_id: "Waiting for RAM"})
    center.store.hold_owner(task["ownerKey"], True)
    status = service.status(identity)
    assert status["waitingReason"] == "Waiting for RAM" and status["task"]["held"] is True

    assert service.launch(identity, plan, "launch")["status"] == "queued"
    assert [row["attempt"] for row in center.store.list(limit=None)] == [1]
    assert service.legacy_executor.calls == []


def test_owner_and_title_are_propagated_to_the_task(job, center):
    service, identity, plan = job
    state = service.launch(
        identity,
        plan,
        "launch",
        task_owner={"kind": "experiment", "id": "experiment-1", "title": "Study"},
        task_title="Refit · custom",
    )
    task = center.store.get(state["taskId"])
    owner = center.store.owner(task["ownerKey"])
    assert (owner["kind"], owner["id"], owner["title"]) == ("experiment", "experiment-1", "Study")
    assert owner["projectId"] == service.store.project_id
    assert task["title"] == "Refit · custom"
    assert task["labels"]["experimentId"] == "experiment-1"


def queued_training(center, tmp_path, count=3):
    """Folds an earlier experiment queued; they run in submission order."""
    center.client.enqueue(
        {
            "kind": "experiment",
            "id": "earlier",
            "projectId": "project-earlier",
            "projectFolder": str(tmp_path / "earlier"),
            "title": "Earlier experiment",
        },
        [
            {
                "id": f"fold-{index}",
                "kind": "mil-fold",
                "adapter": "generic",
                "title": f"Fold {index}",
                "group": {"kind": "mil-batch", "id": "batch"},
                "planOrder": index,
                "request": {"lane": "cpu", "cpuThreads": 1, "ramGb": 0.1},
                "command": {
                    "argv": [sys.executable, "-c", "pass"],
                    "cwd": str(tmp_path),
                    "log": str(tmp_path / f"fold-{index}.log"),
                },
            }
            for index in range(count)
        ],
    )


def test_a_single_evaluation_goes_ahead_of_queued_training(job, center, tmp_path):
    service, identity, plan = job
    queued_training(center, tmp_path)
    state = service.launch(identity, plan, "launch")
    task_id = state["taskId"]
    assert center.store.get(task_id)["priority"] == "interactive"
    queue = [row["id"] for row in center.store.list(states=("queued",), limit=None)]
    assert queue == [task_id, "fold-0", "fold-1", "fold-2"]
    # A resume keeps it ahead, also for a task first queued at normal priority.
    center.store.set_fields(task_id, priority="normal")
    center.store.transition(task_id, from_states="queued", to_state="interrupted")
    service.launch(identity, plan, "resume", resume=True)
    assert center.store.get(task_id)["priority"] == "interactive"


def test_refits_and_bulk_members_keep_their_turn(job, center, tmp_path):
    service, identity, plan = job
    queued_training(center, tmp_path)
    owner = {"kind": "evaluation-batch", "id": "bulk-1", "title": "Bulk"}
    state = service.launch(identity, plan, "launch", task_owner=owner)
    assert center.store.get(state["taskId"])["priority"] == "normal"
    queue = [row["id"] for row in center.store.list(states=("queued",), limit=None)]
    assert queue[-1] == state["taskId"]
    priority = compute_module.compute_priority
    assert priority({"kind": "interpretation"}, {"kind": "model-interpretation"}) == "interactive"
    assert priority({"kind": "refit"}, {"kind": "predictor-refit"}) == "normal"
    assert priority({"kind": "refit"}, {"kind": "experiment"}) == "normal"
    # A resume (Task Center "Retry failed", or Resume on the evaluation page) passes no
    # owner; the member stays in its bulk batch and keeps its turn.
    center.store.transition(state["taskId"], from_states="queued", to_state="interrupted")
    service.launch(identity, plan, "resume", resume=True)
    task = center.store.get(state["taskId"])
    assert task["state"] == "queued" and task["priority"] == "normal"
    assert center.store.owner(task["ownerKey"])["kind"] == "evaluation-batch"


def hold_runner_lock(center):
    import fcntl

    center.store.path.parent.mkdir(parents=True, exist_ok=True)
    stream = open(center.store.path.parent / "runner.lock", "a")  # noqa: SIM115
    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return stream


def test_compute_submissions_wake_a_stopped_runner_and_report_it(job, center, monkeypatch):
    from histopilot.taskcenter import launcher

    service, identity, plan = job
    calls = []
    monkeypatch.setattr(launcher, "ensure_runner", lambda: calls.append(1) or {"started": True})
    # An injected client (tests, tools) never starts a runner.
    state = service.launch(identity, plan, "launch")
    assert calls == []
    task_id = state["taskId"]
    served = ComputeJobService(service.store, runtime=runtime, execution_mode="task-center")
    served.legacy_executor = LegacyExecutor()
    center.store.transition(task_id, from_states="queued", to_state="interrupted")
    served.launch(identity, plan, "resume", resume=True)
    assert calls == [1]
    status = served.status(identity)
    assert status["status"] == "queued" and status["task"]["runnerAlive"] is False
    lock = hold_runner_lock(center)
    try:
        assert served.status(identity)["task"]["runnerAlive"] is True
    finally:
        lock.close()
    # Code running inside a task (a pinned coordinator's refits) never starts one.
    monkeypatch.setenv("HISTOPILOT_TASK_ID", "task-coordinator")
    center.store.transition(task_id, from_states="queued", to_state="interrupted")
    served.launch(identity, plan, "resume-2", resume=True)
    assert calls == [1]
    monkeypatch.delenv("HISTOPILOT_TASK_ID")

    def broken():
        raise RuntimeError("tmux exploded")

    # A runner that cannot start never fails the accepted launch.
    monkeypatch.setattr(launcher, "ensure_runner", broken)
    center.store.transition(task_id, from_states="queued", to_state="interrupted")
    assert served.launch(identity, plan, "resume-3", resume=True)["status"] == "queued"


def test_cancel_while_queued_is_cancelled_and_never_starts(job, center):
    service, identity, plan = job
    state = service.launch(identity, plan, "launch")
    result = service.cancel(identity, "cancel")
    assert result["status"] == "cancelled" and result["cancellationRequested"]
    assert center.state(state["taskId"]) == "cancelled"
    folder = service.folder(identity)
    assert read_json(folder / "state.json")["status"] == "queued"  # derived, never saved
    assert service.cancel(identity, "cancel")["status"] == "cancelled"
    runner = center.runner()
    runner.tick()
    assert center.state(state["taskId"]) == "cancelled"
    assert not (folder / "worker.log").exists()


def test_stopped_task_reads_interrupted_and_resume_requeues_the_same_task(job, center):
    service, identity, plan = job
    state = service.launch(identity, plan, "launch")
    task_id = state["taskId"]
    center.store.transition(task_id, from_states="queued", to_state="interrupted")
    assert service.status(identity)["status"] == "interrupted"
    resumed = service.launch(identity, plan, "resume", resume=True)
    assert (resumed["status"], resumed["attempt"], resumed["taskAttempt"]) == ("queued", 2, 2)
    task = center.store.get(task_id)
    assert (task["state"], task["attempt"]) == ("queued", 2)
    assert center.store.events(task_id)[-1]["detail"]["reason"] == "launch"
    assert service.status(identity)["status"] == "queued"


def test_cancel_reaches_a_paused_job_whose_requeue_is_pending(job, center):
    service, identity, plan = job
    task_id = service.launch(identity, plan, "launch")["taskId"]
    # The runner concluded a paused attempt and has not requeued it yet.
    center.store.transition(
        task_id,
        from_states="queued",
        to_state="interrupted",
        bookkeeping={"hook": "on_requeue", "reason": "pause"},
    )
    assert service.status(identity)["status"] == "interrupted"
    service.cancel(identity, "cancel")
    assert (service.folder(identity) / "cancel.requested").exists()
    assert center.store.get(task_id)["stopRequest"] == "cancel"
    assert not center.store.requeue([task_id], reason="pause", unless_cancelled=True)
    # Without a pending requeue an interrupted job is left alone.
    other = service.folder(identity) / "cancel.requested"
    other.unlink()
    center.store.set_fields(task_id, bookkeeping=None)
    service.cancel(identity, "cancel-again")
    assert not other.exists()


def test_unavailable_task_center_never_reads_as_stopped(job, center, monkeypatch):
    service, identity, plan = job
    service.launch(identity, plan, "launch")

    def unavailable(_task_id):
        from histopilot.storage.project_lock import StorageError

        raise StorageError("down", "TASK_CENTER_UNAVAILABLE", 503)

    monkeypatch.setattr(center.client, "task", unavailable)
    status = service.status(identity)
    assert status["status"] == "queued" and status["task"]["unknown"] is True
    assert "unavailable" in status["waitingReason"]


def test_resume_waits_for_a_stopping_attempt_and_requeues_a_normalized_task(job, center):
    service, identity, plan = job
    task_id = service.launch(identity, plan, "launch")["taskId"]
    folder = service.folder(identity)
    # The worker recorded its outcome and exited; the runner has not concluded the task.
    center.store.transition(task_id, from_states="queued", to_state="running")
    write_json(folder / "state.json", {**read_json(folder / "state.json"), "status": "interrupted"})
    before = (folder / "state.json").read_bytes()
    assert service.status(identity)["status"] == "interrupted"
    with pytest.raises(StorageError) as caught:
        service.launch(identity, plan, "resume", resume=True)
    assert caught.value.code == "COMPUTE_ACTIVE"
    # The finishing attempt's exit bookkeeping still reads that attempt's own state.
    assert (folder / "state.json").read_bytes() == before

    center.store.transition(task_id, from_states="running", to_state="interrupted")
    resumed = service.launch(identity, plan, "resume", resume=True)
    assert (resumed["status"], resumed["taskAttempt"]) == ("queued", 2)
    request = center.store.get(task_id)["request"]
    assert request == normalize_request(request)  # set_fields normalizes a requeued request
    assert {"sharedFiles", "exclusiveGpu", "service", "graceSeconds"} <= request.keys()


OOM_ONCE = r"""
import json, os, pathlib, sys
folder = pathlib.Path(sys.argv[1])
state = json.loads((folder / "state.json").read_text())
state["process"] = {"pid": os.getpid()}
if not (folder / "oom-once").exists():
    (folder / "oom-once").write_text("1")
    # What a pinned worker records for a CUDA out-of-memory error: a failed job, exit 0.
    state.update(status="failed", error="RuntimeError: CUDA out of memory. Tried to allocate")
else:
    state.update(status="completed", error=None, result={"epochsCompleted": 3})
(folder / "state.json").write_text(json.dumps(state))
"""


def test_a_compute_job_that_ran_out_of_memory_is_requeued_once(job, center):
    service, identity, plan = job
    task_id = service.launch(identity, plan, "launch")["taskId"]
    folder = service.folder(identity)
    task = center.store.get(task_id)
    command = {**task["command"], "argv": [sys.executable, "-c", OOM_ONCE, str(folder)]}
    center.store.set_fields(task_id, command=command, request={**task["request"], "vramGb": 2.0})
    runner = center.runner()
    center.tick_until(runner, lambda: center.state(task_id) == "succeeded")
    task = center.store.get(task_id)
    assert task["attempt"] == 2 and task["adapterData"]["oomRetries"] == 1
    assert task["request"]["vramGb"] == 3.0
    reasons = [(event["detail"] or {}).get("reason") for event in center.store.events(task_id)]
    assert "oom-backoff" in reasons
    saved = read_json(folder / "state.json")
    assert (saved["status"], saved["attempt"], saved["taskAttempt"]) == ("completed", 2, 2)


def test_cancel_signals_a_live_worker_whose_task_already_finished(job, center):
    service, identity, plan = job
    task_id = service.launch(identity, plan, "launch")["taskId"]
    folder = service.folder(identity)
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    try:
        saved = read_json(folder / "state.json")
        write_json(
            folder / "state.json",
            {
                **saved,
                "status": "running",
                "process": process_identity(child.pid),
                "processGroupId": child.pid,
            },
        )
        # The runner lost track of the task, but its worker is still alive.
        center.store.transition(task_id, from_states="queued", to_state="interrupted")
        assert service.status(identity)["status"] == "running"
        service.cancel(identity, "cancel")
        assert child.wait(timeout=10) == -signal.SIGTERM
        assert (folder / "cancel.requested").exists()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_legacy_records_and_archives_keep_the_tmux_worker(job, center, monkeypatch):
    service, identity, plan = job
    # A record first launched before the Task Center ...
    legacy = LegacyExecutor()
    ComputeJobService(service.store, executor=legacy, runtime=runtime).launch(
        identity, plan, "legacy-launch"
    )
    state = read_json(service.folder(identity) / "state.json")
    assert "executor" not in state and state["sessionName"].startswith("hp-evaluation-")
    service.legacy_executor = legacy
    assert service.status(identity)["executor"] == "tmux"
    assert service.status(identity)["status"] == "queued"
    legacy.sessions.clear()
    assert service.status(identity)["status"] == "interrupted"
    # ... resumes from its pinned archive, which predates the Task Center protocol.
    monkeypatch.setattr(compute_module, "archive_protocol", lambda _path: None)
    resumed = service.launch(identity, plan, "legacy-resume", resume=True)
    assert "executor" not in resumed and "taskId" not in resumed
    assert len(legacy.calls) == 2 and center.store.list(limit=None) == []
    assert service.status(identity)["status"] == "queued"


def test_managed_worker_fences_foreign_records_and_reports_busy_output(job, center):
    service, identity, plan = job
    state = service.launch(identity, plan, "launch")
    folder = service.folder(identity)
    before = (folder / "state.json").read_bytes()
    foreign = worker(folder, "task-of-another-launch")
    assert foreign.returncode == 0, foreign.stderr
    assert "not assigned" in foreign.stdout
    assert (folder / "state.json").read_bytes() == before

    with output_lock(folder):
        busy = worker(folder, state["taskId"])
    # The busy contract: EX_TEMPFAIL, record untouched.
    assert busy.returncode == 75 and "OUTPUT_BUSY" in busy.stderr
    assert (folder / "state.json").read_bytes() == before

    # The adapter re-queues a busy exit instead of failing the record ...
    task = center.store.get(state["taskId"])
    (folder / "worker.log").write_text(busy.stderr)
    task = {**task, "process": {"pid": 2**22 + 1, "startTicks": 1, "bootId": "x"}}
    exit = {
        "returncode": 75,
        "lost": False,
        "signalled": False,
        "killed": False,
        "stopReason": None,
    }
    result = ComputeJobAdapter().on_exit(task, exit, center.context())
    assert (result["state"], result["exitReason"]) == ("requeue", "busy")
    # ... and so does a worker pinned before it, which exits 1 with the code in its log.
    exit = {**exit, "returncode": 1}
    result = ComputeJobAdapter().on_exit(task, exit, center.context())
    assert (result["state"], result["exitReason"]) == ("requeue", "busy")
    # worker.log is appended across attempts: an older busy line never re-queues a crash.
    with (folder / "worker.log").open("a") as log:
        log.write("Traceback (most recent call last):\nModuleNotFoundError: No module named 'x'\n")
    result = ComputeJobAdapter().on_exit(task, exit, center.context())
    assert (result["state"], result["exitReason"]) == ("interrupted", "interrupted")

    # The fence never re-runs a record its own task already completed ...
    completed = {**read_json(folder / "state.json"), "status": "completed"}
    write_json(folder / "state.json", completed)
    finished = (folder / "state.json").read_bytes()
    again = worker(folder, state["taskId"])
    assert again.returncode == 0 and "not assigned" in again.stdout
    assert (folder / "state.json").read_bytes() == finished
    # ... and a queued record whose cancellation was requested ends cancelled, unstarted.
    write_json(folder / "state.json", {**completed, "status": "queued"})
    write_json(folder / "cancel.requested", {"at": "now"})
    stopped = worker(folder, state["taskId"])
    assert stopped.returncode == 0, stopped.stderr
    saved = read_json(folder / "state.json")
    assert (saved["status"], saved["result"]) == ("cancelled", None)
    assert "gpu" not in saved  # it stopped before choosing a device


def test_managed_worker_waits_out_a_busy_workspace(tmp_path, monkeypatch):
    calls = []

    def verify(_plan):
        calls.append(1)
        if len(calls) < 3:
            raise StorageError("Another operation is changing this workspace.", "PROJECT_BUSY")
        return "verified"

    monkeypatch.setattr(compute_worker, "verify_plan_inputs", verify)
    monkeypatch.setattr(compute_worker.time, "sleep", lambda _seconds: None)
    assert compute_worker._verify_inputs({}, tmp_path, managed=True) == "verified"
    assert len(calls) == 3
    # A workspace busy for longer than the short wait yields the slot (exit 75).
    calls.clear()
    with pytest.raises(compute_worker.Busy):
        compute_worker._verify_inputs({}, tmp_path, managed=True, wait=0)
    # Unmanaged workers keep failing fast, and a cancel ends the wait.
    calls.clear()
    with pytest.raises(StorageError):
        compute_worker._verify_inputs({}, tmp_path, managed=False)
    calls.clear()
    write_json(tmp_path / "cancel.requested", {"at": "now"})
    with pytest.raises(KeyboardInterrupt):
        compute_worker._verify_inputs({}, tmp_path, managed=True)
    # Other failures are never retried.
    monkeypatch.setattr(
        compute_worker, "verify_plan_inputs", lambda _plan: (_ for _ in ()).throw(ValueError("x"))
    )
    with pytest.raises(ValueError):
        compute_worker._verify_inputs({}, tmp_path, managed=True)


# -- Adapters ---------------------------------------------------------------------------------


def compute_task(folder, task_id="task-compute", attempt=1):
    return {
        "id": task_id,
        "attempt": attempt,
        "process": {"pid": 4321, "startTicks": 1, "bootId": "x"},
        "command": {"log": str(folder / "worker.log")},
        "adapterData": {"computeFolder": str(folder), "recordId": "record", "kind": "refit"},
    }


def saved(folder, task_id="task-compute", **state):
    plan = {"kind": "refit", "recordId": "record"}
    write_json(folder / "plan.json", plan)
    write_json(
        folder / "state.json",
        {
            "recordId": "record",
            "planHash": plan_hash(plan),
            "taskId": task_id,
            "attempt": 1,
            "process": {"pid": 4321, "startTicks": 1, "bootId": "x"},
            **state,
        },
    )


def outcome_of(adapter, task, ctx, **exit):
    exit = {
        "returncode": 0,
        "lost": False,
        "signalled": False,
        "killed": False,
        "stopReason": None,
        **exit,
    }
    result = adapter.on_exit(task, exit, ctx)
    return result["state"], result["exitReason"]


def test_compute_adapter_classifies_saved_outcomes_not_exit_codes(tmp_path, center):
    adapter, ctx, folder = ComputeJobAdapter(), center.context(), tmp_path / "job"
    folder.mkdir()
    task = compute_task(folder)
    saved(folder, status="completed", result={"epochsCompleted": 7, "runId": "record"})
    result = adapter.on_exit(task, {"returncode": 0, "stopReason": None}, ctx)
    assert (result["state"], result["measurement"]) == ("succeeded", {"epochs": 7})
    saved(folder, status="failed", error="RuntimeError: CUDA out of memory. Tried to allocate")
    assert outcome_of(adapter, task, ctx) == ("failed", "oom")
    saved(folder, status="failed", error="ValueError: bad input")
    assert outcome_of(adapter, task, ctx) == ("failed", "error")
    saved(folder, status="cancelled", error="Compute cancellation requested.")
    assert outcome_of(adapter, task, ctx, stopReason="cancel") == ("cancelled", "cancelled")
    saved(folder, status="interrupted")
    assert outcome_of(adapter, task, ctx, stopReason="pause") == ("requeue", "paused")
    assert outcome_of(adapter, task, ctx) == ("interrupted", "interrupted")
    saved(folder, status="running")  # killed before recording an outcome
    assert outcome_of(adapter, task, ctx, returncode=-9) == ("interrupted", "interrupted")
    assert outcome_of(adapter, task, ctx, returncode=None, lost=True) == ("interrupted", "lost")
    assert outcome_of(adapter, task, ctx, returncode=-9, stopReason="pause") == (
        "requeue",
        "paused",
    )
    write_json(folder / "cancel.requested", {"at": "now"})
    assert outcome_of(adapter, task, ctx, returncode=-9) == ("cancelled", "cancelled")
    (folder / "cancel.requested").unlink()
    saved(folder, task_id="task-other", status="completed")
    assert outcome_of(adapter, task, ctx) == ("succeeded", "already-complete")
    assert adapter.prepare(task, ctx)["skip"]["state"] == "cancelled"
    saved(folder, task_id="task-other", status="queued")
    assert outcome_of(adapter, task, ctx) == ("cancelled", "cancelled")
    saved(folder, status="completed")
    assert adapter.prepare(task, ctx)["skip"]["exitReason"] == "already-complete"
    saved(folder, status="queued")
    assert adapter.prepare(task, ctx) is None
    # progress.json outlives attempts: a snapshot older than this attempt is not its progress.
    write_json(folder / "progress.json", {"epoch": 3, "maxEpochs": 9})
    started = {
        **task,
        "startedAt": utc_now_iso(),
        "command": {**task["command"], "progress": str(folder / "progress.json")},
    }
    earlier = time.time() - 3600
    os.utime(folder / "progress.json", (earlier, earlier))
    assert adapter.progress(started, ctx) is None
    # File times come from the coarse kernel clock and can trail utc_now_iso() slightly.
    later = time.time() + 5
    os.utime(folder / "progress.json", (later, later))
    assert adapter.progress(started, ctx)["epoch"] == 3


def test_compute_adapter_requeues_only_its_own_resumable_record(tmp_path, center):
    adapter, ctx, folder = ComputeJobAdapter(), center.context(), tmp_path / "job"
    folder.mkdir()
    task = compute_task(folder, attempt=3)
    saved(folder, status="running", waitingReason="stale", error="stale")
    assert adapter.can_requeue(task, ctx)
    adapter.on_requeue(task, ctx)
    state = read_json(folder / "state.json")
    assert (state["status"], state["attempt"], state["taskAttempt"]) == ("queued", 2, 4)
    assert state["autoResumedAt"] and state["error"] is None and "waitingReason" not in state
    write_json(folder / "cancel.requested", {"at": "now"})
    assert not adapter.can_requeue(task, ctx)
    (folder / "cancel.requested").unlink()
    for status in ("completed", "failed", "cancelled"):
        saved(folder, status=status)
        assert not adapter.can_requeue(task, ctx)
    saved(folder, status="interrupted")
    write_json(folder / "plan.json", {"kind": "refit", "changed": True})
    assert not adapter.can_requeue(task, ctx)
    saved(folder, task_id="task-other", status="interrupted")
    assert not adapter.can_requeue(task, ctx)
    from histopilot.taskcenter.adapters.base import AdapterError

    with pytest.raises(AdapterError) as caught:
        adapter.on_requeue(task, ctx)
    assert caught.value.fatal


def test_coordinator_adapter_reads_the_coordinator_state(tmp_path, center):
    adapter, ctx, folder = CoordinatorAdapter(), center.context(), tmp_path / "coordinator"
    folder.mkdir()
    task = {"id": "task-coordinator", "adapterData": {"coordinatorFolder": str(folder)}}

    def with_status(status, **extra):
        write_json(folder / "state.json", {"status": status, "taskId": task["id"], **extra})

    with_status("completed")
    assert outcome_of(adapter, task, ctx) == ("succeeded", "ok")
    assert adapter.prepare(task, ctx)["skip"]["state"] == "succeeded"
    with_status("cancelled")
    assert outcome_of(adapter, task, ctx) == ("cancelled", "cancelled")
    with_status("attention", error={"code": "X", "message": "Refit stopped."})
    result = adapter.on_exit(task, {"returncode": 0, "stopReason": None}, ctx)
    assert (result["state"], result["error"]) == ("failed", "Refit stopped.")
    assert not adapter.can_requeue(task, ctx)
    with_status("running")
    assert outcome_of(adapter, task, ctx, returncode=-15, stopReason="pause") == (
        "requeue",
        "paused",
    )
    assert outcome_of(adapter, task, ctx, returncode=None, lost=True) == ("interrupted", "lost")
    assert adapter.can_requeue(task, ctx)
    write_json(folder / "cancel.requested", {"at": "now"})
    assert outcome_of(adapter, task, ctx, returncode=-15) == ("cancelled", "cancelled")
    assert not adapter.can_requeue(task, ctx)


def test_a_coordinator_that_found_the_project_busy_runs_again(tmp_path, center):
    adapter, ctx, folder = CoordinatorAdapter(), center.context(), tmp_path / "coordinator"
    folder.mkdir()
    log = folder / "worker.log"
    task = {
        "id": "task-coordinator",
        "adapterData": {"coordinatorFolder": str(folder)},
        "command": {"log": str(log)},
    }
    write_json(folder / "state.json", {"status": "queued", "taskId": task["id"]})
    busy = (
        "histopilot.storage.project_lock.StorageError: Another operation is changing this "
        "workspace. Retry after it finishes.\n"
    )
    log.write_text("Traceback (most recent call last):\n" + busy)
    # A coordinator pinned before the busy contract: exit 1, the lock error last in its log.
    assert outcome_of(adapter, task, ctx, returncode=1) == ("requeue", "busy")
    log.write_text(
        "Traceback (most recent call last):\nhistopilot.storage.project_lock.StorageError: "
        "Another operation is writing this project. Retry after it finishes.\n"
    )
    assert outcome_of(adapter, task, ctx, returncode=1) == ("requeue", "busy")
    # Only the latest attempt counts: an older busy exit never turns a crash into a retry.
    log.write_text(busy + "Traceback (most recent call last):\nValueError: broken plan\n")
    assert outcome_of(adapter, task, ctx, returncode=1) == ("interrupted", "interrupted")
    # The busy contract: exit 75, whatever the log says.
    assert outcome_of(adapter, task, ctx, returncode=75) == ("requeue", "busy")


def test_a_busy_attention_is_requeued_and_reopened_not_failed(tmp_path, center):
    """gej3/gej4: the coordinator recorded ``attention`` with PROJECT_BUSY and exited 0."""
    adapter, ctx, folder = CoordinatorAdapter(), center.context(), tmp_path / "coordinator"
    folder.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    task = {
        "id": "task-coordinator",
        "projectFolder": str(project),
        "adapterData": {"coordinatorFolder": str(folder)},
        "command": {"log": str(folder / "worker.log")},
    }
    busy = {"code": "PROJECT_BUSY", "message": "Another operation is changing this workspace."}
    done = {"status": "completed", "error": None}
    waiting = {"status": "waiting", "error": None}

    def attention(error, *items):
        write_json(
            folder / "state.json",
            {"status": "attention", "taskId": task["id"], "error": error, "items": list(items)},
        )

    attention(busy, done, waiting)
    assert outcome_of(adapter, task, ctx, returncode=0) == ("requeue", "busy")
    assert outcome_of(adapter, task, ctx, returncode=None, lost=True) == ("requeue", "busy")
    assert adapter.can_requeue(task, ctx)
    adapter.on_requeue(task, ctx)
    state = read_json(folder / "state.json")
    assert (state["status"], state["error"]) == ("queued", None)
    assert [item["status"] for item in state["items"]] == ["completed", "waiting"]

    # An item that failed only on a busy lock is retried, finished items are kept.
    attention(busy, done, {"status": "failed", "error": busy})
    assert outcome_of(adapter, task, ctx, returncode=0) == ("requeue", "busy")
    adapter.on_requeue(task, ctx)
    state = read_json(folder / "state.json")
    assert state["status"] == "queued"
    assert [item["status"] for item in state["items"]] == ["completed", "waiting"]

    # A real failure still needs attention, even next to a busy item.
    real = {"code": "EXPERIMENT_REFIT_STOPPED", "message": "Refit stopped."}
    attention(real, {"status": "failed", "error": real}, {"status": "failed", "error": busy})
    result = adapter.on_exit(task, {"returncode": 0, "stopReason": None}, ctx)
    assert (result["state"], result["error"]) == ("failed", "Refit stopped.")
    # A busy error first retries the busy item; the real failure then surfaces.
    attention(busy, {"status": "failed", "error": busy}, {"status": "failed", "error": real})
    assert outcome_of(adapter, task, ctx, returncode=0) == ("requeue", "busy")
    adapter.on_requeue(task, ctx)
    state = read_json(folder / "state.json")
    assert state["status"] == "attention"
    assert [item["status"] for item in state["items"]] == ["waiting", "failed"]

    # A cancel wins over the requeue.
    attention(busy, waiting)
    assert outcome_of(adapter, task, ctx, returncode=0, stopReason="cancel") == (
        "cancelled",
        "cancelled",
    )
    write_json(folder / "cancel.requested", {"at": "now"})
    assert outcome_of(adapter, task, ctx, returncode=0) == ("cancelled", "cancelled")
    assert not adapter.can_requeue(task, ctx)


BUSY_AFTER_RELEASE = r"""
import pathlib, sys, time
release = pathlib.Path(sys.argv[1])
while not release.exists():
    time.sleep(0.02)
sys.exit(75)
"""


def test_a_coordinator_that_exited_busy_while_no_runner_ran_is_requeued(coordinator, center):
    """Its wrapper's exit record keeps the busy exit (75) across the runner restart."""
    service, identity, _tmux, tmp_path = coordinator
    service.launch(identity, "start")
    folder = service.folder(identity)
    task_id = ids.coordinator_task_id(str(folder))
    release = tmp_path / "release-coordinator"
    task = center.store.get(task_id)
    command = {**task["command"], "argv": [sys.executable, "-c", BUSY_AFTER_RELEASE, str(release)]}
    center.store.set_fields(task_id, command=command)
    first = center.runner()
    center.tick_until(first, lambda: center.state(task_id) == "running")
    first.close()  # a deploy restarts the runner; the coordinator keeps running
    process = center.store.get(task_id)["process"]
    release.write_text("go")
    while procs.leader_alive(process) or procs.leader_alive(process["wrapper"]):
        time.sleep(0.02)
    center.runner()
    task = center.store.get(task_id)
    assert (task["state"], task["attempt"]) == ("queued", 2)
    assert task["exit"] is None  # cleared by the requeue; the event keeps the reason
    assert any(
        (event["detail"] or {}).get("exitReason") == "busy"
        for event in center.store.events(task_id)
    )


BUSY_THEN_DONE = r"""
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
state = json.loads(path.read_text())
marker = path.with_name("busy-once")
if not marker.exists():
    marker.write_text("1")
    busy = {"code": "PROJECT_BUSY", "message": "Another operation is changing this workspace."}
    state["items"][0].update(status="failed", error=busy)
    state.update(status="attention", error=busy)
else:
    assert state["status"] == "queued", state["status"]
    assert not [item for item in state["items"] if item["status"] == "failed"], state["items"]
    state["status"] = "completed"
path.write_text(json.dumps(state))
"""


def test_a_coordinator_that_exits_busy_is_requeued_and_resumes(coordinator, center, monkeypatch):
    from histopilot.taskcenter import runner as runner_module

    monkeypatch.setattr(runner_module, "BUSY_BACKOFF_SECONDS", 0.05)
    service, identity, _tmux, tmp_path = coordinator
    service.launch(identity, "start")
    folder = service.folder(identity)
    task_id = ids.coordinator_task_id(str(folder))
    task = center.store.get(task_id)
    command = {**task["command"]}
    command["argv"] = [sys.executable, "-c", BUSY_THEN_DONE, str(folder / "state.json")]
    center.store.set_fields(task_id, command=command)

    runner = center.runner()
    center.tick_until(runner, lambda: center.state(task_id) in TERMINAL - {"interrupted"})
    task = center.store.get(task_id)
    assert (task["state"], task["attempt"]) == ("succeeded", 2)
    events = [event["detail"] or {} for event in center.store.events(task_id)]
    assert any(detail.get("exitReason") == "busy" for detail in events)
    assert service.status(identity)["status"] == "completed"


# -- Real pinned refit worker under an in-process runner ------------------------------------------


@pytest.fixture
def refit(tmp_path, center, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("lightning")
    from histopilot.schemas.development import ResourcePolicy
    from histopilot.schemas.predictors import LaunchRefit

    support = runpy.run_path(str(HERE / "test_refit_predictors.py"))
    predictors, _ = support["registry"].__wrapped__(tmp_path)
    selection, _, _ = support["refit_candidate"](predictors, epochs=(12, 12))
    jobs = ComputeJobService(
        predictors.store,
        runtime=lambda: {**runtime(), "host": None},
        execution_mode="task-center",
        task_center=center.client,
    )
    jobs.legacy_executor = LegacyExecutor()
    refits, record, _ = support["create"](predictors, selection, jobs)
    # The worker gets a private TMPDIR: any lease it wrote would appear under it.
    private = tmp_path / "worker-tmp"
    private.mkdir(mode=0o700)
    monkeypatch.setenv("TMPDIR", str(private))

    def launch(operation="launch", *, resume=False):
        return refits.launch(
            record["id"],
            LaunchRefit(
                operationId=operation,
                resources=ResourcePolicy(
                    gpuIds=[], dataLoaderWorkers=0, cpuThreadsPerRun=1, ramGbPerRun=0.01
                ),
            ),
            resume=resume,
        )

    return refits, jobs, record, launch, private


def saved_status(jobs, identity):
    return read_json(jobs.folder(identity) / "state.json")["status"]


def test_real_managed_refit_runs_through_the_runner_without_worker_leases(refit, center):
    refits, jobs, record, launch, private = refit
    identity = record["id"]
    state = launch()
    task_id = state["taskId"]
    owner = center.store.owner(center.store.get(task_id)["ownerKey"])
    assert (owner["kind"], owner["id"]) == ("experiment", record["manifest"]["experimentId"])
    queued = jobs.status(identity)
    assert (queued["status"], queued["task"]["state"]) == ("queued", "queued")

    runner = center.runner()
    center.tick_until(runner, lambda: saved_status(jobs, identity) == "running", timeout=120)
    running = jobs.status(identity)
    assert running["status"] == "running" and running["liveProcesses"]
    assert running["task"]["state"] == "running" and running["gpu"] is None
    own = [lease for lease in leases.read_leases() if lease.get("taskId") == task_id]
    assert len(own) == 1 and own[0]["kind"] == "task-center" and own[0]["live"]

    center.tick_until(runner, lambda: center.state(task_id) in TERMINAL, timeout=180)
    assert center.state(task_id) == "succeeded", center.store.get(task_id)
    final = jobs.status(identity)
    assert (final["status"], final["executor"]) == ("completed", "task-center")
    saved = read_json(jobs.folder(identity) / "state.json")
    assert "waitingReason" not in saved and saved["taskId"] == task_id
    # The runner held the only lease; the worker never touched the registry.
    assert not [lease for lease in leases.read_leases() if lease.get("taskId") == task_id]
    assert not (private / f"histopilot-training-{UID}").exists()
    assert (private / f"histopilot-packing-{UID}").is_dir()  # the worker used this TMPDIR
    measurement = center.store.measurements()[0]
    assert measurement["taskId"] == task_id and measurement["epochs"] == 12
    published = refits.publish(identity, "publish-managed-refit")
    assert published["manifest"]["method"] == "refit"


def test_real_managed_refit_cancelled_while_queued_or_running_is_cancelled(refit, center):
    refits, jobs, record, launch, _private = refit
    identity = record["id"]
    folder = jobs.folder(identity)
    task_id = launch()["taskId"]
    runner = center.runner()
    refits.cancel(identity, "cancel-queued")
    runner.tick()
    assert center.state(task_id) == "cancelled"
    assert jobs.status(identity)["status"] == "cancelled"
    assert saved_status(jobs, identity) == "queued" and not (folder / "worker.log").exists()

    resumed = launch("resume", resume=True)
    assert (resumed["taskAttempt"], center.state(task_id)) == (2, "queued")
    center.tick_until(runner, lambda: saved_status(jobs, identity) == "running", timeout=120)
    refits.cancel(identity, "cancel-running")
    assert center.state(task_id) == "stopping"
    center.tick_until(runner, lambda: center.state(task_id) in TERMINAL, timeout=120)
    assert center.state(task_id) == "cancelled", center.store.get(task_id)
    assert saved_status(jobs, identity) == "cancelled"  # the worker's own verdict
    assert jobs.status(identity)["status"] == "cancelled"


def test_real_managed_refit_is_auto_resumed_after_a_lost_runner(refit, center):
    _refits, jobs, record, launch, _private = refit
    identity = record["id"]
    task_id = launch()["taskId"]
    first = center.runner()
    center.tick_until(first, lambda: saved_status(jobs, identity) == "running", timeout=120)
    # A crash takes the worker, its wrapper and the runner down together.
    process = center.store.get(task_id)["process"]
    os.kill(process["wrapper"]["pid"], signal.SIGKILL)
    procs.kill_group(process)
    first._procs[task_id].wait(timeout=10)
    first.close()

    second = center.runner()  # startup reconcile: lost -> interrupted -> auto-resume
    task = center.store.get(task_id)
    assert (task["state"], task["attempt"]) == ("queued", 2)
    assert center.store.events(task_id)[-1]["detail"]["autoResumed"] is True
    saved = read_json(jobs.folder(identity) / "state.json")
    assert (saved["status"], saved["attempt"], saved["taskAttempt"]) == ("queued", 2, 2)
    assert saved["autoResumedAt"]
    assert jobs.status(identity)["status"] == "queued"
    center.tick_until(second, lambda: center.state(task_id) in TERMINAL, timeout=180)
    assert center.state(task_id) == "succeeded", center.store.get(task_id)
    assert jobs.status(identity)["status"] == "completed"


# -- Predictor coordinator ------------------------------------------------------------------------

STAND_IN_COORDINATOR = r"""
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
state = json.loads(path.read_text())
state["status"] = "completed"
path.write_text(json.dumps(state))
"""

FOLD = r"""
import pathlib, sys, time
time.sleep(0.2)
flag = pathlib.Path(sys.argv[1])
sys.exit(int(flag.read_text()) if flag.exists() else 0)
"""


@pytest.fixture
def coordinator(tmp_path, center):
    from histopilot.application.experiment_predictors import ExperimentPredictorService

    support = runpy.run_path(str(HERE / "test_experiment_predictors.py"))
    legacy, identity, _jobs, _executor, _selections = support["integrated"].__wrapped__(
        support["registry"].__wrapped__(tmp_path)
    )
    service = ExperimentPredictorService(
        legacy.store,
        legacy.filesystem,
        training=legacy.training,
        refits=legacy.refits,
        runtime=legacy.runtime,
        execution_mode="task-center",
        task_center=center.client,
    )
    tmux = support["Executor"]()
    service.legacy_executor = service.executor.legacy = tmux  # never probe real tmux
    return service, identity, tmux, tmp_path


def batch_tasks(service, identity, center, tmp_path, *, failing=None):
    """The fold and collection tasks W3 enqueues for each experiment batch (generic stand-ins)."""
    store = service.store
    batch = store.get_draft(identity)["payload"]["submission"]["batchIds"][0]
    folder = store.folder / "training" / batch
    runs = read_json(folder / "plan.json")["runs"]

    def spec(task_id, kind, order, **extra):
        log = tmp_path / "batch-tasks" / f"{order}.log"
        return {
            "id": task_id,
            "kind": kind,
            "adapter": "generic",
            "title": task_id,
            "group": {"kind": "mil-batch", "id": batch},
            "planOrder": order,
            "request": {"lane": "cpu", "cpuThreads": 1, "ramGb": 0.1},
            "command": {
                "argv": [sys.executable, "-c", FOLD, str(tmp_path / f"flag-{order}")],
                "cwd": str(tmp_path),
                "log": str(log),
            },
            **extra,
        }

    folds = [ids.fold_task_id(str(folder), run["id"]) for run in runs]
    final = ids.collect_task_id(str(folder), True)
    progress = ids.collect_task_id(str(folder), False)
    if failing is not None:
        (tmp_path / f"flag-{failing}").write_text("1")
    center.client.enqueue(
        {
            "kind": "experiment",
            "id": identity,
            "projectId": store.project_id,
            "projectFolder": str(store.folder),
            "title": "Experiment",
        },
        [
            *(spec(task, "mil-fold", index) for index, task in enumerate(folds)),
            spec(progress, "mil-collect", len(folds), adapterData={"final": False}),
            spec(
                final,
                "mil-collect",
                len(folds) + 1,
                adapterData={"final": True},
                dependsOn=[{"task": task, "condition": "terminal"} for task in folds],
            ),
        ],
    )
    return folds, final, progress


def test_coordinator_waits_for_final_results_and_a_failed_fold_does_not_block_it(
    coordinator, center
):
    service, identity, tmux, tmp_path = coordinator
    folds, final, progress = batch_tasks(service, identity, center, tmp_path, failing=0)
    launched = service.launch(identity, "start")
    folder = service.folder(identity)
    task_id = ids.coordinator_task_id(str(folder))
    task = center.store.get(task_id)
    assert task["state"] == "blocked" and launched["status"] == "queued"
    # The coordinator decides per batch, so only each batch's final results gate it.
    assert {row["task"]: row["condition"] for row in center.store.dependencies(task_id)} == {
        final: "terminal"
    }
    assert progress not in {row["task"] for row in center.store.dependencies(task_id)}
    assert task["request"]["service"] is True and task["request"]["lane"] == "cpu"
    # The coordinator probes CUDA for its refits, so the CPU lane's GPU mask is lifted.
    assert task["command"]["argv"][1:3] == ["-u", "CUDA_VISIBLE_DEVICES"]
    assert task["command"]["argv"][4:] == [
        "-u",
        "-m",
        "histopilot.workers.experiment_predictors",
        str(folder / "plan.json"),
    ]
    owner = center.store.owner(task["ownerKey"])
    assert (owner["kind"], owner["id"]) == ("experiment", identity)
    status = service.status(identity)
    assert (status["status"], status["executor"]) == ("queued", "task-center")
    assert "fold batches" in status["waitingReason"]
    assert service.launch(identity, "start")["status"] == "queued"
    assert center.store.get(task_id)["attempt"] == 1 and tmux.launches == []

    # Stand in for the real coordinator: record completion like the worker would.
    command = {**task["command"]}
    command["argv"] = [sys.executable, "-c", STAND_IN_COORDINATOR, str(folder / "state.json")]
    center.store.set_fields(task_id, command=command)

    runner = center.runner()
    center.tick_until(runner, lambda: center.state(task_id) in TERMINAL)
    assert center.state(folds[0]) == "failed"
    assert center.state(task_id) == "succeeded"
    started = center.store.get(task_id)["startedAt"]
    assert all(center.store.get(task)["finishedAt"] <= started for task in [*folds, final])
    assert service.status(identity)["status"] == "completed"


def test_cancel_drops_a_coordinator_that_is_still_waiting(coordinator, center):
    service, identity, _tmux, tmp_path = coordinator
    batch_tasks(service, identity, center, tmp_path)
    service.launch(identity, "start")
    task_id = ids.coordinator_task_id(str(service.folder(identity)))
    assert center.state(task_id) == "blocked"
    assert service.cancel(identity, "cancel")["status"] == "cancelled"
    assert center.state(task_id) == "cancelled"
    assert service.status(identity)["status"] == "cancelled"


def test_a_cancelled_task_center_coordinator_resumes(coordinator, center):
    service, identity, tmux, tmp_path = coordinator
    _folds, final, _progress = batch_tasks(service, identity, center, tmp_path)
    service.launch(identity, "start")
    folder = service.folder(identity)
    task_id = ids.coordinator_task_id(str(folder))
    cancelled = service.cancel(identity, "cancel")
    assert (cancelled["status"], cancelled["retryable"]) == ("cancelled", True)
    assert center.state(task_id) == "cancelled"
    assert {item["status"] for item in cancelled["items"]} == {"cancelled"}

    resumed = service.launch(identity, "resume", resume=True)
    assert resumed["status"] == "queued" and not (folder / "cancel.requested").exists()
    assert {item["status"] for item in resumed["items"]} == {"waiting"}
    task = center.store.get(task_id)
    # Still waiting for the batch's final results, now on its second attempt.
    assert (task["state"], task["attempt"]) == ("blocked", 2)
    assert {row["task"] for row in center.store.dependencies(task_id)} == {final}
    assert service.launch(identity, "resume", resume=True)["status"] == "queued"
    assert center.store.get(task_id)["attempt"] == 2 and tmux.launches == []


def test_a_coordinator_submission_wakes_the_runner_and_reports_it(coordinator, center, monkeypatch):
    from histopilot.application.experiment_predictors import TaskCenterExperimentExecutor
    from histopilot.taskcenter import launcher

    service, identity, tmux, tmp_path = coordinator
    batch_tasks(service, identity, center, tmp_path)
    calls = []
    monkeypatch.setattr(launcher, "ensure_runner", lambda: calls.append(1) or {"started": True})
    service.executor = TaskCenterExperimentExecutor(legacy=tmux)
    service._task_center = None
    service.launch(identity, "start")
    assert calls == [1]
    status = service.status(identity)
    assert (status["executor"], status["runnerAlive"]) == ("task-center", False)
    lock = hold_runner_lock(center)
    try:
        assert service.status(identity)["runnerAlive"] is True
    finally:
        lock.close()


def test_coordinator_without_batch_tasks_or_protocol(coordinator, center, monkeypatch):
    service, identity, tmux, _tmp_path = coordinator
    from histopilot.application import experiment_predictors

    # Batches launched before the Task Center add no dependencies ...
    service.launch(identity, "start")
    task_id = ids.coordinator_task_id(str(service.folder(identity)))
    assert center.state(task_id) == "queued"
    assert center.store.dependencies(task_id) == []
    center.store.cancel_pending([task_id])
    state_path = service.folder(identity) / "state.json"
    state = read_json(state_path)
    write_json(state_path, {**state, "status": "attention"})
    # ... and a coordinator archive pinned before the Task Center resumes in tmux.
    monkeypatch.setattr(experiment_predictors, "archive_protocol", lambda _path: None)
    resumed = service.launch(identity, "resume", resume=True)
    assert resumed["status"] == "queued" and "executor" not in resumed
    assert len(tmux.launches) == 1
    assert "executor" not in read_json(state_path)
    assert center.state(task_id) == "cancelled"


# -- Background bulk evaluation submission --------------------------------------------------------


@pytest.fixture
def bulk(tmp_path, monkeypatch, center):
    support = runpy.run_path(str(HERE / "test_bulk_evaluations.py"))
    service, _predictor, cohort, executor = support["bulk"].__wrapped__(tmp_path, monkeypatch)
    service.background = True
    service._task_center = center.client
    return service, cohort, executor, support


def test_background_bulk_submission_is_a_task_and_members_read_queued(bulk, center, monkeypatch):
    from histopilot.schemas.bulk_evaluations import BulkEvaluationSelection
    from histopilot.taskcenter import jobs

    service, cohort, executor, support = bulk
    choice = BulkEvaluationSelection(cohortId=cohort["id"])
    request = support["run_request"](choice, service.preview(choice))
    result = service.run(request)
    assert executor.calls == []  # nothing launched inside the request
    assert result["status"] == "queued" and not result["submitted"]
    assert [row["status"] for row in result["items"] if row["eligible"]] == ["queued"]
    assert result["counts"]["planned"] == 0 and result["counts"]["queued"] == 1
    store = service.store
    task_id = ids.bulk_submit_task_id(str(store.folder), result["id"])
    task = center.store.get(task_id)
    assert (task["state"], task["kind"], task["priority"]) == (
        "queued",
        "bulk-submit",
        "interactive",
    )
    assert task["request"]["lane"] == "cpu" and task["adapterData"]["requeueSafe"] is True
    argv = task["command"]["argv"]
    assert argv[1:3] == ["-u", "CUDA_VISIBLE_DEVICES"]  # members' device review probes CUDA
    assert argv[4:8] == ["-u", "-m", "histopilot.taskcenter.jobs", "bulk-submit"]
    assert argv[argv.index("--batch") + 1] == result["id"]
    assert argv[argv.index("--project-folder") + 1] == str(store.folder)
    assert argv[argv.index("--project-id") + 1] == store.project_id
    assert [argv[i + 1] for i, value in enumerate(argv) if value == "--data-root"] == [
        str(root) for root in service.filesystem.roots
    ]
    assert Path(task["command"]["cwd"], "histopilot", "taskcenter", "jobs.py").is_file()
    owner = center.store.owner(task["ownerKey"])
    assert (owner["kind"], owner["id"]) == ("evaluation-batch", result["id"])
    assert service.run(request)["id"] == result["id"]
    assert len(center.store.list(limit=None)) == 1

    # The task's entrypoint submits members exactly as the inline path would.
    monkeypatch.setattr(compute_module, "training_runtime", runtime)
    assert (
        jobs.main(
            [
                "bulk-submit",
                "--project-folder",
                str(store.folder),
                "--project-id",
                store.project_id,
                "--batch",
                result["id"],
                *[
                    value
                    for root in service.filesystem.roots
                    for value in ("--data-root", str(root))
                ],
            ]
        )
        == 0
    )
    member = next(row for row in service.get(result["id"])["items"] if row["eligible"])
    evaluation = ids.compute_task_id(str(store.folder / "compute-jobs" / member["evaluationId"]))
    evaluation_task = center.store.get(evaluation)
    evaluation_owner = center.store.owner(evaluation_task["ownerKey"])
    assert (evaluation_owner["kind"], evaluation_owner["id"]) == ("evaluation-batch", result["id"])
    assert evaluation_task["ownerKey"] == task["ownerKey"]
    after = service.get(result["id"])
    assert after["submitted"] and member["status"] == "queued"
    assert member["execution"]["executor"] == "task-center"


def test_background_submission_wakes_the_runner(bulk, center, monkeypatch):
    from histopilot.schemas.bulk_evaluations import BulkEvaluationSelection
    from histopilot.taskcenter import launcher

    service, cohort, _executor, support = bulk
    calls = []
    monkeypatch.setattr(launcher, "ensure_runner", lambda: calls.append(1) or {"started": True})
    choice = BulkEvaluationSelection(cohortId=cohort["id"])
    request = support["run_request"](choice, service.preview(choice))
    service._default_task_center = False
    service.run(request)
    assert calls == []  # an injected client never starts a runner
    service._default_task_center = True
    service.run(request)  # a replay of a still-queued submission wakes it too
    assert calls == [1]


def test_background_submission_retry_and_cancel(bulk, center):
    from histopilot.schemas.bulk_evaluations import BulkEvaluationSelection

    service, cohort, _executor, support = bulk
    choice = BulkEvaluationSelection(cohortId=cohort["id"])
    request = support["run_request"](choice, service.preview(choice))
    result = service.run(request)
    task_id = ids.bulk_submit_task_id(str(service.store.folder), result["id"])
    center.store.transition(task_id, from_states="queued", to_state="failed")
    assert service.get(result["id"])["items"][0]["status"] == "planned"
    service.run(request)  # an exact retry re-queues a submission that stopped early
    assert (center.state(task_id), center.store.get(task_id)["attempt"]) == ("queued", 2)
    cancelled = service.cancel(result["id"], "cancel")
    assert center.state(task_id) == "cancelled"
    assert cancelled["items"][0]["status"] == "cancelled"


def test_background_submission_waits_out_a_busy_workspace(bulk, center, monkeypatch):
    from histopilot.application.bulk_evaluations import BulkEvaluationService
    from histopilot.schemas.bulk_evaluations import BulkEvaluationSelection
    from histopilot.taskcenter import jobs

    service, cohort, _executor, support = bulk
    choice = BulkEvaluationSelection(cohortId=cohort["id"])
    batch = service.run(support["run_request"](choice, service.preview(choice)))["id"]
    calls, sleeps = [], []

    def submit(_self, _batch, failures):
        calls.append(_batch["id"])
        if len(calls) <= failures:
            raise StorageError("Another operation is changing this workspace.", "PROJECT_BUSY")

    store, roots = service.store, [str(root) for root in service.filesystem.roots]
    monkeypatch.setattr(BulkEvaluationService, "_submit", lambda s, b: submit(s, b, 2))
    jobs.bulk_submit(store.folder, store.project_id, batch, roots, sleep=sleeps.append)
    assert calls == [batch] * 3 and len(sleeps) == 2  # replays only submit missing members

    calls.clear()
    monkeypatch.setattr(jobs, "BUSY_WAIT_SECONDS", 0)
    with pytest.raises(jobs.Busy):
        jobs.bulk_submit(store.folder, store.project_id, batch, roots, sleep=sleeps.append)
    assert calls == [batch]
    # The busy contract: the task exits 75 and the runner requeues it.
    calls.clear()
    argv = ["bulk-submit", "--project-folder", str(store.folder), "--project-id"]
    argv += [store.project_id, "--batch", batch, *(f"--data-root={root}" for root in roots)]
    assert jobs.main(argv) == jobs.BUSY_EXIT

    def broken(_self, _batch):
        raise StorageError("Changed.", "EVALUATION_BATCH_CHANGED")

    monkeypatch.setattr(BulkEvaluationService, "_submit", broken)
    monkeypatch.setattr(jobs, "BUSY_WAIT_SECONDS", 600)
    with pytest.raises(StorageError) as caught:
        jobs.bulk_submit(store.folder, store.project_id, batch, roots, sleep=sleeps.append)
    assert caught.value.code == "EVALUATION_BATCH_CHANGED"


def test_orchestration_tasks_see_the_hosts_gpus_on_the_cpu_lane(tmp_path):
    probe = [sys.executable, "-c", "import os; print(os.environ.get('CUDA_VISIBLE_DEVICES'))"]
    task = {"id": "task-probe", "attempt": 1}

    def spawned(argv, name):
        log = tmp_path / f"{name}.log"
        command = {"argv": argv, "cwd": str(tmp_path), "env": {}, "log": str(log)}
        child = procs.spawn(command, lane="cpu", gpu=None, task=task)
        assert child.wait(timeout=30) == 0
        return log.read_text().strip()

    # The runner hides GPUs from CPU-lane tasks; a coordinator's runtime probe would then
    # report cudaAvailable False and refuse every GPU refit.
    assert spawned(probe, "masked") == ""
    assert spawned(compute_module.host_gpu_argv(probe), "restored") == "None"
