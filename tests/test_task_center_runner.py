"""The runner end to end, with real short-lived Python tasks in their own sessions."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from support.task_center import Center as TaskCenter
from support.task_center import fake_host

from histopilot.storage.project_lock import StorageError
from histopilot.taskcenter import leases, procs
from histopilot.taskcenter import runner as runner_module
from histopilot.taskcenter.adapters.base import BUSY_EXIT, AdapterError, outcome
from histopilot.taskcenter.adapters.generic import GenericAdapter
from histopilot.taskcenter.launcher import runner_alive
from histopilot.taskcenter.model import TERMINAL
from histopilot.taskcenter.runner import BUSY_WAIT, Runner, RunnerBusy, host_stop

# Waits for <folder>/release (whose text is the exit code). On SIGTERM it writes
# <folder>/terminated and exits 0, unless told to ignore the signal.
TASK = r"""
import pathlib, signal, sys, time
folder, mode = pathlib.Path(sys.argv[1]), sys.argv[2]
def stop(*_):
    (folder / "terminated").write_text("terminated")
    sys.exit(0)
signal.signal(signal.SIGTERM, signal.SIG_IGN if mode == "ignore" else stop)
(folder / "started").write_text("started")
release = folder / "release"
while not release.exists():
    time.sleep(0.02)
sys.exit(int(release.read_text() or 0))
"""


class Center(TaskCenter):
    """This test's Task Center, plus synthetic generic tasks that each run TASK in a folder."""

    def __init__(self, tmp_path):
        super().__init__()
        self.tmp_path = tmp_path
        self.project = str(tmp_path / "project")

    def runner(self, *, stop=None, **options):
        if stop is None:
            return super().runner(**options)
        # Only a stop given to start() can interrupt startup reconciliation.
        runner = Runner(
            self.store,
            **{
                "host_probe": fake_host(),
                "cuda_probe": lambda gpu: ("ok", None),
                "sample_interval": 0.0,
                "host_interval": 0.0,
                "log": self.logs.append,
                **options,
            },
        )
        self.runners.append(runner)
        runner.start(stop=stop)
        return runner

    def folder(self, identity):
        return self.tmp_path / "tasks" / identity

    def spec(self, identity, *, group, order=0, mode="graceful", request=None, **extra):
        folder = self.folder(identity)
        return {
            "id": identity,
            "kind": "generic",
            "adapter": "generic",
            "title": identity,
            "group": {"kind": "batch", "id": group},
            "planOrder": order,
            "request": {
                "lane": "gpu",
                "cpuThreads": 1,
                "ramGb": 0.5,
                "vramGb": 1.0,
                **(request or {}),
            },
            "command": {
                "argv": [sys.executable, "-c", TASK, str(folder), mode],
                "cwd": str(self.tmp_path),
                "log": str(folder / "run.log"),
                "progress": str(folder / "progress.json"),
            },
            **extra,
        }

    def enqueue(self, owner, *specs):
        return self.client.enqueue(
            {
                "kind": "experiment",
                "id": owner,
                "projectId": "project-1",
                "projectFolder": self.project,
                "title": f"Experiment {owner}",
            },
            list(specs),
        )

    def release(self, identity, code=0):
        # Atomic, so a task never reads a half-written file as exit code 0.
        staged = self.folder(identity) / "release.tmp"
        staged.write_text(str(code))
        staged.replace(self.folder(identity) / "release")

    def started(self, identity):
        return (self.folder(identity) / "started").exists()

    def running(self):
        return {task["id"] for task in self.store.list(states=("running",), limit=None)}


@pytest.fixture
def registry(tmp_path, monkeypatch):
    folder = tmp_path / "tmp"
    folder.mkdir(mode=0o700)
    monkeypatch.setattr(tempfile, "tempdir", str(folder))
    return folder / f"histopilot-training-{os.getuid()}"


@pytest.fixture
def center(tmp_path, registry, _task_center_state):
    value = Center(tmp_path)
    yield value
    value.cleanup()


def test_fifo_fill_across_owners_and_cancel_refills_from_the_next_owner(center):
    center.store.update_settings({"gpuSlots": {"0": 2}})
    center.enqueue("A", *(center.spec(f"a{i}", group="gA", order=i) for i in range(1, 4)))
    center.enqueue("B", *(center.spec(f"b{i}", group="gB", order=i) for i in range(1, 4)))
    center.enqueue("C", *(center.spec(f"c{i}", group="gC", order=i) for i in range(1, 3)))
    runner = center.runner()

    result = runner.tick()
    assert result["started"] == ["a1", "a2"]
    assert result["waiting"]["a3"] == "Waiting for a GPU slot (2/2)"
    assert center.store.get("b1")["waitingReason"] == "Waiting for a GPU slot (2/2)"

    center.release("a1")
    center.tick_until(runner, lambda: center.running() == {"a2", "a3"})
    assert center.state("a1") == "succeeded"
    # Owner A's last task and owner B's first task share the GPU.
    center.release("a2")
    center.tick_until(runner, lambda: center.running() == {"a3", "b1"})
    center.release("a3")
    center.tick_until(runner, lambda: center.running() == {"b1", "b2"})
    assert center.state("b3") == "queued" and center.state("c1") == "queued"
    center.tick_until(runner, lambda: all(center.started(item) for item in ("b1", "b2")))

    cancelled = center.client.cancel_group("batch", "gB", center.project)
    assert cancelled == {"cancelled": ["b3"], "stopping": ["b1", "b2"]}
    center.tick_until(runner, lambda: center.running() == {"c1", "c2"})
    for identity in ("b1", "b2"):
        task = center.store.get(identity)
        assert task["state"] == "cancelled"
        assert task["exit"]["reason"] == "cancelled"
        assert task["exit"]["signalled"] is True and task["exit"]["killed"] is False
        assert (center.folder(identity) / "terminated").exists()  # stopped gracefully
    assert center.state("b3") == "cancelled"
    assert center.store.get("b3")["startedAt"] is None
    for identity in ("c1", "c2"):
        center.release(identity)
    center.tick_until(runner, lambda: not center.running())
    assert center.state("c2") == "succeeded"


def test_stop_escalates_to_sigkill_after_the_grace_period(center):
    center.enqueue(
        "A", center.spec("stubborn", group="g", mode="ignore", request={"graceSeconds": 0.3})
    )
    runner = center.runner()
    center.tick_until(runner, lambda: center.started("stubborn"))
    center.client.cancel_task("stubborn")
    assert runner.tick()["stopped"] == ["stubborn"]
    task = center.store.get("stubborn")
    assert task["state"] == "stopping" and task["signalledAt"] and task["killedAt"] is None
    assert procs.alive(task["process"])
    time.sleep(0.4)
    center.tick_until(runner, lambda: center.state("stubborn") == "cancelled")
    task = center.store.get("stubborn")
    assert task["killedAt"] is not None
    assert task["exit"]["killed"] is True and task["exit"]["signalled"] is True
    assert not (center.folder("stubborn") / "terminated").exists()


def test_pause_requeues_on_hold_and_release_resumes(center):
    enqueued = center.enqueue("A", center.spec("fold", group="g"))
    key = enqueued["owner"]["key"]
    runner = center.runner()
    center.tick_until(runner, lambda: center.started("fold"))
    center.store.hold_owner(key, True)
    assert center.store.request_stop(["fold"], "pause") == ["fold"]
    center.tick_until(runner, lambda: center.state("fold") == "queued")
    task = center.store.get("fold")
    assert task["attempt"] == 2 and task["exit"] is None
    assert (center.folder("fold") / "terminated").exists()
    runner.tick()
    runner.tick()
    assert center.state("fold") == "queued"
    assert center.store.get("fold")["waitingReason"] == "On hold"
    assert center.store.owner(key)["held"] is True
    history = [(event["fromState"], event["toState"]) for event in center.store.events("fold")]
    assert history[-3:] == [
        ("running", "stopping"),
        ("stopping", "interrupted"),
        (
            "interrupted",
            "queued",
        ),
    ]
    center.store.hold_owner(key, False)
    (center.folder("fold") / "started").unlink()
    center.tick_until(runner, lambda: center.state("fold") == "running")
    assert center.store.get("fold")["attempt"] == 2
    center.release("fold")
    center.tick_until(runner, lambda: center.state("fold") == "succeeded")


def test_a_hold_committed_during_admission_keeps_the_task_queued(center):
    enqueued = center.enqueue(
        "A", center.spec("early", group="g"), center.spec("late", group="g", order=1)
    )
    key = enqueued["owner"]["key"]
    runner = center.runner()
    spawn = runner.spawner
    spawned = []

    def hold_while_spawning(command, **options):
        # The owner is held after the runner checked it but before the task is marked running.
        spawned.append(options["task"]["id"])
        if spawned == ["early", "late"]:
            center.store.hold_owner(key, True)
        return spawn(command, **options)

    runner.spawner = hold_while_spawning
    runner.tick()
    assert center.state("early") == "running"
    assert center.state("late") == "queued" and spawned == ["early", "late"]
    # Held before the spawn: nothing is started at all.
    runner.tick()
    assert spawned == ["early", "late"] and center.state("late") == "queued"
    assert center.store.get("late")["waitingReason"] == "On hold"
    center.store.hold_owner(key, False)
    center.tick_until(runner, lambda: center.state("late") == "running")
    assert spawned == ["early", "late", "late"]
    center.release("early")
    center.release("late")
    center.tick_until(
        runner, lambda: {center.state("early"), center.state("late")} == {"succeeded"}
    )


def test_cancel_group_stops_a_task_started_after_the_listing(center):
    center.enqueue("A", center.spec("fold", group="g"))
    stale = center.store.list(limit=None)
    runner = center.runner()
    center.tick_until(runner, lambda: center.started("fold"))
    center.client._group_tasks = lambda *args: stale  # listed while still queued
    result = center.client.cancel_group("batch", "g", center.project)
    assert result == {"cancelled": [], "stopping": ["fold"]}
    center.tick_until(runner, lambda: center.state("fold") == "cancelled")


class Resumable(GenericAdapter):
    def __init__(self):
        self.requeued = []

    def on_requeue(self, task, ctx):
        self.requeued.append(task["id"])


def test_startup_reconciles_lost_tasks_adopts_live_ones_and_auto_resumes(center, registry):
    project_folder = center.folder("adopted")
    project_folder.mkdir(parents=True)
    live = subprocess.Popen(
        [sys.executable, "-c", TASK, str(project_folder), "graceful"],
        start_new_session=True,
    )
    try:
        center.enqueue(
            "A",
            center.spec("lost-safe", group="g", adapterData={"requeueSafe": True}),
            center.spec("lost-unsafe", group="g", order=1),
            center.spec("adopted", group="g", order=2),
        )
        lost = {"pid": 999_999, "startTicks": 1, "bootId": "an-earlier-boot"}
        for identity in ("lost-safe", "lost-unsafe"):
            center.store.transition(
                identity,
                from_states="queued",
                to_state="running",
                started_at="2026-01-01T00:00:00+00:00",
                process=lost,
                gpu=0,
            )
        stale = leases.write_task_lease(
            center.store.get("lost-safe"),
            lost,
            0,
            cpus=1,
            ram_gb=0.5,
            runs_per_gpu=4,
            supervisor={"pid": 999_998, "startTicks": 1, "bootId": "an-earlier-boot"},
        )
        while not (project_folder / "started").exists():
            time.sleep(0.02)
        center.store.transition(
            "adopted",
            from_states="queued",
            to_state="running",
            started_at="2026-01-01T00:00:00+00:00",
            process=procs.identity(live.pid),
            gpu=0,
        )
        adapter = Resumable()
        runner = center.runner(adapters=lambda name: adapter)

        safe = center.store.get("lost-safe")
        assert (safe["state"], safe["attempt"]) == ("queued", 2)
        assert center.store.events("lost-safe")[-1]["detail"]["autoResumed"] is True
        assert adapter.requeued == ["lost-safe"]
        unsafe = center.store.get("lost-unsafe")
        assert unsafe["state"] == "interrupted" and unsafe["exit"]["lost"] is True
        assert unsafe["exit"]["reason"] == "lost"
        assert center.state("adopted") == "running"
        assert not (registry / stale).exists()
        assert center.store.measurements() == []  # lost attempts are not measurements

        center.tick_until(runner, lambda: center.state("lost-safe") == "running")
        assert center.state("adopted") == "running"
        (project_folder / "release").write_text("0")
        live.wait(timeout=10)
        center.tick_until(runner, lambda: center.state("adopted") != "running")
        # An adopted process's exit status is unobservable, so its outcome is unknown.
        assert center.store.get("adopted")["exit"]["reason"] == "interrupted"
        center.release("lost-safe")
        center.tick_until(runner, lambda: center.state("lost-safe") == "succeeded")
    finally:
        if live.poll() is None:
            live.kill()
            live.wait(timeout=5)


def test_a_restarted_runner_classifies_exits_from_the_wrapper_record(center):
    """Adopted tasks used to end "interrupted" whatever their exit code; the wrapper's
    exit.json keeps it across runner restarts (for example on every deploy)."""
    center.enqueue(
        "A",
        center.spec("plain", group="g"),
        center.spec("busy", group="g", order=1),
        center.spec("killed", group="g", order=2),
        center.spec("running", group="g", order=3),
    )
    first = center.runner()
    center.tick_until(first, lambda: all(center.started(t) for t in ("plain", "busy", "killed")))
    first.close()  # the runner stops (a deploy); its tasks keep running
    center.release("plain", 0)
    center.release("busy", 75)  # the busy contract: exit 75
    os.killpg(center.store.get("killed")["process"]["pid"], signal.SIGKILL)
    for identity in ("plain", "busy", "killed"):
        process = center.store.get(identity)["process"]
        while procs.leader_alive(process) or procs.leader_alive(process["wrapper"]):
            time.sleep(0.02)
    second = center.runner()  # startup reconciliation reads the exit records
    assert center.state("plain") == "succeeded"
    assert center.store.get("plain")["exit"]["returncode"] == 0
    busy = center.store.get("busy")
    assert (busy["state"], busy["attempt"]) == ("queued", 2)  # requeued, not interrupted
    killed = center.store.get("killed")
    assert (killed["state"], killed["exit"]["returncode"]) == ("failed", -signal.SIGKILL)
    assert "signal 9" in killed["error"]
    # A task adopted alive is classified the same way once it exits.
    assert center.state("running") == "running"
    center.release("running", 0)
    center.tick_until(second, lambda: center.state("running") != "running")
    assert center.state("running") == "succeeded"
    assert list(second.journal_dir.glob("*.exit.json")) == []


def _wait_until_exited(center, *identities):
    for identity in identities:
        process = center.store.get(identity)["process"]
        while procs.leader_alive(process) or procs.leader_alive(process["wrapper"]):
            time.sleep(0.02)


def test_a_shutdown_while_no_runner_ran_resumes_its_tasks_instead_of_failing_them(center):
    """A graceful shutdown signals every process: the wrapper survives and records the
    exit (Lightning turns SIGTERM into exit 1). Such exits, and any exit recorded in an
    earlier boot, are interruptions that resume; a plain failure in this boot still fails."""
    safe = {"adapterData": {"requeueSafe": True}}
    center.enqueue(
        "A",
        center.spec("shutdown", group="g", **safe),
        center.spec("earlier-boot", group="g", order=1, **safe),
        center.spec("crashed", group="g", order=2, **safe),
    )
    first = center.runner()
    center.tick_until(first, lambda: all(center.started(t) for t in ("shutdown", "earlier-boot")))
    center.tick_until(first, lambda: center.started("crashed"))
    first.close()
    os.kill(center.store.get("shutdown")["process"]["wrapper"]["pid"], signal.SIGTERM)
    time.sleep(0.2)  # the wrapper notes the signal before its task exits
    for identity in ("shutdown", "earlier-boot", "crashed"):
        center.release(identity, 1)
    _wait_until_exited(center, "shutdown", "earlier-boot", "crashed")
    record = Path(first._journal(center.store.get("shutdown"))["exit"])
    assert json.loads(record.read_text())["hostSignal"] == signal.SIGTERM
    earlier = Path(first._journal(center.store.get("earlier-boot"))["exit"])
    value = json.loads(earlier.read_text())
    assert value["bootId"] and value["hostSignal"] is None
    earlier.write_text(json.dumps({**value, "bootId": "an-earlier-boot"}))

    center.runner()
    for identity in ("shutdown", "earlier-boot"):
        task = center.store.get(identity)
        assert (task["state"], task["attempt"]) == ("queued", 2), identity
        assert center.store.events(identity)[-1]["detail"]["autoResumed"] is True
    crashed = center.store.get("crashed")
    assert (crashed["state"], crashed["exit"]["returncode"]) == ("failed", 1)


def test_a_host_signal_the_task_center_never_sent_interrupts_instead_of_failing(center):
    safe = {"adapterData": {"requeueSafe": True}}
    center.enqueue(
        "A",
        center.spec("hangup", group="g", **safe),
        center.spec("session-end", group="g", order=1, **safe),
    )
    runner = center.runner()
    center.tick_until(runner, lambda: center.started("hangup") and center.started("session-end"))
    # SIGHUP to the task alone (the test task does not handle it): interrupted, and it
    # waits for a retry, since nothing says the machine is going down.
    os.kill(center.store.get("hangup")["process"]["pid"], signal.SIGHUP)
    center.tick_until(runner, lambda: center.state("hangup") in TERMINAL)
    hangup = center.store.get("hangup")
    assert (hangup["state"], hangup["attempt"]) == ("interrupted", 1)
    assert hangup["exit"]["lost"] is True and hangup["exit"]["returncode"] == -signal.SIGHUP
    assert "bookkeeping" not in hangup or not hangup["bookkeeping"]
    # The wrapper was signalled too (a closed login session, a shutdown): resumed.
    process = center.store.get("session-end")["process"]
    os.kill(process["wrapper"]["pid"], signal.SIGTERM)
    time.sleep(0.2)
    os.kill(process["pid"], signal.SIGHUP)
    center.tick_until(runner, lambda: center.store.get("session-end")["attempt"] == 2)
    events = center.store.events("session-end")
    assert any((event["detail"] or {}).get("autoResumed") for event in events)


@pytest.mark.parametrize(
    ("fields", "code", "record", "expected"),
    [
        ({}, -signal.SIGTERM, None, "signal"),
        ({}, 1, {"hostSignal": 15}, "shutdown"),
        ({}, 1, {"bootId": "an-earlier-boot"}, "shutdown"),
        ({}, 1, {"bootId": "this-boot"}, None),
        ({}, -signal.SIGKILL, None, None),  # the OOM killer: the task's own outcome
        ({}, BUSY_EXIT, {"bootId": "an-earlier-boot"}, None),
        ({}, 0, {"hostSignal": 15}, None),
        ({"stopRequest": "pause"}, -signal.SIGTERM, {"hostSignal": 15}, None),
        ({"signalledAt": "2026-01-01T00:00:00+00:00"}, -signal.SIGTERM, None, None),
    ],
)
def test_host_stop_tells_host_signals_from_task_failures(fields, code, record, expected):
    task = {"stopRequest": None, "signalledAt": None, **fields}
    assert host_stop(task, code, record, "this-boot") == expected


def test_a_journaled_start_is_adopted_or_undone_by_the_next_runner(center):
    center.enqueue("A", center.spec("adopt", group="g"), center.spec("undo", group="g", order=1))
    probe = Runner(center.store, host_probe=fake_host(), log=center.logs.append)
    # A runner that died between spawning a task and recording it: the start is journaled
    # (state "starting") and the wrapper recorded the process.
    for identity in ("adopt", "undo"):
        center.store.transition(identity, from_states=("queued",), to_state="starting", gpu=0)
    task = center.store.get("adopt")
    child = procs.spawn(task["command"], lane="gpu", gpu=0, task=task, journal=probe._journal(task))
    try:
        while not center.started("adopt"):
            time.sleep(0.02)
        children = []
        runner = center.runner(spawner=recording_spawner(children))
        adopted = center.store.get("adopt")
        assert adopted["state"] == "running" and adopted["process"]["pid"] == child.pid
        assert adopted["attempt"] == 1 and adopted["lease"]
        # Never started: queued again, same attempt.
        undone = center.store.get("undo")
        assert (undone["state"], undone["attempt"]) == ("queued", 1)
        assert center.store.events("undo")[-1]["detail"]["event"] == "start-undone"
        center.tick_until(runner, lambda: center.started("undo"))
        assert [spawned.pid for spawned in children] == [center.store.get("undo")["process"]["pid"]]
        center.release("adopt")
        center.release("undo")
        center.tick_until(
            runner, lambda: {center.state("adopt"), center.state("undo")} == {"succeeded"}
        )
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


def test_a_task_cancelled_while_starting_has_its_journaled_process_stopped(center, registry):
    """The runner died between spawning a task and recording it, then a cancel arrived
    (``starting`` → ``stopping`` without a process): the next runner finds the process in
    the journal and stops it, instead of concluding the task while it runs on untracked."""
    center.store.update_settings({"gpuSlots": {"0": 3}})
    center.enqueue("A", center.spec("stop", group="g"))
    probe = Runner(center.store, host_probe=fake_host(), log=center.logs.append)
    center.store.transition("stop", from_states=("queued",), to_state="starting", gpu=0)
    task = center.store.get("stop")
    child = procs.spawn(task["command"], lane="gpu", gpu=0, task=task, journal=probe._journal(task))
    try:
        while not center.started("stop"):
            time.sleep(0.02)
        center.client.cancel_task("stop")
        stopping = center.store.get("stop")
        assert stopping["state"] == "stopping" and not stopping["process"]
        runner = center.runner()
        adopted = center.store.get("stop")
        assert adopted["state"] == "stopping" and adopted["process"]["pid"] == child.pid
        # The adopted lease shares the GPU like a launched one (3 slots), never exclusively.
        assert json.loads((registry / adopted["lease"]).read_text())["runsPerGpu"] == 3
        center.tick_until(runner, lambda: center.state("stop") == "cancelled")
        assert (center.folder("stop") / "terminated").exists()  # stopped by the runner
        child.wait(timeout=10)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


class FlakyExit(GenericAdapter):
    def __init__(self, failures=1):
        self.failures = failures
        self.calls = 0

    def on_exit(self, task, exit, ctx):
        self.calls += 1
        if self.failures:
            self.failures -= 1
            raise AdapterError("PROJECT_BUSY", transient=True)
        return super().on_exit(task, exit, ctx)


def test_a_failing_exit_hook_is_retried_with_a_backoff_not_every_tick(center):
    center.enqueue("A", center.spec("fold", group="g"))
    adapter = FlakyExit(failures=100)
    clock = [0.0]
    runner = center.runner(adapters=lambda name: adapter, clock=lambda: clock[0])
    runner.tick()
    center.release("fold")
    center.tick_until(runner, lambda: center.store.get("fold")["bookkeeping"])
    calls = adapter.calls
    runner.tick()  # the first retry is immediate ...
    assert adapter.calls == calls + 1
    for _ in range(5):  # ... the next waits 2 s
        runner.tick()
    assert adapter.calls == calls + 1
    clock[0] += 2.1
    runner.tick()
    assert adapter.calls == calls + 2
    clock[0] += 2.1  # then 4 s
    runner.tick()
    assert adapter.calls == calls + 2
    # A cancel is not kept waiting by the backoff.
    center.store.request_stop(["fold"], "cancel")
    runner.tick()
    assert adapter.calls == calls + 3
    adapter.failures = 0
    clock[0] += 60
    center.tick_until(runner, lambda: center.state("fold") == "cancelled")


def test_transient_exit_bookkeeping_is_retried_after_releasing_capacity(center):
    center.store.update_settings({"gpuSlots": {"0": 1}})
    center.enqueue("A", center.spec("first", group="g"), center.spec("second", group="g", order=1))
    adapter = FlakyExit()
    runner = center.runner(adapters=lambda name: adapter)
    assert runner.tick()["started"] == ["first"]
    center.release("first")
    result = center.tick_until(runner, lambda: center.store.get("first")["bookkeeping"])
    task = center.store.get("first")
    assert task["state"] == "running" and task["lease"] is None
    assert task["bookkeeping"]["hook"] == "on_exit" and task["exit"]["returncode"] == 0
    assert result["started"] == ["second"]  # the exited task no longer holds its slot
    runner.tick()
    task = center.store.get("first")
    assert task["state"] == "succeeded" and task["bookkeeping"] is None
    center.release("second")
    center.tick_until(runner, lambda: center.state("second") == "succeeded")


def test_exclusive_keys_serialize_tasks(center):
    center.enqueue(
        "A",
        center.spec("collect-1", group="g", exclusiveKey="collect:x", request={"lane": "cpu"}),
        center.spec(
            "collect-2", group="g", order=1, exclusiveKey="collect:x", request={"lane": "cpu"}
        ),
        center.spec("other", group="g", order=2, request={"lane": "cpu"}),
    )
    runner = center.runner()
    result = runner.tick()
    assert result["started"] == ["collect-1", "other"]
    assert result["waiting"]["collect-2"] == "Waiting for a related task to finish"
    center.release("collect-1")
    center.tick_until(runner, lambda: center.state("collect-2") == "running")
    for identity in ("collect-2", "other"):
        center.release(identity)
    center.tick_until(runner, lambda: not center.running())


def test_leases_are_visible_to_legacy_schedulers_and_measurements_are_recorded(center, registry):
    center.store.update_settings({"gpuSlots": {"0": 2}})
    center.enqueue(
        "A",
        center.spec(
            "fold", group="g", request={"cpuThreads": 2, "dataWorkers": 1, "workloadKey": "wk"}
        ),
    )
    runner = center.runner()
    runner.tick()
    task = center.store.get("fold")
    lease = registry / task["lease"]
    assert task["lease"] == f"lease-{task['process']['pid']}.json" and lease.exists()
    value = json.loads(lease.read_text())
    assert (value["taskId"], value["cpus"], value["runsPerGpu"], value["gpu"]) == ("fold", 3, 2, 0)
    # Other writers of the registry read it as live load; the runner knows it as its own.
    [read] = leases.read_leases()
    assert (read["taskId"], read["live"], read["gpu"]) == ("fold", True, 0)
    assert leases.foreign([read], {"fold"}) == []
    (center.folder("fold") / "progress.json").write_text(json.dumps({"epoch": 3, "maxEpochs": 9}))
    center.tick_until(
        runner, lambda: (center.store.get("fold")["progress"] or {}).get("epoch") == 3
    )
    task = center.store.get("fold")
    assert task["progress"]["updatedAt"] and task["resources"]["meanConcurrency"] == 1.0
    assert task["resources"]["peakPrivateRamGb"] > 0
    center.release("fold")
    center.tick_until(runner, lambda: center.state("fold") == "succeeded")
    assert not lease.exists()
    [measurement] = center.store.measurements()
    assert measurement["taskId"] == "fold" and measurement["exitReason"] == "ok"
    assert measurement["workloadKey"] == "wk" and measurement["gpuIndex"] == 0
    assert measurement["gpuName"] == "Test GPU" and measurement["gpuUuid"] == "GPU-0"
    assert measurement["wallSeconds"] > 0 and measurement["meanConcurrency"] == 1.0
    assert measurement["peakPrivateRamGb"] > 0 and measurement["meanCpuCores"] is not None


class OutOfMemory(GenericAdapter):
    def on_exit(self, task, exit, ctx):
        if exit.get("returncode"):
            return outcome("failed", "oom", "CUDA out of memory", measurement={"peakVramGb": 5.0})
        return super().on_exit(task, exit, ctx)

    def can_requeue(self, task, ctx):
        return True


def test_out_of_memory_is_retried_once_with_more_memory(center):
    key = center.enqueue("A", center.spec("fold", group="g", request={"vramGb": 2.0}))["owner"][
        "key"
    ]
    # Nothing outside HistoPilot uses the GPU, so the OOM is the task's own.
    runner = center.runner(adapters=lambda name: OutOfMemory(), host_probe=fake_host(free=23.5))
    runner.tick()
    center.store.hold_owner(key, True)  # keeps the retry queued until the next release file
    center.release("fold", 3)
    center.tick_until(runner, lambda: center.store.get("fold")["attempt"] == 2)
    task = center.store.get("fold")
    assert task["state"] == "queued" and task["request"]["vramGb"] == 3.0
    assert task["adapterData"]["oomRetries"] == 1
    (center.folder("fold") / "release").unlink()
    center.store.hold_owner(key, False)
    center.tick_until(runner, lambda: center.state("fold") == "running")
    center.release("fold", 3)
    center.tick_until(runner, lambda: center.state("fold") == "failed")
    task = center.store.get("fold")
    assert task["attempt"] == 2 and task["error"] == "CUDA out of memory"
    assert [row["exitReason"] for row in center.store.measurements()] == ["oom", "oom"]
    assert center.store.measurements()[0]["peakVramGb"] == 5.0


def test_an_oom_beside_memory_used_outside_histopilot_waits_instead_of_spending_its_retry(center):
    key = center.enqueue("A", center.spec("fold", group="g", request={"vramGb": 2.0}))["owner"][
        "key"
    ]
    # 24 GiB GPU, 10 GiB free and no other task: 14 GiB are held by another program.
    host = {"free": 10.0}
    probe = fake_host(free=10.0)
    runner = center.runner(
        adapters=lambda name: OutOfMemory(),
        host_probe=lambda: {
            **probe(),
            "gpus": [{**probe()["gpus"][0], "freeMemoryGb": host["free"]}],
        },
    )
    runner.tick()
    center.store.hold_owner(key, True)
    center.release("fold", 3)
    center.tick_until(runner, lambda: center.store.get("fold")["attempt"] == 2)
    task = center.store.get("fold")
    # Requeued as it was: no bigger request, the OOM retry still unspent.
    assert task["state"] == "queued" and task["request"]["vramGb"] == 2.0
    assert task["adapterData"] == {"contentionRequeues": 1}
    assert center.store.events("fold")[-1]["detail"]["reason"] == "gpu-contention"
    assert any("outside HistoPilot" in line for line in center.logs)
    (center.folder("fold") / "release").unlink()
    (center.folder("fold") / "started").unlink()
    center.store.hold_owner(key, False)
    host["free"] = 23.5  # the other program is gone; the next OOM is the task's own
    center.tick_until(runner, lambda: center.started("fold"))
    center.release("fold", 3)
    center.tick_until(runner, lambda: center.store.get("fold")["attempt"] == 3)
    task = center.store.get("fold")
    assert task["request"]["vramGb"] == 3.0
    assert task["adapterData"] == {"contentionRequeues": 1, "oomRetries": 1}


class DeviceFailure(GenericAdapter):
    def __init__(self, error="CUDA error: unknown error", *, requeue=False):
        self.error = error
        self.requeue = requeue

    def on_exit(self, task, exit, ctx):
        if exit.get("returncode"):
            return outcome("failed", "cuda_failure", self.error)
        return super().on_exit(task, exit, ctx)

    def can_requeue(self, task, ctx):
        return self.requeue


class Probe:
    """A CUDA probe that answers from a script: failures first, then success."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def __call__(self, gpu):
        self.calls.append(gpu)
        return self.results.pop(0) if self.results else ("ok", None)


def test_a_device_loss_fences_the_gpu_until_it_answers_and_the_fence_survives_restarts(center):
    center.enqueue(
        "A",
        center.spec("first", group="g"),
        center.spec("second", group="g", order=1),
        center.spec("third", group="g", order=2),
    )
    clock = [1000.0]
    probe = Probe(("failed", "cuInit: unknown error (999)"))
    options = {
        "adapters": lambda name: DeviceFailure(),
        "clock": lambda: clock[0],
        "cuda_probe": probe,
    }
    runner = center.runner(**options)
    runner.tick()
    assert center.running() == {"first", "second", "third"}
    center.release("first", 1)
    center.tick_until(runner, lambda: center.state("first") == "interrupted")
    # The device failed, not the task: interrupted rather than failed (this adapter
    # declines the resume, so it stays interrupted).
    first = center.store.get("first")
    assert first["exit"]["reason"] == "cuda_failure" and first["bookkeeping"] is None
    assert center.store.runner()["gpuFences"]["0"]["verified"] is False
    # Running work is left alone; queued GPU work waits out the fence.
    center.enqueue("A", center.spec("fourth", group="g", order=3))
    runner.tick()
    fourth = center.store.get("fourth")
    assert fourth["state"] == "queued"
    assert fourth["waitingReason"].startswith("GPU 0 reported a device failure; retrying in 60 s")
    # A task that fails while the GPU is fenced is part of the same incident.
    center.release("second", 1)
    center.tick_until(runner, lambda: center.state("second") == "interrupted")
    assert runner._gpu_fences[0]["failures"] == 1
    # After the fence, a fresh process must initialize CUDA there before work resumes.
    clock[0] += 61
    center.tick_until(runner, lambda: "not usable" in (center.store.get("fourth")["waitingReason"]))
    assert "checking again in 60 s" in center.store.get("fourth")["waitingReason"]
    assert center.state("fourth") == "queued"
    clock[0] += 61
    center.tick_until(runner, lambda: center.state("fourth") == "running")
    assert probe.calls == [0, 0]
    assert center.store.runner()["gpuFences"]["0"]["verified"] is True
    # The next incident after the GPU was verified doubles the fence ...
    center.release("third", 1)
    center.tick_until(runner, lambda: center.state("third") == "interrupted")
    center.enqueue("A", center.spec("fifth", group="g", order=4))
    runner.tick()
    assert "retrying in 120 s" in center.store.get("fifth")["waitingReason"]
    # ... and a restarted runner keeps it.
    runner.close()
    restarted = center.runner(**options)
    restarted.tick()
    assert restarted._gpu_fences[0]["failures"] == 2
    assert "reported a device failure; retrying in" in center.store.get("fifth")["waitingReason"]
    assert center.state("fifth") == "queued"
    clock[0] += 121
    center.tick_until(restarted, lambda: center.state("fifth") == "running")
    # A success on the GPU clears its failure history.
    center.release("fifth")
    center.tick_until(restarted, lambda: center.state("fifth") == "succeeded")
    assert restarted._gpu_fences == {} and center.store.runner()["gpuFences"] == {}
    center.release("fourth")
    center.tick_until(restarted, lambda: center.state("fourth") == "succeeded")


def test_a_device_loss_requeues_the_task_without_spending_a_retry(center):
    center.enqueue("A", center.spec("fold", group="g"))
    clock = [0.0]
    runner = center.runner(
        adapters=lambda name: DeviceFailure(requeue=True),
        clock=lambda: clock[0],
        host_probe=fake_host(free=23.5),
    )
    runner.tick()
    center.release("fold", 1)
    center.tick_until(runner, lambda: center.store.get("fold")["attempt"] == 2)
    task = center.store.get("fold")
    assert task["state"] == "queued" and task["adapterData"]["deviceLossRequeues"] == 1
    assert "oomRetries" not in task["adapterData"]
    assert center.store.events("fold")[-1]["detail"]["reason"] == "device-lost"
    (center.folder("fold") / "release").unlink()
    (center.folder("fold") / "started").unlink()
    runner.tick()
    assert center.state("fold") == "queued"  # the GPU is still fenced
    clock[0] += 61
    center.tick_until(runner, lambda: center.started("fold"))
    center.release("fold")
    center.tick_until(runner, lambda: center.state("fold") == "succeeded")
    # The automatic budget is bounded; a manual retry starts a new one.
    limit = runner_module.DEVICE_LOSS_MAX_REQUEUES
    assert runner._device_loss_retry({"adapterData": {"deviceLossRequeues": limit - 1}})
    assert not runner._device_loss_retry({"adapterData": {"deviceLossRequeues": limit}})
    center.store.transition("fold", from_states=("succeeded",), to_state="failed")
    center.store.requeue(["fold"], reason="retry")
    assert "deviceLossRequeues" not in center.store.get("fold")["adapterData"]


@pytest.mark.parametrize(
    "error",
    [
        "RuntimeError: CUDA error: device-side assert triggered\nCUDA kernel errors might be "
        "asynchronously reported; check the driver log",
        "RuntimeError: CUDA error: an illegal memory access was encountered",
        "CUDA error: unspecified launch failure",
    ],
)
def test_task_faults_on_the_gpu_fail_the_task_without_fencing_the_gpu(center, error):
    center.store.update_settings({"gpuSlots": {"0": 1}})
    center.enqueue("A", center.spec("broken", group="g"), center.spec("next", group="g", order=1))
    runner = center.runner(adapters=lambda name: DeviceFailure(error))
    runner.tick()
    center.release("broken", 1)
    center.tick_until(runner, lambda: center.state("next") == "running")
    assert center.store.get("broken")["exit"]["reason"] == "cuda_failure"
    assert runner._gpu_fences == {}
    center.release("next")
    center.tick_until(runner, lambda: center.state("next") == "succeeded")


@pytest.mark.parametrize(
    ("error", "lost"),
    [
        ("RuntimeError: CUDA error: unknown error", True),
        ("RuntimeError: CUDA error: the device has been lost", True),
        ("the device is lost", True),
        ("cudaErrorUnknown", True),
        ("Unable to determine the device handle for GPU 0000:01:00.0: GPU is lost", True),
        ("GPU has fallen off the bus", True),
        ("CUDA error: CUDA-capable device(s) is/are busy or unavailable", True),
        ("CUDA error: initialization error", True),
        ("Failed to initialize NVML: Driver/library version mismatch", True),
        ("CUDA error: device-side assert triggered", False),
        ("CUDA error: an illegal memory access was encountered", False),
        ("CUDA error: unspecified launch failure", False),
        (None, False),
    ],
)
def test_device_loss_covers_every_lost_device_the_worker_reports(error, lost):
    assert runner_module.device_lost(error) is lost


class Busy(GenericAdapter):
    def on_exit(self, task, exit, ctx):
        if exit.get("returncode") == 75:
            return outcome("requeue", "busy")
        return super().on_exit(task, exit, ctx)


def test_a_busy_task_backs_off_before_it_is_spawned_again(center):
    center.enqueue("A", center.spec("collect", group="g"))
    clock = [0.0]
    runner = center.runner(adapters=lambda name: Busy(), clock=lambda: clock[0])
    runner.tick()
    center.release("collect", 75)
    center.tick_until(runner, lambda: center.store.get("collect")["attempt"] == 2)
    (center.folder("collect") / "release").unlink()
    (center.folder("collect") / "started").unlink()
    runner.tick()
    assert center.state("collect") == "queued"
    assert center.store.get("collect")["waitingReason"] == BUSY_WAIT
    clock[0] += 5.1
    center.tick_until(runner, lambda: center.started("collect"))
    center.release("collect", 75)
    center.tick_until(runner, lambda: center.store.get("collect")["attempt"] == 3)
    (center.folder("collect") / "release").unlink()
    clock[0] += 5.1  # the second busy exit doubled the wait
    runner.tick()
    assert center.store.get("collect")["waitingReason"] == BUSY_WAIT
    clock[0] += 5.0
    center.tick_until(runner, lambda: center.state("collect") == "running")
    center.release("collect")
    center.tick_until(runner, lambda: center.state("collect") == "succeeded")
    assert runner._busy == {}


class BusyProject(GenericAdapter):
    def on_exit(self, task, exit, ctx):
        if exit.get("returncode") == 75:
            return outcome("requeue", "busy", "Waiting for the project to become idle.")
        return super().on_exit(task, exit, ctx)


def test_a_task_busy_again_and_again_backs_off_longer_and_says_why(center):
    """An export waiting through hours of training must not relaunch every minute."""
    center.enqueue("A", center.spec("export", group="g"))
    clock = [0.0]
    runner = center.runner(adapters=lambda name: BusyProject(), clock=lambda: clock[0])
    runner.tick()
    runner._busy["export"] = {"count": 9, "until": 0.0, "reason": None}  # nine tries so far
    center.release("export", 75)
    center.tick_until(runner, lambda: center.store.get("export")["attempt"] == 2)
    (center.folder("export") / "release").unlink()
    (center.folder("export") / "started").unlink()
    runner.tick()
    assert center.store.get("export")["waitingReason"] == (
        "Waiting for the project to become idle (10 tries so far; trying again later)"
    )
    clock[0] += 599.0  # a minute used to be the longest wait
    runner.tick()
    assert center.state("export") == "queued" and not center.started("export")
    clock[0] += 1.1
    center.tick_until(runner, lambda: center.started("export"))


class Preparing(GenericAdapter):
    def prepare(self, task, ctx):
        if task["id"] == "done":
            return {"skip": {"state": "succeeded", "exitReason": "already-complete"}}
        if task["id"] == "broken":
            raise AdapterError("plan is invalid", fatal=True)
        if task["id"] == "busy":
            raise AdapterError("Project is busy; retrying")
        return {"request": {**task["request"], "vramGb": 2.5}}

    def on_started(self, task, identity, gpu, ctx):
        if task["id"] == "rejected":
            raise AdapterError("state.json is unreadable", transient=False)


def test_adapter_prepare_and_start_hooks(center, tmp_path):
    center.enqueue(
        "A",
        center.spec("done", group="g"),
        center.spec("broken", group="g", order=1),
        center.spec("busy", group="g", order=2),
        center.spec("rejected", group="g", order=3, request={"graceSeconds": 0.2}),
        center.spec("refined", group="g", order=4),
    )
    unstartable = center.spec("unstartable", group="g", order=5)
    unstartable["command"]["cwd"] = str(tmp_path / "missing")
    center.enqueue("A", unstartable)
    runner = center.runner(adapters=lambda name: Preparing())
    result = runner.tick()
    assert center.store.get("done")["exit"] == {"reason": "already-complete", "error": None}
    assert center.state("done") == "succeeded" and center.store.get("done")["startedAt"] is None
    assert center.store.get("broken")["error"] == "plan is invalid"
    assert center.state("broken") == "failed"
    assert center.store.get("busy")["waitingReason"] == "Project is busy; retrying"
    assert center.store.get("refined")["request"]["vramGb"] == 2.5
    assert center.state("unstartable") == "failed"
    assert "Cannot start the task" in center.store.get("unstartable")["error"]
    assert set(result["started"]) == {"rejected", "refined"}
    assert center.state("rejected") == "stopping"
    center.tick_until(runner, lambda: center.state("rejected") == "failed")
    assert center.store.get("rejected")["error"] == "state.json is unreadable"
    center.release("refined")
    center.tick_until(runner, lambda: center.state("refined") == "succeeded")


def test_paused_queue_admits_nothing_and_a_second_runner_is_refused(center):
    center.store.update_settings({"paused": True})
    center.enqueue("A", center.spec("fold", group="g"))
    runner = center.runner()
    assert runner.tick()["started"] == []
    assert runner_alive(runner.lock_path) is True
    second = Runner(center.store, host_probe=fake_host(), log=center.logs.append)
    second.lock_wait = 0.1
    with pytest.raises(RunnerBusy):
        second.start()
    row = center.store.runner()
    assert row["pid"] == os.getpid() and row["state"] == "running" and row["codeHash"]
    assert row["sample"]["gpus"][0]["name"] == "Test GPU"
    runner.run_forever(interval=0.01, stop=lambda: True)
    assert runner_alive(runner.lock_path) is False
    assert center.store.runner()["state"] == "stopped"
    center.store.update_settings({"paused": False})
    restarted = center.runner()
    assert restarted.tick()["started"] == ["fold"]
    center.tick_until(restarted, lambda: center.started("fold"))
    os.kill(center.store.get("fold")["process"]["pid"], signal.SIGTERM)
    center.tick_until(restarted, lambda: center.state("fold") == "succeeded")
    assert (center.folder("fold") / "terminated").exists()


def test_oom_backoff_is_capped_by_the_largest_gpu_and_fails_what_cannot_fit(center):
    # 24 GiB GPU with the default reserve max(1, 5%) = 1.2 GiB: 22.8 GiB is the most any
    # task can ever be admitted with.
    center.store.update_settings({"gpuSlots": {"0": 2}})
    key = center.enqueue("A", center.spec("big", group="g", request={"vramGb": 16.0}))["owner"][
        "key"
    ]
    center.enqueue("B", center.spec("huge", group="h", request={"vramGb": 22.8}))
    runner = center.runner(adapters=lambda name: OutOfMemory(), host_probe=fake_host(free=23.5))
    assert runner.tick()["started"] == ["big"]
    center.store.hold_owner(key, True)  # keeps the retry queued
    center.release("big", 3)
    center.tick_until(runner, lambda: center.store.get("big")["attempt"] == 2)
    big = center.store.get("big")
    assert big["state"] == "queued" and big["request"]["vramGb"] == 22.8  # not 16 * 1.5 = 24
    assert big["adapterData"]["oomRetries"] == 1
    center.tick_until(runner, lambda: center.started("huge"))
    center.release("huge", 3)
    center.tick_until(runner, lambda: center.state("huge") == "failed")
    huge = center.store.get("huge")
    assert huge["attempt"] == 1 and huge["bookkeeping"] is None
    assert huge["error"].startswith("This task needs more GPU memory than this machine has")
    assert "22.8 GiB requested" in huge["error"] and huge["exit"]["reason"] == "oom"
    assert "requeued huge" not in " ".join(center.logs)


def live_task_processes(center, task_id):
    """Live processes of this test's task ``task_id`` (other test runs may share task ids)."""
    folder = str(center.folder(task_id)).encode()
    pids = []
    for proc in os.scandir("/proc"):
        if not proc.name.isdigit():
            continue
        try:
            with open(f"/proc/{proc.name}/environ", "rb") as stream:
                environ = stream.read().split(b"\0")
            with open(f"/proc/{proc.name}/cmdline", "rb") as stream:
                argv = stream.read().split(b"\0")
            with open(f"/proc/{proc.name}/stat") as stream:
                state = stream.read().rsplit(")", 1)[1].split()[0]
        except OSError:
            continue
        if (
            state not in {"Z", "X"}
            and f"HISTOPILOT_TASK_ID={task_id}".encode() in environ
            and folder in argv
        ):
            pids.append(int(proc.name))
    return pids


def failing_start(store, *, times):
    """Make the queued->running transition raise, as a locked or full SQLite store does."""
    original = store.transition
    calls = {"left": times}

    def transition(task_id, **fields):
        if fields.get("to_state") == "running" and calls["left"]:
            calls["left"] -= 1
            raise StorageError(
                "The Task Center store is unavailable: database is locked",
                "TASK_CENTER_UNAVAILABLE",
                503,
            )
        return original(task_id, **fields)

    store.transition = transition


def recording_spawner(children):
    def spawner(command, **options):
        child = procs.spawn(command, **options)
        children.append(child)
        return child

    return spawner


def test_a_start_that_cannot_be_recorded_leaves_no_process_behind(center, registry):
    center.enqueue("A", center.spec("fold", group="g"))
    failing_start(center.store, times=1)
    children = []
    runner = center.runner(spawner=recording_spawner(children))
    assert runner.tick()["started"] == []
    task = center.store.get("fold")
    assert task["state"] == "queued" and task["process"] is None
    assert task["waitingReason"].startswith("Cannot record the task start")
    [first] = children
    assert first.returncode == -signal.SIGKILL  # killed and reaped, never left untracked
    assert list(registry.glob("lease-*.json")) == []
    assert live_task_processes(center, "fold") == []
    assert runner.tick()["started"] == ["fold"]
    center.tick_until(runner, lambda: center.started("fold"))
    assert live_task_processes(center, "fold") == [children[1].pid]
    center.release("fold")
    center.tick_until(runner, lambda: center.state("fold") == "succeeded")


def test_a_store_that_keeps_failing_never_accumulates_task_processes(center):
    center.enqueue("A", center.spec("fold", group="g"))
    failing_start(center.store, times=10)
    children = []
    runner = center.runner(spawner=recording_spawner(children))
    for _ in range(4):
        runner.tick()
        assert live_task_processes(center, "fold") == []
    assert len(children) == 4 and all(child.returncode is not None for child in children)
    assert center.state("fold") == "queued"


def orphan(center, identity, *, gpu=0, supervisor=None):
    """A task process a runner started but never recorded (it died in between)."""
    task = center.store.get(identity)
    child = procs.spawn(task["command"], lane=task["request"]["lane"], gpu=gpu, task=task)
    lease = leases.write_task_lease(
        task,
        procs.identity(child.pid),
        gpu,
        cpus=1,
        ram_gb=0.5,
        runs_per_gpu=4,
        supervisor=supervisor or {"pid": 999_998, "startTicks": 1, "bootId": "an-earlier-boot"},
    )
    return child, lease


def test_a_live_untracked_process_of_a_queued_task_is_adopted_not_started_again(center):
    center.enqueue("A", center.spec("fold", group="g"))
    child, lease = orphan(center, "fold")
    try:
        while not center.started("fold"):
            time.sleep(0.02)
        children = []
        runner = center.runner(spawner=recording_spawner(children))
        task = center.store.get("fold")
        assert task["state"] == "running" and task["process"]["pid"] == child.pid
        assert (task["lease"], task["gpu"]) == (lease, 0)
        assert center.store.events("fold")[-1]["detail"]["adopted"] == child.pid
        for _ in range(3):
            runner.tick()
        assert children == [] and live_task_processes(center, "fold") == [child.pid]
        center.release("fold")
        child.wait(timeout=10)
        center.tick_until(runner, lambda: center.state("fold") != "running")
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


def test_a_process_of_another_live_runner_is_neither_adopted_nor_stopped(center):
    # Another state directory's runner shares this lease registry and a task id.
    center.enqueue("A", center.spec("fold", group="g"))
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    child, _ = orphan(center, "fold", supervisor=procs.identity(other.pid))
    try:
        while not center.started("fold"):
            time.sleep(0.02)
        children = []
        runner = center.runner(spawner=recording_spawner(children))
        for _ in range(3):
            runner.tick()
        task = center.store.get("fold")
        assert (task["state"], task["waitingReason"]) == (
            "queued",
            runner_module.EARLIER_PROCESS_WAIT,
        )
        assert children == [] and child.poll() is None
        other.kill()
        other.wait(timeout=10)
        runner.tick()  # its supervisor is gone: this store's runner adopts the process
        task = center.store.get("fold")
        assert task["state"] == "running" and task["process"]["pid"] == child.pid
        center.release("fold")
        child.wait(timeout=10)
        center.tick_until(runner, lambda: center.state("fold") != "running")
        assert children == []
    finally:
        for process in (child, other):
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)


def test_an_untracked_process_of_a_finished_task_is_stopped_and_counts_as_load(center, registry):
    center.store.update_settings({"gpuSlots": {"0": 1}, "cancelGraceSeconds": 5})
    center.enqueue("A", center.spec("old", group="g", mode="ignore"))
    assert center.store.cancel_pending(["old"]) == ["old"]
    child, lease = orphan(center, "old")
    try:
        while not center.started("old"):
            time.sleep(0.02)
        center.enqueue("B", center.spec("next", group="h"))
        clock = [0.0]
        runner = center.runner(clock=lambda: clock[0])
        runner.tick()
        assert center.state("old") == "cancelled" and center.state("next") == "queued"
        # Its lease stays and occupies the GPU slot until the process is gone.
        assert center.store.get("next")["waitingReason"] == "Waiting for a GPU slot (1/1)"
        assert (registry / lease).exists() and child.poll() is None
        clock[0] += 6  # SIGTERM is ignored: killed after the grace period
        runner.tick()
        child.wait(timeout=10)
        assert child.returncode == -signal.SIGKILL
        center.tick_until(runner, lambda: center.state("next") == "running")
        assert not (registry / lease).exists()
        center.release("next")
        center.tick_until(runner, lambda: center.state("next") == "succeeded")
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


class PauseThenCancel(GenericAdapter):
    """Sees a pause, while a cancel of the same task commits during its classification."""

    def __init__(self):
        self.stops = []

    def on_exit(self, task, exit, ctx):
        self.stops.append(exit["stopReason"])
        if exit["stopReason"] == "pause":
            ctx.store.request_stop([task["id"]], "cancel")
        return super().on_exit(task, exit, ctx)


def test_a_cancel_during_a_paused_exit_cancels_instead_of_requeueing(center):
    key = center.enqueue("A", center.spec("fold", group="g"))["owner"]["key"]
    adapter = PauseThenCancel()
    runner = center.runner(adapters=lambda name: adapter)
    center.tick_until(runner, lambda: center.started("fold"))
    center.store.hold_owner(key, True)
    center.store.request_stop(["fold"], "pause")
    center.tick_until(runner, lambda: center.state("fold") != "stopping")
    task = center.store.get("fold")
    assert (task["state"], task["attempt"]) == ("cancelled", 1)
    assert task["exit"]["reason"] == "cancelled" and task["bookkeeping"] is None
    assert adapter.stops == ["pause", "cancel"]


class DeferredRequeue(GenericAdapter):
    def __init__(self):
        self.requeues = 0

    def on_requeue(self, task, ctx):
        self.requeues += 1
        raise AdapterError("PROJECT_BUSY", transient=True)


def test_a_cancel_reaches_a_task_awaiting_its_requeue(center):
    collect = center.spec(
        "collect",
        group="g",
        order=1,
        request={"lane": "cpu"},
        dependsOn=[{"task": "fold", "condition": "terminal"}],
    )
    center.enqueue("A", center.spec("fold", group="g"), collect)
    adapter = DeferredRequeue()
    runner = center.runner(adapters=lambda name: adapter)
    center.tick_until(runner, lambda: center.started("fold"))
    center.store.request_stop(["fold"], "pause")
    center.tick_until(runner, lambda: center.state("fold") == "interrupted")
    runner.tick()
    task = center.store.get("fold")
    assert task["bookkeeping"]["hook"] == "on_requeue" and adapter.requeues >= 1
    # Not finished for its dependents while it may run again.
    assert center.state("collect") == "blocked"
    assert center.client.group("batch", "g", center.project)["live"] == 2
    assert center.client.cancel_group("batch", "g", center.project) == {
        "cancelled": ["collect"],
        "stopping": ["fold"],
    }
    requeues = adapter.requeues
    runner.tick()
    task = center.store.get("fold")
    assert (task["state"], task["attempt"]) == ("cancelled", 1)
    assert task["bookkeeping"] is None and task["exit"]["stopReason"] == "cancel"
    assert adapter.requeues == requeues


def lost_task(center, identity, **extra):
    center.store.transition(
        identity,
        from_states="queued",
        to_state="running",
        started_at="2026-01-01T00:00:00+00:00",
        process={"pid": 999_999, "startTicks": 1, "bootId": "an-earlier-boot"},
        gpu=0,
        **extra,
    )


class UndecidedProbe(GenericAdapter):
    """can_requeue raises a transient error until ``ready``, like a failing runtime probe."""

    def __init__(self, failures=10**6):
        self.failures = failures
        self.calls = 0

    def can_requeue(self, task, ctx):
        self.calls += 1
        if self.calls <= self.failures:
            raise AdapterError("the training runtime probe failed", transient=True)
        return True


def test_an_undecided_auto_resume_is_retried_with_backoff_and_holds_back_dependents(center):
    collect = center.spec(
        "collect",
        group="g",
        order=1,
        request={"lane": "cpu"},
        dependsOn=[{"task": "fold", "condition": "terminal"}],
    )
    center.enqueue("A", center.spec("fold", group="g"), collect)
    lost_task(center, "fold")
    adapter = UndecidedProbe(failures=2)
    clock = [0.0]
    runner = center.runner(adapters=lambda name: adapter, clock=lambda: clock[0])
    task = center.store.get("fold")
    assert task["state"] == "interrupted" and task["bookkeeping"]["hook"] == "requeue_intent"
    assert task["bookkeeping"]["reason"] == "auto-resume" and adapter.calls == 1
    clock[0] += 29
    runner.tick()
    assert adapter.calls == 1 and center.state("collect") == "blocked"
    clock[0] += 2  # 30 s after the first attempt
    runner.tick()
    assert adapter.calls == 2 and center.state("fold") == "interrupted"
    clock[0] += 59  # the backoff doubled to 60 s
    runner.tick()
    assert adapter.calls == 2
    clock[0] += 2
    runner.tick()
    task = center.store.get("fold")
    assert (task["attempt"], task["bookkeeping"]) == (2, None)
    assert task["state"] in {"queued", "running"}
    assert any((event["detail"] or {}).get("autoResumed") for event in center.store.events("fold"))
    assert center.state("collect") == "blocked"  # its dependency runs again first


def test_an_auto_resume_that_stays_undecided_gives_up_after_two_hours(center):
    collect = center.spec(
        "collect",
        group="g",
        order=1,
        request={"lane": "cpu"},
        dependsOn=[{"task": "fold", "condition": "terminal"}],
    )
    center.enqueue("A", center.spec("fold", group="g"), collect)
    lost_task(center, "fold")
    moment = [datetime(2026, 9, 1, tzinfo=UTC)]
    clock = [0.0]
    runner = center.runner(
        adapters=lambda name: UndecidedProbe(),
        now=lambda: moment[0].isoformat(),
        clock=lambda: clock[0],
    )
    assert center.store.get("fold")["bookkeeping"]["hook"] == "requeue_intent"
    moment[0] += timedelta(hours=2, seconds=1)
    clock[0] += 3600
    runner.tick()
    task = center.store.get("fold")
    assert (task["state"], task["bookkeeping"]) == ("interrupted", None)
    assert task["error"] == "Auto-resume gave up: the training runtime probe failed"
    runner.tick()
    assert center.state("collect") in {"queued", "running"}


class DiesWhileDeciding(GenericAdapter):
    armed = True

    def can_requeue(self, task, ctx):
        if DiesWhileDeciding.armed:
            DiesWhileDeciding.armed = False
            raise KeyboardInterrupt("the runner is killed while probing the runtime")
        return True


def test_an_auto_resume_intent_survives_a_runner_that_dies_while_deciding(center):
    center.enqueue("A", center.spec("fold", group="g"))
    lost_task(center, "fold")
    adapter = DiesWhileDeciding()
    with pytest.raises(KeyboardInterrupt):
        center.runner(adapters=lambda name: adapter)
    task = center.store.get("fold")
    assert task["state"] == "interrupted" and task["bookkeeping"]["hook"] == "requeue_intent"
    center.runner(adapters=lambda name: adapter)  # the next runner resumes the intent
    task = center.store.get("fold")
    assert (task["state"], task["attempt"], task["bookkeeping"]) == ("queued", 2, None)


def test_a_stop_during_startup_leaves_unreconciled_tasks_to_the_next_runner(center):
    center.enqueue(
        "A",
        center.spec("first", group="g", adapterData={"requeueSafe": True}),
        center.spec("second", group="g", order=1, adapterData={"requeueSafe": True}),
    )
    lost_task(center, "first")
    lost_task(center, "second")
    stop = threading.Event()
    stop.set()
    center.runner(stop=stop).shutdown()
    assert {center.state("first"), center.state("second")} == {"running"}
    center.runner()
    assert {center.state("first"), center.state("second")} == {"queued"}


def test_a_stop_during_startup_leaves_undecided_requeues_to_the_next_runner(center):
    center.enqueue("A", center.spec("first", group="g"), center.spec("second", group="g", order=1))
    for identity in ("first", "second"):
        lost_task(center, identity)
        assert (
            center.store.conclude(
                identity,
                to_state="interrupted",
                stop_request=None,
                follow_up={
                    "hook": "requeue_intent",
                    "reason": "auto-resume",
                    "since": "2026-01-01T00:00:00+00:00",
                },
                exit={"reason": "lost"},
            )
            == "concluded"
        )
    adapter = UndecidedProbe(failures=0)
    center.runner(adapters=lambda name: adapter, stop=lambda: adapter.calls >= 1).shutdown()
    assert adapter.calls == 1
    assert sorted(center.state(identity) for identity in ("first", "second")) == [
        "interrupted",
        "queued",
    ]
    center.runner(adapters=lambda name: adapter)
    assert {center.state("first"), center.state("second")} == {"queued"}


def test_admission_holds_the_registry_lock_against_legacy_schedulers(center, registry):
    center.enqueue("A", center.spec("fold", group="g", request={"vramGb": 4.0}))
    legacy = []
    armed = [True]

    def reader():
        seen = leases.read_leases()
        if armed:
            armed.clear()
            # Another writer (a legacy scheduler of another checkout) dispatches right after
            # the runner's lock-free read, holding the registry lock as every writer does.
            with leases.registry_lock() as folder:
                child = subprocess.Popen(
                    [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
                )
                legacy.append(child)
                (folder / f"lease-{child.pid}.json").write_text(
                    json.dumps(
                        {
                            "process": procs.identity(child.pid),
                            "processGroupId": child.pid,
                            "gpu": 0,
                            "cpus": 4,
                            "ramGb": 8.0,
                            "runsPerGpu": 1,
                            "batchId": "legacy",
                            "runId": "r1",
                        }
                    )
                )
        return seen

    def spawner(command, **options):
        # Legacy writers cannot take the registry between the decision and the lease.
        with pytest.raises(StorageError) as error:
            with leases.registry_lock(timeout=0):
                pass
        assert error.value.code == "PROJECT_BUSY"
        return procs.spawn(command, **options)

    runner = center.runner(lease_reader=reader, spawner=spawner)
    try:
        result = runner.tick()
        assert result["started"] == []
        assert result["waiting"]["fold"] == "GPU 0 is reserved exclusively by another job"
        legacy[0].kill()
        legacy[0].wait(timeout=5)
        center.tick_until(runner, lambda: center.state("fold") == "running")
        center.release("fold")
        center.tick_until(runner, lambda: center.state("fold") == "succeeded")
    finally:
        for child in legacy:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)


def test_the_runner_prunes_leases_of_dead_workers_every_few_minutes(center, registry):
    registry.mkdir(mode=0o700)
    me = procs.identity(os.getpid())

    def lease(name, identity, **extra):
        values = {"process": identity, "gpu": 0, "cpus": 4, "ramGb": 8.0, "runsPerGpu": 1}
        (registry / name).write_text(json.dumps({**values, **extra}))

    dead = {"pid": 999_999, "startTicks": 1, "bootId": "an-earlier-boot"}
    lease("lease-999999.json", dead, batchId="crashed")
    lease(f"lease-{me['pid']}.json", me, processGroupId=me["pid"], batchId="alive")
    lease("lease-123456-preparation-0.json", {**dead, "pid": 123_456}, supervisor=me)
    clock = [1000.0]
    runner = center.runner(clock=lambda: clock[0])
    runner.tick()
    names = {path.name for path in registry.glob("lease-*.json")}
    assert names == {f"lease-{me['pid']}.json", "lease-123456-preparation-0.json"}
    assert any(
        "pruned 1 lease(s) of dead workers: lease-999999.json" in line for line in center.logs
    )
    # Another crash is pruned only once the interval has passed.
    lease("lease-999998.json", {**dead, "pid": 999_998})
    clock[0] += runner_module.LEASE_PRUNE_SECONDS / 2
    runner.tick()
    assert (registry / "lease-999998.json").exists()
    clock[0] += runner_module.LEASE_PRUNE_SECONDS
    with leases.registry_lock():
        runner.tick()  # a busy registry waits for the next interval, quietly
    assert (registry / "lease-999998.json").exists()
    assert not any("lease pruning" in line for line in center.logs)
    clock[0] += runner_module.LEASE_PRUNE_SECONDS
    runner.tick()
    assert not (registry / "lease-999998.json").exists()
    assert (registry / f"lease-{me['pid']}.json").exists()

    def unsafe(**_options):
        raise StorageError("Training resource registry is unsafe.", "TRAINING_REGISTRY_UNSAFE")

    runner.lease_pruner = unsafe
    clock[0] += runner_module.LEASE_PRUNE_SECONDS
    runner.tick()  # logged; the tick goes on
    assert any("lease pruning failed" in line for line in center.logs)


def test_a_busy_registry_or_an_unwritable_lease_starts_nothing(center, registry):
    center.enqueue("A", center.spec("fold", group="g"))
    children = []
    runner = center.runner(
        spawner=recording_spawner(children),
        lease_lock=lambda: leases.registry_lock(timeout=0.05),
    )
    with leases.registry_lock():
        result = runner.tick()
    assert children == [] and result["waiting"]["fold"] == "Waiting for the resource lease registry"
    assert center.store.get("fold")["waitingReason"] == "Waiting for the resource lease registry"

    def unwritable(*args, **kwargs):
        raise OSError("No space left on device")

    runner.lease_writer = unwritable
    result = runner.tick()
    assert result["started"] == [] and len(children) == 1
    assert children[0].returncode == -signal.SIGKILL
    assert center.store.get("fold")["waitingReason"].startswith(
        "Cannot publish the task's resource lease"
    )
    assert center.state("fold") == "queued" and list(registry.glob("lease-*.json")) == []
    runner.lease_writer = leases.write_task_lease
    center.tick_until(runner, lambda: center.state("fold") == "running")
    center.release("fold")
    center.tick_until(runner, lambda: center.state("fold") == "succeeded")


def test_foreign_load_in_measurements_is_refreshed_while_nothing_is_queued(center):
    foreign = {"file": "lease-999.json", "taskId": None, "kind": None, "gpu": 0, "cpus": 1}
    present = [True]

    def reader():
        rows = leases.read_leases()
        if present:
            rows.append({**foreign, "ramGb": 0.0, "runsPerGpu": 4, "live": True})
        return rows

    runner = center.runner(lease_reader=reader)
    center.enqueue("A", center.spec("fold", group="g"))
    runner.tick()
    assert center.store.get("fold")["resources"]["concurrency"] == 2
    present.clear()  # the legacy run finished; nothing is queued and admission is paused
    center.store.update_settings({"paused": True})
    runner.tick()
    assert center.store.get("fold")["resources"]["concurrency"] == 1
    center.store.update_settings({"paused": False})
    center.release("fold")
    center.tick_until(runner, lambda: center.state("fold") == "succeeded")


def test_code_hash_covers_exactly_the_given_files(tmp_path):
    first, second, missing = (tmp_path / name for name in ("a.py", "b.py", "gone.py"))
    first.write_text("a = 1\n")
    second.write_text("b = 2\n")
    files = [str(first), str(second), str(missing)]
    value = runner_module.code_hash(files)
    assert value == runner_module.code_hash(list(reversed(files)))
    missing.write_text("")
    assert runner_module.code_hash(files) != value  # an empty file is not an absent one
    missing.unlink()
    assert runner_module.code_hash(files) == value
    second.write_text("b = 3\n")
    assert runner_module.code_hash(files) != value
    assert runner_module.code_hash() == runner_module.code_hash(None)


def test_the_runner_records_every_histopilot_module_it_runs(center, monkeypatch):
    monkeypatch.setattr(runner_module, "_LOADED_AT", float("inf"))  # edits by others
    runner = center.runner()
    runner.tick()
    row = center.store.runner()
    from histopilot.workers import training_process

    assert str(Path(training_process.__file__).resolve()) in row["codeFiles"]
    assert str(Path(runner_module.__file__).resolve()) in row["codeFiles"]
    assert all(Path(name).is_absolute() for name in row["codeFiles"])
    assert row["codeHash"] == runner_module.code_hash(row["codeFiles"])
    assert runner_module.code_current(row) is True
    # A module file changed after the runner loaded its code: the running code is unknown.
    monkeypatch.setattr(runner_module, "_LOADED_AT", 0.0)
    runner._record_code(force=True)
    row = center.store.runner()
    assert row["codeHash"].startswith("changed-after-start-")
    assert runner_module.code_current(row) is False


def test_the_runner_truncates_the_store_log_once_per_interval(center, monkeypatch):
    calls = []
    monkeypatch.setattr(center.store, "checkpoint", lambda: calls.append(1) or True)
    clock = [0.0]
    runner = center.runner(clock=lambda: clock[0])
    runner.tick()
    runner.tick()
    assert len(calls) == 1
    clock[0] += runner_module.CHECKPOINT_SECONDS + 1
    runner.tick()
    assert len(calls) == 2
