"""Task store schema, enqueue, dependency, transition and settings semantics."""

import sqlite3
import sys

import pytest

from histopilot.storage.project_lock import StorageError
from histopilot.taskcenter import TaskCenterClient, TaskStore, default_client, ids, paths
from histopilot.taskcenter.model import normalize_command, normalize_request


def owner(tmp_path, identity="A", kind="experiment"):
    return {
        "kind": kind,
        "id": identity,
        "projectId": "project-1",
        "projectFolder": str(tmp_path / "project"),
        "title": f"Experiment {identity}",
    }


def spec(tmp_path, identity, *, group="g1", order=0, **extra):
    return {
        "id": identity,
        "kind": "mil-fold",
        "adapter": "generic",
        "title": identity,
        "group": {"kind": "mil-batch", "id": group},
        "planOrder": order,
        "request": {"lane": "gpu", "vramGb": 1.0},
        "command": {
            "argv": [sys.executable, "-c", "pass"],
            "cwd": str(tmp_path),
            "log": str(tmp_path / f"{identity}.log"),
        },
        **extra,
    }


@pytest.fixture
def store(tmp_path):
    return TaskStore(tmp_path / "state" / "task-center.sqlite")


def test_schema_uses_wal_and_refuses_a_newer_store(store):
    store.initialize()
    connection = sqlite3.connect(store.path)
    try:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute(
            "SELECT value FROM meta WHERE key='schemaVersion'"
        ).fetchone() == ("1",)
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master")}
        assert {"owners", "task_groups", "tasks", "task_dependencies", "task_events"} <= tables
        assert {"operations", "settings", "measurements", "runner"} <= tables
        connection.execute("UPDATE meta SET value='2' WHERE key='schemaVersion'")
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(StorageError) as error:
        TaskStore(store.path).initialize()
    assert (error.value.code, error.value.status_code) == ("TASK_CENTER_STORE_NEWER", 409)


def test_default_store_lives_in_the_private_state_directory(tmp_path, monkeypatch):
    state = paths.state_dir()
    assert state == (tmp_path / "histopilot-state").resolve()
    assert state.stat().st_mode & 0o777 == 0o700
    assert TaskStore().path == state / "task-center.sqlite"
    assert default_client() is default_client()
    assert default_client().store.path == state / "task-center.sqlite"
    state.chmod(0o777)
    with pytest.raises(StorageError) as error:
        paths.state_dir()
    assert error.value.code == "TASK_CENTER_STATE_UNSAFE"
    state.chmod(0o700)
    monkeypatch.delenv(paths.STATE_ENV)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg"))
    assert paths.state_dir() == (tmp_path / "xdg" / "histopilot").resolve()


def test_normalization_fills_defaults_and_rejects_bad_values(tmp_path):
    request = normalize_request({"lane": "gpu", "vramGb": 2})
    assert request["cpuThreads"] == 2 and request["ramGb"] == 4.0 and request["vramGb"] == 2.0
    assert request["service"] is False and request["graceSeconds"] is None
    with pytest.raises(StorageError):
        normalize_request({"cpuThreads": True})
    with pytest.raises(StorageError):
        normalize_request({"lane": "tpu"})
    command = normalize_command(
        {"argv": ["python"], "cwd": str(tmp_path), "log": str(tmp_path / "x.log"), "env": {"T": 2}}
    )
    assert command["env"] == {"T": "2"} and command["progress"] is None
    with pytest.raises(StorageError):
        normalize_command({"argv": ["python"], "cwd": "relative", "log": str(tmp_path / "x")})


def test_enqueue_is_idempotent_and_orders_by_priority_owner_group_and_plan(store, tmp_path):
    first = store.enqueue(
        owner(tmp_path, "A"),
        [spec(tmp_path, "a2", order=1), spec(tmp_path, "a1", order=0)],
        operation_id="op-a",
    )
    assert first["created"] == ["a2", "a1"] and first["tasks"] == ["a2", "a1"]
    assert first["owner"]["key"] == ids.owner_key("experiment", "A", str(tmp_path / "project"))
    assert (
        store.enqueue(
            owner(tmp_path, "A"),
            [spec(tmp_path, "a2", order=1), spec(tmp_path, "a1", order=0)],
            operation_id="op-a",
        )
        == first
    )
    with pytest.raises(StorageError) as error:
        store.enqueue(owner(tmp_path, "A"), [spec(tmp_path, "a3")], operation_id="op-a")
    assert error.value.code == "OPERATION_CONFLICT"
    again = store.enqueue(owner(tmp_path, "A"), [spec(tmp_path, "a1")])
    assert again["created"] == []
    store.enqueue(owner(tmp_path, "A"), [spec(tmp_path, "a4", group="g2", order=0)])
    store.enqueue(
        owner(tmp_path, "B"),
        [spec(tmp_path, "b1"), spec(tmp_path, "b-collect", order=5, priority="interactive")],
    )
    queue = [task["id"] for task in store.list(states=("queued",))]
    assert queue == ["b-collect", "a1", "a2", "a4", "b1"]
    a4 = store.get("a4")
    assert a4["planOrder"] == 2 * 1_000_000 and store.get("a1")["planOrder"] == 1_000_000
    assert a4["group"] == {"kind": "mil-batch", "id": "g2"}
    assert store.events("a1")[0]["toState"] == "queued"


def test_enqueue_updates_only_pending_tasks(store, tmp_path):
    store.enqueue(owner(tmp_path), [spec(tmp_path, "t")])
    store.enqueue(owner(tmp_path), [{**spec(tmp_path, "t"), "title": "renamed"}])
    assert store.get("t")["title"] == "renamed"
    assert store.transition("t", from_states="queued", to_state="running", started_at="x")
    store.enqueue(owner(tmp_path), [{**spec(tmp_path, "t"), "title": "ignored"}])
    assert store.get("t")["title"] == "renamed"
    assert not store.transition("t", from_states="queued", to_state="running")
    assert store.transition("t", from_states="running", to_state="failed")
    store.enqueue(owner(tmp_path), [{**spec(tmp_path, "t"), "title": "ignored"}])
    task = store.get("t")
    assert task["state"] == "failed" and task["title"] == "renamed" and task["finishedAt"]


def test_enqueue_merges_owner_labels_from_every_service(store, tmp_path):
    labelled = {**owner(tmp_path), "labels": {"experimentId": "A", "batchName": "grid"}}
    key = store.enqueue(labelled, [spec(tmp_path, "fold")])["owner"]["key"]
    store.enqueue(owner(tmp_path), [spec(tmp_path, "refit", group="refit")])
    assert store.owner(key)["labels"] == {"experimentId": "A", "batchName": "grid"}
    store.enqueue({**owner(tmp_path), "labels": {"batchName": "second"}}, [spec(tmp_path, "b2")])
    assert store.owner(key)["labels"] == {"experimentId": "A", "batchName": "second"}


def test_idle_owner_rejoins_the_back_of_the_queue(store, tmp_path):
    store.enqueue(owner(tmp_path, "A"), [spec(tmp_path, "a1")])
    store.enqueue(owner(tmp_path, "B"), [spec(tmp_path, "b1")])
    store.transition("a1", from_states="queued", to_state="succeeded")
    store.enqueue(owner(tmp_path, "A"), [spec(tmp_path, "a2", group="g2")])
    assert [task["id"] for task in store.list(states=("queued",))] == ["b1", "a2"]


def test_dependencies_promote_by_condition_without_cascading(store, tmp_path):
    store.enqueue(
        owner(tmp_path),
        [
            spec(tmp_path, "fold"),
            spec(tmp_path, "needs-success", dependsOn=[{"task": "fold"}]),
            spec(tmp_path, "needs-terminal", dependsOn=[{"task": "fold", "condition": "terminal"}]),
        ],
    )
    assert store.get("needs-success")["state"] == "blocked"
    assert store.get("needs-terminal")["state"] == "blocked"
    assert store.promote_ready() == []
    store.transition("fold", from_states="queued", to_state="failed")
    assert store.promote_ready() == ["needs-terminal"]
    # A task that needs its dependency to succeed can never run now: it is cancelled.
    cancelled = store.get("needs-success")
    assert cancelled["state"] == "cancelled" and "did not succeed (fold)" in cancelled["error"]
    assert cancelled["exit"]["reason"] == "cancelled"
    assert store.events("needs-success")[-1]["detail"] == {
        "event": "dependency-failed",
        "dependency": "fold",
    }
    assert store.dependencies("needs-success") == [
        {"task": "fold", "condition": "succeeded", "state": "failed"}
    ]
    assert store.dependents(["fold"]) == ["needs-success", "needs-terminal"]
    # A retry of both runs them again in order.
    store.requeue(["fold", "needs-success"], reason="retry")
    assert store.get("needs-success")["state"] == "blocked"
    store.transition("fold", from_states="queued", to_state="succeeded")
    assert store.promote_ready() == ["needs-success"]
    with pytest.raises(StorageError) as error:
        store.enqueue(owner(tmp_path), [spec(tmp_path, "x", dependsOn=[{"task": "missing"}])])
    assert error.value.code == "TASK_DEPENDENCY_UNKNOWN"


def test_a_failed_dependency_cancels_its_dependents_in_turn(store, tmp_path):
    store.enqueue(
        owner(tmp_path),
        [
            spec(tmp_path, "a"),
            spec(tmp_path, "b", dependsOn=[{"task": "a"}]),
            spec(tmp_path, "c", dependsOn=[{"task": "b"}]),
            spec(tmp_path, "after", dependsOn=[{"task": "b", "condition": "terminal"}]),
        ],
    )
    # A dependency that may still be requeued has not failed yet for its dependents.
    store.transition(
        "a", from_states="queued", to_state="failed", bookkeeping={"hook": "requeue_intent"}
    )
    assert store.promote_ready() == [] and store.get("b")["state"] == "blocked"
    store.set_fields("a", bookkeeping=None)
    assert store.promote_ready() == ["after"]
    assert [store.get(task)["state"] for task in ("b", "c")] == ["cancelled", "cancelled"]
    assert store.events("c")[-1]["detail"]["dependency"] == "b"
    # Retrying the failed task alone brings back what its failure cancelled, in turn ...
    store.enqueue(owner(tmp_path), [spec(tmp_path, "mine", dependsOn=[{"task": "a"}])])
    store.transition("mine", from_states=("blocked",), to_state="cancelled")  # a user's cancel
    assert store.requeue(["a"], reason="retry") == ["a"]
    assert [store.get(task)["state"] for task in ("a", "b", "c")] == [
        "queued",
        "blocked",
        "blocked",
    ]
    assert store.get("b")["attempt"] == 2 and store.get("b")["error"] is None
    assert store.events("b")[-1]["detail"]["event"] == "dependency-requeued"
    # ... but never what a user cancelled.
    assert store.get("mine")["state"] == "cancelled"
    store.transition("a", from_states=("queued",), to_state="succeeded")
    assert store.promote_ready() == ["b"]


def test_requeue_bumps_the_attempt_and_clears_the_previous_run(store, tmp_path):
    store.enqueue(owner(tmp_path), [spec(tmp_path, "t"), spec(tmp_path, "done")])
    store.transition(
        "t",
        from_states="queued",
        to_state="running",
        started_at="2026-01-01T00:00:00+00:00",
        process={"pid": 1234, "startTicks": 1, "bootId": "b"},
        gpu=0,
        lease="lease-1234.json",
    )
    store.request_stop(["t"], "cancel")
    store.transition(
        "t", from_states="stopping", to_state="interrupted", exit={"reason": "lost"}, error="gone"
    )
    store.transition("done", from_states="queued", to_state="succeeded")
    assert store.requeue(
        ["t", "done"],
        reason="auto-resume",
        request_patch={"vramGb": 3.0},
        adapter_data_patch={"oomRetries": 1},
        manual=False,
    ) == ["t"]
    task = store.get("t")
    assert (task["state"], task["attempt"]) == ("queued", 2)
    for field in ("process", "gpu", "lease", "stopRequest", "exit", "error", "startedAt"):
        assert task[field] is None, field
    assert task["request"]["vramGb"] == 3.0 and task["adapterData"] == {"oomRetries": 1}
    assert store.events("t")[-1]["detail"]["autoResumed"] is True
    # A manual retry resets the automatic retry budgets and keeps everything else.
    store.set_fields("t", adapter_data={"oomRetries": 1, "deviceLossRequeues": 2, "runId": "r"})
    store.transition("t", from_states="queued", to_state="failed")
    assert store.requeue(["t"], reason="retry") == ["t"]
    assert store.get("t")["adapterData"] == {"runId": "r"}
    assert store.requeue(["done"], reason="rerun", include_succeeded=True) == ["done"]
    assert store.get("done")["attempt"] == 2


def test_cancel_pending_and_stop_requests(store, tmp_path):
    store.enqueue(
        owner(tmp_path),
        [
            spec(tmp_path, "run"),
            spec(tmp_path, "wait"),
            spec(tmp_path, "blocked", dependsOn=[{"task": "run"}]),
        ],
    )
    store.transition("run", from_states="queued", to_state="running")
    assert store.cancel_pending(["run", "wait", "blocked"]) == ["wait", "blocked"]
    assert store.get("wait")["exit"] == {"reason": "cancelled"}
    assert store.get("run")["state"] == "running"
    assert store.request_stop(["run", "wait"], "pause") == ["run"]
    task = store.get("run")
    assert (task["state"], task["stopRequest"]) == ("stopping", "pause")
    assert store.request_stop(["run"], "pause") == ["run"]
    assert store.request_stop(["run"], "cancel") == ["run"]
    assert store.get("run")["stopRequest"] == "cancel"
    assert store.request_stop(["run"], "pause") == []


def test_hold_and_move_owners(store, tmp_path):
    for identity in "ABC":
        store.enqueue(owner(tmp_path, identity), [spec(tmp_path, identity.lower())])
    store.enqueue(owner(tmp_path, "D"), [spec(tmp_path, "d")])
    store.transition("d", from_states="queued", to_state="succeeded")

    def order():
        return [item["id"] for item in store.owners()]

    assert order() == ["A", "B", "C"]
    key = {item["id"]: item["key"] for item in store.owners(live_only=False)}
    store.move_owner(key["C"], "top")
    assert order() == ["C", "A", "B"]
    store.move_owner(key["C"], "down")
    assert order() == ["A", "C", "B"]
    store.move_owner(key["A"], "bottom")
    assert order() == ["C", "B", "A"]
    store.move_owner(key["A"], "up")
    assert order() == ["C", "A", "B"]
    assert [task["id"] for task in store.list(states=("queued",))] == ["c", "a", "b"]
    held = store.hold_owner(key["A"], True)
    assert held["held"] is True and held["counts"]["queued"] == 1
    assert store.owners(live_only=False)[-1]["id"] == "D"
    with pytest.raises(StorageError):
        store.move_owner(key["A"], "sideways")


def test_settings_merge_with_defaults_and_validate(store):
    settings = store.settings()
    assert settings["defaultGpuSlots"] == 4 and settings["gpuSlots"] == {}
    assert settings["reserves"] == {"cpuThreads": 2, "ramGb": None, "vramGb": None}
    updated = store.update_settings({"gpuSlots": {"0": 5}, "reserves": {"ramGb": 8}})
    assert updated["gpuSlots"] == {"0": 5}
    assert updated["reserves"] == {"cpuThreads": 2, "ramGb": 8, "vramGb": None}
    store.update_settings({"gpuSlots": {"1": 2}, "paused": True})
    assert store.settings()["gpuSlots"] == {"0": 5, "1": 2}
    assert store.settings()["paused"] is True
    for patch in (
        {"gpuSlots": {"0": 17}},
        {"unknown": 1},
        {"paused": "yes"},
        {"defaults": {"cpuThreadsPerRun": 0}},
    ):
        with pytest.raises(StorageError) as error:
            store.update_settings(patch)
        assert error.value.code == "TASK_CENTER_SETTINGS_INVALID"
    assert store.settings()["gpuSlots"] == {"0": 5, "1": 2}


def test_null_setting_restores_the_automatic_value(store):
    assert store.update_settings({"cpuTaskSlots": 3})["cpuTaskSlots"] == 3
    assert store.update_settings({"cpuTaskSlots": None})["cpuTaskSlots"] is None
    assert store.settings()["cpuTaskSlots"] is None


def test_measurements_operations_runner_row_and_sessions(store, tmp_path):
    store.record_measurement(
        {
            "taskId": "t",
            "attempt": 1,
            "kind": "mil-fold",
            "workloadKey": "k",
            "workload": {"model": "abmil"},
            "peakVramGb": 2.5,
            "exitReason": "ok",
        }
    )
    store.record_measurement({"taskId": "u", "kind": "compute-job", "workloadKey": "other"})
    rows = store.measurements(workload_key="k")
    assert len(rows) == 1 and rows[0]["workload"] == {"model": "abmil"}
    assert rows[0]["peakVramGb"] == 2.5 and rows[0]["wallSeconds"] is None
    assert [row["taskId"] for row in store.measurements()] == ["u", "t"]
    assert store.measurements(kinds=("compute-job",))[0]["taskId"] == "u"
    store.record_operation("op", "hash", {"ok": True})
    store.record_operation("op", "hash", {"ignored": True})
    assert store.operation("op")["result"] == {"ok": True}
    with pytest.raises(StorageError):
        store.record_operation("op", "other", None)
    assert store.runner() is None
    store.write_runner(pid=5, start_ticks=6, boot_id="b", state="running", sample={"at": "now"})
    store.write_runner(heartbeat_at="later")
    runner = store.runner()
    assert (runner["pid"], runner["heartbeatAt"], runner["sample"]) == (5, "later", {"at": "now"})
    store.enqueue(owner(tmp_path), [spec(tmp_path, "s1", sessionName="hp-x")])
    assert store.by_session("hp-x")["id"] == "s1"
    with pytest.raises(StorageError) as error:
        store.enqueue(owner(tmp_path), [spec(tmp_path, "s2", sessionName="hp-x")])
    assert error.value.code == "TASK_CENTER_CONFLICT"


def test_client_group_cancel_and_requeue_group(store, tmp_path):
    client = TaskCenterClient(store)
    folds = [spec(tmp_path, f"fold-{index}", order=index) for index in range(3)]
    collect = spec(
        tmp_path,
        "collect",
        order=9,
        priority="interactive",
        dependsOn=[{"task": item["id"], "condition": "terminal"} for item in folds],
    )
    collect["kind"] = "mil-collect"
    client.enqueue(owner(tmp_path), [*folds, collect])
    project = str(tmp_path / "project")
    group = client.group("mil-batch", "g1", project)
    assert group["live"] == 4 and group["counts"]["blocked"] == 1
    assert group["owner"]["id"] == "A" and group["held"] is False
    store.set_waiting_reasons({"fold-0": "Waiting for a GPU slot (4/4)"})
    assert client.group("mil-batch", "g1", project)["waitingReason"] == (
        "Waiting for a GPU slot (4/4)"
    )
    store.transition("fold-0", from_states="queued", to_state="succeeded")
    store.transition("fold-1", from_states="queued", to_state="running")
    result = client.cancel_group("mil-batch", "g1", project, exclude_kinds=("mil-collect",))
    assert result == {"cancelled": ["fold-2"], "stopping": ["fold-1"]}
    assert client.task("collect")["state"] == "blocked"
    store.transition("fold-1", from_states="stopping", to_state="cancelled")
    store.promote_ready()
    store.transition("collect", from_states="queued", to_state="succeeded")
    requeued = client.requeue_group(
        "mil-batch", "g1", project, kinds=("mil-fold", "mil-collect"), reason="resume"
    )
    assert sorted(requeued) == ["collect", "fold-1", "fold-2"]
    assert client.task("fold-0")["state"] == "succeeded"
    assert client.task("collect")["state"] == "blocked"
    assert client.task("fold-1")["attempt"] == 2
    assert client.cancel_task("fold-1")["state"] == "cancelled"
    assert client.requeue_task("fold-1", reason="retry") is True
    assert client.cancel_task("missing") is None
    assert client.runner_alive() is False


def conclude_with_intent(store, task_id, hook="requeue_intent"):
    store.transition(task_id, from_states="queued", to_state="running", started_at="x")
    return store.conclude(
        task_id,
        to_state="interrupted",
        stop_request=None,
        follow_up={"hook": hook, "reason": "auto-resume", "since": "2026-01-01T00:00:00+00:00"},
        exit={"reason": "lost"},
    )


@pytest.mark.parametrize("hook", ["requeue_intent", "on_requeue"])
def test_a_task_awaiting_its_requeue_is_not_finished_for_dependents(store, tmp_path, hook):
    client = TaskCenterClient(store)
    store.enqueue(
        owner(tmp_path),
        [
            spec(tmp_path, "fold"),
            spec(tmp_path, "collect", dependsOn=[{"task": "fold", "condition": "terminal"}]),
        ],
    )
    assert conclude_with_intent(store, "fold", hook) == "concluded"
    fold = store.get("fold")
    assert fold["state"] == "interrupted" and fold["bookkeeping"]["hook"] == hook
    assert [task["id"] for task in store.pending_bookkeeping()] == ["fold"]
    assert store.promote_ready() == []
    assert store.get("collect")["state"] == "blocked"
    store.enqueue(
        owner(tmp_path),
        [spec(tmp_path, "late", dependsOn=[{"task": "fold", "condition": "terminal"}])],
    )
    assert store.get("late")["state"] == "blocked"
    # The owner stays in the live queue and the group still has work ahead.
    assert [item["id"] for item in store.owners()] == ["A"]
    group = client.group("mil-batch", "g1", str(tmp_path / "project"))
    assert group["live"] == 3 and group["awaitingRequeue"] == 1
    assert store.settle_bookkeeping("fold", "other-hook") is False
    assert store.settle_bookkeeping("fold", hook, error="decided") is True
    assert store.get("fold")["error"] == "decided" and store.get("fold")["bookkeeping"] is None
    assert store.promote_ready() == ["collect", "late"]


def test_a_cancel_wins_over_a_requeue_in_the_store(store, tmp_path):
    client = TaskCenterClient(store)
    store.enqueue(owner(tmp_path), [spec(tmp_path, "t"), spec(tmp_path, "u")])
    store.transition("t", from_states="queued", to_state="running", started_at="x")
    store.request_stop(["t"], "pause")
    store.request_stop(["t"], "cancel")  # lands while the runner classifies the paused exit
    intent = {"hook": "on_requeue", "reason": "paused"}
    assert store.conclude("t", to_state="interrupted", stop_request="pause", follow_up=intent) == (
        "stale"
    )
    assert store.get("t")["state"] == "stopping"
    assert store.conclude("t", to_state="cancelled", stop_request="cancel") == "concluded"
    assert store.conclude("t", to_state="cancelled", stop_request="cancel") == "gone"
    with pytest.raises(ValueError):
        store.conclude("t", to_state="queued", stop_request=None)

    # A cancel after the conclusion marks the task awaiting its requeue.
    assert conclude_with_intent(store, "u", "on_requeue") == "concluded"
    assert store.request_stop(["u"], "pause") == []
    assert client.cancel_task("u")["stopRequest"] == "cancel"
    assert store.events("u")[-1]["detail"] == {"stopRequest": "cancel"}
    assert store.request_stop(["u"], "cancel") == ["u"]
    assert store.requeue(["u"], reason="auto-resume", unless_cancelled=True) == []
    assert store.get("u")["state"] == "interrupted"
    # A manual retry is a new decision and still works.
    assert store.requeue(["u"], reason="retry") == ["u"]
    assert store.get("u")["stopRequest"] is None


def test_the_runner_row_lists_its_code_files_and_old_stores_gain_the_column(tmp_path):
    path = tmp_path / "old" / "task-center.sqlite"
    path.parent.mkdir(mode=0o700)
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute("INSERT INTO meta VALUES('schemaVersion', '1')")
    connection.execute(
        "CREATE TABLE runner(id INTEGER PRIMARY KEY CHECK(id = 1), pid INTEGER, "
        "start_ticks INTEGER, boot_id TEXT, code_hash TEXT, protocol INTEGER, started_at TEXT, "
        "heartbeat_at TEXT, state TEXT, message TEXT, sample TEXT)"
    )
    connection.execute("INSERT INTO runner(id, pid, code_hash) VALUES(1, 7, 'old')")
    connection.commit()
    connection.close()
    store = TaskStore(path)
    row = store.runner()
    assert (row["pid"], row["codeHash"], row["codeFiles"]) == (7, "old", None)
    files = ["/abs/histopilot/a.py", "/abs/histopilot/b.py"]
    store.write_runner(code_hash="new", code_files=files)
    assert store.runner()["codeFiles"] == files and store.runner()["codeHash"] == "new"
    TaskStore(path).initialize()  # idempotent on a migrated store


def test_checkpoint_truncates_the_write_ahead_log_left_by_heartbeats(tmp_path):
    store = TaskStore(tmp_path / "task-center.sqlite")
    files = [f"/checkout/histopilot/module_{index}.py" for index in range(120)]
    store.write_runner(pid=5, state="running", code_files=files)
    for beat in range(200):
        store.write_runner(heartbeat_at=str(beat), sample={"at": str(beat)})
    wal = tmp_path / "task-center.sqlite-wal"
    assert wal.stat().st_size > 0
    with store._connection() as db:
        assert db.execute("PRAGMA journal_size_limit").fetchone()[0] == 8 * 1024 * 1024
    assert store.checkpoint() is True
    assert wal.stat().st_size == 0
    assert store.runner()["heartbeatAt"] == "199"
