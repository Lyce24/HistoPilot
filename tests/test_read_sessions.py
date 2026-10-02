"""A read session shares one connection per store and thread, and forgets nothing it must not."""

import inspect
import json
import threading

import pytest

from histopilot.application import compute_jobs
from histopilot.storage import read_sessions
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import ScientificStore
from histopilot.taskcenter.store import TaskStore


def counting(monkeypatch, owner, name):
    calls = []
    original = getattr(owner, name)

    def counted(*args, **kwargs):
        calls.append(threading.get_ident())
        return original(*args, **kwargs)

    static = isinstance(inspect.getattr_static(owner, name), staticmethod)
    monkeypatch.setattr(owner, name, staticmethod(counted) if static else counted)
    return calls


@pytest.fixture
def store(tmp_path):
    folder = tmp_path / "project"
    folder.mkdir()
    value = ScientificStore(folder, "project-sessions")
    value.initialize()
    for index in range(3):
        value.create_draft("experiment", f"Draft {index}", {"type": "note", "index": index})
    return value


def test_a_session_reads_through_one_connection_and_closes_it(store, monkeypatch):
    opened = counting(monkeypatch, ScientificStore, "_open_connection")
    closed = counting(monkeypatch, ScientificStore, "_close_connection")
    drafts = store.list_drafts()
    for draft in drafts:
        store.get_draft(draft["id"])
    outside = len(opened)
    assert outside >= 4
    opened.clear()
    closed.clear()
    with read_sessions.read_session():
        with read_sessions.read_session():  # nested sessions share the outer one
            for draft in drafts:
                assert store.get_draft(draft["id"]) == draft
        assert store.list_drafts() == drafts
        assert len(opened) == 1 and not closed
    assert len(closed) == 1


def test_writes_inside_a_session_are_seen_by_its_reads(store):
    with read_sessions.read_session():
        before = len(store.list_drafts())
        store.create_draft("experiment", "Written meanwhile", {"type": "note"})
        assert len(store.list_drafts()) == before + 1


def test_each_thread_of_a_session_opens_its_own_connection(store, monkeypatch):
    opened = counting(monkeypatch, ScientificStore, "_open_connection")
    with read_sessions.read_session():
        store.list_drafts()
        context = __import__("contextvars").copy_context()
        worker = threading.Thread(target=lambda: context.run(store.list_drafts))
        worker.start()
        worker.join()
        store.list_drafts()
    assert len(set(opened)) == 2 and len(opened) == 2


def test_task_center_reads_share_a_connection_and_writes_do_not(tmp_path, monkeypatch):
    tasks = TaskStore(tmp_path / "state" / "task-center.sqlite")
    tasks.initialize()
    opened = counting(monkeypatch, TaskStore, "_open")
    with read_sessions.read_session():
        for _ in range(5):
            assert tasks.get("task-missing") is None
        tasks.counts()
        assert len(opened) == 1
        tasks.update_settings({})
        assert len(opened) == 2


def test_remember_computes_once_per_session_and_never_keeps_a_failure():
    calls = []

    def compute():
        calls.append(1)
        return len(calls)

    assert read_sessions.remember(("k",), compute) == 1
    assert read_sessions.remember(("k",), compute) == 2  # outside a session: every call
    with read_sessions.read_session():
        assert read_sessions.remember(("k",), compute) == 3
        assert read_sessions.remember(("k",), compute) == 3
        failures = []

        def failing():
            failures.append(1)
            raise StorageError("Corrupt.", "STORAGE_CORRUPT")

        for _ in range(2):
            with pytest.raises(StorageError):
                read_sessions.remember(("bad",), failing)
        assert len(failures) == 2


def test_a_changed_plan_file_is_hashed_again(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"runs": [1, 2, 3]}))
    first = compute_jobs.plan_hash(path)
    assert compute_jobs.plan_hash(path) == first
    replacement = tmp_path / "plan.next"
    replacement.write_text(json.dumps({"runs": [1, 2, 4]}))
    replacement.replace(path)
    assert compute_jobs.plan_hash(path) != first


def test_sessions_take_turns_and_a_long_wait_reads_anyway(monkeypatch):
    monkeypatch.setattr(read_sessions, "_TURNS", threading.Semaphore(1))
    monkeypatch.setattr(read_sessions, "TURN_SECONDS", 0.2)
    order = []
    entered = threading.Event()
    release = threading.Event()

    def first():
        with read_sessions.read_session():
            order.append("first in")
            entered.set()
            release.wait(5)
        order.append("first out")

    def second():
        entered.wait(5)
        with read_sessions.read_session():
            order.append("second in")

    threads = [threading.Thread(target=first), threading.Thread(target=second)]
    for thread in threads:
        thread.start()
    entered.wait(5)
    threads[1].join(1)
    # The second session waited its turn up to TURN_SECONDS, then read without one.
    assert order == ["first in", "second in"]
    release.set()
    threads[0].join(5)
    # A nested session never waits for a turn of its own.
    monkeypatch.setattr(read_sessions, "_TURNS", threading.Semaphore(1))
    with read_sessions.read_session(), read_sessions.read_session():
        assert read_sessions.active()
    assert read_sessions._TURNS.acquire(timeout=0)
