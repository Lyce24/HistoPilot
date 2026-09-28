"""SQLite connections never checkpoint on close and survive concurrent open/close.

SQLite 3.51.0 deadlocks two threads of one process when one closes the last connection to
a WAL database (checkpoint on close) while another opens or closes a connection to it.
"""

import sqlite3
import subprocess
import sys
import textwrap

import pytest

from histopilot.storage import sqlite_connections
from histopilot.storage.database import Database
from histopilot.storage.scientific import ScientificStore
from histopilot.taskcenter.store import TaskStore

NO_CHECKPOINT = getattr(sqlite3, "SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE", None)
pytestmark = pytest.mark.skipif(
    NO_CHECKPOINT is None, reason="this Python cannot configure SQLite connections"
)


def test_every_connection_skips_the_close_time_checkpoint(tmp_path):
    store = TaskStore(tmp_path / "state" / "task-center.sqlite")
    store.initialize()
    task_connection = store._open()
    try:
        assert task_connection.getconfig(NO_CHECKPOINT)
    finally:
        sqlite_connections.close(task_connection)

    project = tmp_path / "project"
    project.mkdir()
    scientific = ScientificStore(project, "project-1")
    scientific.initialize()
    with scientific._connection() as connection:
        assert connection.getconfig(NO_CHECKPOINT)

    database = Database(tmp_path / "workspace")
    database.initialize()
    raw = database.engine.raw_connection()
    try:
        assert raw.driver_connection.getconfig(NO_CHECKPOINT)
    finally:
        raw.close()


def test_the_task_store_survives_concurrent_open_and_close(tmp_path):
    # Three threads opening, reading and closing store connections deadlocked SQLite 3.51.0
    # within about a thousand calls; run it in a child so a regression cannot hang the suite.
    script = textwrap.dedent(
        f"""
        import os, threading, time
        from pathlib import Path
        from histopilot.taskcenter.store import TaskStore
        store = TaskStore(Path({str(tmp_path)!r}) / "task-center.sqlite")
        store.initialize()
        calls, end = [0], time.monotonic() + 3
        def work():
            while time.monotonic() < end:
                store.counts()
                calls[0] += 1
        threads = [threading.Thread(target=work, daemon=True) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=20)
        print(calls[0], flush=True)
        os._exit(1 if any(thread.is_alive() for thread in threads) else 0)
        """
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
        )
    except subprocess.TimeoutExpired:
        pytest.fail("Concurrent Task Center store calls deadlocked inside SQLite.")
    assert result.returncode == 0, result.stderr
    assert int(result.stdout.split()[-1]) > 100
