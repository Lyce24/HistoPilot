"""Machine-level SQLite task store: schema, queries and conditional state transitions.

Every method opens its own connection. Writes run inside ``BEGIN IMMEDIATE`` so the API
and the runner serialize on SQLite's own lock; conditional transitions make every write
safe to repeat.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path

from histopilot.storage import sqlite_connections
from histopilot.storage.project_lock import (
    StorageError,
    ensure_managed_directory,
    reject_symlink_components,
)
from histopilot.taskcenter import ids, paths
from histopilot.taskcenter.model import (
    ACTIVE,
    DEFAULT_SETTINGS,
    LIVE,
    PENDING,
    PLAN_ORDER_SPAN,
    REQUEUE_HOOKS,
    SCHEMA_VERSION,
    STATES,
    STOP_REASONS,
    TERMINAL,
    apply_settings_patch,
    merge_settings,
    normalize_owner,
    normalize_request,
    normalize_task,
    utc_now_iso,
)

__all__ = ["DEFAULT_SETTINGS", "TaskStore"]

SCHEMA = (
    "CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    """CREATE TABLE IF NOT EXISTS owners(
  key TEXT PRIMARY KEY, kind TEXT NOT NULL, id TEXT NOT NULL,
  project_id TEXT NOT NULL, project_folder TEXT NOT NULL, workspace TEXT,
  title TEXT NOT NULL, labels TEXT NOT NULL DEFAULT '{}',
  queue_seq INTEGER NOT NULL, held INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS task_groups(
  owner_key TEXT NOT NULL REFERENCES owners(key), kind TEXT NOT NULL, id TEXT NOT NULL,
  project_folder TEXT NOT NULL, seq INTEGER NOT NULL, created_at TEXT NOT NULL,
  PRIMARY KEY(kind, id, project_folder))""",
    """CREATE TABLE IF NOT EXISTS tasks(
  id TEXT PRIMARY KEY, kind TEXT NOT NULL, adapter TEXT NOT NULL,
  owner_key TEXT NOT NULL REFERENCES owners(key), group_kind TEXT, group_id TEXT,
  project_folder TEXT NOT NULL,
  title TEXT NOT NULL, labels TEXT NOT NULL DEFAULT '{}',
  priority TEXT NOT NULL DEFAULT 'normal', plan_order INTEGER NOT NULL DEFAULT 0,
  exclusive_key TEXT, session_name TEXT,
  state TEXT NOT NULL, attempt INTEGER NOT NULL DEFAULT 1,
  request TEXT NOT NULL, command TEXT NOT NULL, adapter_data TEXT NOT NULL DEFAULT '{}',
  process TEXT, gpu INTEGER, lease TEXT,
  stop_request TEXT, stop_requested_at TEXT, signalled_at TEXT, killed_at TEXT,
  exit TEXT, progress TEXT, resources TEXT, waiting_reason TEXT, error TEXT, bookkeeping TEXT,
  created_at TEXT NOT NULL, queued_at TEXT, started_at TEXT, finished_at TEXT,
  updated_at TEXT NOT NULL)""",
    "CREATE INDEX IF NOT EXISTS tasks_state ON tasks(state)",
    "CREATE INDEX IF NOT EXISTS tasks_owner ON tasks(owner_key)",
    "CREATE INDEX IF NOT EXISTS tasks_group ON tasks(group_kind, group_id, project_folder)",
    "CREATE UNIQUE INDEX IF NOT EXISTS tasks_session ON tasks(session_name) "
    "WHERE session_name IS NOT NULL",
    """CREATE TABLE IF NOT EXISTS task_dependencies(task_id TEXT NOT NULL, depends_on TEXT NOT NULL,
  condition TEXT NOT NULL CHECK(condition IN ('succeeded','terminal')),
  PRIMARY KEY(task_id, depends_on))""",
    """CREATE TABLE IF NOT EXISTS task_events(seq INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id TEXT NOT NULL, at TEXT NOT NULL, attempt INTEGER, from_state TEXT, to_state TEXT,
  detail TEXT)""",
    """CREATE TABLE IF NOT EXISTS operations(operation_id TEXT PRIMARY KEY,
  request_hash TEXT NOT NULL, result TEXT, created_at TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT NOT NULL,
  updated_at TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS measurements(seq INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id TEXT, attempt INTEGER, kind TEXT, lane TEXT, workload_key TEXT, workload TEXT,
  gpu_index INTEGER, gpu_name TEXT, gpu_uuid TEXT, peak_vram_gb REAL, peak_private_ram_gb REAL,
  mean_cpu_cores REAL, epochs INTEGER, wall_seconds REAL, seconds_per_epoch REAL,
  mean_concurrency REAL, exit_reason TEXT, started_at TEXT, finished_at TEXT,
  recorded_at TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS runner(id INTEGER PRIMARY KEY CHECK(id = 1), pid INTEGER,
  start_ticks INTEGER, boot_id TEXT, code_hash TEXT, protocol INTEGER, started_at TEXT,
  heartbeat_at TEXT, state TEXT, message TEXT, sample TEXT, code_files TEXT)""",
)
# Columns added after schema version 1 shipped; older stores gain them on open.
ADDED_COLUMNS = (
    ("runner", "code_files", "TEXT"),
    ("runner", "gpu_fences", "TEXT"),
    ("runner", "checkout_root", "TEXT"),
)
# Automatic-retry budgets in a task's adapter data. A manual retry or resume starts over.
RETRY_COUNTERS = ("oomRetries", "contentionRequeues", "deviceLossRequeues")
# The runner writes a heartbeat every few seconds while the service reads constantly, so the
# write-ahead log rarely restarts on its own; cap the file it leaves behind after a checkpoint.
WAL_SIZE_LIMIT = 8 * 1024 * 1024

JSON_COLUMNS = frozenset(
    {
        "labels",
        "request",
        "command",
        "adapter_data",
        "process",
        "exit",
        "progress",
        "resources",
        "bookkeeping",
    }
)
# Columns a caller may change directly; state changes go through transition().
MUTABLE_COLUMNS = {
    "title": "title",
    "labels": "labels",
    "priority": "priority",
    "exclusive_key": "exclusiveKey",
    "session_name": "sessionName",
    "request": "request",
    "command": "command",
    "adapter_data": "adapterData",
    "process": "process",
    "gpu": "gpu",
    "lease": "lease",
    "stop_request": "stopRequest",
    "stop_requested_at": "stopRequestedAt",
    "signalled_at": "signalledAt",
    "killed_at": "killedAt",
    "exit": "exit",
    "progress": "progress",
    "resources": "resources",
    "waiting_reason": "waitingReason",
    "error": "error",
    "bookkeeping": "bookkeeping",
    "queued_at": "queuedAt",
    "started_at": "startedAt",
    "finished_at": "finishedAt",
    "attempt": "attempt",
}
_CAMEL_COLUMNS = {camel: column for column, camel in MUTABLE_COLUMNS.items()}
RUN_CLEARED = (
    "process",
    "gpu",
    "lease",
    "stop_request",
    "stop_requested_at",
    "signalled_at",
    "killed_at",
    "exit",
    "started_at",
    "finished_at",
    "error",
    "bookkeeping",
    "waiting_reason",
)
MEASUREMENT_COLUMNS = {
    "taskId": "task_id",
    "attempt": "attempt",
    "kind": "kind",
    "lane": "lane",
    "workloadKey": "workload_key",
    "workload": "workload",
    "gpuIndex": "gpu_index",
    "gpuName": "gpu_name",
    "gpuUuid": "gpu_uuid",
    "peakVramGb": "peak_vram_gb",
    "peakPrivateRamGb": "peak_private_ram_gb",
    "meanCpuCores": "mean_cpu_cores",
    "epochs": "epochs",
    "wallSeconds": "wall_seconds",
    "secondsPerEpoch": "seconds_per_epoch",
    "meanConcurrency": "mean_concurrency",
    "exitReason": "exit_reason",
    "startedAt": "started_at",
    "finishedAt": "finished_at",
    "recordedAt": "recorded_at",
}
RUNNER_COLUMNS = {
    "pid": "pid",
    "startTicks": "start_ticks",
    "bootId": "boot_id",
    "codeHash": "code_hash",
    "protocol": "protocol",
    "startedAt": "started_at",
    "heartbeatAt": "heartbeat_at",
    "state": "state",
    "message": "message",
    "sample": "sample",
    "codeFiles": "code_files",
    "gpuFences": "gpu_fences",
    "checkoutRoot": "checkout_root",
}
RUNNER_JSON_COLUMNS = frozenset({"sample", "code_files", "gpu_fences"})
_TERMINAL_SQL = ",".join(f"'{state}'" for state in sorted(TERMINAL))
_REQUEUE_HOOKS_SQL = ",".join(f"'{hook}'" for hook in sorted(REQUEUE_HOOKS))
# A concluded task the runner may still requeue keeps its owner in the live queue.
LIVE_OR_AWAITING_SQL = (
    "(t.state IN ({live}) OR (t.state IN ({terminal}) AND "
    "json_extract(t.bookkeeping, '$.hook') IN ({hooks})))".format(
        live=",".join(f"'{state}'" for state in sorted(LIVE)),
        terminal=_TERMINAL_SQL,
        hooks=_REQUEUE_HOOKS_SQL,
    )
)


def _encode(value) -> str | None:
    if value is None:
        return None
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _decode(value):
    return None if value is None else json.loads(value)


def request_hash(value) -> str:
    return hashlib.sha256(_encode(value).encode()).hexdigest()


def _states(value) -> tuple[str, ...]:
    return (value,) if isinstance(value, str) else tuple(value)


def _marks(values) -> str:
    return ",".join("?" for _ in values)


class TaskStore:
    def __init__(self, path: Path | None = None, *, now=utc_now_iso):
        self.path = Path(path) if path is not None else paths.store_path()
        self._now = now
        self._ready = False
        self._init_lock = threading.Lock()

    # -- connections -------------------------------------------------------------------

    def _open(self) -> sqlite3.Connection:
        connection = sqlite_connections.connect(
            self.path, timeout=5, check_same_thread=False, isolation_level=None
        )
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout=5000")
            mode = connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            if str(mode).lower() != "wal":
                raise StorageError(
                    "The Task Center store requires SQLite write-ahead logging on a local disk.",
                    "TASK_CENTER_STORE_UNSUPPORTED",
                    503,
                )
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute(f"PRAGMA journal_size_limit={WAL_SIZE_LIMIT}")
            return connection
        except BaseException:
            sqlite_connections.close(connection)
            raise

    def initialize(self) -> None:
        with self._init_lock:
            if self._ready:
                return
            try:
                reject_symlink_components(self.path)
                if not self.path.parent.is_dir():
                    ensure_managed_directory(self.path.parent)
                connection = self._open()
                try:
                    connection.execute("BEGIN IMMEDIATE")
                    try:
                        has_meta = connection.execute(
                            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'"
                        ).fetchone()
                        version = None
                        if has_meta:
                            row = connection.execute(
                                "SELECT value FROM meta WHERE key='schemaVersion'"
                            ).fetchone()
                            version = int(row[0]) if row else None
                        if version is not None and version > SCHEMA_VERSION:
                            raise StorageError(
                                "The Task Center store was created by a newer HistoPilot. "
                                "Update this checkout before using it.",
                                "TASK_CENTER_STORE_NEWER",
                                409,
                            )
                        for statement in SCHEMA:
                            connection.execute(statement)
                        for table, column, kind in ADDED_COLUMNS:
                            present = {
                                row[1] for row in connection.execute(f"PRAGMA table_info({table})")
                            }
                            if column not in present:
                                connection.execute(
                                    f"ALTER TABLE {table} ADD COLUMN {column} {kind}"
                                )
                        if version is None or version < SCHEMA_VERSION:
                            connection.execute(
                                "INSERT OR REPLACE INTO meta(key, value) VALUES('schemaVersion', ?)",
                                (str(SCHEMA_VERSION),),
                            )
                        connection.execute("COMMIT")
                    except BaseException:
                        if connection.in_transaction:
                            connection.execute("ROLLBACK")
                        raise
                finally:
                    sqlite_connections.close(connection)
            except sqlite3.Error as error:
                raise StorageError(
                    f"The Task Center store is unavailable: {error}",
                    "TASK_CENTER_UNAVAILABLE",
                    503,
                ) from error
            self._ready = True

    @contextmanager
    def _connection(self, *, write=False):
        self.initialize()
        try:
            connection = self._open()
        except sqlite3.Error as error:
            raise StorageError(
                f"The Task Center store is unavailable: {error}", "TASK_CENTER_UNAVAILABLE", 503
            ) from error
        try:
            if write:
                connection.execute("BEGIN IMMEDIATE")
            yield connection
            if write:
                connection.execute("COMMIT")
        except sqlite3.IntegrityError as error:
            self._rollback(connection)
            raise StorageError(
                f"The Task Center store rejected a conflicting change: {error}",
                "TASK_CENTER_CONFLICT",
                409,
            ) from error
        except sqlite3.Error as error:
            self._rollback(connection)
            raise StorageError(
                f"The Task Center store is unavailable: {error}", "TASK_CENTER_UNAVAILABLE", 503
            ) from error
        except BaseException:
            self._rollback(connection)
            raise
        finally:
            sqlite_connections.close(connection)

    @staticmethod
    def _rollback(connection) -> None:
        try:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
        except sqlite3.Error:
            pass

    # -- row mapping ---------------------------------------------------------------------

    @staticmethod
    def _task(row) -> dict:
        return {
            "id": row["id"],
            "kind": row["kind"],
            "adapter": row["adapter"],
            "ownerKey": row["owner_key"],
            "group": {"kind": row["group_kind"], "id": row["group_id"]}
            if row["group_kind"] is not None
            else None,
            "projectFolder": row["project_folder"],
            "title": row["title"],
            "labels": _decode(row["labels"]) or {},
            "priority": row["priority"],
            "planOrder": row["plan_order"],
            "exclusiveKey": row["exclusive_key"],
            "sessionName": row["session_name"],
            "state": row["state"],
            "attempt": row["attempt"],
            "request": _decode(row["request"]),
            "command": _decode(row["command"]),
            "adapterData": _decode(row["adapter_data"]) or {},
            "process": _decode(row["process"]),
            "gpu": row["gpu"],
            "lease": row["lease"],
            "stopRequest": row["stop_request"],
            "stopRequestedAt": row["stop_requested_at"],
            "signalledAt": row["signalled_at"],
            "killedAt": row["killed_at"],
            "exit": _decode(row["exit"]),
            "progress": _decode(row["progress"]),
            "resources": _decode(row["resources"]),
            "waitingReason": row["waiting_reason"],
            "error": row["error"],
            "bookkeeping": _decode(row["bookkeeping"]),
            "createdAt": row["created_at"],
            "queuedAt": row["queued_at"],
            "startedAt": row["started_at"],
            "finishedAt": row["finished_at"],
            "updatedAt": row["updated_at"],
        }

    @staticmethod
    def _owner(row, counts: dict | None = None) -> dict:
        return {
            "key": row["key"],
            "kind": row["kind"],
            "id": row["id"],
            "projectId": row["project_id"],
            "projectFolder": row["project_folder"],
            "workspace": row["workspace"],
            "title": row["title"],
            "labels": _decode(row["labels"]) or {},
            "queueSeq": row["queue_seq"],
            "held": bool(row["held"]),
            "createdAt": row["created_at"],
            "updatedAt": row["updated_at"],
            "counts": counts if counts is not None else {},
        }

    @staticmethod
    def _owner_counts(db, key: str) -> dict:
        counts = dict.fromkeys(STATES, 0)
        for row in db.execute(
            "SELECT state, COUNT(*) AS n FROM tasks WHERE owner_key=? GROUP BY state", (key,)
        ):
            counts[row["state"]] = row["n"]
        return counts

    def _owner_dict(self, db, key: str) -> dict | None:
        row = db.execute("SELECT * FROM owners WHERE key=?", (key,)).fetchone()
        return self._owner(row, self._owner_counts(db, key)) if row else None

    @staticmethod
    def _columns(fields: dict) -> dict:
        columns = {}
        for name, value in fields.items():
            column = name if name in MUTABLE_COLUMNS else _CAMEL_COLUMNS.get(name)
            if column is None:
                raise ValueError(f"Task field {name} cannot be changed directly.")
            if column == "request" and value is not None:
                value = normalize_request(value)
            columns[column] = _encode(value) if column in JSON_COLUMNS else value
        return columns

    def _event(self, db, task_id, attempt, from_state, to_state, detail=None) -> None:
        db.execute(
            "INSERT INTO task_events(task_id, at, attempt, from_state, to_state, detail) "
            "VALUES(?,?,?,?,?,?)",
            (task_id, self._now(), attempt, from_state, to_state, _encode(detail)),
        )

    @staticmethod
    def _dependencies_met(db, task_id: str) -> bool:
        for row in db.execute(
            "SELECT d.condition, t.state, t.bookkeeping FROM task_dependencies d "
            "LEFT JOIN tasks t ON t.id = d.depends_on WHERE d.task_id=?",
            (task_id,),
        ):
            state = row["state"]
            if state is None:
                return False
            # A task that may still be requeued (for example a lost fold awaiting its
            # auto-resume) has not finished yet for the tasks that depend on it.
            if (
                state in TERMINAL
                and (_decode(row["bookkeeping"]) or {}).get("hook") in REQUEUE_HOOKS
            ):
                return False
            if row["condition"] == "succeeded" and state != "succeeded":
                return False
            if row["condition"] == "terminal" and state not in TERMINAL:
                return False
        return True

    # -- enqueue and lookup ----------------------------------------------------------------

    def enqueue(self, owner: dict, tasks: list[dict], *, operation_id: str | None = None) -> dict:
        owner = normalize_owner(owner)
        specs = [normalize_task(task) for task in tasks]
        batch = [spec["id"] for spec in specs]
        if len(set(batch)) != len(batch):
            raise StorageError("Task ids must be unique in one request.", "TASK_INVALID", 422)
        digest = request_hash({"owner": owner, "tasks": specs})
        key = ids.owner_key(owner["kind"], owner["id"], owner["projectFolder"])
        now = self._now()
        with self._connection(write=True) as db:
            if operation_id is not None:
                receipt = db.execute(
                    "SELECT request_hash, result FROM operations WHERE operation_id=?",
                    (operation_id,),
                ).fetchone()
                if receipt is not None:
                    if receipt["request_hash"] != digest:
                        raise StorageError(
                            "This operation id was already used for a different request.",
                            "OPERATION_CONFLICT",
                            409,
                        )
                    return _decode(receipt["result"])
            existing = db.execute("SELECT * FROM owners WHERE key=?", (key,)).fetchone()
            was_live = existing is not None and (
                db.execute(
                    f"SELECT 1 FROM tasks t WHERE t.owner_key=? AND {LIVE_OR_AWAITING_SQL} LIMIT 1",
                    (key,),
                ).fetchone()
                is not None
            )
            if existing is None:
                db.execute(
                    "INSERT INTO owners(key, kind, id, project_id, project_folder, workspace, "
                    "title, labels, queue_seq, held, created_at, updated_at) VALUES("
                    "?,?,?,?,?,?,?,?,(SELECT COALESCE(MAX(queue_seq), 0) + 1 FROM owners),"
                    "0,?,?)",
                    (
                        key,
                        owner["kind"],
                        owner["id"],
                        owner["projectId"],
                        owner["projectFolder"],
                        owner["workspace"],
                        owner["title"],
                        _encode(owner["labels"]),
                        now,
                        now,
                    ),
                )
            else:
                # Several services queue work for one owner; each adds labels, none erases them.
                labels = {**(_decode(existing["labels"]) or {}), **owner["labels"]}
                db.execute(
                    "UPDATE owners SET title=?, labels=?, project_id=?, "
                    "workspace=COALESCE(?, workspace), updated_at=? WHERE key=?",
                    (
                        owner["title"],
                        _encode(labels),
                        owner["projectId"],
                        owner["workspace"],
                        now,
                        key,
                    ),
                )
            created = []
            for spec in specs:
                group = spec["group"] or {"kind": owner["kind"], "id": owner["id"]}
                group_row = db.execute(
                    "SELECT seq FROM task_groups WHERE kind=? AND id=? AND project_folder=?",
                    (group["kind"], group["id"], owner["projectFolder"]),
                ).fetchone()
                if group_row is None:
                    seq = db.execute(
                        "SELECT COALESCE(MAX(seq), 0) + 1 FROM task_groups WHERE owner_key=?",
                        (key,),
                    ).fetchone()[0]
                    db.execute(
                        "INSERT INTO task_groups(owner_key, kind, id, project_folder, seq, "
                        "created_at) VALUES(?,?,?,?,?,?)",
                        (key, group["kind"], group["id"], owner["projectFolder"], seq, now),
                    )
                else:
                    seq = group_row["seq"]
                row = db.execute("SELECT state FROM tasks WHERE id=?", (spec["id"],)).fetchone()
                if row is not None:
                    if row["state"] in PENDING:
                        db.execute(
                            "UPDATE tasks SET title=?, labels=?, request=?, command=?, "
                            "updated_at=? WHERE id=?",
                            (
                                spec["title"],
                                _encode(spec["labels"]),
                                _encode(spec["request"]),
                                _encode(spec["command"]),
                                now,
                                spec["id"],
                            ),
                        )
                    continue
                for dependency in spec["dependsOn"]:
                    if dependency["task"] not in batch and (
                        db.execute(
                            "SELECT 1 FROM tasks WHERE id=?", (dependency["task"],)
                        ).fetchone()
                        is None
                    ):
                        raise StorageError(
                            f"Task {spec['id']} depends on unknown task {dependency['task']}.",
                            "TASK_DEPENDENCY_UNKNOWN",
                            422,
                        )
                    db.execute(
                        "INSERT OR IGNORE INTO task_dependencies(task_id, depends_on, condition) "
                        "VALUES(?,?,?)",
                        (spec["id"], dependency["task"], dependency["condition"]),
                    )
                state = "queued" if self._dependencies_met(db, spec["id"]) else "blocked"
                db.execute(
                    "INSERT INTO tasks(id, kind, adapter, owner_key, group_kind, group_id, "
                    "project_folder, title, labels, priority, plan_order, exclusive_key, "
                    "session_name, state, attempt, request, command, adapter_data, created_at, "
                    "queued_at, updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,1,?,?,?,?,?,?)",
                    (
                        spec["id"],
                        spec["kind"],
                        spec["adapter"],
                        key,
                        group["kind"],
                        group["id"],
                        owner["projectFolder"],
                        spec["title"],
                        _encode(spec["labels"]),
                        spec["priority"],
                        seq * PLAN_ORDER_SPAN + spec["planOrder"],
                        spec["exclusiveKey"],
                        spec["sessionName"],
                        state,
                        _encode(spec["request"]),
                        _encode(spec["command"]),
                        _encode(spec["adapterData"]),
                        now,
                        now if state == "queued" else None,
                        now,
                    ),
                )
                self._event(db, spec["id"], 1, None, state, {"event": "enqueued"})
                created.append(spec["id"])
            if created and existing is not None and not was_live:
                # An idle owner that receives new work joins the back of the queue.
                db.execute(
                    "UPDATE owners SET queue_seq=(SELECT COALESCE(MAX(queue_seq), 0) + 1 "
                    "FROM owners) WHERE key=?",
                    (key,),
                )
            result = {"owner": self._owner_dict(db, key), "tasks": batch, "created": created}
            if operation_id is not None:
                db.execute(
                    "INSERT INTO operations(operation_id, request_hash, result, created_at) "
                    "VALUES(?,?,?,?)",
                    (operation_id, digest, _encode(result), now),
                )
            return result

    def requeue(
        self,
        task_ids: Iterable[str],
        *,
        reason: str,
        include_succeeded=False,
        request_patch: dict | None = None,
        adapter_data_patch: dict | None = None,
        unless_cancelled: bool = False,
        manual: bool = True,
    ) -> list[str]:
        """Start another attempt of terminal tasks.

        ``unless_cancelled`` (the runner's automatic requeues) skips a task whose cancel was
        requested after it concluded; checked in this transaction, so the cancel wins.
        A ``manual`` requeue (a retry or resume, not the runner's own) resets the automatic
        retry budgets, so an earlier OOM or device loss never limits the new run.
        """
        allowed = TERMINAL if include_succeeded else TERMINAL - {"succeeded"}
        now = self._now()
        requeued = []
        with self._connection(write=True) as db:
            for task_id in dict.fromkeys(task_ids):
                row = db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
                if row is None or row["state"] not in allowed:
                    continue
                if unless_cancelled and row["stop_request"] == "cancel":
                    continue
                request = _decode(row["request"])
                if request_patch:
                    request = normalize_request({**request, **request_patch})
                adapter_data = _decode(row["adapter_data"]) or {}
                if manual:
                    adapter_data = {
                        key: value
                        for key, value in adapter_data.items()
                        if key not in RETRY_COUNTERS
                    }
                if adapter_data_patch:
                    adapter_data = {**adapter_data, **adapter_data_patch}
                state = "queued" if self._dependencies_met(db, task_id) else "blocked"
                cleared = ", ".join(f"{column}=NULL" for column in RUN_CLEARED)
                db.execute(
                    f"UPDATE tasks SET state=?, attempt=attempt + 1, request=?, adapter_data=?, "
                    f"{cleared}, queued_at=?, updated_at=? WHERE id=?",
                    (
                        state,
                        _encode(request),
                        _encode(adapter_data),
                        now if state == "queued" else None,
                        now,
                        task_id,
                    ),
                )
                detail = {"reason": reason}
                if reason == "auto-resume":
                    detail["autoResumed"] = True
                if request_patch:
                    detail["requestPatch"] = request_patch
                self._event(db, task_id, row["attempt"] + 1, row["state"], state, detail)
                requeued.append(task_id)
            self._revive_dependents(db, requeued, reason, now)
        return requeued

    def _revive_dependents(self, db, task_ids: list[str], reason: str, now: str) -> None:
        """Dependents that ``promote_ready`` cancelled because one of these tasks failed wait
        for its new attempt again (and theirs for them); a user's own cancel is kept."""
        frontier = list(task_ids)
        while frontier:
            rows = db.execute(
                "SELECT DISTINCT t.id, t.attempt FROM task_dependencies d JOIN tasks t "
                f"ON t.id = d.task_id WHERE d.depends_on IN ({_marks(frontier)}) "
                "AND t.state='cancelled'",
                frontier,
            ).fetchall()
            frontier = []
            for row in rows:
                last = db.execute(
                    "SELECT detail FROM task_events WHERE task_id=? ORDER BY seq DESC LIMIT 1",
                    (row["id"],),
                ).fetchone()
                detail = (_decode(last["detail"]) if last else None) or {}
                if detail.get("event") != "dependency-failed":
                    continue
                cleared = ", ".join(f"{column}=NULL" for column in RUN_CLEARED)
                db.execute(
                    f"UPDATE tasks SET state='blocked', attempt=attempt + 1, {cleared}, "
                    "queued_at=NULL, updated_at=? WHERE id=?",
                    (now, row["id"]),
                )
                self._event(
                    db,
                    row["id"],
                    row["attempt"] + 1,
                    "cancelled",
                    "blocked",
                    {"reason": reason, "event": "dependency-requeued"},
                )
                frontier.append(row["id"])

    def get(self, task_id: str) -> dict | None:
        with self._connection() as db:
            row = db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        return self._task(row) if row else None

    def by_session(self, session_name: str) -> dict | None:
        with self._connection() as db:
            row = db.execute("SELECT * FROM tasks WHERE session_name=?", (session_name,)).fetchone()
        return self._task(row) if row else None

    def list(
        self,
        *,
        states=None,
        owner_key=None,
        group=None,
        project_folder=None,
        kinds=None,
        limit: int | None = 500,
        order="queue",
    ) -> list[dict]:
        clauses, params = [], []
        if states is not None:
            states = _states(states)
            if not states:
                return []
            clauses.append(f"t.state IN ({_marks(states)})")
            params.extend(states)
        if owner_key is not None:
            clauses.append("t.owner_key=?")
            params.append(owner_key)
        if group is not None:
            kind, identity = (
                (group["kind"], group["id"]) if isinstance(group, dict) else tuple(group)
            )
            clauses.append("t.group_kind=? AND t.group_id=?")
            params.extend((kind, identity))
        if project_folder is not None:
            clauses.append("t.project_folder=?")
            params.append(str(project_folder))
        if kinds is not None:
            kinds = _states(kinds)
            if not kinds:
                return []
            clauses.append(f"t.kind IN ({_marks(kinds)})")
            params.extend(kinds)
        if order == "queue":
            ordering = (
                "CASE t.priority WHEN 'interactive' THEN 0 ELSE 1 END, o.queue_seq, "
                "t.plan_order, t.created_at, t.id"
            )
        elif order == "recent":
            ordering = "COALESCE(t.finished_at, t.updated_at) DESC, t.id"
        else:
            raise ValueError(f"Unknown task order: {order}")
        sql = "SELECT t.* FROM tasks t JOIN owners o ON o.key = t.owner_key"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY " + ordering
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        with self._connection() as db:
            return [self._task(row) for row in db.execute(sql, params)]

    def pending_bookkeeping(self) -> list[dict]:
        """Tasks with a pending runner hook: a transient adapter failure to retry, or a
        concluded task whose requeue is still to be decided or carried out."""
        with self._connection() as db:
            rows = db.execute(
                "SELECT * FROM tasks WHERE bookkeeping IS NOT NULL ORDER BY updated_at"
            ).fetchall()
        return [self._task(row) for row in rows]

    def counts(self) -> dict:
        counts = dict.fromkeys(STATES, 0)
        with self._connection() as db:
            for row in db.execute("SELECT state, COUNT(*) AS n FROM tasks GROUP BY state"):
                counts[row["state"]] = row["n"]
        return counts

    def owner(self, key: str) -> dict | None:
        with self._connection() as db:
            return self._owner_dict(db, key)

    def owners(self, *, live_only=True) -> list[dict]:
        with self._connection() as db:
            if live_only:
                rows = db.execute(
                    "SELECT * FROM owners o WHERE EXISTS (SELECT 1 FROM tasks t "
                    f"WHERE t.owner_key=o.key AND {LIVE_OR_AWAITING_SQL}) "
                    "ORDER BY o.queue_seq"
                ).fetchall()
            else:
                rows = db.execute("SELECT * FROM owners ORDER BY queue_seq").fetchall()
            counts = {}
            for row in db.execute(
                "SELECT owner_key, state, COUNT(*) AS n FROM tasks GROUP BY owner_key, state"
            ):
                counts.setdefault(row["owner_key"], dict.fromkeys(STATES, 0))[row["state"]] = row[
                    "n"
                ]
        return [self._owner(row, counts.get(row["key"], dict.fromkeys(STATES, 0))) for row in rows]

    def dependencies(self, task_id: str) -> list[dict]:
        with self._connection() as db:
            rows = db.execute(
                "SELECT d.depends_on, d.condition, t.state FROM task_dependencies d "
                "LEFT JOIN tasks t ON t.id = d.depends_on WHERE d.task_id=? ORDER BY d.depends_on",
                (task_id,),
            ).fetchall()
        return [
            {"task": row["depends_on"], "condition": row["condition"], "state": row["state"]}
            for row in rows
        ]

    def dependents(self, task_ids: Iterable[str]) -> list[str]:
        task_ids = list(task_ids)
        if not task_ids:
            return []
        with self._connection() as db:
            rows = db.execute(
                "SELECT DISTINCT task_id FROM task_dependencies "
                f"WHERE depends_on IN ({_marks(task_ids)}) ORDER BY task_id",
                task_ids,
            ).fetchall()
        return [row["task_id"] for row in rows]

    def events(self, task_id: str, limit=50) -> list[dict]:
        with self._connection() as db:
            rows = db.execute(
                "SELECT * FROM task_events WHERE task_id=? ORDER BY seq DESC LIMIT ?",
                (task_id, int(limit)),
            ).fetchall()
        return [
            {
                "seq": row["seq"],
                "taskId": row["task_id"],
                "at": row["at"],
                "attempt": row["attempt"],
                "fromState": row["from_state"],
                "toState": row["to_state"],
                "detail": _decode(row["detail"]),
            }
            for row in reversed(rows)
        ]

    # -- transitions -----------------------------------------------------------------------

    def transition(self, task_id: str, *, from_states, to_state: str, **fields) -> bool:
        if to_state not in STATES:
            raise ValueError(f"Unknown task state: {to_state}")
        detail = fields.pop("detail", None)
        # Refuse the change when the task's owner is held, checked in the same transaction
        # so a hold committed before the runner's start is never missed.
        unless_owner_held = fields.pop("unless_owner_held", False)
        columns = self._columns(fields)
        now = self._now()
        if to_state == "queued":
            columns.setdefault("queued_at", now)
        if to_state in TERMINAL:
            columns.setdefault("finished_at", now)
        with self._connection(write=True) as db:
            if (
                unless_owner_held
                and db.execute(
                    "SELECT 1 FROM tasks t JOIN owners o ON o.key=t.owner_key WHERE t.id=? AND o.held",
                    (task_id,),
                ).fetchone()
            ):
                return False
            return self._transition(db, task_id, _states(from_states), to_state, columns, detail)

    def conclude(
        self, task_id: str, *, to_state: str, stop_request, follow_up=None, **fields
    ) -> str:
        """Move an active task to a terminal state together with the requeue that follows it.

        ``follow_up`` (bookkeeping such as a requeue intent) commits with the transition, so
        a runner that dies before acting on it leaves the intent to the next runner. It is
        refused as "stale" when a cancel was requested after the caller read
        ``stop_request``: a cancel always wins over a requeue. Returns "concluded",
        "stale" or "gone" (the task is no longer active).
        """
        if to_state not in TERMINAL:
            raise ValueError(f"Not a terminal task state: {to_state}")
        detail = fields.pop("detail", None)
        columns = self._columns({**fields, "bookkeeping": follow_up})
        columns.setdefault("finished_at", self._now())
        with self._connection(write=True) as db:
            row = db.execute(
                "SELECT state, stop_request FROM tasks WHERE id=?", (task_id,)
            ).fetchone()
            if row is None or row["state"] not in ACTIVE:
                return "gone"
            if (
                follow_up is not None
                and row["stop_request"] == "cancel"
                and stop_request != "cancel"
            ):
                return "stale"
            self._transition(db, task_id, tuple(ACTIVE), to_state, columns, detail)
            return "concluded"

    def settle_bookkeeping(self, task_id: str, hook: str, **fields) -> bool:
        """Clear a pending ``hook`` and set ``fields``, only while that hook is still pending."""
        columns = self._columns({**fields, "bookkeeping": None})
        assignments = ", ".join(f"{column}=?" for column in columns)
        with self._connection(write=True) as db:
            row = db.execute("SELECT bookkeeping FROM tasks WHERE id=?", (task_id,)).fetchone()
            if row is None or (_decode(row["bookkeeping"]) or {}).get("hook") != hook:
                return False
            db.execute(
                f"UPDATE tasks SET {assignments}, updated_at=? WHERE id=?",
                (*columns.values(), self._now(), task_id),
            )
            return True

    def _transition(self, db, task_id, from_states, to_state, columns, detail) -> bool:
        row = db.execute("SELECT state, attempt FROM tasks WHERE id=?", (task_id,)).fetchone()
        if row is None or row["state"] not in from_states:
            return False
        assignments = "".join(f", {column}=?" for column in columns)
        db.execute(
            f"UPDATE tasks SET state=?{assignments}, updated_at=? WHERE id=? AND state=?",
            (to_state, *columns.values(), self._now(), task_id, row["state"]),
        )
        self._event(db, task_id, row["attempt"], row["state"], to_state, detail)
        return True

    def cancel_pending(self, task_ids: Iterable[str]) -> list[str]:
        now = self._now()
        columns = {
            "exit": _encode({"reason": "cancelled"}),
            "finished_at": now,
            "waiting_reason": None,
        }
        with self._connection(write=True) as db:
            return [
                task_id
                for task_id in dict.fromkeys(task_ids)
                if self._transition(db, task_id, tuple(PENDING), "cancelled", columns, None)
            ]

    def request_stop(self, task_ids: Iterable[str], reason: str) -> list[str]:
        """Ask the runner to stop active tasks; a cancel supersedes an earlier pause.

        A cancel also reaches a concluded task that is awaiting its requeue, which the runner
        then concludes as cancelled instead.
        """
        if reason not in STOP_REASONS:
            raise ValueError(f"Unknown stop reason: {reason}")
        now = self._now()
        stopping = []
        with self._connection(write=True) as db:
            for task_id in dict.fromkeys(task_ids):
                row = db.execute(
                    "SELECT state, attempt, stop_request, bookkeeping FROM tasks WHERE id=?",
                    (task_id,),
                ).fetchone()
                if row is None:
                    continue
                awaiting = (
                    row["state"] in TERMINAL
                    and (_decode(row["bookkeeping"]) or {}).get("hook") in REQUEUE_HOOKS
                )
                if awaiting:
                    if reason == "cancel":
                        if row["stop_request"] != "cancel":
                            db.execute(
                                "UPDATE tasks SET stop_request='cancel', stop_requested_at=?, "
                                "updated_at=? WHERE id=?",
                                (now, now, task_id),
                            )
                            self._event(
                                db,
                                task_id,
                                row["attempt"],
                                row["state"],
                                row["state"],
                                {"stopRequest": "cancel"},
                            )
                        stopping.append(task_id)
                    continue
                if row["state"] in ("starting", "running"):
                    self._transition(
                        db,
                        task_id,
                        ("starting", "running"),
                        "stopping",
                        {"stop_request": reason, "stop_requested_at": now},
                        {"stopRequest": reason},
                    )
                    stopping.append(task_id)
                elif row["state"] == "stopping":
                    if row["stop_request"] == "pause" and reason == "cancel":
                        db.execute(
                            "UPDATE tasks SET stop_request='cancel', updated_at=? WHERE id=?",
                            (now, task_id),
                        )
                    if row["stop_request"] == reason or reason == "cancel":
                        stopping.append(task_id)
        return stopping

    def set_fields(self, task_id: str, **fields) -> None:
        if not fields:
            return
        columns = self._columns(fields)
        assignments = ", ".join(f"{column}=?" for column in columns)
        with self._connection(write=True) as db:
            db.execute(
                f"UPDATE tasks SET {assignments}, updated_at=? WHERE id=?",
                (*columns.values(), self._now(), task_id),
            )

    def set_waiting_reasons(self, reasons: dict) -> None:
        """Record why pending tasks wait, in one transaction; unchanged rows are not touched."""
        if not reasons:
            return
        now = self._now()
        with self._connection(write=True) as db:
            for task_id, reason in reasons.items():
                db.execute(
                    "UPDATE tasks SET waiting_reason=?, updated_at=? WHERE id=? "
                    "AND state IN ('blocked', 'queued') AND waiting_reason IS NOT ?",
                    (reason, now, task_id, reason),
                )

    def hold_owner(self, key: str, held: bool) -> dict:
        with self._connection(write=True) as db:
            db.execute(
                "UPDATE owners SET held=?, updated_at=? WHERE key=?",
                (int(bool(held)), self._now(), key),
            )
            owner = self._owner_dict(db, key)
        if owner is None:
            raise StorageError("This task owner does not exist.", "TASK_OWNER_NOT_FOUND", 404)
        return owner

    def move_owner(self, key: str, position: str) -> dict:
        """Reorder owners with live tasks; the queue_seq values are permuted, never created."""
        if position not in ("top", "up", "down", "bottom"):
            raise StorageError("Move an owner to top, up, down or bottom.", "TASK_INVALID", 422)
        with self._connection(write=True) as db:
            if db.execute("SELECT 1 FROM owners WHERE key=?", (key,)).fetchone() is None:
                raise StorageError("This task owner does not exist.", "TASK_OWNER_NOT_FOUND", 404)
            rows = db.execute(
                "SELECT key, queue_seq FROM owners o WHERE EXISTS (SELECT 1 FROM tasks t "
                f"WHERE t.owner_key=o.key AND {LIVE_OR_AWAITING_SQL}) ORDER BY queue_seq"
            ).fetchall()
            order = [row["key"] for row in rows]
            if key in order:
                index = order.index(key)
                order.pop(index)
                target = {
                    "top": 0,
                    "up": max(0, index - 1),
                    "down": min(len(order), index + 1),
                    "bottom": len(order),
                }[position]
                order.insert(target, key)
                now = self._now()
                for owner_key, seq in zip(order, [row["queue_seq"] for row in rows], strict=True):
                    db.execute(
                        "UPDATE owners SET queue_seq=?, updated_at=? WHERE key=? AND queue_seq<>?",
                        (seq, now, owner_key, seq),
                    )
            return self._owner_dict(db, key)

    @staticmethod
    def _failed_dependency(db, task_id: str) -> str | None:
        """A dependency that must succeed but finished otherwise, and will not run again."""
        for row in db.execute(
            "SELECT d.depends_on, t.state, t.bookkeeping FROM task_dependencies d "
            "JOIN tasks t ON t.id = d.depends_on WHERE d.task_id=? AND d.condition='succeeded'",
            (task_id,),
        ):
            if (
                row["state"] in TERMINAL
                and row["state"] != "succeeded"
                and (_decode(row["bookkeeping"]) or {}).get("hook") not in REQUEUE_HOOKS
            ):
                return row["depends_on"]
        return None

    def promote_ready(self) -> list[str]:
        """Queue blocked tasks whose dependencies are met; cancel those that can never be.

        A task that needs a dependency to succeed is cancelled once that dependency failed,
        was cancelled or was interrupted for good (section 4.2 of the design), and so are its
        own dependents in turn. Retrying the dependency's owner requeues them again.
        """
        now = self._now()
        promoted = []
        with self._connection(write=True) as db:
            changed = True
            while changed:
                changed = False
                for row in db.execute("SELECT id FROM tasks WHERE state='blocked'").fetchall():
                    failed = self._failed_dependency(db, row["id"])
                    if failed is not None:
                        message = f"A task it depends on did not succeed ({failed})."
                        self._transition(
                            db,
                            row["id"],
                            ("blocked",),
                            "cancelled",
                            {
                                "exit": _encode({"reason": "cancelled", "error": message}),
                                "error": message,
                                "finished_at": now,
                                "waiting_reason": None,
                            },
                            {"event": "dependency-failed", "dependency": failed},
                        )
                        changed = True
                    elif self._dependencies_met(db, row["id"]):
                        self._transition(
                            db,
                            row["id"],
                            ("blocked",),
                            "queued",
                            {"queued_at": now, "waiting_reason": None},
                            {"event": "dependencies-met"},
                        )
                        promoted.append(row["id"])
        return promoted

    # -- settings, operations, measurements, runner ---------------------------------------

    def _stored_settings(self, db) -> dict:
        return {row["key"]: _decode(row["value"]) for row in db.execute("SELECT * FROM settings")}

    def settings(self) -> dict:
        with self._connection() as db:
            return merge_settings(self._stored_settings(db))

    def update_settings(self, patch: dict) -> dict:
        with self._connection(write=True) as db:
            current = merge_settings(self._stored_settings(db))
            updated = apply_settings_patch(current, patch)
            now = self._now()
            for key in patch:
                # A null setting (e.g. automatic cpuTaskSlots) is stored as JSON null, not SQL NULL.
                value = json.dumps(
                    updated[key], sort_keys=True, separators=(",", ":"), allow_nan=False
                )
                db.execute(
                    "INSERT OR REPLACE INTO settings(key, value, updated_at) VALUES(?,?,?)",
                    (key, value, now),
                )
            return updated

    def operation(self, operation_id: str) -> dict | None:
        with self._connection() as db:
            row = db.execute(
                "SELECT * FROM operations WHERE operation_id=?", (operation_id,)
            ).fetchone()
        if row is None:
            return None
        return {
            "operationId": row["operation_id"],
            "requestHash": row["request_hash"],
            "result": _decode(row["result"]),
            "createdAt": row["created_at"],
        }

    def record_operation(self, operation_id: str, request_hash: str, result) -> None:
        with self._connection(write=True) as db:
            row = db.execute(
                "SELECT request_hash FROM operations WHERE operation_id=?", (operation_id,)
            ).fetchone()
            if row is not None:
                if row["request_hash"] != request_hash:
                    raise StorageError(
                        "This operation id was already used for a different request.",
                        "OPERATION_CONFLICT",
                        409,
                    )
                return
            db.execute(
                "INSERT INTO operations(operation_id, request_hash, result, created_at) "
                "VALUES(?,?,?,?)",
                (operation_id, request_hash, _encode(result), self._now()),
            )

    def record_measurement(self, row: dict) -> None:
        values = {column: row[name] for name, column in MEASUREMENT_COLUMNS.items() if name in row}
        values["workload"] = _encode(values.get("workload"))
        values["recorded_at"] = values.get("recorded_at") or self._now()
        with self._connection(write=True) as db:
            db.execute(
                f"INSERT INTO measurements({', '.join(values)}) VALUES({_marks(values)})",
                tuple(values.values()),
            )

    def measurements(self, *, workload_key=None, kinds=None, limit=2000) -> list[dict]:
        """Most recent first."""
        clauses, params = [], []
        if workload_key is not None:
            clauses.append("workload_key=?")
            params.append(workload_key)
        if kinds is not None:
            kinds = _states(kinds)
            if not kinds:
                return []
            clauses.append(f"kind IN ({_marks(kinds)})")
            params.extend(kinds)
        sql = "SELECT * FROM measurements"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY seq DESC LIMIT ?"
        params.append(int(limit))
        with self._connection() as db:
            rows = db.execute(sql, params).fetchall()
        result = []
        for row in rows:
            item = {name: row[column] for name, column in MEASUREMENT_COLUMNS.items()}
            item["workload"] = _decode(item["workload"])
            result.append(item)
        return result

    def runner(self) -> dict | None:
        with self._connection() as db:
            row = db.execute("SELECT * FROM runner WHERE id=1").fetchone()
        if row is None:
            return None
        value = {name: row[column] for name, column in RUNNER_COLUMNS.items()}
        value["sample"] = _decode(value["sample"])
        value["codeFiles"] = _decode(value["codeFiles"])
        value["gpuFences"] = _decode(value["gpuFences"]) or {}
        return value

    def checkpoint(self) -> bool:
        """Copy the write-ahead log into the database and truncate it; False while readers block."""
        with self._connection() as db:
            busy, _log_frames, _copied = db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        return not busy

    def write_runner(self, **fields) -> None:
        values = {}
        for name, value in fields.items():
            column = RUNNER_COLUMNS.get(name, name)
            if column not in RUNNER_COLUMNS.values():
                raise ValueError(f"Unknown runner field: {name}")
            values[column] = _encode(value) if column in RUNNER_JSON_COLUMNS else value
        if not values:
            return
        assignments = ", ".join(f"{column}=excluded.{column}" for column in values)
        with self._connection(write=True) as db:
            db.execute(
                f"INSERT INTO runner(id, {', '.join(values)}) VALUES(1, {_marks(values)}) "
                f"ON CONFLICT(id) DO UPDATE SET {assignments}",
                tuple(values.values()),
            )
