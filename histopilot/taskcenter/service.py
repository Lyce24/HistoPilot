"""API-facing Task Center service: read models, capacity settings and routed actions.

Polled reads touch only the task store, this workspace's project registry, the lease
registry (read-only), a cached host snapshot and the cancel marker of a batch whose final
collection waits. They never take project locks, probe worker runtimes or delete leases. Actions on tasks of projects registered in this
workspace go through the application services that own the science records, so batch,
compute and coordinator receipts stay consistent with the queue.
"""

import hashlib
import os
import sqlite3
import stat
import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from urllib.parse import quote

from histopilot.storage.project_lock import StorageError
from histopilot.taskcenter import capacity, estimator, launcher, leases, paths
from histopilot.taskcenter.client import TaskCenterClient, default_client
from histopilot.taskcenter.model import (
    ACTIVE,
    LIVE,
    PENDING,
    PROTOCOL_VERSION,
    STATES,
    TERMINAL,
    awaiting_requeue,
    seconds_between,
    utc_now_iso,
)
from histopilot.taskcenter.store import LIVE_OR_AWAITING_SQL, request_hash

LOG_TAIL_BYTES = 16 * 1024
DEFAULT_TASK_SECONDS = 600.0
HOST_CACHE_SECONDS = 5.0
SAMPLE_FRESH_SECONDS = 30.0
MEDIAN_CACHE_SECONDS = 30.0
RUNNER_EXIT_SECONDS = 15.0
RUNNER_RESTART_WAIT_SECONDS = 600.0
RESTART_PENDING_NOTE = "The old runner is finishing its current step; it restarts when that ends."
RETRYABLE = frozenset({"failed", "cancelled", "interrupted"})
MIL_KINDS = frozenset({"mil-fold", "mil-collect"})
REFIT_KINDS = frozenset({"refit", "predictor-refit"})
EVALUATION_KINDS = frozenset({"evaluation", "model-evaluation"})
INTERPRETATION_KINDS = frozenset({"interpretation", "model-interpretation"})
COMPUTE_KINDS = REFIT_KINDS | EVALUATION_KINDS | INTERPRETATION_KINDS
# Phase 3 records (Area D): task kind or owner kind -> record kind of its stage link.
PREPARATION_KINDS = {
    "extraction": "extraction",
    "extraction-validation": "extraction",
    "packing": "feature-pack",
    "feature-pack": "feature-pack",
    "archive": "archive",
}
STATE_ALIASES = {"live": LIVE, "history": TERMINAL, "pending": PENDING, "active": ACTIVE}
# The "live" alias lists what runs before what waits, so a short list shows running work.
LIVE_RANK = {"starting": 0, "running": 0, "stopping": 0, "queued": 1, "blocked": 2}
OWNER_ACTIONS = ("hold", "release", "stop", "cancel", "retry", "move")
TASK_ACTIONS = ("cancel", "retry")
MOVE_POSITIONS = ("top", "up", "down", "bottom")
OPERATION_PREFIX = "task-center-api:"
TASK_RESOURCE_KEYS = (
    "privateRamGb",
    "peakPrivateRamGb",
    "cpuCores",
    "meanCpuCores",
    "meanConcurrency",
    "gpuName",
    "vramGb",
    "peakVramGb",
    "sampledAt",
)
# Polled views (snapshot, rollup) re-check the runner's source files at most this often;
# the summary endpoint still checks on every call.
CODE_CHECK_SECONDS = 10.0
# Machine- and project-wide rollups report failures from this window, so an old failure
# does not keep a stage chip or the job tray asking for attention forever.
RECENT_FAILURE_SECONDS = 24 * 3600.0
HISTORY_PAGE_LIMIT = 200
EVENT_LIMIT = 200
# Record kinds as the science pages name them, and the owner, compute and group kinds the
# task store uses for the same records.
RECORD_KIND_ALIASES = {
    "refit": ("refit", "predictor-refit"),
    "evaluation": ("evaluation", "model-evaluation"),
    "interpretation": ("interpretation", "model-interpretation", "attention-interpretation"),
}
for _family in list(RECORD_KIND_ALIASES.values()):
    for _alias in _family:
        RECORD_KIND_ALIASES.setdefault(_alias, _family)
# Commands carry these environment names; anything else (for example a token) is hidden.
VISIBLE_ENV_PREFIXES = ("HISTOPILOT_", "PYTHON", "CUDA_", "OMP_", "MKL_", "OPENBLAS_")
# A progress collection that stopped short is superseded by its batch's final collection
# (an owner retry never reruns it), so an owner or record rollup does not count it.
SUPERSEDED_PROGRESS_SQL = (
    "NOT (t.kind = 'mil-collect' AND json_extract(t.adapter_data, '$.final') IS 0 "
    "AND t.state IN ('failed', 'cancelled', 'interrupted'))"
)
# The training batch of a task, as ``TaskCenterService._batch_id`` reads it.
TASK_BATCH_SQL = (
    "COALESCE(CASE WHEN t.group_kind = 'mil-batch' THEN t.group_id END, "
    "json_extract(t.adapter_data, '$.batchId'), json_extract(t.labels, '$.batchId'))"
)


def record_kinds(kind: str) -> tuple[str, ...]:
    return RECORD_KIND_ALIASES.get(kind, (kind,))


def _first_line(text) -> str:
    """The most informative line of an error: the last non-empty traceback line."""
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        return ""
    if lines[0].startswith("Traceback"):
        return lines[-1][:500]
    return lines[0][:500]


def explain_failure(task: dict) -> dict | None:
    """A plain-language cause for an unsuccessful task and whether retrying it is safe.

    ``retry`` is ``safe`` (nothing to fix first), ``check`` (look at the machine first),
    ``after-fix`` (it fails again until the cause is fixed) or ``unknown``.
    """
    state = task.get("state")
    if state not in ("failed", "interrupted", "cancelled"):
        return None
    exit_record = task.get("exit") if isinstance(task.get("exit"), dict) else {}
    reason = exit_record.get("reason")
    error = str(exit_record.get("error") or task.get("error") or "")
    code = exit_record.get("returncode")
    text = error.lower()
    line = _first_line(error)

    def result(title, cause, retry, advice=None):
        return {"title": title, "cause": cause, "retry": retry, "advice": advice, "detail": line}

    if state == "cancelled" or reason == "cancelled":
        return result(
            "Cancelled",
            "It was stopped on request. Completed work is kept.",
            "safe",
            "Retry resumes it; training resumes from its last checkpoint.",
        )
    if (
        reason == "busy"
        or code == 75
        or "project_busy" in text
        or "another operation is changing this workspace" in text
        or "blockingioerror" in text
        or "resource temporarily unavailable" in text
    ):
        return result(
            "The project was busy",
            "Another HistoPilot operation was changing this project when the task started, "
            "so it stopped without changing anything.",
            "safe",
            "Retry once the other operation has finished.",
        )
    if reason == "oom" or "out of memory" in text or "cuda error: out of memory" in text:
        return result(
            "Out of GPU memory",
            "The task needed more GPU memory than was free. It is retried once with more "
            "memory automatically.",
            "safe",
            "If it failed again, lower the parallel GPU tasks or wait for other GPU work to "
            "finish, then retry.",
        )
    if reason == "cuda_failure":
        return result(
            "GPU device failure",
            "The GPU stopped responding (a driver or device error).",
            "check",
            "Check the GPU (for example with nvidia-smi) before retrying.",
        )
    if reason == "lost" or exit_record.get("lost"):
        return result(
            "Process lost",
            "Its process ended without an exit record, for example after a reboot or a "
            "runner restart.",
            "safe",
            "Retry resumes it.",
        )
    if state == "interrupted" or reason in ("interrupted", "paused"):
        return result(
            "Interrupted",
            "It stopped before finishing, for example when the runner or machine restarted.",
            "safe",
            "Retry resumes it.",
        )
    if "no module named" in text or "modulenotfounderror" in text:
        return result(
            "A Python package is missing",
            "The worker's environment lacks a module it needs.",
            "after-fix",
            "Repair the training environment (uv sync --locked --extra training), then retry.",
        )
    if "no space left on device" in text:
        return result(
            "The disk is full",
            "A write failed because the disk has no space left.",
            "after-fix",
            "Free disk space, then retry.",
        )
    if code in (-9, 137) or exit_record.get("killed"):
        return result(
            "Killed by the system",
            "The process was killed, often because the machine ran out of RAM.",
            "check",
            "Lower the parallel tasks or close other memory-heavy programs, then retry.",
        )
    if "filenotfounderror" in text or "no such file or directory" in text:
        return result(
            "A required file is missing",
            "The task could not find a file it reads.",
            "after-fix",
            "Check that the project's slides, features and outputs are still in place.",
        )
    return result(
        "The task failed",
        line
        or (f"It exited with code {code}." if code not in (None, 0) else "It reported an error."),
        "unknown",
        "The log below shows what happened. Retrying repeats the same work.",
    )


def derived_operation_id(action: str, operation_id: str, target: str) -> str:
    """The operation id handed to an owning service for one target of a Task Center action."""
    digest = hashlib.sha256((operation_id + target).encode()).hexdigest()[:32]
    return f"task-center-{action}-{digest}"


def _unavailable(error: Exception) -> StorageError:
    return StorageError(
        f"The Task Center store is unavailable: {error}", "TASK_CENTER_UNAVAILABLE", 503
    )


def _norm(folder) -> str:
    return os.path.normpath(str(folder))


def _query(value) -> str:
    return quote(str(value), safe="")


def _log_tail(path, limit: int = LOG_TAIL_BYTES) -> tuple[str | None, bool]:
    """The end of a task log without following symlinks or reading special files."""
    if not isinstance(path, str) or not os.path.isabs(path):
        return None, False
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return None, False
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            return None, False
        start = max(0, info.st_size - limit)
        stream.seek(start)
        data = stream.read(limit)
    return data.decode("utf-8", errors="replace"), start > 0


def _attempts(events: list[dict], task: dict) -> list[dict]:
    """One row per attempt from the state-change journal: when it started and how it ended."""
    attempts: dict[int, dict] = {}
    for event in events:
        number = event.get("attempt") or 1
        row = attempts.setdefault(
            number,
            {
                "attempt": number,
                "startedAt": None,
                "endedAt": None,
                "state": None,
                "exitReason": None,
                "gpu": None,
            },
        )
        detail = event.get("detail") if isinstance(event.get("detail"), dict) else {}
        target = event.get("toState")
        if target in ("starting", "running") and row["startedAt"] is None:
            row["startedAt"] = event["at"]
            if detail.get("gpu") is not None:
                row["gpu"] = detail.get("gpu")
        if target in TERMINAL:
            row["endedAt"] = event["at"]
            row["state"] = target
            row["exitReason"] = detail.get("exitReason") or row["exitReason"]
    current = attempts.get(task.get("attempt") or 1)
    if current is not None and current["state"] is None:
        current["state"] = task.get("state")
    return [attempts[number] for number in sorted(attempts)]


def _progress_collect(task: dict) -> bool:
    return task["kind"] == "mil-collect" and (task.get("adapterData") or {}).get("final") is False


def _plan_resumes(plan: dict) -> bool:
    """Whether a retry plan (see ``TaskCenterService._retry_plan``) resumes anything now."""
    return any(plan[name] for name in ("batches", "coordinators", "computes", "direct"))


def _pausable(task: dict) -> bool:
    """Short bookkeeping tasks and long-lived services finish on their own; a pause would
    only interrupt their record writes."""
    return not (task.get("request") or {}).get("service") and task["kind"] != "mil-collect"


def _cancel_marked(task: dict) -> bool:
    folder = (task.get("adapterData") or {}).get("batchFolder")
    return (
        isinstance(folder, str)
        and os.path.isabs(folder)
        and os.path.exists(os.path.join(folder, "cancel.json"))
    )


def _busy_batches(tasks, key) -> set:
    """The batches (named by ``key(task)``) whose training service would refuse a resume.

    A fold may still run or be auto-resumed, or a collection is writing results. A batch
    that was not cancelled also stays running until its final collection records it, even
    while that collection waits behind a hold or a stopped runner.
    """
    busy, finals = set(), {}
    for task in tasks:
        if task["kind"] == "mil-fold":
            if task["state"] in LIVE or awaiting_requeue(task):
                busy.add(key(task))
        elif task["kind"] == "mil-collect":
            if task["state"] in ACTIVE:
                busy.add(key(task))
            elif task["state"] in PENDING and (task.get("adapterData") or {}).get("final") is True:
                finals.setdefault(key(task), task)
    busy.update(
        batch for batch, task in finals.items() if batch not in busy and not _cancel_marked(task)
    )
    return busy


def _owner_cancel_reaches(task: dict) -> bool:
    """Whether an owner-wide Cancel still changes this task.

    It leaves a batch's collections to record the outcome, a task stopping for a cancel is
    already being cancelled, and a concluded fold awaiting its auto-resume is cancelled
    instead of resumed.
    """
    if task["stopRequest"] == "cancel" and task["state"] not in PENDING:
        return False
    if task["kind"] == "mil-collect" and TaskCenterService._batch_id(task) is not None:
        return False
    return task["state"] in LIVE or (task["kind"] == "mil-fold" and awaiting_requeue(task))


def _default_service(name: str, store, filesystem):
    if name == "training":
        from histopilot.application.training import TrainingService

        return TrainingService(store, filesystem)
    if name == "predictors":
        from histopilot.application.experiment_predictors import ExperimentPredictorService

        return ExperimentPredictorService(store, filesystem)
    if name == "compute":
        from histopilot.application.compute_jobs import ComputeJobService

        return ComputeJobService(store, filesystem)
    if name == "refits":
        from histopilot.application.refits import RefitService

        return RefitService(store, filesystem)
    if name == "evaluations":
        from histopilot.application.evaluation_runs import EvaluationRunService

        return EvaluationRunService(store, filesystem)
    if name == "interpretations":
        from histopilot.application.interpretation import InterpretationService

        return InterpretationService(store, filesystem)
    if name == "bulk":
        from histopilot.application.bulk_evaluations import BulkEvaluationService

        return BulkEvaluationService(store, filesystem)
    if name == "experiments":
        from histopilot.application.model_experiments import ModelExperimentService

        return ModelExperimentService(store, filesystem)
    raise ValueError(f"Unknown Task Center service: {name}")


class _Errors:
    """Run every routed call of one action; report the first failure after trying all."""

    def __init__(self):
        self.first: StorageError | None = None

    def run(self, function, *args, **kwargs) -> bool:
        try:
            function(*args, **kwargs)
            return True
        except StorageError as error:
            self.first = self.first or error
            return False
        except OSError as error:
            # A project file problem, not an unavailable task store.
            self.first = self.first or StorageError(
                f"The owning record could not be updated: {error}", "TASK_ACTION_FAILED", 503
            )
            return False

    def raise_first(self) -> None:
        if self.first is not None:
            raise self.first


class TaskCenterService:
    """Workspace view of the machine-wide Task Center.

    ``services`` maps service names (training, predictors, compute, refits, evaluations,
    interpretations, bulk) to ``factory(store, filesystem)`` overrides, for tests.
    """

    def __init__(
        self,
        projects,
        filesystem,
        workspace: Path,
        *,
        client: TaskCenterClient | None = None,
        host_probe=None,
        services: dict | None = None,
    ):
        self.projects = projects
        self.filesystem = filesystem
        self.workspace = Path(workspace)
        self._client = client
        self.host_probe = host_probe or capacity.host
        self.services = dict(services or {})
        self._lock = threading.Lock()
        self._host_cache: tuple[float, dict] | None = None
        self._median_cache: tuple[float, dict] | None = None
        self._code_cache: tuple[tuple, str] | None = None
        self._code_checked: tuple[float, tuple | None, str | None] | None = None
        self._log_cache: dict[str, tuple[tuple, tuple]] = {}
        self._operations: dict[str, list] = {}
        self._restart_thread: threading.Thread | None = None

    # -- plumbing ------------------------------------------------------------------------

    @property
    def client(self) -> TaskCenterClient:
        if self._client is not None:
            return self._client
        try:
            return default_client()
        except OSError as error:
            raise _unavailable(error) from error

    @property
    def store(self):
        return self.client.store

    def _call(self, function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (OSError, sqlite3.Error) as error:
            raise _unavailable(error) from error

    @contextmanager
    def _serialized(self, operation_id: str):
        """Run requests that reuse one operation id one at a time.

        The receipt is checked before and recorded after the action, so a duplicate sent
        while the first is still acting must wait for that receipt instead of acting again.
        """
        with self._lock:
            entry = self._operations.setdefault(operation_id, [threading.Lock(), 0])
            entry[1] += 1
        try:
            with entry[0]:
                yield
        finally:
            with self._lock:
                entry[1] -= 1
                if not entry[1]:
                    self._operations.pop(operation_id, None)

    def _service(self, name: str, store):
        factory = self.services.get(name)
        if factory is not None:
            return factory(store, self.filesystem)
        return _default_service(name, store, self.filesystem)

    def _registry(self) -> dict[str, str]:
        """Resolved project folder -> project id for projects registered in this workspace.

        Reads the workspace database only; project folders and their locks are untouched.
        """
        return self._registry_details()[0]

    def _registry_details(self) -> tuple[dict[str, str], dict[str, str]]:
        """``(folder -> project id, project id -> name)`` for this workspace's projects."""
        if self.projects is None:
            return {}, {}
        from sqlalchemy import select

        from histopilot.storage.database import Record

        with self.projects.database.sessions.begin() as session:
            rows = [
                (record.id, record.payload)
                for record in session.scalars(select(Record).where(Record.kind == "project"))
            ]
        folders, names = {}, {}
        for identity, payload in rows:
            if isinstance(payload, dict) and isinstance(payload.get("storagePath"), str):
                folders[_norm(payload["storagePath"])] = identity
                if isinstance(payload.get("name"), str):
                    names[identity] = payload["name"]
        return folders, names

    def _rows(self, sql: str, params=()) -> list:
        """A read-only query on the task store for aggregate views the store has no method
        for (rollups, grouped history). Never takes a write transaction."""
        with self.store._connection() as db:
            return db.execute(sql, list(params)).fetchall()

    def _host(self) -> dict:
        """The runner's recent host sample, else a briefly cached probe (never CUDA)."""
        try:
            row = self.store.runner()
        except StorageError:
            row = None
        sample = (row or {}).get("sample") or {}
        at = sample.get("at")
        age = seconds_between(at, utc_now_iso()) if at else None
        fresh = age is not None and age <= SAMPLE_FRESH_SECONDS
        if fresh and isinstance(sample.get("host"), dict) and sample["host"].get("cpuCount"):
            return {**sample["host"], "gpus": list(sample.get("gpus") or []), "sampledAt": at}
        with self._lock:
            cached = self._host_cache
            if cached is not None and time.monotonic() - cached[0] < HOST_CACHE_SECONDS:
                return cached[1]
        try:
            value = dict(self.host_probe())
        except Exception as error:  # an unreadable host leaves capacity unknown, not broken
            value = {"gpus": [], "gpuProbeError": f"Host probe failed: {error}"}
        value["sampledAt"] = utc_now_iso()
        with self._lock:
            self._host_cache = (time.monotonic(), value)
        return value

    def _medians(self) -> dict[str, float]:
        """Median wall seconds of successful measured tasks, per workload key."""
        with self._lock:
            cached = self._median_cache
            if cached is not None and time.monotonic() - cached[0] < MEDIAN_CACHE_SECONDS:
                return cached[1]
        samples: dict[str, list[float]] = {}
        for row in self.store.measurements(limit=2000):
            key, wall = row.get("workloadKey"), row.get("wallSeconds")
            if (
                key
                and row.get("exitReason") == "ok"
                and isinstance(wall, (int, float))
                and wall > 0
            ):
                samples.setdefault(key, []).append(float(wall))
        value = {key: median(values) for key, values in samples.items()}
        with self._lock:
            self._median_cache = (time.monotonic(), value)
        return value

    def _code_hash(self, files: list[str] | None = None, *, max_age: float | None = None) -> str:
        """The current hash of the runner's code, recomputed only when a source file changes.

        ``files`` are the modules the runner recorded as loaded; without them the hash
        covers the Task Center package only (runners that predate that record).
        ``max_age`` lets polled views reuse a check made that many seconds ago instead of
        stating every module again.
        """
        from histopilot.taskcenter.runner import code_hash

        wanted = None if files is None else tuple(files)
        if max_age is not None:
            with self._lock:
                checked = self._code_checked
            if (
                checked is not None
                and checked[1] == wanted
                and time.monotonic() - checked[0] < max_age
            ):
                return checked[2]
        value = self._code_hash_now(files, code_hash)
        with self._lock:
            self._code_checked = (time.monotonic(), wanted, value)
        return value

    def _code_hash_now(self, files, code_hash) -> str:
        if files is None:
            root = Path(__file__).resolve().parent
            sources = sorted([*root.glob("*.py"), *(root / "adapters").glob("*.py")])
        else:
            sources = [Path(name) for name in files]
        signature = []
        for path in sources:
            try:
                info = path.stat()
            except OSError:
                signature.append((str(path), None, None))
                continue
            signature.append((str(path), info.st_mtime_ns, info.st_size))
        key = (None if files is None else tuple(files), tuple(signature))
        with self._lock:
            if self._code_cache is not None and self._code_cache[0] == key:
                return self._code_cache[1]
        try:
            value = code_hash() if files is None else code_hash(list(files))
        except OSError:
            # A source file replaced mid-read must not fail the polled summary.
            return self._code_cache[1] if self._code_cache is not None else None
        with self._lock:
            self._code_cache = (key, value)
        return value

    # -- read models ---------------------------------------------------------------------

    def _context(self, owners: dict | None = None) -> dict:
        """One consistent read of owners, live tasks and derived queue facts.

        ``owners`` narrows the owner read (it must include every owner with live tasks).
        """
        store = self.store
        if owners is None:
            owners = {owner["key"]: owner for owner in store.owners(live_only=False)}
        live = store.list(states=LIVE, limit=None)
        # Concluded tasks the runner may still requeue (a lost fold whose auto-resume is
        # undecided) are not finished work: the store keeps their owners in the queue.
        awaiting = {
            task["id"]: task for task in store.pending_bookkeeping() if awaiting_requeue(task)
        }
        registry, project_names = self._registry_details()
        positions = {}
        for task in live:
            if task["state"] in PENDING:
                positions[task["id"]] = len(positions) + 1
        awaiting_owners = {task["ownerKey"] for task in awaiting.values()}
        live_owners = sorted(
            (
                owner
                for owner in owners.values()
                if any(owner["counts"].get(state) for state in LIVE)
                or owner["key"] in awaiting_owners
            ),
            key=lambda owner: owner["queueSeq"],
        )
        aggregates = {}
        for task in [*live, *awaiting.values()]:
            item = aggregates.setdefault(
                task["ownerKey"],
                {
                    "lanes": {"gpu": 0, "cpu": 0},
                    "waiting": None,
                    "queued": False,
                    "batch": None,
                    "purpose": None,
                    "tasks": [],
                    "pausable": 0,
                    "cancellable": 0,
                },
            )
            item["tasks"].append(task)
            if _owner_cancel_reaches(task):
                item["cancellable"] += 1
            if task["id"] not in awaiting:
                lane = (task["request"] or {}).get("lane", "cpu")
                item["lanes"][lane] = item["lanes"].get(lane, 0) + 1
            if task["state"] in ("starting", "running") and _pausable(task):
                item["pausable"] += 1
            if task["state"] == "queued" and not item["queued"]:
                item["queued"] = True
                item["waiting"] = task["waitingReason"]
            group = task["group"] or {}
            if group.get("kind") == "mil-batch" and item["batch"] is None:
                item["batch"] = group.get("id")
            # Compute owners carry no labels; an inference evaluation says so on its task.
            item["purpose"] = item["purpose"] or (task["labels"] or {}).get("purpose")
        return {
            "owners": owners,
            "live": live,
            "awaiting": awaiting,
            "registry": registry,
            "projectNames": project_names,
            "purposes": {},
            "retryable": None,
            "positions": positions,
            "ownerPositions": {owner["key"]: index + 1 for index, owner in enumerate(live_owners)},
            "liveOwners": len(live_owners),
            "aggregates": aggregates,
            "busyBatches": _busy_batches(
                [*live, *awaiting.values()],
                lambda task: (_norm(task["projectFolder"]), self._batch_id(task)),
            ),
        }

    def _project(self, owner: dict | None, registry: dict) -> str | None:
        if not owner:
            return None
        return registry.get(_norm(owner["projectFolder"]))

    def _eta(self, live: list[dict], owners: dict, slots: int) -> tuple[dict, dict]:
        """Queue drain estimate: expected GPU seconds over GPU slots, filled in queue order."""
        medians = self._medians()
        now = utc_now_iso()
        slots = max(1, slots)
        total, measured, counted = 0.0, True, 0
        running_left: dict[str, float] = {}
        pending = []
        for task in live:
            request = task["request"] or {}
            if request.get("lane") != "gpu":
                continue
            expected = medians.get(request.get("workloadKey"))
            if expected is None:
                expected, measured = DEFAULT_TASK_SECONDS, False
            counted += 1
            if task["state"] in ACTIVE:
                elapsed = seconds_between(task["startedAt"], now) or 0.0
                remaining = max(0.0, expected - elapsed)
                total += remaining
                key = task["ownerKey"]
                running_left[key] = max(running_left.get(key, 0.0), remaining)
            else:
                total += expected
                pending.append((task["ownerKey"], expected))
        summary = {
            "seconds": round(total / slots),
            "basis": "measured" if counted and measured else "estimated",
        }
        # A pending task starts when the work ahead of it drains through the slots and
        # then needs its own full duration; held owners do not start at all.
        ahead = total - sum(expected for _key, expected in pending)
        per_owner: dict[str, float] = dict(running_left)
        for key, expected in pending:
            if (owners.get(key) or {}).get("held"):
                per_owner.pop(key, None)
                continue
            per_owner[key] = max(per_owner.get(key, 0.0), ahead / slots + expected)
            ahead += expected
        held = {key for key, owner in owners.items() if owner.get("held")}
        return summary, {key: round(value) for key, value in per_owner.items() if key not in held}

    def _task_link(
        self, task: dict, owner: dict | None, project_id: str | None, context: dict | None = None
    ) -> str | None:
        if project_id is None:
            return None
        labels = task.get("labels") or {}
        data = task.get("adapterData") or {}
        group = task.get("group") or {}
        owner = owner or {}
        target = None
        if task["kind"] in MIL_KINDS:
            batch = (
                (group.get("id") if group.get("kind") == "mil-batch" else None)
                or labels.get("batchId")
                or data.get("batchId")
            )
            experiment = labels.get("experimentId") or (
                owner.get("id") if owner.get("kind") == "experiment" else None
            )
            if batch or experiment:
                target = self._experiment_hash(experiment or f"legacy-{batch}", batch)
        elif task["kind"] == "compute-job":
            kind = data.get("kind") or group.get("kind")
            record = data.get("recordId") or group.get("id")
            experiment = labels.get("experimentId") or (
                owner.get("id") if owner.get("kind") == "experiment" else None
            )
            if experiment and kind in REFIT_KINDS:
                # An experiment's predictor refits are read on the experiment, not in the
                # historical refit list.
                target = self._experiment_hash(experiment, None)
            else:
                target = self._compute_hash(kind, record, labels)
        elif task["kind"] == "predictor-coordinator":
            experiment = self._experiment_id(task, owner)
            target = self._experiment_hash(experiment, None) if experiment else None
        elif task["kind"] == "bulk-submit":
            batch = self._bulk_id(task, owner)
            purpose = self._owner_purpose(task["ownerKey"], context)
            target = self._bulk_hash(batch, purpose)
        elif task["kind"] in PREPARATION_KINDS:
            target = self._preparation_hash(
                PREPARATION_KINDS[task["kind"]], labels.get("recordId") or group.get("id"), labels
            )
        if target is None:
            return self._owner_link(owner, project_id, None, labels.get("purpose"), context)
        return f"?project={_query(project_id)}{target}"

    @staticmethod
    def _experiment_hash(experiment: str, batch: str | None) -> str:
        value = f"#experiments?experiment={_query(experiment)}&tab=runs"
        return value + (f"&batch={_query(batch)}" if batch else "")

    @staticmethod
    def _bulk_hash(batch: str | None, purpose: str | None) -> str:
        page = "inference" if purpose == "inference" else "evaluation"
        return f"#{page}?batch={_query(batch)}" if batch else f"#{page}"

    def _owner_purpose(self, key: str | None, context: dict | None) -> str | None:
        """An evaluation batch's purpose, which only its member tasks carry (inference)."""
        if not key:
            return None
        cache = context["purposes"] if context is not None else {}
        if key not in cache:
            aggregate = ((context or {}).get("aggregates") or {}).get(key) or {}
            purpose = aggregate.get("purpose")
            if purpose is None:
                rows = self._rows(
                    "SELECT json_extract(labels, '$.purpose') AS purpose FROM tasks "
                    "WHERE owner_key=? AND json_extract(labels, '$.purpose') IS NOT NULL LIMIT 1",
                    (key,),
                )
                purpose = rows[0]["purpose"] if rows else None
            cache[key] = purpose
        return cache[key]

    @staticmethod
    def _preparation_hash(kind: str, record: str | None, labels: dict) -> str | None:
        """Stage links of extraction, feature-pack and archive records."""
        if kind == "archive":
            return "#operations"
        if not record:
            return "#features"
        if kind == "extraction":
            return f"#features?extraction={_query(record)}"
        source = labels.get("featureSetId")
        prefix = f"source={_query(source)}&" if source else ""
        return f"#features?{prefix}packing={_query(record)}"

    @staticmethod
    def _compute_hash(kind, record, labels: dict) -> str | None:
        if not record:
            return None
        if kind in REFIT_KINDS:
            return f"#post-development?tab=refits&refit={_query(record)}"
        if kind in EVALUATION_KINDS:
            page = "inference" if labels.get("purpose") == "inference" else "evaluation"
            return f"#{page}?evaluation={_query(record)}"
        if kind in INTERPRETATION_KINDS:
            return f"#interpretation?interpretation={_query(record)}"
        return None

    def _owner_link(
        self,
        owner: dict,
        project_id: str | None,
        batch: str | None,
        purpose: str | None = None,
        context: dict | None = None,
    ) -> str | None:
        if project_id is None or not owner:
            return None
        kind, identity = owner.get("kind"), owner.get("id")
        labels = {"purpose": purpose, **(owner.get("labels") or {})}
        if kind == "experiment":
            target = self._experiment_hash(identity, batch)
        elif kind == "mil-batch":
            target = self._experiment_hash(
                labels.get("experimentId") or f"legacy-{identity}", identity
            )
        elif kind in COMPUTE_KINDS:
            target = self._compute_hash(kind, identity, labels)
        elif kind == "evaluation-batch":
            target = self._bulk_hash(
                identity, labels.get("purpose") or self._owner_purpose(owner.get("key"), context)
            )
        elif kind in PREPARATION_KINDS:
            target = self._preparation_hash(PREPARATION_KINDS[kind], identity, labels)
        else:
            target = None
        return f"?project={_query(project_id)}{target}" if target else None

    def _task_json(self, task: dict, context: dict) -> dict:
        owner = context["owners"].get(task["ownerKey"])
        if owner is None:
            # A narrowed context (snapshot, one task) reads other owners on demand.
            owner = context["owners"][task["ownerKey"]] = self.store.owner(task["ownerKey"])
        project_id = self._project(owner, context["registry"])
        same = project_id is not None
        request = task["request"] or {}
        resources = task["resources"]
        exit_record = task["exit"]
        state = task["state"]
        # The runner is still deciding whether a concluded task awaiting its requeue runs
        # again; a cancel reaches such a fold (it is then cancelled instead).
        awaiting = awaiting_requeue(task)
        retry = same and state in RETRYABLE and not awaiting
        if retry and task["kind"] in MIL_KINDS:
            # A batch resumes as a whole; the owning service refuses while it still runs.
            batch = (_norm(task["projectFolder"]), self._batch_id(task))
            retry = batch not in context["busyBatches"]
        cancel = same and task["stopRequest"] != "cancel"
        cancel = cancel and (state in LIVE or (awaiting and task["kind"] == "mil-fold"))
        return {
            "id": task["id"],
            "kind": task["kind"],
            "title": task["title"],
            "state": state,
            "attempt": task["attempt"],
            "priority": task["priority"],
            "lane": request.get("lane"),
            "gpu": task["gpu"],
            "owner": {
                "key": task["ownerKey"],
                "kind": (owner or {}).get("kind"),
                "id": (owner or {}).get("id"),
                "title": (owner or {}).get("title"),
                "projectId": (owner or {}).get("projectId"),
                "projectName": context["projectNames"].get(project_id) if same else None,
                "projectFolder": (owner or {}).get("projectFolder", task["projectFolder"]),
                "sameWorkspace": same,
                "held": bool((owner or {}).get("held")),
                "queueSeq": (owner or {}).get("queueSeq"),
            },
            "group": task["group"],
            "labels": task["labels"],
            "queuePosition": context["positions"].get(task["id"]),
            "waitingReason": task["waitingReason"],
            "progress": task["progress"],
            "resources": {key: resources.get(key) for key in TASK_RESOURCE_KEYS}
            if isinstance(resources, dict)
            else None,
            "request": {
                key: request.get(key)
                for key in ("lane", "cpuThreads", "dataWorkers", "ramGb", "vramGb", "service")
            },
            "exit": {
                "reason": exit_record.get("reason"),
                "returncode": exit_record.get("returncode"),
                "error": exit_record.get("error") or task["error"],
                "stopReason": exit_record.get("stopReason"),
                "killed": bool(exit_record.get("killed")),
                "lost": bool(exit_record.get("lost")),
                "signalled": bool(exit_record.get("signalled")),
            }
            if isinstance(exit_record, dict)
            else None,
            "error": task["error"],
            "failure": explain_failure(task),
            "stopRequest": task["stopRequest"],
            "stopRequestedAt": task.get("stopRequestedAt"),
            "createdAt": task["createdAt"],
            "queuedAt": task["queuedAt"],
            "startedAt": task["startedAt"],
            "finishedAt": task["finishedAt"],
            "updatedAt": task["updatedAt"],
            "link": self._task_link(task, owner, project_id, context),
            "logPath": (task["command"] or {}).get("log"),
            "awaitingRequeue": awaiting,
            "actions": {"cancel": cancel, "retry": retry},
        }

    def _owner_json(self, owner: dict, context: dict) -> dict:
        project_id = self._project(owner, context["registry"])
        same = project_id is not None
        counts = {state: int(owner["counts"].get(state, 0)) for state in STATES}
        aggregate = context["aggregates"].get(owner["key"]) or {}
        position = context["ownerPositions"].get(owner["key"])
        # Aggregated tasks are the live ones and those awaiting a requeue.
        pending_work = bool(aggregate.get("tasks")) or any(counts[state] for state in LIVE)
        retryable = sum(counts[state] for state in RETRYABLE)
        waiting = aggregate.get("waiting")
        if waiting is None and pending_work:
            if owner["held"]:
                waiting = "Held. Its queued tasks start after it is released."
            elif counts["blocked"] and not (
                counts["queued"] or any(counts[state] for state in ACTIVE)
            ):
                waiting = "Waiting for earlier tasks to finish."
        return {
            "key": owner["key"],
            "kind": owner["kind"],
            "id": owner["id"],
            "title": owner["title"],
            "projectId": owner["projectId"],
            "projectName": context["projectNames"].get(project_id) if same else None,
            "projectFolder": owner["projectFolder"],
            "workspace": owner["workspace"],
            "sameWorkspace": same,
            "held": owner["held"],
            "queueSeq": owner["queueSeq"],
            "position": position,
            "counts": counts,
            "lanes": aggregate.get("lanes") or {"gpu": 0, "cpu": 0},
            "waitingReason": waiting,
            "purpose": aggregate.get("purpose"),
            "labels": owner["labels"],
            "createdAt": owner["createdAt"],
            "updatedAt": owner["updatedAt"],
            "etaSeconds": context.get("ownerEta", {}).get(owner["key"]),
            "link": self._owner_link(
                owner, project_id, aggregate.get("batch"), aggregate.get("purpose"), context
            ),
            "actions": {
                "hold": same and pending_work and not owner["held"],
                "release": same and owner["held"],
                "stop": same and aggregate.get("pausable", 0) > 0,
                "cancel": same and aggregate.get("cancellable", 0) > 0,
                "retry": same
                and retryable > 0
                and self._owner_retryable(owner, aggregate, context),
                "moveUp": same and position is not None and position > 1,
                "moveDown": same and position is not None and position < context["liveOwners"],
            },
        }

    def _owner_retryable(self, owner: dict, aggregate: dict, context: dict | None = None) -> bool:
        """Whether an owner retry would resume something now, rather than wait on a batch
        or resume nothing (for example when only a superseded progress collection failed)."""
        live = aggregate.get("tasks") or []
        if context is None:
            failed = self.store.list(owner_key=owner["key"], states=RETRYABLE, limit=None)
        else:
            if context.get("retryable") is None:
                # One read for every live owner instead of one per owner and poll.
                keys = sorted(context["aggregates"])
                states = sorted(RETRYABLE)
                rows = (
                    self._rows(
                        f"SELECT * FROM tasks WHERE state IN ({','.join('?' * len(states))}) "
                        f"AND owner_key IN ({','.join('?' * len(keys))})",
                        [*states, *keys],
                    )
                    if keys
                    else []
                )
                grouped: dict[str, list] = {key: [] for key in keys}
                for row in rows:
                    item = self.store._task(row)
                    grouped.setdefault(item["ownerKey"], []).append(item)
                context["retryable"] = grouped
            if owner["key"] not in context["retryable"]:
                # An owner without live work (history, one focused owner) reads its own.
                context["retryable"][owner["key"]] = self.store.list(
                    owner_key=owner["key"], states=RETRYABLE, limit=None
                )
            failed = context["retryable"][owner["key"]]
        cache = context.setdefault("trashedBatches", {}) if context is not None else {}
        trashed = self._trashed_batches(owner, cache)
        tasks = self._without_batches([*live, *failed], lambda batch: batch not in trashed)
        return _plan_resumes(self._retry_plan(tasks, owner, whole_owner=True))

    def _trashed_batches(self, owner: dict, cache: dict) -> set:
        """Batches in the Trash of an experiment owner's project, from its lifecycle
        metadata: a lock-free file read, once per project and request, never the project
        store. A polled read cannot see the experiment's submission, but a submitted
        experiment only ever ran submitted batches, so the Trash is what an owner retry
        (``_current_tasks``) leaves out beyond them. Empty when unknown."""
        folder, project_id = owner.get("projectFolder"), owner.get("projectId")
        if owner.get("kind") != "experiment" or not folder or not project_id:
            return set()
        if folder not in cache:
            from histopilot.storage.lifecycle import LifecycleStore

            try:
                records = LifecycleStore(Path(folder), project_id).read()["records"]
            except (StorageError, OSError):
                records = {}
            cache[folder] = {
                key.removeprefix("configuration:")
                for key, item in records.items()
                if key.startswith("configuration:") and item.get("state") == "trashed"
            }
        return cache[folder]

    def _without_batches(self, tasks, keep) -> list:
        """``tasks`` without the finished ones whose batch ``keep`` refuses. Live work and
        tasks of no batch always stay, as in the rollup's ``batchIds`` filter."""
        return [
            task
            for task in tasks
            if task["state"] in LIVE
            or awaiting_requeue(task)
            or (batch := self._batch_id(task)) is None
            or keep(batch)
        ]

    def _current_tasks(self, tasks, owner, project_id) -> list:
        """An experiment owner's tasks without the finished ones of batches it no longer
        resumes (trashed, or outside its submission), the batches its status counts
        (``ModelExperimentService.task_batch_ids``); all of them when that is unknown."""
        if (owner or {}).get("kind") != "experiment":
            return tasks
        try:
            experiments = self._service("experiments", self._project_store(project_id))
            current = experiments.task_batch_ids(owner["id"])
        except (StorageError, OSError, ValueError, KeyError, TypeError):
            return tasks  # the training service still refuses what it cannot resume
        return self._without_batches(tasks, lambda batch: batch in current)

    def _runner_view(self, *, code_max_age: float | None = None) -> dict:
        store = self.store
        row = store.runner() or {}
        alive = launcher.runner_alive(store.path.parent / "runner.lock")
        recorded = row.get("codeHash")
        files = row.get("codeFiles")
        # The runner records every module it loaded; any of them changing makes it stale.
        files = [str(name) for name in files] if isinstance(files, list) and files else None
        checkout = self._code_hash(files, max_age=code_max_age) if recorded else None
        # An unknown hash on either side never asks the user to restart the runner.
        current = (
            recorded == checkout and row.get("protocol") in (None, PROTOCOL_VERSION)
            if recorded and checkout
            else True
        )
        from histopilot.taskcenter.runner import foreign_checkout

        # A runner started from another checkout runs that checkout's code (Area A, 1.14).
        other = foreign_checkout(row)
        return {
            "alive": alive,
            "heartbeatAt": row.get("heartbeatAt"),
            "startedAt": row.get("startedAt"),
            "pid": row.get("pid"),
            "state": (row.get("state") or "running") if alive else "stopped",
            "message": row.get("message"),
            "codeHash": recorded,
            "codeCurrent": current and not other,
            "autostart": paths.autostart_enabled(),
            **({"checkoutRoot": row["checkoutRoot"]} if row.get("checkoutRoot") else {}),
            **({"otherCheckout": other} if other else {}),
        }

    def _capacity_view(self, settings, host, live) -> tuple[dict, dict]:
        running = [
            task
            for task in live
            if task["state"] in ACTIVE and (task.get("bookkeeping") or {}).get("hook") != "on_exit"
        ]
        own = {task["id"] for task in live if task["state"] in ACTIVE}
        lease_error = None
        try:
            foreign = leases.foreign(leases.read_leases(), own)
        except (StorageError, OSError) as error:
            foreign, lease_error = [], str(error)
        effective = capacity.effective(settings, host)
        usage = capacity.usage(running, foreign, host)
        gpus = []
        for gpu in host.get("gpus") or []:
            index = gpu.get("index")
            used = usage["gpus"].get(index) or {}
            gpus.append(
                {
                    "index": index,
                    "name": gpu.get("name"),
                    "slots": effective["gpuSlots"].get(index),
                    "usedSlots": used.get("slots", 0),
                    "foreignSlots": used.get("foreign", 0),
                    "totalMemoryGb": gpu.get("totalMemoryGb"),
                    "usedMemoryGb": gpu.get("usedMemoryGb"),
                    "freeMemoryGb": gpu.get("freeMemoryGb"),
                    "committedMemoryGb": round(used.get("committedVramGb", 0.0), 3),
                    "reserveMemoryGb": effective["vramReserves"].get(index),
                    "utilizationPercent": gpu.get("utilizationPercent"),
                    "exclusive": bool(used.get("exclusive")),
                }
            )
        view = {
            "gpus": gpus,
            "cpu": {
                "logical": host.get("cpuCount"),
                "physical": host.get("physicalCpuCount"),
                "committedThreads": usage["cpu"]["committedThreads"],
                "reserveThreads": effective["reserves"]["cpuThreads"],
                "cpuTaskSlots": effective["cpuTaskSlots"],
                "usedCpuTasks": usage["cpu"]["cpuTasks"],
            },
            "ram": {
                "totalGb": host.get("totalRamGb"),
                "availableGb": host.get("availableRamGb"),
                "reserveGb": effective["reserves"]["ramGb"],
                "committedGb": round(usage["ram"]["committedGb"], 3),
            },
            "sampledAt": host.get("sampledAt"),
        }
        if host.get("gpuProbeError"):
            view["gpuProbeError"] = host["gpuProbeError"]
        foreign_view = [
            {
                "kind": lease.get("kind") or "legacy",
                "gpu": lease.get("gpu"),
                "cpus": lease.get("cpus"),
                "ramGb": lease.get("ramGb"),
                "runsPerGpu": lease.get("runsPerGpu"),
                # Which legacy batch or run holds the lease, when it says.
                **{key: lease[key] for key in ("batchId", "runId") if lease.get(key)},
            }
            for lease in foreign
        ]
        return view, {
            "foreignLeases": foreign_view,
            "effective": effective,
            "leaseError": lease_error,
        }

    def summary(self) -> dict:
        return self._call(self._summary)

    def _summary(self) -> dict:
        store = self.store
        live = store.list(states=LIVE, limit=None)
        owners = {owner["key"]: owner for owner in store.owners(live_only=True)}
        return self._summary_from(live, owners)[0]

    def _summary_from(self, live, owners, *, code_max_age=None) -> tuple[dict, dict]:
        """The summary for already-read live tasks and live owners, and the per-owner ETAs."""
        store = self.store
        settings = store.settings()
        host = self._host()
        view, extra = self._capacity_view(settings, host, live)
        slots = sum(extra["effective"]["gpuSlots"].values()) or settings["defaultGpuSlots"]
        eta, owner_eta = self._eta(live, owners, slots)
        result = {
            "runner": self._runner_view(code_max_age=code_max_age),
            "paused": settings["paused"],
            "capacity": view,
            "counts": store.counts(),
            "recentFailures": self._recent_failures(),
            "eta": eta,
            "foreignLeases": extra["foreignLeases"],
            "workspace": str(self.workspace),
            "updatedAt": utc_now_iso(),
        }
        if extra["leaseError"]:
            result["leaseError"] = extra["leaseError"]
        return result, owner_eta

    def _recent_failures(self, where: str = "", params=()) -> int:
        """Failed or interrupted tasks that ended within the recent-failure window."""
        since = datetime.fromtimestamp(time.time() - RECENT_FAILURE_SECONDS, UTC).isoformat()
        rows = self._rows(
            "SELECT COUNT(*) AS n FROM tasks t JOIN owners o ON o.key = t.owner_key "
            "WHERE t.state IN ('failed','interrupted') "
            "AND COALESCE(t.finished_at, t.updated_at) >= ?" + (f" AND ({where})" if where else ""),
            [since, *params],
        )
        return int(rows[0]["n"]) if rows else 0

    def tasks(
        self, *, state=None, owner=None, project=None, kind=None, limit=200, offset=None
    ) -> dict:
        return self._call(
            self._tasks,
            state=state,
            owner=owner,
            project=project,
            kind=kind,
            limit=limit,
            offset=offset,
        )

    @staticmethod
    def _states(value) -> tuple | None:
        if value is None or not str(value).strip():
            return None
        states = set()
        for item in str(value).split(","):
            item = item.strip()
            if not item:
                continue
            if item in STATE_ALIASES:
                states |= STATE_ALIASES[item]
            elif item in STATES:
                states.add(item)
            else:
                raise StorageError(
                    f"Unknown task state filter: {item}.", "TASK_CENTER_FILTER_INVALID", 422
                )
        return tuple(sorted(states)) if states else None

    def _project_folders(self, project: str, registry: dict, owners=None) -> set[str]:
        """Folders of a project given by id (this workspace's registry, plus owners that
        recorded that id) or by absolute folder."""
        if os.path.isabs(project):
            return {_norm(project)}
        folders = {folder for folder, identity in registry.items() if identity == project}
        if owners is None:
            rows = self._rows(
                "SELECT DISTINCT project_folder FROM owners WHERE project_id=?", (project,)
            )
            folders |= {_norm(row["project_folder"]) for row in rows}
        else:
            folders |= {
                _norm(item["projectFolder"]) for item in owners if item["projectId"] == project
            }
        return folders

    def _tasks(self, *, state, owner, project, kind, limit, offset=None) -> dict:
        store = self.store
        states = self._states(state)
        kinds = [item.strip() for item in (kind or "").split(",") if item.strip()] or None
        order = "recent" if states and set(states) <= TERMINAL else "queue"
        live_first = "live" in {item.strip() for item in str(state or "").split(",")}
        # Only live owners decide queue positions; other owners are read as rows need them.
        context = self._context({item["key"]: item for item in store.owners(live_only=True)})
        folders = None
        if project:
            folders = self._project_folders(project, context["registry"])
        paged = offset is not None
        offset = offset or 0
        if folders is not None and not folders:
            return {"tasks": [], **({"offset": offset, "hasMore": False} if paged else {})}
        single = next(iter(folders)) if folders and len(folders) == 1 else None
        window = limit + offset + 1
        rows = store.list(
            states=states,
            owner_key=owner or None,
            project_folder=single,
            kinds=kinds,
            limit=None if live_first or (folders and single is None) else window,
            order=order,
        )
        if folders and single is None:
            rows = [row for row in rows if _norm(row["projectFolder"]) in folders]
        if live_first:
            # Stable: queue order is kept within running, queued and blocked tasks.
            rows = sorted(rows, key=lambda row: LIVE_RANK.get(row["state"], len(LIVE_RANK)))
        more = len(rows) > offset + limit
        rows = rows[offset : offset + limit]
        result = {"tasks": [self._task_json(row, context) for row in rows]}
        if paged:
            result.update(offset=offset, hasMore=more)
        return result

    def task(self, task_id: str) -> dict:
        return self._call(self._task, task_id)

    def _require_task(self, task_id: str) -> dict:
        task = self.store.get(task_id)
        if task is None:
            raise StorageError("This task does not exist.", "TASK_NOT_FOUND", 404)
        return task

    def _task(self, task_id: str) -> dict:
        task = self._require_task(task_id)
        store = self.store
        owners = {item["key"]: item for item in store.owners(live_only=True)}
        context = self._context(owners)
        command = task["command"] or {}
        tail, truncated, size = self._cached_log_tail(command.get("log"))
        events = store.events(task_id, limit=EVENT_LIMIT)
        dependencies = store.dependencies(task_id)
        titles = self._titles([item["task"] for item in dependencies])
        dependents = self._dependents(task_id)
        return {
            **self._task_json(task, context),
            "command": {
                "argv": list(command.get("argv") or []),
                "cwd": command.get("cwd"),
                "env": {
                    key: value
                    for key, value in (command.get("env") or {}).items()
                    if key.startswith(VISIBLE_ENV_PREFIXES)
                },
                "log": command.get("log"),
                "progress": command.get("progress"),
                "result": command.get("result"),
            },
            "pid": (task.get("process") or {}).get("pid")
            if isinstance(task.get("process"), dict)
            else None,
            "sessionName": task.get("sessionName"),
            "attempts": _attempts(events, task),
            "events": events,
            "dependencies": dependencies,
            "dependents": dependents,
            "taskTitles": {
                **titles,
                **{item["task"]: item["title"] for item in dependents},
            },
            "logTail": tail,
            "logTruncated": truncated,
            "logSize": size,
        }

    def _titles(self, task_ids) -> dict[str, str]:
        task_ids = list(task_ids)
        if not task_ids:
            return {}
        rows = self._rows(
            f"SELECT id, title, state FROM tasks WHERE id IN ({','.join('?' * len(task_ids))})",
            task_ids,
        )
        return {row["id"]: row["title"] for row in rows}

    def _dependents(self, task_id: str) -> list[dict]:
        rows = self._rows(
            "SELECT t.id, t.title, t.state FROM task_dependencies d JOIN tasks t ON t.id = d.task_id "
            "WHERE d.depends_on=? ORDER BY t.plan_order, t.id LIMIT 50",
            (task_id,),
        )
        return [{"task": row["id"], "title": row["title"], "state": row["state"]} for row in rows]

    def _cached_log_tail(self, path) -> tuple[str | None, bool, int | None]:
        """The log tail, re-read only when the file changed since the last poll."""
        if not isinstance(path, str) or not os.path.isabs(path):
            return None, False, None
        try:
            info = os.stat(path, follow_symlinks=False)
        except OSError:
            return None, False, None
        signature = (info.st_ino, info.st_size, info.st_mtime_ns)
        with self._lock:
            cached = self._log_cache.get(path)
        if cached is not None and cached[0] == signature:
            return cached[1]
        tail, truncated = _log_tail(path)
        value = (tail, truncated, info.st_size if tail is not None else None)
        with self._lock:
            if len(self._log_cache) > 64:
                self._log_cache.clear()
            self._log_cache[path] = (signature, value)
        return value

    def log_file(self, task_id: str):
        """An open binary stream over a task's whole log (no symlinks, regular files only)."""
        task = self._call(self._require_task, task_id)
        path = (task["command"] or {}).get("log")
        if not isinstance(path, str) or not os.path.isabs(path):
            raise StorageError("This task has no log.", "TASK_LOG_NOT_FOUND", 404)
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except OSError as error:
            raise StorageError(
                f"The task log cannot be read: {error.strerror or error}", "TASK_LOG_NOT_FOUND", 404
            ) from error
        stream = os.fdopen(descriptor, "rb")
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            stream.close()
            raise StorageError("The task log is not a regular file.", "TASK_LOG_NOT_FOUND", 404)
        return stream, os.path.basename(path)

    def owners(self, *, scope="live") -> dict:
        if scope not in ("live", "all"):
            raise StorageError("Owner scope is live or all.", "TASK_CENTER_FILTER_INVALID", 422)
        return self._call(self._owners, scope)

    def owner(self, key: str) -> dict:
        return self._call(self._owner_view, key)

    def _owner_context(self, owners: dict | None = None) -> dict:
        context = self._context(owners)
        settings = self.store.settings()
        slots = sum(capacity.effective(settings, self._host())["gpuSlots"].values())
        _eta, context["ownerEta"] = self._eta(
            context["live"], context["owners"], slots or settings["defaultGpuSlots"]
        )
        return context

    def _owners(self, scope: str) -> dict:
        context = self._owner_context()
        owners = sorted(context["owners"].values(), key=lambda owner: owner["queueSeq"])
        if scope == "live":
            owners = [owner for owner in owners if owner["key"] in context["ownerPositions"]]
        return {"owners": [self._owner_json(owner, context) for owner in owners]}

    def _owner_view(self, key: str) -> dict:
        """One owner, reading only the owners that decide its queue position."""
        store = self.store
        owner = store.owner(key)
        if owner is None:
            raise StorageError("This task owner does not exist.", "TASK_OWNER_NOT_FOUND", 404)
        owners = {item["key"]: item for item in store.owners(live_only=True)}
        owners[key] = owner
        return self._owner_json(owner, self._owner_context(owners))

    # -- polled page views -----------------------------------------------------------------

    def snapshot(self) -> dict:
        """Summary, running tasks and the live queue from one consistent read.

        The Task Center page polls this instead of summary, running tasks and owners
        separately; only live owners are read, never the whole owner history.
        """
        return self._call(self._snapshot)

    def _snapshot(self) -> dict:
        store = self.store
        owners = {item["key"]: item for item in store.owners(live_only=True)}
        context = self._context(dict(owners))
        live = context["live"]
        summary, context["ownerEta"] = self._summary_from(
            live, owners, code_max_age=CODE_CHECK_SECONDS
        )
        queue = sorted(
            (owner for owner in owners.values() if owner["key"] in context["ownerPositions"]),
            key=lambda owner: owner["queueSeq"],
        )
        return {
            "summary": summary,
            "running": [self._task_json(task, context) for task in live if task["state"] in ACTIVE],
            "owners": [self._owner_json(owner, context) for owner in queue],
            "pendingCount": sum(1 for task in live if task["state"] in PENDING),
            "updatedAt": summary["updatedAt"],
        }

    def history(self, *, project=None, kind=None, state=None, limit=25, offset=0) -> dict:
        return self._call(
            self._history, project=project, kind=kind, state=state, limit=limit, offset=offset
        )

    def _history(self, *, project, kind, state, limit, offset) -> dict:
        """Finished tasks grouped by owner, newest first, with server-side filters and paging."""
        states = [item for item in (self._states(state) or sorted(TERMINAL)) if item in TERMINAL]
        if not states:
            return {"groups": [], "total": 0, "offset": offset, "limit": limit}
        registry, _names = self._registry_details()
        clauses = [f"t.state IN ({','.join('?' * len(states))})"]
        params: list = list(states)
        if project:
            folders = sorted(self._project_folders(project, registry))
            if not folders:
                return {"groups": [], "total": 0, "offset": offset, "limit": limit}
            clauses.append(f"t.project_folder IN ({','.join('?' * len(folders))})")
            params.extend(folders)
        kinds = [item.strip() for item in (kind or "").split(",") if item.strip()]
        if kinds:
            clauses.append(f"t.kind IN ({','.join('?' * len(kinds))})")
            params.extend(kinds)
        where = " AND ".join(clauses)
        total = self._rows(
            f"SELECT COUNT(DISTINCT t.owner_key) AS n FROM tasks t WHERE {where}", params
        )
        rows = self._rows(
            "SELECT t.owner_key AS owner_key, COUNT(*) AS n, "
            "SUM(t.state='succeeded') AS succeeded, SUM(t.state='failed') AS failed, "
            "SUM(t.state='cancelled') AS cancelled, SUM(t.state='interrupted') AS interrupted, "
            "MAX(COALESCE(t.finished_at, t.updated_at)) AS last_at, MIN(t.started_at) AS first_at "
            f"FROM tasks t WHERE {where} GROUP BY t.owner_key ORDER BY last_at DESC, t.owner_key "
            "LIMIT ? OFFSET ?",
            [*params, int(limit), int(offset)],
        )
        keys = [row["owner_key"] for row in rows]
        owners = self._owners_by_key(keys)
        live = {item["key"]: item for item in self.store.owners(live_only=True)}
        context = self._owner_context({**live, **owners})
        failures = {}
        if keys:
            marks = ",".join("?" * len(keys))
            for row in self._rows(
                "SELECT * FROM (SELECT t.*, ROW_NUMBER() OVER (PARTITION BY t.owner_key "
                "ORDER BY COALESCE(t.finished_at, t.updated_at) DESC) AS rank FROM tasks t "
                f"WHERE t.owner_key IN ({marks}) AND t.state IN ('failed','interrupted')) "
                "WHERE rank = 1",
                keys,
            ):
                failures[row["owner_key"]] = self._failure_view(self.store._task(row))
        groups = []
        for row in rows:
            owner = owners.get(row["owner_key"])
            if owner is None:
                continue
            groups.append(
                {
                    "owner": self._owner_json(owner, context),
                    "finished": {
                        "total": int(row["n"]),
                        "succeeded": int(row["succeeded"] or 0),
                        "failed": int(row["failed"] or 0),
                        "cancelled": int(row["cancelled"] or 0),
                        "interrupted": int(row["interrupted"] or 0),
                    },
                    "lastFinishedAt": row["last_at"],
                    "firstStartedAt": row["first_at"],
                    "lastFailure": failures.get(row["owner_key"]),
                }
            )
        return {
            "groups": groups,
            "total": int(total[0]["n"]) if total else 0,
            "offset": offset,
            "limit": limit,
        }

    def _owners_by_key(self, keys) -> dict:
        keys = list(keys)
        if not keys:
            return {}
        marks = ",".join("?" * len(keys))
        counts: dict[str, dict] = {}
        for row in self._rows(
            f"SELECT owner_key, state, COUNT(*) AS n FROM tasks WHERE owner_key IN ({marks}) "
            "GROUP BY owner_key, state",
            keys,
        ):
            counts.setdefault(row["owner_key"], dict.fromkeys(STATES, 0))[row["state"]] = row["n"]
        return {
            row["key"]: self.store._owner(row, counts.get(row["key"], dict.fromkeys(STATES, 0)))
            for row in self._rows(f"SELECT * FROM owners WHERE key IN ({marks})", keys)
        }

    @staticmethod
    def _failure_view(task: dict) -> dict:
        explained = explain_failure(task) or {}
        exit_record = task.get("exit") if isinstance(task.get("exit"), dict) else {}
        return {
            "taskId": task["id"],
            "title": task["title"],
            "state": task["state"],
            "reason": exit_record.get("reason"),
            "message": explained.get("title"),
            "cause": explained.get("cause"),
            "retry": explained.get("retry"),
            "at": task.get("finishedAt") or task.get("updatedAt"),
        }

    # -- rollup: the stage pages' run status -----------------------------------------------

    def rollup(
        self,
        *,
        owner=None,
        owner_kind=None,
        owner_id=None,
        record_kind=None,
        record_id=None,
        project=None,
        kinds=None,
        record_ids=None,
        batch_ids=None,
    ) -> dict:
        """What a stage page shows about its runs, read from the task store only.

        Scope, in order of precedence: one owner key; an owner by kind and id; a science
        record by kind and id (task labels ``recordKind``/``computeKind`` + ``recordId``,
        or an owner of that kind and id), or several records of one kind (``record_ids``,
        comma-separated); a project (optionally some task kinds); the whole machine. Never
        takes a project lock.

        ``batch_ids`` (comma-separated) names the training batches the page still keeps (an
        experiment's retained batches and its current submission): concluded tasks of any
        other batch (trashed, or from an earlier plan) are left out, since trashing a batch
        never removes its tasks. Owner and record scopes also leave out progress collections
        that stopped short; their batch's final collection supersedes them. ``retryable``
        says whether the owner's retry (owner scopes) or a retry of the record's own tasks
        (record scopes) would resume anything; project and machine scopes report null.
        """
        ids = [item.strip() for item in (record_ids or "").split(",") if item.strip()]
        if record_id:
            ids = [record_id, *ids]
        if (owner_kind is None) != (owner_id is None) or (ids and not record_kind):
            raise StorageError(
                "Give both ownerKind and ownerId, and recordKind with any recordId.",
                "TASK_CENTER_FILTER_INVALID",
                422,
            )
        return self._call(
            self._rollup,
            owner=owner,
            owner_kind=owner_kind,
            owner_id=owner_id,
            record_kind=record_kind,
            record_ids=ids,
            project=project,
            kinds=kinds,
            batch_ids=[item.strip() for item in (batch_ids or "").split(",") if item.strip()],
        )

    def _rollup(
        self, *, owner, owner_kind, owner_id, record_kind, record_ids, project, kinds, batch_ids
    ):
        store = self.store
        registry, names = self._registry_details()
        kinds = [item.strip() for item in (kinds or "").split(",") if item.strip()] or None
        scope = {
            key: value
            for key, value in (
                ("owner", owner),
                ("ownerKind", owner_kind),
                ("ownerId", owner_id),
                ("recordKind", record_kind),
                ("recordId", record_ids[0] if len(record_ids) == 1 else None),
                ("recordIds", ",".join(record_ids) if len(record_ids) > 1 else None),
                ("project", project),
                ("kinds", ",".join(kinds) if kinds else None),
                ("batchIds", ",".join(batch_ids) if batch_ids else None),
            )
            if value
        }
        # An owner or record (or every record of one kind) reports its own failures until
        # they are retried; a project or the machine only recent ones.
        narrow = bool(owner or owner_id or record_kind)
        clauses, params = [], []
        empty = False
        folders = sorted(self._project_folders(project, registry)) if project else None
        if folders is not None and not folders:
            empty = True
        if owner:
            clauses.append("t.owner_key=?")
            params.append(owner)
        elif owner_id:
            aliases = record_kinds(owner_kind)
            rows = self._rows(
                f"SELECT key, project_folder FROM owners WHERE id=? AND kind IN "
                f"({','.join('?' * len(aliases))})",
                [owner_id, *aliases],
            )
            keys = [
                row["key"]
                for row in rows
                if folders is None or _norm(row["project_folder"]) in folders
            ]
            if not keys:
                empty = True
            clauses.append(f"t.owner_key IN ({','.join('?' * max(1, len(keys)))})")
            params.extend(keys or [""])
        elif record_kind:
            aliases = record_kinds(record_kind)
            marks = ",".join("?" * len(aliases))
            kind_match = (
                f"(json_extract(t.labels, '$.recordKind') IN ({marks}) OR "
                f"json_extract(t.labels, '$.computeKind') IN ({marks}) OR t.group_kind IN ({marks}))"
            )
            if record_ids:
                ids = ",".join("?" * len(record_ids))
                clauses.append(
                    f"((json_extract(t.labels, '$.recordId') IN ({ids}) AND {kind_match}) "
                    f"OR (o.id IN ({ids}) AND o.kind IN ({marks})))"
                )
                params.extend([*record_ids, *aliases, *aliases, *aliases, *record_ids, *aliases])
            else:
                # Every record of this kind (for example all refits of a project).
                clauses.append(f"({kind_match} OR o.kind IN ({marks}))")
                params.extend([*aliases, *aliases, *aliases, *aliases])
        if folders:
            clauses.append(f"t.project_folder IN ({','.join('?' * len(folders))})")
            params.extend(folders)
        if kinds:
            clauses.append(f"t.kind IN ({','.join('?' * len(kinds))})")
            params.extend(kinds)
        if narrow:
            clauses.append(SUPERSEDED_PROGRESS_SQL)
        if batch_ids:
            # Live work always counts: it holds the machine whatever its batch's fate.
            clauses.append(
                f"({TASK_BATCH_SQL} IS NULL OR {TASK_BATCH_SQL} IN "
                f"({','.join('?' * len(batch_ids))}) OR {LIVE_OR_AWAITING_SQL})"
            )
            params.extend(batch_ids)
        where = " AND ".join(clauses) or "1=1"
        base = f"FROM tasks t JOIN owners o ON o.key = t.owner_key WHERE {where}"

        counts = dict.fromkeys(STATES, 0)
        by_kind: dict[str, dict] = {}
        started_at = finished_at = None
        scope_live: list[dict] = []
        owner_keys: list[str] = []
        last_failure = None
        if not empty:
            for row in self._rows(
                "SELECT t.kind AS kind, t.state AS state, COUNT(*) AS n, "
                f"MIN(t.started_at) AS first_at, MAX(t.finished_at) AS last_at {base} "
                "GROUP BY t.kind, t.state",
                params,
            ):
                counts[row["state"]] += row["n"]
                item = by_kind.setdefault(row["kind"], dict.fromkeys(STATES, 0))
                item[row["state"]] += row["n"]
                if row["first_at"] and (started_at is None or row["first_at"] < started_at):
                    started_at = row["first_at"]
                if row["last_at"] and (finished_at is None or row["last_at"] > finished_at):
                    finished_at = row["last_at"]
            scope_live = [
                store._task(row)
                for row in self._rows(f"SELECT t.* {base} AND {LIVE_OR_AWAITING_SQL}", params)
            ]
            owner_keys = [
                row["owner_key"]
                for row in self._rows(
                    f"SELECT DISTINCT t.owner_key AS owner_key {base} LIMIT 2", params
                )
            ]
            failure_where = "t.state IN ('failed','interrupted')"
            failure_params = list(params)
            if not narrow:
                failure_where += " AND COALESCE(t.finished_at, t.updated_at) >= ?"
                failure_params.append(
                    datetime.fromtimestamp(time.time() - RECENT_FAILURE_SECONDS, UTC).isoformat()
                )
            failed_rows = self._rows(
                f"SELECT t.* {base} AND {failure_where} "
                "ORDER BY COALESCE(t.finished_at, t.updated_at) DESC LIMIT 1",
                failure_params,
            )
            if failed_rows:
                last_failure = self._failure_view(store._task(failed_rows[0]))
            recent_failures = (
                None
                if narrow
                else int(
                    self._rows(f"SELECT COUNT(*) AS n {base} AND {failure_where}", failure_params)[
                        0
                    ]["n"]
                )
            )
        else:
            recent_failures = None if narrow else 0

        # Queue facts are machine-wide: positions, holds, the runner and the ETA.
        live = store.list(states=LIVE, limit=None)
        live_owners = {item["key"]: item for item in store.owners(live_only=True)}
        settings = store.settings()
        host = self._host()
        slots = sum(capacity.effective(settings, host)["gpuSlots"].values())
        eta_all, eta_owner = self._eta(live, live_owners, slots or settings["defaultGpuSlots"])
        pending_order = [task["id"] for task in live if task["state"] in PENDING]
        positions = {identity: index + 1 for index, identity in enumerate(pending_order)}
        queue = sorted(
            (
                item
                for item in live_owners.values()
                if any(item["counts"].get(state) for state in LIVE)
                or any(task["ownerKey"] == item["key"] for task in scope_live)
            ),
            key=lambda item: item["queueSeq"],
        )
        owner_positions = {item["key"]: index + 1 for index, item in enumerate(queue)}
        runner_alive = launcher.runner_alive(store.path.parent / "runner.lock")
        retryable = None
        if narrow:
            retryable = not empty and self._scope_retryable(
                base, params, scope_live, live_owners, registry, whole_owner=bool(owner_id or owner)
            )

        live_states = [task for task in scope_live if task["state"] in LIVE]
        active = [task for task in live_states if task["state"] in ACTIVE]
        pending = sorted(
            (task for task in live_states if task["state"] in PENDING),
            key=lambda task: positions.get(task["id"], len(positions) + 1),
        )
        awaiting = [task for task in scope_live if task["state"] not in LIVE]
        held = any((live_owners.get(task["ownerKey"]) or {}).get("held") for task in pending)
        stopping = [task for task in active if task["state"] == "stopping"]
        cancel_requested = any(task.get("stopRequest") == "cancel" for task in live_states)
        if not scope_live:
            # A record or owner reports its own failures until they are retried; broader
            # scopes (a project, the machine) only recent ones.
            if not sum(counts.values()):
                state = "not-started"
            elif (counts["failed"] or counts["interrupted"]) if narrow else recent_failures:
                state = "attention"
            elif narrow and counts["cancelled"]:
                state = "cancelled"
            else:
                state = "completed"
        elif active:
            state = "stopping" if len(stopping) == len(active) else "running"
        elif pending and not runner_alive:
            state = "runner-stopped"
        elif held:
            state = "held"
        else:
            state = "queued"
        waiting = None
        if state in ("queued", "held", "runner-stopped") or (pending and not active):
            waiting = next(
                (task["waitingReason"] for task in pending if task["waitingReason"]), None
            )
            if state == "runner-stopped":
                waiting = "The Task Center runner is stopped; queued tasks start once it runs."
            elif held:
                waiting = "Held in the Task Center; it starts after it is released there."
            elif settings.get("paused") and waiting is None:
                waiting = "The queue is paused."
            elif waiting is None and awaiting and not pending:
                waiting = "Waiting to be resumed automatically."
            elif (
                waiting is None and pending and all(task["state"] == "blocked" for task in pending)
            ):
                waiting = "Waiting for earlier tasks to finish."
        scope_owner_keys = {task["ownerKey"] for task in scope_live}
        if narrow or kinds or project:
            etas = [eta_owner[key] for key in scope_owner_keys if key in eta_owner]
            eta = (
                {"seconds": max(etas), "basis": eta_all["basis"]}
                if etas and (active or pending)
                else None
            )
        else:
            eta = eta_all if live else None
        single_owner = owner_keys[0] if len(owner_keys) == 1 else None
        owner_row = (
            live_owners.get(single_owner) or store.owner(single_owner) if single_owner else None
        )
        project_id = self._project(owner_row, registry) if owner_row else None
        if project and not os.path.isabs(project):
            project_id = project_id or project
        current = None
        if active:
            first = sorted(active, key=lambda task: task.get("startedAt") or "")[0]
            current = {
                "taskId": first["id"],
                "title": first["title"],
                "kind": first["kind"],
                "labels": first["labels"],
                "progress": first["progress"],
                "startedAt": first["startedAt"],
            }
        total = sum(counts.values())
        focus = None
        if state == "attention" and last_failure:
            focus = last_failure["taskId"]
        elif total == 1 and (scope_live or last_failure):
            focus = scope_live[0]["id"] if scope_live else last_failure["taskId"]
        if not narrow and state != "attention":
            focus = None
        href = self._deep_link(
            owner=single_owner if narrow else None,
            task=focus,
            project=project_id,
            kind=kinds[0] if kinds and len(kinds) == 1 else None,
        )
        return {
            "scope": scope,
            "state": state,
            "counts": counts,
            "byKind": {
                kind: {
                    "counts": item,
                    "completed": item["succeeded"],
                    "total": sum(item.values()),
                }
                for kind, item in sorted(by_kind.items())
            },
            "progress": {"completed": counts["succeeded"], "total": total} if narrow else None,
            "live": len(live_states) + len(awaiting),
            "active": len(active),
            "pending": len(pending) + len(awaiting),
            "held": held,
            "position": owner_positions.get(single_owner) if single_owner else None,
            "queuePosition": positions.get(pending[0]["id"]) if pending else None,
            "waitingReason": waiting,
            "eta": eta,
            "runnerAlive": runner_alive,
            "paused": bool(settings.get("paused")),
            "stopRequest": "cancel" if cancel_requested else ("pause" if stopping else None),
            "lastFailure": last_failure,
            "recentFailures": recent_failures,
            "retryable": retryable,
            "current": current,
            "startedAt": started_at,
            "finishedAt": None if scope_live else finished_at,
            "ownerKey": single_owner,
            "ownerKind": (owner_row or {}).get("kind"),
            "ownerId": (owner_row or {}).get("id"),
            "title": (owner_row or {}).get("title"),
            "projectId": project_id,
            "projectName": names.get(project_id) if project_id else None,
            "href": href,
            "updatedAt": utc_now_iso(),
        }

    def _scope_retryable(self, base, params, scope_live, live_owners, registry, *, whole_owner):
        """Whether a retry would resume any of a rollup's work now: the owner's retry for an
        owner scope (as its ``actions.retry``), a retry of the scope's own tasks otherwise."""
        tasks = {task["id"]: task for task in scope_live}
        for row in self._rows(
            f"SELECT t.* {base} AND t.state IN ({','.join('?' * len(RETRYABLE))})",
            [*params, *sorted(RETRYABLE)],
        ):
            task = self.store._task(row)
            tasks.setdefault(task["id"], task)
        by_owner: dict[str, list] = {}
        for task in tasks.values():
            by_owner.setdefault(task["ownerKey"], []).append(task)
        trashed_cache: dict = {}
        for key, items in by_owner.items():
            owner = live_owners.get(key) or self.store.owner(key)
            if owner is None or self._project(owner, registry) is None:
                continue
            if whole_owner:
                # As the owner's own retry offer (``_owner_retryable``).
                trashed = self._trashed_batches(owner, trashed_cache)
                items = self._without_batches(items, lambda batch: batch not in trashed)
            if _plan_resumes(self._retry_plan(items, owner, whole_owner=whole_owner)):
                return True
        return False

    @staticmethod
    def _deep_link(*, owner=None, task=None, project=None, kind=None, state=None) -> str:
        """``#task-center?owner=…&task=…&project=…&kind=…&state=…`` (empty values dropped)."""
        values = [
            (name, value)
            for name, value in (
                ("owner", owner),
                ("task", task),
                ("project", project),
                ("kind", kind),
                ("state", state),
            )
            if value
        ]
        query = "&".join(f"{name}={_query(value)}" for name, value in values)
        return "#task-center" + (f"?{query}" if query else "")

    # -- capacity ------------------------------------------------------------------------

    def capacity(self) -> dict:
        return self._call(self._capacity)

    def _capacity(self) -> dict:
        store = self.store
        settings = store.settings()
        host = self._host()
        effective = capacity.effective(settings, host)
        return {
            "settings": settings,
            "effective": {
                "gpuSlots": {str(index): count for index, count in effective["gpuSlots"].items()},
                "cpuTaskSlots": effective["cpuTaskSlots"],
                "reserves": effective["reserves"],
                "vramReserves": {
                    str(index): value for index, value in effective["vramReserves"].items()
                },
            },
            "suggestion": self._suggestion(settings, host),
        }

    def _suggestion(self, settings: dict, host: dict) -> dict:
        store = self.store
        live = store.list(states=LIVE, limit=None)
        workloads, seen = [], set()
        for task in live:
            request = task["request"] or {}
            workload = request.get("workload")
            if request.get("lane") != "gpu" or not isinstance(workload, dict):
                continue
            key = (
                request.get("workloadKey")
                or workload.get("key")
                or estimator.workload_key(workload)
            )
            if key not in seen:
                seen.add(key)
                workloads.append({**workload, "key": key})
        running = [task for task in live if task["state"] in ACTIVE]
        folders = set(self._registry())
        folders |= {_norm(owner["projectFolder"]) for owner in store.owners(live_only=False)}
        try:
            observations = estimator.gather_observations(
                sorted(folders), measurements=store.measurements()
            )
            return estimator.suggest(workloads, observations, host, settings, running=running)
        except StorageError:
            raise
        except Exception as error:  # evidence problems degrade to a hardware-only suggestion
            fallback = estimator.suggest([], [], host, settings, running=running)
            fallback.setdefault("findings", []).append(
                {
                    "code": "CAPACITY_EVIDENCE_UNAVAILABLE",
                    "severity": "warning",
                    "message": f"Measured evidence could not be read: {error}",
                }
            )
            return fallback

    def update_capacity(self, changes: dict, operation_id: str) -> dict:
        with self._serialized(operation_id):
            return self._call(self._update_capacity, changes, operation_id)

    def _update_capacity(self, changes: dict, operation_id: str) -> dict:
        store = self.store
        if self._replayed({"capacity": changes}, operation_id):
            return self._capacity()
        patch = {}
        if changes.get("parallelGpuTasks") is not None:
            count = changes["parallelGpuTasks"]
            host = self._host()
            indices = {str(gpu["index"]) for gpu in host.get("gpus") or [] if "index" in gpu}
            indices |= set(store.settings()["gpuSlots"])
            patch["defaultGpuSlots"] = count
            patch["gpuSlots"] = dict.fromkeys(sorted(indices), count)
        if changes.get("gpuSlots") is not None:
            patch["gpuSlots"] = {**patch.get("gpuSlots", {}), **changes["gpuSlots"]}
        if "cpuTaskSlots" in changes:
            patch["cpuTaskSlots"] = changes["cpuTaskSlots"]
        for key in ("paused", "autoResume"):
            if changes.get(key) is not None:
                patch[key] = changes[key]
        if changes.get("defaults"):
            patch["defaults"] = {
                key: value for key, value in changes["defaults"].items() if value is not None
            }
        if patch:
            store.update_settings(patch)
        self._record({"capacity": changes}, operation_id, {"settings": sorted(patch)})
        return self._capacity()

    # -- runner --------------------------------------------------------------------------

    def start_runner(self, operation_id: str) -> dict:
        with self._serialized(operation_id):
            return self._call(self._start_runner, operation_id)

    def _start_runner(self, operation_id: str) -> dict:
        receipt = self._receipt({"runner": "start"}, operation_id)
        if receipt is not None:
            return {"runner": self._runner_view(), "result": receipt.get("result")}
        result = launcher.ensure_runner()
        self._record({"runner": "start"}, operation_id, {"result": result})
        return {"runner": self._runner_view(), "result": result}

    def restart_runner(self, operation_id: str) -> dict:
        with self._serialized(operation_id):
            return self._call(self._restart_runner, operation_id)

    def _restart_runner(self, operation_id: str) -> dict:
        receipt = self._receipt({"runner": "restart"}, operation_id)
        if receipt is not None:
            return {"runner": self._runner_view(), "result": receipt.get("result")}
        import shutil

        if not paths.autostart_enabled():
            # Never stop a runner this service is not allowed to start again.
            result = {"started": False, "reason": "disabled"}
        elif shutil.which("tmux") is None:
            result = {"started": False, "reason": "tmux unavailable"}
        else:
            stopped = launcher.stop_runner(wait=5.0)
            # The runner releases its lock before its process and tmux session end; a new
            # session started in that window is refused as a duplicate.
            if stopped.get("stopped") and not self._await_runner_exit(RUNNER_EXIT_SECONDS):
                # It honours the stop only between steps; starting now would find it alive
                # and leave no runner once it exits.
                self._restart_when_exited()
                result = {
                    "started": False,
                    "pending": True,
                    "note": RESTART_PENDING_NOTE,
                    "stopped": stopped,
                }
            else:
                result = {**launcher.ensure_runner(), "stopped": stopped}
        self._record({"runner": "restart"}, operation_id, {"result": result})
        return {"runner": self._runner_view(), "result": result}

    def _restart_when_exited(self) -> None:
        """Start the runner in the background once the stopped one has exited."""
        lock = self.store.path.parent / "runner.lock"

        def restart():
            try:
                if self._await_runner_exit(RUNNER_RESTART_WAIT_SECONDS, lock, poll=1.0):
                    launcher.ensure_runner()
            except Exception:  # Start runner stays available; nothing waits on this thread
                pass

        with self._lock:
            if self._restart_thread is not None and self._restart_thread.is_alive():
                return
            self._restart_thread = threading.Thread(
                target=restart, name="task-center-runner-restart", daemon=True
            )
            self._restart_thread.start()

    def _await_runner_exit(
        self, timeout: float, lock: Path | None = None, *, poll: float = 0.1
    ) -> bool:
        lock = lock or self.store.path.parent / "runner.lock"
        deadline = time.monotonic() + timeout
        while True:
            if not launcher.runner_alive(lock) and not self._runner_session_exists():
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(poll)

    @staticmethod
    def _runner_session_exists() -> bool:
        """Whether the runner's tmux session still exists (read-only query)."""
        import shutil
        import subprocess

        tmux = shutil.which("tmux")
        if tmux is None:
            return False
        try:
            result = subprocess.run(
                [tmux, "has-session", "-t", "=" + launcher.session_name()],
                capture_output=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return result.returncode == 0

    # -- operation receipts --------------------------------------------------------------

    def _receipt(self, request: dict, operation_id: str) -> dict | None:
        prior = self.store.operation(OPERATION_PREFIX + operation_id)
        if prior is None:
            return None
        if prior["requestHash"] != request_hash(request):
            raise StorageError(
                "This operation id was already used for a different request.",
                "OPERATION_CONFLICT",
                409,
            )
        return prior.get("result") or {}

    def _replayed(self, request: dict, operation_id: str) -> bool:
        return self._receipt(request, operation_id) is not None

    def _record(self, request: dict, operation_id: str, result: dict) -> None:
        self.store.record_operation(
            OPERATION_PREFIX + operation_id,
            request_hash(request),
            {**result, "at": utc_now_iso()},
        )

    # -- actions -------------------------------------------------------------------------

    def _workspace_project(self, owner: dict | None):
        """The scientific store of an owner's project, which must be registered here."""
        project_id = self._project(owner, self._registry())
        if project_id is None:
            raise StorageError(
                "This task belongs to a project of another HistoPilot workspace. Manage it "
                "from the workspace that submitted it.",
                "TASK_OWNER_OTHER_WORKSPACE",
                409,
            )
        return project_id

    def _project_store(self, project_id: str):
        return self.projects.scientific_store(project_id)

    def task_action(self, task_id: str, action: str, operation_id: str) -> dict:
        with self._serialized(operation_id):
            return self._call(self._task_action, task_id, action, operation_id)

    def _task_action(self, task_id: str, action: str, operation_id: str) -> dict:
        if action not in TASK_ACTIONS:
            raise StorageError(f"Unknown task action: {action}.", "TASK_ACTION_INVALID", 422)
        task = self._require_task(task_id)
        owner = self.store.owner(task["ownerKey"])
        project_id = self._workspace_project(owner)
        request = {"scope": "task", "task": task_id, "action": action}
        if not self._replayed(request, operation_id):
            awaiting = awaiting_requeue(task)
            if action == "cancel" and (task["state"] in LIVE or awaiting):
                self._cancel_tasks([task], owner, project_id, operation_id, whole_owner=False)
            elif action == "retry" and task["state"] in RETRYABLE and not awaiting:
                self._retry_tasks([task], owner, project_id, operation_id)
            self._record(request, operation_id, {"task": task_id, "action": action})
        return self._task_json(self._require_task(task_id), self._context())

    def owner_action(
        self, key: str, action: str, operation_id: str, *, position: str | None = None
    ) -> dict:
        with self._serialized(operation_id):
            return self._call(self._owner_action, key, action, operation_id, position)

    def _owner_action(self, key, action, operation_id, position) -> dict:
        if action not in OWNER_ACTIONS:
            raise StorageError(f"Unknown owner action: {action}.", "TASK_ACTION_INVALID", 422)
        if action == "move" and position not in MOVE_POSITIONS:
            raise StorageError(
                "Move an owner to top, up, down or bottom.", "TASK_ACTION_INVALID", 422
            )
        store = self.store
        owner = store.owner(key)
        if owner is None:
            raise StorageError("This task owner does not exist.", "TASK_OWNER_NOT_FOUND", 404)
        project_id = self._workspace_project(owner)
        request = {"scope": "owner", "owner": key, "action": action}
        if action == "move":
            request["position"] = position
        if not self._replayed(request, operation_id):
            if action == "hold":
                store.hold_owner(key, True)
            elif action == "release":
                store.hold_owner(key, False)
            elif action == "move":
                store.move_owner(key, position)
            elif action == "stop":
                store.hold_owner(key, True)
                # A second pass catches a task the runner admitted from a candidate list
                # read just before the hold committed.
                for _ in range(2):
                    store.request_stop(
                        [
                            task["id"]
                            for task in store.list(
                                owner_key=key, states=("starting", "running"), limit=None
                            )
                            if _pausable(task)
                        ],
                        "pause",
                    )
            elif action == "cancel":
                live = store.list(owner_key=key, states=LIVE, limit=None)
                # A lost fold awaiting its auto-resume is cancelled instead of resumed.
                live += [
                    task
                    for task in store.pending_bookkeeping()
                    if task["ownerKey"] == key
                    and task["kind"] == "mil-fold"
                    and awaiting_requeue(task)
                ]
                if live:
                    self._cancel_tasks(live, owner, project_id, operation_id, whole_owner=True)
                if (store.owner(key) or {}).get("held"):
                    # Only what a cancel leaves to conclude remains (collections recording
                    # the cancelled batches, a stopping coordinator); a hold would keep
                    # those queued forever.
                    store.hold_owner(key, False)
            elif action == "retry":
                tasks = store.list(owner_key=key, limit=None)
                self._retry_tasks(tasks, owner, project_id, operation_id, whole_owner=True)
            self._record(request, operation_id, {"owner": key, "action": action})
        return self._owner_view(key)

    @staticmethod
    def _experiment_id(task: dict, owner: dict | None) -> str | None:
        owner = owner or {}
        return (
            (task.get("adapterData") or {}).get("experimentId")
            or (task.get("labels") or {}).get("experimentId")
            or (owner.get("id") if owner.get("kind") == "experiment" else None)
        )

    @staticmethod
    def _batch_id(task: dict) -> str | None:
        group = task.get("group") or {}
        return (
            (group.get("id") if group.get("kind") == "mil-batch" else None)
            or (task.get("adapterData") or {}).get("batchId")
            or (task.get("labels") or {}).get("batchId")
        )

    @staticmethod
    def _run_id(task: dict) -> str | None:
        return (task.get("adapterData") or {}).get("runId") or (task.get("labels") or {}).get(
            "runId"
        )

    @staticmethod
    def _compute_target(task: dict) -> tuple[str | None, str | None]:
        data, group = task.get("adapterData") or {}, task.get("group") or {}
        return data.get("kind") or group.get("kind"), data.get("recordId") or group.get("id")

    @staticmethod
    def _bulk_id(task: dict, owner: dict | None) -> str | None:
        owner = owner or {}
        return (
            (task.get("adapterData") or {}).get("batchId")
            or (task.get("labels") or {}).get("batchId")
            or (owner.get("id") if owner.get("kind") == "evaluation-batch" else None)
            or (task.get("group") or {}).get("id")
        )

    def _cancel_tasks(self, tasks, owner, project_id, operation_id, *, whole_owner) -> None:
        """Route cancellation to the services that own each task's science record.

        Batch, compute and coordinator cancels write their cancel receipts before the
        runner signals anything; tasks without such a record are cancelled in the store.
        """
        errors = _Errors()

        def derive(target):
            return derived_operation_id("cancel", operation_id, target)

        coordinators, batches, folds, computes, bulks, direct = {}, {}, {}, {}, {}, []
        bulk_owner = (owner or {}).get("kind") == "evaluation-batch"
        for task in tasks:
            kind = task["kind"]
            if kind in MIL_KINDS:
                batch = self._batch_id(task)
                run = self._run_id(task) if kind == "mil-fold" else None
                if batch is not None and whole_owner:
                    if kind == "mil-fold":
                        batches.setdefault(batch, []).append(task)
                    # An owner-wide cancel leaves a running collection alone: it records
                    # the outcome of folds that already finished and ends by itself.
                elif batch is not None and run is not None:
                    # The batch records the cancelled run, so its results say cancelled.
                    folds.setdefault(batch, []).append(run)
                else:
                    direct.append(task)
                continue
            if kind == "predictor-coordinator":
                experiment = self._experiment_id(task, owner)
                if experiment:
                    coordinators.setdefault(experiment, []).append(task)
                    continue
            elif kind == "compute-job":
                if whole_owner and bulk_owner:
                    bulks.setdefault(owner["id"], []).append(task)
                    continue
                record_kind, record = self._compute_target(task)
                if record and record_kind in COMPUTE_KINDS:
                    computes.setdefault(record, []).append(task)
                    continue
            elif kind == "bulk-submit":
                batch = self._bulk_id(task, owner)
                if batch:
                    bulks.setdefault(batch, []).append(task)
                    continue
            direct.append(task)
        if whole_owner and bulk_owner:
            bulks.setdefault(owner["id"], [])
        store = (
            self._project_store(project_id)
            if coordinators or batches or folds or computes or bulks
            else None
        )
        for experiment in coordinators:
            errors.run(self._service("predictors", store).cancel, experiment, derive(experiment))
        for batch in batches:
            errors.run(self._service("training", store).cancel, batch, derive(batch))
        for batch, runs in folds.items():
            errors.run(
                self._service("training", store).cancel_runs, batch, sorted(runs), derive(batch)
            )
        for record, items in computes.items():
            if errors.run(self._service("compute", store).cancel, record, derive(record)):
                # Never started: nothing to signal, and the record already asked to stop.
                self._cancel_pending(items)
        for batch, items in bulks.items():
            if errors.run(self._service("bulk", store).cancel, batch, derive(batch)):
                self._cancel_pending(items)
        for task in direct:
            self.client.cancel_task(task["id"])
        errors.raise_first()

    def _cancel_pending(self, tasks) -> None:
        pending = [
            task["id"]
            for task in tasks
            if (self.store.get(task["id"]) or {}).get("state") in PENDING
        ]
        if pending:
            self.store.cancel_pending(pending)

    def _retry_plan(self, tasks, owner, *, whole_owner) -> dict:
        """Which owning services a retry of ``tasks`` resumes, and which batches must wait.

        A batch waits while its owning service would refuse the resume (``busy``, see
        ``_busy_batches``). Other concluded tasks awaiting their requeue wait for the runner.
        """
        by_batch: dict[str, list] = {}
        running = _busy_batches(tasks, self._batch_id)
        coordinators, computes, direct = {}, {}, []
        # A live, resumable or cancelled coordinator decides about its refits; one that
        # completed had every refit succeed, so a failed refit was launched separately.
        coordinator_owns_refits = any(
            task["kind"] == "predictor-coordinator" and task["state"] != "succeeded"
            for task in tasks
        )
        for task in tasks:
            kind = task["kind"]
            if _progress_collect(task):
                # Progress summaries are idempotent and superseded by the final collection.
                if not whole_owner and task["state"] in RETRYABLE and not awaiting_requeue(task):
                    direct.append(task)
            elif kind in MIL_KINDS:
                batch = self._batch_id(task)
                if batch:
                    by_batch.setdefault(batch, []).append(task)
            elif task["state"] not in RETRYABLE or awaiting_requeue(task):
                continue
            elif kind == "predictor-coordinator":
                experiment = self._experiment_id(task, owner)
                # A cancelled coordinator resumes too: Task Center experiments stay resumable.
                if experiment and task["state"] in ("failed", "interrupted", "cancelled"):
                    coordinators[experiment] = task
            elif kind == "compute-job":
                record_kind, record = self._compute_target(task)
                # A coordinator relaunches the refits it owns when it resumes.
                if whole_owner and coordinator_owns_refits and record_kind in REFIT_KINDS:
                    continue
                if record and record_kind in COMPUTE_KINDS:
                    computes[record] = record_kind
            else:
                direct.append(task)
        batches, busy = [], []
        for batch, items in by_batch.items():
            if not any(item["state"] in RETRYABLE for item in items):
                continue
            if whole_owner and batch in running:
                busy.append(batch)
                continue
            batches.append(batch)
        return {
            "batches": batches,
            "busy": busy,
            "coordinators": coordinators,
            "computes": computes,
            "direct": direct,
        }

    def _retry_tasks(self, tasks, owner, project_id, operation_id, *, whole_owner=False) -> None:
        """Resume unsuccessful work through its owning service (a new attempt, same task)."""
        errors = _Errors()

        def derive(target):
            return derived_operation_id("retry", operation_id, target)

        if whole_owner:
            # Failed folds of a trashed or unsubmitted batch are never resumed: the training
            # service refuses them, which used to fail the whole retry.
            tasks = self._current_tasks(tasks, owner, project_id)
        plan = self._retry_plan(tasks, owner, whole_owner=whole_owner)
        batches, coordinators, computes = plan["batches"], plan["coordinators"], plan["computes"]
        if not (batches or coordinators or computes or plan["direct"]) and plan["busy"]:
            raise StorageError(
                "Wait for the running tasks of this batch to finish before retrying it.",
                "TASK_RETRY_BUSY",
                409,
            )
        store = self._project_store(project_id) if batches or coordinators or computes else None
        for batch in batches:
            errors.run(self._service("training", store).launch, batch, derive(batch), resume=True)
        for record, record_kind in computes.items():
            errors.run(self._resume_compute, store, record_kind, record, derive(record))
        for experiment in coordinators:
            errors.run(
                self._service("predictors", store).launch,
                experiment,
                derive(experiment),
                resume=True,
            )
        for task in plan["direct"]:
            self.client.requeue_task(task["id"], reason="retry")
        errors.raise_first()

    def _resume_compute(self, store, kind: str, record: str, operation_id: str):
        if kind in REFIT_KINDS:
            from histopilot.schemas.predictors import LaunchRefit

            return self._service("refits", store).launch(
                record, LaunchRefit(operationId=operation_id), resume=True
            )
        if kind in EVALUATION_KINDS:
            return self._service("evaluations", store).launch(record, operation_id, resume=True)
        return self._service("interpretations", store).launch(record, operation_id, resume=True)
