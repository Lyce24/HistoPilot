"""The machine-wide Task Center runner: admission, supervision, stops and recovery.

The runner treats every task as an opaque command in its own session. It never imports
worker code and never reads project stores; adapters translate process exits into the
records they own. Tasks outlive the runner and are re-adopted by verified identity.
"""

import hashlib
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from contextlib import ExitStack, contextmanager
from datetime import timedelta
from pathlib import Path

from histopilot.storage.io import utc_now
from histopilot.storage.project_lock import StorageError
from histopilot.taskcenter import adapters as adapter_registry
from histopilot.taskcenter import capacity, leases, paths, procs
from histopilot.taskcenter.adapters.base import BUSY_EXIT, AdapterError, RunnerContext, outcome
from histopilot.taskcenter.model import (
    ACTIVE,
    PROTOCOL_VERSION,
    TERMINAL,
    awaiting_requeue,
    normalize_command,
    normalize_request,
    parse_iso,
    seconds_between,
)
from histopilot.taskcenter.store import MEASUREMENT_COLUMNS, TaskStore

HEARTBEAT_SECONDS = 2.0
# Truncate the store's write-ahead log this often; the heartbeat alone adds megabytes an hour.
CHECKPOINT_SECONDS = 300.0
DRAIN_SECONDS = 5.0
OOM_VRAM_FACTOR = 1.5
# An OOM on a GPU where memory this runner cannot account for is in use (another program,
# or a task using more than it asked for) is requeued without raising the task's request or
# spending its OOM retry; admission then waits until that memory is free. Bounded, so a
# misattributed OOM still reaches the normal backoff.
FOREIGN_VRAM_GB = 1.0
CONTENTION_MAX_REQUEUES = 3
# A lost device interrupts its task and requeues it once the device answers again, up to
# this many times per run; the failures that follow are the task's own.
DEVICE_LOSS_MAX_REQUEUES = 3
DEFAULT_EXIT_REASONS = {
    "succeeded": "ok",
    "failed": "error",
    "cancelled": "cancelled",
    "interrupted": "interrupted",
    "requeue": "paused",
}
OUTCOME_STATES = frozenset(DEFAULT_EXIT_REASONS)
ON_HOLD = "On hold"
# After a device failure a GPU takes no new work for a while, doubling with each incident,
# and then only once a fresh process can initialize CUDA on it (the probe runs off the
# loop). One broken device cannot fail the whole queue in turn; the fence is saved on the
# runner row, so a restarted runner keeps it.
GPU_FENCE_SECONDS = 60.0
GPU_FENCE_MAX_SECONDS = 1800.0
# Error text of a lost or broken device. Device-side asserts and illegal memory accesses
# come from one task's own inputs or code, so they fail that task and fence nothing.
DEVICE_LOSS = re.compile(
    r"(?:device|gpu) (?:has been |is )?lost|fallen off|unknown error|cudaerrorunknown"
    r"|no cuda-capable device|initialization error|busy or unavailable|uncorrectable ecc"
    r"|\bxid\b|nvml|driver",
    re.IGNORECASE,
)
TASK_FAULT = re.compile(r"device-side assert|illegal memory access", re.IGNORECASE)
# A task requeued because its output is busy waits before it is spawned again, doubling
# up to a minute. After BUSY_PATIENT_AFTER busy attempts in a row (hours of training can
# hold a project) it waits up to ten minutes, so an export waiting for an idle project
# does not relaunch a worker every minute; from BUSY_REPORT_AFTER on, its waiting reason
# names the cause and the number of tries.
BUSY_BACKOFF_SECONDS = 5.0
BUSY_BACKOFF_MAX_SECONDS = 60.0
BUSY_BACKOFF_LONG_SECONDS = 600.0
BUSY_PATIENT_AFTER = 10
BUSY_REPORT_AFTER = 3
BUSY_WAIT = "Waiting for another worker to release this task's output"
EXCLUSIVE_WAIT = "Waiting for a related task to finish"
# An automatic requeue the adapter cannot decide yet (for example while the training
# runtime probe fails after a reboot) is retried with a backoff, then given up.
INTENT_RETRY_SECONDS = 30.0
INTENT_RETRY_MAX_SECONDS = 300.0
INTENT_GIVE_UP_SECONDS = 2 * 3600.0
# A hook that failed transiently (for example on a busy batch lock) is retried with a
# backoff, so stuck hooks never slow every tick down.
HOOK_RETRY_SECONDS = 2.0
HOOK_RETRY_MAX_SECONDS = 60.0
# How long a task's leader may be gone before its wrapper's exit record must be there, and
# how long one tick waits for a wrapper it spawned to record an exit it just saw.
EXIT_RECORD_WAIT_SECONDS = 30.0
EXIT_RECORD_GRACE_SECONDS = 2.0
# Wrapper journals (start and exit records) older than this are swept at startup.
JOURNAL_KEEP_SECONDS = 7 * 86400.0
# How often leases whose owner is confirmed dead are pruned from the shared registry.
LEASE_PRUNE_SECONDS = 300.0
LEASE_PRUNE_LOCK_SECONDS = 0.2
REGISTRY_WAIT = "Waiting for the resource lease registry"
EARLIER_PROCESS_WAIT = "Waiting for an earlier process of this task to exit"
NEEDS_MORE_VRAM = "This task needs more GPU memory than this machine has"
STOP_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
# Source files changed after this moment may differ from the code this process runs.
_LOADED_AT = time.time()


class RunnerBusy(RuntimeError):
    """Another runner holds the runner lock for this state directory."""


def code_hash(files: list[str] | None = None) -> str:
    """Fingerprint of runner code.

    Without ``files``: taskcenter/*.py and its adapters (what runners recorded before they
    listed their modules). With ``files``: exactly those absolute paths, in any order; a
    missing file hashes as absent.
    """
    digest = hashlib.sha256()
    if files is None:
        root = Path(__file__).resolve().parent
        for path in sorted([*root.glob("*.py"), *(root / "adapters").glob("*.py")]):
            digest.update(str(path.relative_to(root)).encode() + b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        return digest.hexdigest()
    for name in sorted(set(files)):
        digest.update(str(name).encode() + b"\0")
        try:
            content = Path(name).read_bytes()
        except (FileNotFoundError, NotADirectoryError):
            digest.update(b"\0absent")
        else:
            digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def _row_checkout(row: dict) -> str | None:
    """The checkout a runner row names. Runners from before ``checkoutRoot`` was recorded
    are placed by the absolute paths of the modules they loaded (``codeFiles``)."""
    if row.get("checkoutRoot"):
        return row["checkoutRoot"]
    for name in row.get("codeFiles") or []:
        folders = Path(name).parts[:-1]
        if "histopilot" in folders:
            package = len(folders) - 1 - folders[::-1].index("histopilot")
            return str(Path(*folders[:package])) if package else None
    return None


def foreign_checkout(row: dict | None) -> str | None:
    """The checkout a runner row was started from when it is not this process's checkout."""
    recorded = _row_checkout(row or {})
    return recorded if recorded and recorded != paths.checkout_root() else None


def code_current(row: dict | None) -> bool:
    """Whether a runner row's recorded code matches the files on disk of this checkout.

    A runner started from another checkout (the state directory is per OS user, so every
    checkout shares one runner) is never current here. An unreadable file or a row without
    a hash never asks for a restart.
    """
    if not row:
        return False
    if foreign_checkout(row):
        return False
    recorded = row.get("codeHash")
    if not recorded:
        return True
    files = row.get("codeFiles")
    try:
        return recorded == (code_hash(files) if files else code_hash())
    except OSError:
        return True


def loaded_code_files() -> list[str]:
    """Absolute paths of every histopilot module this process has loaded."""
    import histopilot

    root = Path(histopilot.__file__).resolve().parent
    files = set()
    for module in list(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if not isinstance(name, str) or not name.endswith(".py"):
            continue
        path = Path(name).resolve()
        if path.is_relative_to(root):
            files.add(str(path))
    return sorted(files)


def _recorded_hash(files: list[str]) -> str:
    """The hash to record for ``files``; one that never matches when a file changed since
    this process loaded its code, because the loaded code is then unknown."""
    value = code_hash(files)
    for name in files:
        try:
            if os.stat(name).st_mtime > _LOADED_AT:
                return "changed-after-start-" + value
        except OSError:
            continue
    return value


@contextmanager
def stop_signals(event: threading.Event):
    """Turn SIGTERM/SIGINT/SIGHUP into ``event`` inside the block (main thread only)."""
    previous = {}
    if threading.current_thread() is threading.main_thread():
        for signum in STOP_SIGNALS:
            previous[signum] = signal.signal(signum, lambda *_: event.set())
    try:
        yield event
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _print_log(message: str) -> None:
    print(f"{utc_now()} {message}", flush=True)


def _abort_error(task: dict) -> str | None:
    """The message of a fatal on_started hook that must fail this task once it exits."""
    pending = task.get("bookkeeping") or {}
    if pending.get("hook") == "abort":
        return pending.get("error") or "The task adapter rejected the started task."
    return pending.get("abortError")


def _boot_id() -> str | None:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return None


def host_stop(task: dict, returncode: int | None, record: dict | None, boot: str | None):
    """Why a failed exit the Task Center never asked for came from the host, or None.

    "shutdown": the task's wrapper, outside the task's session, received a stop signal
    too (the machine or the login session was going down), or the exit was recorded in an
    earlier boot. "signal": the task alone died of SIGTERM/SIGINT/SIGHUP that nobody in
    the Task Center sent. Either way the task was interrupted, not failed; a crash, a
    SIGKILL (the OOM killer) or a busy exit stays the task's own outcome.
    """
    if task["stopRequest"] or task["signalledAt"] or returncode in (None, 0, BUSY_EXIT):
        return None
    record = record or {}
    received = record.get("hostSignal")
    if type(received) is int and received in STOP_SIGNALS:
        return "shutdown"
    recorded_boot = record.get("bootId")
    if isinstance(recorded_boot, str) and boot and recorded_boot != boot:
        return "shutdown"
    if returncode < 0 and -returncode in STOP_SIGNALS:
        return "signal"
    return None


def device_lost(error: str | None) -> bool:
    """Whether a CUDA failure message describes a lost or broken device."""
    text = error or ""
    return not TASK_FAULT.search(text) and bool(DEVICE_LOSS.search(text))


def _same_process(first, second) -> bool:
    return (
        isinstance(first, dict)
        and isinstance(second, dict)
        and first.get("pid") == second.get("pid")
        and first.get("startTicks") == second.get("startTicks")
    )


def _normalize_outcome(value) -> dict:
    if not isinstance(value, dict) or value.get("state") not in OUTCOME_STATES:
        return outcome(
            "failed", "error", f"The task adapter returned an invalid outcome: {value!r}"
        )
    return {
        "state": value["state"],
        "exitReason": value.get("exitReason") or DEFAULT_EXIT_REASONS[value["state"]],
        "error": value.get("error"),
        "measurement": value.get("measurement")
        if isinstance(value.get("measurement"), dict)
        else None,
    }


class Runner:
    lock_wait = 2.0  # absorbs launcher.runner_alive() probes that briefly hold the lock

    def __init__(
        self,
        store: TaskStore,
        *,
        now=utc_now,
        clock=time.monotonic,
        host_probe=capacity.host,
        spawner=procs.spawn,
        lease_reader=leases.read_leases,
        lease_writer=leases.write_task_lease,
        lease_remover=leases.remove_task_lease,
        lease_lock=leases.registry_lock,
        lease_pruner=leases.prune_dead_leases,
        adapters=adapter_registry.adapter,
        cuda_probe=procs.cuda_probe,
        sample_interval=10.0,
        host_interval=5.0,
        log=None,
    ):
        self.store = store
        self.now = now
        self.clock = clock
        self.host_probe = host_probe
        self.spawner = spawner
        self.lease_reader = lease_reader
        self.lease_writer = lease_writer
        self.lease_remover = lease_remover
        self.lease_lock = lease_lock
        self.lease_pruner = lease_pruner
        self.adapters = adapters
        self.cuda_probe = cuda_probe
        self.sample_interval = sample_interval
        self.host_interval = host_interval
        self.log = log or _print_log
        self.identity: dict | None = None
        self._procs = {}  # task id -> Popen for children this runner spawned
        self._draining = {}  # task id -> clock when surviving group members were signalled
        self._samples = {}  # task id -> (clock, cpu seconds) of the previous telemetry sample
        self._foreign_gpu = Counter()  # live foreign leases per GPU, refreshed every tick
        # GPU index -> {"failures", "until", "verified", "error"} after device failures
        self._gpu_fences = {}
        self._probes = {}  # GPU index -> {"result"} of a CUDA probe running in a thread
        self._busy = {}  # task id -> {"count", "until"} after "busy" requeues
        self._intents = {}  # task id -> {"count", "until"} for undecided requeues
        self._hooks = {}  # task id -> {"count", "until"} for hooks being retried
        self._exit_waits = {}  # task id -> clock since its leader was seen gone
        self._orphans = {}  # lease file -> {"pid", "at", "killed"} for untracked processes
        self._registry_error: str | None = None
        self._host: dict | None = None
        self._host_at: float | None = None
        self._heartbeat_at: float | None = None
        self._checkpoint_at: float | None = None
        self._pruned_at: float | None = None
        self._sample_at: float | None = None
        self._lock_descriptor: int | None = None
        self._stop = None
        self._code_modules: int | None = None
        self._shared = {}

    # -- lifecycle ---------------------------------------------------------------------------

    @property
    def lock_path(self) -> Path:
        return self.store.path.parent / "runner.lock"

    def _acquire_lock(self) -> None:
        import fcntl

        descriptor = os.open(self.lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        deadline = time.monotonic() + self.lock_wait
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    os.close(descriptor)
                    raise RunnerBusy(
                        f"Another Task Center runner holds {self.lock_path}."
                    ) from None
                time.sleep(0.05)
        self._lock_descriptor = descriptor

    def close(self) -> None:
        """Release the runner lock. Tasks keep running in their own sessions."""
        if self._lock_descriptor is not None:
            import fcntl

            try:
                fcntl.flock(self._lock_descriptor, fcntl.LOCK_UN)
            finally:
                os.close(self._lock_descriptor)
                self._lock_descriptor = None

    def start(self, *, lock=True, stop=None) -> None:
        """Take the runner lock, record this runner and reconcile what the last one left.

        ``stop`` (an Event or a callable) ends startup reconciliation early; whatever is
        left stays recorded for the next runner.
        """
        self._stop = stop
        self.store.initialize()  # creates the state directory that holds the lock
        if lock:
            self._acquire_lock()
        try:
            self.identity = procs.identity(os.getpid())
            if self.adapters is adapter_registry.adapter:
                adapter_registry.preload()
            now = self.now()
            files = loaded_code_files()
            self._code_modules = len(sys.modules)
            self.store.write_runner(
                pid=self.identity["pid"],
                start_ticks=self.identity["startTicks"],
                boot_id=self.identity["bootId"],
                code_hash=_recorded_hash(files),
                code_files=files,
                checkout_root=paths.checkout_root(),
                protocol=PROTOCOL_VERSION,
                started_at=now,
                heartbeat_at=now,
                state="running",
                message=None,
            )
            self._load_fences()
            self._refresh_host(self.clock(), force=True)
            self._guard("journal sweep", None, self._sweep_journals)
            self.reconcile_startup()
        except BaseException:
            self.close()
            raise
        self.log(f"runner started (pid {self.identity['pid']})")

    def _stopping(self) -> bool:
        stop = self._stop
        if stop is None:
            return False
        return stop.is_set() if hasattr(stop, "is_set") else bool(stop())

    def run_forever(self, *, interval=1.0, stop=None) -> None:
        """Tick until SIGTERM/SIGINT/SIGHUP or ``stop`` (an Event or a callable).

        Tasks are never signalled on exit; the next runner adopts them.
        """
        event = stop if hasattr(stop, "wait") else threading.Event()
        requested = stop if callable(stop) and not hasattr(stop, "wait") else (lambda: False)
        with stop_signals(event):
            try:
                while not event.is_set() and not requested():
                    try:
                        self.tick()
                    except Exception:  # the loop must survive any single tick
                        self.log("tick failed:\n" + traceback.format_exc())
                    if requested():
                        break
                    event.wait(interval)
            finally:
                self.shutdown()

    def shutdown(self) -> None:
        """Record the stop and release the lock. Tasks keep running in their own sessions."""
        try:
            self.store.write_runner(state="stopped", heartbeat_at=self.now())
        except Exception:
            self.log("cannot record runner shutdown:\n" + traceback.format_exc())
        finally:
            self.close()
        self.log("runner stopped; tasks keep running")

    # -- helpers -------------------------------------------------------------------------------

    def _context(self) -> RunnerContext:
        return RunnerContext(
            store=self.store,
            now=self.now,
            settings=self.store.settings(),
            host=self._host or {},
            log=self.log,
            shared=self._shared,
        )

    def _adapter(self, task: dict):
        return self.adapters(task["adapter"])

    @staticmethod
    def _pending_exit(task: dict) -> bool:
        return (task.get("bookkeeping") or {}).get("hook") == "on_exit"

    def _guard(self, action: str, task: dict | None, function, *args) -> None:
        try:
            function(*args)
        except Exception:  # one task must never stall the others
            label = f" {task['id']}" if task else ""
            self.log(f"{action}{label} failed:\n" + traceback.format_exc())

    def _refresh_host(self, clock: float, *, force=False) -> None:
        if not force and self._host_at is not None and clock - self._host_at < self.host_interval:
            return
        self._host_at = clock
        try:
            self._host = self.host_probe()
        except Exception:  # keep the last snapshot
            self.log("host probe failed:\n" + traceback.format_exc())

    def _heartbeat(self, clock: float) -> None:
        if self._heartbeat_at is not None and clock - self._heartbeat_at < HEARTBEAT_SECONDS:
            return
        self._heartbeat_at = clock
        host = self._host or {}
        now = self.now()
        self.store.write_runner(
            heartbeat_at=now,
            state="running",
            sample={
                "host": {key: value for key, value in host.items() if key != "gpus"},
                "gpus": host.get("gpus") or [],
                "at": now,
            },
        )

    def _record_code(self, *, force=False) -> None:
        """Record every histopilot module loaded so far; lazy imports extend the list."""
        if not force and len(sys.modules) == self._code_modules:
            return
        self._code_modules = len(sys.modules)
        files = loaded_code_files()
        self.store.write_runner(code_files=files, code_hash=_recorded_hash(files))

    def _device(self, gpu: int | None) -> dict:
        return next(
            (item for item in (self._host or {}).get("gpus") or [] if item["index"] == gpu), {}
        )

    # -- wrapper journals ------------------------------------------------------------------

    @property
    def journal_dir(self) -> Path:
        return self.store.path.parent / "exits"

    def _journal(self, task: dict, attempt: int | None = None) -> dict:
        """Where the wrapper records this attempt's start and exit."""
        digest = hashlib.sha256(task["id"].encode()).hexdigest()[:32]
        base = f"{digest}-{attempt if attempt is not None else task['attempt']}"
        return {
            "spawn": str(self.journal_dir / f"{base}.spawn.json"),
            "exit": str(self.journal_dir / f"{base}.exit.json"),
        }

    def _drop_journal(self, task: dict, attempt: int | None = None) -> None:
        for path in self._journal(task, attempt).values():
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                pass  # swept at a later startup

    def _sweep_journals(self) -> None:
        if not self.journal_dir.is_dir():
            return
        cutoff = time.time() - JOURNAL_KEEP_SECONDS
        for path in self.journal_dir.iterdir():
            try:
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink()
            except OSError:
                continue

    def _recorded_exit(self, task: dict) -> tuple[bool, dict | None]:
        """(ready, record): the wrapper's exit record of a task whose leader is gone.

        Not ready while its wrapper is still writing the record (briefly); a wrapper that is
        gone without one (killed, or a task started before the wrapper) is unknown.
        """
        recorded = procs.read_exit(self._journal(task)["exit"])
        if recorded is not None:
            self._exit_waits.pop(task["id"], None)
            return True, recorded
        wrapper = (task.get("process") or {}).get("wrapper")
        if wrapper and procs.leader_alive(wrapper):
            since = self._exit_waits.setdefault(task["id"], self.clock())
            if self.clock() - since < EXIT_RECORD_WAIT_SECONDS:
                return False, None
        self._exit_waits.pop(task["id"], None)
        return True, None

    def _checkpoint(self, clock: float) -> None:
        if self._checkpoint_at is not None and clock - self._checkpoint_at < CHECKPOINT_SECONDS:
            return
        self._checkpoint_at = clock
        # A busy result leaves the log for the next attempt; nothing depends on it finishing.
        self.store.checkpoint()

    # -- tick ----------------------------------------------------------------------------------

    def tick(self) -> dict:
        result = {"started": [], "finished": [], "stopped": [], "waiting": {}}
        clock = self.clock()
        first = self._heartbeat_at is None
        self._refresh_host(clock)
        self._guard("heartbeat", None, self._heartbeat, clock)
        ctx = self._context()
        self._retry_bookkeeping(ctx, result)
        self._stops(ctx, result)
        self._reconcile(ctx, result)
        self._guard("lease pruning", None, self._prune_leases, clock)
        registry = self._lease_housekeeping(ctx, result)
        self.store.promote_ready()
        if not ctx.settings["paused"]:
            self._admit(ctx, result, registry)
        self._telemetry(ctx, clock)
        self._guard("code record", None, lambda: self._record_code(force=first))
        self._guard("store checkpoint", None, self._checkpoint, clock)
        return result

    def _retry_bookkeeping(self, ctx: RunnerContext, result: dict, *, requeues_only=False) -> None:
        pending = self.store.pending_bookkeeping()
        if not requeues_only:
            waiting = {task["id"] for task in pending}
            self._intents = {key: value for key, value in self._intents.items() if key in waiting}
            self._hooks = {key: value for key, value in self._hooks.items() if key in waiting}
        clock = self.clock()
        for task in pending:
            if requeues_only and not awaiting_requeue(task):
                continue
            if requeues_only and self._stopping():
                return  # a startup being stopped leaves the rest to the next runner
            hook = (task["bookkeeping"] or {}).get("hook")
            if hook in ("on_exit", "on_started", "on_requeue"):
                # Each try of a failing hook may wait on a lock: back off per task, except
                # right after a cancel arrives. The entry is dropped once the hook is done.
                cancel = task["stopRequest"] == "cancel"
                entry = self._hooks.get(task["id"])
                if entry is not None and entry["until"] > clock and (entry["cancel"] or not cancel):
                    continue
                count = (entry or {}).get("count", 0) + 1
                delay = min(HOOK_RETRY_MAX_SECONDS, HOOK_RETRY_SECONDS * 2 ** (count - 1))
                self._hooks[task["id"]] = {"count": count, "until": clock + delay, "cancel": cancel}
            self._guard("bookkeeping retry", task, self._retry_one, task, ctx, result)

    def _retry_one(self, task: dict, ctx: RunnerContext, result: dict) -> None:
        pending = task["bookkeeping"]
        hook = pending.get("hook")
        if hook == "on_exit" and task["state"] in ACTIVE:
            exit = dict(pending.get("exit") or {})
            self._conclude(task, exit, ctx, result, auto_resume=bool(pending.get("autoResume")))
        elif hook == "on_started" and task["state"] in ACTIVE:
            try:
                self._adapter(task).on_started(task, task["process"], task["gpu"], ctx)
            except AdapterError as error:
                if error.fatal:
                    self._abort(task, str(error))
                return
            self.store.set_fields(task["id"], bookkeeping=None)
        elif hook == "on_requeue" and task["state"] in TERMINAL:
            self._requeue(task, ctx, result)
        elif hook == "requeue_intent" and task["state"] in TERMINAL:
            self._resolve_intent(task, ctx, result)
        elif hook != "abort":
            self.store.set_fields(task["id"], bookkeeping=None)

    def _stops(self, ctx: RunnerContext, result: dict) -> None:
        for task in self.store.list(states=ACTIVE, limit=None):
            if task["state"] != "stopping" and not task["stopRequest"]:
                continue
            if self._pending_exit(task) or not task["process"]:
                continue
            self._guard("stop", task, self._stop_one, task, ctx, result)

    def _stop_one(self, task: dict, ctx: RunnerContext, result: dict) -> None:
        identity = task["process"]
        if task["signalledAt"] is None:
            procs.signal_leader(identity, signal.SIGTERM)
            self.store.set_fields(task["id"], signalled_at=self.now())
            result["stopped"].append(task["id"])
            self.log(f"stopping {task['id']} ({task['stopRequest'] or 'abort'})")
            return
        grace = task["request"].get("graceSeconds")
        if grace is None:
            grace = ctx.settings["cancelGraceSeconds"]
        elapsed = seconds_between(task["signalledAt"], self.now())
        if elapsed is not None and elapsed > grace and procs.alive(identity):
            procs.kill_group(identity)
            if task["killedAt"] is None:
                self.store.set_fields(task["id"], killed_at=self.now())
                self.log(f"killed {task['id']} after {grace:g} s grace")

    def _reconcile(self, ctx: RunnerContext, result: dict) -> None:
        for task in self.store.list(states=ACTIVE, limit=None):
            if not self._pending_exit(task):
                self._guard("reconcile", task, self._reconcile_one, task, ctx, result)

    def _reconcile_one(self, task: dict, ctx: RunnerContext, result: dict) -> None:
        identity = task["process"]
        if task["state"] in ("starting", "stopping") and not identity:
            self._resolve_starting(task, ctx, result)
            return
        child = self._procs.get(task["id"])
        returncode = None
        if child is not None:
            returncode = child.poll()
            leader_gone = returncode is not None
            if (
                not leader_gone
                and isinstance(child, procs.Spawned)
                and identity
                and not procs.leader_alive(identity)
            ):
                # The task ended and its wrapper is recording the exit status right now.
                returncode = self._settled(child)
                leader_gone = returncode is not None
            if not leader_gone and getattr(child, "orphaned", False):
                # Its wrapper died without an exit record: follow the task by identity.
                self._procs.pop(task["id"], None)
                child = None
        if child is None:
            leader_gone = not procs.leader_alive(identity) if identity else True
        if not leader_gone:
            return
        if identity and procs.alive(identity):
            # The leader exited but its session still has members: drain them first.
            began = self._draining.get(task["id"])
            if began is None:
                self._draining[task["id"]] = self.clock()
                procs.signal_group(identity, signal.SIGTERM)
            elif self.clock() - began > DRAIN_SECONDS:
                procs.kill_group(identity)
            return
        record = getattr(child, "record", None)
        if child is None and identity:
            # An adopted task: its wrapper recorded the exit status the runner cannot see.
            ready, record = self._recorded_exit(task)
            if not ready:
                return
            returncode = record["returncode"] if record else None
        exit = self._exit(task, returncode, lost=child is None and not identity, record=record)
        # A shutdown resumes the task like a reboot would; a lone signal waits for a retry.
        self._finish(task, exit, ctx, result, auto_resume=exit.get("hostStop") == "shutdown")

    def _settled(self, child) -> int | None:
        try:
            return child.wait(timeout=EXIT_RECORD_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            return None  # asked again next tick

    def _resolve_starting(self, task: dict, ctx: RunnerContext, result: dict) -> None:
        """A start that was journaled but never recorded (the runner died or the store
        refused the write): adopt the task's live process, conclude one that already
        ended, or queue the task again without spending an attempt.

        A task cancelled or paused while starting (``stopping``, no process) is resolved the
        same way, so its journaled process is stopped like any other instead of running on
        untracked; with no process at all, the stop concludes it.
        """
        journal = self._journal(task)
        spawned = procs.read_spawn(journal["spawn"])
        if spawned is not None and procs.alive(spawned["process"]):
            self._adopt_started(task, spawned, ctx)
            return
        recorded = procs.read_exit(journal["exit"]) if spawned is not None else None
        if recorded is not None:
            identity = {**spawned["process"], "wrapper": spawned.get("wrapper")}
            self.store.set_fields(task["id"], process=identity, started_at=self.now())
            current = self.store.get(task["id"]) or task
            self.log(f"{task['id']} ran and exited while no runner recorded it")
            exit = self._exit(current, recorded["returncode"], lost=False, record=recorded)
            self._finish(current, exit, ctx, result, auto_resume=exit.get("hostStop") == "shutdown")
            return
        if task["state"] == "stopping":
            self._finish(task, self._exit(task, None, lost=True), ctx, result)
            return
        self._unstart(task, "its process never started or was stopped before it was recorded")

    def _unstart(self, task: dict, why: str) -> bool:
        """Put a journaled start back in the queue; the attempt was never run."""
        self._drop_journal(task)
        undone = self.store.transition(
            task["id"],
            from_states=("starting",),
            to_state="queued",
            process=None,
            gpu=None,
            lease=None,
            started_at=None,
            detail={"event": "start-undone", "why": why},
        )
        if undone:
            self.log(f"{task['id']} queued again: {why}")
        return undone

    def _adopt_started(self, task: dict, spawned: dict, ctx: RunnerContext) -> None:
        """Record the live process of a journaled start, with a lease if it has none."""
        identity = spawned["process"]
        process = {**identity, "wrapper": spawned.get("wrapper")}
        gpu = task["gpu"] if task["request"]["lane"] == "gpu" else None
        lease = task["lease"]
        if lease is None:
            try:
                # The GPU's configured slots, as ``_launch`` writes them: a lease with one
                # run per GPU would tell legacy schedulers that the task holds it alone.
                slots = capacity.effective(ctx.settings, self._host or {})["gpuSlots"]
                with self.lease_lock():
                    lease = self.lease_writer(
                        task,
                        identity,
                        gpu,
                        cpus=task["request"]["cpuThreads"] + task["request"]["dataWorkers"],
                        ram_gb=task["request"]["ramGb"],
                        runs_per_gpu=slots.get(gpu, 1) if gpu is not None else 1,
                        supervisor=self.identity,
                        locked=True,
                    )
            except Exception:  # it runs anyway; the lease is housekeeping for legacy schedulers
                self.log(f"cannot publish the lease of {task['id']}:\n" + traceback.format_exc())
        device = self._device(gpu)
        if not self.store.transition(
            task["id"],
            from_states=("starting",),
            to_state="running",
            started_at=self.now(),
            process=process,
            gpu=gpu,
            lease=lease,
            progress=None,
            resources={"gpuName": device.get("name"), "gpuUuid": device.get("uuid")},
            detail={"adopted": identity["pid"], "gpu": gpu},
        ):
            current = self.store.get(task["id"])
            if current is not None and current["state"] == "stopping":
                self.store.set_fields(task["id"], process=process, gpu=gpu, lease=lease)
            return
        self.log(f"adopted the journaled start of {task['id']} (pid {identity['pid']})")
        current = self.store.get(task["id"]) or task
        self._started_hook(current, identity, gpu, ctx)

    def _exit(
        self, task: dict, returncode: int | None, *, lost: bool, record: dict | None = None
    ) -> dict:
        """The exit adapters classify. One the host caused (``host_stop``) reads as lost:
        interrupted, never failed, and resumable like a task lost with the machine."""
        boot = (self.identity or {}).get("bootId") or _boot_id()
        host = None if lost else host_stop(task, returncode, record, boot)
        return {
            "returncode": returncode,
            "lost": lost or host is not None,
            "hostStop": host,
            "signalled": task["signalledAt"] is not None,
            "killed": task["killedAt"] is not None,
            "stopReason": task["stopRequest"],
        }

    def _finish(
        self, task: dict, exit: dict, ctx: RunnerContext, result: dict, *, auto_resume=False
    ) -> None:
        """A task's process group is gone: release capacity, then let the adapter classify."""
        self._procs.pop(task["id"], None)
        self._draining.pop(task["id"], None)
        self._samples.pop(task["id"], None)
        if task["lease"]:
            try:
                self.lease_remover(task["lease"])
            except Exception:  # stale own leases are removed on a later tick
                self.log(f"cannot remove lease {task['lease']}:\n" + traceback.format_exc())
        self._conclude(task, exit, ctx, result, auto_resume=auto_resume)

    def _conclude(
        self, task: dict, exit: dict, ctx: RunnerContext, result: dict, *, auto_resume=False
    ) -> None:
        # The row is read again so the adapter sees the current stop request. A cancel that
        # lands while the adapter classifies the exit makes the conclusion stale; the second
        # pass classifies the exit as the cancel it has become.
        for _ in range(2):
            current = self.store.get(task["id"])
            if current is None or current["state"] not in ACTIVE:
                return
            exit = {
                **exit,
                "stopReason": current["stopRequest"] or exit.get("stopReason"),
                "signalled": bool(exit.get("signalled")) or current["signalledAt"] is not None,
                "killed": bool(exit.get("killed")) or current["killedAt"] is not None,
            }
            try:
                raw = self._adapter(current).on_exit(current, exit, ctx)
            except AdapterError as error:
                if error.transient:
                    self.store.set_fields(
                        current["id"],
                        exit=exit,
                        lease=None,
                        bookkeeping={
                            "hook": "on_exit",
                            "exit": exit,
                            "autoResume": auto_resume,
                            "abortError": _abort_error(current),
                        },
                    )
                    self.log(f"exit bookkeeping for {current['id']} will be retried: {error}")
                    return
                raw = outcome("failed", "error", str(error))
            except Exception as error:  # adapter bugs fail the task, not the runner
                self.log(f"adapter on_exit for {current['id']} failed:\n" + traceback.format_exc())
                raw = outcome("failed", "error", f"The task adapter failed: {error}")
            status = self._apply(
                current, exit, _normalize_outcome(raw), ctx, result, auto_resume=auto_resume
            )
            if status != "stale":
                return

    def _apply(
        self, task: dict, exit: dict, result_outcome: dict, ctx, result, *, auto_resume
    ) -> str:
        """Conclude the task with the requeue that follows it, in one transaction."""
        state = result_outcome["state"]
        aborted = _abort_error(task)
        if aborted is not None:
            state = "failed"
            result_outcome = {
                **result_outcome,
                "state": "failed",
                "exitReason": "error",
                "error": aborted,
            }
        stop = exit.get("stopReason")
        if state == "requeue" and stop == "cancel":
            # A cancel supersedes a requeue the adapter asked for (a pause or a busy output).
            state = "cancelled"
            result_outcome = {**result_outcome, "state": "cancelled", "exitReason": "cancelled"}
        now = self.now()
        terminal = "interrupted" if state == "requeue" else state
        reason = result_outcome["exitReason"]
        error = result_outcome["error"]
        follow_up = None
        lost_device = (
            reason == "cuda_failure" and task.get("gpu") is not None and device_lost(error)
        )
        if state == "requeue":
            follow_up = {"hook": "on_requeue", "reason": reason or "requeue", "since": now}
        elif stop != "cancel" and lost_device and self._device_loss_retry(task):
            # The device failed, not the task: interrupted, and resumed once the GPU answers.
            terminal = "interrupted"
            follow_up = self._retry_intent(task, "device-lost", "deviceLossRequeues", now)
        elif stop != "cancel" and reason == "oom" and self._contention(task, ctx):
            follow_up = self._retry_intent(task, "gpu-contention", "contentionRequeues", now)
        elif stop != "cancel" and reason == "oom" and self._oom_retry(task):
            follow_up, error = self._oom_backoff(task, ctx, now, error)
        elif (
            stop != "cancel"
            and auto_resume
            and terminal == "interrupted"
            and ctx.settings["autoResume"]
        ):
            follow_up = {"hook": "requeue_intent", "reason": "auto-resume", "since": now}
        record = {
            "reason": reason,
            "returncode": exit.get("returncode"),
            "lost": bool(exit.get("lost")),
            "signalled": bool(exit.get("signalled")),
            "killed": bool(exit.get("killed")),
            "stopReason": stop,
            "error": error,
        }
        status = self.store.conclude(
            task["id"],
            to_state=terminal,
            stop_request=stop,
            follow_up=follow_up,
            exit=record,
            error=error,
            finished_at=now,
            lease=None,
            waiting_reason=None,
            detail={"exitReason": reason},
        )
        if status != "concluded":
            return status
        self._drop_journal(task)
        self._exit_waits.pop(task["id"], None)
        self._hooks.pop(task["id"], None)
        if task["startedAt"] and not exit.get("lost"):
            self._guard("measurement", task, self._measure, task, record, result_outcome, now)
        result["finished"].append(task["id"])
        self.log(f"finished {task['id']}: {terminal} ({reason})")
        if task.get("gpu") is not None:
            if lost_device:
                self._fence_gpu(task["gpu"], error)
            elif terminal == "succeeded" and task["gpu"] in self._gpu_fences:
                self._gpu_fences.pop(task["gpu"], None)
                self._save_fences()
        if reason == "busy" and state == "requeue":
            count = self._busy.get(task["id"], {}).get("count", 0) + 1
            cap = (
                BUSY_BACKOFF_MAX_SECONDS
                if count < BUSY_PATIENT_AFTER
                else BUSY_BACKOFF_LONG_SECONDS
            )
            delay = min(cap, BUSY_BACKOFF_SECONDS * 2 ** (count - 1))
            self._busy[task["id"]] = {
                "count": count,
                "until": self.clock() + delay,
                "reason": error,
            }
        else:
            self._busy.pop(task["id"], None)
        if follow_up is not None:
            self._follow_up(task["id"], ctx, result)
        return status

    @staticmethod
    def _oom_retry(task: dict) -> bool:
        """One OOM retry per run: pauses, auto-resumes and busy requeues never count, and
        a manual retry or resume starts a new budget (the store resets the counters)."""
        return int(task["adapterData"].get("oomRetries") or 0) < 1

    @staticmethod
    def _device_loss_retry(task: dict) -> bool:
        return int(task["adapterData"].get("deviceLossRequeues") or 0) < DEVICE_LOSS_MAX_REQUEUES

    @staticmethod
    def _retry_intent(task: dict, reason: str, counter: str, now: str) -> dict:
        """A requeue intent that spends one unit of ``counter``, never the OOM retry."""
        spent = int(task["adapterData"].get(counter) or 0)
        return {
            "hook": "requeue_intent",
            "reason": reason,
            "since": now,
            "adapterDataPatch": {counter: spent + 1},
        }

    def _contention(self, task: dict, ctx: RunnerContext) -> bool:
        """Whether an OOM happened on a GPU holding memory no HistoPilot task accounts for.

        Measured right after the exit (the task's own memory is gone): used memory beyond
        the GPU's reserve and the requests of the other tasks running there. Legacy leases
        on the GPU have no request, so their GPUs are never judged.
        """
        gpu = task.get("gpu")
        if gpu is None or task["request"]["lane"] != "gpu":
            return False
        if int(task["adapterData"].get("contentionRequeues") or 0) >= CONTENTION_MAX_REQUEUES:
            return False
        if self._foreign_gpu.get(gpu):
            return False
        self._refresh_host(self.clock(), force=True)
        device = self._device(gpu)
        total, free = device.get("totalMemoryGb"), device.get("freeMemoryGb")
        if total is None or free is None:
            return False
        reserve = capacity.effective(ctx.settings, self._host or {})["vramReserves"].get(gpu, 0.0)
        others = sum(
            other["request"]["vramGb"]
            for other in self.store.list(states=ACTIVE, limit=None)
            if other["id"] != task["id"]
            and other.get("gpu") == gpu
            and other["request"]["lane"] == "gpu"
        )
        unexplained = total - free - reserve - others
        if unexplained < FOREIGN_VRAM_GB:
            return False
        self.log(
            f"{task['id']} ran out of GPU memory while {unexplained:.1f} GiB of GPU {gpu} was in "
            "use outside HistoPilot; it waits for that memory instead of asking for more"
        )
        return True

    def _vram_cap(self, ctx: RunnerContext) -> float | None:
        """The largest VRAM request any GPU here can admit: its total less its reserve."""
        host = ctx.host or {}
        gpus = [gpu for gpu in host.get("gpus") or [] if gpu.get("totalMemoryGb") is not None]
        if not gpus:
            return None
        reserves = capacity.effective(ctx.settings, host)["vramReserves"]
        largest = max(gpu["totalMemoryGb"] - reserves.get(gpu["index"], 0.0) for gpu in gpus)
        return max(0.0, math.floor(largest * 1000) / 1000)

    def _oom_backoff(self, task: dict, ctx: RunnerContext, now: str, error):
        """The OOM retry intent with a raised VRAM request, or a failure when none can fit."""
        vram = float(task["request"].get("vramGb") or 0.0)
        cap = self._vram_cap(ctx)
        if cap is not None and vram >= cap:
            return None, (
                f"{NEEDS_MORE_VRAM}: it ran out of GPU memory with {vram:.1f} GiB requested, "
                f"and the largest GPU offers {cap:.1f} GiB after its reserve."
            )
        raised = round(vram * OOM_VRAM_FACTOR, 3)
        if cap is not None:
            raised = min(raised, cap)
        retries = int(task["adapterData"].get("oomRetries") or 0)
        return {
            "hook": "requeue_intent",
            "reason": "oom-backoff",
            "since": now,
            "requestPatch": {"vramGb": raised},
            "adapterDataPatch": {"oomRetries": retries + 1},
        }, error

    @staticmethod
    def _fence_seconds(failures: int) -> float:
        return min(GPU_FENCE_MAX_SECONDS, GPU_FENCE_SECONDS * 2 ** (failures - 1))

    def _fence_gpu(self, gpu: int, error: str | None = None) -> None:
        now = self.clock()
        fence = self._gpu_fences.get(gpu)
        if fence is not None and not fence["verified"]:
            return  # the same incident: failures ending while fenced do not extend it
        failures = (fence or {}).get("failures", 0) + 1
        seconds = self._fence_seconds(failures)
        self._gpu_fences[gpu] = {
            "failures": failures,
            "until": now + seconds,
            "verified": False,
            "error": (error or "")[:500] or None,
        }
        self._save_fences()
        self.log(
            f"GPU {gpu} reported a device failure; no new work on it for {seconds:.0f} s "
            "and until a fresh process can use it"
        )

    def _save_fences(self) -> None:
        """Keep fences on the runner row (wall-clock times), so a restart keeps them."""
        clock, now = self.clock(), self.now()
        start = parse_iso(now)
        saved = {
            str(gpu): {
                "failures": fence["failures"],
                "verified": fence["verified"],
                "error": fence.get("error"),
                "until": (start + timedelta(seconds=max(0.0, fence["until"] - clock))).isoformat()
                if start
                else now,
            }
            for gpu, fence in self._gpu_fences.items()
        }
        self._guard("fence record", None, lambda: self.store.write_runner(gpu_fences=saved))

    def _load_fences(self) -> None:
        row = self.store.runner() or {}
        clock, now = self.clock(), self.now()
        fences = {}
        for gpu, fence in (row.get("gpuFences") or {}).items():
            try:
                remaining = seconds_between(now, fence.get("until")) or 0.0
                fences[int(gpu)] = {
                    "failures": int(fence.get("failures") or 1),
                    "until": clock + max(0.0, remaining),
                    "verified": bool(fence.get("verified")),
                    "error": fence.get("error"),
                }
            except (TypeError, ValueError, AttributeError):
                continue
        self._gpu_fences = fences
        for gpu, fence in fences.items():
            if not fence["verified"]:
                self.log(f"GPU {gpu} stays fenced after a device failure seen by an earlier runner")

    def _probe_result(self, gpu: int):
        """The finished CUDA probe of ``gpu``, starting one in a thread when none runs."""
        entry = self._probes.get(gpu)
        if entry is None:
            entry = self._probes[gpu] = {"result": None}

            def probe():
                try:
                    entry["result"] = self.cuda_probe(gpu)
                except Exception as error:  # a broken probe is a failed check
                    entry["result"] = ("failed", str(error))

            threading.Thread(target=probe, name=f"cuda-probe-{gpu}", daemon=True).start()
            return None
        if entry["result"] is None:
            return None
        self._probes.pop(gpu, None)
        return entry["result"]

    def _fence_reason(self, gpu: int, fence: dict, now: float) -> str | None:
        """Why ``gpu`` takes no new work, or None once it is usable again."""
        if fence["verified"]:
            return None
        if fence["until"] > now:
            return (
                f"GPU {gpu} reported a device failure; retrying in "
                f"{max(1, round(fence['until'] - now))} s"
            )
        result = self._probe_result(gpu)
        if result is None:
            return f"GPU {gpu} reported a device failure; checking that it works again"
        status, message = result
        if status in ("ok", "unavailable"):
            if status == "unavailable":
                self.log(f"GPU {gpu} cannot be checked ({message}); reopening it after the fence")
            else:
                self.log(f"GPU {gpu} answers again; it takes new work")
            fence["verified"] = True
            self._save_fences()
            return None
        fence["until"] = now + self._fence_seconds(fence["failures"])
        self._save_fences()
        self.log(f"GPU {gpu} still fails its check: {message}")
        return (
            f"GPU {gpu} is not usable ({message}); checking again in "
            f"{max(1, round(fence['until'] - now))} s"
        )

    def _gpu_host(self, host: dict) -> tuple[dict, str | None]:
        """The host without fenced GPUs, and a waiting reason when none is left."""
        now = self.clock()
        fenced = {}
        for gpu, fence in list(self._gpu_fences.items()):
            reason = self._fence_reason(gpu, fence, now)
            if reason:
                fenced[gpu] = reason
        if not fenced:
            return host, None
        gpus = [gpu for gpu in host.get("gpus") or [] if gpu["index"] not in fenced]
        return {**host, "gpus": gpus}, (None if gpus else "; ".join(fenced.values()))

    # -- requeues ------------------------------------------------------------------------------

    def _follow_up(self, task_id: str, ctx: RunnerContext, result: dict) -> None:
        task = self.store.get(task_id)
        if task is None or not awaiting_requeue(task):
            return
        if task["bookkeeping"]["hook"] == "on_requeue":
            self._requeue(task, ctx, result)
        else:
            self._resolve_intent(task, ctx, result)

    def _resolve_intent(self, task: dict, ctx: RunnerContext, result: dict) -> None:
        """Decide a requeue intent: requeue, a definitive no, or retry later."""
        pending = task["bookkeeping"]
        reason = pending.get("reason") or "requeue"
        if task["stopRequest"] == "cancel":
            self._cancel_instead(task, ctx, result)
            return
        retry = self._intents.get(task["id"])
        if retry is not None and retry["until"] > self.clock():
            return
        if reason == "auto-resume" and not ctx.settings["autoResume"]:
            allowed = False
        else:
            try:
                allowed = bool(self._adapter(task).can_requeue(task, ctx))
            except AdapterError as error:
                if error.transient and error.retry_after is not None:
                    # Not ready yet (a probe is running): ask again soon, without backoff.
                    entry = self._intents.get(task["id"], {"count": 0})
                    self._intents[task["id"]] = {**entry, "until": self.clock() + error.retry_after}
                    return
                if error.transient:
                    self._defer_intent(task, str(error))
                    return
                self.log(f"{task['id']} cannot be requeued: {error}")
                allowed = False
            except Exception:  # adapter bugs decide "no", not the runner's fate
                self.log(f"adapter can_requeue for {task['id']} failed:\n" + traceback.format_exc())
                allowed = False
        self._intents.pop(task["id"], None)
        if not allowed:
            if self.store.settle_bookkeeping(task["id"], "requeue_intent"):
                self.log(f"{task['id']} is not requeued ({reason})")
            return
        self._requeue(task, ctx, result)

    def _defer_intent(self, task: dict, message: str) -> None:
        pending = task["bookkeeping"]
        elapsed = seconds_between(pending.get("since"), self.now())
        if elapsed is None or elapsed >= INTENT_GIVE_UP_SECONDS:
            self._intents.pop(task["id"], None)
            if self.store.settle_bookkeeping(
                task["id"], "requeue_intent", error=f"Auto-resume gave up: {message}"
            ):
                self.log(f"auto-resume of {task['id']} gave up: {message}")
            return
        count = self._intents.get(task["id"], {}).get("count", 0) + 1
        delay = min(INTENT_RETRY_MAX_SECONDS, INTENT_RETRY_SECONDS * 2 ** (count - 1))
        self._intents[task["id"]] = {"count": count, "until": self.clock() + delay}
        self.log(
            f"requeue of {task['id']} ({pending.get('reason')}) undecided: {message}; "
            f"retrying in {delay:.0f} s"
        )

    def _requeue(self, task: dict, ctx: RunnerContext, result: dict) -> None:
        """Carry out a decided requeue unless a cancel arrived first."""
        pending = task["bookkeeping"]
        reason = pending.get("reason") or "requeue"
        if task["stopRequest"] == "cancel":
            self._cancel_instead(task, ctx, result)
            return
        try:
            self._adapter(task).on_requeue(task, ctx)
        except AdapterError as error:
            if error.transient:
                self.store.set_fields(task["id"], bookkeeping={**pending, "hook": "on_requeue"})
                self.log(f"requeue of {task['id']} deferred: {error}")
            elif self.store.settle_bookkeeping(task["id"], pending["hook"], error=str(error)):
                self.log(f"requeue of {task['id']} refused: {error}")
            return
        except Exception as error:  # adapter bugs end the requeue, not the runner
            self.log(f"adapter on_requeue for {task['id']} failed:\n" + traceback.format_exc())
            self.store.settle_bookkeeping(
                task["id"], pending["hook"], error=f"The task adapter failed: {error}"
            )
            return
        if self.store.requeue(
            [task["id"]],
            reason=reason,
            request_patch=pending.get("requestPatch"),
            adapter_data_patch=pending.get("adapterDataPatch"),
            unless_cancelled=True,
            manual=False,
        ):
            self.log(f"requeued {task['id']} ({reason})")
            return
        current = self.store.get(task["id"])
        if current is not None and current["stopRequest"] == "cancel" and awaiting_requeue(current):
            self._cancel_instead(current, ctx, result)

    def _cancel_instead(self, task: dict, ctx: RunnerContext, result: dict) -> None:
        """A cancel reached a concluded task awaiting its requeue: conclude it cancelled.

        The adapter classifies the exit again as a cancel so its own record agrees; work
        that completed or failed on its own keeps that outcome.
        """
        previous = task.get("exit") or {}
        exit = {
            "returncode": previous.get("returncode"),
            "lost": bool(previous.get("lost")),
            "signalled": bool(previous.get("signalled")),
            "killed": bool(previous.get("killed")),
            "stopReason": "cancel",
        }
        try:
            raw = self._adapter(task).on_exit(task, exit, ctx)
        except AdapterError as error:
            if error.transient:
                self.log(f"cancel of {task['id']} will be retried: {error}")
                return
            raw = outcome("cancelled", "cancelled")
        except Exception:
            self.log(f"adapter on_exit for {task['id']} failed:\n" + traceback.format_exc())
            raw = outcome("cancelled", "cancelled")
        value = _normalize_outcome(raw)
        if value["state"] in ("succeeded", "failed", "cancelled"):
            state, reason, error = value["state"], value["exitReason"], value["error"]
        else:
            state, reason, error = "cancelled", "cancelled", None
        if self.store.transition(
            task["id"],
            from_states=TERMINAL,
            to_state=state,
            exit={**previous, "reason": reason, "stopReason": "cancel", "error": error},
            error=error,
            bookkeeping=None,
            finished_at=self.now(),
            detail={"exitReason": reason, "requeueCancelled": True},
        ):
            self._intents.pop(task["id"], None)
            result["finished"].append(task["id"])
            self.log(f"finished {task['id']}: {state} (cancelled before its requeue)")

    def _measure(self, task: dict, record: dict, result_outcome: dict, now: str) -> None:
        resources = task.get("resources") or {}
        measurement = result_outcome["measurement"] or {}
        request = task["request"]
        wall = seconds_between(task["startedAt"], now)
        row = {
            "taskId": task["id"],
            "attempt": task["attempt"],
            "kind": task["kind"],
            "lane": request["lane"],
            "workloadKey": request.get("workloadKey"),
            "workload": request.get("workload"),
            "gpuIndex": task["gpu"],
            "gpuName": resources.get("gpuName"),
            "gpuUuid": resources.get("gpuUuid"),
            "peakVramGb": None,
            "peakPrivateRamGb": resources.get("peakPrivateRamGb"),
            "meanCpuCores": resources.get("meanCpuCores"),
            "epochs": None,
            "wallSeconds": wall,
            "secondsPerEpoch": None,
            "meanConcurrency": resources.get("meanConcurrency"),
            "exitReason": record["reason"],
            "startedAt": task["startedAt"],
            "finishedAt": now,
        }
        row.update(
            {
                key: value
                for key, value in measurement.items()
                if key in MEASUREMENT_COLUMNS and value is not None
            }
        )
        if row["secondsPerEpoch"] is None and row["epochs"] and row["wallSeconds"]:
            row["secondsPerEpoch"] = row["wallSeconds"] / row["epochs"]
        self.store.record_measurement(row)

    def _abort(self, task: dict, message: str) -> None:
        """A fatal on_started hook: stop the task and fail it once its processes are gone."""
        if task["process"]:
            procs.signal_leader(task["process"], signal.SIGTERM)
        self.store.transition(
            task["id"],
            from_states=("starting", "running"),
            to_state="stopping",
            signalled_at=self.now(),
            error=message,
            bookkeeping={"hook": "abort", "error": message},
            detail={"abort": message},
        )
        self.log(f"aborting {task['id']}: {message}")

    def _started_hook(self, task: dict, identity: dict, gpu: int | None, ctx) -> None:
        try:
            self._adapter(task).on_started(task, identity, gpu, ctx)
        except AdapterError as error:
            if error.transient:
                self.store.set_fields(task["id"], bookkeeping={"hook": "on_started"})
            else:
                self._abort(task, str(error))
        except Exception as error:
            self.log(f"adapter on_started for {task['id']} failed:\n" + traceback.format_exc())
            self._abort(task, f"The task adapter failed: {error}")

    # -- leases --------------------------------------------------------------------------------

    def _prune_leases(self, clock: float) -> None:
        """Every few minutes, remove leases whose owner process is confirmed dead.

        Older checkouts' tmux schedulers pruned them as they admitted work; without them
        a crashed worker's lease would count as foreign load forever. A busy registry is
        tried again at the next interval.
        """
        if self._pruned_at is not None and clock - self._pruned_at < LEASE_PRUNE_SECONDS:
            return
        self._pruned_at = clock
        try:
            removed = self.lease_pruner(timeout=LEASE_PRUNE_LOCK_SECONDS)
        except StorageError as error:
            if error.code == "PROJECT_BUSY":
                return
            raise
        if removed:
            self.log(f"pruned {len(removed)} lease(s) of dead workers: {', '.join(removed)}")

    def _lease_housekeeping(self, ctx: RunnerContext, result: dict) -> dict:
        """Read the lease registry once per tick.

        Drops stale leases of this store's tasks, adopts or stops task processes the store
        does not track, and refreshes the foreign load that telemetry attributes to GPUs.
        """
        try:
            registry = self.lease_reader()
        except Exception as error:  # unknown registry: nothing foreign can be attributed
            self._foreign_gpu = Counter()
            message = str(error)
            if message != self._registry_error:
                self.log(f"cannot read the resource lease registry: {message}")
            self._registry_error = message
            return {"registry": None, "error": message, "foreign": []}
        self._registry_error = None
        active = {task["id"]: task for task in self.store.list(states=ACTIVE, limit=None)}
        kept, orphans = [], set()
        for lease in registry:
            verdict = "keep"
            try:
                verdict = self._own_lease(lease, active, ctx, result)
            except Exception:
                self.log(f"lease {lease.get('file')} check failed:\n" + traceback.format_exc())
            if verdict == "drop":
                continue
            if verdict == "own":
                active[lease["taskId"]] = self.store.get(lease["taskId"]) or {
                    "id": lease["taskId"],
                    "process": lease.get("process"),
                }
            if verdict == "orphan":
                orphans.add(lease["file"])
            kept.append(lease)
        self._orphans = {key: value for key, value in self._orphans.items() if key in orphans}
        foreign = self._foreign(kept, set(active))
        self._foreign_gpu = Counter(
            lease["gpu"] for lease in foreign if lease.get("gpu") is not None
        )
        return {"registry": kept, "error": None, "foreign": foreign}

    def _foreign(self, registry: list[dict], own_ids: set) -> list[dict]:
        """Live leases that are not accounted as this runner's running tasks.

        A runner lease counts while its task process lives: its supervisor (a runner) being
        alive says nothing about the task. Untracked processes being stopped count too.
        """
        foreign = []
        for lease in registry:
            if lease.get("invalid"):
                continue
            if lease.get("file") in self._orphans:
                foreign.append(lease)
            elif lease.get("kind") == "task-center" and lease.get("process"):
                if lease.get("taskId") not in own_ids and procs.alive(lease["process"]):
                    foreign.append(lease)
            elif lease.get("live") and lease.get("taskId") not in own_ids:
                foreign.append(lease)
        return foreign

    def _own_lease(self, lease: dict, active: dict, ctx: RunnerContext, result: dict) -> str:
        """Classify one lease: "keep", "drop" (removed), "own" (adopted) or "orphan"."""
        if lease.get("invalid") or lease.get("kind") != "task-center" or not lease.get("taskId"):
            return "keep"
        task_id, identity = lease["taskId"], lease.get("process")
        task = active.get(task_id)
        if task is not None and _same_process(task.get("process"), identity):
            return "keep"
        if task is None:
            task = self.store.get(task_id)
            if task is None:
                return "keep"  # a task of another state directory's runner
        if not identity or not procs.alive(identity):
            try:
                self.lease_remover(lease["file"])
            except Exception:
                self.log(f"cannot remove stale lease {lease['file']}")
                return "keep"
            return "drop"
        supervisor = lease.get("supervisor")
        if not _same_process(supervisor, self.identity) and procs.leader_alive(supervisor):
            # Another live runner (another state directory sharing this registry) runs it.
            return "keep"
        # A live process this store does not track, e.g. after a runner died between the
        # spawn and recording the start: a queued task continues with it, others stop it.
        if task["state"] in ("queued", "starting") and self._adopt(task, lease, ctx):
            return "own"
        self._stop_orphan(task, lease, ctx)
        return "orphan"

    def _adopt(self, task: dict, lease: dict, ctx: RunnerContext) -> bool:
        identity = lease["process"]
        gpu = lease.get("gpu") if task["request"]["lane"] == "gpu" else None
        device = self._device(gpu)
        if not self.store.transition(
            task["id"],
            from_states=("queued", "starting"),
            to_state="running",
            started_at=self.now(),
            process=identity,
            gpu=gpu,
            lease=lease["file"],
            waiting_reason=None,
            progress=None,
            resources={"gpuName": device.get("name"), "gpuUuid": device.get("uuid")},
            detail={"adopted": identity.get("pid"), "gpu": gpu},
        ):
            return False
        self.log(f"adopted an untracked process of {task['id']} (pid {identity.get('pid')})")
        current = self.store.get(task["id"]) or task
        self._started_hook(current, identity, gpu, ctx)
        return True

    def _stop_orphan(self, task: dict, lease: dict, ctx: RunnerContext) -> None:
        """Stop a process of a task that must not run, as gracefully as a cancel."""
        identity = lease["process"]
        entry = self._orphans.get(lease["file"])
        if entry is None or entry["pid"] != identity.get("pid"):
            if not procs.signal_leader(identity, signal.SIGTERM):
                procs.signal_group(identity, signal.SIGTERM)
            self._orphans[lease["file"]] = {
                "pid": identity.get("pid"),
                "at": self.clock(),
                "killed": False,
            }
            self.log(
                f"stopping an untracked process of {task['id']} ({task['state']}, "
                f"pid {identity.get('pid')})"
            )
            return
        grace = task["request"].get("graceSeconds")
        if grace is None:
            grace = ctx.settings["cancelGraceSeconds"]
        if self.clock() - entry["at"] > grace:
            procs.kill_group(identity)
            if not entry["killed"]:
                entry["killed"] = True
                self.log(f"killed an untracked process of {task['id']} after {grace:g} s grace")

    # -- admission -----------------------------------------------------------------------------

    def _admit(self, ctx: RunnerContext, result: dict, registry: dict) -> None:
        host = self._host
        queued = self.store.list(states=("queued",), limit=None)
        if not queued:
            return
        waiting = result["waiting"]
        if host is None:
            for task in queued:
                waiting[task["id"]] = "Waiting for host capacity information"
            self._write_waiting(queued, waiting)
            return
        active = self.store.list(states=ACTIVE, limit=None)
        own = {task["id"] for task in active}
        pending = own | {task["id"] for task in queued}
        self._busy = {key: value for key, value in self._busy.items() if key in pending}
        if registry["registry"] is None:  # unknown registry: admit nothing
            for task in queued:
                waiting[task["id"]] = (
                    f"Cannot read the resource lease registry: {registry['error']}"
                )
            self._write_waiting(queued, waiting)
            return
        running = [task for task in active if not self._pending_exit(task)]
        effective = capacity.effective(ctx.settings, host)
        usage = capacity.usage(running, registry["foreign"], host, now=self.now())
        claims = []
        exclusive = {task["exclusiveKey"] for task in active if task["exclusiveKey"]}
        held = {owner["key"] for owner in self.store.owners(live_only=True) if owner["held"]}
        gpu_host, fenced = self._gpu_host(host)
        blocked = {"all": None, "gpu": fenced, "cpu": None}
        for task in queued:
            request = task["request"]
            lane_scope = (
                "gpu" if request["lane"] == "gpu" else (None if request.get("service") else "cpu")
            )
            if task["ownerKey"] in held:
                waiting[task["id"]] = ON_HOLD
            elif self._busy.get(task["id"], {}).get("until", 0) > self.clock():
                waiting[task["id"]] = self._busy_wait(task["id"])
            elif blocked["all"]:
                waiting[task["id"]] = blocked["all"]
            elif task["exclusiveKey"] and task["exclusiveKey"] in exclusive:
                waiting[task["id"]] = EXCLUSIVE_WAIT
            elif lane_scope and blocked[lane_scope]:
                waiting[task["id"]] = blocked[lane_scope]
            else:
                try:
                    prepared = self._prepare(task, ctx, result)
                    if prepared is None:
                        continue
                    target = gpu_host if lane_scope == "gpu" else host
                    # A cheap check without the registry lock first; the launch checks again
                    # under the lock before it spawns.
                    admitted, gpu, reason = capacity.admit(prepared, usage, target, effective)
                    scope = capacity.blocking_scope(reason)
                    if admitted:
                        admitted, gpu, reason, scope = self._launch(
                            prepared,
                            effective,
                            ctx,
                            result,
                            running=running,
                            claims=claims,
                            host=target,
                            own=own,
                        )
                    if reason:
                        waiting[task["id"]] = reason
                        if scope:
                            blocked[scope] = reason
                    if admitted:
                        capacity.claim(usage, prepared, gpu)
                        claims.append((prepared, gpu))
                        if prepared["exclusiveKey"]:
                            exclusive.add(prepared["exclusiveKey"])
                except Exception:
                    self.log(f"admission of {task['id']} failed:\n" + traceback.format_exc())
        self._write_waiting(queued, waiting)

    def _busy_wait(self, task_id: str) -> str:
        entry = self._busy[task_id]
        if entry["count"] < BUSY_REPORT_AFTER:
            return BUSY_WAIT
        cause = (entry.get("reason") or BUSY_WAIT).strip().rstrip(".")
        return f"{cause} ({entry['count']} tries so far; trying again later)"

    def _write_waiting(self, queued: list[dict], waiting: dict) -> None:
        changes = {
            task["id"]: waiting[task["id"]]
            for task in queued
            if task["id"] in waiting and waiting[task["id"]] != task["waitingReason"]
        }
        if changes:
            self.store.set_waiting_reasons(changes)

    def _prepare(self, task: dict, ctx: RunnerContext, result: dict) -> dict | None:
        try:
            prepared = self._adapter(task).prepare(task, ctx)
            if not prepared:
                return task
            if "skip" in prepared:
                skip = _normalize_outcome(prepared["skip"])
                state = skip["state"] if skip["state"] in TERMINAL else "succeeded"
                if self.store.transition(
                    task["id"],
                    from_states=("queued",),
                    to_state=state,
                    exit={"reason": skip["exitReason"], "error": skip["error"]},
                    error=skip["error"],
                    waiting_reason=None,
                    detail={"skipped": skip["exitReason"]},
                ):
                    result["finished"].append(task["id"])
                    self.log(f"skipped {task['id']}: {skip['exitReason']}")
                return None
            changes = {}
            if prepared.get("request"):
                changes["request"] = normalize_request(prepared["request"])
            if prepared.get("command"):
                changes["command"] = normalize_command(prepared["command"])
            if changes:
                self.store.set_fields(task["id"], **changes)
                task = {**task, **changes}
            return task
        except AdapterError as error:
            if error.transient:
                result["waiting"][task["id"]] = str(error)
                return None
            self._fail_queued(task, str(error), result)
            return None
        except (ValueError, TypeError) as error:
            self._fail_queued(task, f"The task adapter prepared an invalid task: {error}", result)
            return None

    def _fail_queued(self, task: dict, message: str, result: dict) -> None:
        if self.store.transition(
            task["id"],
            from_states=("queued",),
            to_state="failed",
            exit={"reason": "error", "error": message},
            error=message,
            waiting_reason=None,
        ):
            result["finished"].append(task["id"])
            self.log(f"failed {task['id']} before start: {message}")

    def _admit_locked(self, task, effective, *, running, claims, host, own):
        """Admission against a registry read under its lock, so no legacy scheduler can
        take the same capacity between this decision and the new task's lease."""
        registry = self.lease_reader()
        for lease in registry:
            if (
                lease.get("taskId") == task["id"]
                and lease.get("kind") == "task-center"
                and lease.get("process")
                and procs.alive(lease["process"])
            ):
                return False, None, EARLIER_PROCESS_WAIT, None
        foreign = self._foreign(registry, own | {claimed["id"] for claimed, _ in claims})
        usage = capacity.usage(running, foreign, host, now=self.now())
        for claimed, gpu in claims:
            capacity.claim(usage, claimed, gpu)
        admitted, gpu, reason = capacity.admit(task, usage, host, effective)
        return admitted, gpu, reason, capacity.blocking_scope(reason)

    def _launch(self, task, effective, ctx: RunnerContext, result: dict, **admission):
        """Start an admitted task; returns (started, gpu, waiting reason, blocking scope).

        The registry lock is held from the final admission through the lease write, as
        legacy schedulers do. The start is journaled first (``queued`` → ``starting``) and
        the wrapper records the spawned process, so a runner that dies at any point leaves
        either a queued task, or a start the next runner adopts or undoes. No child
        survives a start that cannot be recorded.
        """
        request = task["request"]
        owner = self.store.owner(task["ownerKey"])
        if owner is not None and owner["held"]:  # held after this pass listed the owners
            return False, None, ON_HOLD, None
        stack = ExitStack()
        try:
            stack.enter_context(self.lease_lock())
        except Exception as error:
            busy = getattr(error, "code", None) == "PROJECT_BUSY"
            reason = REGISTRY_WAIT if busy else f"Cannot use the resource lease registry: {error}"
            return False, None, reason, "all"
        with stack:
            admitted, gpu, reason, scope = self._admit_locked(task, effective, **admission)
            if not admitted:
                return False, None, reason, scope
            if not self.store.transition(
                task["id"],
                from_states=("queued",),
                to_state="starting",
                gpu=gpu,
                waiting_reason=None,
                detail={"gpu": gpu},
                unless_owner_held=True,
            ):
                return False, None, None, None  # cancelled, held or changed since listing
            journal = self._journal(task)
            try:
                child = self.spawner(
                    task["command"], lane=request["lane"], gpu=gpu, task=task, journal=journal
                )
            except Exception as error:  # a command that cannot start fails the task
                self._drop_journal(task)
                self._fail_starting(task, f"Cannot start the task: {error}", result)
                return False, None, None, None
            lease = None
            try:
                identity = procs.identity(child.pid)
                lease = self.lease_writer(
                    task,
                    identity,
                    gpu,
                    cpus=request["cpuThreads"] + request["dataWorkers"],
                    ram_gb=request["ramGb"],
                    runs_per_gpu=effective["gpuSlots"].get(gpu, 1) if gpu is not None else 1,
                    supervisor=self.identity,
                    locked=True,
                )
            except BaseException as error:
                # Never run without a lease: legacy schedulers could not see the task.
                self._discard(child, lease, locked=True, task=task)
                self._guard("start undo", task, self._unstart, task, "its lease was not written")
                if not isinstance(error, Exception):
                    raise
                self.log(f"cannot publish the lease of {task['id']}:\n" + traceback.format_exc())
                return False, None, f"Cannot publish the task's resource lease: {error}", None
        device = self._device(gpu)
        wrapper = getattr(child, "wrapper_identity", None)
        process = {**identity, "wrapper": wrapper} if wrapper else identity
        fields = {
            "started_at": self.now(),
            "process": process,
            "gpu": gpu,
            "lease": lease,
            "progress": None,
            "resources": {"gpuName": device.get("name"), "gpuUuid": device.get("uuid")},
        }
        try:
            started = self.store.transition(
                task["id"],
                from_states=("starting",),
                to_state="running",
                waiting_reason=None,
                detail={"pid": child.pid, "gpu": gpu},
                unless_owner_held=True,
                **fields,
            )
        except BaseException as error:
            self._discard(child, lease, task=task)
            self._guard("start undo", task, self._unstart, task, "its start was not recorded")
            if not isinstance(error, Exception):
                raise
            self.log(f"cannot record the start of {task['id']}:\n" + traceback.format_exc())
            return False, None, f"Cannot record the task start: {error}", None
        if not started:
            current = self.store.get(task["id"])
            if current is not None and current["state"] == "stopping":
                # Cancelled or paused while starting: record the process so the stop
                # reaches it like any running task.
                self.store.set_fields(task["id"], **fields)
                self._procs[task["id"]] = child
                result["started"].append(task["id"])
                return True, gpu, None, None
            # Held while starting: undo the start.
            self._discard(child, lease, task=task)
            self._guard("start undo", task, self._unstart, task, "its owner was held")
            return (
                False,
                None,
                ON_HOLD if current and current["state"] == "starting" else None,
                None,
            )
        self._procs[task["id"]] = child
        result["started"].append(task["id"])
        self.log(f"started {task['id']} ({task['title']}) pid {child.pid} gpu {gpu}")
        task = self.store.get(task["id"]) or task
        self._started_hook(task, identity, gpu, ctx)
        return True, gpu, None, None

    def _fail_starting(self, task: dict, message: str, result: dict) -> None:
        if self.store.transition(
            task["id"],
            from_states=("starting",),
            to_state="failed",
            exit={"reason": "error", "error": message},
            error=message,
            waiting_reason=None,
        ):
            result["finished"].append(task["id"])
            self.log(f"failed {task['id']} before start: {message}")

    def _discard(self, child, lease: str | None, *, locked=False, task=None) -> None:
        """Kill a just-spawned child and its session, reap it and withdraw its lease."""
        try:
            # Its own session: the group id is its pid, reserved until the child is reaped.
            os.killpg(child.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            child.wait(timeout=5)
        except Exception:
            self.log(f"cannot reap discarded process {child.pid}:\n" + traceback.format_exc())
        if lease:
            try:
                self.lease_remover(lease, locked=locked)
            except Exception:  # removed on a later tick once its process is gone
                self.log(f"cannot remove lease {lease}:\n" + traceback.format_exc())
        if task is not None:
            self._drop_journal(task)  # never adopted or concluded by a later runner

    # -- telemetry -----------------------------------------------------------------------------

    def _telemetry(self, ctx: RunnerContext, clock: float) -> None:
        if self._sample_at is not None and clock - self._sample_at < self.sample_interval:
            return
        self._sample_at = clock
        running = [
            task
            for task in self.store.list(states=ACTIVE, limit=None)
            if task["process"] and not self._pending_exit(task)
        ]
        on_gpu = Counter(
            task["gpu"]
            for task in running
            if task["gpu"] is not None and task["request"]["lane"] == "gpu"
        )
        for task in running:
            self._guard("telemetry", task, self._sample, task, ctx, clock, on_gpu)

    def _sample(self, task: dict, ctx: RunnerContext, clock: float, on_gpu: Counter) -> None:
        identity = task["process"]
        now = self.now()
        resources = dict(task["resources"] or {})
        memory = procs.private_memory_gb(identity)
        if memory is not None:
            resources["privateRamGb"] = round(memory, 4)
            resources["peakPrivateRamGb"] = round(
                max(memory, resources.get("peakPrivateRamGb") or 0.0), 4
            )
        cpu = procs.cpu_seconds(identity)
        previous = self._samples.get(task["id"])
        if cpu is not None:
            if previous and previous[1] is not None and clock > previous[0]:
                resources["cpuCores"] = round(
                    max(0.0, (cpu - previous[1]) / (clock - previous[0])), 3
                )
            elapsed = seconds_between(task["startedAt"], now)
            if elapsed and elapsed > 0:
                resources["meanCpuCores"] = round(cpu / elapsed, 3)
        if task["gpu"] is not None and task["request"]["lane"] == "gpu":
            count = on_gpu[task["gpu"]] + self._foreign_gpu.get(task["gpu"], 0)
            step = clock - previous[0] if previous else 0.0
            area = float(resources.get("concurrencySeconds") or 0.0) + count * step
            observed = float(resources.get("observedSeconds") or 0.0) + step
            resources.update(
                concurrency=count,
                concurrencySeconds=area,
                observedSeconds=observed,
                meanConcurrency=round(area / observed, 3) if observed > 0 else float(count),
            )
        self._samples[task["id"]] = (clock, cpu)
        resources["sampledAt"] = now
        fields = {"resources": resources}
        try:
            progress = self._adapter(task).progress(task, ctx)
        except Exception:  # progress never blocks supervision
            progress = None
        if progress is not None:
            fields["progress"] = progress
        if task["command"].get("progress") and not task["request"].get("service"):
            reference = (progress or task["progress"] or {}).get("updatedAt") or task["startedAt"]
            idle = seconds_between(reference, now)
            reason = (
                f"No progress for {int(idle // 60)} min"
                if idle is not None and idle > ctx.settings["stallMinutes"] * 60
                else None
            )
            current = task["waitingReason"]
            if reason != current and (reason or (current or "").startswith("No progress")):
                fields["waiting_reason"] = reason
        self.store.set_fields(task["id"], **fields)

    # -- startup -------------------------------------------------------------------------------

    def reconcile_startup(self) -> None:
        """Adopt live tasks, classify and maybe resume lost ones, resume pending requeues."""
        ctx = self._context()
        result = {"started": [], "finished": [], "stopped": [], "waiting": {}}
        for task in self.store.list(states=ACTIVE, limit=None):
            if self._stopping():
                return  # the rest stays recorded for the next runner
            if self._pending_exit(task):
                continue
            identity = task["process"]
            if task["state"] in ("starting", "stopping") and not identity:
                self._guard("startup reconcile", task, self._resolve_starting, task, ctx, result)
                continue
            if identity and procs.alive(identity):
                self.log(f"adopted {task['id']} (pid {identity.get('pid')})")
                continue
            recorded = procs.read_exit(self._journal(task)["exit"]) if identity else None
            if recorded is None and identity and procs.leader_alive(identity.get("wrapper")):
                continue  # its wrapper is writing the exit record; the next tick reads it
            # With the wrapper's exit record the exit status is known; without it the
            # process was lost (for example with the machine). A record of a task the host
            # stopped (a shutdown before this boot) reads as lost too (``host_stop``).
            returncode = recorded["returncode"] if recorded else None
            self._guard(
                "startup reconcile",
                task,
                lambda task=task, code=returncode, record=recorded: self._finish(
                    task,
                    self._exit(task, code, lost=code is None, record=record),
                    ctx,
                    result,
                    auto_resume=True,
                ),
            )
        if self._stopping():
            return
        self._retry_bookkeeping(ctx, result, requeues_only=True)
        self._guard("lease cleanup", None, self._lease_housekeeping, ctx, result)
