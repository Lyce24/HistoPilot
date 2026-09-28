"""Standalone stdlib supervisor, compatible with TRIDENT's Python 3.10 environment.

Invoke this file by absolute path with a private JSON plan, never with `-m`:
TRIDENT's interpreter need not have HistoPilot or Pydantic installed. The parent
service owns plan validation, process persistence and artifact coverage checking.

Two modes share this file:

- Task Center (``"managed": true`` in the plan, or ``HISTOPILOT_TASK_MANAGED=1``): how
  HistoPilot runs extractions. The runner holds the task's resource lease; this file keeps
  TRIDENT in the task's process group so a cancel or a runner restart reaches it, runs no
  validation (that is a separate CPU task) and writes progress JSON for stall detection.
- Standalone (a plan without ``managed``, run by hand): TRIDENT runs in its own session,
  then the plan's artifact validation command. It takes no resource lease.
"""

import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

PROGRESS_SECONDS = 15.0
PROGRESS_REFRESH_SECONDS = 60.0
STOP_GRACE_SECONDS = 10.0
LOG_TAIL_BYTES = 64 * 1024


def _now():
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017 - worker supports Python 3.10


def _write_result(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _process_identity(pid):
    """Record Linux process identity so PID reuse never implies a live worker."""
    identity = {"pid": pid, "startTicks": None, "bootId": None}
    try:
        # comm (field 2) may contain spaces or parentheses. starttime is field 22.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        identity["startTicks"] = int(fields[19])
        identity["bootId"] = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except (OSError, ValueError, IndexError):
        # A very short command may already have exited; its completion record
        # follows immediately. Unavailable identity must not imply a live PID.
        pass
    return identity


def _worker_environment(python_path, *, cwd=None):
    """Resolve native SDPC libraries in the selected worker's Python environment.

    Looking up the top-level package spec avoids importing OpenSDPC before the
    dynamic linker has its search path. This also supports editable installs,
    without borrowing packages from a different project or Python environment.
    """
    env = os.environ.copy()
    # DataLoader processes already supply parallelism. Avoid a native thread
    # pool per worker unless the operator explicitly configured one.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env.setdefault(name, "1")
    if not sys.platform.startswith("linux"):
        return env
    probe = (
        "import importlib.util, json; "
        "spec = importlib.util.find_spec('opensdpc'); "
        "print(json.dumps(list(spec.submodule_search_locations or []) if spec else []))"
    )
    try:
        result = subprocess.run(
            [python_path, "-c", probe],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
            env=env,
            cwd=cwd,
        )
        package_roots = json.loads(result.stdout)
        if not isinstance(package_roots, list) or any(
            not isinstance(root, str) for root in package_roots
        ):
            return env
    except (OSError, ValueError, subprocess.SubprocessError):
        # OpenSDPC is optional for other slide formats. The actual TRIDENT
        # command remains responsible for reporting unavailable dependencies.
        return env
    libraries = []
    for root in package_roots:
        for relative in ("LINUX", "LINUX/ffmpeg"):
            directory = Path(root) / relative
            if directory.is_dir() and str(directory) not in libraries:
                libraries.append(str(directory))
    if libraries:
        previous = env.get("LD_LIBRARY_PATH")
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [*libraries, *([previous] if previous else [])]
        )
    return env


def _worker_command(command):
    """Bootstrap recognized batch scripts; retain historical fixture commands."""
    script_index = 2 if len(command) > 1 and command[1] == "-u" else 1
    if len(command) <= script_index:
        return command
    script = Path(command[script_index])
    if script.name != "run_batch_of_slides.py" or not script.is_file():
        return command
    return [
        command[0], "-u", str(Path(__file__).with_name("bootstrap.py")),
        *command[script_index:],
    ]


# -- dead TRIDENT locks ---------------------------------------------------------------------

# A lock older than its live PID's process by more than this was left by an earlier owner
# of that PID. Anything closer may be a clock step (below), so the live writer keeps it.
PID_REUSE_MARGIN_SECONDS = 30 * 86400.0


def _boot_time():
    try:
        for line in Path("/proc/stat").read_text().splitlines():
            if line.startswith("btime "):
                return float(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


def _process_start(pid):
    """Epoch seconds when ``pid`` started; False when no such process runs; None if unknown."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return False
    except (OSError, IndexError):
        return None
    try:
        if fields[0] in ("Z", "X"):
            return False
        ticks = int(fields[19])
    except (ValueError, IndexError):
        return None
    boot = _boot_time()
    if boot is None:
        return None
    return boot + ticks / os.sysconf("SC_CLK_TCK")


def _lock_owner_dead(lock, *, now, host, max_age_seconds):
    """Whether the process that wrote a TRIDENT lock is certainly gone.

    TRIDENT locks record ``{"pid", "hostname", "created_at"}``. On this host the owner is
    dead when the PID no longer runs. A live PID keeps its lock: ``btime`` in /proc/stat,
    and every process start derived from it, moves forward when the clock is stepped
    (WSL2 does after the host slept), so timing cannot tell a live writer that started
    before the step from a reused PID; only a lock ``PID_REUSE_MARGIN_SECONDS`` older than
    the process marks reuse. A lock without an owner this host can check (empty,
    unreadable or from another host) is dead once older than ``max_age_seconds``, as in
    TRIDENT's own ``clear_dead_locks``.
    """
    pid = created = owner_host = None
    try:
        with open(lock, encoding="utf-8") as stream:
            raw = stream.read(4096).strip()
        if raw:
            data = json.loads(raw)
            if isinstance(data, dict):
                pid, owner_host, created = data.get("pid"), data.get("hostname"), data.get("created_at")
    except (OSError, ValueError, UnicodeError):
        pass
    try:
        created = float(created) if created is not None else None
    except (TypeError, ValueError):
        created = None
    if type(pid) is int and pid > 0 and owner_host == host:
        started = _process_start(pid)
        if started is False:
            return True
        return (
            started is not None
            and created is not None
            and started > created + PID_REUSE_MARGIN_SECONDS
        )
    try:
        reference = created if created is not None else os.path.getmtime(lock)
    except OSError:
        return False
    return now - reference >= max_age_seconds


def clear_dead_locks(root, *, max_age_hours=24.0, log=None):
    """Remove TRIDENT ``.lock`` files under ``root`` whose writer is gone.

    TRIDENT skips every slide whose output has a ``.lock`` next to it, and treats an
    output without one as finished. A lock left by a dead writer therefore hides the slide
    forever, and its output (if any) may be truncated: such an output is moved aside to
    ``<name>.stale-<epoch>`` so TRIDENT produces it again. Nothing is deleted except the
    lock. Callers guarantee that no other extraction writes to ``root`` meanwhile.
    """
    stats = {"scanned": 0, "removed": 0, "movedAside": 0, "kept": 0}
    if not root or not os.path.isdir(root):
        return stats
    now, host = time.time(), socket.gethostname()
    max_age_seconds = float(max_age_hours) * 3600.0
    for directory, _folders, names in os.walk(root):
        for name in names:
            if not name.endswith(".lock"):
                continue
            lock = os.path.join(directory, name)
            stats["scanned"] += 1
            try:
                if os.path.islink(lock) or not _lock_owner_dead(
                    lock, now=now, host=host, max_age_seconds=max_age_seconds
                ):
                    stats["kept"] += 1
                    continue
                target = lock[: -len(".lock")]
                if os.path.lexists(target):
                    aside = f"{target}.stale-{int(now)}"
                    os.replace(target, aside)
                    stats["movedAside"] += 1
                    if log:
                        log(f"moved an output of a dead TRIDENT writer aside: {aside}")
                os.unlink(lock)
                stats["removed"] += 1
            except OSError as error:
                stats["kept"] += 1
                if log:
                    log(f"cannot clear TRIDENT lock {lock}: {error}")
    return stats


# -- progress -------------------------------------------------------------------------------


def _progress_builder():
    """``progress.build_progress`` loaded by path: it is stdlib-only, like this runner."""
    try:
        spec = importlib.util.spec_from_file_location(
            "_histopilot_trident_progress", Path(__file__).with_name("progress.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.build_progress
    except Exception:  # progress is optional telemetry
        return None


def _log_tail(path):
    try:
        with open(path, "rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            stream.seek(max(0, size - LOG_TAIL_BYTES))
            text = stream.read(LOG_TAIL_BYTES).decode(errors="replace")
            return text, os.fstat(stream.fileno()).st_mtime
    except OSError:
        return "", None


def _read_peak(path):
    if not path:
        return None
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        peak = value.get("cudaPeakReservedBytes")
        return peak if type(peak) is int and peak > 0 else None
    except (OSError, ValueError, AttributeError):
        return None


class _Progress:
    """Writes the Task Center progress file from the job log.

    The file is rewritten when the derived progress changes, or at least once a minute
    while the log keeps growing. A silent (hung) TRIDENT stops refreshing it, so the Task
    Center's stall detection sees no progress.
    """

    def __init__(self, plan):
        self.path = Path(plan["progressPath"]) if plan.get("progressPath") else None
        self.log_path = plan.get("logPath")
        self.peak_path = plan.get("peakPath")
        self.job = {}
        if plan.get("jobPath"):
            try:
                self.job = json.loads(Path(plan["jobPath"]).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self.job = {}
        self.build = _progress_builder() if self.path else None
        self.checked = 0.0
        self.written = 0.0
        self.log_mtime = None
        self.last = None

    def update(self, *, force=False):
        if self.path is None or self.build is None:
            return
        clock = time.monotonic()
        if not force and clock - self.checked < PROGRESS_SECONDS:
            return
        self.checked = clock
        text, mtime = _log_tail(self.log_path)
        try:
            job = {
                **self.job,
                "state": "running",
                "result": None,
                "_logUpdatedAt": datetime.fromtimestamp(mtime, timezone.utc).isoformat()  # noqa: UP017
                if mtime
                else None,
            }
            value = self.build(job, text)
        except Exception:  # never stop extraction for telemetry
            return
        comparable = {
            key: value.get(key)
            for key in ("stage", "label", "completed", "total", "currentSlide", "scope")
        }
        grew = mtime is not None and mtime != self.log_mtime
        if not force and comparable == self.last and not (
            grew and clock - self.written >= PROGRESS_REFRESH_SECONDS
        ):
            return
        value.update(phase=value.get("label"), message=value.get("detail"))
        if value.get("unit") == "slides" and value.get("total"):
            value.update(completedSlides=value.get("completed") or 0, totalSlides=value["total"])
        peak = _read_peak(self.peak_path)
        if peak:
            value["cudaPeakReservedBytes"] = peak
        try:
            _write_result(self.path, value)
        except (OSError, ValueError):
            return
        self.last, self.written, self.log_mtime = comparable, clock, mtime


# -- runner ---------------------------------------------------------------------------------


def _signal_child(process, signum, *, own_group):
    """TRIDENT has its own session when run standalone; under the Task Center it shares the
    task's process group, whose members the Task Center drains after this runner exits."""
    try:
        if own_group:
            os.kill(process.pid, signum)
        else:
            os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass


def _stop_child(process, *, own_group):
    _signal_child(process, signal.SIGTERM, own_group=own_group)
    try:
        process.wait(timeout=STOP_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        _signal_child(process, signal.SIGKILL, own_group=own_group)
        process.wait()
    if not own_group:
        # Descendants can outlive a parent that accepted SIGTERM.
        _signal_child(process, signal.SIGKILL, own_group=False)


def run_plan(path):
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    managed = bool(plan.get("managed")) or os.environ.get("HISTOPILOT_TASK_MANAGED") == "1"
    result_path = Path(plan["resultPath"])
    cancel_path = Path(plan["cancelPath"]) if plan.get("cancelPath") else None
    process = None
    received = []
    started = _now()

    def cancel_requested():
        return bool(cancel_path and cancel_path.exists())

    def stopping():
        return bool(received) or cancel_requested()

    def stop(signum, _frame):
        received.append(signum)
        if process is not None and process.poll() is None:
            _signal_child(process, signal.SIGTERM, own_group=managed)

    def stopped_state():
        # Only a requested cancel is a cancellation. A signal from the host (a closed
        # terminal, a shutdown) interrupts the run, which can be resumed.
        return "cancelled" if cancel_requested() else "interrupted"

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, stop)
    result = {"state": "failed", "exitCode": None, "startedAt": started}
    if managed:
        result.update(
            managed=True,
            taskId=os.environ.get("HISTOPILOT_TASK_ID"),
            taskAttempt=int(os.environ["HISTOPILOT_TASK_ATTEMPT"])
            if os.environ.get("HISTOPILOT_TASK_ATTEMPT", "").isdecimal()
            else None,
        )
    progress = _Progress(plan) if managed else None
    try:
        commands = [("TRIDENT", plan["command"])]
        if plan.get("validationCommand") is not None and not managed:
            commands.append(("Artifact validation", plan["validationCommand"]))
        for phase, command in commands:
            if (
                not isinstance(command, list)
                or not command
                or any(not isinstance(arg, str) or "\x00" in arg for arg in command)
            ):
                raise ValueError(f"{phase} command must be a nonempty argv list")
        if cancel_requested():
            result["state"] = "cancelled"
            return 0
        with Path(plan["logPath"]).open("a", encoding="utf-8", buffering=1) as log:
            if plan.get("processPath"):
                _write_result(Path(plan["processPath"]), _process_identity(os.getpid()))
            locks = plan.get("clearDeadLocks")
            if isinstance(locks, dict) and locks.get("root") and not stopping():
                # Every attempt owns its output exclusively: locks of dead writers would
                # make TRIDENT skip their slides and fail validation.
                stats = clear_dead_locks(
                    locks["root"],
                    max_age_hours=locks.get("maxAgeHours") or 24.0,
                    log=lambda message: log.write(f"[{_now()}] {message}\n"),
                )
                result["deadLocks"] = stats
                log.write(
                    f"[{_now()}] Dead TRIDENT locks: removed={stats['removed']} "
                    f"movedAside={stats['movedAside']} kept={stats['kept']} "
                    f"scanned={stats['scanned']}\n"
                )
            for phase, command in commands:
                worker_env = (
                    _worker_environment(command[0], cwd=plan.get("cwd"))
                    if phase == "TRIDENT"
                    else None
                )
                if phase == "TRIDENT" and plan.get("peakPath") and worker_env is not None:
                    worker_env["HISTOPILOT_TRIDENT_PEAK_PATH"] = str(plan["peakPath"])
                # Cancellation during discovery or between stages prevents
                # the next worker from starting.
                if stopping():
                    result["state"] = stopped_state()
                    break
                log.write(f"[{_now()}] Starting {phase} worker\n")
                effective_command = _worker_command(command) if phase == "TRIDENT" else command
                if effective_command != command:
                    log.write(f"[{_now()}] HistoPilot TRIDENT performance bootstrap enabled\n")
                process = subprocess.Popen(
                    effective_command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    shell=False,
                    start_new_session=not managed,
                    cwd=plan.get("cwd"),
                    env=worker_env,
                )
                if plan.get("processPath"):
                    _write_result(
                        Path(plan["processPath"]),
                        {**_process_identity(process.pid), "phase": phase},
                    )
                # Each phase has its own process group and durable identity.
                while process.poll() is None:
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        pass
                    if progress is not None:
                        progress.update()
                    if stopping():
                        _stop_child(process, own_group=managed)
                        break
                if progress is not None:
                    progress.update(force=True)  # where TRIDENT ended, before its exit line
                result["exitCode"] = process.returncode
                if phase == "TRIDENT":
                    result["tridentExitCode"] = process.returncode
                result["state"] = (
                    stopped_state()
                    if stopping()
                    else "succeeded"
                    if process.returncode == 0
                    else "failed"
                )
                if received:
                    result["signal"] = received[0]
                if result["state"] == "failed":
                    result["error"] = (
                        f"{phase} exited with status {process.returncode}; see worker log."
                    )
                elif result["state"] == "interrupted":
                    result["error"] = (
                        f"{phase} was stopped by signal {received[0] if received else '?'} "
                        "without a cancel request; resume to continue."
                    )
                log.write(f"[{_now()}] {phase} {result['state']} (exit {process.returncode})\n")
                if result["state"] != "succeeded":
                    break
    except Exception as error:
        result["error"] = str(error)
        if stopping():
            result["state"] = stopped_state()
        traceback.print_exc()
        if process is not None and process.poll() is None:
            _stop_child(process, own_group=managed)
    finally:
        peak = _read_peak(plan.get("peakPath"))
        if peak:
            result["cudaPeakReservedBytes"] = peak
        result["cancelRequested"] = cancel_requested()
        result["finishedAt"] = _now()
        _write_result(result_path, result)
    return 0 if result["state"] in {"succeeded", "cancelled"} else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: runner.py PLAN.json")
    raise SystemExit(run_plan(sys.argv[1]))
