"""Start, probe and stop the per-user runner. tmux hosts it; the runner lock is the authority."""

import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from histopilot.storage.project_lock import StorageError
from histopilot.taskcenter import paths


def session_name() -> str:
    return f"hp-runner-{os.getuid()}"


def runner_alive(lock_path: Path | None = None) -> bool:
    """Probe the runner lock without blocking; a held lock means a live runner."""
    import fcntl

    path = Path(lock_path) if lock_path is not None else paths.runner_lock_path()
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError:
        return False  # No lock file: no runner has ever started here.
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    except OSError:
        return False
    else:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return False
    finally:
        os.close(descriptor)


def _repository_root() -> Path:
    import histopilot

    return Path(histopilot.__file__).resolve().parents[1]


# Variables that describe one task process, not the machine; a runner started from inside a
# task (for example a bulk submission) must not hand them to every task it spawns.
TASK_ONLY_ENV = frozenset(
    {
        "HISTOPILOT_TASK_ID",
        "HISTOPILOT_TASK_ATTEMPT",
        "HISTOPILOT_TASK_GPU",
        "HISTOPILOT_TASK_MANAGED",
    }
)


def _forwarded_environment(state: Path) -> list[str]:
    """``NAME=value`` pairs the runner needs from the process that starts it.

    A tmux session created on an existing tmux server gets the server's environment, not
    this process's, so settings such as HISTOPILOT_TRAINING_PYTHON are passed explicitly.
    """
    values = {
        name: value
        for name, value in os.environ.items()
        if name.startswith("HISTOPILOT_")
        and name not in TASK_ONLY_ENV
        and name != paths.AUTOSTART_ENV
    }
    for name in ("PATH", "LD_LIBRARY_PATH"):
        if os.environ.get(name):
            values[name] = os.environ[name]
    values.update(
        {
            "PYTHONUNBUFFERED": "1",
            paths.STATE_ENV: str(state),
            "TMPDIR": tempfile.gettempdir(),
        }
    )
    ordered = ["PYTHONUNBUFFERED", paths.STATE_ENV, "TMPDIR"]
    names = ordered + sorted(name for name in values if name not in ordered)
    return [f"{name}={values[name]}" for name in names]


def ensure_runner(*, python: str | None = None) -> dict:
    """Start the runner in tmux when it is not running. Never raises for tmux problems."""
    if not paths.autostart_enabled():
        return {"started": False, "reason": "disabled"}
    try:
        state = paths.state_dir()
        if runner_alive(state / "runner.lock"):
            return {"started": False, "alive": True}
        tmux = shutil.which("tmux")
        if tmux is None:
            return {"started": False, "reason": "tmux unavailable"}
        command = "cd " + shlex.quote(str(_repository_root())) + " && "
        command += shlex.join(
            [
                "env",
                *_forwarded_environment(state),
                python or sys.executable,
                "-u",
                "-m",
                "histopilot.cli",
                "runner",
                "run",
            ]
        )
        command += " >> " + shlex.quote(str(state / "runner.log")) + " 2>&1"
        session = session_name()
        for attempt in range(2):
            try:
                subprocess.run(
                    [tmux, "new-session", "-d", "-s", session, command],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=True,
                )
                return {"started": True, "session": session}
            except subprocess.CalledProcessError as error:
                detail = (error.stderr or error.stdout or "").strip()
                if "duplicate session" not in detail:
                    raise
                # The session belongs to a runner that is starting, or to one that already
                # released its lock and is exiting; start again once the latter is gone.
                outcome = _await_session(tmux, session, state / "runner.lock")
                if outcome == "alive":
                    return {"started": False, "alive": True}
                if outcome == "busy" or attempt:
                    return {"started": True, "note": "already starting"}
        return {"started": True, "note": "already starting"}
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "").strip()
        return {"started": False, "reason": f"tmux failed: {detail or error.returncode}"}
    except FileNotFoundError:
        return {"started": False, "reason": "tmux unavailable"}
    except (OSError, subprocess.SubprocessError) as error:
        return {"started": False, "reason": f"tmux failed: {error}"}
    except StorageError as error:
        return {"started": False, "reason": str(error)}


def _session_exists(tmux: str, session: str) -> bool:
    result = subprocess.run(
        [tmux, "has-session", "-t", "=" + session], capture_output=True, timeout=5, check=False
    )
    return result.returncode == 0


SESSION_WAIT_SECONDS = 10.0


def _await_session(tmux: str, session: str, lock: Path) -> str:
    """Wait for a runner session to take its lock ("alive") or to end ("gone")."""
    deadline = time.monotonic() + SESSION_WAIT_SECONDS
    while True:
        if runner_alive(lock):
            return "alive"
        if not _session_exists(tmux, session):
            return "gone"
        if time.monotonic() >= deadline:
            return "busy"
        time.sleep(0.2)


def stop_runner(*, wait: float = 5.0) -> dict:
    """SIGTERM the recorded runner after verifying its identity. Tasks keep running."""
    from histopilot.taskcenter.store import TaskStore
    from histopilot.workers.training_process import confirmed_process_alive

    store = TaskStore()
    row = store.runner()
    if not row or not row.get("pid"):
        return {"stopped": False, "reason": "no runner recorded"}
    lock = store.path.parent / "runner.lock"
    if not runner_alive(lock):
        return {"stopped": False, "reason": "not running"}  # a live runner holds its lock
    identity = {"pid": row["pid"], "startTicks": row["startTicks"], "bootId": row["bootId"]}
    if identity["pid"] == os.getpid():
        return {"stopped": False, "reason": "the runner is this process"}
    try:
        if not confirmed_process_alive(identity):
            return {"stopped": False, "reason": "not running"}
        os.kill(identity["pid"], signal.SIGTERM)
    except ProcessLookupError:
        return {"stopped": False, "reason": "not running"}
    except (StorageError, PermissionError) as error:
        return {"stopped": False, "reason": str(error)}
    # The runner releases its lock before its process ends; wait for both.
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if not runner_alive(lock) and not _process_alive(identity):
            break
        time.sleep(0.1)
    exited = not runner_alive(lock) and not _process_alive(identity)
    return {"stopped": True, "pid": identity["pid"], "exited": exited}


def restart_runner(*, wait: float = 20.0) -> dict:
    """Stop the recorded runner, wait for it to exit, then start a new one.

    Tasks keep running and the new runner adopts them. When the old runner is still
    finishing a step after ``wait`` seconds, the result is ``{"started": False, "pending":
    True, ...}``; the old runner then exits on its own and must be started again.
    """
    if not paths.autostart_enabled():
        return {"started": False, "reason": "disabled"}
    stopped = stop_runner(wait=wait)
    if stopped.get("stopped") and not stopped.get("exited"):
        return {
            "started": False,
            "pending": True,
            "stopped": stopped,
            "note": "The runner is finishing its current step; start it again once it exits.",
        }
    return {**ensure_runner(), "stopped": stopped}


def await_runner_exit(timeout: float, *, poll: float = 0.5) -> bool:
    """Wait until no runner holds the lock and the runner's tmux session has ended."""
    lock = paths.runner_lock_path()
    tmux = shutil.which("tmux")
    deadline = time.monotonic() + timeout
    while True:
        try:
            session = tmux is not None and _session_exists(tmux, session_name())
        except (OSError, subprocess.SubprocessError):
            session = False
        if not runner_alive(lock) and not session:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(poll)


def _process_alive(identity: dict) -> bool:
    from histopilot.workers.training_process import confirmed_process_alive

    try:
        return confirmed_process_alive(identity)
    except (OSError, StorageError):
        return False
