"""Machine-level Task Center locations and environment switches, resolved at call time."""

import os
from pathlib import Path

from histopilot.storage.project_lock import StorageError, ensure_managed_directory

STATE_ENV = "HISTOPILOT_STATE_DIR"
AUTOSTART_ENV = "HISTOPILOT_TASK_CENTER_AUTOSTART"  # "0" disables ensure_runner()


def state_dir() -> Path:
    """Private per-user state directory shared by every workspace on this machine."""
    configured = os.environ.get(STATE_ENV)
    if configured:
        path = Path(configured)
    else:
        xdg = os.environ.get("XDG_STATE_HOME")
        # The XDG specification says relative values are invalid and must be ignored.
        path = (
            Path(xdg) / "histopilot"
            if xdg and Path(xdg).is_absolute()
            else Path("~/.local/state/histopilot")
        )
    path = path.expanduser().resolve()
    ensure_managed_directory(path)
    info = path.stat()
    if info.st_uid != os.getuid() or info.st_mode & 0o022:
        raise StorageError(
            f"The Task Center state directory {path} must be owned by this user and not writable "
            "by other users.",
            "TASK_CENTER_STATE_UNSAFE",
            403,
        )
    return path


def store_path() -> Path:
    return state_dir() / "task-center.sqlite"


def runner_lock_path() -> Path:
    return state_dir() / "runner.lock"


def autostart_enabled() -> bool:
    return os.environ.get(AUTOSTART_ENV, "1") != "0"


def checkout_root() -> str:
    """The checkout (or installation) this process runs HistoPilot from."""
    return str(Path(__file__).resolve().parents[2])
