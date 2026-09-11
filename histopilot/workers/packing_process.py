"""Durable CPU worker execution and cross-project output ownership."""

import fcntl
import hashlib
import json
import os
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

from histopilot.storage.project_lock import (
    StorageError,
    _reject_symlink_components,
    ensure_managed_directory,
    fsync_directory,
    writer_lock,
)


def registry_directory() -> Path:
    path = Path(tempfile.gettempdir()) / f"histopilot-packing-{os.getuid()}"
    ensure_managed_directory(path)
    info = path.stat()
    if info.st_uid != os.getuid() or info.st_mode & 0o022:
        raise StorageError("Packing lock storage is unsafe.", "PACKING_UNSAFE_LOCK", 403)
    return path


def output_key(output: str | Path) -> str:
    return hashlib.sha256(str(output).encode()).hexdigest()


@contextmanager
def registry_lock():
    folder = registry_directory()
    with writer_lock(folder, timeout=5):
        yield folder


@contextmanager
def output_lock(output: str | Path):
    """A process-held lock survives service restarts and cannot be stolen by retries."""
    path = registry_directory() / f"{output_key(output)}.lock"
    _reject_symlink_components(path)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.getuid():
            raise StorageError("Packing output lock is unsafe.", "PACKING_UNSAFE_LOCK", 403)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise StorageError("A worker is already using this output.", "OUTPUT_BUSY") from error
        yield
    finally:
        os.close(descriptor)


def process_metadata() -> dict:
    pid = os.getpid()
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return {
        "pid": pid,
        "bootId": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "startTicks": int(fields[19]),
    }


def live_process(folder: Path) -> dict | None:
    path = folder / "process.json"
    if not path.exists():
        return None
    _reject_symlink_components(path)
    try:
        from histopilot.storage.scientific import ScientificStore

        process = json.loads(ScientificStore._read_file(path, 4096))
        pid = process.get("pid")
        if type(pid) is not int or pid <= 1:
            return None
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if (
            process.get("bootId") == boot
            and process.get("startTicks") == int(fields[19])
            and fields[0] != "Z"
        ):
            return process
    except (OSError, ValueError, IndexError, AttributeError):
        pass
    return None


def write_json(path: Path, value: dict) -> None:
    _reject_symlink_components(path)
    content = json.dumps(value, indent=2, allow_nan=False).encode() + b"\n"
    if len(content) > 64 * 1024 * 1024:
        raise StorageError("Packing metadata exceeds its size limit.", "PACKING_LIMIT", 413)
    descriptor, temporary = tempfile.mkstemp(prefix=".packing-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


class TmuxPackingExecutor:
    def available(self) -> bool:
        return shutil.which("tmux") is not None

    def running(self, session: str) -> bool:
        if not self.available():
            return False
        result = subprocess.run(
            ["tmux", "has-session", "-t", f"={session}"], capture_output=True, timeout=10
        )
        if result.returncode and b"Operation not permitted" in result.stderr:
            raise RuntimeError("Cannot inspect tmux sessions: permission denied.")
        return result.returncode == 0

    def launch(self, session: str, runner: Path, plan: Path) -> None:
        subprocess.run(["tmux", "ls"], capture_output=True, timeout=10)
        if self.running(session):
            raise RuntimeError("This packing session already exists.")
        # Execute the script by absolute path; it establishes the checkout import root.
        command = shlex.join([sys.executable, "-u", str(runner), str(plan)])
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", session, command],
            capture_output=True,
            check=True,
            timeout=15,
        )
