"""Durable worker metadata, cross-project output locks and their registry housekeeping."""

import fcntl
import hashlib
import json
import os
import re
import stat
import tempfile
import time
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
    """A process-held lock survives service restarts and cannot be stolen by retries.

    The housekeeping sweep may unlink an idle lock file while holding its lock; a holder
    that locked such an unlinked file owns nothing, so it checks the path afterwards.
    """
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
        try:
            current = os.stat(path, follow_symlinks=False)
        except FileNotFoundError:
            current = None
        if current is None or (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino):
            raise StorageError("A worker is already using this output.", "OUTPUT_BUSY")
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
    """The identity in ``folder/process.json`` while that exact process still runs.

    Shared by extraction, packing and archive records (boot id + start ticks, so a
    reused PID or a reboot never reads as a live worker).
    """
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


KEYED = re.compile(r"[0-9a-f]{64}")
SWEEP_MAX_AGE_HOURS = 24.0
SWEEP_INTERVAL_SECONDS = 6 * 3600.0
SWEEP_BUDGET_SECONDS = 5.0
SWEEP_STAMP = ".last-sweep"


def _older_than(path: Path, seconds: float, now: float) -> bool:
    try:
        return now - path.stat(follow_symlinks=False).st_mtime >= seconds
    except OSError:
        return False


def _claim_finished(claim: dict, now: float, max_age: float, claim_path: Path) -> bool:
    """Whether an output claim of a pre-Task Center packing job no longer protects it.

    Older checkouts wrote a claim per tmux job. A claim protects its job while the job's
    worker process lives, or while the job has neither a receipt nor a cancel marker and
    the claim is younger than ``max_age`` (a launch whose worker has not recorded itself
    yet). Published packs need no claim: their folders are non-empty and new outputs
    inside or around them are refused.
    """
    job_path = claim.get("jobPath")
    if not isinstance(job_path, str) or not job_path.startswith("/"):
        return _older_than(claim_path, max_age, now)
    folder = Path(job_path).parent
    if live_process(folder):
        return False
    if not (folder / "job.json").exists():
        return True
    if (folder / "result.json").exists() or (folder / "cancelled").exists():
        return True
    return _older_than(claim_path, max_age, now)


def _unlock_idle(path: Path) -> bool:
    """Unlink an idle lock file while holding its lock (holders re-check the inode)."""
    try:
        descriptor = os.open(path, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return False
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            return False
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        current = os.stat(path, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino):
            return False
        path.unlink()
        return True
    except OSError:
        return False
    finally:
        os.close(descriptor)


def sweep_registry(
    folder: Path | None = None,
    *,
    max_age_hours: float = SWEEP_MAX_AGE_HOURS,
    budget_seconds: float = SWEEP_BUDGET_SECONDS,
    batch: int = 200,
    clock=time.monotonic,
    now=time.time,
) -> dict:
    """Remove finished or dead output claims and idle, old lock files from the registry.

    Packing jobs are Task Center tasks and write no claims; this only drains what jobs
    of older checkouts and old test runs left behind. Malformed claims are tolerated (removed
    once old) instead of blocking packing machine-wide. Claims are checked in small
    batches under the registry lock, so submissions are never held up for long. Stops
    after ``budget_seconds``; the next sweep continues.
    """
    folder = folder or registry_directory()
    stats = {"claimsRemoved": 0, "claimsKept": 0, "locksRemoved": 0, "complete": True}
    started, current = clock(), now()
    max_age = float(max_age_hours) * 3600.0
    claims = [
        path
        for path in folder.glob("*.claim.json")
        if KEYED.fullmatch(path.name.removesuffix(".claim.json"))
    ]
    for offset in range(0, len(claims), batch):
        if clock() - started > budget_seconds:
            stats["complete"] = False
            return stats
        with writer_lock(folder, timeout=5):
            for path in claims[offset : offset + batch]:
                try:
                    claim = json.loads(path.read_bytes()[:65536])
                    if not isinstance(claim, dict):
                        raise ValueError("claim is not an object")
                except FileNotFoundError:
                    continue
                except (OSError, ValueError, UnicodeError):
                    claim = None
                finished = (
                    _older_than(path, max_age, current)
                    if claim is None
                    else _claim_finished(claim, current, max_age, path)
                )
                if finished:
                    path.unlink(missing_ok=True)
                    stats["claimsRemoved"] += 1
                else:
                    stats["claimsKept"] += 1
    for path in folder.glob("*.lock"):
        # Only output locks (<sha256>.lock); never the registry's own writer lock.
        if not KEYED.fullmatch(path.name.removesuffix(".lock")):
            continue
        if clock() - started > budget_seconds:
            stats["complete"] = False
            break
        if _older_than(path, max_age, current) and _unlock_idle(path):
            stats["locksRemoved"] += 1
    return stats


def maybe_sweep_registry(folder: Path | None = None, *, interval=SWEEP_INTERVAL_SECONDS):
    """Run ``sweep_registry`` at most once per ``interval`` per machine; never raises.

    An unfinished sweep (its time budget ran out) resumes a minute later.
    """
    try:
        folder = folder or registry_directory()
        stamp = folder / SWEEP_STAMP
        if stamp.exists() and not _older_than(stamp, interval, time.time()):
            return None
        stamp.touch()
        stats = sweep_registry(folder)
        if not stats["complete"]:
            soon = time.time() - interval + 60
            os.utime(stamp, (soon, soon))
        return stats
    except (OSError, StorageError):
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
