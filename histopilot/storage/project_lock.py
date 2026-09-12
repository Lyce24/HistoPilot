"""Filesystem durability primitives shared by project setup and scientific state.

Locks cover a single operation, not the lifetime of a web-service registry. Keep
the lock inode in place so separate services always coordinate on the same file.
"""

import os
import stat
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

LOCK_FILE = ".histopilot-write.lock"


class StorageError(ValueError):
    def __init__(self, message: str, code: str, status_code: int = 409):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


def _reject_symlink_components(path: Path) -> None:
    for component in (path, *path.parents):
        if component.is_symlink():
            raise StorageError(
                "Managed project storage cannot contain symbolic links.", "STORAGE_UNSAFE_PATH", 403
            )


def fsync_directory(path: Path) -> None:
    """Persist directory-entry changes, failing if this filesystem cannot do so."""
    _reject_symlink_components(path)
    if os.name != "posix":
        raise StorageError(
            "Scientific storage currently requires POSIX directory synchronization support.",
            "STORAGE_UNSUPPORTED",
        )
    descriptor = None
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        os.fsync(descriptor)
    except OSError as error:
        raise StorageError(
            "The experiment filesystem cannot synchronize directory changes.",
            "STORAGE_SYNC_FAILED",
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def ensure_managed_directory(path: Path) -> None:
    """Create a managed directory durably without following symbolic links."""
    _reject_symlink_components(path)
    try:
        if path.exists():
            if not path.is_dir():
                raise StorageError(
                    "A managed storage directory is occupied by another file type.",
                    "STORAGE_UNSAFE_PATH",
                    403,
                )
            return
        if not path.parent.is_dir():
            ensure_managed_directory(path.parent)
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            # Independent workers may initialize a shared registry concurrently.
            # Accept only a real directory, applying the same path checks again.
            _reject_symlink_components(path)
            if not path.is_dir():
                raise StorageError(
                    "A managed storage directory is occupied by another file type.",
                    "STORAGE_UNSAFE_PATH",
                    403,
                ) from None
        fsync_directory(path.parent)
    except OSError as error:
        raise StorageError(
            "A project storage directory cannot be created.", "STORAGE_WRITE_FAILED", 403
        ) from error


@contextmanager
def writer_lock(folder: Path, *, timeout: float = 0) -> Iterator[None]:
    """Acquire a shared process lock; reads may wait briefly, mutations fail fast."""
    _reject_symlink_components(folder)
    if os.name != "posix":
        raise StorageError(
            "Scientific storage currently requires POSIX project locking support.",
            "STORAGE_UNSUPPORTED",
        )
    import fcntl

    path = folder / LOCK_FILE
    descriptor = None
    locked = False
    try:
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise StorageError(
                "The project writer lock must be a regular file without aliases.",
                "STORAGE_UNSAFE_PATH",
                403,
            )
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as error:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise StorageError(
                        "Another operation is writing this experiment. Retry after it finishes.",
                        "PROJECT_BUSY",
                    ) from error
                time.sleep(min(0.01, remaining))
        locked = True
        current = os.stat(path, follow_symlinks=False)
        if (metadata.st_dev, metadata.st_ino) != (current.st_dev, current.st_ino):
            raise StorageError("The project writer lock changed.", "STORAGE_UNSAFE_PATH", 403)
        os.fsync(descriptor)
        fsync_directory(folder)
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
            descriptor = None
        raise StorageError(
            "The experiment filesystem cannot acquire a durable project writer lock.",
            "STORAGE_LOCK_FAILED",
            403,
        ) from error
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
            descriptor = None
        raise
    try:
        yield
    finally:
        if descriptor is not None:
            try:
                if locked:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
