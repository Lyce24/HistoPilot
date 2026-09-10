"""Advisory process lock for one control service per local workspace."""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def service_lock(workspace: Path) -> Iterator[None]:
    workspace.mkdir(parents=True, exist_ok=True)
    path = workspace / "service.lock"
    # Keep the inode in place; unlinking a held lock could allow duplicate services.
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt

            stream.seek(0, 2)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(f"A control service already holds {path}") from exc
        else:
            import fcntl

            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError(f"A control service already holds {path}") from exc
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
