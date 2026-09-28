"""Process-wide rules for opening and closing SQLite connections.

SQLite 3.51.0's unix VFS can deadlock two threads of one process: closing the last
connection to a WAL database checkpoints under an EXCLUSIVE lock request, which holds the
file's lock mutex while taking the global VFS mutex, and opening or closing another
connection to that file takes the same mutexes in the opposite order. Once it happens every
later SQLite call in the process blocks. Every connection therefore skips the checkpoint on
close (WAL files are still checkpointed automatically as they grow), and connections this
module opens and closes do so one at a time.
"""

import sqlite3
from threading import RLock

LIFECYCLE_LOCK = RLock()
_NO_CHECKPOINT_ON_CLOSE = getattr(sqlite3, "SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE", None)


def prepare(connection: sqlite3.Connection) -> sqlite3.Connection:
    """Disable the close-time checkpoint; safe for connections opened by any library."""
    if _NO_CHECKPOINT_ON_CLOSE is not None and hasattr(connection, "setconfig"):
        connection.setconfig(_NO_CHECKPOINT_ON_CLOSE, True)
    return connection


def connect(*args, **kwargs) -> sqlite3.Connection:
    with LIFECYCLE_LOCK:
        connection = sqlite3.connect(*args, **kwargs)
    try:
        return prepare(connection)
    except BaseException:
        close(connection)
        raise


def close(connection: sqlite3.Connection) -> None:
    with LIFECYCLE_LOCK:
        connection.close()
