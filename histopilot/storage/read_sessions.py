"""One connection per store for the reads of a call that lists many records.

A list reads hundreds of records, and each read used to open its own SQLite connection:
opening one checks the database files, sets its pragmas and, because of SQLite 3.51's
open/close deadlock, waits its turn behind every other open and close in the process
(``sqlite_connections``). Inside ``read_session()`` a store keeps the first connection it
opens for reads and reuses it until the block ends, so a large project's lists cost a few
connections instead of thousands.

Reads still see every committed change: a shared connection holds no transaction between
statements. Writes open their own connections as before. A connection is kept per thread,
so code that hands work to another thread opens its own there. Nested blocks share the
outer one. Values that are costly to probe and cannot change within one call, such as
whether the Task Center runner is alive, can be remembered for the block with
``remember``.
"""

import functools
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_SESSION: ContextVar[dict | None] = ContextVar("histopilot_read_session", default=None)


@contextmanager
def read_session() -> Iterator[None]:
    """Share one read connection per store and thread until the block ends."""
    if _SESSION.get() is not None:
        yield
        return
    session: dict = {"connections": {}, "values": {}}
    token = _SESSION.set(session)
    try:
        yield
    finally:
        _SESSION.reset(token)
        for connection, close in session["connections"].values():
            try:
                close(connection)
            except Exception:  # closing a shared reader never hides the call's own outcome
                pass


def active() -> bool:
    return _SESSION.get() is not None


def connection(key: tuple, open_connection: Callable, close: Callable):
    """The session's connection for ``key`` in this thread, opened on first use with
    ``open_connection()`` and closed with ``close(connection)`` when the session ends;
    None outside a session."""
    session = _SESSION.get()
    if session is None:
        return None
    key = (*key, threading.get_ident())
    entry = session["connections"].get(key)
    if entry is None:
        entry = (open_connection(), close)
        session["connections"][key] = entry
    return entry[0]


def remember(key: tuple, compute: Callable):
    """``compute()``, worked out once per session for ``key``; every call outside one."""
    session = _SESSION.get()
    if session is None:
        return compute()
    values = session["values"]
    if key not in values:
        values[key] = compute()
    return values[key]


def reads(method: Callable) -> Callable:
    """Run ``method`` inside a read session: for list methods that read many records."""

    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        with read_session():
            return method(*args, **kwargs)

    return wrapper


def store_key(store) -> str:
    """What identifies a store in session keys: its project folder, else the object."""
    folder = getattr(store, "folder", None)
    return str(folder) if folder is not None else f"object:{id(store)}"
