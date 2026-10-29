"""Structured logs that carry the request and the task they belong to.

The API's request middleware puts each request's ID in a context variable and the Task
Center runner names the task it acts on; a worker reads both from the environment the
runner starts it with (``HISTOPILOT_TASK_ID``, ``HISTOPILOT_REQUEST_ID``). Every record is
stamped with them in the thread that made it.

A process routes its records once, at startup:

- ``configure`` (the service and the runner) writes JSON Lines to a size-rotated file in
  the Task Center state directory, and to stderr only warnings and errors (the service) or,
  with ``HISTOPILOT_LOG_STDERR=1``, everything (containers). Each sink has a bounded queue
  and a thread of its own, so logging never waits for a disk or a terminal: a sink that
  falls behind drops records instead of holding up a request. (A service once hung because
  its access log filled a terminal nobody read.)
- ``configure_worker`` writes HistoPilot's own records in place to the worker's stderr,
  which is its task's log.

``HISTOPILOT_LOG_FORMAT`` (``text``, the default, or ``json``) picks the format of stderr;
files are always JSON Lines. ``HISTOPILOT_LOG_LEVEL`` sets the level (default ``INFO``).
Standard library only: pinned compute archives import their own copy of this module.
"""

import contextvars
import copy
import json
import logging
import os
import queue
import re
import sys
import threading
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

FORMAT_ENV = "HISTOPILOT_LOG_FORMAT"
LEVEL_ENV = "HISTOPILOT_LOG_LEVEL"
STDERR_ENV = "HISTOPILOT_LOG_STDERR"
TASK_ENV = "HISTOPILOT_TASK_ID"
REQUEST_ENV = "HISTOPILOT_REQUEST_ID"
ROTATE_BYTES = 10 * 1024 * 1024
KEEP_ROTATED = 5
# How many records a sink may fall behind by before it drops new ones.
QUEUE_RECORDS = 10_000
# How long closing waits for a sink to write what it still holds.
CLOSE_SECONDS = 2.0
# A request ID a client may choose: what tracing systems send, never a character that
# could forge a log line or a header.
_REQUEST_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}")
_BARE = re.compile(r"[^\s=\"]+")
_ON = frozenset({"1", "true", "yes", "on"})

REQUEST_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)
TASK_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar("task_id", default=None)
# A worker's IDs, read from its environment: they hold in every thread it starts.
_PROCESS: dict[str, str | None] = {"request_id": None, "task_id": None}
_CONTEXT = ("request_id", "task_id")
# What every LogRecord has; anything else on a record came from a call's ``extra``.
# Uvicorn adds ``color_message``, its message again with terminal colour codes.
_STANDARD = frozenset(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {
    "message",
    "asctime",
    "taskName",
    "color_message",
}
_PLAIN = logging.Formatter()


def valid_request_id(value) -> str | None:
    """``value`` if a client may use it as a request ID, else None."""
    return value if isinstance(value, str) and _REQUEST_ID.fullmatch(value) else None


def new_request_id() -> str:
    return uuid.uuid4().hex


def current_request() -> str | None:
    """The request being served, or in a worker the request that queued its task."""
    return REQUEST_ID.get() or _PROCESS["request_id"]


def current_task() -> str | None:
    return TASK_ID.get() or _PROCESS["task_id"]


@contextmanager
def bound(*, task: str | None = None, request: str | None = None):
    """Stamp the records made inside the block with this task or request."""
    tokens = []
    if task is not None:
        tokens.append((TASK_ID, TASK_ID.set(task)))
    if request is not None:
        tokens.append((REQUEST_ID, REQUEST_ID.set(request)))
    try:
        yield
    finally:
        for variable, token in reversed(tokens):
            variable.reset(token)


def stamp(record: logging.LogRecord) -> bool:
    """Give a record the IDs of the context it is made in, unless its call named its own.

    A handler filter: it runs in the thread that logs, before any queue."""
    if getattr(record, "request_id", None) is None:
        record.request_id = current_request()
    if getattr(record, "task_id", None) is None:
        record.task_id = current_task()
    return True


def level() -> int:
    """HISTOPILOT_LOG_LEVEL as a number; INFO when unset or unknown."""
    value = logging.getLevelName(os.environ.get(LEVEL_ENV, "").strip().upper() or "INFO")
    return value if isinstance(value, int) else logging.INFO


def stream_format() -> str:
    return "json" if os.environ.get(FORMAT_ENV, "").strip().lower() == "json" else "text"


def _timestamp(created: float) -> str:
    moment = datetime.fromtimestamp(created, UTC).isoformat(timespec="milliseconds")
    return moment.replace("+00:00", "Z")


def _fields(record: logging.LogRecord) -> dict:
    """What a call added with ``extra``."""
    return {
        key: value
        for key, value in vars(record).items()
        if key not in _STANDARD and key not in _CONTEXT and not key.startswith("_")
    }


def _traceback(formatter: logging.Formatter, record: logging.LogRecord) -> str | None:
    text = record.exc_text
    if not text and record.exc_info:
        text = formatter.formatException(record.exc_info)
    if record.stack_info:
        text = (text + "\n" if text else "") + formatter.formatStack(record.stack_info)
    return text or None


class JsonFormatter(logging.Formatter):
    """One JSON object per line: ``ts``, ``level``, ``logger``, ``msg``, ``request_id``,
    ``task_id``, the process's ``component``, the record's own fields and, for an error,
    its traceback as ``exc``. A field never replaces one of these."""

    def __init__(self, component: str | None = None):
        super().__init__()
        self.component = component

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": _timestamp(record.created),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": getattr(record, "request_id", None),
            "task_id": getattr(record, "task_id", None),
        }
        if self.component:
            entry["component"] = self.component
        for key, value in _fields(record).items():
            entry.setdefault(key, value)
        error = _traceback(self, record)
        if error:
            entry["exc"] = error
        return json.dumps(entry, ensure_ascii=False, default=str)


def _text(value) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if _BARE.fullmatch(text) else json.dumps(text, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """``ts LEVEL logger: message key=value ...``, then any traceback: for people."""

    def format(self, record: logging.LogRecord) -> str:
        fields = {key: getattr(record, key, None) for key in _CONTEXT} | _fields(record)
        line = (
            f"{_timestamp(record.created)} {record.levelname} {record.name}: {record.getMessage()}"
        )
        pairs = " ".join(
            f"{key}={_text(value)}" for key, value in fields.items() if value is not None
        )
        if pairs:
            line = f"{line} {pairs}"
        error = _traceback(self, record)
        return f"{line}\n{error}" if error else line


def formatter(kind: str, component: str | None = None) -> logging.Formatter:
    return JsonFormatter(component) if kind == "json" else TextFormatter()


class _FileSink(RotatingFileHandler):
    def handleError(self, record):
        """A full or vanished disk loses the record quietly: the usual report would go to
        stderr, which must not be written from here."""


class _StderrSink(logging.StreamHandler):
    """stderr as it is when a record is written, so a replaced stream is followed."""

    def __init__(self):
        logging.Handler.__init__(self)

    @property
    def stream(self):
        return sys.stderr

    def handleError(self, record):
        """A closed or broken stderr loses the record quietly."""


# Not logging.handlers.QueueHandler and QueueListener: the handler folds a traceback into
# the message and reports its own failures on stderr, and the listener feeds every
# destination from one thread (a stuck terminal would stall the file too) and waits for it
# without a limit when it stops.
class _Sink:
    """One destination, fed through a bounded queue by a writer thread of its own."""

    def __init__(self, name: str, handler: logging.Handler, threshold: int):
        self.name = name
        self.handler = handler
        self.threshold = threshold
        self.dropped = 0
        self.queue: queue.Queue = queue.Queue(QUEUE_RECORDS)
        self.thread = threading.Thread(
            target=self._drain, name=f"histopilot-log-{name}", daemon=True
        )
        self.thread.start()

    def offer(self, record: logging.LogRecord) -> None:
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            self.dropped += 1

    def _drain(self) -> None:
        while True:
            record = self.queue.get()
            if record is None:
                return
            try:
                self.handler.handle(record)
            except Exception:  # noqa: BLE001 - one bad record must not end the writer
                pass

    def close(self) -> None:
        try:
            self.queue.put(None, timeout=CLOSE_SECONDS)
        except queue.Full:
            pass
        self.thread.join(CLOSE_SECONDS)
        if not self.thread.is_alive():  # a writer stuck on its stream keeps the handler
            self.handler.close()


def _prepared(record: logging.LogRecord) -> logging.LogRecord:
    """A copy another thread can format later: the message rendered, the traceback as
    text, and the IDs of the context the record was made in."""
    record = copy.copy(record)
    stamp(record)
    try:
        record.msg = record.getMessage()
    except Exception:  # noqa: BLE001 - a mismatched format is shown, never raised here
        record.msg = f"{record.msg!r} (arguments did not fit: {record.args!r})"
    record.args = None
    if record.exc_info:
        record.exc_text = record.exc_text or _PLAIN.formatException(record.exc_info)
        record.exc_info = None
    return record


class _Fanout(logging.Handler):
    """Hands each record to every sink at its level, without waiting for any of them."""

    def __init__(self, sinks: list[_Sink]):
        super().__init__()
        self.sinks = sinks
        self.closed = False

    def emit(self, record: logging.LogRecord) -> None:
        try:
            prepared = _prepared(record)
        except Exception:  # noqa: BLE001 - logging never raises into the code that logs
            return
        for sink in self.sinks:
            if prepared.levelno >= sink.threshold:
                sink.offer(prepared)

    def handleError(self, record):
        """Never write a report to stderr from the thread that logged."""

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            for sink in self.sinks:
                sink.close()
        super().close()


class Routing:
    """What ``configure`` installed: how many records each sink dropped, and ``close``."""

    def __init__(self, handler: _Fanout, previous_level: int):
        self.handler = handler
        self.previous_level = previous_level

    def dropped(self) -> dict[str, int]:
        return {sink.name: sink.dropped for sink in self.handler.sinks}

    def close(self) -> None:
        global _ROUTING
        root = logging.getLogger()
        root.removeHandler(self.handler)
        root.setLevel(self.previous_level)
        logging.captureWarnings(False)
        self.handler.close()
        if _ROUTING is self:
            _ROUTING = None


_ROUTING: Routing | None = None


def _requested(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _ON


def configure(component: str, *, file: Path | None = None, stderr_warnings=False) -> Routing:
    """Route this process's records at HISTOPILOT_LOG_LEVEL: as JSON Lines to ``file``
    (rotated at 10 MiB, five kept), and to stderr with HISTOPILOT_LOG_STDERR=1, or else
    only its warnings and errors when ``stderr_warnings``. A file that cannot be opened
    is left out: the process runs without it rather than not at all."""
    global _ROUTING
    if _ROUTING is not None:
        _ROUTING.close()
    threshold = level()
    sinks = []
    if file is not None:
        try:
            handler = _FileSink(
                file,
                maxBytes=ROTATE_BYTES,
                backupCount=KEEP_ROTATED,
                encoding="utf-8",
                errors="backslashreplace",
            )
        except OSError:
            handler = None
        if handler is not None:
            handler.setFormatter(JsonFormatter(component))
            sinks.append(_Sink("file", handler, threshold))
    everything = _requested(STDERR_ENV)
    if everything or stderr_warnings:
        handler = _StderrSink()
        handler.setFormatter(formatter(stream_format(), component))
        floor = threshold if everything else max(threshold, logging.WARNING)
        sinks.append(_Sink("stderr", handler, floor))
    root = logging.getLogger()
    routing = Routing(_Fanout(sinks), root.level)
    root.addHandler(routing.handler)
    root.setLevel(threshold)
    # Python's warnings join the records, so nothing writes to stderr on its own.
    logging.captureWarnings(True)
    _ROUTING = routing
    return routing


def dropped() -> dict[str, int]:
    """Records each sink of this process dropped because it fell behind; {} unrouted."""
    return _ROUTING.dropped() if _ROUTING is not None else {}


_WORKER: logging.Handler | None = None


def configure_worker() -> None:
    """A Task Center worker logs to its stderr, its task's log, as it goes, stamped with
    the task and the request the runner started it for, at HISTOPILOT_LOG_LEVEL.

    Only HistoPilot's own records are routed. Other libraries print as they always have:
    Lightning, for one, keeps a console handler of its own and would otherwise show every
    line twice."""
    global _WORKER
    _PROCESS["task_id"] = os.environ.get(TASK_ENV) or None
    _PROCESS["request_id"] = valid_request_id(os.environ.get(REQUEST_ENV))
    if _WORKER is not None:
        return
    _WORKER = _StderrSink()
    _WORKER.setFormatter(formatter(stream_format(), "worker"))
    _WORKER.addFilter(stamp)
    own = logging.getLogger("histopilot")
    own.addHandler(_WORKER)
    own.setLevel(level())
    own.propagate = False
