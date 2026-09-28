"""Bounded warm slide workers; the historical module name keeps SDPC callers compatible.

OpenSDPC, OpenSlide and bounded raster fallback all run outside the API process.
"""

import atexit
import json
import os
import select
import shutil
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import BoundedSemaphore, Event, Lock, Thread

from histopilot.adapters.trident.runner import _worker_environment
from histopilot.adapters.trident.runtime import discover_runtime
from histopilot.storage.project_lock import StorageError, reject_symlink_components

# Bound both native concurrency and accumulated allocations in vendor decoders.
_READERS = BoundedSemaphore(2)
_POOL_LOCK = Lock()
_POOL = []
_REAPER = None
_REAPER_STOP = None
READER_TIMEOUT = 20
READER_MAX_REQUESTS = 128
READER_MAX_AGE = 300
READER_IDLE_TIMEOUT = 60
MAX_PNG_BYTES = 20 * 1024 * 1024
_SETUP_MESSAGE = (
    "SDPC viewing needs OpenSDPC, Pillow and OpenSlide in the slide reader environment. "
    "Use the working TRIDENT environment (HISTOPILOT_TRIDENT_PYTHON), or set "
    "HISTOPILOT_SDPC_PYTHON to an interpreter with those dependencies."
)
_OPENSLIDE_SETUP_MESSAGE = (
    "Slide viewing needs Pillow and OpenSlide in the HistoPilot Python environment. "
    "Install the optional imaging dependencies (histopilot[imaging]) and the OpenSlide library."
)
_ERRORS = {
    "STORAGE_UNSAFE_PATH": 403,
    "SLIDE_VIEWER_UNAVAILABLE": 503,
    "SLIDE_FORMAT_UNSUPPORTED": 422,
    "SLIDE_GEOMETRY_INVALID": 422,
    "SLIDE_RASTER_TOO_LARGE": 422,
    "SLIDE_VIEW_INVALID": 422,
    "SLIDE_VIEW_TOO_LARGE": 422,
    "SLIDE_SOURCE_CHANGED": 409,
}


def _python():
    configured = os.environ.get("HISTOPILOT_SDPC_PYTHON")
    python = (
        str(Path(shutil.which(configured) or configured).expanduser().absolute())
        if configured
        else discover_runtime()["pythonPath"]
    )
    if not Path(python).is_file() or not os.access(python, os.X_OK):
        raise StorageError(_SETUP_MESSAGE, "SLIDE_VIEWER_UNAVAILABLE", 503)
    return python


def _openslide_python():
    # OpenSlide uses the API's imaging environment; only SDPC requires TRIDENT.
    python = str(Path(sys.executable).absolute())
    if not Path(python).is_file() or not os.access(python, os.X_OK):
        raise StorageError(_OPENSLIDE_SETUP_MESSAGE, "SLIDE_VIEWER_UNAVAILABLE", 503)
    return python


def _source_stamp(path):
    reject_symlink_components(path)
    try:
        info = path.stat()
        if not path.is_file():
            raise OSError("Not a regular slide")
        return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]
    except OSError as error:
        raise StorageError(
            "The slide changed or is no longer available. Reopen it before viewing.",
            "SLIDE_SOURCE_CHANGED",
            409,
        ) from error


def _runtime_key(python):
    # A changed reader environment must not reuse an already-imported package.
    keys = (
        (name, value)
        for name, value in os.environ.items()
        if name.startswith(("HISTOPILOT_", "PYTHON", "LD_", "DYLD_"))
        or name
        in {
            "PATH",
            "CONDA_PREFIX",
            "VIRTUAL_ENV",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
        }
    )
    info = Path(python).stat()
    return python, info.st_mtime_ns, info.st_ino, tuple(sorted(keys))


def _failed():
    return StorageError(
        "The slide decoder could not read this slide. Retry the view; "
        "the HistoPilot service is still available.",
        "SLIDE_READER_FAILED",
        422,
    )


class _Reader:
    def __init__(self, key, python, backend="opensdpc"):
        self.key = key
        self.backend = backend
        self.setup_message = _SETUP_MESSAGE if backend == "opensdpc" else _OPENSLIDE_SETUP_MESSAGE
        self.busy = True
        self.closed = False
        self.requests = 0
        self.created = self.last_used = time.monotonic()
        self.temporary = TemporaryDirectory(prefix="histopilot-sdpc-")
        self.folder = Path(self.temporary.name)
        self.process = None
        try:
            env = (
                _worker_environment(python, cwd=self.folder)
                if backend == "opensdpc"
                else os.environ.copy()
            )
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                env[name] = "1"
            self.process = subprocess.Popen(
                [python, str(Path(__file__).with_name("sdpc_worker.py"))],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=env,
                cwd=self.folder,
                bufsize=0,
            )
            os.set_blocking(self.process.stdin.fileno(), False)
        except OSError as error:
            self.close()
            raise StorageError(self.setup_message, "SLIDE_VIEWER_UNAVAILABLE", 503) from error
        except BaseException:
            self.close()
            raise

    def expired(self, now):
        return (
            self.closed
            or self.process.poll() is not None
            or self.requests >= READER_MAX_REQUESTS
            or now - self.created >= READER_MAX_AGE
            # Retire before the child's idle deadline to avoid an EOF race.
            or now - self.last_used >= READER_IDLE_TIMEOUT * 0.8
        )

    def close(self):
        if self.closed:
            return
        self.closed = True
        process = self.process
        if process is not None:
            if process.poll() is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
            process.wait(timeout=2)
            if process.stdin:
                process.stdin.close()
            if process.stdout:
                process.stdout.close()
        self.temporary.cleanup()

    def request(self, path, stamp, max_size, region):
        self.requests += 1
        request = {
            "id": self.requests,
            "path": str(path),
            "sourceStamp": stamp,
            "reader": self.backend,
            "maxSize": max_size,
            "region": region,
            "output": str(self.folder),
            "maxRequests": READER_MAX_REQUESTS,
            "maxAge": READER_MAX_AGE,
            "idleTimeout": READER_IDLE_TIMEOUT,
        }
        try:
            payload = json.dumps(request, allow_nan=False).encode() + b"\n"
        except (ValueError, TypeError) as error:
            raise StorageError(
                "Provide finite numeric viewport bounds.", "SLIDE_VIEW_INVALID", 422
            ) from error
        if len(payload) > 32768:
            raise StorageError("The viewport request is too large.", "SLIDE_VIEW_INVALID", 422)
        try:
            # Never accept a prior request's files after a child crash.
            for name in ("result.json", "view.png"):
                (self.folder / name).unlink(missing_ok=True)
            deadline = time.monotonic() + READER_TIMEOUT
            written = 0
            while written < len(payload):
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([], [self.process.stdin], [], remaining)[1]:
                    self.close()
                    raise StorageError(
                        "The slide reader timed out. Retry a smaller view or check the slide file.",
                        "SLIDE_READER_TIMEOUT",
                        504,
                    )
                try:
                    written += os.write(self.process.stdin.fileno(), payload[written:])
                except BlockingIOError:
                    continue
            response = b""
            while not response.endswith(b"\n"):
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                    self.close()
                    raise StorageError(
                        "The slide reader timed out. Retry a smaller view or check the slide file.",
                        "SLIDE_READER_TIMEOUT",
                        504,
                    )
                chunk = os.read(self.process.stdout.fileno(), 32)
                if not chunk or len(response) + len(chunk) > 32:
                    self.close()
                    raise _failed()
                response += chunk
            if response != b"OK\n":
                raise ValueError("Invalid reader acknowledgement")
            result_path = self.folder / "result.json"
            if result_path.stat().st_size > 16384:
                raise ValueError("Oversized metadata")
            result = json.loads(result_path.read_bytes())
            if not isinstance(result, dict) or result.get("id") != self.requests:
                raise ValueError("Invalid reader result")
            if _source_stamp(path) != stamp:
                raise StorageError(
                    "The slide changed while it was being read. Reopen it.",
                    "SLIDE_SOURCE_CHANGED",
                    409,
                )
            if "error" in result:
                code = result["error"]["code"]
                if code not in _ERRORS:
                    raise ValueError("Unknown reader error")
                message = (
                    self.setup_message
                    if code == "SLIDE_VIEWER_UNAVAILABLE"
                    else result["error"]["message"]
                )
                raise StorageError(message, code, _ERRORS[code])
            if max_size is None:
                return result["metadata"]
            png = self.folder / "view.png"
            if not 0 < png.stat().st_size <= MAX_PNG_BYTES:
                raise ValueError("Invalid rendered image size")
            content = png.read_bytes()
            if not content.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError("Invalid rendered image")
            return content
        except StorageError as error:
            # A bad viewport is recoverable; reader/source failures retire it.
            if error.code not in {"SLIDE_VIEW_INVALID", "SLIDE_VIEW_TOO_LARGE"}:
                self.close()
            raise
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.close()
            raise _failed() from error


def _reap(stop):
    while not stop.wait(5):
        with _POOL_LOCK:
            for reader in list(_POOL):
                if not reader.busy and reader.expired(time.monotonic()):
                    _POOL.remove(reader)
                    reader.close()


def _acquire_reader(key, python, backend):
    global _REAPER, _REAPER_STOP
    with _POOL_LOCK:
        for reader in list(_POOL):
            if not reader.busy and (
                reader.expired(time.monotonic()) or (reader.key[0] == key[0] and reader.key != key)
            ):
                _POOL.remove(reader)
                reader.close()
        for reader in _POOL:
            if not reader.busy and reader.key == key:
                reader.busy = True
                return reader
        if len(_POOL) >= 2:
            idle = min(
                (reader for reader in _POOL if not reader.busy), key=lambda reader: reader.last_used
            )
            _POOL.remove(idle)
            idle.close()
        reader = _Reader(key, python, backend)
        _POOL.append(reader)
        if _REAPER is None or not _REAPER.is_alive() or _REAPER_STOP.is_set():
            _REAPER_STOP = Event()
            _REAPER = Thread(
                target=_reap, args=(_REAPER_STOP,), name="slide-reader-cleanup", daemon=True
            )
            _REAPER.start()
        return reader


def close_readers():
    """Stop native children and remove their private IPC files at service shutdown."""
    with _POOL_LOCK:
        if _REAPER_STOP is not None:
            _REAPER_STOP.set()
        readers = list(_POOL)
        _POOL.clear()
        for reader in readers:
            reader.close()


def _read(path, *, backend, max_size=None, region=None):
    """Share two native slots across formats, with a deadline covering open and read."""
    if not _READERS.acquire(timeout=2):
        raise StorageError(
            "Slide readers are busy. Retry this view shortly.", "SLIDE_READER_BUSY", 503
        )
    reader = None
    try:
        python = _python() if backend == "opensdpc" else _openslide_python()
        path = Path(path).absolute()
        stamp = _source_stamp(path)
        key = (str(path), tuple(stamp), backend, _runtime_key(python))
        reader = _acquire_reader(key, python, backend)
        return reader.request(path, stamp, max_size, region)
    finally:
        if reader is not None:
            with _POOL_LOCK:
                reader.busy = False
                reader.last_used = time.monotonic()
                if reader.expired(reader.last_used):
                    if reader in _POOL:
                        _POOL.remove(reader)
                    reader.close()
        _READERS.release()


def read_sdpc(path, *, max_size=None, region=None):
    """Read through the configured OpenSDPC environment, isolated from the API."""
    return _read(path, backend="opensdpc", max_size=max_size, region=region)


def read_openslide(path, *, max_size=None, region=None):
    """Isolate native OpenSlide and bounded Pillow fallback in the current environment."""
    return _read(path, backend="openslide", max_size=max_size, region=region)


atexit.register(close_readers)
