"""Small exclusive leases for warm OpenSlide handles; no optional imports at startup."""

import atexit
import time
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Condition, Timer

from histopilot.storage.project_lock import StorageError


@dataclass(eq=False)
class _Entry:
    key: tuple
    reader: object = None
    busy: bool = True
    used: float = 0
    discard: bool = False
    timer: object = None


class ReaderCache:
    def __init__(self, capacity=2, idle_seconds=60, wait_seconds=2):
        self.capacity = capacity
        self.idle_seconds = idle_seconds
        self.wait_seconds = wait_seconds
        self._condition = Condition()
        self._entries = []

    @staticmethod
    def _close(entry):
        if entry.timer is not None:
            entry.timer.cancel()
            entry.timer = None
        if entry.reader is not None:
            try:
                entry.reader.close()
            except Exception:
                pass  # A failing cleanup must still release the cache slot.
            finally:
                entry.reader = None

    def _expire(self, entry):
        with self._condition:
            if entry not in self._entries or entry.busy:
                return
            if time.monotonic() - entry.used < self.idle_seconds:
                return
            self._entries.remove(entry)
            self._close(entry)
            self._condition.notify_all()

    @contextmanager
    def lease(self, key, factory):
        deadline = time.monotonic() + self.wait_seconds
        created = False
        with self._condition:
            while True:
                idle = [entry for entry in self._entries if not entry.busy]
                entry = next((entry for entry in idle if entry.key == key), None)
                if entry is not None:
                    entry.busy = True
                    if entry.timer is not None:
                        entry.timer.cancel()
                        entry.timer = None
                    break
                if len(self._entries) >= self.capacity and idle:
                    old = min(idle, key=lambda item: item.used)
                    self._entries.remove(old)
                    self._close(old)
                if len(self._entries) < self.capacity:
                    entry = _Entry(key)
                    self._entries.append(entry)
                    created = True
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise StorageError(
                        "Slide readers are busy. Retry this view shortly.", "SLIDE_READER_BUSY", 503
                    )
                self._condition.wait(remaining)
        try:
            if created:
                entry.reader = factory()
            yield entry.reader
        except BaseException:
            # OpenSlide errors latch on the handle; never reuse a failed reader.
            entry.discard = True
            raise
        finally:
            with self._condition:
                entry.busy = False
                entry.used = time.monotonic()
                if entry.discard or entry.reader is None:
                    if entry in self._entries:
                        self._entries.remove(entry)
                    self._close(entry)
                else:
                    entry.timer = Timer(self.idle_seconds, self._expire, (entry,))
                    entry.timer.daemon = True
                    entry.timer.start()
                self._condition.notify_all()

    def close(self):
        with self._condition:
            for entry in list(self._entries):
                entry.discard = True
                if not entry.busy:
                    self._entries.remove(entry)
                    self._close(entry)
            self._condition.notify_all()


OPENSLIDE_READERS = ReaderCache()
atexit.register(OPENSLIDE_READERS.close)
