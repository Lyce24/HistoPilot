"""Bounded lossless slide-image results shared across requests and viewer reopens."""

import time
from collections import OrderedDict
from concurrent.futures import Future, TimeoutError
from threading import Lock

from histopilot.storage.project_lock import StorageError


class ImageCache:
    """LRU encoded bytes, with one render per key and bounded waiting callers."""

    def __init__(
        self,
        *,
        max_bytes=128 * 1024**2,
        max_entries=512,
        idle_seconds=600,
        wait_seconds=25,
        max_pending=64,
        clock=time.monotonic,
    ):
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.idle_seconds = idle_seconds
        self.wait_seconds = wait_seconds
        self.max_pending = max_pending
        self._clock = clock
        self._lock = Lock()
        self._entries = OrderedDict()
        self._pending = {}
        self._bytes = 0
        self._generation = 0

    @staticmethod
    def _busy():
        return StorageError(
            "Slide images are still being prepared. Retry this view shortly.",
            "SLIDE_READER_BUSY",
            503,
        )

    def _remove(self, key):
        content, _used = self._entries.pop(key)
        self._bytes -= len(content)

    def _expire(self, now):
        while self._entries:
            key, (_content, used) = next(iter(self._entries.items()))
            if now - used < self.idle_seconds:
                break
            self._remove(key)

    def get_or_create(self, key, render):
        with self._lock:
            now = self._clock()
            self._expire(now)
            if key in self._entries:
                content, _used = self._entries[key]
                self._entries[key] = content, now
                self._entries.move_to_end(key)
                return content
            future = self._pending.get(key)
            leader = future is None
            generation = self._generation
            if leader:
                if len(self._pending) >= self.max_pending:
                    raise self._busy()
                future = Future()
                self._pending[key] = future
        if not leader:
            try:
                return future.result(timeout=self.wait_seconds)
            except TimeoutError as error:
                raise self._busy() from error
        try:
            content = render()
            if not isinstance(content, bytes):
                raise TypeError("Slide image renderers must return immutable bytes")
            with self._lock:
                if generation == self._generation and len(content) <= self.max_bytes:
                    now = self._clock()
                    self._expire(now)
                    while self._entries and (
                        len(self._entries) >= self.max_entries
                        or self._bytes + len(content) > self.max_bytes
                    ):
                        self._remove(next(iter(self._entries)))
                    if self.max_entries > 0:
                        self._entries[key] = content, now
                        self._bytes += len(content)
                self._pending.pop(key, None)
            future.set_result(content)
            return content
        except BaseException as error:
            with self._lock:
                self._pending.pop(key, None)
            future.set_exception(error)
            raise

    def discard(self, key):
        with self._lock:
            if key in self._entries:
                self._remove(key)

    def clear(self):
        """Drop cached bytes and prevent current renders from refilling this generation."""
        with self._lock:
            self._entries.clear()
            self._bytes = 0
            self._generation += 1


SLIDE_IMAGES = ImageCache()
