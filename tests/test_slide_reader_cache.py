"""Warm-reader reuse must stay bounded and never retain invalid or in-use handles."""

import sys
import threading
from types import SimpleNamespace

import pytest

from histopilot.storage.project_lock import StorageError
from histopilot.viewer.reader_cache import OPENSLIDE_READERS, ReaderCache
from histopilot.viewer.slide_images import _inspect_open_slide, _open


class Reader:
    def __init__(self):
        self.closed = 0

    def close(self):
        self.closed += 1


def test_reuses_exclusive_warm_reader_and_evicts_least_recent_idle():
    pool = ReaderCache(capacity=2)
    a, b, c = Reader(), Reader(), Reader()
    try:
        with pool.lease(("a",), lambda: a) as first:
            assert first is a
        with pool.lease(("b",), lambda: b):
            pass
        with pool.lease(("a",), lambda: pytest.fail("reopened warm slide")) as second:
            assert second is a and not a.closed
        with pool.lease(("c",), lambda: c):
            assert b.closed == 1 and a.closed == 0
    finally:
        pool.close()
    assert (a.closed, b.closed, c.closed) == (1, 1, 1)


def test_failed_reader_is_closed_and_cannot_poison_next_view():
    pool = ReaderCache()
    reader = Reader()
    with pytest.raises(RuntimeError, match="native read failed"):
        with pool.lease(("a",), lambda: reader):
            raise RuntimeError("native read failed")
    assert reader.closed == 1
    fresh = Reader()
    with pool.lease(("a",), lambda: fresh) as value:
        assert value is fresh
    pool.close()
    assert fresh.closed == 1


def test_open_failure_releases_capacity():
    pool = ReaderCache(capacity=1)

    def fail():
        raise OSError("cannot open")

    with pytest.raises(OSError):
        with pool.lease(("a",), fail):
            pass
    with pool.lease(("b",), Reader):
        pass
    pool.close()


def test_busy_pool_is_bounded_and_shutdown_waits_for_active_lease():
    pool = ReaderCache(capacity=1, wait_seconds=0.01)
    reader = Reader()
    with pool.lease(("a",), lambda: reader):
        with pytest.raises(StorageError) as error:
            with pool.lease(("a",), lambda: pytest.fail("opened beyond capacity")):
                pass
        assert error.value.code == "SLIDE_READER_BUSY"
        pool.close()
        assert reader.closed == 0
    assert reader.closed == 1


def test_waiting_reader_wakes_when_active_lease_returns():
    pool = ReaderCache(capacity=1, wait_seconds=1)
    reader, started, completed = Reader(), threading.Event(), threading.Event()
    failures = []

    def read():
        started.set()
        try:
            with pool.lease(("a",), lambda: pytest.fail("reopened queued slide")) as value:
                assert value is reader
            completed.set()
        except BaseException as error:
            failures.append(error)

    with pool.lease(("a",), lambda: reader):
        thread = threading.Thread(target=read)
        thread.start()
        assert started.wait(1)
        assert not completed.is_set()
    thread.join(2)
    pool.close()
    assert not thread.is_alive() and completed.is_set() and not failures


def test_idle_expiration_releases_native_resources():
    pool = ReaderCache(idle_seconds=0.02)
    closed = threading.Event()
    reader = Reader()

    def close():
        reader.closed += 1
        closed.set()

    reader.close = close
    with pool.lease(("a",), lambda: reader):
        pass
    assert closed.wait(1)
    pool.close()
    assert reader.closed == 1


def test_source_stat_changes_force_fresh_openslide_reader(tmp_path, monkeypatch):
    pytest.importorskip("PIL.Image")
    opened = []

    class FakeSlide(Reader):
        dimensions = (1024, 512)
        level_downsamples = [1, 4]

        @staticmethod
        def detect_format(path):
            return "test"

        def __init__(self, path):
            super().__init__()
            opened.append(self)

    class SlideError(Exception):
        pass

    monkeypatch.setitem(
        sys.modules, "openslide", SimpleNamespace(OpenSlide=FakeSlide, OpenSlideError=SlideError)
    )
    path = tmp_path / "slide.svs"
    path.write_bytes(b"initial")

    def inspect_slide(value):
        # This unit exercises the worker-local handle cache, below process isolation.
        with _open(value) as (slide, backend):
            return _inspect_open_slide(slide, backend)

    OPENSLIDE_READERS.close()
    try:
        assert inspect_slide(path)["width"] == 1024
        assert inspect_slide(path)["width"] == 1024
        assert len(opened) == 1
        path.write_bytes(b"changed content")
        assert inspect_slide(path)["width"] == 1024
        assert len(opened) == 2
        link = tmp_path / "link.svs"
        link.symlink_to(path)
        with pytest.raises(StorageError):
            inspect_slide(link)
        assert len(opened) == 2
    finally:
        OPENSLIDE_READERS.close()
    assert all(reader.closed == 1 for reader in opened)
