import io
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from histopilot.storage.project_lock import StorageError
from histopilot.viewer.image_cache import ImageCache
from histopilot.viewer.slide_images import render_slide


def test_lru_bounds_bytes_count_and_idle_expiry():
    now = [0]
    cache = ImageCache(max_bytes=6, max_entries=2, idle_seconds=10, clock=lambda: now[0])
    cache.get_or_create("a", lambda: b"aaa")
    cache.get_or_create("b", lambda: b"bbb")
    assert cache.get_or_create("a", lambda: pytest.fail("Cached bytes should be reused")) == b"aaa"
    cache.get_or_create("c", lambda: b"cc")
    assert list(cache._entries) == ["a", "c"]
    assert cache._bytes == 5
    now[0] = 11
    assert cache.get_or_create("d", lambda: b"dddd") == b"dddd"
    assert list(cache._entries) == ["d"]
    assert cache._bytes == 4
    # Large images may be returned, but cannot expand the retained cache.
    assert cache.get_or_create("large", lambda: b"1234567") == b"1234567"
    assert list(cache._entries) == ["d"]
    cache.get_or_create("e", lambda: b"eee")
    assert list(cache._entries) == ["e"]
    assert cache._bytes == 3


def test_concurrent_identical_requests_share_one_render():
    cache = ImageCache()
    started, release, follower_entered = Event(), Event(), Event()
    renders = []

    def render():
        renders.append(True)
        started.set()
        assert release.wait(3)
        return b"same-pixels"

    def follower():
        follower_entered.set()
        return cache.get_or_create("tile", render)

    with ThreadPoolExecutor(max_workers=2) as pool:
        leader = pool.submit(cache.get_or_create, "tile", render)
        assert started.wait(3)
        second = pool.submit(follower)
        assert follower_entered.wait(3)
        release.set()
        assert leader.result(3) == second.result(3) == b"same-pixels"
    assert len(renders) == 1
    assert not cache._pending


def test_different_tiles_can_render_concurrently():
    cache = ImageCache()
    started_a, started_b = Event(), Event()

    def render(started, other):
        started.set()
        assert other.wait(3)
        return b"tile"

    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(cache.get_or_create, "a", lambda: render(started_a, started_b))
        b = pool.submit(cache.get_or_create, "b", lambda: render(started_b, started_a))
        assert a.result(3) == b.result(3) == b"tile"


def test_failed_render_can_be_retried_and_never_poisoned():
    cache = ImageCache()
    with pytest.raises(StorageError, match="failed"):
        cache.get_or_create(
            "tile", lambda: (_ for _ in ()).throw(StorageError("failed", "SLIDE_READER_FAILED"))
        )
    assert not cache._pending
    assert cache.get_or_create("tile", lambda: b"recovered") == b"recovered"


def test_pending_bound_timeout_and_clear_do_not_cancel_active_render():
    cache = ImageCache(wait_seconds=0.01, max_pending=1)
    started, release = Event(), Event()

    def render():
        started.set()
        assert release.wait(3)
        return b"pixels"

    with ThreadPoolExecutor(max_workers=1) as pool:
        active = pool.submit(cache.get_or_create, "a", render)
        assert started.wait(3)
        for key in ("b", "a"):
            with pytest.raises(StorageError) as error:
                cache.get_or_create(key, lambda: pytest.fail("Must not duplicate work"))
            assert error.value.code == "SLIDE_READER_BUSY"
        cache.clear()
        release.set()
        assert active.result(3) == b"pixels"
    assert not cache._entries
    assert not cache._pending
    assert cache._bytes == 0


@pytest.fixture
def image_cache(monkeypatch):
    from histopilot.viewer import image_cache

    cache = ImageCache()
    monkeypatch.setattr(image_cache, "SLIDE_IMAGES", cache)
    return cache


def test_image_reuse_preserves_png_pixels_and_source_replacement(
    tmp_path, monkeypatch, image_cache
):
    from PIL import Image

    from histopilot.viewer import sdpc

    path = tmp_path / "slide.png"
    Image.new("RGB", (100, 100), "red").save(path)
    original = sdpc.read_openslide
    renders = []

    def counted(*args, **kwargs):
        renders.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(sdpc, "read_openslide", counted)
    first = render_slide(path, max_size=64, region=[0, 0, 64, 64])
    second = render_slide(path, max_size=64, region=(0, 0, 64, 64))
    assert first is second
    assert len(renders) == 1
    assert Image.open(io.BytesIO(second)).getpixel((10, 10)) == (255, 0, 0)
    Image.new("RGB", (100, 100), "blue").save(path)
    replaced = render_slide(path, max_size=64, region=(0, 0, 64, 64))
    assert len(renders) == 2
    assert Image.open(io.BytesIO(replaced)).getpixel((10, 10)) == (0, 0, 255)


def test_source_change_during_cache_hit_is_rejected(tmp_path, monkeypatch, image_cache):
    from PIL import Image

    path = tmp_path / "slide.png"
    Image.new("RGB", (100, 100), "red").save(path)
    render_slide(path)
    original = image_cache.get_or_create

    def changed_after_hit(key, render):
        content = original(key, render)
        Image.new("RGB", (100, 100), "blue").save(path)
        return content

    monkeypatch.setattr(image_cache, "get_or_create", changed_after_hit)
    with pytest.raises(StorageError) as error:
        render_slide(path)
    assert error.value.code == "SLIDE_SOURCE_CHANGED"
    assert not image_cache._entries


def test_source_change_during_render_is_not_published(tmp_path, monkeypatch, image_cache):
    from PIL import Image

    from histopilot.viewer import sdpc

    path = tmp_path / "slide.png"
    Image.new("RGB", (100, 100), "red").save(path)
    original = sdpc.read_openslide

    def changed(*args, **kwargs):
        content = original(*args, **kwargs)
        Image.new("RGB", (100, 100), "blue").save(path)
        return content

    monkeypatch.setattr(sdpc, "read_openslide", changed)
    with pytest.raises(StorageError) as error:
        render_slide(path)
    assert error.value.code == "SLIDE_SOURCE_CHANGED"
    assert not image_cache._entries
    assert not image_cache._pending


def test_cache_does_not_bypass_symlink_or_missing_source_checks(tmp_path, image_cache):
    from PIL import Image

    path = tmp_path / "slide.png"
    target = tmp_path / "other.png"
    Image.new("RGB", (100, 100), "red").save(path)
    Image.new("RGB", (100, 100), "blue").save(target)
    render_slide(path)
    path.unlink()
    with pytest.raises(StorageError) as error:
        render_slide(path)
    assert error.value.code == "SLIDE_SOURCE_CHANGED"
    path.symlink_to(target)
    with pytest.raises(StorageError) as error:
        render_slide(path)
    assert error.value.code == "STORAGE_UNSAFE_PATH"


def test_sdpc_cache_changes_with_reader_runtime(tmp_path, monkeypatch, image_cache):
    from histopilot.viewer import sdpc

    path = tmp_path / "slide.sdpc"
    path.write_bytes(b"example slide")
    runtime, renders = ["runtime-a"], []
    monkeypatch.setattr(sdpc, "_python", lambda: "python")
    monkeypatch.setattr(sdpc, "_runtime_key", lambda python: runtime[0])

    def render(*args, **kwargs):
        renders.append(runtime[0])
        return runtime[0].encode()

    monkeypatch.setattr(sdpc, "read_sdpc", render)
    assert render_slide(path) == render_slide(path) == b"runtime-a"
    runtime[0] = "runtime-b"
    assert render_slide(path) == b"runtime-b"
    assert renders == ["runtime-a", "runtime-b"]


@pytest.mark.parametrize(
    "region", [[False, 0, 10, 10], [0, 0, float("nan"), 10], [0, 0], "invalid"]
)
def test_invalid_viewports_rejected_before_cache_access(tmp_path, monkeypatch, image_cache, region):
    monkeypatch.setattr(
        image_cache, "get_or_create", lambda *args: pytest.fail("Invalid cache key")
    )
    with pytest.raises(StorageError) as error:
        render_slide(tmp_path / "absent.png", region=region)
    assert error.value.code == "SLIDE_VIEW_INVALID"
