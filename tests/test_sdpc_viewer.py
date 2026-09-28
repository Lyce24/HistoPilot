"""SDPC native process isolation and exact level-0 attention-to-tissue alignment."""

import io
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from test_interpretation import save
from test_interpretation import study as study

from histopilot.storage.project_lock import StorageError
from histopilot.viewer import sdpc, slide_images

Image = pytest.importorskip("PIL.Image")


class Pyramid:
    level_dimensions = [(1025, 769), (256, 192), (64, 48)]
    level_downsamples = [1.0, 4.0, 16.0]

    def __init__(self):
        self.reads = []

    def read_region(self, origin, level, size):
        x, y = (math.floor(v / self.level_downsamples[level]) for v in origin)
        self.reads.append((origin, level, size))
        width, height = self.level_dimensions[level]
        assert 0 <= x < width and 0 <= y < height
        assert x + size[0] <= width and y + size[1] <= height
        result = Image.new("RGB", size)
        result.putdata(
            [
                ((x + dx) % 256, (y + dy) % 256, 100)
                for dy in range(size[1])
                for dx in range(size[0])
            ]
        )
        return result


def test_sdpc_off_grid_crop_matches_exact_level0_field_of_view():
    slide = Pyramid()
    content = slide_images._render_open_slide(
        slide, "opensdpc", max_size=64, region=(35.5, 51.25, 513, 333)
    )
    assert slide.reads == [((35, 51), 1, (130, 85))]
    # Independent crop of the full pyramid level is a reference for the affine
    # transform: a floored-origin crop without the residual fails this equality.
    full = Pyramid().read_region((0, 0), 1, (256, 192))
    expected = full.resize(
        (64, 42), Image.Resampling.LANCZOS, box=(35.5 / 4, 51.25 / 4, 548.5 / 4, 384.25 / 4)
    )
    with Image.open(io.BytesIO(content)) as image:
        # Lanczos differs only in its support at the local tile's boundary.
        assert image.crop((2, 2, 62, 40)).tobytes() == expected.crop((2, 2, 62, 40)).tobytes()


def test_sdpc_truncated_pyramid_edges_are_clamped_and_padded():
    slide = Pyramid()
    content = slide_images._render_open_slide(slide, "opensdpc", max_size=64, region=None)
    assert slide.reads == [((0, 0), 2, (64, 48))]
    with Image.open(io.BytesIO(content)) as image:
        assert image.size == (64, 48)
    # An entire viewport in the missing coarse edge never reaches the decoder.
    slide = Pyramid()
    content = slide_images._render_open_slide(
        slide, "opensdpc", max_size=64, region=(1024, 0, 1, 769)
    )
    assert slide.reads == []
    with Image.open(io.BytesIO(content)) as image:
        assert image.size == (1, 64)
        assert image.getextrema() == ((255, 255),) * 3


def test_sparse_sdpc_pyramid_refuses_huge_intermediate_before_native_read():
    slide = Pyramid()
    slide.level_dimensions = [(100000, 100000)]
    slide.level_downsamples = [1.0]
    with pytest.raises(StorageError) as error:
        slide_images._render_open_slide(slide, "opensdpc", max_size=64, region=None)
    assert error.value.code == "SLIDE_VIEW_TOO_LARGE"
    assert slide.reads == []


@pytest.fixture(autouse=True)
def clean_reader_pool():
    sdpc.close_readers()
    yield
    sdpc.close_readers()


@pytest.fixture
def reader_runtime(tmp_path, monkeypatch):
    # Exercise actual child process IPC and imports without installing a native
    # decoder in CI. A fake package lives only on the subprocess search path.
    package = tmp_path / "opensdpc"
    package.mkdir()
    source = package / "__init__.py"
    source.write_text("""
from PIL import Image
class OpenSdpc:
    level_dimensions = [(1024, 512), (256, 128)]
    level_downsamples = [1.0, 4.0]
    def __init__(self, path): pass
    def read_region(self, origin, level, size):
        return Image.new("RGB", size, (120, 30, 60))
    def close(self): pass
""")
    monkeypatch.setenv("HISTOPILOT_SDPC_PYTHON", sys.executable)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    path = tmp_path / "slide with spaces.SDPC"
    path.touch()
    return path, source


def test_sdpc_uses_child_interpreter_and_preserves_service_import_boundary(reader_runtime):
    path, _ = reader_runtime
    before = set(sys.modules)
    metadata = slide_images.inspect_slide(path)
    assert metadata == {
        "width": 1024,
        "height": 512,
        "backend": "opensdpc",
        "levelDownsamples": [1.0, 4.0],
        "coordinateSpace": "level0",
    }
    with Image.open(io.BytesIO(slide_images.render_slide(path, max_size=128))) as image:
        assert image.size == (128, 64)
        assert image.getpixel((64, 32)) == (120, 30, 60)
    assert not ({"opensdpc", "openslide", "torch", "cv2"} & (set(sys.modules) - before))
    # Invalid geometry returns a typed viewer error, without raster fallback.
    with pytest.raises(StorageError) as error:
        slide_images.render_slide(path, max_size=128, region=(1000, 0, 25, 1))
    assert error.value.code == "SLIDE_VIEW_INVALID"


@pytest.mark.parametrize(
    "body,code",
    [
        ("raise ImportError('missing optional dependencies')", "SLIDE_VIEWER_UNAVAILABLE"),
        ("import os; os._exit(139)", "SLIDE_READER_FAILED"),
    ],
)
def test_sdpc_import_failure_and_native_exit_are_contained(reader_runtime, body, code):
    path, source = reader_runtime
    source.write_text(body)
    with pytest.raises(StorageError) as error:
        slide_images.inspect_slide(path)
    assert error.value.code == code
    if code == "SLIDE_VIEWER_UNAVAILABLE":
        assert "HISTOPILOT_SDPC_PYTHON" in str(error.value)
    assert sdpc._READERS.acquire(blocking=False)
    sdpc._READERS.release()


def test_sdpc_timeout_kills_reader_and_releases_capacity(reader_runtime, monkeypatch):
    path, source = reader_runtime
    source.write_text("import time; time.sleep(5)")
    monkeypatch.setattr(sdpc, "READER_TIMEOUT", 0.1)
    with pytest.raises(StorageError) as error:
        slide_images.inspect_slide(path)
    assert error.value.code == "SLIDE_READER_TIMEOUT"
    assert sdpc._READERS.acquire(blocking=False)
    sdpc._READERS.release()


def test_sdpc_missing_configured_interpreter_has_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setenv("HISTOPILOT_SDPC_PYTHON", str(tmp_path / "no-python"))
    with pytest.raises(StorageError) as error:
        slide_images.inspect_slide(tmp_path / "slide.sdpc")
    assert error.value.code == "SLIDE_VIEWER_UNAVAILABLE"
    assert "HISTOPILOT_TRIDENT_PYTHON" in str(error.value)


def test_sdpc_rejects_symlink_before_launch(reader_runtime, monkeypatch):
    path, _ = reader_runtime
    link = path.with_name("link.sdpc")
    link.symlink_to(path)
    monkeypatch.setattr(sdpc, "read_sdpc", lambda *a, **k: pytest.fail("launched reader"))
    with pytest.raises(StorageError) as error:
        slide_images.inspect_slide(link)
    assert error.value.status_code == 403


def test_sdpc_study_freezes_geometry_and_serves_original_tissue(study, reader_runtime, monkeypatch):
    service, selection, executor = study
    path, _ = reader_runtime
    selected = selection.model_copy(
        update={"slides": [selection.slides[0].model_copy(update={"slidePath": str(path)})]}
    )
    document, _ = save((service, selected, executor))
    row = document["manifest"]["slides"][0]
    assert row["backend"] == "opensdpc"
    assert (row["width"], row["height"]) == (1024, 512)
    assert row["coordinateSpace"] == "level0" and row["patchCount"] == 3
    with Image.open(
        io.BytesIO(
            service.image(document["id"], row["slideId"], region=(35, 51, 513, 333), max_size=64)
        )
    ) as image:
        assert image.size == (64, 42)
        assert image.getpixel((32, 21)) == (120, 30, 60)

    def failed(*args, **kwargs):
        raise StorageError("Reader timed out.", "SLIDE_READER_TIMEOUT", 504)

    monkeypatch.setattr(sdpc, "read_sdpc", failed)
    with pytest.raises(StorageError) as error:
        service.image(document["id"], row["slideId"])
    assert error.value.code == "SLIDE_READER_TIMEOUT" and error.value.status_code == 504


def test_warm_reader_reuses_process_and_open_slide_despite_native_stdout(reader_runtime):
    path, source = reader_runtime
    opened = path.parent / "opens.txt"
    source.write_text(
        source.read_text().replace(
            "def __init__(self, path): pass",
            f"def __init__(self, path):\n        import os\n        os.write(1, b'native decoder chatter\\n')\n        open({str(opened)!r}, 'a').write(str(os.getpid()) + '\\n')",
        )
    )
    slide_images.inspect_slide(path)
    first = sdpc._POOL[0]
    with Image.open(io.BytesIO(slide_images.render_slide(path, max_size=128))) as image:
        assert image.size == (128, 64)
    slide_images.render_slide(path, max_size=64, region=(17, 23, 257, 111))
    assert sdpc._POOL == [first] and first.requests == 3
    assert len(opened.read_text().splitlines()) == 1
    assert first.process.poll() is None
    folder = first.folder
    sdpc.close_readers()
    assert first.process.poll() is not None and not folder.exists()


def test_warm_reader_reopens_when_slide_identity_changes(reader_runtime):
    path, _ = reader_runtime
    slide_images.inspect_slide(path)
    old = sdpc._POOL[0]
    path.write_bytes(b"slide replaced")
    slide_images.inspect_slide(path)
    assert len(sdpc._POOL) == 1 and sdpc._POOL[0] is not old
    assert old.process.poll() is not None and not old.folder.exists()


def test_warm_reader_does_not_reuse_changed_python_environment(reader_runtime, monkeypatch):
    path, _ = reader_runtime
    slide_images.inspect_slide(path)
    old = sdpc._POOL[0]
    monkeypatch.setenv("HISTOPILOT_TEST_READER_VERSION", "changed")
    slide_images.inspect_slide(path)
    assert len(sdpc._POOL) == 1 and sdpc._POOL[0] is not old
    assert old.process.poll() is not None


def test_warm_reader_recovers_after_native_crash(reader_runtime):
    path, source = reader_runtime
    body = source.read_text()
    # Import succeeds and the child opens the slide, then native read crashes.
    source.write_text(
        body.replace('return Image.new("RGB", size, (120, 30, 60))', "import os; os._exit(139)")
    )
    slide_images.inspect_slide(path)
    old = sdpc._POOL[0]
    with pytest.raises(StorageError) as error:
        slide_images.render_slide(path, max_size=128)
    assert error.value.code == "SLIDE_READER_FAILED"
    assert old.process.poll() is not None and not old.folder.exists()
    assert sdpc._POOL == []
    source.write_text(body)
    with Image.open(io.BytesIO(slide_images.render_slide(path, max_size=128))) as image:
        assert image.getpixel((64, 32)) == (120, 30, 60)


def test_warm_reader_recovers_after_timeout(reader_runtime, monkeypatch):
    path, source = reader_runtime
    body = source.read_text()
    source.write_text(
        body.replace(
            'return Image.new("RGB", size, (120, 30, 60))',
            'import time; time.sleep(5)\n        return Image.new("RGB", size)',
        )
    )
    slide_images.inspect_slide(path)
    old = sdpc._POOL[0]
    monkeypatch.setattr(sdpc, "READER_TIMEOUT", 0.05)
    with pytest.raises(StorageError) as error:
        slide_images.render_slide(path, max_size=128)
    assert error.value.code == "SLIDE_READER_TIMEOUT"
    assert old.process.poll() is not None and not old.folder.exists()
    assert sdpc._POOL == []
    source.write_text(body)
    monkeypatch.setattr(sdpc, "READER_TIMEOUT", 2)
    assert slide_images.inspect_slide(path)["width"] == 1024


@pytest.mark.parametrize("limit", ["requests", "age", "idle"])
def test_warm_reader_recycles_at_lifetime_limits(reader_runtime, monkeypatch, limit):
    path, _ = reader_runtime
    if limit == "requests":
        monkeypatch.setattr(sdpc, "READER_MAX_REQUESTS", 2)
    slide_images.inspect_slide(path)
    old = sdpc._POOL[0]
    if limit == "requests":
        slide_images.inspect_slide(path)
        assert old.process.poll() is not None
    elif limit == "age":
        old.created -= sdpc.READER_MAX_AGE + 1
    else:
        old.last_used -= sdpc.READER_IDLE_TIMEOUT + 1
    slide_images.inspect_slide(path)
    assert sdpc._POOL[0] is not old and old.process.poll() is not None


def test_child_exits_when_idle_even_without_parent_cleanup(reader_runtime, monkeypatch):
    path, _ = reader_runtime
    monkeypatch.setattr(sdpc, "READER_IDLE_TIMEOUT", 0.15)
    slide_images.inspect_slide(path)
    old = sdpc._POOL[0]
    assert old.process.wait(timeout=2) == 0
    assert slide_images.inspect_slide(path)["width"] == 1024
    assert sdpc._POOL[0] is not old


def test_two_concurrent_readers_are_bounded_and_busy_requests_do_not_queue_forever(reader_runtime):
    path, source = reader_runtime
    source.write_text(
        source.read_text().replace(
            'return Image.new("RGB", size, (120, 30, 60))',
            'import time; time.sleep(0.1)\n        return Image.new("RGB", size, (120, 30, 60))',
        )
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Distinct tile sizes exercise both native slots; identical requests now share one render.
        futures = [
            executor.submit(slide_images.render_slide, path, max_size=128 + index)
            for index in range(2)
        ]
        assert all(future.result(timeout=5).startswith(b"\x89PNG") for future in futures)
    assert len(sdpc._POOL) == 2
    assert all(not reader.busy for reader in sdpc._POOL)
    assert sdpc._READERS.acquire(blocking=False)
    assert sdpc._READERS.acquire(blocking=False)
    try:
        started = time.monotonic()
        with pytest.raises(StorageError) as error:
            slide_images.inspect_slide(path)
        assert error.value.code == "SLIDE_READER_BUSY"
        assert time.monotonic() - started < 3
    finally:
        sdpc._READERS.release()
        sdpc._READERS.release()


def test_source_change_during_native_read_rejects_result_and_retires_worker(reader_runtime):
    path, source = reader_runtime
    source.write_text(
        source.read_text().replace(
            'return Image.new("RGB", size, (120, 30, 60))',
            f'open({str(path)!r}, "ab").write(b"changed")\n        return Image.new("RGB", size)',
        )
    )
    with pytest.raises(StorageError) as error:
        slide_images.render_slide(path, max_size=128)
    assert error.value.code == "SLIDE_SOURCE_CHANGED"
    assert sdpc._POOL == []


def test_stalled_child_cannot_block_writing_request_forever(reader_runtime, monkeypatch):
    path, _ = reader_runtime
    original_popen = sdpc.subprocess.Popen
    # Preserve a real environment while bypassing the optional-library probe.
    import os

    monkeypatch.setattr(sdpc, "_worker_environment", lambda *a, **kw: os.environ.copy())
    monkeypatch.setattr(
        sdpc.subprocess,
        "Popen",
        lambda command, **kwargs: original_popen(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            **kwargs,
        ),
    )
    monkeypatch.setattr(sdpc, "READER_TIMEOUT", 0.1)
    started = time.monotonic()
    with pytest.raises(StorageError) as error:
        sdpc.read_sdpc(path, max_size=128, region="x" * 30000)
    assert error.value.code == "SLIDE_READER_TIMEOUT"
    assert time.monotonic() - started < 2
    assert sdpc._POOL == []
