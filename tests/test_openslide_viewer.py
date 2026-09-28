"""OpenSlide open/read hangs and crashes stay outside the API process."""

import io
import sys
import time

import pytest
from PIL import Image

from histopilot.storage.project_lock import StorageError
from histopilot.viewer import sdpc, slide_images
from histopilot.viewer.image_cache import SLIDE_IMAGES


@pytest.fixture
def reader_runtime(tmp_path, monkeypatch):
    package = tmp_path / "openslide"
    package.mkdir()
    (package / "__init__.py").write_text("""
import os
import time
from pathlib import Path
from PIL import Image
class OpenSlideError(Exception): pass
class OpenSlide:
    dimensions = (1024, 512)
    level_dimensions = [(1024, 512), (256, 128)]
    level_downsamples = [1.0, 4.0]
    @staticmethod
    def detect_format(path): return "test-pyramid"
    def __init__(self, path):
        self.path = Path(path)
        self.mode = self.path.read_text()
        with self.path.with_suffix(".events").open("a") as log: log.write("opened\\n")
        os.write(1, b"native decoder logging must not enter the IPC channel\\n")
        if self.mode == "hang-open": time.sleep(60)
        if self.mode == "crash-open": os._exit(17)
    def get_best_level_for_downsample(self, downsample): return 1 if downsample >= 4 else 0
    def read_region(self, origin, level, size):
        if self.mode == "hang-read": time.sleep(60)
        if self.mode == "crash-read": os._exit(18)
        if self.mode == "change-read": self.path.write_text("blue")
        return Image.new("RGBA", size, "blue" if self.mode == "blue" else "red")
    def close(self):
        with self.path.with_suffix(".events").open("a") as log: log.write("closed\\n")
""")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setenv("HISTOPILOT_SDPC_PYTHON", sys.executable)
    path = tmp_path / "slide.svs"
    path.write_text("red")
    sdpc.close_readers()
    SLIDE_IMAGES.clear()
    yield path
    sdpc.close_readers()
    SLIDE_IMAGES.clear()


def test_openslide_metadata_and_pixels_share_warm_child_without_parent_import(reader_runtime):
    path = reader_runtime
    before = set(sys.modules)
    assert slide_images.inspect_slide(path) == {
        "width": 1024,
        "height": 512,
        "backend": "openslide",
        "levelDownsamples": [1.0, 4.0],
        "coordinateSpace": "level0",
    }
    reader = sdpc._POOL[0]
    content = slide_images.render_slide(path, max_size=128, region=(16, 20, 512, 256))
    with Image.open(io.BytesIO(content)) as image:
        assert image.size == (128, 64)
        assert image.getpixel((20, 20)) == (255, 0, 0)
    assert sdpc._POOL[0] is reader
    assert reader.requests == 2
    assert path.with_suffix(".events").read_text().splitlines() == ["opened"]
    assert not ({"openslide", "opensdpc"} & (set(sys.modules) - before))
    folder = reader.folder
    sdpc.close_readers()
    assert reader.process.poll() is not None
    assert not folder.exists()


@pytest.mark.parametrize("mode", ["hang-open", "hang-read", "crash-open", "crash-read"])
def test_openslide_hang_or_crash_retires_child_and_next_request_recovers(
    reader_runtime, monkeypatch, mode
):
    path = reader_runtime
    path.write_text(mode)
    monkeypatch.setattr(sdpc, "READER_TIMEOUT", 0.5)
    started = time.monotonic()
    with pytest.raises(StorageError) as error:
        slide_images.render_slide(path, max_size=128)
    assert time.monotonic() - started < 3
    assert error.value.code == (
        "SLIDE_READER_TIMEOUT" if mode.startswith("hang") else "SLIDE_READER_FAILED"
    )
    assert error.value.status_code == (504 if mode.startswith("hang") else 422)
    assert sdpc._POOL == []
    assert not SLIDE_IMAGES._entries
    path.write_text("blue")
    monkeypatch.setattr(sdpc, "READER_TIMEOUT", 2)
    content = slide_images.render_slide(path, max_size=128)
    with Image.open(io.BytesIO(content)) as image:
        assert image.getpixel((0, 0)) == (0, 0, 255)
    assert len(sdpc._POOL) == 1


def test_openslide_mutation_during_native_read_never_publishes_pixels(reader_runtime):
    path = reader_runtime
    path.write_text("change-read")
    with pytest.raises(StorageError) as error:
        slide_images.render_slide(path, max_size=128)
    assert error.value.code == "SLIDE_SOURCE_CHANGED"
    assert sdpc._POOL == []
    assert not SLIDE_IMAGES._entries


def test_openslide_changed_source_and_environment_get_fresh_workers(reader_runtime, monkeypatch):
    path = reader_runtime
    slide_images.inspect_slide(path)
    first = sdpc._POOL[0]
    path.write_text("blue")
    slide_images.inspect_slide(path)
    second = sdpc._POOL[0]
    assert second is not first
    assert first.process.poll() is not None
    monkeypatch.setenv("HISTOPILOT_VIEWER_TEST_GENERATION", "new")
    slide_images.inspect_slide(path)
    assert sdpc._POOL[0] is not second
    assert second.process.poll() is not None


def test_backend_identity_is_explicit_and_both_formats_share_two_worker_limit(reader_runtime):
    path = reader_runtime
    package = path.parent / "opensdpc"
    package.mkdir()
    (package / "__init__.py").write_text("from openslide import OpenSlide as OpenSdpc\n")
    assert sdpc.read_openslide(path)["backend"] == "openslide"
    first = sdpc._POOL[0]
    assert sdpc.read_sdpc(path)["backend"] == "opensdpc"
    assert len(sdpc._POOL) == 1  # The other backend for the same source retires its idle worker.
    assert sdpc._POOL[0] is not first
    assert first.process.poll() is not None
    assert sdpc._POOL[0].backend == "opensdpc"
    for index in range(3):
        other = path.with_name(f"other-{index}.svs")
        other.write_text("red")
        assert sdpc.read_openslide(other)["backend"] == "openslide"
        assert len(sdpc._POOL) <= 2


def test_missing_imaging_dependencies_are_actionable_and_do_not_import_in_api(
    reader_runtime,
):
    path = reader_runtime
    package = path.parent / "PIL"
    package.mkdir()
    (package / "__init__.py").write_text("raise ImportError('test missing Pillow')\n")
    with pytest.raises(StorageError) as error:
        slide_images.inspect_slide(path)
    assert error.value.code == "SLIDE_VIEWER_UNAVAILABLE"
    assert "histopilot[imaging]" in str(error.value)
    assert "OpenSlide" in str(error.value)
    assert sdpc._POOL == []
