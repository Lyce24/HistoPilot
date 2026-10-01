"""Optimized SDPC reads retain owned pixels and release each native buffer."""

import ctypes
import gc
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from histopilot.adapters.trident import bootstrap


def legacy_reader():
    path = Path(__file__).parent / "fixtures/sdpc_legacy_reader.py"
    spec = importlib.util.spec_from_file_location("legacy_sdpc_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_known_decoder_keeps_pixels_after_native_buffer_disposal():
    module = legacy_reader()
    pixels = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
    allocations, disposed, collections = [], [], []

    def decode(_slide, destination, width, height, x, y, level):
        assert (width, height, x, y, level) == (2, 1, 10, 20, 1)
        buffer = (ctypes.c_uint8 * 6)(*pixels.ravel())
        allocations.append(buffer)
        ctypes.cast(destination, ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)))[0] = ctypes.cast(
            buffer, ctypes.POINTER(ctypes.c_uint8)
        )

    def dispose(pointer):
        disposed.append(ctypes.addressof(pointer.contents))
        ctypes.memset(pointer, 0, 6)

    module.POINTER = ctypes.POINTER
    module.c_uint8 = ctypes.c_uint8
    module.byref = ctypes.byref
    module.so = SimpleNamespace(SqGetRoiRgbOfSpecifyLayer=decode, Dispose=dispose)
    module.Image = SimpleNamespace(fromarray=lambda array: array.copy())
    module.gc = SimpleNamespace(collect=lambda: collections.append(True))
    reader = module.OldSdpc.__new__(module.OldSdpc)
    reader.sdpc = object()
    reader.level_downsamples = (1, 2)
    reader.getRgb = lambda pointer, width, height: np.ctypeslib.as_array(
        pointer, (height, width, 3)
    )
    baseline = reader.read_region((20, 40), 1, (2, 1))
    collect_function, gc_enabled = gc.collect, gc.isenabled()
    optimized = bootstrap._replacement(module.OldSdpc.read_region, "read_region")
    assert optimized is not None
    result = optimized(reader, (20, 40), 1, (2, 1))
    np.testing.assert_array_equal(result, baseline)
    np.testing.assert_array_equal(result, pixels[..., ::-1])
    assert len(disposed) == 2
    assert len(collections) == 1  # only the original reader collects the full heap
    assert gc.collect is collect_function and gc.isenabled() == gc_enabled


def test_metadata_reuses_the_owned_handle():
    module = legacy_reader()
    reader = module.OldSdpc.__new__(module.OldSdpc)
    opened = []
    handle = SimpleNamespace(
        contents=SimpleNamespace(
            picHead=SimpleNamespace(contents=SimpleNamespace(rate=40, scale=0.5))
        )
    )
    reader.readSdpc = lambda path: opened.append(path) or handle
    reader.getLevelCount = lambda: 2
    reader.getLevelDownsamples = lambda: (1, 2)
    reader.getLevelDimensions = lambda: ((100, 200), (50, 100))
    optimized = bootstrap._replacement(module.OldSdpc.__init__, "__init__")
    assert optimized is not None
    optimized(reader, "slide.sdpc")
    assert opened == ["slide.sdpc"]
    assert reader.sdpc is handle
    assert reader.scan_magnification == 40
    assert reader.sampling_rate == 0.5


def test_unknown_implementation_and_opt_out_are_untouched(monkeypatch):
    def unknown_reader(self, location, level, size):
        raise AssertionError("must never be called")

    assert bootstrap._replacement(unknown_reader, "read_region") is None
    monkeypatch.setenv("HISTOPILOT_SDPC_OPTIMIZATIONS", "0")
    monkeypatch.setattr(bootstrap.importlib, "import_module", lambda _: 1 / 0)
    assert bootstrap.optimize_sdpc() == {"status": "disabled", "applied": []}


def test_optional_decoder_absence_does_not_block_other_formats(monkeypatch):
    monkeypatch.delenv("HISTOPILOT_SDPC_OPTIMIZATIONS", raising=False)

    def absent(_):
        raise ImportError("No optional SDPC backend")

    monkeypatch.setattr(bootstrap.importlib, "import_module", absent)
    assert bootstrap.optimize_sdpc()["status"] == "unavailable"


def test_bootstrap_preserves_upstream_argv_and_main_semantics(tmp_path, monkeypatch):
    script = tmp_path / "run_batch_of_slides.py"
    script.write_text(
        "import sys\nassert __name__ == '__main__'\n"
        "assert sys.argv[1:] == ['--wsi_dir', 'a path with spaces']\n"
    )
    monkeypatch.setattr(bootstrap, "optimize_sdpc", lambda: {"status": "fixture"})
    monkeypatch.setattr(sys, "argv", ["bootstrap.py"])
    monkeypatch.setattr(sys, "path", list(sys.path))
    bootstrap.main([str(script), "--wsi_dir", "a path with spaces"])
