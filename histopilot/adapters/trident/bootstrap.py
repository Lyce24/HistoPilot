"""Worker-local SDPC performance fixes; standalone for TRIDENT's Python 3.10.

The pinned OpenSDPC decoder frees its native tile buffer, then unnecessarily
scans the entire Python heap after every tile. It also opens extra native slide
handles just to read metadata. These transformations preserve the decoder,
pixel copy, native buffer disposal and automatic Python garbage collection.

Only known function implementations are changed, in this worker process.
Unknown/new backends retain upstream behavior. Installed files are never edited.
"""

import ast
import hashlib
import importlib
import inspect
import json
import os
import runpy
import sys
import textwrap
from functools import update_wrapper
from pathlib import Path

_KNOWN = {
    "read_region": "e0a47b7e4c256d96847e8fd92a0ada281e1ffbff6ba828b9196658f89bf15c73",
    "__init__": "2ecdfefb464572a8aca759e76b863ef8bcec2c00f1a95d2a9470fdaba639d147",
    "_lazy_initialize": "feeda851873c5c1d031b376d1979e5c69f11b68459ecd7cb5121b4b667ef9600",
}


def _fingerprint(tree):
    # Python 3.12 added an empty type_params field to FunctionDef. Worker and
    # service interpreters must recognize the same unparameterized function.
    try:
        normalized = ast.dump(tree, include_attributes=False, show_empty=True)
    except TypeError:  # show_empty was added in Python 3.13
        normalized = ast.dump(tree, include_attributes=False)
    normalized = normalized.replace(", type_params=[]", "")
    return hashlib.sha256(normalized.encode()).hexdigest()


class _Optimize(ast.NodeTransformer):
    def __init__(self, method):
        self.method = method
        self.changes = 0

    def visit_Expr(self, node):
        if self.method == "read_region" and ast.dump(node.value) == ast.dump(
            ast.parse("gc.collect()", mode="eval").body
        ):
            self.changes += 1
            return None
        return self.generic_visit(node)

    def visit_Call(self, node):
        expression = ast.unparse(node)
        if self.method == "__init__" and expression == "self.readSdpc(self.sdpcPath)":
            # The first call creates the owned handle. Only metadata reads
            # after that assignment reuse it.
            self.changes += 1
            if self.changes > 1:
                return ast.copy_location(ast.parse("self.sdpc", mode="eval").body, node)
        if self.method == "_lazy_initialize":
            if expression == "self.img.readSdpc(self.slide_path)":
                self.changes += 1
                return ast.copy_location(ast.parse("self.img.sdpc", mode="eval").body, node)
            if expression == "super()":
                # A recompiled top-level method has no implicit __class__ cell.
                return ast.copy_location(
                    ast.parse("super(SDPCWSI, self)", mode="eval").body, node
                )
        return self.generic_visit(node)


def _replacement(function, method):
    """Return a replacement only for the exact supported implementation."""
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    except (OSError, TypeError, SyntaxError):
        return None
    if _fingerprint(tree) != _KNOWN[method]:
        return None
    transform = _Optimize(method)
    tree = transform.visit(tree)
    expected = 3 if method == "__init__" else 1
    if transform.changes != expected:
        return None
    ast.fix_missing_locations(tree)
    namespace = {}
    exec(compile(tree, inspect.getfile(function), "exec"), function.__globals__, namespace)
    return update_wrapper(namespace[function.__name__], function)


def optimize_sdpc():
    """Apply known reader fixes, returning diagnostics for the durable job log."""
    if os.environ.get("HISTOPILOT_SDPC_OPTIMIZATIONS", "1") == "0":
        return {"status": "disabled", "applied": []}
    try:
        backend = importlib.import_module("opensdpc.opensdpc")
    except (ImportError, OSError, RuntimeError) as error:
        # Other slide formats must work without an optional SDPC installation.
        return {"status": "unavailable", "applied": [], "reason": str(error)}
    reader = getattr(backend, "OldSdpc", None)
    if reader is None:
        return {"status": "unsupported", "applied": []}
    if getattr(reader, "_histopilot_optimized", False):
        return {"status": "already-applied", "applied": list(reader._histopilot_optimized)}
    applied = []
    for method, label in (("read_region", "no-per-tile-full-gc"), ("__init__", "reuse-metadata-handle")):
        replacement = _replacement(getattr(reader, method, None), method)
        if replacement is not None:
            setattr(reader, method, replacement)
            applied.append(label)
    if "reuse-metadata-handle" in applied:
        try:
            wsi_module = importlib.import_module("trident.wsi_objects.SDPCWSI")
        except (ImportError, OSError):
            wsi_module = None
        if wsi_module is not None:
            replacement = _replacement(wsi_module.SDPCWSI._lazy_initialize, "_lazy_initialize")
            if replacement is not None:
                wsi_module.SDPCWSI._lazy_initialize = replacement
                applied.append("reuse-mpp-handle")
    if applied:
        reader._histopilot_optimized = tuple(applied)
    return {"status": "applied" if applied else "unsupported", "applied": applied}


def _peak_reserved_bytes():
    """Peak CUDA memory reserved by this process, without importing or initializing torch."""
    torch = sys.modules.get("torch")
    try:
        if torch is None or not torch.cuda.is_initialized():
            return None
        return max(
            int(torch.cuda.max_memory_reserved(device))
            for device in range(torch.cuda.device_count())
        )
    except Exception:  # telemetry never disturbs extraction
        return None


def report_peak_memory(path, interval=20.0):
    """Keep ``path`` holding this process's peak reserved CUDA memory (Task Center sizing).

    A daemon thread samples the counter while TRIDENT runs and once more at exit, so the
    supervisor can record the measured VRAM of an extraction for later requests.
    """
    import atexit
    import tempfile
    import threading

    target = Path(path)
    state = {"peak": 0}

    def write():
        peak = _peak_reserved_bytes()
        if not peak or peak <= state["peak"]:
            return
        state["peak"] = peak
        try:
            descriptor, temporary = tempfile.mkstemp(prefix=".peak-", dir=target.parent)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump({"cudaPeakReservedBytes": peak}, stream)
            os.replace(temporary, target)
        except OSError:
            pass

    stop = threading.Event()

    def loop():
        while not stop.wait(interval):
            write()

    threading.Thread(target=loop, name="histopilot-peak-memory", daemon=True).start()
    atexit.register(write)
    return stop


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or Path(argv[0]).name != "run_batch_of_slides.py":
        raise SystemExit("Usage: bootstrap.py /path/to/run_batch_of_slides.py [TRIDENT arguments]")
    script = Path(argv[0]).resolve(strict=True)
    sys.path.insert(0, str(script.parent))
    if os.environ.get("HISTOPILOT_TRIDENT_PEAK_PATH"):
        report_peak_memory(os.environ["HISTOPILOT_TRIDENT_PEAK_PATH"])
    print("[HistoPilot] SDPC optimization: " + json.dumps(optimize_sdpc()), flush=True)
    sys.argv = [str(script), *argv[1:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
