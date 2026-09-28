"""Bounded warm native reader, invoked by filename in the slide-reader Python.

The API imports no native decoder. A child retains one slide briefly, exits on
EOF/idle/request/age limits, and is killed by its parent on decode timeout.
"""

import json
import os
import select
import sys
import time
from contextlib import ExitStack
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from histopilot.storage.project_lock import StorageError, _reject_symlink_components  # noqa: E402
from histopilot.viewer.slide_images import (  # noqa: E402
    _inspect_open_slide,
    _open,
    _render_open_slide,
)


def _stamp(path):
    _reject_symlink_components(path)
    try:
        info = path.stat()
        if not path.is_file():
            raise OSError("Not a regular file")
        return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]
    except OSError as error:
        raise StorageError(
            "The slide changed or is unavailable.", "SLIDE_SOURCE_CHANGED", 409
        ) from error


def main():
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (24 * 1024**2, 24 * 1024**2))
        resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    except (ImportError, OSError, ValueError):
        pass
    # Reserve a private acknowledgement descriptor, then silence fd 1 before
    # native imports. Vendor printf/logging must never corrupt the IPC stream.
    acknowledgement = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    with open(os.devnull, "wb") as sink:
        os.dup2(sink.fileno(), sys.stdout.fileno())
    slide = None
    decoder = None
    identity = None
    contexts = ExitStack()
    started = time.monotonic()
    idle_timeout, max_age, max_requests = 60, 300, 128
    count = 0
    try:
        while count < max_requests and time.monotonic() - started < max_age:
            remaining = min(idle_timeout, max_age - (time.monotonic() - started))
            if remaining <= 0 or not select.select([sys.stdin], [], [], remaining)[0]:
                break
            raw = sys.stdin.buffer.readline(32769)
            if not raw or len(raw) > 32768 or not raw.endswith(b"\n"):
                break
            request = json.loads(raw)
            folder = Path(request["output"])
            # Limits are supplied by our trusted parent, capped in the child.
            idle_timeout = min(60, max(0.01, float(request["idleTimeout"])))
            max_age = min(300, max(0.01, float(request["maxAge"])))
            max_requests = min(128, max(1, int(request["maxRequests"])))
            count += 1
            result = {"id": request["id"]}
            try:
                path = Path(request["path"])
                stamp = _stamp(path)
                if stamp != request["sourceStamp"]:
                    raise StorageError(
                        "The slide changed before it was read.", "SLIDE_SOURCE_CHANGED", 409
                    )
                reader = request.get("reader", "opensdpc")
                if reader not in {"opensdpc", "openslide"}:
                    raise StorageError("Unknown slide reader.", "SLIDE_VIEW_INVALID", 422)
                current_identity = (str(path), stamp, reader)
                if slide is None:
                    if reader == "opensdpc":
                        try:
                            from opensdpc import OpenSdpc
                        except (ImportError, OSError, RuntimeError) as error:
                            raise StorageError(
                                "OpenSDPC is unavailable.", "SLIDE_VIEWER_UNAVAILABLE", 503
                            ) from error
                        slide = OpenSdpc(str(path))
                        decoder = "opensdpc"
                    else:
                        slide, decoder = contexts.enter_context(_open(path))
                    identity = current_identity
                elif identity != current_identity:
                    raise StorageError(
                        "The slide changed. Reopen it before viewing.", "SLIDE_SOURCE_CHANGED", 409
                    )
                metadata = _inspect_open_slide(slide, decoder)
                if request["maxSize"] is not None:
                    content = _render_open_slide(
                        slide, decoder, max_size=request["maxSize"], region=request["region"]
                    )
                    (folder / "view.png").write_bytes(content)
                if _stamp(path) != stamp:
                    raise StorageError(
                        "The slide changed while it was read.", "SLIDE_SOURCE_CHANGED", 409
                    )
                result["metadata"] = metadata
            except StorageError as error:
                result["error"] = {"code": error.code, "message": str(error)}
            except Exception:
                result["error"] = {
                    "code": "SLIDE_FORMAT_UNSUPPORTED",
                    "message": "The slide or requested region cannot be decoded.",
                }
            (folder / "result.json").write_text(json.dumps(result, allow_nan=False))
            acknowledgement.write(b"OK\n")
            if "error" in result and result["error"]["code"] not in {
                "SLIDE_VIEW_INVALID",
                "SLIDE_VIEW_TOO_LARGE",
            }:
                break
    finally:
        try:
            if slide is not None and decoder == "opensdpc":
                slide.close()
            contexts.close()
        finally:
            acknowledgement.close()


if __name__ == "__main__":
    main()
