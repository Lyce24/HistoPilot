"""Bounded attention viewport reads from authenticated, object-free NumPy arrays."""

import hashlib
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from threading import Lock

from histopilot.storage.attention_inputs import file_stamp
from histopilot.storage.packed import _source
from histopilot.storage.project_lock import StorageError

_VERIFIED = OrderedDict()
_VERIFIED_LOCK = Lock()
MAX_ARRAY_BYTES = 80 * 1024 * 1024
RANK_CHUNK_ROWS = 65536


def coordinate_bounds(slide, minimum_x, minimum_y, maximum_x, maximum_y):
    """Exact coverage of all patch footprints, clipped to the original slide."""
    return {
        "x": float(minimum_x),
        "y": float(minimum_y),
        "width": float(min(slide["width"], maximum_x + slide["patchWidthLevel0"]) - minimum_x),
        "height": float(min(slide["height"], maximum_y + slide["patchHeightLevel0"]) - minimum_y),
    }


@contextmanager
def _attention_array(path, expected, slide):
    """Authenticate every access mode and close the read-only map after use.

    A changed inode, ctime, mtime, size, checksum receipt or path invalidates the
    verification cache. Region filtering operates on the read-only numeric array;
    the API materializes only the requested rows, never millions of JSON objects.
    """
    import numpy as np

    path = Path(path)
    try:
        if expected.get("path") != str(path) or not 0 < expected.get("bytes", 0) <= MAX_ARRAY_BYTES:
            raise ValueError("Attention array receipt is invalid.")
        with _source(path) as (stream, stamp):
            if stamp["sizeBytes"] != expected["bytes"]:
                raise ValueError("Attention array size changed.")
            signature = (str(path), expected["sha256"], tuple(sorted(stamp.items())))
            with _VERIFIED_LOCK:
                verified = signature in _VERIFIED
            if not verified:
                digest = hashlib.sha256()
                while block := stream.read(1024 * 1024):
                    digest.update(block)
                if digest.hexdigest() != expected["sha256"]:
                    raise ValueError("Attention array checksum changed.")
                with _VERIFIED_LOCK:
                    _VERIFIED[signature] = True
                    _VERIFIED.move_to_end(signature)
                    while len(_VERIFIED) > 128:
                        _VERIFIED.popitem(last=False)
            array = np.load(path, mmap_mode="r", allow_pickle=False, max_header_size=4096)
            try:
                if (
                    not isinstance(array, np.memmap)
                    or array.dtype != np.dtype("<f8")
                    or array.shape != (slide["patchCount"], 4)
                ):
                    raise ValueError(
                        "Attention array dimensions or dtype differ from the frozen slide."
                    )
                if file_stamp(path) != {"path": str(path), **stamp}:
                    raise ValueError("Attention array changed while opening its viewport.")
                yield array
            finally:
                mapping = getattr(array, "_mmap", None)
                if mapping is not None:
                    mapping.close()
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise StorageError(
            f"Attention output changed or is invalid: {error}", "INTERPRETATION_RESULT_CHANGED", 409
        ) from error


def _validate_values(values, slide):
    import numpy as np

    if (
        not np.isfinite(values).all()
        or np.any(values < 0)
        or np.any(values[:, 2:] > 1)
        or np.any(values[:, :2] != np.floor(values[:, :2]))
        or np.any(values[:, 0] >= slide["width"])
        or np.any(values[:, 1] >= slide["height"])
    ):
        raise ValueError("Attention array contains invalid coordinates or weights.")


def _patches(array, indices, slide):
    values = array[indices]
    _validate_values(values, slide)
    return [
        {
            "index": int(index),
            "x": int(row[0]),
            "y": int(row[1]),
            "weight": float(row[2]),
            "percentile": float(row[3]),
        }
        for index, row in zip(indices, values, strict=True)
    ]


def attention_page(path, expected, slide, *, offset, limit, region=None):
    import numpy as np

    with _attention_array(path, expected, slide) as array:
        indices = None
        if region:
            x, y, width, height = region
            indices = np.flatnonzero(
                (array[:, 0] < x + width)
                & (array[:, 0] + slide["patchWidthLevel0"] > x)
                & (array[:, 1] < y + height)
                & (array[:, 1] + slide["patchHeightLevel0"] > y)
            )
        total = len(array) if indices is None else len(indices)
        selected = (
            np.arange(min(offset, total), min(offset + limit, total))
            if indices is None
            else indices[offset : offset + limit]
        )
        patches = _patches(array, selected, slide)
    return {"total": total, "offset": offset, "limit": limit, "patches": patches}


def attention_top(path, expected, slide, *, limit):
    """Scan the entire slide with bounded scratch space and deterministic tie order.

    Partition one fixed-size block at a time, keep its strongest k rows, then
    merge at most 2k candidates. Equal weights keep the lowest original indices;
    no viewport, percentile rounding, or display cap affects the ranking.
    """
    import numpy as np

    if type(limit) is not int or not 1 <= limit <= 20:
        raise StorageError("Choose 1–20 attention locations.", "INTERPRETATION_VIEW_INVALID", 422)
    with _attention_array(path, expected, slide) as array:
        best = np.empty(0, dtype=np.int64)
        minimum_x, minimum_y = slide["width"], slide["height"]
        maximum_x = maximum_y = 0
        for start in range(0, len(array), RANK_CHUNK_ROWS):
            block = array[start : start + RANK_CHUNK_ROWS]
            _validate_values(block, slide)
            minimum_x = min(minimum_x, float(block[:, 0].min()))
            minimum_y = min(minimum_y, float(block[:, 1].min()))
            maximum_x = max(maximum_x, float(block[:, 0].max()))
            maximum_y = max(maximum_y, float(block[:, 1].max()))
            weights = block[:, 2]
            count = min(limit, len(block))
            threshold = np.partition(weights, len(block) - count)[len(block) - count]
            stronger = np.flatnonzero(weights > threshold)
            tied = np.flatnonzero(weights == threshold)[: count - len(stronger)]
            candidates = np.concatenate((best, start + stronger, start + tied))
            best = candidates[np.lexsort((candidates, -array[candidates, 2]))[:limit]]
        patches = _patches(array, best, slide)
    return {
        "scope": "whole_slide",
        "total": slide["patchCount"],
        "returned": len(patches),
        "offset": 0,
        "limit": limit,
        "coordinateBounds": coordinate_bounds(slide, minimum_x, minimum_y, maximum_x, maximum_y)
        if len(patches)
        else None,
        "patches": [{**patch, "rank": rank} for rank, patch in enumerate(patches, 1)],
    }
