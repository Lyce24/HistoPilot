"""Streaming feature/coordinate validation for scientifically aligned attention maps."""

import hashlib
import math
import time
from contextlib import ExitStack
from pathlib import Path

from histopilot.storage.packed import _source, _stamp
from histopilot.storage.project_lock import StorageError, _reject_symlink_components

MAX_PATCHES = 2_000_000
MAX_FEATURE_BYTES = 8 * 1024**3
CHUNK_ROWS = 8192


def file_stamp(path):
    path = Path(path)
    _reject_symlink_components(path)
    if not path.is_file():
        raise ValueError("An interpretation source is missing or no longer a regular file.")
    return {"path": str(path), **_stamp(path.stat())}


def _dataset(handle, key):
    import h5py

    if not isinstance(handle.get(key, getlink=True), h5py.HardLink):
        raise ValueError(f"{key} must be an embedded HDF5 dataset, without linked files.")
    value = handle[key]
    if not isinstance(value, h5py.Dataset) or value.is_virtual or value.external:
        raise ValueError(f"{key} must not use virtual or external HDF5 storage.")
    return value


def _metadata(handle, dataset):
    names = (
        "encoder_id",
        "encoderId",
        "encoder",
        "patch_encoder",
        "coordinate_space",
        "coordinateSpace",
        "patch_size_level0",
        "patch_width_level0",
        "patch_height_level0",
        "patch_size",
        "patch_level",
        "level0_magnification",
        "target_magnification",
        "slide_id",
        "slideId",
    )
    result = {}
    for obj in (handle, dataset):
        for key in names:
            if key in obj.attrs:
                value = obj.attrs[key]
                if hasattr(value, "item"):
                    value = value.item()
                if isinstance(value, bytes):
                    value = value.decode("utf-8")
                if not isinstance(value, (str, int, float)) or isinstance(value, bool):
                    raise ValueError(f"The coordinate or feature attribute {key} must be a scalar.")
                if key in result and result[key] != value:
                    raise ValueError(f"Conflicting HDF5 metadata for {key}.")
                result[key] = value
    return result


def _geometry(attributes, selection, slide):
    for key in ("coordinate_space", "coordinateSpace"):
        if key in attributes and attributes[key] not in {
            "level0",
            "level_0",
            "level-0",
            "level0_pixels",
        }:
            raise ValueError("Coordinate metadata must describe level-0 pixels.")
    inferred_width = attributes.get("patch_width_level0", attributes.get("patch_size_level0"))
    inferred_height = attributes.get("patch_height_level0", attributes.get("patch_size_level0"))
    if inferred_width is None and "patch_size" in attributes:
        if all(key in attributes for key in ("level0_magnification", "target_magnification")):
            if attributes["target_magnification"] <= 0:
                raise ValueError("Target magnification must be positive.")
            inferred_width = inferred_height = (
                attributes["patch_size"]
                * attributes["level0_magnification"]
                / attributes["target_magnification"]
            )
        elif "patch_level" in attributes:
            level = attributes["patch_level"]
            downsamples = slide["levelDownsamples"]
            if type(level) is not int or not 0 <= level < len(downsamples):
                raise ValueError("Patch level is outside this slide's pyramid.")
            inferred_width = inferred_height = attributes["patch_size"] * downsamples[level]
    width, height = selection.get("patchWidthLevel0"), selection.get("patchHeightLevel0")
    for explicit, inferred in ((width, inferred_width), (height, inferred_height)):
        if (
            explicit is not None
            and inferred is not None
            and not math.isclose(explicit, inferred, rel_tol=1e-6)
        ):
            raise ValueError("Provided patch geometry differs from the coordinate metadata.")
    width, height = width or inferred_width, height or inferred_height
    if any(
        not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 < v <= 1000000
        for v in (width, height)
    ):
        raise ValueError(
            "Enter patch width and height in level-0 pixels; source metadata cannot establish them."
        )
    return float(width), float(height)


def inspect_inputs(selection, slide, contract, *, load=False, deadline=None):
    """Hash every feature and XY row; never subsample or reorder the scientific bag.

    Separate coordinate files with embedded feature coordinates are compared row by
    row. Without embedded coordinates, explicit row-alignment confirmation is retained
    as provenance: equal row counts alone cannot prove matching patch order.
    """
    if selection.get("sourceFormat", "h5") == "packed":
        from histopilot.storage.attention_packs import inspect_packed_inputs

        return inspect_packed_inputs(selection, slide, contract, load=load, deadline=deadline)
    if selection.get("sourceFormat", "h5") != "h5":
        raise ValueError("Select HDF5 patch features or a verified feature pack.")
    import h5py
    import numpy as np

    if selection.get("confirmRowAlignment") is not True:
        raise ValueError(
            "Confirm that feature and coordinate rows describe the same patches in the same order."
        )
    feature_path = Path(selection["featurePath"])
    coordinate_path = Path(selection.get("coordinatesPath") or feature_path)
    with ExitStack() as stack:
        streams = {}
        for path in dict.fromkeys((feature_path, coordinate_path)):
            if Path(f"{path}.lock").exists() or Path(f"{path}.lock").is_symlink():
                raise ValueError(
                    "A feature or coordinate extraction lock remains; wait for extraction to complete."
                )
            streams[path] = stack.enter_context(_source(path))
        handles = {
            path: stack.enter_context(h5py.File(stream[0], "r")) for path, stream in streams.items()
        }
        features = _dataset(handles[feature_path], selection["featureKey"])
        coordinates = _dataset(handles[coordinate_path], selection["coordinatesKey"])
        if (
            features.ndim != 2
            or not 0 < features.shape[0] <= MAX_PATCHES
            or features.dtype.kind != "f"
            or features.dtype.itemsize not in {2, 4, 8}
            or features.shape[1] != contract["dimensions"]
            or str(features.dtype) != contract["dtype"]
            or features.size * features.dtype.itemsize > MAX_FEATURE_BYTES
        ):
            raise ValueError(
                "Features must be a finite floating-point N × D matrix matching the predictor's exact dimensions and dtype (maximum 2 million patches / 8 GiB)."
            )
        if coordinates.shape != (features.shape[0], 2) or coordinates.dtype.kind not in {"i", "u"}:
            raise ValueError(
                "Coordinates must be an integer N × 2 matrix with exactly one XY row per feature row."
            )
        feature_attributes = _metadata(handles[feature_path], features)
        coordinate_attributes = _metadata(handles[coordinate_path], coordinates)
        for key in ("encoder_id", "encoderId", "encoder", "patch_encoder"):
            if key in feature_attributes and feature_attributes[key] != contract["encoderId"]:
                raise ValueError(
                    "The feature encoder metadata differs from the selected predictor."
                )
        for attributes in (feature_attributes, coordinate_attributes):
            for key in ("slide_id", "slideId"):
                if key in attributes and attributes[key] not in {
                    selection["slideId"],
                    Path(selection["slidePath"]).stem,
                }:
                    raise ValueError("HDF5 slide identity differs from the selected slide.")
        width, height = _geometry(coordinate_attributes, selection, slide)
        copied = None
        if (
            feature_path != coordinate_path
            and handles[feature_path].get(selection["coordinatesKey"], getlink=True) is not None
        ):
            copied = _dataset(handles[feature_path], selection["coordinatesKey"])
            if copied.shape != coordinates.shape or copied.dtype.kind not in {"i", "u"}:
                raise ValueError(
                    "Embedded feature coordinates differ from the selected coordinate rows."
                )
        feature_digest, coord_digest = hashlib.sha256(), hashlib.sha256()
        loaded_features, loaded_coords = [], []
        for start in range(0, len(features), CHUNK_ROWS):
            if deadline is not None and time.monotonic() > deadline:
                raise ValueError(
                    "Review exceeded its 45-second inspection budget. Select fewer slides or use faster local storage."
                )
            stop = start + CHUNK_ROWS
            feature_block, coord_block = features[start:stop], coordinates[start:stop]
            if not np.isfinite(feature_block).all():
                raise ValueError("Feature vectors contain NaN or infinite values.")
            if (
                np.any(coord_block < 0)
                or np.any(coord_block[:, 0] >= slide["width"])
                or np.any(coord_block[:, 1] >= slide["height"])
            ):
                raise ValueError(
                    "Coordinate origins must lie inside the selected slide in level-0 pixels."
                )
            if copied is not None and not np.array_equal(copied[start:stop], coord_block):
                raise ValueError(
                    "Feature and selected coordinate row order differ; attention would be misaligned."
                )
            feature_digest.update(feature_block.tobytes(order="C"))
            coord_digest.update(coord_block.astype("<i8", copy=False).tobytes(order="C"))
            if load:
                loaded_features.append(feature_block.astype(np.float32))
                loaded_coords.append(coord_block.astype(np.int64))
        evidence = {
            "patchCount": len(features),
            "dimensions": features.shape[1],
            "dtype": str(features.dtype),
            "featureSha256": feature_digest.hexdigest(),
            "coordinatesSha256": coord_digest.hexdigest(),
            "patchWidthLevel0": width,
            "patchHeightLevel0": height,
            "coordinateSpace": "level0",
            "alignment": "embedded_verified"
            if feature_path == coordinate_path or copied is not None
            else "user_confirmed",
            "sourceStamps": {
                str(path): {"path": str(path), **stream[1]} for path, stream in streams.items()
            },
        }
    if load:
        return evidence, np.concatenate(loaded_features), np.concatenate(loaded_coords)
    return evidence


def verify_sources(slides):
    for slide in slides:
        if slide.get("sourceFormat") == "packed":
            from histopilot.storage.pack_import import pack_file_stamps

            if pack_file_stamps(Path(slide["packPath"])) != slide.get("packStamps"):
                raise ValueError("The selected feature pack changed after review.")
        expected = slide["slideSource"]
        if file_stamp(expected["path"]) != expected:
            raise ValueError("The selected slide image changed after review.")
        for path, expected in slide["sourceStamps"].items():
            if file_stamp(path) != expected:
                raise ValueError("The selected features or coordinates changed after review.")


def as_input_error(error):
    return StorageError(str(error), "INTERPRETATION_INPUT_INVALID", 422)
