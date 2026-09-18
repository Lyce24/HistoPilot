"""Bounded feature validation and immutable, independently readable training packs.

Arrays use OceanPath schema v1: little-endian row-major features and int32 XY,
with a Parquet slide index. HistoPilot's manifest and checksums retain scientific
evidence and authenticate the complete artifact without consulting live sources.
This preserves supported tensor precision, not the original HDF5 containers.
``meta.json:source_inventory_sha256`` carries HistoPilot's selected tensor-content
identity, not OceanPath's legacy directory-stat fingerprint. Validate the artifact
with ``validate_pack`` and open OceanPath's reader without ``verify_source``;
OceanPath's default live-source training preflight is not an archive adapter.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
import sys
import tempfile
from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np

CHUNK_BYTES = 16 * 1024 * 1024


def _h5py():
    """Import h5py lazily so the control service never loads HDF5 libraries on startup."""
    import h5py

    return h5py


FORMAT = "oceanpath-packed-v1"
STAMP_KEYS = ("sizeBytes", "mtimeNs", "ctimeNs", "deviceId", "inode")
PAYLOADS = ("features.bin", "coords.bin", "index.parquet", "meta.json")
MAX_METADATA_BYTES = 262144
_LOCATION_KEYS = {
    "path",
    "coordinatepath",
    "coordinatespath",
    "savetodir",
    "savedir",
    "outputdir",
    "outputpath",
    "featuredirectory",
    "coordinatesdirectory",
    "jobdirectory",
    "sourcedir",
    "slidepath",
    "wsipath",
    "wsi",
    "file",
    "filename",
    "name",
    "tag",
    "tags",
    "label",
    "labels",
    "versionlabel",
    "datasetid",
    "featuresetid",
    "mtime",
    "mtime_ns",
    "ctime",
    "sizebytes",
    "mtimens",
    "ctimens",
    "deviceid",
    "inode",
    "createdat",
    "updatedat",
}
_EVIDENCE_KEYS = {
    "encoder",
    "encoder_id",
    "encoderid",
    "model",
    "model_name",
    "checkpoint_sha256",
    "checkpointsha256",
    "patch_size",
    "target_patch_size",
    "target_mag",
    "target_mpp",
    "target_magnification",
    "representation",
    "feature_kind",
    "featurekind",
    "coordinate_space",
    "coordinatespace",
}


class PackedStoreError(ValueError):
    """Invalid, changed, unsupported, or corrupt feature data."""


class PackingCancelled(RuntimeError):
    """The caller cancelled a validation or packing operation."""


class _HashedWriter:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def write(self, data):
        self.stream.write(data)
        self.digest.update(data)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value: object) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _cancel(cancelled):
    if cancelled is not None and cancelled():
        raise PackingCancelled("Feature preparation was cancelled.")


def _progress(callback, stage, completed, total, slide=None, unit="patches"):
    if callback:
        callback(
            {
                "stage": stage,
                "completed": completed,
                "total": total,
                "unit": unit,
                "currentSlide": slide,
                "percent": round(100 * completed / max(total, 1), 2),
            }
        )


def _stamp(info) -> dict:
    return dict(
        zip(
            STAMP_KEYS,
            (info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_dev, info.st_ino),
            strict=True,
        )
    )


def _no_links(path: Path):
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise PackedStoreError(f"Feature paths must not traverse symbolic links: {path}")


def _same_stamp(actual: dict, expected: dict, path: Path):
    if any(actual[key] != expected[key] for key in STAMP_KEYS if key in expected):
        raise PackedStoreError(f"Feature source changed since it was saved: {path}")


@contextmanager
def _source(path: Path, expected: dict | None = None):
    _no_links(path)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise PackedStoreError(f"Feature input must be a regular file: {path}")
            before = _stamp(info)
            _same_stamp(before, expected or {}, path)
            yield stream, before
            _same_stamp(_stamp(os.fstat(stream.fileno())), before, path)
        after = path.lstat()
        if not stat.S_ISREG(after.st_mode):
            raise PackedStoreError(f"Feature source changed during validation: {path}")
        _same_stamp(_stamp(after), before, path)
    except OSError as error:
        raise PackedStoreError(f"Cannot read feature source {path}: {error}") from error


def _stream_hash(stream, cancelled=None, chunk_bytes=CHUNK_BYTES) -> str:
    stream.seek(0)
    digest = hashlib.sha256()
    while True:
        _cancel(cancelled)
        block = stream.read(chunk_bytes)
        if not block:
            return digest.hexdigest()
        digest.update(block)


def _file_evidence(path, cancelled=None, chunk_bytes=CHUNK_BYTES):
    with _source(path) as (stream, stamp):
        return {"path": str(path), **stamp, "sha256": _stream_hash(stream, cancelled, chunk_bytes)}


def _attributes(handle) -> dict:
    def plain(value):
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, (list, tuple)):
            return [plain(item) for item in value]
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float) and math.isfinite(value):
            return value
        raise PackedStoreError("HDF5 attributes must contain finite JSON-compatible values.")

    if len(handle.attrs) > 128:
        raise PackedStoreError("HDF5 attributes exceed the metadata limit.")
    result = {}
    for key in handle.attrs:
        attribute = handle.attrs.get_id(key)
        if (
            attribute.shape is None
            or math.prod(attribute.shape) * attribute.dtype.itemsize > MAX_METADATA_BYTES
        ):
            raise PackedStoreError("HDF5 attributes exceed the metadata limit.")
        result[key] = plain(handle.attrs[key])
        if len(_json_bytes(result)) > MAX_METADATA_BYTES:
            raise PackedStoreError("HDF5 attributes exceed the metadata limit.")
    return result


def _semantic(value):
    """Keep scientific metadata while removing labels, locators, and mutable bookkeeping."""
    if isinstance(value, dict):
        return {
            key: _semantic(item)
            for key, item in sorted(value.items())
            if key.lower().replace("-", "") not in _LOCATION_KEYS
            and not key.lower().endswith(("_path", "_dir", "_directory"))
        }
    if isinstance(value, list):
        return [_semantic(item) for item in value]
    if isinstance(value, str) and (value.startswith("/") or value.startswith("file://")):
        return "<external-locator>"
    return value


def _dataset(handle, key):
    h5py = _h5py()
    if not isinstance(handle.get(key, getlink=True), h5py.HardLink):
        raise PackedStoreError(f"{key} must be an embedded HDF5 dataset.")
    dataset = handle[key]
    if not isinstance(dataset, h5py.Dataset) or dataset.is_virtual or dataset.external:
        raise PackedStoreError(f"{key} cannot use external or virtual storage.")
    return dataset


def _configuration(configuration, *, allow_slide=False):
    try:
        manifest = configuration["manifest"]
        entries = manifest["files"]
        if manifest.get("kind") != "feature" or not entries:
            raise PackedStoreError("Select a saved feature version with at least one slide.")
        # Generic validation also accepts slide vectors. Materialization remains
        # patch-only; its coordinate-bearing format cannot represent slide vectors.
        kind = manifest.get("spec", {}).get("featureKind", "patch")
        if kind not in {"patch", "slide"} or manifest.get("featureKind", kind) != kind:
            raise PackedStoreError("The saved feature inventory has inconsistent feature kinds.")
        if kind != "patch" and not allow_slide:
            raise PackedStoreError(
                "Packing v1 supports patch features only. A slide encoder writes one "
                "embedding per slide, which is already small enough to read directly."
            )
        ids = [entry["slideId"] for entry in entries]
        if any(not isinstance(item, str) or not item for item in ids) or len(ids) != len(set(ids)):
            raise PackedStoreError("The saved feature inventory has empty or duplicate slide IDs.")
        for entry in entries:
            if not Path(entry["path"]).is_absolute():
                raise PackedStoreError("Saved feature paths must be absolute.")
            if entry.get("featureKind", kind) != kind:
                raise PackedStoreError(
                    "The saved feature inventory has inconsistent feature kinds."
                )
        return manifest, sorted(entries, key=lambda item: item["slideId"])
    except (KeyError, TypeError) as error:
        raise PackedStoreError("The saved feature configuration is incomplete.") from error


def _conflicts(attributes, observed):
    for values in attributes.values():
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            if key.lower() not in _EVIDENCE_KEYS or value in (None, ""):
                continue
            normalized = key.lower()
            if normalized in {"encoder", "encoder_id", "encoderid"}:
                normalized = "encoder"
                value = {"uni": "uni_v1", "uni2": "uni_v2"}.get(str(value), value)
            encoded = _json_bytes(value)
            if normalized in observed and observed[normalized] != encoded:
                raise PackedStoreError(f"Selected slides have conflicting {key} evidence.")
            observed[normalized] = encoded


def _scan(
    configuration,
    *,
    dtype=None,
    outputs=None,
    cast_dtype=None,
    progress=None,
    cancelled=None,
    chunk_bytes=CHUNK_BYTES,
):
    """``cast_dtype`` additionally digests each source tensor at that precision.

    An existing pack may store a faithful reduced-precision copy of float32 sources. Comparing
    it needs the digest the source would have had at the pack's precision; the native digest
    stays authoritative for same-precision packs and for source identity.
    """
    if not isinstance(chunk_bytes, int) or chunk_bytes <= 0:
        raise PackedStoreError("chunk_bytes must be a positive integer.")
    manifest, entries = _configuration(configuration)
    total = sum(int(item["patchCount"]) for item in entries)
    if total <= 0:
        raise PackedStoreError("Feature inventory must contain positive patch counts.")
    reports, semantic_slides, provenance, observed = [], [], [], {}
    seen_inodes, source_stamps = set(), {}
    completed, dimensions, source_dtype, output_dtype = 0, None, None, None
    conversion = {"changedValues": 0, "maxAbsoluteError": 0.0, "overflowPolicy": "reject"}
    encoder = manifest.get("layout", {}).get("encoderId")
    if encoder:
        _conflicts({"features": {"encoder": encoder}}, observed)
    for entry in entries:
        _cancel(cancelled)
        slide = entry["slideId"]
        path = Path(entry["path"])
        coords_path = Path(entry.get("coordinatePath") or path)
        with ExitStack() as stack:
            stream, stamp = stack.enter_context(_source(path, entry))
            identity = stamp["deviceId"], stamp["inode"]
            if identity in seen_inodes:
                raise PackedStoreError("Different slide IDs reference the same feature file.")
            seen_inodes.add(identity)
            feature_handle = stack.enter_context(_h5py().File(stream, "r"))
            features = _dataset(feature_handle, "features")
            if len(features.shape) != 2 or not all(features.shape) or features.dtype.kind != "f":
                raise PackedStoreError(f"{slide}: features must be nonempty floating [N,D].")
            count, dim = features.shape
            current_dtype = features.dtype.name
            if (
                count != entry["patchCount"]
                or dim != entry["dimensions"]
                or np.dtype(entry["dtype"]) != features.dtype
            ):
                raise PackedStoreError(f"{slide}: feature header changed since it was saved.")
            if dimensions is not None and (dim != dimensions or current_dtype != source_dtype):
                raise PackedStoreError("Selected features have inconsistent dimensions or dtypes.")
            dimensions, source_dtype = dim, current_dtype
            if outputs is not None:
                if dtype == "preserve":
                    if current_dtype not in {"float16", "float32"}:
                        raise PackedStoreError(
                            "Preserved packs support float16 or float32 sources."
                        )
                    output_dtype = current_dtype
                elif dtype == "float16":
                    output_dtype = "float16"
                else:
                    raise PackedStoreError("Packing dtype must be preserve or float16.")
            if coords_path == path:
                coords_handle, coords_stream, coords_stamp = feature_handle, stream, stamp
            else:
                coords_stream, coords_stamp = stack.enter_context(
                    _source(coords_path, entry.get("coordinateFile", {}))
                )
                coords_handle = stack.enter_context(_h5py().File(coords_stream, "r"))
            coords = _dataset(coords_handle, "coords")
            if coords.shape != (count, 2) or coords.dtype.kind not in {"i", "u"}:
                raise PackedStoreError(f"{slide}: coords must have one integer XY pair per row.")
            coordinate_dtype = coords.dtype.name
            attrs = {
                "file": _attributes(feature_handle),
                "features": _attributes(features),
                "coords": _attributes(coords),
            }
            if "attributes" in entry and entry["attributes"] != attrs:
                raise PackedStoreError(f"{slide}: feature attributes changed since being saved.")
            coord_file_attrs = _attributes(coords_handle)
            if (
                coords_path != path
                and "attributes" in entry.get("coordinateFile", {})
                and entry["coordinateFile"]["attributes"] != coord_file_attrs
            ):
                raise PackedStoreError(f"{slide}: coordinate attributes changed since being saved.")
            for category in ("features", "coords"):
                if attrs[category].get("name", slide) != slide:
                    raise PackedStoreError(f"{slide}: stored slide name does not match its ID.")
            _conflicts({**attrs, "coordinateFile": coord_file_attrs}, observed)
            if any(
                attrs[category].get("feature_kind", "patch") != "patch"
                for category in ("file", "features")
            ):
                raise PackedStoreError("Packing v1 supports patch features only.")
            feature_hash, coord_hash = hashlib.sha256(), hashlib.sha256()
            cast_hash = hashlib.sha256() if cast_dtype is not None else None
            rows = max(1, chunk_bytes // max(dim * features.dtype.itemsize * 4 + 64, 1))
            for start in range(0, count, rows):
                _cancel(cancelled)
                end = min(count, start + rows)
                block, coordinates = features[start:end, :], coords[start:end, :]
                if not np.isfinite(block).all():
                    raise PackedStoreError(f"{slide}: features contain NaN or infinity.")
                if np.any(coordinates < 0):
                    raise PackedStoreError(f"{slide}: coordinates must be nonnegative integer XY.")
                if outputs is not None and np.any(coordinates > np.iinfo(np.int32).max):
                    raise PackedStoreError(
                        f"{slide}: packing requires nonnegative int32 XY coordinates."
                    )
                feature_hash.update(
                    np.ascontiguousarray(block, dtype=features.dtype.newbyteorder("<")).tobytes()
                )
                if cast_hash is not None:
                    reduced = np.dtype(cast_dtype).newbyteorder("<")
                    if np.any(np.abs(block) > np.finfo(reduced).max):
                        raise PackedStoreError(f"{slide}: feature values overflow {cast_dtype}.")
                    cast_hash.update(np.ascontiguousarray(block, dtype=reduced).tobytes())
                coord_hash.update(np.ascontiguousarray(coordinates, dtype="<u8").tobytes())
                if outputs is not None:
                    target = np.dtype(output_dtype).newbyteorder("<")
                    if np.any(np.abs(block) > np.finfo(target).max):
                        raise PackedStoreError(f"{slide}: feature values overflow {output_dtype}.")
                    converted = np.ascontiguousarray(block, dtype=target)
                    if current_dtype != output_dtype:
                        error = np.abs(block.astype(np.float64) - converted.astype(np.float64))
                        conversion["changedValues"] += int(np.count_nonzero(error))
                        conversion["maxAbsoluteError"] = max(
                            conversion["maxAbsoluteError"], float(error.max())
                        )
                    outputs[0].write(converted.tobytes())
                    outputs[1].write(np.ascontiguousarray(coordinates, dtype="<i4").tobytes())
                completed += end - start
                _progress(progress, "packing" if outputs else "validating", completed, total, slide)
            # Hash complete original containers, while they remain pinned to the same inode.
            feature_handle.close()
            if coords_handle is not feature_handle:
                coords_handle.close()
            _progress(progress, "checksumming", completed, total, slide)
            feature_sha = _stream_hash(stream, cancelled, chunk_bytes)
            coord_sha = (
                feature_sha
                if coords_path == path
                else _stream_hash(coords_stream, cancelled, chunk_bytes)
            )
            source_stamps[str(path)] = stamp
            source_stamps[str(coords_path)] = coords_stamp
            report = {
                "slideId": slide,
                "path": str(path),
                **stamp,
                "sha256": feature_sha,
                "coordinatePath": str(coords_path),
                "coordinateFile": {
                    "path": str(coords_path),
                    **coords_stamp,
                    "sha256": coord_sha,
                    "attributes": coord_file_attrs,
                },
                "patchCount": count,
                "dimensions": dim,
                "dtype": current_dtype,
                "coordinateDtype": coordinate_dtype,
                "coordinateSpace": entry.get("coordinateSpace", "unspecified"),
                "attributes": attrs,
                "featureTensorSha256": feature_hash.hexdigest(),
                "coordinateTensorSha256": coord_hash.hexdigest(),
                **(
                    {"featureTensorCastSha256": cast_hash.hexdigest()}
                    if cast_hash is not None
                    else {}
                ),
            }
            reports.append(report)
            semantic_slides.append(
                {
                    "slideId": slide,
                    "patchCount": count,
                    "dimensions": dim,
                    "dtype": current_dtype,
                    "attributes": {key: _semantic(value) for key, value in attrs.items()},
                    "coordinateFileAttributes": _semantic(coord_file_attrs),
                    "coordinateSpace": report["coordinateSpace"],
                    "featureTensorSha256": feature_hash.hexdigest(),
                    "coordinateTensorSha256": coord_hash.hexdigest(),
                }
            )
    for item in manifest.get("provenance", []):
        _cancel(cancelled)
        path = Path(item["path"])
        with _source(path, item) as (stream, stamp):
            if stamp["sizeBytes"] > MAX_METADATA_BYTES:
                raise PackedStoreError("Source provenance exceeds the metadata limit.")
            raw = stream.read(MAX_METADATA_BYTES + 1)
            checksum = hashlib.sha256(raw).hexdigest()
            if checksum != item.get("sha256"):
                raise PackedStoreError(f"Recorded source provenance changed: {path}")
            data = json.loads(raw)
            if "configuration" in item and data != item["configuration"]:
                raise PackedStoreError(f"Recorded source provenance changed: {path}")
            provenance.append(
                {"path": str(path), **stamp, "sha256": checksum, "configuration": data}
            )
            source_stamps[str(path)] = stamp
    semantic = {
        "schemaVersion": 1,
        "featureKind": "patch",
        "slides": semantic_slides,
        "encoderId": encoder,
        "provenance": sorted(
            (_semantic(item["configuration"]) for item in provenance), key=_json_bytes
        ),
    }
    result = {
        "schemaVersion": 1,
        "valid": True,
        "featureSetId": configuration.get("id"),
        "sourceBindingHash": configuration.get("contentHash"),
        "sourceContentHash": _digest(semantic),
        "semanticIdentity": semantic,
        "featureKind": "patch",
        "tensorValidationComplete": True,
        "provenanceComplete": False,
        "provenanceStatus": "Recorded metadata; encoder checkpoint is not authenticated.",
        "files": reports,
        "provenance": provenance,
        "slideCount": len(reports),
        "totalPatches": completed,
        "dimensions": dimensions,
        "sourceDtype": source_dtype,
        "coordinateDtypes": sorted({item["coordinateDtype"] for item in reports}),
        "coordinateValidation": "nonnegative-int32-xy" if outputs else "nonnegative-integer-xy",
        "validationScope": [
            "exact-membership",
            "full-source-checksums",
            "finite-features",
            "integer-coordinate-range",
            "frozen-source-unchanged",
        ],
        "sourceStamps": source_stamps,
    }
    if manifest.get("sourceExtraction") is not None:
        # The frozen application snapshot is evidence, not a claim that a checkpoint
        # has been authenticated. Job IDs/commands/locators do not identify tensors.
        result["sourceExtraction"] = manifest["sourceExtraction"]
    _check_sources(result)
    _cancel(cancelled)
    return result, output_dtype, conversion


def _check_sources(report):
    for locator, stamp in report["sourceStamps"].items():
        path = Path(locator)
        _no_links(path)
        try:
            _same_stamp(_stamp(path.stat()), stamp, path)
        except OSError as error:
            raise PackedStoreError(f"Feature source changed during validation: {path}") from error


def _scan_slide_features(configuration, *, progress=None, cancelled=None, chunk_bytes=CHUNK_BYTES):
    """Validate one vector per slide without inventing coordinates or packing it.

    Both accepted container shapes identify the same tensor. Read columns in
    bounded chunks, and hash complete pinned containers separately from tensor
    identity, just as patch validation does.
    """
    if not isinstance(chunk_bytes, int) or chunk_bytes <= 0:
        raise PackedStoreError("chunk_bytes must be a positive integer.")
    manifest, entries = _configuration(configuration, allow_slide=True)
    reports, semantic_slides, provenance, observed = [], [], [], {}
    seen_inodes, source_stamps = set(), {}
    dimensions, source_dtype = None, None
    encoder = manifest.get("layout", {}).get("encoderId")
    if encoder:
        _conflicts({"features": {"encoder": encoder}}, observed)
    for index, entry in enumerate(entries):
        _cancel(cancelled)
        slide, path = entry["slideId"], Path(entry["path"])
        if entry.get("coordinatePath") or entry.get("coordinateFile"):
            raise PackedStoreError(f"{slide}: slide embeddings cannot declare patch coordinates.")
        with _source(path, entry) as (stream, stamp):
            identity = stamp["deviceId"], stamp["inode"]
            if identity in seen_inodes:
                raise PackedStoreError("Different slide IDs reference the same feature file.")
            seen_inodes.add(identity)
            with _h5py().File(stream, "r") as handle:
                features = _dataset(handle, "features")
                shape = features.shape
                if (
                    len(shape) not in {1, 2}
                    or not all(shape)
                    or (len(shape) == 2 and shape[0] != 1)
                    or features.dtype.kind != "f"
                ):
                    raise PackedStoreError(
                        f"{slide}: a slide embedding must be one nonempty floating-point vector."
                    )
                dim, current_dtype = shape[-1], features.dtype.name
                if (
                    entry["patchCount"] != 1
                    or dim != entry["dimensions"]
                    or np.dtype(entry["dtype"]) != features.dtype
                ):
                    raise PackedStoreError(f"{slide}: feature header changed since it was saved.")
                if dimensions is not None and (dim != dimensions or current_dtype != source_dtype):
                    raise PackedStoreError(
                        "Selected features have inconsistent dimensions or dtypes."
                    )
                dimensions, source_dtype = dim, current_dtype
                attrs = {
                    "file": _attributes(handle),
                    "features": _attributes(features),
                    "coords": {},
                }
                if "attributes" in entry and entry["attributes"] != attrs:
                    raise PackedStoreError(
                        f"{slide}: feature attributes changed since being saved."
                    )
                if attrs["features"].get("name", slide) != slide:
                    raise PackedStoreError(f"{slide}: stored slide name does not match its ID.")
                for category in ("file", "features"):
                    for key in ("feature_kind", "featureKind"):
                        if attrs[category].get(key, "slide") != "slide":
                            raise PackedStoreError(
                                f"{slide}: feature metadata is not a slide embedding."
                            )
                _conflicts(attrs, observed)
                feature_hash = hashlib.sha256()
                columns = max(1, chunk_bytes // max(features.dtype.itemsize * 4, 1))
                for start in range(0, dim, columns):
                    _cancel(cancelled)
                    end = min(dim, start + columns)
                    block = features[start:end] if len(shape) == 1 else features[0, start:end]
                    if not np.isfinite(block).all():
                        raise PackedStoreError(f"{slide}: features contain NaN or infinity.")
                    feature_hash.update(
                        np.ascontiguousarray(
                            block, dtype=features.dtype.newbyteorder("<")
                        ).tobytes()
                    )
                _progress(progress, "validating", index + 1, len(entries), slide, unit="slides")
            _progress(progress, "checksumming", index + 1, len(entries), slide, unit="slides")
            checksum = _stream_hash(stream, cancelled, chunk_bytes)
            source_stamps[str(path)] = stamp
            semantic = {
                "slideId": slide,
                "patchCount": 1,
                "dimensions": dim,
                "dtype": current_dtype,
                "attributes": {key: _semantic(value) for key, value in attrs.items()},
                "coordinateSpace": "slide",
                "featureTensorSha256": feature_hash.hexdigest(),
            }
            semantic_slides.append(semantic)
            reports.append(
                {
                    **semantic,
                    "attributes": attrs,
                    "path": str(path),
                    **stamp,
                    "sha256": checksum,
                    "featureKind": "slide",
                }
            )
    for item in manifest.get("provenance", []):
        _cancel(cancelled)
        path = Path(item["path"])
        with _source(path, item) as (stream, stamp):
            if stamp["sizeBytes"] > MAX_METADATA_BYTES:
                raise PackedStoreError("Source provenance exceeds the metadata limit.")
            raw = stream.read(MAX_METADATA_BYTES + 1)
            checksum = hashlib.sha256(raw).hexdigest()
            if checksum != item.get("sha256"):
                raise PackedStoreError(f"Recorded source provenance changed: {path}")
            data = json.loads(raw)
            if "configuration" in item and data != item["configuration"]:
                raise PackedStoreError(f"Recorded source provenance changed: {path}")
            provenance.append(
                {"path": str(path), **stamp, "sha256": checksum, "configuration": data}
            )
            source_stamps[str(path)] = stamp
    semantic = {
        "schemaVersion": 1,
        "featureKind": "slide",
        "slides": semantic_slides,
        "encoderId": encoder,
        "provenance": sorted(
            (_semantic(item["configuration"]) for item in provenance), key=_json_bytes
        ),
    }
    report = {
        "schemaVersion": 1,
        "valid": True,
        "featureSetId": configuration.get("id"),
        "sourceBindingHash": configuration.get("contentHash"),
        "sourceContentHash": _digest(semantic),
        "semanticIdentity": semantic,
        "featureKind": "slide",
        "tensorValidationComplete": True,
        "provenanceComplete": False,
        "provenanceStatus": "Recorded metadata; encoder checkpoint is not authenticated.",
        "files": reports,
        "provenance": provenance,
        "slideCount": len(reports),
        "totalPatches": len(reports),
        "dimensions": dimensions,
        "sourceDtype": source_dtype,
        "coordinateDtypes": [],
        "coordinateValidation": "not-applicable",
        "validationScope": [
            "exact-membership",
            "full-source-checksums",
            "finite-features",
            "single-slide-vector",
            "frozen-source-unchanged",
        ],
        "sourceStamps": source_stamps,
    }
    if manifest.get("sourceExtraction") is not None:
        report["sourceExtraction"] = manifest["sourceExtraction"]
    return report


def validate_features(
    configuration: dict,
    *,
    cast_dtype=None,
    progress=None,
    cancelled=None,
    chunk_bytes=CHUNK_BYTES,
) -> dict:
    """Validate every selected row and checksum every source, without writing tensors."""
    if configuration.get("manifest", {}).get("spec", {}).get("featureKind", "patch") == "slide":
        if cast_dtype is not None:
            raise PackedStoreError("Slide embeddings are validated at their native precision.")
        report = _scan_slide_features(
            configuration, progress=progress, cancelled=cancelled, chunk_bytes=chunk_bytes
        )
    else:
        report, _, _ = _scan(
            configuration,
            cast_dtype=cast_dtype,
            progress=progress,
            cancelled=cancelled,
            chunk_bytes=chunk_bytes,
        )
    _progress(
        progress,
        "complete",
        report["totalPatches"],
        report["totalPatches"],
        unit="slides" if report["featureKind"] == "slide" else "patches",
    )
    _check_sources(report)
    _cancel(cancelled)
    return report


def _safe_destination(configuration, destination):
    manifest, entries = _configuration(configuration)
    _no_links(destination)
    destination = destination.resolve()
    sources = {Path(item["path"]).resolve().parent for item in entries}
    sources.update(
        Path(item.get("coordinatePath") or item["path"]).resolve().parent for item in entries
    )
    for key in ("featureDirectory", "coordinatesDirectory"):
        value = manifest.get("layout", {}).get(key)
        if value:
            sources.add(Path(value).resolve())
    for key in ("path", "coordinatesPath"):
        value = manifest.get("spec", {}).get(key)
        if value:
            sources.add(Path(value).resolve())
    sources.update(Path(item["path"]).resolve() for item in manifest.get("provenance", []))
    if any(
        destination == source
        or destination.is_relative_to(source)
        or source.is_relative_to(destination)
        for source in sources
    ):
        raise PackedStoreError("Pack destination must not overlap source files or directories.")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise PackedStoreError(
            "Pack destination must be new or empty; existing data is never replaced."
        )
    return destination


def _write_json(path, data):
    with path.open("xb") as stream:
        stream.write(_json_bytes(data) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_dir(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _identity(manifest):
    return {
        "format": manifest["format"],
        "schemaVersion": manifest["schemaVersion"],
        "sourceContentHash": manifest["sourceContentHash"],
        "dtypePolicy": manifest["dtypePolicy"],
        "outputDtype": manifest["outputDtype"],
        "endianness": manifest["endianness"],
        "representation": manifest["representation"],
        "slides": [
            {key: item[key] for key in ("slideId", "offset", "patchCount")}
            for item in manifest["slides"]
        ],
        "payloads": {
            name: manifest["files"][name]["sha256"] for name in ("features.bin", "coords.bin")
        },
    }


def build_pack(
    configuration: dict,
    output_path: Path,
    *,
    dtype="preserve",
    progress=None,
    cancelled=None,
    chunk_bytes=CHUNK_BYTES,
) -> dict:
    """Validate and stream a pack, then durably publish a fresh destination atomically.

    The caller owns cross-process job/destination locks. An existing empty folder
    may be selected; neither a source nor any existing nonempty output is removed.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if dtype not in {"preserve", "float16"}:
        raise PackedStoreError("Packing dtype must be preserve or float16.")
    destination = _safe_destination(configuration, Path(output_path).absolute())
    _cancel(cancelled)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.packing-", dir=destination.parent))
    try:
        with (
            (staging / "features.bin").open("xb") as fout,
            (staging / "coords.bin").open("xb") as cout,
        ):
            feature_writer, coordinate_writer = _HashedWriter(fout), _HashedWriter(cout)
            validation, output_dtype, conversion = _scan(
                configuration,
                dtype=dtype,
                outputs=(feature_writer, coordinate_writer),
                progress=progress,
                cancelled=cancelled,
                chunk_bytes=chunk_bytes,
            )
            for stream in (fout, cout):
                stream.flush()
                os.fsync(stream.fileno())
        offset, slides = 0, []
        for item in validation["files"]:
            slides.append(
                {
                    "slideId": item["slideId"],
                    "offset": offset,
                    "patchCount": item["patchCount"],
                    "attributes": item["attributes"],
                    "coordinateSpace": item["coordinateSpace"],
                }
            )
            offset += item["patchCount"]
        table = pa.table(
            {
                "slide_id": pa.array([item["slideId"] for item in slides], pa.string()),
                "offset": pa.array([item["offset"] for item in slides], pa.int64()),
                "n_patches": pa.array([item["patchCount"] for item in slides], pa.int64()),
            }
        )
        pq.write_table(table, staging / "index.parquet")
        with (staging / "index.parquet").open("rb") as stream:
            os.fsync(stream.fileno())
        source_manifest = configuration["manifest"]
        meta = {
            "schema_version": 1,
            "feat_dim": validation["dimensions"],
            "feat_dtype": output_dtype,
            "coord_dim": 2,
            "n_slides": len(slides),
            "total_patches": offset,
            "has_coords": True,
            "source_dir": source_manifest.get("layout", {}).get("featureDirectory", ""),
            "source_inventory_sha256": validation["sourceContentHash"],
        }
        _write_json(staging / "meta.json", meta)
        files = {}
        for name in PAYLOADS:
            _progress(progress, "verifying-output", len(files), len(PAYLOADS), unit="files")
            evidence = _file_evidence(staging / name, cancelled, chunk_bytes)
            if name in {"features.bin", "coords.bin"}:
                expected_hash = (
                    feature_writer if name == "features.bin" else coordinate_writer
                ).digest.hexdigest()
                if evidence["sha256"] != expected_hash:
                    raise PackedStoreError(
                        f"Pack payload readback differs from streamed values: {name}"
                    )
            files[name] = {
                "path": name,
                "sizeBytes": evidence["sizeBytes"],
                "sha256": evidence["sha256"],
            }
        manifest = {
            "schemaVersion": 1,
            "format": FORMAT,
            "featureKind": "patch",
            "featureSetId": configuration.get("id"),
            "sourceContentHash": validation["sourceContentHash"],
            "sourceBindingHash": configuration.get("contentHash"),
            "writer": "histopilot",
            "writerVersion": "1",
            "dtypePolicy": dtype,
            "sourceDtype": validation["sourceDtype"],
            "outputDtype": output_dtype,
            "endianness": "little",
            "order": "C",
            "coordinateDtype": "int32",
            "coordinateDimensions": 2,
            "representation": "all-patch-rows-all-feature-columns",
            "slideCount": len(slides),
            "totalPatches": offset,
            "dimensions": validation["dimensions"],
            "conversion": conversion,
            "validation": validation,
            "slides": slides,
            "files": files,
            "preservesSourcePrecision": output_dtype == validation["sourceDtype"],
            "preservesOriginalContainers": False,
            "oceanPathCompatibility": "schema-v1-on-little-endian-hosts; verify this artifact independently of live sources",
        }
        if validation.get("sourceExtraction") is not None:
            manifest["sourceExtraction"] = validation["sourceExtraction"]
        manifest["id"] = manifest["materializationId"] = "pack-" + _digest(_identity(manifest))
        manifest["manifestContentHash"] = _digest(manifest)
        _write_json(staging / "manifest.json", manifest)
        checksum_entries = {name: files[name]["sha256"] for name in PAYLOADS}
        checksum_entries["manifest.json"] = _file_evidence(staging / "manifest.json")["sha256"]
        _write_json(staging / "checksums.json", {"algorithm": "sha256", "files": checksum_entries})
        validate_pack(staging, full=False)
        _progress(progress, "publishing", offset, offset)
        _check_sources(validation)
        _cancel(cancelled)
        _safe_destination(configuration, destination)
        _fsync_dir(staging)
        if destination.exists():
            destination.rmdir()  # Fails safely if another writer populated it.
        os.rename(staging, destination)  # POSIX cannot replace a nonempty directory.
        try:
            _fsync_dir(destination.parent)
        except OSError:
            # No ready artifact should remain when durable publication failed.
            os.rename(destination, staging)
            raise
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _read_json(path):
    try:
        with _source(path) as (stream, stamp):
            if stamp["sizeBytes"] > 128 * 1024 * 1024:
                raise PackedStoreError(f"Pack metadata is unreasonably large: {path.name}")
            return json.load(stream)
    except (ValueError, TypeError) as error:
        raise PackedStoreError(f"Invalid pack metadata {path.name}: {error}") from error


def validate_pack(path: Path, *, full=True) -> dict:
    """Verify a self-contained pack; full=True also hashes every payload byte.

    Checksums detect corruption, not a malicious rewrite of every digest. Callers
    pin the returned materialization ID in their scientific artifact registry.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    _no_links(path)
    try:
        manifest = _read_json(path / "manifest.json")
        saved_hash = manifest.get("manifestContentHash")
        if saved_hash != _digest(
            {key: value for key, value in manifest.items() if key != "manifestContentHash"}
        ):
            raise PackedStoreError("Pack manifest checksum does not match its content.")
        if manifest["schemaVersion"] != 1 or manifest["format"] != FORMAT:
            raise PackedStoreError("Unsupported feature pack format.")
        if (
            manifest["endianness"] != "little"
            or manifest["order"] != "C"
            or manifest["outputDtype"] not in {"float16", "float32"}
            or manifest["coordinateDtype"] != "int32"
            or manifest["coordinateDimensions"] != 2
        ):
            raise PackedStoreError("Unsupported feature pack array layout.")
        if manifest["id"] != "pack-" + _digest(_identity(manifest)):
            raise PackedStoreError("Pack materialization identity does not match its content.")
        if manifest["materializationId"] != manifest["id"]:
            raise PackedStoreError("Pack materialization IDs disagree.")
        validation = manifest["validation"]
        if (
            validation["sourceContentHash"] != manifest["sourceContentHash"]
            or _digest(validation["semanticIdentity"]) != manifest["sourceContentHash"]
            or not validation["tensorValidationComplete"]
        ):
            raise PackedStoreError("Pack source content identity is invalid.")
        checksums = _read_json(path / "checksums.json")
        if checksums["algorithm"] != "sha256" or set(checksums["files"]) != {
            *PAYLOADS,
            "manifest.json",
        }:
            raise PackedStoreError("Pack checksum inventory is incomplete.")
        if set(manifest["files"]) != set(PAYLOADS):
            raise PackedStoreError("Pack payload inventory is incomplete.")
        for name in (*PAYLOADS, "manifest.json"):
            payload = path / name
            _no_links(payload)
            if not payload.is_file():
                raise PackedStoreError(f"Pack payload is missing: {name}")
            if name in manifest["files"]:
                record = manifest["files"][name]
                if record["path"] != name or record["sha256"] != checksums["files"][name]:
                    raise PackedStoreError(f"Pack payload checksum metadata disagrees: {name}")
                if payload.stat().st_size != record["sizeBytes"]:
                    raise PackedStoreError(f"Pack payload byte length is incorrect: {name}")
            if full or name in {"manifest.json", "meta.json", "index.parquet"}:
                if _file_evidence(payload)["sha256"] != checksums["files"][name]:
                    raise PackedStoreError(f"Pack payload checksum mismatch: {name}")
        meta = _read_json(path / "meta.json")
        expected_meta = {
            "schema_version": 1,
            "feat_dim": manifest["dimensions"],
            "feat_dtype": manifest["outputDtype"],
            "coord_dim": 2,
            "n_slides": manifest["slideCount"],
            "total_patches": manifest["totalPatches"],
            "has_coords": True,
            "source_inventory_sha256": manifest["sourceContentHash"],
        }
        if any(meta.get(key) != value for key, value in expected_meta.items()):
            raise PackedStoreError("OceanPath metadata disagrees with the HistoPilot manifest.")
        table = pq.read_table(path / "index.parquet")
        if (
            table.column_names != ["slide_id", "offset", "n_patches"]
            or table.schema.field("slide_id").type != pa.string()
            or table.schema.field("offset").type != pa.int64()
            or table.schema.field("n_patches").type != pa.int64()
        ):
            raise PackedStoreError("Pack index must contain string IDs and int64 offsets/counts.")
        records = table.to_pylist()
        if len(records) != manifest["slideCount"] or len(records) != len(manifest["slides"]):
            raise PackedStoreError("Pack slide count disagrees with its index.")
        offset, seen = 0, set()
        for record, slide in zip(records, manifest["slides"], strict=True):
            slide_id, count = record["slide_id"], record["n_patches"]
            if (
                not isinstance(slide_id, str)
                or not slide_id
                or slide_id in seen
                or not isinstance(count, int)
                or count <= 0
                or record["offset"] != offset
                or slide != {**slide, "slideId": slide_id, "offset": offset, "patchCount": count}
            ):
                raise PackedStoreError("Pack index has invalid membership, lengths, or offsets.")
            seen.add(slide_id)
            offset += count
        if not seen or offset != manifest["totalPatches"] or manifest["dimensions"] <= 0:
            raise PackedStoreError("Pack dimensions or total patch counts are invalid.")
        expected_sizes = {
            "features.bin": offset
            * manifest["dimensions"]
            * np.dtype(manifest["outputDtype"]).itemsize,
            "coords.bin": offset * 2 * 4,
        }
        if any(
            manifest["files"][name]["sizeBytes"] != size for name, size in expected_sizes.items()
        ):
            raise PackedStoreError("Pack array byte lengths disagree with its index.")
        return manifest
    except (KeyError, TypeError, OSError, OverflowError) as error:
        raise PackedStoreError(f"Malformed or incomplete feature pack: {error}") from error


class PackedFeatureStore:
    """Read exact slide IDs and selected rows through bounded memory-mapped arrays."""

    def __init__(self, path: Path, *, verify=True):
        self.path = Path(path)
        self.manifest = validate_pack(self.path, full=verify)
        self.slide_ids = tuple(item["slideId"] for item in self.manifest["slides"])
        self._slides = {item["slideId"]: item for item in self.manifest["slides"]}
        self._features = np.memmap(
            self.path / "features.bin",
            mode="r",
            dtype=np.dtype(self.manifest["outputDtype"]).newbyteorder("<"),
            shape=(self.manifest["totalPatches"], self.manifest["dimensions"]),
        )
        self._coords = np.memmap(
            self.path / "coords.bin",
            mode="r",
            dtype="<i4",
            shape=(self.manifest["totalPatches"], 2),
        )

    def _read(self, array, slide_id, rows):
        slide = self._slides[slide_id]
        start, count = slide["offset"], slide["patchCount"]
        values = array[start : start + count]
        if rows is not None:
            if isinstance(rows, slice):
                values = values[rows]
            else:
                selection = np.asarray(rows)
                if selection.ndim != 1 or selection.dtype.kind not in {"i", "u"}:
                    if selection.size == 0 and selection.ndim == 1:
                        selection = selection.astype(np.int64)
                    else:
                        raise PackedStoreError(
                            "Selected rows must be a one-dimensional integer list."
                        )
                if np.any(selection < 0) or np.any(selection >= count):
                    raise PackedStoreError("Selected row index is outside this slide.")
                values = values[selection]
        return np.array(values, copy=True)

    def read_features(self, slide_id: str, rows=None):
        return self._read(self._features, slide_id, rows)

    def read_coords(self, slide_id: str, rows=None):
        return self._read(self._coords, slide_id, rows)

    @property
    def oceanpath_compatible(self):
        return sys.byteorder == "little"

    def close(self):
        for array in (self._features, self._coords):
            array._mmap.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
