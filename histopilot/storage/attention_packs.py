"""Exact, source-authenticated patch rows from native and registered OceanPath packs."""

import hashlib
import time
from collections import OrderedDict
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from threading import Lock

from histopilot.storage.pack_import import _layout, pack_file_stamps
from histopilot.storage.packed import PackedFeatureStore, _source

_LAYOUTS = OrderedDict()
_LAYOUT_LOCK = Lock()
_PAYLOADS_VERIFIED = OrderedDict()
_PAYLOAD_LOCK = Lock()


def _deadline(deadline):
    if deadline is not None and time.monotonic() > deadline:
        raise ValueError(
            "Review exceeded its 45-second inspection budget. Select fewer slides or use faster local storage."
        )


def _cached_layout(path):
    stamps = pack_file_stamps(path)
    key = (
        str(path),
        tuple((name, tuple(sorted(stamp.items()))) for name, stamp in sorted(stamps.items())),
    )
    with _LAYOUT_LOCK:
        cached = _LAYOUTS.get(key)
        if cached is not None:
            _LAYOUTS.move_to_end(key)
            return cached
    value = _layout(path)
    if value["packStamps"] != stamps:
        raise ValueError("Feature pack changed while its index was opened.")
    # Large metadata remains bounded by the existing reader, but is not retained
    # indefinitely in the control process after this request completes.
    metadata_bytes = sum(
        stamp["sizeBytes"]
        for name, stamp in stamps.items()
        if name not in {"features.bin", "coords.bin"}
    )
    if metadata_bytes <= 16 * 1024**2:
        with _LAYOUT_LOCK:
            _LAYOUTS[key] = value
            _LAYOUTS.move_to_end(key)
            while len(_LAYOUTS) > 2:
                _LAYOUTS.popitem(last=False)
    return value


def _verify_payloads(path, stamps, expected, *, deadline=None):
    """Authenticate standalone or converted native packs once per complete file stamp."""
    for name in ("features.bin", "coords.bin"):
        digest = expected.get(name)
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("The packed payload checksum evidence is incomplete.")
        key = (str(path / name), digest, tuple(sorted(stamps[name].items())))
        with _PAYLOAD_LOCK:
            if key in _PAYLOADS_VERIFIED:
                _PAYLOADS_VERIFIED.move_to_end(key)
                continue
        with _source(path / name, stamps[name]) as (stream, _):
            actual = hashlib.sha256()
            while block := stream.read(1024 * 1024):
                _deadline(deadline)
                actual.update(block)
            if actual.hexdigest() != digest:
                raise ValueError(f"Packed {name} checksum differs from its verified evidence.")
        with _PAYLOAD_LOCK:
            _PAYLOADS_VERIFIED[key] = True
            while len(_PAYLOADS_VERIFIED) > 128:
                _PAYLOADS_VERIFIED.popitem(last=False)


class _VerifiedRows(PackedFeatureStore):
    """Reuse the existing exact-row reader after one authenticated layout check.

    The caller pins every pack file and has already validated the layout. This
    constructor avoids rehashing and reparsing a shared multi-slide manifest for
    each selected slide, while retaining PackedFeatureStore's row semantics.
    """

    def __init__(self, path, layout):
        import numpy as np

        self.path = path
        self.manifest = layout["manifest"]
        self.slide_ids = tuple(row["slideId"] for row in layout["slides"])
        self._slides = {row["slideId"]: row for row in layout["slides"]}
        meta = layout["meta"]
        self._features = np.memmap(
            path / "features.bin",
            mode="r",
            dtype=np.dtype(meta["feat_dtype"]).newbyteorder("<"),
            shape=(meta["total_patches"], meta["feat_dim"]),
        )
        try:
            self._coords = np.memmap(
                path / "coords.bin", mode="r", dtype="<i4", shape=(meta["total_patches"], 2)
            )
        except BaseException:
            self._features._mmap.close()
            raise


def _attributes(source):
    """Recover patch geometry and encoder metadata saved during source verification."""
    sections = source.get("attributes", {})
    relevant = {
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
    }
    feature_attributes = {}
    coordinate_attributes = {}
    for target, values in (
        (feature_attributes, sections.get("file", {})),
        (feature_attributes, sections.get("features", {})),
        (coordinate_attributes, source.get("coordinateFile", {}).get("attributes", {})),
        (coordinate_attributes, sections.get("coords", {})),
    ):
        for key, value in values.items():
            if key not in relevant:
                continue
            if key in target and target[key] != value:
                raise ValueError(f"Conflicting saved pack source metadata: {key}.")
            target[key] = value
    declared_space = source.get("coordinateSpace")
    if declared_space not in {None, "unspecified"}:
        recorded_space = coordinate_attributes.get("coordinate_space", declared_space)
        level_zero_names = {"level0", "level_0", "level-0", "level0_pixels"}
        if (
            recorded_space != declared_space
            and not {recorded_space, declared_space} <= level_zero_names
        ):
            raise ValueError("Packed coordinate-space evidence is inconsistent.")
        coordinate_attributes["coordinate_space"] = declared_space
    return feature_attributes, coordinate_attributes


def inspect_packed_inputs(selection, slide, contract, *, load=False, deadline=None):
    """Read the exact indexed rows from features.bin and coords.bin, never HDF5.

    Registered packs carry source verification evidence and exact file stamps.
    A native standalone pack authenticates its full payload against its manifest.
    Legacy packs require registered, same-precision source verification evidence.
    Selected tensor hashes, encoder, dtype, coordinates and geometry are rechecked
    in both cases; original HDF5 containers are not needed at inference time.
    """
    import numpy as np

    from histopilot.storage.attention_inputs import (
        CHUNK_ROWS,
        MAX_FEATURE_BYTES,
        MAX_PATCHES,
        _geometry,
    )

    _deadline(deadline)
    if selection.get("coordinatesPath"):
        raise ValueError(
            "Packed attention uses the pack's own coords.bin rows; remove the separate coordinate path."
        )
    path = Path(selection["packPath"])
    if not path.is_absolute():
        raise ValueError("Select an absolute feature pack folder.")
    layout = _cached_layout(path)
    meta, native, stamps = layout["meta"], layout["manifest"], layout["packStamps"]
    supplied = selection.get("packEvidence")
    proof = supplied or native
    if not isinstance(proof, dict):
        raise ValueError(
            "Legacy feature packs require registered, verified source and encoder evidence."
        )
    if supplied and proof.get("packStamps") != stamps:
        raise ValueError("The selected feature pack changed after source verification.")
    if native and proof.get("materializationId") != native["materializationId"]:
        raise ValueError("The selected native pack differs from the registered materialization.")
    if meta["feat_dim"] != contract["dimensions"] or meta["feat_dtype"] != contract["dtype"]:
        raise ValueError(
            "Packed feature dimensions and dtype must match the selected predictor exactly."
        )
    identity = selection.get("packSlideId")
    selected = [row for row in layout["slides"] if row["slideId"] == identity]
    if len(selected) != 1:
        raise ValueError(f"The selected slide {identity!r} has no exact row in this feature pack.")
    packed_row = selected[0]
    count, dimensions = packed_row["patchCount"], meta["feat_dim"]
    if (
        not 0 < count <= MAX_PATCHES
        or count * dimensions * np.dtype(meta["feat_dtype"]).itemsize > MAX_FEATURE_BYTES
    ):
        raise ValueError(
            "Packed attention is limited to 2 million patches and 8 GiB of feature vectors per slide."
        )
    validation = proof.get("validation", {})
    if (
        validation.get("tensorValidationComplete") is not True
        or not proof.get("sourceContentHash")
        or validation.get("sourceContentHash") != proof["sourceContentHash"]
    ):
        raise ValueError("The feature pack lacks complete source tensor verification evidence.")
    source_rows = [row for row in validation.get("files", []) if row.get("slideId") == identity]
    if (
        len(source_rows) != 1
        or source_rows[0].get("patchCount") != count
        or source_rows[0].get("dimensions") != dimensions
    ):
        raise ValueError("The packed slide differs from its verified source membership.")
    source = source_rows[0]
    feature_attrs, coordinate_attrs = _attributes(source)
    encoders = [validation.get("semanticIdentity", {}).get("encoderId")]
    encoders.extend(
        feature_attrs.get(key) for key in ("encoder_id", "encoderId", "encoder", "patch_encoder")
    )
    if native:
        encoders.append(native["validation"].get("semanticIdentity", {}).get("encoderId"))
    encoders = [value for value in encoders if value is not None]
    if not encoders or any(value != contract["encoderId"] for value in encoders):
        raise ValueError(
            "The packed feature encoder differs from the selected predictor or cannot be verified."
        )
    if not native and (
        proof.get("verification") != "exact-source-values"
        or proof.get("preservesSourcePrecision") is not True
    ):
        raise ValueError(
            "Legacy attention packs require exact-source-values, same-precision verification."
        )
    width, height = _geometry(coordinate_attrs, selection, slide)
    payload_hashes = proof.get("payloadHashes") or {
        name: item["sha256"]
        for name, item in (native or proof).get("files", {}).items()
        if name in {"features.bin", "coords.bin"}
    }
    if set(payload_hashes) != {"features.bin", "coords.bin"}:
        raise ValueError("Packed feature and coordinate checksum evidence is incomplete.")
    if native and any(
        payload_hashes[name] != native["files"][name]["sha256"] for name in payload_hashes
    ):
        raise ValueError("The registered payload hashes differ from the native pack manifest.")
    converted = source.get("dtype") != meta["feat_dtype"]
    if converted and not native:
        raise ValueError("A legacy pack cannot change the verified feature precision.")
    if not supplied or converted:
        _verify_payloads(path, stamps, payload_hashes, deadline=deadline)
    compact = {
        key: proof[key]
        for key in (
            "id",
            "materializationId",
            "sourceContentHash",
            "featureSetId",
            "verification",
            "preservesSourcePrecision",
            "sourceDtype",
            "outputDtype",
            "dimensions",
        )
        if key in proof
    }
    compact.update(
        packStamps=deepcopy(stamps),
        payloadHashes=deepcopy(payload_hashes),
        validation={
            "tensorValidationComplete": True,
            "sourceContentHash": proof["sourceContentHash"],
            "semanticIdentity": {"encoderId": contract["encoderId"]},
            "files": [deepcopy(source)],
        },
    )
    feature_digest, coordinate_digest = hashlib.sha256(), hashlib.sha256()
    features_result = np.empty((count, dimensions), dtype=np.float32) if load else None
    coords_result = np.empty((count, 2), dtype=np.int64) if load else None
    with ExitStack() as stack:
        # Pin all metadata and both payloads while the independent reader maps them.
        for name, stamp in stamps.items():
            stack.enter_context(_source(path / name, stamp))
        reader = _VerifiedRows(path, layout)
        stack.callback(reader.close)
        for start in range(0, count, CHUNK_ROWS):
            _deadline(deadline)
            stop = min(count, start + CHUNK_ROWS)
            block = reader.read_features(identity, slice(start, stop))
            coords = reader.read_coords(identity, slice(start, stop))
            if (
                block.shape != (stop - start, dimensions)
                or coords.shape != (stop - start, 2)
                or not np.isfinite(block).all()
            ):
                raise ValueError(
                    "Packed rows contain invalid dimensions or nonfinite feature vectors."
                )
            if (
                np.any(coords < 0)
                or np.any(coords[:, 0] >= slide["width"])
                or np.any(coords[:, 1] >= slide["height"])
            ):
                raise ValueError(
                    "Packed coordinate origins must lie inside the selected slide in level-0 pixels."
                )
            feature_digest.update(
                block.astype(np.dtype(meta["feat_dtype"]).newbyteorder("<"), copy=False).tobytes(
                    order="C"
                )
            )
            coordinate_digest.update(coords.astype("<i8", copy=False).tobytes(order="C"))
            if load:
                features_result[start:stop] = block
                coords_result[start:stop] = coords
    if pack_file_stamps(path) != stamps:
        raise ValueError("The selected feature pack changed during attention preparation.")
    if not converted and feature_digest.hexdigest() != source.get("featureTensorSha256"):
        raise ValueError("Packed feature values differ from the verified slide tensor.")
    if coordinate_digest.hexdigest() != source.get("coordinateTensorSha256"):
        raise ValueError("Packed coordinates differ from the verified slide tensor or row order.")
    evidence = {
        "patchCount": count,
        "dimensions": dimensions,
        "dtype": meta["feat_dtype"],
        "featureSha256": feature_digest.hexdigest(),
        "coordinatesSha256": coordinate_digest.hexdigest(),
        "patchWidthLevel0": width,
        "patchHeightLevel0": height,
        "coordinateSpace": "level0",
        "alignment": "packed_verified",
        "packStamps": deepcopy(stamps),
        "packEvidence": compact,
        "packedRow": dict(packed_row),
        "sourceStamps": {
            str(path / name): {"path": str(path / name), **stamp} for name, stamp in stamps.items()
        },
    }
    return (evidence, features_result, coords_result) if load else evidence
