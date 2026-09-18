"""Read-only registration of existing packs against a frozen source inventory.

Header agreement is a preview, never proof of tensor equivalence. Verification
reads actual packed rows and compares their hashes with independently scanned
source tensors, without modifying either representation or trusting old paths.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from histopilot.storage.packed import (
    CHUNK_BYTES,
    FORMAT,
    PackedStoreError,
    _cancel,
    _check_sources,
    _configuration,
    _digest,
    _no_links,
    _progress,
    _read_json,
    _source,
    validate_features,
    validate_pack,
)

PACK_NAMES = (
    "features.bin",
    "coords.bin",
    "index.parquet",
    "meta.json",
    "manifest.json",
    "checksums.json",
)
MAX_INDEX_BYTES = 128 * 1024 * 1024
MAX_SLIDES = 50000


def pack_file_stamps(path: Path) -> dict:
    """Pin the files consumed by a pack reader, without reading tensor payloads."""
    path = Path(path)
    _no_links(path)
    result = {}
    for name in PACK_NAMES:
        target = path / name
        if target.exists() or target.is_symlink():
            with _source(target) as (_, stamp):
                result[name] = stamp
    return result


def _layout(path: Path) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    _no_links(path)
    before = pack_file_stamps(path)
    if ("manifest.json" in before) != ("checksums.json" in before):
        raise PackedStoreError(
            "HistoPilot packs require both manifest.json and checksums.json. "
            "This folder is incomplete; restore the missing file or create a new pack."
        )
    meta = _read_json(path / "meta.json")
    if not isinstance(meta, dict):
        raise PackedStoreError("Pack meta.json must contain a JSON object.")
    for key in ("schema_version", "feat_dim", "coord_dim", "n_slides", "total_patches"):
        if type(meta.get(key)) is not int or meta[key] <= 0:
            raise PackedStoreError(f"Pack metadata requires a positive integer {key}.")
    if meta["schema_version"] != 1 or meta["feat_dtype"] not in {"float16", "float32"}:
        raise PackedStoreError("Only OceanPath v1 float16 or float32 packs are supported.")
    if meta.get("has_coords") is not True or meta["coord_dim"] != 2:
        raise PackedStoreError("Attaching a pack requires one int32 XY coordinate pair per patch.")
    if meta["n_slides"] > MAX_SLIDES or meta["feat_dim"] > 1000000:
        raise PackedStoreError("Pack dimensions or slide count exceed the supported limits.")
    with _source(path / "index.parquet") as (stream, stamp):
        if stamp["sizeBytes"] > MAX_INDEX_BYTES:
            raise PackedStoreError("Pack index exceeds the metadata size limit.")
        parquet = pq.ParquetFile(stream)
        if parquet.metadata.num_rows > MAX_SLIDES:
            raise PackedStoreError("Pack index exceeds the slide count limit.")
        table = parquet.read()
    required = ["slide_id", "offset", "n_patches"]
    if any(name not in table.column_names for name in required):
        raise PackedStoreError("Pack index requires slide_id, offset and n_patches columns.")
    if table.schema.field("slide_id").type != pa.string() or any(
        table.schema.field(name).type != pa.int64() for name in required[1:]
    ):
        raise PackedStoreError("Pack index must use exact string IDs and int64 offsets/counts.")
    records, seen, offset = table.select(required).to_pylist(), set(), 0
    slides = []
    for record in records:
        identity, count = record["slide_id"], record["n_patches"]
        if (
            not isinstance(identity, str)
            or not identity
            or identity in seen
            or type(count) is not int
            or count <= 0
            or record["offset"] != offset
        ):
            raise PackedStoreError(
                "Pack index has duplicate IDs, invalid counts or noncontiguous offsets."
            )
        seen.add(identity)
        slides.append({"slideId": identity, "patchCount": count, "offset": offset})
        offset += count
    if len(slides) != meta["n_slides"] or offset != meta["total_patches"]:
        raise PackedStoreError("Pack index counts disagree with meta.json.")
    expected = {
        "features.bin": offset * meta["feat_dim"] * np.dtype(meta["feat_dtype"]).itemsize,
        "coords.bin": offset * 2 * 4,
    }
    for name, size in expected.items():
        if before.get(name, {}).get("sizeBytes") != size:
            raise PackedStoreError(
                f"{name} byte length differs from the declared array: expected {size:,} bytes."
            )
    manifest = validate_pack(path, full=False) if "manifest.json" in before else None
    if before != pack_file_stamps(path):
        raise PackedStoreError("Pack files changed while their structure was inspected.")
    return {"meta": meta, "slides": slides, "manifest": manifest, "packStamps": before}


def _reduces(source_dtype: str, pack_dtype: str) -> bool:
    """A pack may hold the source at lower float precision, verified against the cast source."""
    try:
        source, packed = np.dtype(source_dtype), np.dtype(pack_dtype)
    except TypeError:
        return False
    return (
        source.kind == packed.kind == "f"
        and packed.itemsize < source.itemsize
        and packed.name in {"float16", "float32"}
    )


def inspect_existing_pack(configuration: dict, path: Path) -> dict:
    """Compare metadata with the entire frozen inventory; do not read tensor data."""
    _configuration(configuration)
    try:
        layout = _layout(path)
    except (AttributeError, KeyError, TypeError, ValueError, OSError) as error:
        if isinstance(error, PackedStoreError):
            raise
        raise PackedStoreError(f"Cannot inspect this feature pack: {error}") from error
    meta, files = layout["meta"], configuration["manifest"]["files"]
    source = {item["slideId"]: item for item in files}
    packed = {item["slideId"]: item for item in layout["slides"]}
    missing, extra = sorted(source.keys() - packed.keys()), sorted(packed.keys() - source.keys())
    mismatch = [
        {
            "slideId": identity,
            "sourcePatches": source[identity]["patchCount"],
            "packPatches": packed[identity]["patchCount"],
        }
        for identity in sorted(source.keys() & packed.keys())
        if source[identity]["patchCount"] != packed[identity]["patchCount"]
    ]
    dimensions = {item["dimensions"] for item in files}
    dtypes = {item["dtype"] for item in files}
    source_count = sum(item["patchCount"] for item in files)
    dimension = next(iter(dimensions)) if len(dimensions) == 1 else None
    source_dtype = next(iter(dtypes)) if len(dtypes) == 1 else None
    findings = []

    def finding(code, message):
        findings.append({"severity": "warning", "code": code, "message": message})

    if missing or extra:
        finding(
            "PACK_SLIDE_MISMATCH",
            f"Pack membership differs: {len(missing)} source slides missing and {len(extra)} extra packed slides.",
        )
    if mismatch or source_count != meta["total_patches"]:
        finding(
            "PACK_PATCH_COUNT_MISMATCH",
            f"Patch counts differ on {len(mismatch)} slides: source {source_count:,}, pack {meta['total_patches']:,} patches. Select a pack from the same extraction.",
        )
    if dimensions != {meta["feat_dim"]}:
        finding(
            "PACK_DIMENSION_MISMATCH",
            f"Feature dimensions differ: source {dimension}, pack {meta['feat_dim']}.",
        )
    reduced = (
        len(dtypes) == 1 and source_dtype is not None and _reduces(source_dtype, meta["feat_dtype"])
    )
    if dtypes != {meta["feat_dtype"]} and not reduced:
        finding(
            "PACK_DTYPE_MISMATCH",
            f"Feature precision differs: source {source_dtype}, pack {meta['feat_dtype']}. "
            "A pack may store the source at the same or lower float precision, never higher.",
        )
    summary = {
        "format": FORMAT,
        "formatVariant": "histopilot" if layout["manifest"] else "oceanpath-legacy",
        "sourceDtype": source_dtype,
        "precision": "reduced" if reduced else "exact",
        "slideCount": len(packed),
        "totalPatches": meta["total_patches"],
        "dimensions": meta["feat_dim"],
        "outputDtype": meta["feat_dtype"],
        "featureBytes": layout["packStamps"]["features.bin"]["sizeBytes"],
        "coordinateBytes": layout["packStamps"]["coords.bin"]["sizeBytes"],
        "totalBytes": sum(item["sizeBytes"] for item in layout["packStamps"].values()),
        "expectedFeatureBytes": source_count * dimension * np.dtype(source_dtype).itemsize
        if dimension and source_dtype
        else None,
        "expectedCoordinateBytes": source_count * 8,
        "sourceContainerBytes": sum(item.get("sizeBytes", 0) for item in files),
        "sourcePatchCount": source_count,
        "missingSlideCount": len(missing),
        "extraSlideCount": len(extra),
        "mismatchedSlideCount": len(mismatch),
        "missingSlides": missing[:20],
        "extraSlides": extra[:20],
        "mismatchedSlides": mismatch[:20],
    }
    return {**layout, "summary": summary, "matchesFeatures": not findings, "findings": findings}


def verify_existing_pack(
    configuration: dict,
    path: Path,
    *,
    expected_stamps=None,
    progress=None,
    cancelled=None,
    chunk_bytes=CHUNK_BYTES,
) -> dict:
    """Read all source and packed values; register only exact same-precision data."""
    path = Path(path)
    _cancel(cancelled)
    inspection = inspect_existing_pack(configuration, path)
    stamps = inspection["packStamps"]
    if expected_stamps is not None and stamps != expected_stamps:
        raise PackedStoreError(
            "Pack files changed since the attachment preview. Review the folder again."
        )
    if not inspection["matchesFeatures"]:
        raise PackedStoreError(" ".join(item["message"] for item in inspection["findings"]))
    meta, strong = inspection["meta"], inspection["manifest"]
    reduced = inspection["summary"]["precision"] == "reduced"
    validation = validate_features(
        configuration,
        cast_dtype=meta["feat_dtype"] if reduced else None,
        progress=progress,
        cancelled=cancelled,
        chunk_bytes=chunk_bytes,
    )
    digest_key = "featureTensorCastSha256" if reduced else "featureTensorSha256"
    source = {item["slideId"]: item for item in validation["files"]}
    feature_digest, coord_digest = hashlib.sha256(), hashlib.sha256()
    completed = 0
    with (
        _source(path / "features.bin", stamps["features.bin"]) as (features, _),
        _source(path / "coords.bin", stamps["coords.bin"]) as (coords, _),
    ):
        row_bytes = meta["feat_dim"] * np.dtype(meta["feat_dtype"]).itemsize
        rows = max(1, chunk_bytes // (row_bytes + 32))
        for slide in inspection["slides"]:
            fhash, chash = hashlib.sha256(), hashlib.sha256()
            for start in range(0, slide["patchCount"], rows):
                _cancel(cancelled)
                count = min(rows, slide["patchCount"] - start)
                block, coordinates = features.read(count * row_bytes), coords.read(count * 8)
                if len(block) != count * row_bytes or len(coordinates) != count * 8:
                    raise PackedStoreError("Pack payload ended before its declared rows.")
                feature_digest.update(block)
                coord_digest.update(coordinates)
                fhash.update(block)
                values = np.frombuffer(coordinates, dtype="<i4")
                if np.any(values < 0):
                    raise PackedStoreError(
                        f"{slide['slideId']}: packed coordinates must be nonnegative."
                    )
                chash.update(values.astype("<u8").tobytes())
                completed += count
                _progress(
                    progress, "comparing-pack", completed, meta["total_patches"], slide["slideId"]
                )
            expected = source[slide["slideId"]]
            if fhash.hexdigest() != expected[digest_key]:
                raise PackedStoreError(
                    f"{slide['slideId']}: packed feature values differ from the selected source"
                    + (
                        f" at {meta['feat_dtype']} precision, despite matching shape."
                        if reduced
                        else ", despite matching shape."
                    )
                )
            if chash.hexdigest() != expected["coordinateTensorSha256"]:
                raise PackedStoreError(
                    f"{slide['slideId']}: packed coordinates differ from the selected source."
                )
    hashes = {"features.bin": feature_digest.hexdigest(), "coords.bin": coord_digest.hexdigest()}
    if strong:
        for name, digest in hashes.items():
            if strong["files"][name]["sha256"] != digest:
                raise PackedStoreError(f"{name} checksum differs from the pack manifest.")
        materialization = strong["materializationId"]
    else:
        materialization = "pack-" + _digest(
            {
                "format": FORMAT,
                "slides": inspection["slides"],
                "dimensions": meta["feat_dim"],
                "dtype": meta["feat_dtype"],
                "payloads": hashes,
            }
        )
    _check_sources(validation)
    if stamps != pack_file_stamps(path):
        raise PackedStoreError("Pack files changed during content verification.")
    _cancel(cancelled)
    identity = "pack-" + _digest(
        {
            "featureSetId": configuration["id"],
            "materializationId": materialization,
            "path": str(path),
            "packStamps": stamps,
        }
    )
    artifact = {
        "id": identity,
        "materializationId": materialization,
        "featureSetId": configuration["id"],
        "sourceFeatureSetId": strong.get("featureSetId") if strong else None,
        "sourceContentHash": validation["sourceContentHash"],
        "originalSourceContentHash": strong.get("sourceContentHash")
        if strong
        else meta.get("source_inventory_sha256"),
        "sourceBindingHash": configuration.get("contentHash"),
        "outputPath": str(path),
        "origin": "existing",
        "format": FORMAT,
        "formatVariant": inspection["summary"]["formatVariant"],
        "verification": "exact-cast-source-values" if reduced else "exact-source-values",
        "dtypePolicy": meta["feat_dtype"] if reduced else "preserve",
        "sourceDtype": validation["sourceDtype"],
        "outputDtype": meta["feat_dtype"],
        "preservesSourcePrecision": not reduced,
        "slideCount": meta["n_slides"],
        "totalPatches": meta["total_patches"],
        "dimensions": meta["feat_dim"],
        "validation": validation,
        "packStamps": stamps,
        "slides": inspection["slides"],
        "payloadHashes": hashes,
        "packInspection": inspection["summary"],
    }
    _progress(progress, "complete", completed, meta["total_patches"])
    return artifact
