"""Bounded validation of native TRIDENT outputs without loading embedding tensors."""

import json
import math
import time
from pathlib import Path

from histopilot.application.features import (
    ENCODER_ALIASES,
    MAX_FILES,
    SCAN_SECONDS,
    FeatureService,
    _attributes,
    _regular_file,
)

MAX_GEOJSON_BYTES = 16 * 1024 * 1024


def complete_coverage(report: dict, expected: int) -> bool:
    """Accept historical receipts without the optional flag, but never implicit counts."""
    keys = ("completedSlides", "missingSlides")
    unvalidated = report.get("unvalidatedSlides", 0)
    findings = report.get("findings", [])
    return (
        type(expected) is int
        and expected > 0
        and all(type(report.get(key)) is int and report[key] >= 0 for key in keys)
        and report["completedSlides"] == expected
        and report["missingSlides"] == 0
        and type(unvalidated) is int
        and unvalidated == 0
        and report.get("inspectionComplete", True) is True
        and isinstance(findings, list)
        and all(isinstance(item, dict) and item.get("severity") != "error" for item in findings)
    )


def _dataset(handle, key):
    import h5py

    if not isinstance(handle.get(key, getlink=True), h5py.HardLink):
        raise ValueError(f"{key} must be an embedded HDF5 dataset.")
    value = handle[key]
    if not isinstance(value, h5py.Dataset) or value.is_virtual or value.external:
        raise ValueError(f"{key} cannot use external or virtual storage.")
    return value


def _unlocked(path: Path) -> None:
    lock = Path(f"{path}.lock")
    if lock.exists() or lock.is_symlink():
        raise ValueError(f"TRIDENT lock remains for {path.name}; the output is incomplete.")


def _coord_attributes(attrs: dict, name: str, options: dict) -> None:
    if attrs.get("name") != name:
        raise ValueError("Coordinate slide identity is missing or differs from the expected slide.")
    for key in ("patch_size", "patch_size_level0", "target_magnification", "level0_magnification"):
        value = attrs.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"Coordinate attribute {key} must be a positive number.")
    for key, option, default in (
        ("patch_size", "patch_size", 256),
        ("target_magnification", "mag", 20),
        ("overlap", "overlap", 0),
    ):
        if attrs.get(key) != options.get(option, default):
            raise ValueError(f"Coordinate attribute {key} differs from the extraction settings.")


def _coordinates(path: Path, name: str, options: dict) -> int:
    import h5py

    _unlocked(path)
    with _regular_file(path) as (stream, _stamp):
        with h5py.File(stream, "r") as handle:
            value = _dataset(handle, "coords")
            if len(value.shape) != 2 or value.shape[1] != 2 or value.dtype.kind not in {"i", "u"}:
                raise ValueError("coords must be an integer matrix with shape (N, 2).")
            _coord_attributes(_attributes(value), name, options)
            return value.shape[0]


def _matching_coordinates(feature_path: Path, coordinate_path: Path, started: float) -> None:
    """Compare copied coordinate rows in bounded chunks, without reading embeddings."""
    import h5py
    import numpy as np

    with _regular_file(feature_path) as (features_stream, _):
        with _regular_file(coordinate_path) as (coords_stream, _):
            with (
                h5py.File(features_stream, "r") as features,
                h5py.File(coords_stream, "r") as source,
            ):
                copied, original = _dataset(features, "coords"), _dataset(source, "coords")
                if copied.shape != original.shape:
                    raise ValueError(
                        "Copied feature coordinates differ from the patch coordinate shape."
                    )
                for start in range(0, len(original), 65536):
                    if time.monotonic() - started > SCAN_SECONDS:
                        raise ValueError(
                            "Coordinate comparison exceeded its bounded inspection time."
                        )
                    if not np.array_equal(
                        copied[start : start + 65536], original[start : start + 65536]
                    ):
                        raise ValueError(
                            "Copied feature coordinates differ from the requested patch rows."
                        )


def _source_inputs(job: dict) -> None:
    """Recheck recorded source and checkpoint identities after extraction has finished."""
    inputs = [*job.get("inputFiles", []), *(slide for slide in job["slides"] if slide.get("path"))]
    for source in inputs:
        with _regular_file(Path(source["path"])) as (_, current):
            expected = {**source, "sizeBytes": source.get("sizeBytes", source.get("size"))}
            if any(expected.get(key) != value for key, value in current.items()):
                raise ValueError(f"Extraction input changed: {source['path']}")


def _segmentation(path: Path, contour_path: Path) -> bool:
    _unlocked(contour_path)
    _unlocked(path)
    with _regular_file(path) as (stream, stamp):
        if stamp["sizeBytes"] > MAX_GEOJSON_BYTES:
            raise ValueError("GeoJSON exceeds the bounded inspection size limit.")
        raw = stream.read(MAX_GEOJSON_BYTES + 1)
        if len(raw) > MAX_GEOJSON_BYTES:
            raise ValueError("GeoJSON exceeds the bounded inspection size limit.")
        data = json.loads(raw)
    if not isinstance(data, dict) or data.get("type") != "FeatureCollection":
        raise ValueError("Segmentation must be a GeoJSON FeatureCollection.")
    features = data.get("features")
    if not isinstance(features, list):
        raise ValueError("Segmentation must contain a GeoJSON features array.")
    for feature in features:
        if not isinstance(feature, dict) or feature.get("type") != "Feature":
            raise ValueError("Segmentation contains an invalid GeoJSON feature.")
        geometry = feature.get("geometry")
        if not isinstance(geometry, dict) or geometry.get("type") not in {
            "Polygon",
            "MultiPolygon",
        }:
            raise ValueError("Tissue segmentation must contain Polygon or MultiPolygon geometry.")
        polygons = geometry.get("coordinates")
        if geometry["type"] == "Polygon":
            polygons = [polygons]
        if not isinstance(polygons, list) or not polygons:
            raise ValueError("Segmentation geometry has no polygon coordinates.")
        for polygon in polygons:
            if not isinstance(polygon, list) or not polygon:
                raise ValueError("Segmentation geometry has no polygon rings.")
            for ring in polygon:
                if not isinstance(ring, list) or len(ring) < 4 or ring[0] != ring[-1]:
                    raise ValueError(
                        "Segmentation polygon rings must be closed with at least four points."
                    )
                for point in ring:
                    if (
                        not isinstance(point, list)
                        or len(point) < 2
                        or any(
                            isinstance(value, bool)
                            or not isinstance(value, (int, float))
                            or (isinstance(value, float) and not math.isfinite(value))
                            for value in point
                        )
                    ):
                        raise ValueError("Segmentation coordinates must be finite numeric points.")
    return bool(features)


def _pooled_features(path: Path, name: str) -> None:
    import h5py

    _unlocked(path)
    with _regular_file(path) as (stream, _stamp):
        with h5py.File(stream, "r") as handle:
            value = _dataset(handle, "features")
            if len(value.shape) not in {1, 2} or not all(value.shape) or value.dtype.kind != "f":
                raise ValueError(
                    "Slide features must be a nonempty floating-point vector or matrix."
                )
            attrs = _attributes(value)
            if "name" in attrs and attrs["name"] != name:
                raise ValueError("Slide feature identity differs from the expected slide.")
            if handle.get("coords", getlink=True) is not None:
                coords = _dataset(handle, "coords")
                # TRIDENT squeezes its copied patch coordinates for a single-patch slide.
                if coords.dtype.kind not in {"i", "u"} or not (
                    coords.shape == (2,)
                    or (len(coords.shape) == 2 and coords.shape[0] > 0 and coords.shape[1] == 2)
                ):
                    raise ValueError(
                        "Copied slide-feature coordinates have an invalid shape or dtype."
                    )


def inspect_outputs(job: dict) -> dict:
    """Count slides whose requested stages produced usable, unlocked native artifacts.

    Empty tissue/coordinate outputs are valid completed stages with scientific warnings.
    Empty embeddings cannot count as a completed feature extraction. Copied coordinates
    must match the extraction grid; embedding values remain for full content validation.
    """
    options = job["spec"]["options"]
    task = options.get("task", "all")
    output = Path(job["outputPath"])
    layout = job.get("outputLayout", {})
    coords = Path(
        layout.get("coordsDir")
        or output
        / (
            options.get("coords_dir")
            or f"{options.get('mag', 20):g}x_{options.get('patch_size', 256)}px_{options.get('overlap', 0)}px_overlap"
        )
    )
    encoder = options.get("slide_encoder") or options.get("patch_encoder", "uni_v1")
    feature_dir = Path(
        layout.get("featuresDir")
        or coords
        / (f"{'slide_features' if options.get('slide_encoder') else 'features'}_{encoder}")
    )
    geojson_dir = Path(layout.get("geojsonDir") or output / "contours_geojson")
    contour_dir = Path(layout.get("contoursDir") or output / "contours")
    patches_dir = Path(layout.get("patchesDir") or coords / "patches")
    slides = job["slides"]
    result = {
        "completedSlides": 0,
        "missingSlides": 0,
        "unvalidatedSlides": len(slides),
        "inspectionComplete": not slides,
        "findings": [],
    }
    if task in {"feat", "all"}:
        result["featurePath"] = str(feature_dir)

    def finding(name, code, message, severity="error"):
        result["findings"].append(
            {
                "slideId": name,
                "severity": severity,
                "code": code,
                "message": message,
            }
        )

    if not output.is_absolute() or any(
        not path.is_relative_to(output) or ".." in path.parts
        for path in (coords, feature_dir, geojson_dir, contour_dir, patches_dir)
    ):
        finding(
            None, "INVALID_OUTPUT_LAYOUT", "Artifact paths must stay inside the extraction output."
        )
        return result
    if task not in {"seg", "coords", "feat", "all"}:
        finding(None, "INVALID_OUTPUT_TASK", "Unknown extraction stage.")
        return result
    started = time.monotonic()
    try:
        _source_inputs(job)
    except (OSError, ValueError, KeyError, TypeError) as error:
        finding(None, "EXTRACTION_INPUT_CHANGED", str(error))
        return result
    for index, slide in enumerate(slides):
        if index >= MAX_FILES or time.monotonic() - started > SCAN_SECONDS:
            finding(
                None,
                "ARTIFACT_SCAN_LIMIT",
                "Output inspection reached its limit; remaining slides were not validated.",
            )
            break
        name = slide.get("name") or slide.get("slideId") or slide.get("id")
        result["unvalidatedSlides"] -= 1
        if (
            not isinstance(name, str)
            or not name
            or name in {".", ".."}
            or any(char in name for char in ("/", "\\", "\x00"))
        ):
            finding(None, "INVALID_SLIDE_NAME", "The output slide name is unsafe.")
            result["missingSlides"] += 1
            continue
        try:
            if task in {"seg", "all"}:
                has_tissue = _segmentation(
                    geojson_dir / f"{name}.geojson", contour_dir / f"{name}.jpg"
                )
                if not has_tissue:
                    finding(
                        name, "EMPTY_TISSUE", f"{name}: segmentation contains no tissue.", "warning"
                    )
            count = None
            if task in {"coords", "all"}:
                count = _coordinates(patches_dir / f"{name}_patches.h5", name, options)
                if not count:
                    finding(
                        name,
                        "EMPTY_COORDINATES",
                        f"{name}: coordinate extraction produced zero patches.",
                        "warning",
                    )
            if task in {"feat", "all"}:
                path = feature_dir / f"{name}.h5"
                if options.get("slide_encoder"):
                    _pooled_features(path, name)
                else:
                    _unlocked(path)
                    coordinate_path = patches_dir / f"{name}_patches.h5"
                    _unlocked(coordinate_path)
                    if count is None:
                        count = _coordinates(coordinate_path, name, options)
                    header = FeatureService._header(path, coordinate_path)
                    _coord_attributes(header["attributes"]["coords"], name, options)
                    attrs = header["attributes"]["features"]
                    if "name" in attrs and attrs["name"] != name:
                        raise ValueError("Feature slide identity differs from the expected slide.")
                    actual_encoder = attrs.get("encoder")
                    if actual_encoder is not None and (
                        not isinstance(actual_encoder, str)
                        or ENCODER_ALIASES.get(actual_encoder, actual_encoder)
                        != ENCODER_ALIASES.get(encoder, encoder)
                    ):
                        raise ValueError("Feature encoder differs from the extraction settings.")
                    if count is not None and count != header["patchCount"]:
                        raise ValueError(
                            "Coordinate and feature artifacts contain different patch counts."
                        )
                    if header["coordinateSource"] == "embedded":
                        _matching_coordinates(path, coordinate_path, started)
            result["completedSlides"] += 1
        except (OSError, ValueError, KeyError, RuntimeError, TypeError) as error:
            finding(name, "INVALID_EXTRACTION_ARTIFACT", f"{name}: {error}")
            result["missingSlides"] += 1
    result["inspectionComplete"] = result["unvalidatedSlides"] == 0
    return result
