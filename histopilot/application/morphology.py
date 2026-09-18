"""Real, bounded feature exploration and frozen-dataset visual quality evidence.

Exploratory indexes are process-local caches, never cohort or predictor state. Every
cache access revalidates its frozen source binding. Reviews are managed separately.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from threading import Lock, Semaphore

import h5py
import numpy as np

from histopilot.application.feature_bundles import FeatureBundleService
from histopilot.application.slide_reviews import SlideReviewService, dataset_rows
from histopilot.domain.features import representation_kind
from histopilot.schemas.morphology import MorphologyIndexRequest
from histopilot.storage.attention_inputs import _dataset, _geometry
from histopilot.storage.filesystem import LocalFilesystem
from histopilot.storage.packed import PackedStoreError, _same_stamp, _source, _stamp
from histopilot.storage.project_lock import StorageError
from histopilot.viewer.slide_images import allowed_file, inspect_slide, render_slide

MAX_INDEX_VALUES = 8_000_000  # At most 32 MB of float32 patch features per index.
MAX_PATCHES = 8192
MAX_COORDINATES = 2_000_000
INDEX_SECONDS = 20
_INDEXES = OrderedDict()
_CACHE_LOCK = Lock()
_BUILD_SLOT = Semaphore(1)


def _error(message, code="MORPHOLOGY_INVALID", status=422):
    return StorageError(message, code, status)


def sample_rows(count, limit):
    """Fixed, distinct whole-bag positions; no replacement and no hidden randomness."""
    return np.linspace(0, count - 1, min(count, limit), dtype=np.int64)


def normalized(values):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise _error("Features contain non-finite values. Validate the source again.")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms != 0).astype("float32")


def projection(values):
    """PCA through a small sample Gram matrix, with a deterministic axis sign."""
    centered = values.astype(np.float64) - values.mean(axis=0, dtype=np.float64)
    gram = centered @ centered.T
    eigenvalues, vectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1][:2]
    positive = np.maximum(eigenvalues[order], 0)
    coordinates = vectors[:, order] * np.sqrt(positive)
    for axis in range(coordinates.shape[1]):
        anchor = np.argmax(np.abs(coordinates[:, axis]))
        if coordinates[anchor, axis] < 0:
            coordinates[:, axis] *= -1
    if coordinates.shape[1] == 1:
        coordinates = np.column_stack((coordinates, np.zeros(len(values))))
    total = max(float(np.trace(gram)), 0)
    explained = (positive / total).tolist() if total > 1e-20 else [0.0] * len(order)
    return coordinates.tolist(), [*explained, *([0.0] * (2 - len(explained)))]


class MorphologyService:
    def __init__(self, store, filesystem):
        self.store = store
        self.filesystem = LocalFilesystem((store.folder, *filesystem.roots))
        self.bundles = FeatureBundleService(store, filesystem)

    def records(self, dataset_id):
        try:
            return list(dataset_rows(self.store, dataset_id)[1].values())
        except StorageError as error:
            if error.code != "ARTIFACT_NOT_FOUND":
                raise
            rows = json.loads(self.store.read_artifact(dataset_id, "slides.json"))
            if (
                not isinstance(rows, list)
                or len(rows) > 50000
                or any(
                    not isinstance(row, dict)
                    or not isinstance(row.get("slideId"), str)
                    or not row["slideId"]
                    or not isinstance(row.get("attributes", {}), dict)
                    for row in rows
                )
                or len({row["slideId"] for row in rows}) != len(rows)
            ):
                raise _error(
                    "Legacy frozen dataset records are invalid.", "MORPHOLOGY_DATASET_INVALID"
                )
            return rows

    def slides(self, dataset_id, *, search="", offset=0, limit=100):
        rows = self.records(dataset_id)
        query = search.strip().casefold()
        visible = [
            row
            for row in rows
            if not query or query in f"{row['slideId']} {row.get('patientId') or ''}".casefold()
        ]
        reviews = SlideReviewService(self.store)
        items = []
        for row in visible[offset : offset + limit]:
            review = reviews._read(dataset_id, row["slideId"])
            items.append(
                {
                    "slideId": row["slideId"],
                    "patientId": row.get("patientId"),
                    "hasImage": bool(row.get("slidePath")),
                    "attributes": row.get("attributes", {}),
                    "reviewStatus": review["status"],
                    "reviewRevision": review["revision"],
                }
            )
        return {
            "total": len(rows),
            "matching": len(visible),
            "offset": offset,
            "items": items,
        }

    def _source(self, request):
        rows = self.records(request.datasetId)
        bundle = self.bundles.get(request.featureBundleId)
        if not bundle["current"]:
            raise _error(
                "Feature verification changed. Revalidate the selected bundle.",
                "MORPHOLOGY_SOURCE_CHANGED",
                409,
            )
        feature = self.store.get_configuration(bundle["manifest"]["feature"]["id"])
        source_dataset = feature["manifest"].get("datasetId")
        bundle_dataset = bundle["manifest"].get("datasetId")
        if any(
            value is not None and value != request.datasetId
            for value in (source_dataset, bundle_dataset)
        ):
            raise _error(
                "This feature bundle is scoped to a different frozen dataset. "
                "Use that dataset's bundle or an explicitly verified store-scoped inventory; "
                "matching slide names alone cannot authenticate an image.",
                "MORPHOLOGY_DATASET_MISMATCH",
            )
        if source_dataset is None and feature["manifest"].get("summary", {}).get("scope") not in {
            "store",
            "list",
        }:
            raise _error(
                "The feature inventory has no explicit dataset or store scope.",
                "MORPHOLOGY_SCOPE_REQUIRED",
            )
        files = {item["slideId"]: item for item in feature["manifest"]["files"]}
        if len(files) != len(feature["manifest"]["files"]):
            raise _error("Feature slide identities are ambiguous.")
        return rows, bundle, feature, files

    def _features(self, item, indices, *, coordinates=False):
        """Read only selected rows, with recorded inode/size/time evidence on both ends."""
        path = allowed_file(self.filesystem, item["path"])
        with _source(path, item) as (stream, _):
            with h5py.File(stream, "r") as handle:
                features = _dataset(handle, "features")
                if features.shape != (item["patchCount"], item["dimensions"]):
                    raise _error("The saved feature dimensions changed.")
                values = np.asarray(features[indices], dtype=np.float32)
        coords = None
        if coordinates:
            coord_path = allowed_file(self.filesystem, item.get("coordinatePath", item["path"]))
            expected = item if coord_path == path else item.get("coordinateFile", {})
            with _source(coord_path, expected) as (stream, _):
                with h5py.File(stream, "r") as handle:
                    dataset = _dataset(handle, "coords")
                    if dataset.shape != (item["patchCount"], 2) or dataset.dtype.kind not in "iu":
                        raise _error("Coordinates no longer match the feature row order.")
                    coords = np.asarray(dataset[indices], dtype=np.int64)
        return values, coords

    def build(self, request):
        rows, bundle, feature, files = self._source(request)
        self._require_patch_features(feature)
        dataset = self.store.get_dataset(request.datasetId)
        identity = hashlib.sha256(
            json.dumps(
                {
                    "spec": request.model_dump(),
                    "dataset": dataset["contentHash"],
                    "bundle": bundle["contentHash"],
                    "algorithm": "sampled-pooled-l2-pca-v1",
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        key = (str(self.store.folder), identity)
        with _CACHE_LOCK:
            if key in _INDEXES:
                _INDEXES.move_to_end(key)
                return deepcopy(_INDEXES[key]["public"])
        if not _BUILD_SLOT.acquire(blocking=False):
            raise _error(
                "Another morphology index is being prepared. Try again shortly.",
                "MORPHOLOGY_BUSY",
                409,
            )
        try:
            started = time.monotonic()
            available = sorted(
                (row for row in rows if row["slideId"] in files), key=lambda row: row["slideId"]
            )
            if request.slideIds:
                selected = set(request.slideIds)
                if selected - {row["slideId"] for row in available}:
                    raise _error("Every selected slide must belong to this dataset and bundle.")
                available = [row for row in available if row["slideId"] in selected]
            chosen = available[: request.maxSlides]
            if not chosen:
                raise _error("This frozen dataset and feature bundle have no shared slides.")
            sizes = [
                min(files[row["slideId"]]["patchCount"], request.patchesPerSlide) for row in chosen
            ]
            dimensions = {files[row["slideId"]]["dimensions"] for row in chosen}
            if len(dimensions) != 1:
                raise _error("Selected feature dimensions differ.")
            dimension = next(iter(dimensions))
            if sum(sizes) > MAX_PATCHES or sum(sizes) * dimension > MAX_INDEX_VALUES:
                raise _error(
                    "This exploration exceeds its bounded index. Reduce slides or patches per slide.",
                    "MORPHOLOGY_INDEX_LIMIT",
                    413,
                )
            pooled, patch_features, patches, points = [], [], [], []
            for row in chosen:
                if time.monotonic() - started > INDEX_SECONDS:
                    raise _error(
                        "Feature reads exceeded 20 seconds. Select fewer slides.",
                        "MORPHOLOGY_INDEX_LIMIT",
                        413,
                    )
                item = files[row["slideId"]]
                indices = sample_rows(item["patchCount"], request.patchesPerSlide)
                values, coords = self._features(item, indices, coordinates=True)
                if not np.isfinite(values).all():
                    raise _error("Non-finite patch features cannot be explored.")
                pooled.append(values.mean(axis=0, dtype=np.float64))
                patch_features.append(normalized(values))
                points.append(
                    {
                        "slideId": row["slideId"],
                        "patientId": row.get("patientId"),
                        "attributes": row.get("attributes", {}),
                        "patchCount": item["patchCount"],
                        "sampledPatches": len(indices),
                        "hasImage": bool(row.get("slidePath")),
                    }
                )
                patches.extend(
                    {
                        "slideId": row["slideId"],
                        "patchIndex": int(index),
                        "x": int(xy[0]),
                        "y": int(xy[1]),
                    }
                    for index, xy in zip(indices, coords, strict=True)
                )
            slide_vectors = normalized(pooled)
            xy, explained = projection(slide_vectors)
            for point, location in zip(points, xy, strict=True):
                point.update(x=location[0], y=location[1])
            # Re-run the exact binding check after reads, before publishing cached evidence.
            self._source(request)
            public = {
                "indexId": identity,
                "datasetId": request.datasetId,
                "featureBundleId": request.featureBundleId,
                "encoderId": feature["manifest"].get("layout", {}).get("encoderId")
                or feature["manifest"].get("spec", {}).get("encoderId"),
                "method": "PCA of L2-normalized means of sampled raw patch features",
                "sampling": "Evenly spaced original patch indices; one-patch samples use index 0",
                "candidateSlides": len(available),
                "indexedSlides": len(points),
                "indexedPatches": len(patches),
                "explainedVariance": explained,
                "points": points,
                "patches": patches,
                "warnings": [
                    "Exploratory morphology only; distance is not diagnostic certainty.",
                    "Nearest neighbors use original feature-space cosine similarity, not PCA distance.",
                    "Patch neighbors search only the displayed deterministic sample.",
                    "The index cache is temporary; rebuild after a service restart.",
                ]
                + (
                    [
                        f"Showing the first {len(points)} of {len(available)} matching slides, sorted by ID."
                    ]
                    if len(points) < len(available)
                    else []
                ),
            }
            entry = {
                "public": public,
                "request": request,
                "slides": slide_vectors,
                "patches": np.concatenate(patch_features),
            }
            with _CACHE_LOCK:
                _INDEXES[key] = entry
                while len(_INDEXES) > 2:
                    _INDEXES.popitem(last=False)
            return deepcopy(public)
        except (OSError, ValueError, KeyError) as error:
            if isinstance(error, StorageError):
                raise
            raise _error(str(error)) from error
        finally:
            _BUILD_SLOT.release()

    def neighbors(self, request):
        key = (str(self.store.folder), request.indexId)
        with _CACHE_LOCK:
            index = _INDEXES.get(key)
        if index is None:
            raise _error(
                "This temporary index expired. Build the projection again.",
                "MORPHOLOGY_INDEX_EXPIRED",
                409,
            )
        self._source(index["request"])
        public = index["public"]
        rows = public["points"] if request.mode == "slide" else public["patches"]
        positions = [
            i
            for i, row in enumerate(rows)
            if row["slideId"] == request.slideId
            and (request.mode == "slide" or row["patchIndex"] == request.patchIndex)
        ]
        if len(positions) != 1:
            raise _error("Select one slide or one sampled patch in this index.")
        anchor = positions[0]
        matrix = index["slides" if request.mode == "slide" else "patches"]
        if not np.any(matrix[anchor]):
            raise _error("A zero feature vector has no defined cosine similarity.")
        scores = matrix.astype(np.float64) @ matrix[anchor].astype(np.float64)
        candidates = [
            i
            for i in range(len(rows))
            if i != anchor
            and np.any(matrix[i])
            and (not request.otherSlidesOnly or rows[i]["slideId"] != request.slideId)
        ]
        candidates.sort(
            key=lambda i: (-scores[i], rows[i]["slideId"], rows[i].get("patchIndex", -1))
        )
        return {
            "scope": "indexed_slides" if request.mode == "slide" else "sampled_patches",
            "metric": "cosine_similarity",
            "candidateCount": len(candidates),
            "items": [
                {**rows[i], "similarity": float(np.clip(scores[i], -1, 1))}
                for i in candidates[: request.limit]
            ],
        }

    def _slide(self, dataset_id, slide_id):
        matches = [row for row in self.records(dataset_id) if row["slideId"] == slide_id]
        if len(matches) != 1:
            raise _error("This slide is not in the selected frozen dataset.", status=404)
        row = matches[0]
        if not row.get("slidePath"):
            raise _error("This frozen slide has no linked image.")
        path = allowed_file(self.filesystem, row["slidePath"])
        actual = _stamp(path.stat())
        authenticated = True
        try:
            inventory = json.loads(self.store.read_artifact(dataset_id, "inventory.json"))
            saved = [item for item in inventory if item.get("path") == str(path)]
            if len(saved) != 1:
                raise _error("The image is not uniquely bound to the frozen slide inventory.")
            _same_stamp(actual, saved[0], path)
        except StorageError as error:
            if error.code != "ARTIFACT_NOT_FOUND":
                raise
            authenticated = False
            # Legacy imported datasets have no inventory. Their path is still exact,
            # but the UI explicitly does not claim import-time image authentication.
        except PackedStoreError as error:
            raise _error(
                "The slide image changed since dataset import.", "MORPHOLOGY_SLIDE_CHANGED", 409
            ) from error
        return path, actual, authenticated

    def image(self, dataset_id, slide_id, *, max_size=1024, region=None):
        path, before, _ = self._slide(dataset_id, slide_id)
        content = render_slide(path, max_size=max_size, region=region)
        try:
            _same_stamp(_stamp(path.stat()), before, path)
        except PackedStoreError as error:
            raise _error(
                "The slide changed while it was read.", "MORPHOLOGY_SLIDE_CHANGED", 409
            ) from error
        return content

    def quality(self, dataset_id, slide_id, bundle_id=None):
        path, before, authenticated = self._slide(dataset_id, slide_id)
        geometry = inspect_slide(path)
        result = {
            "slideId": slide_id,
            "datasetId": dataset_id,
            **geometry,
            "patches": [],
            "patchCount": None,
            "patchWidth": None,
            "patchHeight": None,
            "coordinateBounds": None,
            "tissueContours": [],
            "artifactRemoval": None,
            "warnings": [],
        }
        if not authenticated:
            result["warnings"].append(
                "This legacy dataset has no saved image inventory. Its exact path is checked, but import-time image identity is unavailable."
            )
        if bundle_id:
            request = MorphologyIndexRequest(datasetId=dataset_id, featureBundleId=bundle_id)
            _, _, feature, files = self._source(request)
            result["featureKind"] = representation_kind(feature["manifest"])
            item = files.get(slide_id)
            if item and result["featureKind"] == "slide":
                result["warnings"].append(
                    "This bundle contains one embedding per slide. Patch coverage and "
                    "patch similarity are unavailable; original-image review remains available."
                )
            elif item:
                count = item["patchCount"]
                if count > MAX_COORDINATES:
                    raise _error("Coverage inspection is limited to 2 million patches.", status=413)
                coord_path = allowed_file(self.filesystem, item.get("coordinatePath", item["path"]))
                expected = (
                    item if str(coord_path) == item["path"] else item.get("coordinateFile", {})
                )
                with _source(coord_path, expected) as (stream, _):
                    with h5py.File(stream, "r") as handle:
                        coords = _dataset(handle, "coords")
                        if coords.shape != (count, 2) or coords.dtype.kind not in "iu":
                            raise _error("Invalid patch coordinates.")
                        values = np.asarray(coords[:], dtype=np.int64)
                if (
                    np.any(values < 0)
                    or np.any(values[:, 0] >= geometry["width"])
                    or np.any(values[:, 1] >= geometry["height"])
                ):
                    raise _error("Patch origins are outside this exact slide image.")
                attrs = {}
                for source_attrs in (
                    item.get("attributes", {}).get("file", {}),
                    item.get("coordinateFile", {}).get("attributes", {}),
                    item.get("attributes", {}).get("coords", {}),
                ):
                    for name, value in source_attrs.items():
                        if name in attrs and attrs[name] != value:
                            raise _error("Feature and coordinate geometry metadata conflict.")
                        attrs[name] = value
                try:
                    width, height = _geometry(attrs, {}, geometry)
                    result.update(patchWidth=width, patchHeight=height)
                    lower, upper = values.min(axis=0), values.max(axis=0)
                    result["coordinateBounds"] = {
                        "x": int(lower[0]),
                        "y": int(lower[1]),
                        "width": min(geometry["width"], float(upper[0]) + width) - int(lower[0]),
                        "height": min(geometry["height"], float(upper[1]) + height) - int(lower[1]),
                    }
                except ValueError as error:
                    result["warnings"].append(f"Patch footprint unavailable: {error}")
                indices = sample_rows(count, 4096)
                result.update(
                    patchCount=count,
                    patches=[
                        {"patchIndex": int(i), "x": int(values[i, 0]), "y": int(values[i, 1])}
                        for i in indices
                    ],
                    coverageSampled=count > len(indices),
                )
                extraction = feature["manifest"].get("sourceExtraction")
                if extraction:
                    result["artifactRemoval"] = (
                        extraction.get("spec", {}).get("options", {}).get("remove_artifacts")
                    )
                    result["tissueContours"], warnings = self._contours(
                        extraction, slide_id, geometry
                    )
                    result["warnings"].extend(warnings)
            else:
                result["warnings"].append("This slide has no features in the selected bundle.")
        if result.get("featureKind") != "slide":
            result["warnings"].append(
                "Patch coverage is sampling geometry, not a tumor or tissue-quality score."
            )
        if not result["tissueContours"]:
            result["warnings"].append("No usable recorded TRIDENT tissue contours are available.")
        try:
            _same_stamp(_stamp(path.stat()), before, path)
        except PackedStoreError as error:
            raise _error(
                "The slide changed while it was read.", "MORPHOLOGY_SLIDE_CHANGED", 409
            ) from error
        return result

    def _contours(self, extraction, slide_id, geometry):
        """Use only a recorded extraction location and bound total GeoJSON complexity."""
        if Path(slide_id).name != slide_id or slide_id in {".", ".."}:
            return [], ["Contours are unavailable for this slide identifier."]
        try:
            job = json.loads(
                self.store._read_file(
                    self.store.folder / "extractions" / extraction["jobId"] / "job.json",
                    8 * 1024 * 1024,
                )
            )
            folder = job.get("outputLayout", {}).get("geojsonDir")
            if not folder:
                return [], []
            path = Path(folder) / f"{slide_id}.geojson"
            if not path.exists():
                return [], []
            path = allowed_file(self.filesystem, str(path))
            with _source(path) as (stream, _):
                content = stream.read(8 * 1024 * 1024 + 1)
            if len(content) > 8 * 1024 * 1024:
                return [], ["Tissue contours exceed the 8 MB display limit."]
            value = json.loads(content)
            features = (
                value.get("features", []) if value.get("type") == "FeatureCollection" else [value]
            )
            polygons, total = [], 0
            for feature in features:
                shape = feature.get("geometry", feature)
                coordinates = shape.get("coordinates", [])
                groups = (
                    [coordinates]
                    if shape.get("type") == "Polygon"
                    else coordinates
                    if shape.get("type") == "MultiPolygon"
                    else []
                )
                for rings in groups:
                    checked = []
                    for ring in rings:
                        total += len(ring)
                        if total > 50000:
                            return [], ["Tissue contours exceed the 50,000-vertex display limit."]
                        if len(ring) < 4 or any(
                            len(xy) < 2
                            or any(
                                not isinstance(v, (int, float)) or not math.isfinite(v)
                                for v in xy[:2]
                            )
                            or not (
                                0 <= xy[0] <= geometry["width"] and 0 <= xy[1] <= geometry["height"]
                            )
                            for xy in ring
                        ):
                            raise ValueError("Contour coordinates are not valid level-0 positions.")
                        checked.append([[xy[0], xy[1]] for xy in ring])
                    polygons.append(checked)
            return polygons, [
                "TRIDENT contours are read from the recorded extraction output; their current bytes are not part of the frozen feature hash."
            ]
        except (StorageError, OSError, ValueError, KeyError, TypeError) as error:
            return [], [f"Tissue contour display unavailable: {error}"]

    @staticmethod
    def _require_patch_features(feature):
        if representation_kind(feature["manifest"]) != "patch":
            raise _error(
                "Patch exploration requires patch embeddings with coordinates. "
                "Use original-image review for a slide-embedding bundle.",
                "MORPHOLOGY_PATCH_FEATURES_REQUIRED",
            )

    def patch_region(self, dataset_id, slide_id, bundle_id, patch_index):
        path, _, _ = self._slide(dataset_id, slide_id)
        geometry = inspect_slide(path)
        _, _, feature, files = self._source(
            MorphologyIndexRequest(datasetId=dataset_id, featureBundleId=bundle_id)
        )
        self._require_patch_features(feature)
        item = files.get(slide_id)
        if item is None or patch_index < 0 or patch_index >= item["patchCount"]:
            raise _error("The patch index is outside the selected feature bag.")
        attrs = {}
        for metadata in (
            item.get("attributes", {}).get("file", {}),
            item.get("coordinateFile", {}).get("attributes", {}),
            item.get("attributes", {}).get("coords", {}),
        ):
            for name, value in metadata.items():
                if name in attrs and attrs[name] != value:
                    raise _error("Feature and coordinate geometry metadata conflict.")
                attrs[name] = value
        try:
            width, height = _geometry(attrs, {}, geometry)
        except ValueError as error:
            raise _error(f"Exact patch geometry is unavailable: {error}") from error
        _, coords = self._features(item, np.array([patch_index]), coordinates=True)
        x, y = map(int, coords[0])
        if not (0 <= x < geometry["width"] and 0 <= y < geometry["height"]):
            raise _error("The selected patch is outside this exact slide image.")
        return {
            "x": x,
            "y": y,
            "width": min(width, geometry["width"] - x),
            "height": min(height, geometry["height"] - y),
        }

    def patch_image(self, dataset_id, slide_id, bundle_id, patch_index):
        region = self.patch_region(dataset_id, slide_id, bundle_id, patch_index)
        return self.image(
            dataset_id,
            slide_id,
            max_size=512,
            region=tuple(region[key] for key in ("x", "y", "width", "height")),
        )
