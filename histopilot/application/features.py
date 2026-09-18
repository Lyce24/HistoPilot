"""Bounded HDF5 header inspection; never load embeddings or a CUDA model."""

import hashlib
import json
import math
import os
import stat
import time
from contextlib import contextmanager
from pathlib import Path

from histopilot.application.slide_lists import (
    SlideListError,
    list_slide_ids,
    read_slide_list_source,
)
from histopilot.schemas.features import FeatureSpec
from histopilot.schemas.slide_lists import SlideListSource
from histopilot.storage.filesystem import FilesystemError, LocalFilesystem
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import ScientificStore
from histopilot.viewer.slide_images import allowed_file

MAX_FILES = 10000
SCAN_SECONDS = 30
MAX_METADATA_BYTES = 262144
ENCODER_ALIASES = {"uni": "uni_v1", "uni2": "uni_v2"}


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _stamp(info: os.stat_result) -> dict:
    return {
        "sizeBytes": info.st_size,
        "mtimeNs": info.st_mtime_ns,
        "ctimeNs": info.st_ctime_ns,
        "deviceId": info.st_dev,
        "inode": info.st_ino,
    }


@contextmanager
def _regular_file(path: Path):
    """Pin the opened inode and reject links or files replaced during inspection."""
    if any(component.is_symlink() for component in (path, *path.parents)):
        raise ValueError("Feature input cannot traverse symbolic links.")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("Feature input is not a regular file.")
        yield stream, _stamp(before)
        opened_after = os.fstat(stream.fileno())
    after = path.lstat()
    if (
        not stat.S_ISREG(after.st_mode)
        or _stamp(before) != _stamp(after)
        or _stamp(before) != _stamp(opened_after)
    ):
        raise ValueError("Feature source changed during inspection; preview again.")


def _attributes(handle) -> dict:
    """Keep TRIDENT's dataset attributes JSON-safe, without reading embedding arrays."""

    def plain(value):
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, (tuple, list)):
            return [plain(item) for item in value]
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float) and math.isfinite(value):
            return value
        raise ValueError("HDF5 attributes must contain finite, JSON-compatible values.")

    if len(handle.attrs) > 128:
        raise ValueError("HDF5 attribute metadata exceeds the inspection limit.")
    attributes = {}
    for key in handle.attrs:
        attribute = handle.attrs.get_id(key)
        if attribute.shape is None:
            raise ValueError("HDF5 attributes cannot contain null dataspaces.")
        if math.prod(attribute.shape) * attribute.dtype.itemsize > MAX_METADATA_BYTES:
            raise ValueError("HDF5 attribute metadata exceeds the inspection limit.")
        attributes[key] = plain(handle.attrs[key])
        if len(json.dumps(attributes).encode()) > MAX_METADATA_BYTES:
            raise ValueError("HDF5 attribute metadata exceeds the inspection limit.")
    return attributes


def _scope(spec: FeatureSpec) -> str:
    """Name the narrowest thing that decided this feature set."""
    if spec.datasetId is not None:
        return "dataset"
    return "list" if spec.slideList is not None or spec.slideListPath is not None else "store"


class FeatureService:
    def __init__(self, store: ScientificStore, filesystem: LocalFilesystem):
        self.store = store
        self.filesystem = filesystem

    def _list_path(self, value: str) -> Path:
        return allowed_file(self.filesystem, value)

    def _directory(self, value: str) -> Path:
        path = Path(value)
        if any(component.is_symlink() for component in (path, *path.parents)):
            raise FilesystemError("Feature folders must not traverse symbolic links.", 403)
        return self.filesystem.directory(value)

    def _layout(self, root: Path, spec: FeatureSpec) -> dict:
        """Recognize a TRIDENT job, coordinate run, or one encoder's feature directory."""
        feature_root = root
        kind = "flat"
        inferred_encoder = None
        selected_encoder = ENCODER_ALIASES.get(spec.encoderId, spec.encoderId)
        # TRIDENT writes patch bags to features_<encoder> and slide embeddings to
        # slide_features_<encoder>, so each kind discovers only its own folders.
        prefix = "slide_features_" if spec.featureKind == "slide" else "features_"
        if spec.layout != "flat":
            candidates: list[Path] = []
            scanned = 0
            started = time.monotonic()
            pending = [(root, 0)]
            if root.name.startswith(prefix):
                candidates = [root]
                pending = []
            while pending:
                folder, depth = pending.pop()
                with os.scandir(folder) as entries:
                    for item in entries:
                        scanned += 1
                        if scanned > MAX_FILES or time.monotonic() - started > SCAN_SECONDS:
                            raise StorageError(
                                "Feature discovery exceeded its limit; select a smaller folder.",
                                "FEATURE_SCAN_LIMIT",
                                413,
                            )
                        if item.is_symlink():
                            raise FilesystemError(
                                "Feature folders must not contain symbolic links.", 403
                            )
                        if item.is_dir(follow_symlinks=False):
                            path = Path(item.path)
                            if item.name.startswith(prefix):
                                candidates.append(path)
                            elif depth == 0:
                                pending.append((path, depth + 1))
            native_candidates = bool(candidates)
            if selected_encoder:
                candidates = [
                    path
                    for path in candidates
                    if ENCODER_ALIASES.get(
                        path.name.removeprefix(prefix), path.name.removeprefix(prefix)
                    )
                    == selected_encoder
                ]
            if len(candidates) > 1:
                raise StorageError(
                    "Multiple TRIDENT feature sets found; choose an encoder and its specific "
                    "feature folder: " + ", ".join(str(path) for path in sorted(candidates)),
                    "FEATURE_LAYOUT_AMBIGUOUS",
                    422,
                )
            if candidates:
                feature_root = self._directory(str(candidates[0]))
                kind = "trident"
                name = feature_root.name.removeprefix(prefix)
                inferred_encoder = ENCODER_ALIASES.get(name, name)
            elif spec.layout == "trident" or root.name.startswith(prefix) or native_candidates:
                raise StorageError(
                    f"No TRIDENT {prefix}<encoder> directory matches the selected encoder.",
                    "FEATURE_LAYOUT_NOT_FOUND",
                    422,
                )
        coords_root = None
        if spec.coordinatesPath:
            coords_root = self._directory(spec.coordinatesPath)
        elif kind == "trident":
            candidate = feature_root.parent / "patches"
            if self.filesystem._contains(candidate) and (
                candidate.exists() or candidate.is_symlink()
            ):
                coords_root = self._directory(str(candidate))
        # Metadata paths come from the selected layout, never from untrusted savetodir attrs.
        job_root = feature_root.parent.parent if kind == "trident" else None
        if job_root is not None and not self.filesystem._contains(job_root):
            job_root = None
        return {
            "kind": kind,
            "featureDirectory": str(feature_root),
            "coordinatesDirectory": str(coords_root) if coords_root else None,
            "jobDirectory": str(job_root) if job_root else None,
            "encoderId": selected_encoder or inferred_encoder,
        }

    def _provenance(self, layout: dict) -> list[dict]:
        if layout["kind"] != "trident":
            return []
        feature_root = Path(layout["featureDirectory"])
        paths = [
            feature_root.parent / "_config_coords.json",
            feature_root.parent
            / f"_config_feats_{feature_root.name.removeprefix('features_')}.json",
        ]
        if layout["jobDirectory"]:
            job_root = Path(layout["jobDirectory"])
            paths.extend(
                [
                    job_root / "_config_segmentation.json",
                    job_root / "manifest.json",
                    job_root / "_run" / "provenance.json",
                ]
            )
        provenance = []
        for path in paths:
            if not path.exists() and not path.is_symlink():
                continue
            if not self.filesystem._contains(path):
                continue
            with _regular_file(path) as (stream, stamp):
                if stamp["sizeBytes"] > MAX_METADATA_BYTES:
                    raise ValueError("TRIDENT configuration exceeds the metadata size limit.")
                raw = stream.read(MAX_METADATA_BYTES + 1)
                if len(raw) > MAX_METADATA_BYTES:
                    raise ValueError("TRIDENT configuration exceeds the metadata size limit.")
                data = json.loads(raw)
                # Reject non-finite JSON numbers before including the source in a preview hash.
                json.dumps(data, allow_nan=False)
                provenance.append(
                    {
                        "path": str(path),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "configuration": data,
                        **stamp,
                    }
                )
        return provenance

    def _files(self, root: Path, spec: FeatureSpec) -> list[Path]:
        found: list[Path] = []
        scanned = 0
        started = time.monotonic()
        pending = [root]
        while pending:
            directory = pending.pop()
            with os.scandir(directory) as entries:
                for item in entries:
                    scanned += 1
                    if scanned > MAX_FILES or time.monotonic() - started > SCAN_SECONDS:
                        raise StorageError(
                            "Feature discovery exceeded its limit; select a smaller folder.",
                            "FEATURE_SCAN_LIMIT",
                            413,
                        )
                    if item.is_symlink():
                        raise FilesystemError(
                            "Feature folders must not contain symbolic links.", 403
                        )
                    if item.is_dir(follow_symlinks=False) and spec.recursive:
                        pending.append(Path(item.path))
                    elif item.is_file(follow_symlinks=False) and item.name.endswith(
                        spec.fileSuffix
                    ):
                        path = Path(item.path).resolve(strict=True)
                        if not self.filesystem._contains(path) or not path.is_relative_to(root):
                            raise FilesystemError("A feature file is outside permitted roots.", 403)
                        found.append(path)
        return sorted(found)

    @staticmethod
    def _header(
        path: Path, coordinates_path: Path | None = None, *, feature_kind: str = "patch"
    ) -> dict:
        import h5py

        def dataset(handle, key):
            if not isinstance(handle.get(key, getlink=True), h5py.HardLink):
                raise ValueError(f"{key} must be an embedded HDF5 dataset.")
            value = handle[key]
            if not isinstance(value, h5py.Dataset) or value.is_virtual or value.external:
                raise ValueError(f"{key} cannot use external or virtual storage.")
            return value

        def coordinates(handle, count):
            coords = dataset(handle, "coords")
            if coords.shape != (count, 2) or coords.dtype.kind not in {"i", "u"}:
                raise ValueError("coords must contain one integer (x, y) pair per feature row.")
            return _attributes(coords)

        with _regular_file(path) as (stream, stamp):
            with h5py.File(stream, "r") as handle:
                features = dataset(handle, "features")
                if feature_kind == "slide":
                    # A slide encoder produces one embedding for the whole slide.
                    # TRIDENT writes it as a bare vector or a single-row matrix, and
                    # patch coordinates do not apply to it.
                    shape = tuple(features.shape)
                    if (
                        len(shape) not in {1, 2}
                        or not all(shape)
                        or (len(shape) == 2 and shape[0] != 1)
                        or features.dtype.kind != "f"
                    ):
                        raise ValueError(
                            "A slide embedding must be one nonempty floating-point vector "
                            "per slide."
                        )
                    return {
                        "patchCount": 1,
                        "dimensions": shape[-1],
                        "dtype": str(features.dtype),
                        **stamp,
                        "featureKind": "slide",
                        "coordinateSpace": "slide",
                        "attributes": {
                            "file": _attributes(handle),
                            "features": _attributes(features),
                            # Slide embeddings carry no patch grid; keep the key so
                            # callers that read attributes do not special-case them.
                            "coords": {},
                        },
                    }
                if (
                    len(features.shape) != 2
                    or not all(features.shape)
                    or features.dtype.kind != "f"
                ):
                    raise ValueError(
                        "features must be a nonempty, two-dimensional floating-point matrix."
                    )
                result = {
                    "patchCount": features.shape[0],
                    "dimensions": features.shape[1],
                    "dtype": str(features.dtype),
                    **stamp,
                    "attributes": {"file": _attributes(handle), "features": _attributes(features)},
                }
                if handle.get("coords", getlink=True) is not None:
                    result["attributes"]["coords"] = coordinates(handle, features.shape[0])
                    result["coordinatePath"] = str(path)
                    result["coordinateSource"] = "embedded"
                elif coordinates_path is not None:
                    with _regular_file(coordinates_path) as (coords_stream, coords_stamp):
                        with h5py.File(coords_stream, "r") as coords_handle:
                            result["attributes"]["coords"] = coordinates(
                                coords_handle, features.shape[0]
                            )
                            result["coordinateFile"] = {
                                **coords_stamp,
                                "attributes": _attributes(coords_handle),
                            }
                    result["coordinatePath"] = str(coordinates_path)
                    result["coordinateSource"] = "trident-patches"
                else:
                    raise ValueError(
                        "coords is missing; select a TRIDENT folder with patches/ or supply "
                        "the coordinate directory."
                    )
                coord_attrs = result["attributes"]["coords"]
                result["coordinateSpace"] = (
                    "level0_pixels"
                    if "patch_size_level0" in coord_attrs or "level0_magnification" in coord_attrs
                    else "unspecified"
                )
        return result

    def _source_extraction(self, spec: FeatureSpec, layout: dict) -> dict | None:
        """Capture immutable run evidence when attachment follows an in-app extraction."""
        if spec.sourceExtractionJobId is None:
            return None
        from histopilot.application.extraction_artifacts import complete_coverage

        folder = self.store.folder / "extractions" / spec.sourceExtractionJobId
        try:
            documents = {}
            for name in ("job.json", "result.json", "validation.json"):
                documents[name] = json.loads(self.store._read_file(folder / name, 8 * 1024 * 1024))
                if not isinstance(documents[name], dict):
                    raise ValueError("Invalid extraction record.")
            job, result, validation = (
                documents["job.json"],
                documents["result.json"],
                documents["validation.json"],
            )
            options = job.get("spec", {}).get("options", {})
            # A slide encoder names itself; the patch encoder names a patch run.
            slide_encoder = options.get("slide_encoder")
            encoder = slide_encoder or options.get("patch_encoder", "uni_v1")
            if (
                job.get("id") != spec.sourceExtractionJobId
                # The extraction dataset records how the original slides were chosen.
                # A later feature inventory may select another cohort from those same
                # outputs; folder, encoder and completion evidence bind its provenance.
                or options.get("task", "all") not in {"feat", "all"}
                # The attachment must claim the kind the extraction actually wrote,
                # so a slide-embedding run cannot be attached as a patch bag.
                or bool(slide_encoder) != (spec.featureKind == "slide")
                or not isinstance(encoder, str)
                or (
                    layout.get("encoderId")
                    and ENCODER_ALIASES.get(encoder, encoder) != layout["encoderId"]
                )
                or result.get("state") != "succeeded"
                or validation.get("jobId") != job["id"]
                or not complete_coverage(validation, job.get("slideCount"))
                or job.get("outputLayout", {}).get("featureKind") != spec.featureKind
                or Path(job.get("outputLayout", {}).get("featuresDir", "")).resolve()
                != Path(layout["featureDirectory"]).resolve()
            ):
                raise ValueError(
                    f"The extraction must have completed {spec.featureKind} features for this "
                    "encoder and folder."
                )
            evidence = {
                "jobId": job["id"],
                "spec": job["spec"],
                "command": job.get("command"),
                "runtime": job.get("runtime"),
                "inputFiles": job.get("inputFiles", []),
                "validation": validation,
            }
            return {**evidence, "snapshotHash": _hash(evidence)}
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise StorageError(
                f"The linked extraction cannot be verified: {error}",
                "INVALID_SOURCE_EXTRACTION",
                422,
            ) from error

    def preview(self, spec: FeatureSpec) -> dict:
        # The folder defines the feature set unless a slide list narrows it, and a dataset
        # narrows whatever that produced. Coverage against a cohort is decided later, where
        # a protocol binds features to its rows.
        listed = None
        slide_list = None
        if spec.slideList is not None or spec.slideListPath is not None:
            source = spec.slideList or SlideListSource(path=spec.slideListPath)
            try:
                content, label = read_slide_list_source(source, self._list_path)
                listed, declares_mpp, digest = list_slide_ids(
                    content,
                    context=f"Slide list {Path(label).name}",
                )
            except SlideListError as error:
                raise StorageError(
                    f"Invalid slide list: {error}", "INVALID_SLIDE_LIST", 422
                ) from error
            slide_list = {
                "path": label if source.path else None,
                **({"filename": source.filename} if source.filename else {}),
                "sha256": digest,
                "listedCount": len(listed),
                "declaresMpp": declares_mpp,
            }
        expected = None
        if spec.datasetId is not None:
            dataset = self.store.get_dataset(spec.datasetId)
            if dataset["manifest"].get("kind") != "dataset":
                raise StorageError(
                    "Select a dataset frozen through Dataset import.", "INVALID_DATASET", 422
                )
            records = json.loads(self.store.read_artifact(spec.datasetId, "records.json"))
            expected = {row["slideId"] for row in records}
        if listed is not None:
            expected = listed if expected is None else (expected & listed)
        root = self._directory(spec.path)
        layout = self._layout(root, spec)
        extraction = self._source_extraction(spec, layout)
        root = Path(layout["featureDirectory"])
        findings: list[dict] = []
        files: list[dict] = []
        orphan = 0
        seen: set[str] = set()
        inventory = []
        provenance = []
        try:
            try:
                provenance = self._provenance(layout)
            except (OSError, ValueError, RuntimeError) as error:
                findings.append(
                    {
                        "severity": "error",
                        "code": "INVALID_FEATURE_PROVENANCE",
                        "message": f"TRIDENT configuration cannot be inspected: {error}",
                    }
                )
            paths = self._files(root, spec)
            started = time.monotonic()
            for path in paths:
                if time.monotonic() - started > SCAN_SECONDS:
                    raise StorageError(
                        "Feature inspection exceeded 30 seconds; select a smaller folder.",
                        "FEATURE_SCAN_LIMIT",
                        413,
                    )
                stem = path.name[: -len(spec.fileSuffix)]
                slide_id = (
                    stem[: -len(spec.idSuffix)]
                    if spec.idSuffix and stem.endswith(spec.idSuffix)
                    else stem
                )
                info = path.stat()
                inventory.append(
                    {"path": str(path), "size": info.st_size, "mtime": info.st_mtime_ns}
                )
                if spec.idSuffix and not stem.endswith(spec.idSuffix):
                    orphan += 1
                    continue
                if expected is not None and slide_id not in expected:
                    orphan += 1
                    continue
                if slide_id in seen:
                    findings.append(
                        {
                            "severity": "error",
                            "code": "DUPLICATE_FEATURE_ID",
                            "message": f"Multiple feature files match slide {slide_id}.",
                        }
                    )
                    continue
                seen.add(slide_id)
                try:
                    coords_path = (
                        Path(layout["coordinatesDirectory"]) / f"{slide_id}_patches.h5"
                        if layout["coordinatesDirectory"]
                        else None
                    )
                    header = self._header(path, coords_path, feature_kind=spec.featureKind)
                    encoder = header["attributes"]["features"].get("encoder")
                    if encoder is not None and (
                        not isinstance(encoder, str)
                        or (
                            layout["encoderId"]
                            and ENCODER_ALIASES.get(encoder, encoder) != layout["encoderId"]
                        )
                    ):
                        raise ValueError(
                            "The feature encoder attribute differs from the selection."
                        )
                    for attrs in (header["attributes"]["features"], header["attributes"]["coords"]):
                        if "name" in attrs and attrs["name"] != slide_id:
                            raise ValueError("The stored slide name differs from the selected ID.")
                    files.append({"slideId": slide_id, "path": str(path), **header})
                except (OSError, ValueError, KeyError, RuntimeError) as error:
                    findings.append(
                        {
                            "severity": "error",
                            "code": "INVALID_FEATURE_HEADER",
                            "message": f"{path.name}: {error}",
                        }
                    )
        except OSError as error:
            raise FilesystemError("Feature files cannot be inspected.", 403) from error
        dimensions = {item["dimensions"] for item in files}
        encoders = {
            ENCODER_ALIASES.get(
                item["attributes"]["features"]["encoder"], item["attributes"]["features"]["encoder"]
            )
            for item in files
            if item["attributes"]["features"].get("encoder")
        }
        if len(encoders) > 1:
            findings.append(
                {
                    "severity": "error",
                    "code": "INCONSISTENT_ENCODERS",
                    "message": "Selected feature files contain different encoder attributes.",
                }
            )
        elif encoders and not layout["encoderId"]:
            layout["encoderId"] = next(iter(encoders))
        if extraction is not None:
            # Flat imports can discover the encoder only after reading HDF5 headers.
            # Check that final identity against the run before calling the review valid.
            extraction = self._source_extraction(spec, layout)
        if len({(item["deviceId"], item["inode"]) for item in files}) != len(files):
            findings.append(
                {
                    "severity": "error",
                    "code": "ALIASED_FEATURE_FILES",
                    "message": "Different slide identities resolve to the same feature file inode.",
                }
            )
        if len(dimensions) > 1 or len({item["dtype"] for item in files}) > 1:
            findings.append(
                {
                    "severity": "error",
                    "code": "INCONSISTENT_FEATURES",
                    "message": "Selected feature files have inconsistent dimensions or dtypes.",
                }
            )
        if not files:
            findings.append(
                {
                    "severity": "error",
                    "code": "NO_FEATURE_MATCHES",
                    "message": "No valid feature files match this slide selection.",
                }
            )
        missing = len(expected - {item["slideId"] for item in files}) if expected is not None else 0
        if missing:
            findings.append(
                {
                    "severity": "warning",
                    "code": "MISSING_FEATURES",
                    "message": f"{missing} selected slides have no usable features. A protocol uses only slides present in both its dataset and bundle.",
                }
            )
        if orphan:
            findings.append(
                {
                    "severity": "warning",
                    "code": "ORPHAN_FEATURES",
                    "message": f"{orphan} files fall outside the selected slide list, dataset, or filename pattern and are excluded.",
                }
            )
        findings.append(
            {
                "severity": "warning",
                "code": "HEADER_VALIDATION_ONLY",
                "message": "Headers and coverage checked; TRIDENT attributes and configuration "
                "are recorded when present. Full tensor checksums, finite values, coordinate "
                "contents, and encoder checkpoint provenance remain unverified.",
            }
        )
        summary = {
            "slideCount": len(expected) if expected is not None else len(files),
            "scope": _scope(spec),
            "matchedSlides": len(files),
            "missingSlides": missing,
            "orphanFiles": orphan,
            "dimensions": next(iter(dimensions)) if len(dimensions) == 1 else None,
            "patchCount": sum(item["patchCount"] for item in files),
        }
        result = {
            "spec": spec.model_dump(mode="json"),
            "summary": summary,
            **({"slideList": slide_list} if slide_list is not None else {}),
            "files": files,
            "layout": layout,
            "provenance": provenance,
            **({"sourceExtraction": extraction} if extraction is not None else {}),
            "findings": findings,
            "validationLevel": "headers",
            "canFreeze": not any(item["severity"] == "error" for item in findings),
        }
        return {**result, "previewHash": _hash({**result, "inventory": inventory})}

    def freeze(
        self,
        spec: FeatureSpec,
        preview_hash: str,
        operation_id: str,
        *,
        version_label: dict | None = None,
    ) -> dict:
        prior = self.store.configuration_publication(operation_id)
        if prior:
            manifest = prior["manifest"]
            if (
                manifest.get("kind") != "feature"
                or manifest.get("spec") != spec.model_dump(mode="json")
                or manifest.get("previewHash") != preview_hash
            ):
                raise StorageError(
                    "This operation ID belongs to another feature request.", "OPERATION_CONFLICT"
                )
            return self.store.publish_configuration(
                manifest=manifest, operation_id=operation_id, version_label=version_label
            )
        preview = self.preview(spec)
        if preview["previewHash"] != preview_hash:
            raise StorageError(
                "Feature files or settings changed. Preview again before freezing.", "PREVIEW_STALE"
            )
        if not preview["canFreeze"]:
            raise StorageError(
                "Resolve feature inspection errors before freezing.", "FEATURES_INVALID", 422
            )
        manifest = {
            "kind": "feature",
            "schemaVersion": 1,
            "datasetId": spec.datasetId,
            **{
                key: value
                for key, value in preview.items()
                if key not in {"canFreeze", "previewHash"}
            },
            "previewHash": preview_hash,
        }

        def check_sources():
            if self.verify_binding({"manifest": manifest}):
                raise StorageError(
                    "Feature inputs changed before publication. Preview again.", "PREVIEW_STALE"
                )

        return self.store.publish_configuration(
            manifest=manifest,
            operation_id=operation_id,
            version_label=version_label,
            before_publish=check_sources,
        )

    def verify_binding(self, configuration: dict) -> list[dict]:
        """Recheck the saved eligible files at preflight; never silently replace the binding."""
        findings = []
        manifest = configuration["manifest"]
        # Records frozen before slide encoders were supported hold patch bags.
        feature_kind = manifest.get("spec", {}).get("featureKind", "patch")
        if manifest.get("sourceExtraction"):
            try:
                current = self._source_extraction(
                    FeatureSpec.model_validate(manifest["spec"]), manifest["layout"]
                )
                if current != manifest["sourceExtraction"]:
                    raise ValueError("Recorded extraction evidence changed.")
            except (ValueError, KeyError, OSError) as error:
                findings.append(
                    {
                        "severity": "error",
                        "code": "FEATURE_SOURCE_CHANGED",
                        "message": f"Extraction provenance: {error}",
                    }
                )
        started = time.monotonic()
        if len(configuration["manifest"]["files"]) > MAX_FILES:
            raise StorageError(
                "Feature validation exceeded its file limit.", "FEATURE_SCAN_LIMIT", 413
            )
        for item in configuration["manifest"]["files"]:
            if time.monotonic() - started > SCAN_SECONDS:
                findings.append(
                    {
                        "severity": "error",
                        "code": "FEATURE_SCAN_LIMIT",
                        "message": "Feature preflight exceeded its time budget; remaining files were not validated.",
                    }
                )
                break
            try:
                path = Path(item["path"])
                resolved = path.resolve(strict=True)
                if not self.filesystem._contains(resolved) or resolved != path:
                    raise ValueError("File moved outside its declared path or permitted roots.")
                coordinate_path = Path(item.get("coordinatePath", item["path"]))
                coordinate_resolved = coordinate_path.resolve(strict=True)
                if (
                    not self.filesystem._contains(coordinate_resolved)
                    or coordinate_resolved != coordinate_path
                ):
                    raise ValueError("Coordinate file moved outside its path or permitted roots.")
                header = self._header(
                    path,
                    coordinate_path if coordinate_path != path else None,
                    feature_kind=feature_kind,
                )
                # Older flat imports lack TRIDENT metadata; still validate their original stamps.
                if any(header[key] != item[key] for key in header if key in item):
                    raise ValueError("File header or modification metadata changed.")
            except (OSError, ValueError, KeyError, RuntimeError) as error:
                findings.append(
                    {
                        "severity": "error",
                        "code": "FEATURE_SOURCE_CHANGED",
                        "message": f"{item['slideId']}: {error}",
                    }
                )
        for source in configuration["manifest"].get("provenance", []):
            try:
                path = Path(source["path"])
                resolved = path.resolve(strict=True)
                if not self.filesystem._contains(resolved) or resolved != path:
                    raise ValueError("Configuration moved outside its path or permitted roots.")
                with _regular_file(path) as (stream, stamp):
                    if stamp["sizeBytes"] > MAX_METADATA_BYTES:
                        raise ValueError("Configuration exceeds its metadata size limit.")
                    checksum = hashlib.sha256(stream.read(MAX_METADATA_BYTES + 1)).hexdigest()
                if any(stamp[key] != source[key] for key in stamp) or checksum != source["sha256"]:
                    raise ValueError("Recorded TRIDENT configuration changed.")
            except (OSError, ValueError, KeyError, RuntimeError) as error:
                findings.append(
                    {
                        "severity": "error",
                        "code": "FEATURE_SOURCE_CHANGED",
                        "message": f"TRIDENT provenance: {error}",
                    }
                )
        return findings
