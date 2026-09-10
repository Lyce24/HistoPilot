"""Bounded HDF5 header inspection; never load embeddings or a CUDA model."""

import hashlib
import json
import os
import stat
import time
from pathlib import Path

from histopilot.schemas.features import FeatureSpec
from histopilot.storage.filesystem import FilesystemError, LocalFilesystem
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import ScientificStore

MAX_FILES = 10000
SCAN_SECONDS = 30


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


class FeatureService:
    def __init__(self, store: ScientificStore, filesystem: LocalFilesystem):
        self.store = store
        self.filesystem = filesystem

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
    def _header(path: Path) -> dict:
        import h5py

        if any(component.is_symlink() for component in (path, *path.parents)):
            raise ValueError("Feature input cannot traverse symbolic links.")
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("Feature input is not a regular file.")
            # Inspect the opened inode rather than reopening an independently
            # replaceable path; the file-object driver still reads only metadata.
            with h5py.File(stream, "r") as handle:
                for key in ("features", "coords"):
                    if not isinstance(handle.get(key, getlink=True), h5py.HardLink):
                        raise ValueError(f"{key} must be an embedded HDF5 dataset.")
                    value = handle[key]
                    if not isinstance(value, h5py.Dataset) or value.is_virtual or value.external:
                        raise ValueError(f"{key} cannot use external or virtual storage.")
                features, coords = handle["features"], handle["coords"]
                if (
                    len(features.shape) != 2
                    or not all(features.shape)
                    or features.dtype.kind != "f"
                ):
                    raise ValueError(
                        "features must be a nonempty, two-dimensional floating-point matrix."
                    )
                if coords.shape != (features.shape[0], 2) or coords.dtype.kind not in {"i", "u"}:
                    raise ValueError("coords must contain one integer (x, y) pair per feature row.")
                result = {
                    "patchCount": features.shape[0],
                    "dimensions": features.shape[1],
                    "dtype": str(features.dtype),
                    "sizeBytes": before.st_size,
                    "mtimeNs": before.st_mtime_ns,
                    "ctimeNs": before.st_ctime_ns,
                    "deviceId": before.st_dev,
                    "inode": before.st_ino,
                }
            opened_after = os.fstat(stream.fileno())
        after = path.lstat()

        def stamp(info):
            return (info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_dev, info.st_ino)

        if (
            not stat.S_ISREG(after.st_mode)
            or stamp(before) != stamp(after)
            or stamp(before) != stamp(opened_after)
        ):
            raise ValueError("Feature file changed during inspection; preview again.")
        return result

    def preview(self, spec: FeatureSpec) -> dict:
        dataset = self.store.get_dataset(spec.datasetId)
        if dataset["manifest"].get("kind") != "dataset":
            raise StorageError(
                "Select a dataset frozen through Dataset import.", "INVALID_DATASET", 422
            )
        records = json.loads(self.store.read_artifact(spec.datasetId, "records.json"))
        expected = {row["slideId"] for row in records}
        root = self.filesystem.directory(spec.path)
        findings: list[dict] = []
        files: list[dict] = []
        orphan = 0
        seen: set[str] = set()
        inventory = []
        try:
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
                if slide_id not in expected:
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
                    files.append({"slideId": slide_id, "path": str(path), **self._header(path)})
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
                    "message": "No valid feature files match this frozen dataset.",
                }
            )
        missing = len(expected - {item["slideId"] for item in files})
        if missing:
            findings.append(
                {
                    "severity": "warning",
                    "code": "MISSING_FEATURES",
                    "message": f"{missing} slides have no usable features. Protocols requiring them will be blocked.",
                }
            )
        if orphan:
            findings.append(
                {
                    "severity": "warning",
                    "code": "ORPHAN_FEATURES",
                    "message": f"{orphan} files do not match the selected dataset and are excluded.",
                }
            )
        findings.append(
            {
                "severity": "warning",
                "code": "HEADER_VALIDATION_ONLY",
                "message": "Headers and coverage checked. Full tensor checksums, finite-value checks, and encoder provenance remain unverified.",
            }
        )
        summary = {
            "slideCount": len(expected),
            "matchedSlides": len(files),
            "missingSlides": missing,
            "orphanFiles": orphan,
            "dimensions": next(iter(dimensions)) if len(dimensions) == 1 else None,
            "patchCount": sum(item["patchCount"] for item in files),
        }
        result = {
            "spec": spec.model_dump(mode="json"),
            "summary": summary,
            "files": files,
            "findings": findings,
            "validationLevel": "headers",
            "canFreeze": not any(item["severity"] == "error" for item in findings),
        }
        return {**result, "previewHash": _hash({**result, "inventory": inventory})}

    def freeze(self, spec: FeatureSpec, preview_hash: str, operation_id: str) -> dict:
        preview = self.preview(spec)
        if preview["previewHash"] != preview_hash:
            raise StorageError(
                "Feature files or settings changed. Preview again before freezing.", "PREVIEW_STALE"
            )
        if not preview["canFreeze"]:
            raise StorageError(
                "Resolve feature inspection errors before freezing.", "FEATURES_INVALID", 422
            )
        return self.store.publish_configuration(
            manifest={
                "kind": "feature",
                "schemaVersion": 1,
                "datasetId": spec.datasetId,
                **{
                    key: value
                    for key, value in preview.items()
                    if key not in {"canFreeze", "previewHash"}
                },
                "previewHash": preview_hash,
            },
            operation_id=operation_id,
        )

    def verify_binding(self, configuration: dict) -> list[dict]:
        """Recheck the saved eligible files at preflight; never silently replace the binding."""
        findings = []
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
                header = self._header(path)
                if any(header[key] != item[key] for key in header):
                    raise ValueError("File header or modification metadata changed.")
            except (OSError, ValueError, KeyError, RuntimeError) as error:
                findings.append(
                    {
                        "severity": "error",
                        "code": "FEATURE_SOURCE_CHANGED",
                        "message": f"{item['slideId']}: {error}",
                    }
                )
        return findings
