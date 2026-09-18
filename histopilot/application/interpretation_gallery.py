"""Searchable, root-confined slide discovery bound to frozen feature representations."""

import json
import os
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

from histopilot.application.feature_bundles import FeatureBundleService
from histopilot.application.imports import SLIDE_EXTENSIONS
from histopilot.application.predictors import PredictorService, reference
from histopilot.schemas.interpretation import InterpretationSlide
from histopilot.storage.filesystem import FilesystemError
from histopilot.storage.project_lock import StorageError, _reject_symlink_components
from histopilot.viewer.slide_images import allowed_file

MAX_ENTRIES = 20000
MAX_SLIDES = 10000
MAX_DEPTH = 24
SCAN_SECONDS = 5
IMAGE_EXTENSIONS = SLIDE_EXTENSIONS | {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@contextmanager
def directory_entries(path):
    """Pin every directory component without following a link substituted during discovery."""
    descriptor = None
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        descriptor = os.open(path.anchor, flags)
        for component in path.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        with os.scandir(descriptor) as entries:
            yield entries
        current, opened = path.stat(follow_symlinks=False), os.fstat(descriptor)
        if path.resolve(strict=True) != path or (current.st_dev, current.st_ino) != (
            opened.st_dev,
            opened.st_ino,
        ):
            raise StorageError(
                "A slide folder changed during discovery. Browse it again.",
                "INTERPRETATION_FOLDER_CHANGED",
                409,
            )
    finally:
        if descriptor is not None:
            os.close(descriptor)


def allowed_folder(filesystem, value):
    try:
        path = Path(value)
        if not path.is_absolute() or ".." in path.parts or "\x00" in value:
            raise ValueError
        _reject_symlink_components(path)
        return filesystem.directory(value)
    except (FilesystemError, OSError, ValueError, RuntimeError) as error:
        raise StorageError(
            "Select a slide folder inside a configured data root, without symbolic links.",
            "INTERPRETATION_FOLDER_INVALID",
            403,
        ) from error


class InterpretationGalleryService:
    def __init__(self, store, filesystem):
        self.store, self.filesystem = store, filesystem
        self.bundles = FeatureBundleService(store, filesystem)

    def _records(self, dataset_id):
        try:
            return json.loads(self.store.read_artifact(dataset_id, "records.json"))
        except StorageError as error:
            if error.code != "ARTIFACT_NOT_FOUND":
                raise
            return json.loads(self.store.read_artifact(dataset_id, "slides.json"))

    def dataset_source(self, dataset_id, *, records=None):
        """Reuse the bundle's own frozen dataset folder without guessing across cohorts."""
        metadata = {
            "datasetId": dataset_id,
            "datasetName": f"Dataset {dataset_id[-8:]}" if dataset_id else "No dataset",
            "slideFolder": None,
            "slideFolderSource": None,
            "slideFolderFinding": None,
        }
        reason = "The frozen dataset has no unambiguous, accessible slide folder."
        if dataset_id is None:
            # A store-scoped bundle names no cohort, so no slide folder can be resolved from it.
            metadata["slideFolderFinding"] = {
                "severity": "warning",
                "code": "DATASET_SLIDE_FOLDER_UNAVAILABLE",
                "message": "These features are scoped to a slide store, not a frozen dataset. "
                "Attach them to a dataset to open slides for interpretation.",
            }
            return metadata
        try:
            dataset = self.store.get_dataset(dataset_id)
            manifest = dataset["manifest"]
            metadata["datasetName"] = (
                (dataset.get("versionLabel") or {}).get("tag")
                or manifest.get("name")
                or dataset.get("name")
                or metadata["datasetName"]
            )
            records = self._records(dataset_id) if records is None else records
            paths = []
            for row in records:
                value = row.get("slidePath")
                if value is None:
                    continue
                if not isinstance(value, str):
                    raise ValueError("The dataset contains an invalid recorded slide path.")
                path = Path(value)
                if (
                    not path.is_absolute()
                    or ".." in path.parts
                    or any(ord(character) < 32 for character in value)
                    or path.suffix.lower() not in IMAGE_EXTENSIONS
                ):
                    raise ValueError("The dataset contains an invalid recorded slide path.")
                _reject_symlink_components(path)
                if not self.filesystem._contains(path):
                    raise ValueError(
                        "The dataset's recorded slides are outside configured data roots."
                    )
                paths.append(path)
            declared = manifest.get("provenance", {}).get("mapping", {}).get(
                "slideRoot"
            ) or manifest.get("spec", {}).get("slideRoot")
            if declared:
                try:
                    folder = allowed_folder(self.filesystem, declared)
                    if folder == Path(folder.anchor) or any(
                        not path.is_relative_to(folder) for path in paths
                    ):
                        raise ValueError(
                            "The saved folder does not contain this dataset's recorded slides."
                        )
                    return {
                        **metadata,
                        "slideFolder": str(folder),
                        "slideFolderSource": "dataset_import",
                    }
                except (StorageError, ValueError):
                    reason = "The saved dataset slide folder is unavailable or does not match its recorded slides."
            # A shared direct parent is evidence; a broad common ancestor of
            # unrelated folders is not. Nested legacy cohorts require their saved import root.
            parents = {path.parent for path in paths}
            if len(parents) == 1:
                folder = allowed_folder(self.filesystem, str(next(iter(parents))))
                if folder != Path(folder.anchor):
                    return {
                        **metadata,
                        "slideFolder": str(folder),
                        "slideFolderSource": "dataset_records",
                    }
            elif len(parents) > 1:
                reason = "The dataset's slides span multiple folders and no valid dataset import folder is recorded."
        except (StorageError, ValueError, TypeError, KeyError, OSError) as error:
            reason = str(error)
        finding = {
            "severity": "error",
            "code": "INTERPRETATION_DATASET_FOLDER_UNAVAILABLE",
            "message": f"{reason} Open Datasets to link or update its slide folder, then freeze the corresponding feature bundle.",
        }
        return {**metadata, "slideFolderFinding": finding}

    def sources(self):
        items = []
        for bundle in self.bundles.list()["items"]:
            manifest = bundle["manifest"]
            dataset_id = manifest["datasetId"]
            try:
                feature = self.store.get_configuration(manifest["feature"]["id"])["manifest"]
                dataset_id = feature["datasetId"] or dataset_id
                encoder = feature.get("layout", {}).get("encoderId") or feature.get("spec", {}).get(
                    "encoderId"
                )
            except StorageError:
                encoder = None
            summary = manifest["summary"]
            dataset_source = self.dataset_source(dataset_id)
            folder_findings = (
                [dataset_source["slideFolderFinding"]]
                if dataset_source["slideFolderFinding"]
                else []
            )
            items.append(
                {
                    "id": bundle["id"],
                    "name": (bundle.get("versionLabel") or {}).get("tag")
                    or manifest.get("name")
                    or f"Bundle {bundle['id'][-8:]}",
                    "current": bundle["current"] and not folder_findings,
                    "findings": [*bundle["findings"], *folder_findings],
                    **dataset_source,
                    "encoderId": encoder,
                    "dimensions": summary["dimensions"],
                    "dtype": summary["dtype"],
                    "slideCount": summary["slideCount"],
                    "featureSetId": manifest["feature"]["id"],
                    "packs": [
                        {
                            "id": row["id"],
                            "name": Path(row["outputPath"]).name,
                            "outputDtype": row["outputDtype"],
                        }
                        for row in manifest["packs"]
                    ],
                }
            )
        return {"items": items}

    def resolve(self, source, *, require_current=True):
        bundle = self.bundles.get(source.featureBundleId)
        manifest = bundle["manifest"]
        feature = self.store.get_configuration(manifest["feature"]["id"])
        original = feature["manifest"]
        encoder = original.get("layout", {}).get("encoderId") or original.get("spec", {}).get(
            "encoderId"
        )
        pack = None
        if source.packArtifactId:
            if source.packArtifactId not in {item["id"] for item in manifest["packs"]}:
                raise StorageError(
                    "Select a pack included in this frozen feature bundle.",
                    "INTERPRETATION_PACK_MISMATCH",
                    422,
                )
            pack = self.bundles.packing.artifact(source.packArtifactId)
            # get() verifies the exact frozen artifact binding, including validation and stamps.
        contract = {
            "encoderId": encoder,
            "dimensions": manifest["summary"]["dimensions"],
            "dtype": pack["outputDtype"] if pack else manifest["summary"]["dtype"],
        }
        findings = list(bundle["findings"])
        if not encoder:
            findings.append(
                {
                    "code": "INTERPRETATION_ENCODER_MISSING",
                    "message": "The bundle must record its feature encoder identity.",
                    "severity": "error",
                }
            )
        if source.predictorId:
            predictor = PredictorService(self.store, self.filesystem).get(source.predictorId)
            expected = predictor["manifest"]["inputs"]["features"]
            if any(contract[key] != expected.get(key) for key in contract):
                findings.append(
                    {
                        "code": "INTERPRETATION_FEATURE_MISMATCH",
                        "message": "This representation does not match the predictor's encoder, dimensions and dtype. Select a compatible bundle or pack.",
                        "severity": "error",
                    }
                )
            if (predictor["manifest"].get("recipe", {}).get("model", "abmil").lower() not in {"abmil", "nnmil"}
                    or predictor["manifest"].get("recipe", {}).get("inputMode") == "clinical"):
                findings.append(
                    {
                        "code": "INTERPRETATION_MODEL_UNSUPPORTED",
                        "message": "Attention visualization requires an ABMIL or nnMIL predictor.",
                        "severity": "error",
                    }
                )
        if require_current and findings:
            raise StorageError(findings[0]["message"], findings[0]["code"], 422)
        files = defaultdict(list)
        for row in original["files"]:
            files[row["slideId"]].append(row)
        aliases = defaultdict(set)
        for identity in files:
            aliases[identity].add(identity)
        # Preserve custom imported slide IDs when images are moved as a folder.
        records = self._records(original["datasetId"])
        for row in records:
            if row.get("slidePath") and row["slideId"] in files:
                aliases[Path(row["slidePath"]).stem].add(row["slideId"])
        return {
            "bundle": bundle,
            "feature": feature,
            "pack": pack,
            "contract": contract,
            "files": files,
            "aliases": aliases,
            "findings": findings,
            "datasetSource": self.dataset_source(original["datasetId"], records=records),
        }

    def scan(self, value):
        folder = allowed_folder(self.filesystem, value)
        deadline, entries, skipped = time.monotonic() + SCAN_SECONDS, 0, 0
        paths, pending = [], [(folder, 0)]
        while pending:
            directory, depth = pending.pop()
            try:
                _reject_symlink_components(directory)
                with directory_entries(directory) as listing:
                    children = []
                    for entry in listing:
                        entries += 1
                        if entries > MAX_ENTRIES or time.monotonic() > deadline:
                            raise StorageError(
                                "Slide discovery exceeded its bounded scan. Select a smaller slide folder.",
                                "INTERPRETATION_SCAN_LIMIT",
                                413,
                            )
                        if entry.is_symlink():
                            skipped += 1
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            if depth >= MAX_DEPTH:
                                raise StorageError(
                                    "The slide folder is too deeply nested. Select a closer folder.",
                                    "INTERPRETATION_SCAN_LIMIT",
                                    413,
                                )
                            children.append((directory / entry.name, depth + 1))
                        elif (
                            entry.is_file(follow_symlinks=False)
                            and Path(entry.name).suffix.lower() in IMAGE_EXTENSIONS
                        ):
                            paths.append(directory / entry.name)
                            if len(paths) > MAX_SLIDES:
                                raise StorageError(
                                    "Select a folder containing at most 10,000 slides.",
                                    "INTERPRETATION_SCAN_LIMIT",
                                    413,
                                )
                    pending.extend(sorted(children, reverse=True))
            except OSError as error:
                raise StorageError(
                    "A slide subfolder could not be read. Check permissions or select a readable folder.",
                    "INTERPRETATION_FOLDER_UNREADABLE",
                    403,
                ) from error
        paths.sort(key=lambda path: (str(path.relative_to(folder)).casefold(), str(path)))
        return folder, paths, [f"Skipped {skipped} symbolic links."] if skipped else []

    def rows(self, source, *, require_current=False):
        resolved = self.resolve(source, require_current=require_current)
        selected_folder = source.slideFolder or resolved["datasetSource"]["slideFolder"]
        if selected_folder is None:
            finding = resolved["datasetSource"]["slideFolderFinding"]
            raise StorageError(finding["message"], finding["code"], 422)
        folder, paths, warnings = self.scan(selected_folder)
        stems = Counter(path.stem for path in paths)
        items = []
        for path in paths:
            candidates = resolved["aliases"].get(path.stem, set())
            identity = next(iter(candidates)) if len(candidates) == 1 else path.stem
            matched = resolved["files"].get(identity, [])
            reason = None
            if resolved["findings"]:
                reason = resolved["findings"][0]["message"]
            elif stems[path.stem] > 1 or len(candidates) > 1 or len(matched) > 1:
                reason = "Ambiguous slide identity: multiple images or feature rows share this ID. Select a more specific folder or correct the inventory."
            elif not candidates or not matched:
                reason = "No matching slide ID in this frozen feature bundle."
            elif not resolved["pack"] and not matched[0].get("coordinatePath"):
                reason = "This feature row has no verified patch coordinates."
            if reason is None:
                try:
                    InterpretationSlide.logical_identity(identity)
                except ValueError:
                    reason = "The frozen slide ID contains unsupported separators or control characters. Correct its ID before visualization."
            items.append(
                {
                    "key": str(path),
                    "slideId": identity,
                    "slidePath": str(path),
                    "relativePath": str(path.relative_to(folder)),
                    "name": path.name,
                    "available": reason is None,
                    "reason": reason,
                    "patchCount": matched[0]["patchCount"] if len(matched) == 1 else None,
                }
            )
        # Different filename aliases can still resolve to one canonical slide identity.
        counts = Counter(row["slideId"] for row in items if row["available"])
        for row in items:
            if row["available"] and counts[row["slideId"]] > 1:
                row.update(
                    available=False,
                    reason="Ambiguous slide identity: multiple images resolve to the same frozen feature slide.",
                )
        return resolved, folder, items, warnings

    def query(self, request):
        resolved, folder, items, warnings = self.rows(request)
        search = request.search.strip().casefold()
        if search:
            items = [
                row
                for row in items
                if search in f"{row['slideId']} {row['relativePath']}".casefold()
            ]
        total = len(items)
        return {
            "items": items[request.offset : request.offset + request.limit],
            "total": total,
            "offset": request.offset,
            "limit": request.limit,
            "hasMore": request.offset + request.limit < total,
            "folder": str(folder),
            "source": {
                "featureBundleId": request.featureBundleId,
                "packArtifactId": request.packArtifactId,
                **resolved["contract"],
            },
            "warnings": warnings,
        }

    def input(self, resolved, row, *, width=None, height=None):
        if not row["available"]:
            raise StorageError(row["reason"], "INTERPRETATION_SLIDE_UNAVAILABLE", 422)
        source = resolved["files"][row["slideId"]][0]
        result = {
            "slideId": row["slideId"],
            "slidePath": str(allowed_file(self.filesystem, row["slidePath"])),
            "confirmRowAlignment": True,
            "patchWidthLevel0": width,
            "patchHeightLevel0": height,
        }
        if resolved["pack"]:
            pack = resolved["pack"]
            result.update(
                sourceFormat="packed",
                packPath=str(allowed_folder(self.bundles.packing.outputs, pack["outputPath"])),
                packSlideId=row["slideId"],
            )
        else:
            result.update(
                sourceFormat="h5",
                featurePath=str(allowed_file(self.filesystem, source["path"])),
                coordinatesPath=str(allowed_file(self.filesystem, source["coordinatePath"])),
            )
        return InterpretationSlide.model_validate(result)

    def bind(self, selection, *, context=None):
        if context is None:
            resolved, folder, rows, _ = self.rows(selection, require_current=True)
        else:
            resolved, folder, rows = context[:3]
            if (
                selection.featureBundleId != resolved["bundle"]["id"]
                or selection.packArtifactId
                != (resolved["pack"]["id"] if resolved["pack"] else None)
                or selection.slideFolder != str(folder)
            ):
                raise StorageError(
                    "The source selection changed during visualization.",
                    "INTERPRETATION_SOURCE_MISMATCH",
                    409,
                )
        by_path = {row["slidePath"]: row for row in rows}
        normalized = []
        for request in selection.slides:
            path = str(allowed_file(self.filesystem, request.slidePath))
            if path not in by_path:
                raise StorageError(
                    "The selected slide is outside the selected slide folder.",
                    "INTERPRETATION_SLIDE_UNAVAILABLE",
                    422,
                )
            expected = self.input(
                resolved,
                by_path[path],
                width=request.patchWidthLevel0,
                height=request.patchHeightLevel0,
            )
            if expected.model_dump() != request.model_dump():
                raise StorageError(
                    "Slide inputs no longer match the selected frozen bundle and representation.",
                    "INTERPRETATION_SOURCE_MISMATCH",
                    409,
                )
            row = expected.model_dump()
            if resolved["pack"]:
                row["packEvidence"] = resolved["pack"]
            normalized.append(row)
        return (
            normalized,
            [reference(resolved["bundle"]), reference(resolved["feature"])],
            str(folder),
        )
