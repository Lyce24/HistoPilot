"""Durable, project-scoped TRIDENT extraction with server-derived slide manifests."""

import csv
import hashlib
import io
import json
import os
import re
import shutil
import stat
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from histopilot.adapters import trident
from histopilot.adapters.trident.performance import (
    estimate_output_bytes,
    estimate_ram_gb,
    estimate_vram_gb,
    execution_device_count,
    resolve_max_workers,
    uses_gpu,
    workload_key,
)
from histopilot.adapters.trident.progress import build_progress
from histopilot.application import task_records
from histopilot.application.project_outputs import overlaps as _overlaps
from histopilot.application.project_outputs import protected_output
from histopilot.application.slide_lists import (
    SlideListError,
    read_slide_list_source,
    resolve_slide_selection,
)
from histopilot.schemas.extractions import ExtractionSpec
from histopilot.schemas.slide_lists import SlideListSource
from histopilot.storage.filesystem import LocalFilesystem
from histopilot.storage.lifecycle import LifecycleStore, lifecycle_guard
from histopilot.storage.project_lock import (
    StorageError,
    ensure_managed_directory,
    fsync_directory,
    reject_symlink_components,
    writer_lock,
)
from histopilot.storage.scientific import ScientificStore
from histopilot.taskcenter import ids

ACTIVE = {"queued", "starting", "running", "cancelling"}
JOB_ID = re.compile(r"^extraction-[a-f0-9]{32}$")
MAX_JSON = 8 * 1024 * 1024
MAX_LOG_BYTES = 64 * 1024
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
# Refuse to start when the output volume cannot even hold this much of an estimated run.
MIN_FREE_BYTES = 2 * 1024**3
EXTRACTION_GRACE_SECONDS = 30
VALIDATION_GRACE_SECONDS = 10


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _read(path: Path) -> dict:
    content = ScientificStore._read_file(path, MAX_JSON)
    try:
        value = json.loads(content)
        if not isinstance(value, dict):
            raise ValueError
        return value
    except (ValueError, UnicodeError) as error:
        raise StorageError("Extraction metadata is invalid.", "EXTRACTION_CORRUPT") from error


def _write(path: Path, value: dict) -> None:
    reject_symlink_components(path)
    content = json.dumps(value, indent=2, allow_nan=False).encode() + b"\n"
    if len(content) > MAX_JSON:
        raise StorageError("Extraction metadata exceeds its size limit.", "EXTRACTION_LIMIT", 413)
    descriptor, name = tempfile.mkstemp(prefix=".extraction-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        fsync_directory(path.parent)
    finally:
        Path(name).unlink(missing_ok=True)


def _log_tail(path: Path) -> tuple[str, str | None]:
    """Read one bounded snapshot for progress and optional troubleshooting output."""
    reject_symlink_components(path)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return "", None
    with os.fdopen(descriptor, "rb") as handle:
        metadata = os.fstat(handle.fileno())
        if not stat.S_ISREG(metadata.st_mode):
            raise StorageError("Worker log is unsafe.", "EXTRACTION_CORRUPT")
        handle.seek(max(0, metadata.st_size - MAX_LOG_BYTES))
        return (
            handle.read(MAX_LOG_BYTES).decode(errors="replace"),
            datetime.fromtimestamp(metadata.st_mtime, UTC).isoformat(),
        )


class ExtractionService:
    """TRIDENT extraction records.

    Jobs are two Task Center tasks: ``extraction`` (TRIDENT, GPU lane) and
    ``extraction-validation`` (artifact validation, CPU lane, after it). Jobs recorded
    before the Task Center stay readable but can no longer run, resume or be cancelled.
    """

    def __init__(self, store: ScientificStore, filesystem: LocalFilesystem, *, task_center=None):
        self.store = store
        self.filesystem = filesystem
        self.outputs = LocalFilesystem((store.folder, *filesystem.roots))
        self.tasks = task_records.TaskCenterAccess(task_center)
        self.folder = store.folder / "extractions"

    def catalog(self) -> dict:
        return {
            **trident.option_catalog(),
            "runtime": trident.discover_runtime(),
            "defaultOutputPath": str(self.store.folder / "trident"),
        }

    def _path(self, value: str, *, exists=False, directory=False) -> Path:
        path = Path(value)
        if not path.is_absolute() or "\x00" in value or ".." in path.parts:
            raise StorageError(
                "Use an absolute path without parent traversal.", "INVALID_PATH", 422
            )
        reject_symlink_components(path)
        resolved = path.resolve()
        if not self.outputs._contains(resolved):
            raise StorageError("The path is outside configured data roots.", "INVALID_PATH", 403)
        if exists and (not resolved.exists() or (directory and not resolved.is_dir())):
            raise StorageError("The selected input path does not exist.", "INVALID_PATH", 422)
        return resolved

    def _jobs(self) -> list[dict]:
        if not self.folder.exists():
            return []
        reject_symlink_components(self.folder)
        paths = sorted(self.folder.glob("extraction-*/job.json"))
        if len(paths) > 10000:
            raise StorageError("Too many extraction records.", "EXTRACTION_LIMIT", 413)
        return [self._job_record(path.parent.name) for path in paths]

    def _job_record(self, identity: str) -> dict:
        if not JOB_ID.fullmatch(identity):
            raise StorageError("Extraction not found.", "EXTRACTION_NOT_FOUND", 404)
        path = self.folder / identity / "job.json"
        if not path.exists():
            raise StorageError("Extraction not found.", "EXTRACTION_NOT_FOUND", 404)
        job = _read(path)
        if job.get("id") != identity or job.get("projectId") != self.store.project_id:
            raise StorageError("Extraction metadata is inconsistent.", "EXTRACTION_CORRUPT")
        return job

    def _slide_root(self, dataset: dict) -> Path | None:
        """The folder the dataset was imported from; slide lists are written relative to it."""
        manifest = dataset.get("manifest", {})
        declared = manifest.get("provenance", {}).get("mapping", {}).get(
            "slideRoot"
        ) or manifest.get("spec", {}).get("slideRoot")
        if not declared:
            return None
        try:
            return self._path(str(declared), exists=True, directory=True)
        except StorageError:
            # An imported root can be renamed or unmounted; fall back to the linked slides.
            return None

    def _select_slides(self, spec, values, dataset, records, finding):
        """One source, then one optional cohort filter.

        The source is the slide list when the options name one, the slide folder when one is
        chosen, and otherwise the dataset's own linked slide files. A dataset, when given
        alongside a list or folder, narrows that selection to the slides it claims.
        """
        list_path = values.get("custom_list_of_wsis")
        content = None
        list_label = list_path
        if spec.slideList is not None or list_path:
            source = spec.slideList or SlideListSource(path=list_path)
            try:
                content, list_label = read_slide_list_source(
                    source, lambda path: self._path(path, exists=True)
                )
            except SlideListError as error:
                raise StorageError(str(error), "INVALID_SLIDE_LIST", 422) from error
            list_path = list_label if source.path else None
        root = None
        if spec.slideRoot is not None:
            root = self._path(spec.slideRoot, exists=True, directory=True)
        elif content is not None and dataset is not None:
            root = self._slide_root(dataset)
        if content is None and root is None:
            if records is None:
                raise StorageError("Choose a slide folder or a slide list.", "INVALID_SLIDES", 422)
            rows = self._dataset_slides(records, finding)
            # The dataset is the source here; report it the same way as any other selection.
            return rows, {
                "source": "dataset",
                "listPath": None,
                "sha256": None,
                "root": str(self._slide_root(dataset) or ""),
                "initialCount": len(records),
                "selectedCount": len(rows),
                "declaresMpp": False,
                "datasetFiltered": False,
                "outside": [],
                "outsideCount": 0,
                "outsideExamples": [],
                "unlisted": [],
                "unlistedCount": len(records) - len(rows),
                "unlistedExamples": [],
            }
        if root is None:
            raise StorageError(
                "A slide list needs the slide folder its paths are relative to.",
                "SLIDE_ROOT_REQUIRED",
                422,
            )
        try:
            selection = resolve_slide_selection(
                root,
                list_content=content,
                list_path=str(list_path) if list_path else None,
                records=records,
                wsi_ext=values.get("wsi_ext"),
                recursive=spec.recursive,
                context=f"Slide list {Path(list_label).name}" if list_label else "The slide folder",
            )
        except SlideListError as error:
            raise StorageError(
                f"Invalid slide selection: {error}", "INVALID_SLIDE_LIST", 422
            ) from error
        rows = [
            {"slideId": entry.slideId, "path": str(entry.path), "mpp": entry.mpp}
            for entry in selection["slides"]
        ]
        return rows, {key: value for key, value in selection.items() if key != "slides"} | {
            "outsideCount": len(selection["outside"]),
            "outsideExamples": selection["outside"][:5],
            "unlistedCount": len(selection["unlisted"]),
            "unlistedExamples": selection["unlisted"][:5],
            **(
                {"filename": spec.slideList.filename}
                if spec.slideList and spec.slideList.filename
                else {}
            ),
        }

    def _dataset_slides(self, records, finding):
        """Without a list or folder the dataset is the source: its own linked slide files."""
        rows = []
        for row in records:
            if not row.get("slidePath"):
                finding("SLIDE_MISSING", f"{row['slideId']}: no slide file is linked.")
                continue
            rows.append({"slideId": row["slideId"], "path": row["slidePath"], "mpp": None})
        return rows

    def _prepare(self, spec: ExtractionSpec) -> tuple[dict, list[dict]]:
        dataset = None
        if spec.datasetId is not None:
            dataset = self.store.get_dataset(spec.datasetId)
            if dataset["manifest"].get("kind") != "dataset":
                raise StorageError("Select a frozen imported dataset.", "INVALID_DATASET", 422)
        try:
            options = trident.TridentOptions.model_validate(spec.options)
        except ValidationError as error:
            raise StorageError(str(error), "INVALID_TRIDENT_OPTIONS", 422) from error
        # Freeze the resolved count in the preview and durable spec so command
        # generation and resource reservations use exactly the same workers.
        options = options.model_copy(
            update={"max_workers": resolve_max_workers(options.model_dump())}
        )
        values = options.model_dump(mode="json")
        output = self._path(spec.outputPath)
        # Outputs may be siblings of project data, never the project itself or its metadata.
        if protected_output(output, self.store.folder):
            raise StorageError(
                "Choose a dedicated TRIDENT output directory.", "INVALID_OUTPUT", 422
            )
        if any(
            all(
                (parent / name).is_file()
                for name in ("features.bin", "coords.bin", "index.parquet", "meta.json")
            )
            for parent in (output, *output.parents)
        ):
            raise StorageError(
                "An extraction output cannot be inside an existing feature pack.",
                "OUTPUT_IMMUTABLE",
                422,
            )
        findings = []

        def finding(code, message, severity="error"):
            findings.append({"severity": severity, "code": code, "message": message})

        records = (
            json.loads(self.store.read_artifact(spec.datasetId, "records.json"))
            if dataset is not None
            else None
        )
        rows, slide_list = self._select_slides(spec, values, dataset, records, finding)
        slides = []
        names = set()
        physical_slides = {}
        for row in rows:
            path = self._path(row["path"], exists=True)
            if not self.filesystem._contains(path) or not path.is_file():
                raise StorageError(
                    "Slide input is outside configured source roots.", "INVALID_SLIDE", 403
                )
            # TRIDENT names all slide outputs from their filename stem.
            name = path.stem
            if name in names:
                finding("DUPLICATE_SLIDE_NAME", f"TRIDENT output names collide for {name}.")
            names.add(name)
            if name != row["slideId"]:
                finding(
                    "SLIDE_ID_MISMATCH",
                    f"{row['slideId']}: filename stem {name} must match Slide_ID for native TRIDENT output binding.",
                )
            info = path.stat()
            physical = (info.st_dev, info.st_ino)
            if physical in physical_slides:
                finding(
                    "DUPLICATE_SLIDE_ALIAS",
                    f"{row['slideId']} and {physical_slides[physical]} refer to the same physical "
                    "slide file. Select one identity for that slide.",
                )
            else:
                physical_slides[physical] = row["slideId"]
            slides.append(
                {
                    "slideId": row["slideId"],
                    "path": str(path),
                    "name": name,
                    "size": info.st_size,
                    "mtimeNs": info.st_mtime_ns,
                    "ctimeNs": info.st_ctime_ns,
                    "deviceId": info.st_dev,
                    "inode": info.st_ino,
                    **({"mpp": row["mpp"]} if row.get("mpp") is not None else {}),
                }
            )
        if not slides:
            finding("NO_SLIDES", "The slide selection contains no readable slide.")
        if any(Path(row["path"]).is_relative_to(output) for row in slides):
            raise StorageError(
                "The output folder cannot contain source slides.", "INVALID_OUTPUT", 422
            )
        jobs = self._jobs()
        owned = any(item["outputPath"] == str(output) for item in jobs)
        previous_runs = [item for item in jobs if item["outputPath"] == str(output)]
        preprocessing = (
            "segmenter",
            "seg_conf_thresh",
            "remove_holes",
            "remove_artifacts",
            "remove_penmarks",
            "mag",
            "patch_size",
            "overlap",
            "min_tissue_proportion",
            "coords_dir",
            "reader_type",
            "custom_mpp_keys",
        )
        for previous in previous_runs:
            if previous["slides"] != slides or any(
                previous["spec"]["options"].get(key) != values.get(key) for key in preprocessing
            ):
                finding(
                    "OUTPUT_CONFIG_CHANGED",
                    "Source slides or preprocessing settings differ from this output's earlier run. Choose a new output folder.",
                )
                break
        marker = output / ".histopilot-output.json"
        input_files = []
        if options.patch_encoder_ckpt_path:
            checkpoint = self._path(options.patch_encoder_ckpt_path, exists=True)
            info = checkpoint.stat()
            if not stat.S_ISREG(info.st_mode):
                raise StorageError("Select a regular checkpoint file.", "INVALID_CHECKPOINT", 422)
            input_files.append(
                {
                    "path": str(checkpoint),
                    "sizeBytes": info.st_size,
                    "mtimeNs": info.st_mtime_ns,
                    "ctimeNs": info.st_ctime_ns,
                    "deviceId": info.st_dev,
                    "inode": info.st_ino,
                }
            )
        for previous in previous_runs:
            old = previous["spec"]["options"]
            if (
                old.get("task") in {"feat", "all"}
                and (old.get("slide_encoder") or old.get("patch_encoder"))
                == (values.get("slide_encoder") or values.get("patch_encoder"))
                and (
                    previous.get("inputFiles", []) != input_files
                    or any(
                        old.get(key) != values.get(key)
                        for key in ("patch_encoder_ckpt_path", "patch_encoder_img_size")
                    )
                )
            ):
                finding(
                    "ENCODER_CONFIG_CHANGED",
                    "Checkpoint or model input resolution changed for this encoder. Choose a new output folder to avoid reusing old embeddings.",
                )
        if output.is_dir() and marker.exists():
            ownership = _read(marker)
            if ownership.get("projectId") != self.store.project_id:
                finding("OUTPUT_OWNED", "This output folder belongs to another HistoPilot project.")
        if output.exists():
            if not output.is_dir():
                finding("OUTPUT_OCCUPIED", "The output path is not a directory.")
            elif any(output.iterdir()) and not owned:
                finding(
                    "OUTPUT_OCCUPIED",
                    "Choose an empty output folder. Attach existing TRIDENT features in the feature folder panel.",
                )
        if any(
            _overlaps(item["outputPath"], output)
            and self.get(item["id"], include_inactive=True)["state"] in ACTIVE
            for item in jobs
        ):
            finding("OUTPUT_BUSY", "An extraction is already using this output folder.")
        for key in ("patch_encoder_ckpt_path", "slide_encoder_ckpt_path", "custom_list_of_wsis"):
            if values.get(key):
                self._path(values[key], exists=True)
        # TRIDENT clears its cache. Commands receive a unique disposable child, never the parent.
        if values.get("wsi_cache"):
            cache = self._path(values["wsi_cache"])
            if cache.exists() and not cache.is_dir():
                raise StorageError("Select a directory for the cache parent.", "INVALID_CACHE", 422)
        if values.get("coords_dir"):
            coords = Path(values["coords_dir"])
            coords = self._path(str(coords if coords.is_absolute() else output / coords))
            if not coords.is_relative_to(output) or coords == output:
                raise StorageError(
                    "Coordinates must be in a subdirectory of this run's output.",
                    "INVALID_COORDS",
                    422,
                )
        runtime = trident.discover_runtime()
        if not runtime.get("available"):
            finding(
                "TRIDENT_UNAVAILABLE",
                runtime.get("error")
                or runtime.get("message")
                or "Configure a working TRIDENT Python environment and checkout.",
            )
        if options.max_workers == 0:
            finding(
                "TRIDENT_WORKERS_ZERO",
                "This TRIDENT CSV loader requires max_workers of at least 1. Leave it automatic or choose a positive count.",
            )
        devices = values.get("gpus") or [values.get("gpu", 0)]
        if len({device for device in devices if device >= 0}) > 1:
            finding(
                "SINGLE_GPU_TASK",
                "The Task Center runs an extraction on one GPU; the other selected GPUs stay "
                "available for other work.",
                "warning",
            )
        normalized = ExtractionSpec(
            datasetId=spec.datasetId,
            slideRoot=spec.slideRoot,
            slideList=spec.slideList,
            recursive=spec.recursive,
            outputPath=str(output),
            options=values,
        )
        layout = trident.output_layout(options, str(output))
        if values["task"] in {"coords", "feat"}:
            pattern = (
                layout["coordinatePattern"]
                if values["task"] == "feat"
                else str(Path(layout["geojsonDir"]) / "{slide}.geojson")
            )
            missing = sum(not Path(pattern.format(slide=row["name"])).is_file() for row in slides)
            if missing:
                finding(
                    "STAGE_INPUT_MISSING",
                    f"{missing} slides lack {'patch coordinates' if values['task'] == 'feat' else 'segmentation contours'}. Run the preceding stage or choose All stages.",
                )
        if runtime.get("available") and slides:
            self._reader_findings(runtime, slides, values, finding)
        estimated, available = self._space(values, layout, slides, output)
        if available is not None and available < min(estimated, MIN_FREE_BYTES):
            finding(
                "INSUFFICIENT_SPACE",
                f"The output volume is nearly full. Free at least "
                f"{min(estimated, MIN_FREE_BYTES) / 1024**3:.1f} GiB before extracting.",
            )
        elif available is not None and available < estimated:
            finding(
                "LOW_DISK_SPACE",
                f"This run may write about {estimated / 1024**3:.0f} GiB (an estimate), more "
                "than the output volume has free. Free space or choose another output folder.",
                "warning",
            )
        command = []
        if runtime.get("available") and slides:
            common = os.path.commonpath([str(Path(row["path"]).parent) for row in slides])
            command_options = options
            if options.wsi_cache:
                command_options = options.model_copy(
                    update={"wsi_cache": str(Path(options.wsi_cache) / "histopilot-{run}")}
                )
            try:
                command = trident.build_command(
                    command_options,
                    python_path=runtime["pythonPath"],
                    trident_root=runtime["tridentRoot"],
                    wsi_dir=common,
                    job_dir=str(output),
                    custom_list_of_wsis=str(self.folder / "{run}" / "slides.csv"),
                )
            except ValueError as error:
                finding("TRIDENT_OPTION_UNSUPPORTED", str(error))
        result = {
            "spec": normalized.model_dump(mode="json"),
            "slideCount": len(slides),
            "findings": findings,
            "canRun": not any(item["severity"] == "error" for item in findings),
            "runtime": runtime,
            "outputLayout": layout,
            "command": command,
            "customListSha256": slide_list["sha256"] if slide_list else None,
            "slideList": slide_list,
            "inputFiles": input_files,
            "estimatedBytes": estimated,
        }
        # Free space may change between two requests without invalidating the user's intent.
        result["previewHash"] = _hash({**result, "slides": slides})
        result["availableBytes"] = available
        return result, slides

    @staticmethod
    def _final_pattern(values: dict, layout: dict) -> str:
        """The last file TRIDENT writes for a slide in this run's task."""
        if values["task"] == "seg":
            return str(Path(layout["contoursDir"]) / "{slide}.jpg")
        if values["task"] == "coords":
            return layout["coordinatePattern"]
        return layout["featurePattern"]

    def _space(self, values: dict, layout: dict, slides: list, output: Path):
        """Estimated bytes still to write (slides without their final output) and free bytes."""
        pattern = self._final_pattern(values, layout)
        missing = [row for row in slides if not Path(pattern.format(slide=row["name"])).is_file()]
        estimated = estimate_output_bytes(
            values, sum(int(row.get("size") or 0) for row in missing), len(missing)
        )
        ancestor = output
        while not ancestor.exists() and ancestor != ancestor.parent:
            ancestor = ancestor.parent
        try:
            available = shutil.disk_usage(ancestor).free
        except OSError:
            available = None
        return estimated, available

    @staticmethod
    def _reader_findings(runtime: dict, slides: list, values: dict, finding) -> None:
        """TRIDENT imports each slide reader lazily, so a missing one fails mid-run."""
        readers = trident.slide_readers([row["path"] for row in slides], values.get("reader_type"))
        modules = {
            module: reader
            for reader in readers
            for module in trident.READER_MODULES.get(reader, ())
        }
        probe = trident.probe_reader_modules(
            runtime["pythonPath"], list(modules), cwd=runtime.get("tridentRoot")
        )
        if probe.get("error"):
            finding(
                "SLIDE_READER_UNCHECKED",
                f"{probe['error']}. TRIDENT checks its slide readers when it starts.",
                "warning",
            )
            return
        for module, error in sorted(probe.get("modules", {}).items()):
            if error:
                reader = modules[module]
                finding(
                    "SLIDE_READER_UNAVAILABLE",
                    f"TRIDENT's Python cannot import {module}, which the {reader} reader needs "
                    f"for {readers[reader]} selected slide(s): {error}",
                )

    def preview(self, spec: ExtractionSpec) -> dict:
        return self._prepare(spec)[0]

    def submit(self, spec: ExtractionSpec, preview_hash: str, operation_id: str) -> dict:
        with lifecycle_guard(self.store.folder, timeout=5):
            lifecycle = LifecycleStore(self.store.folder, self.store.project_id)
            lifecycle.assert_document_usable(spec.model_dump(mode="json"))
            return self._submit(spec, preview_hash, operation_id)

    def _submit(self, spec: ExtractionSpec, preview_hash: str, operation_id: str) -> dict:
        self.store.initialize()
        with writer_lock(self.store.folder):
            for existing in self._jobs():
                if existing["operationId"] == operation_id:
                    if (
                        existing["requestHash"] != _hash(spec.model_dump(mode="json"))
                        or existing["previewHash"] != preview_hash
                    ):
                        raise StorageError(
                            "Operation ID was used for another extraction.", "OPERATION_CONFLICT"
                        )
                    return self.get(existing["id"])
        # Dataset store calls take their own writer lock; validation must be outside it.
        preview, slides = self._prepare(spec)
        if preview["previewHash"] != preview_hash:
            raise StorageError("Extraction inputs changed. Preview again.", "PREVIEW_STALE")
        if not preview["canRun"]:
            raise StorageError(
                "Resolve extraction preflight findings before starting.", "EXTRACTION_INVALID", 422
            )
        with writer_lock(self.store.folder):
            # Serialize claims of an output directory and idempotency keys.
            for existing in self._jobs():
                if existing["operationId"] == operation_id:
                    if (
                        existing["requestHash"] != _hash(spec.model_dump(mode="json"))
                        or existing["previewHash"] != preview_hash
                    ):
                        raise StorageError(
                            "Operation ID was used for another extraction.", "OPERATION_CONFLICT"
                        )
                    return self.get(existing["id"])
                if (
                    _overlaps(existing["outputPath"], preview["spec"]["outputPath"])
                    and self.get(existing["id"], include_inactive=True)["state"] in ACTIVE
                ):
                    raise StorageError("An extraction already owns this output.", "OUTPUT_BUSY")
            identity = f"extraction-{uuid4().hex}"
            folder = self.folder / identity
            ensure_managed_directory(folder)
            output = self._path(preview["spec"]["outputPath"])
            ensure_managed_directory(output)
            marker = output / ".histopilot-output.json"
            if marker.exists():
                if _read(marker).get("projectId") != self.store.project_id:
                    raise StorageError("Output is owned by another project.", "OUTPUT_BUSY")
            else:
                if any(output.iterdir()):
                    raise StorageError("Output contents changed after preview.", "PREVIEW_STALE")
                try:
                    ScientificStore._write_file(
                        marker, json.dumps({"projectId": self.store.project_id}).encode()
                    )
                except FileExistsError as error:
                    raise StorageError("Another run claimed this output.", "OUTPUT_BUSY") from error
                fsync_directory(output)
            root = Path(os.path.commonpath([str(Path(row["path"]).parent) for row in slides]))
            stream = io.StringIO()
            writer = csv.writer(stream)
            # TRIDENT compacts the mpp column before pairing it with slides, so an
            # all-or-nothing column is the only one that keeps each value on its own slide.
            declared = sum("mpp" in row for row in slides)
            if declared and declared != len(slides):
                raise StorageError(
                    "Every selected slide needs a declared MPP, or none may have one.",
                    "INVALID_SLIDE_LIST",
                    422,
                )
            has_mpp = bool(declared)
            writer.writerow(["wsi", "mpp"] if has_mpp else ["wsi"])
            writer.writerows(
                [
                    [
                        str(Path(row["path"]).relative_to(root)),
                        *([row["mpp"]] if has_mpp else []),
                    ]
                    for row in slides
                ]
            )
            ScientificStore._write_file(folder / "slides.csv", stream.getvalue().encode())
            options = trident.TridentOptions.model_validate(preview["spec"]["options"])
            if options.wsi_cache:
                cache = self._path(str(Path(options.wsi_cache) / f"histopilot-{identity}"))
                if cache.exists():
                    raise StorageError(
                        "The per-run cache is unexpectedly occupied.", "INVALID_CACHE"
                    )
                ensure_managed_directory(cache.parent)
                options = options.model_copy(update={"wsi_cache": str(cache)})
            runtime = preview["runtime"]
            values = options.model_dump(mode="json")
            lane = "gpu" if uses_gpu(values) else "cpu"
            command_options = options
            if lane == "gpu":
                # The runner exposes the GPU it admits the task on as CUDA device 0.
                command_options = options.model_copy(update={"gpu": 0, "gpus": None})
            command = trident.build_command(
                command_options,
                python_path=runtime["pythonPath"],
                trident_root=runtime["tridentRoot"],
                wsi_dir=str(root),
                job_dir=str(output),
                custom_list_of_wsis=str(folder / "slides.csv"),
            )
            job = {
                "id": identity,
                "projectId": self.store.project_id,
                "state": "queued",
                "operationId": operation_id,
                "requestHash": _hash(spec.model_dump(mode="json")),
                "previewHash": preview_hash,
                "spec": preview["spec"],
                "outputPath": str(output),
                "outputLayout": preview["outputLayout"],
                "slideCount": len(slides),
                "slides": slides,
                "command": command,
                "runtime": runtime,
                "inputFiles": preview["inputFiles"],
                "sessionName": None,
                "logPath": str(folder / "worker.log"),
                "createdAt": _now(),
                "updatedAt": _now(),
                "executionMode": task_records.TASK_CENTER,
                "taskId": ids.task_id("extraction", str(folder)),
                "validationTaskId": ids.task_id("extraction-validation", str(folder)),
                "ownerKey": task_records.owner_key("extraction", identity, self.store),
            }
            _write(folder / "job.json", job)
            plan = {
                "command": command,
                "resultPath": str(folder / "result.json"),
                "logPath": job["logPath"],
                "cancelPath": str(folder / "cancelled"),
                "processPath": str(folder / "process.json"),
                "cwd": runtime["tridentRoot"],
                # Each attempt owns its output: locks of dead TRIDENT writers are cleared
                # before TRIDENT starts, or it would skip those slides (resume).
                "clearDeadLocks": {
                    "root": str(output),
                    "maxAgeHours": options.dead_lock_max_age_hours,
                },
                "managed": True,
                "jobPath": str(folder / "job.json"),
                "progressPath": str(folder / "progress.json"),
                "peakPath": str(folder / "trident-peak.json"),
            }
            _write(folder / "plan.json", plan)
            try:
                self._enqueue(job, folder, values, lane, preview["estimatedBytes"])
            except (StorageError, OSError) as error:
                job["state"], job["error"] = (
                    "failed",
                    f"Could not queue the extraction in the Task Center: {error}",
                )
                _write(folder / "job.json", job)
        return self.get(identity)

    def _enqueue(self, job: dict, folder: Path, values: dict, lane: str, estimated: int) -> None:
        """Queue TRIDENT (GPU lane) and its artifact validation (CPU lane, after it)."""
        identity, count = job["id"], job["slideCount"]
        encoder = values.get("slide_encoder") or values.get("patch_encoder")
        stage = {"seg": "Segmentation", "coords": "Patch coordinates"}.get(values["task"])
        subject = stage or f"Extraction · {encoder}"
        labels = {
            "recordKind": "extraction",
            "recordId": identity,
            "projectId": self.store.project_id,
            "extractionId": identity,
            "encoder": encoder,
            "slideCount": count,
        }
        workers = resolve_max_workers(values)
        devices = 1 if lane == "gpu" else execution_device_count(values)
        environment = {"HISTOPILOT_TASK_MANAGED": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        extraction = {
            "id": job["taskId"],
            "kind": "extraction",
            "adapter": "extraction",
            "title": f"{subject} · {count} slides"[:200],
            "group": {"kind": "extraction", "id": identity},
            "labels": {**labels, "phase": "extraction"},
            "exclusiveKey": "trident-output:"
            + hashlib.sha256(job["outputPath"].encode()).hexdigest(),
            "request": {
                "lane": lane,
                "cpuThreads": devices,
                "dataWorkers": min(256, workers * devices),
                "ramGb": estimate_ram_gb(values) * devices,
                "vramGb": estimate_vram_gb(values) if lane == "gpu" else 0.0,
                "graceSeconds": EXTRACTION_GRACE_SECONDS,
                "workloadKey": workload_key(values),
            },
            "command": {
                "argv": [
                    sys.executable,
                    "-u",
                    str(Path(trident.__file__).with_name("runner.py")),
                    str(folder / "plan.json"),
                ],
                "cwd": job["runtime"]["tridentRoot"],
                "env": environment,
                "log": job["logPath"],
                "progress": str(folder / "progress.json"),
                "result": str(folder / "result.json"),
            },
            "adapterData": {
                "extractionFolder": str(folder),
                "jobId": identity,
                "validationTaskId": job["validationTaskId"],
                "outputPath": job["outputPath"],
                "estimatedOutputBytes": int(estimated),
            },
        }
        validation = {
            "id": job["validationTaskId"],
            "kind": "extraction-validation",
            "adapter": "extraction-validation",
            "title": f"Extraction validation · {count} slides",
            "group": {"kind": "extraction", "id": identity},
            "planOrder": 1,
            "labels": {**labels, "phase": "validation"},
            "request": {
                "lane": "cpu",
                "cpuThreads": 1,
                "dataWorkers": 0,
                "ramGb": 2.0,
                "graceSeconds": VALIDATION_GRACE_SECONDS,
            },
            "command": {
                "argv": [
                    sys.executable,
                    "-u",
                    "-m",
                    "histopilot.workers.verify_extraction",
                    str(folder / "job.json"),
                    str(folder / "validation.json"),
                    str(folder / "validation-progress.json"),
                ],
                "cwd": str(REPOSITORY_ROOT),
                "env": environment,
                "log": job["logPath"],
                "progress": str(folder / "validation-progress.json"),
                "result": str(folder / "validation.json"),
            },
            "dependsOn": [{"task": job["taskId"], "condition": "succeeded"}],
            "adapterData": {
                "extractionFolder": str(folder),
                "jobId": identity,
                "extractionTaskId": job["taskId"],
            },
        }
        owner = task_records.owner(
            "extraction",
            identity,
            f"{subject} · {count} slides",
            self.store,
            {"recordKind": "extraction"},
        )
        task_records.enqueue(self.tasks, owner, [extraction, validation])

    def get(self, identity: str, *, logs=False, include_inactive=False) -> dict:
        if not include_inactive:
            LifecycleStore(self.store.folder, self.store.project_id).assert_usable(
                [f"extraction:{identity}"]
            )
        job = self._job_record(identity)
        folder = self.folder / identity
        if task_records.managed_record(job):
            self._task_state(job, folder)
        elif (folder / "result.json").exists():
            # A job from before the Task Center: its saved outcome, never its tmux session.
            result = _read(folder / "result.json")
            job["result"] = result
            job["state"] = result.get("state", "failed")
            job["updatedAt"] = result.get("finishedAt", job["updatedAt"])
            if result.get("error"):
                job["error"] = result["error"]
            if job["state"] == "succeeded" or (folder / "validation.json").exists():
                self._coverage(job)
        else:
            job = task_records.legacy_state(job, ACTIVE)
        # Old workers need no restart: derive progress from their existing log.
        # Polling never opens slide tensors or rescans the output directory.
        log_warning = None
        try:
            output, output_updated_at = _log_tail(folder / "worker.log")
        except (OSError, StorageError):
            # Diagnostic output is optional; it must never prevent cancellation
            # or hide the durable outcome of this or any other job in the list.
            output, output_updated_at = "", None
            log_warning = (
                "Progress details are unavailable because the worker log cannot be read safely."
            )
        job["progress"] = build_progress({**job, "_logUpdatedAt": output_updated_at}, output)
        if log_warning:
            job["progress"]["warnings"].append(log_warning)
        if logs:
            job["logs"] = output
        return {
            key: value
            for key, value in job.items()
            if key not in {"slides", "requestHash", "operationId"}
        }

    def _task_state(self, job: dict, folder: Path) -> None:
        """Derive a Task Center job's state from its two tasks (the store, not processes)."""
        extraction, validation = self.tasks.views([job.get("taskId"), job.get("validationTaskId")])
        job["executor"] = "task-center"
        job["tasks"] = {
            "extraction": task_records.public_view(extraction),
            "validation": task_records.public_view(validation),
        }
        extracted = bool(extraction) and extraction.get("state") == "succeeded"
        current = validation if extracted else extraction
        job["task"] = task_records.public_view(current)
        receipt = _read(folder / "result.json") if (folder / "result.json").exists() else None
        if (
            receipt
            and extraction
            and receipt.get("taskId") == extraction.get("id")
            and receipt.get("taskAttempt") == extraction.get("attempt")
        ):
            job["result"] = receipt
            job["updatedAt"] = receipt.get("finishedAt", job["updatedAt"])
        if current is None:
            if job["state"] in ACTIVE:
                job["state"] = "interrupted"
                job["error"] = (
                    "The Task Center has no task for this extraction. Preview a resume run."
                )
            return
        if current.get("unknown"):
            # An unreadable store never reads as stopped: keep the recorded state.
            job["waitingReason"] = task_records.waiting_reason(current)
            return
        state = task_records.record_state(current)
        if extracted and state in {"queued", "starting"}:
            state = "running"  # TRIDENT finished; its validation waits for a CPU slot
        job["state"] = state
        if task_records.pending(current):
            job["waitingReason"] = task_records.waiting_reason(current)
        if state in {"failed", "cancelled", "interrupted"}:
            job["error"] = (
                current.get("error") or job.get("error") or f"The extraction was {state}."
            )
        if extracted and validation and validation["state"] in {"succeeded", "failed"}:
            # Coverage of this validation attempt's own report (never an older one).
            path = folder / "validation.json"
            report = _read(path) if path.exists() else None
            if report is not None and report.get("taskAttempt") == validation["attempt"]:
                job.setdefault("result", {})
                self._coverage(job)

    def _coverage(self, job: dict) -> None:
        from histopilot.application.extraction_artifacts import complete_coverage

        path = self.folder / job["id"] / "validation.json"
        if not path.exists():
            job["state"] = "failed"
            job["error"] = (
                "The worker did not publish an artifact validation report. Inspect its log before resuming."
            )
            return
        coverage = _read(path)
        if coverage.get("jobId") != job["id"]:
            raise StorageError(
                "Output validation belongs to a different job.", "EXTRACTION_CORRUPT"
            )
        job["result"].update(
            {
                key: value
                for key, value in coverage.items()
                if key not in {"startedAt", "finishedAt"}
            }
        )
        job["result"]["validationStartedAt"] = coverage.get("startedAt")
        job["result"]["validationFinishedAt"] = coverage.get("finishedAt")
        if job["state"] == "succeeded" and not complete_coverage(coverage, job["slideCount"]):
            job["state"] = "failed"
            job["error"] = (
                f"Artifact validation does not confirm all {job['slideCount']} slides: "
                f"{coverage.get('missingSlides', 0)} slides lack valid outputs; "
                f"{coverage.get('unvalidatedSlides', 0)} remain unvalidated. "
                "Inspect worker logs and artifact findings; skipped errors do not count as success."
            )

    def list(self, *, include_inactive=False) -> dict:
        states = LifecycleStore(self.store.folder, self.store.project_id).read()["records"]
        jobs = [self.get(item["id"], include_inactive=True) for item in self._jobs()]
        return {
            "jobs": sorted(
                [
                    job
                    for job in jobs
                    if include_inactive
                    or job["state"] in ACTIVE
                    or states.get(f"extraction:{job['id']}", {}).get("state", "active") == "active"
                ],
                key=lambda item: item["createdAt"],
                reverse=True,
            )
        }

    def cancel(self, identity: str) -> dict:
        with lifecycle_guard(self.store.folder, timeout=5):
            return self._cancel(identity)

    def _cancel(self, identity: str) -> dict:
        with writer_lock(self.store.folder):
            job = self.get(identity, include_inactive=True)
            if not task_records.managed_record(job):
                task_records.refuse_legacy()
            return self._cancel_tasks(identity, job)

    def resume(self, identity: str) -> dict:
        """Run a stopped Task Center extraction again: TRIDENT, then a fresh validation.

        TRIDENT skips slides whose outputs are complete, and every attempt first clears
        the locks of dead writers, so this resumes where the last attempt stopped. A job
        recorded before the Task Center cannot resume; a new preview on its output folder
        continues its work.
        """
        with lifecycle_guard(self.store.folder, timeout=5):
            LifecycleStore(self.store.folder, self.store.project_id).assert_usable(
                [f"extraction:{identity}"]
            )
            with writer_lock(self.store.folder):
                job = self.get(identity)
                if not task_records.managed_record(job):
                    task_records.refuse_legacy()
                if job["state"] in ACTIVE or job["state"] == "succeeded":
                    return job
                # As at submit: the task's exclusive key serializes only identical folders,
                # and this job's dead-lock sweep must never walk a live job's nested output.
                for other in self._jobs():
                    if (
                        other["id"] != identity
                        and _overlaps(other["outputPath"], job["outputPath"])
                        and self.get(other["id"], include_inactive=True)["state"] in ACTIVE
                    ):
                        raise StorageError(
                            "Another extraction is using an overlapping output folder. "
                            "Resume this one after it finishes.",
                            "OUTPUT_BUSY",
                            409,
                        )
                try:
                    requeued = self.tasks.client.store.requeue(
                        [job["taskId"]], reason="resume", include_succeeded=True
                    )
                except (StorageError, OSError) as error:
                    raise StorageError(
                        f"The Task Center cannot resume this extraction: {error}",
                        "TASK_CENTER_UNAVAILABLE",
                        503,
                    ) from error
                if not requeued:
                    raise StorageError(
                        "This extraction is still running in the Task Center.",
                        "EXTRACTION_ACTIVE",
                        409,
                    )
                self.tasks.wake()
        return self.get(identity)

    def _cancel_tasks(self, identity: str, job: dict) -> dict:
        """Mark the cancel (the worker stops on it too), then cancel both tasks.

        The marker names the attempts it cancels, so a later retry from the Task Center
        runs instead of finding a stale cancel.
        """
        if job["state"] not in ACTIVE:
            return self.get(identity, logs=True, include_inactive=True)
        views = self.tasks.views([job.get("taskId"), job.get("validationTaskId")])
        _write(
            self.folder / identity / "cancelled",
            {
                "requestedAt": _now(),
                "attempts": {
                    view["id"]: view["attempt"]
                    for view in views
                    if view and not view.get("unknown")
                },
            },
        )
        for view in views:
            if view and task_records.live(view) and not view.get("unknown"):
                try:
                    self.tasks.client.cancel_task(view["id"])
                except (StorageError, OSError):
                    pass  # the worker still stops on the marker
        return self.get(identity, logs=True, include_inactive=True)
