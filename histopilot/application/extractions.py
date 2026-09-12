"""Durable, project-scoped TRIDENT extraction with server-derived slide manifests."""

import csv
import hashlib
import io
import json
import math
import os
import re
import signal
import stat
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from histopilot.adapters import trident
from histopilot.adapters.trident.progress import build_progress
from histopilot.schemas.extractions import ExtractionSpec
from histopilot.storage.filesystem import LocalFilesystem
from histopilot.storage.lifecycle import LifecycleStore, lifecycle_guard
from histopilot.storage.project_lock import (
    StorageError,
    _reject_symlink_components,
    ensure_managed_directory,
    fsync_directory,
    writer_lock,
)
from histopilot.storage.scientific import ScientificStore
from histopilot.workers.extraction_process import TmuxExtractionExecutor

ACTIVE = {"starting", "running", "cancelling"}
JOB_ID = re.compile(r"^extraction-[a-f0-9]{32}$")
MAX_JSON = 8 * 1024 * 1024
MAX_LOG_BYTES = 64 * 1024


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _overlaps(first: str | Path, second: str | Path) -> bool:
    first, second = Path(first), Path(second)
    return first.is_relative_to(second) or second.is_relative_to(first)


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
    _reject_symlink_components(path)
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
    _reject_symlink_components(path)
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
    def __init__(self, store: ScientificStore, filesystem: LocalFilesystem, executor=None):
        self.store = store
        self.filesystem = filesystem
        self.outputs = LocalFilesystem((store.folder, *filesystem.roots))
        self.executor = executor or TmuxExtractionExecutor()
        self.folder = store.folder / "extractions"

    def catalog(self) -> dict:
        return {
            **trident.option_catalog(),
            "runtime": trident.discover_runtime(),
            "tmuxAvailable": self.executor.available(),
            "defaultOutputPath": str(self.store.folder / "trident"),
        }

    def _path(self, value: str, *, exists=False, directory=False) -> Path:
        path = Path(value)
        if not path.is_absolute() or "\x00" in value or ".." in path.parts:
            raise StorageError(
                "Use an absolute path without parent traversal.", "INVALID_PATH", 422
            )
        _reject_symlink_components(path)
        resolved = path.resolve()
        if not self.outputs._contains(resolved):
            raise StorageError("The path is outside configured data roots.", "INVALID_PATH", 403)
        if exists and (not resolved.exists() or (directory and not resolved.is_dir())):
            raise StorageError("The selected input path does not exist.", "INVALID_PATH", 422)
        return resolved

    def _jobs(self) -> list[dict]:
        if not self.folder.exists():
            return []
        _reject_symlink_components(self.folder)
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

    def _select_csv(self, records: list[dict], value: str) -> tuple[list[dict], str]:
        path = self._path(value, exists=True)
        content = ScientificStore._read_file(path, 2 * 1024 * 1024)
        try:
            table = csv.DictReader(io.StringIO(content.decode("utf-8-sig")))
            if "wsi" not in (table.fieldnames or []):
                raise ValueError("CSV needs a wsi column and optionally mpp")
            available = [row for row in records if row.get("slidePath")]
            common = (
                Path(os.path.commonpath([str(Path(row["slidePath"]).parent) for row in available]))
                if available
                else None
            )
            selected, seen = [], set()
            for item in table:
                name = item.get("wsi", "").strip()
                matches = [
                    row
                    for row in available
                    if name
                    in {
                        row["slidePath"],
                        Path(row["slidePath"]).name,
                        str(Path(row["slidePath"]).relative_to(common)),
                    }
                ]
                if len(matches) != 1 or matches[0]["slideId"] in seen:
                    raise ValueError(
                        f"CSV slide {name!r} is missing, ambiguous, duplicated or outside the frozen dataset"
                    )
                row = dict(matches[0])
                seen.add(row["slideId"])
                if item.get("mpp", "").strip():
                    mpp = float(item["mpp"])
                    if not math.isfinite(mpp) or mpp <= 0:
                        raise ValueError("mpp must be a positive finite number")
                    row["customMpp"] = mpp
                selected.append(row)
            if not selected:
                raise ValueError("CSV contains no selected slides")
            return selected, hashlib.sha256(content).hexdigest()
        except (ValueError, TypeError, UnicodeError, csv.Error) as error:
            raise StorageError(
                f"Invalid TRIDENT slide CSV: {error}", "INVALID_SLIDE_LIST", 422
            ) from error

    def _prepare(self, spec: ExtractionSpec) -> tuple[dict, list[dict]]:
        dataset = self.store.get_dataset(spec.datasetId)
        if dataset["manifest"].get("kind") != "dataset":
            raise StorageError("Select a frozen imported dataset.", "INVALID_DATASET", 422)
        try:
            options = trident.TridentOptions.model_validate(spec.options)
        except ValidationError as error:
            raise StorageError(str(error), "INVALID_TRIDENT_OPTIONS", 422) from error
        values = options.model_dump(mode="json")
        output = self._path(spec.outputPath)
        # Outputs may be siblings of project data, never the project itself or its metadata.
        protected = [
            self.store.folder / name
            for name in (
                "datasets",
                "configurations",
                "extractions",
                "packing",
                "training",
                "compute-jobs",
                "predictor-builds",
                "evaluation-batches",
                "jobs",
                "drafts",
                ".staging",
                ".trash",
                ".git",
                ".codex",
                ".histopilot-write.lock",
                ".histopilot-lifecycle.lock",
                "histopilot-lifecycle.json",
                "histopilot-project.json",
                "histopilot-state.sqlite",
                "histopilot-state.sqlite-wal",
                "histopilot-state.sqlite-shm",
            )
        ]
        if output == self.store.folder or any(
            output.is_relative_to(path) or path.is_relative_to(output) for path in protected
        ):
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

        def finding(code, message):
            findings.append({"severity": "error", "code": code, "message": message})

        records = json.loads(self.store.read_artifact(spec.datasetId, "records.json"))
        csv_hash = None
        if values.get("custom_list_of_wsis"):
            records, csv_hash = self._select_csv(records, values["custom_list_of_wsis"])
        slides = []
        names = set()
        for row in records:
            if not row.get("slidePath"):
                finding("SLIDE_MISSING", f"{row['slideId']}: no slide file is linked.")
                continue
            path = self._path(row["slidePath"], exists=True)
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
                    **({"mpp": row["customMpp"]} if "customMpp" in row else {}),
                }
            )
        if not slides:
            finding("NO_SLIDES", "No readable slides are linked to this dataset.")
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
        if not self.executor.available():
            finding("TMUX_UNAVAILABLE", "Install tmux to run persistent extraction jobs.")
        if options.max_workers == 0:
            finding(
                "TRIDENT_WORKERS_ZERO",
                "This TRIDENT CSV loader requires max_workers of at least 1. Leave it automatic or choose a positive count.",
            )
        normalized = ExtractionSpec(
            datasetId=spec.datasetId, outputPath=str(output), options=values
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
            "canRun": not findings,
            "runtime": runtime,
            "outputLayout": layout,
            "command": command,
            "customListSha256": csv_hash,
            "inputFiles": input_files,
        }
        result["previewHash"] = _hash({**result, "slides": slides})
        return result, slides

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
            has_mpp = any("mpp" in row for row in slides)
            writer.writerow(["wsi", "mpp"] if has_mpp else ["wsi"])
            writer.writerows(
                [
                    [
                        str(Path(row["path"]).relative_to(root)),
                        *([row.get("mpp", "")] if has_mpp else []),
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
            command = trident.build_command(
                options,
                python_path=runtime["pythonPath"],
                trident_root=runtime["tridentRoot"],
                wsi_dir=str(root),
                job_dir=str(output),
                custom_list_of_wsis=str(folder / "slides.csv"),
            )
            job = {
                "id": identity,
                "projectId": self.store.project_id,
                "state": "starting",
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
                "sessionName": f"histopilot-pfm-{identity.removeprefix('extraction-')}",
                "logPath": str(folder / "worker.log"),
                "createdAt": _now(),
                "updatedAt": _now(),
            }
            _write(folder / "job.json", job)
            _write(
                folder / "plan.json",
                {
                    "command": command,
                    "resultPath": str(folder / "result.json"),
                    "logPath": job["logPath"],
                    "cancelPath": str(folder / "cancelled"),
                    "processPath": str(folder / "process.json"),
                    "validationCommand": [
                        sys.executable,
                        "-m",
                        "histopilot.workers.verify_extraction",
                        str(folder / "job.json"),
                        str(folder / "validation.json"),
                    ],
                    "cwd": runtime["tridentRoot"],
                },
            )
            try:
                self.executor.launch(
                    job["sessionName"],
                    Path(trident.__file__).with_name("runner.py"),
                    folder / "plan.json",
                )
                job["state"] = "running"
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                try:
                    started = (
                        self.executor.running(job["sessionName"])
                        or self._live_process(folder)
                        or (folder / "result.json").exists()
                    )
                except (OSError, RuntimeError, subprocess.SubprocessError):
                    job["state"] = "starting"
                    job["error"] = (
                        "Launch acknowledgement was lost. Check worker status before retrying."
                    )
                else:
                    if started:
                        job["state"] = "running"
                    else:
                        job["state"], job["error"] = "failed", f"Could not start TRIDENT: {error}"
            _write(folder / "job.json", job)
        return self.get(identity)

    def get(self, identity: str, *, logs=False, include_inactive=False) -> dict:
        if not include_inactive:
            LifecycleStore(self.store.folder, self.store.project_id).assert_usable(
                [f"extraction:{identity}"]
            )
        job = self._job_record(identity)
        folder = self.folder / identity
        if (folder / "result.json").exists():
            result = _read(folder / "result.json")
            job["result"] = result
            job["state"] = result.get("state", "failed")
            job["updatedAt"] = result.get("finishedAt", job["updatedAt"])
            if result.get("error"):
                job["error"] = result["error"]
            if job["state"] == "succeeded" or (folder / "validation.json").exists():
                self._coverage(job)
        elif (folder / "cancelled").exists():
            job["state"] = (
                "cancelling"
                if self.executor.running(job["sessionName"]) or self._live_process(folder)
                else "cancelled"
            )
        elif job["state"] in ACTIVE:
            try:
                if not self.executor.running(job["sessionName"]) and not self._live_process(folder):
                    job["state"] = "interrupted"
                    job["error"] = (
                        "The tmux session ended without a completion record. Inspect the log, then preview a resume run."
                    )
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                job["error"] = f"Cannot inspect worker status: {error}"
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

    @staticmethod
    def _live_process(folder: Path) -> dict | None:
        path = folder / "process.json"
        if not path.exists():
            return None
        process = _read(path)
        pid = process.get("pid")
        if type(pid) is not int or pid <= 1:
            return None
        try:
            boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            if (
                process.get("bootId") == boot
                and process.get("startTicks") == int(fields[19])
                and fields[0] != "Z"
            ):
                return process
        except (OSError, ValueError, IndexError):
            pass
        return None

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
            if (
                job["state"] not in ACTIVE
                and not self.executor.running(job["sessionName"])
                and not self._live_process(self.folder / identity)
            ):
                return job
            marker = self.folder / identity / "cancelled"
            if not marker.exists():
                ScientificStore._write_file(marker, b"cancelled\n")
                fsync_directory(marker.parent)
            # The standalone runner observes this durable marker and terminates its process group.
            if not self.executor.running(job["sessionName"]):
                process = self._live_process(marker.parent)
                if process:
                    # The original runner may have died. Reconcile the exact child identity
                    # before signaling its group; PID reuse and reboots cannot target another job.
                    try:
                        os.killpg(process["pid"], signal.SIGTERM)
                    except ProcessLookupError:
                        pass
        return self.get(identity, logs=True, include_inactive=True)
