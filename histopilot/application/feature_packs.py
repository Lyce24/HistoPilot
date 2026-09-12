"""Optional feature validation and materialization jobs, bound to frozen inventories."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import shutil
import stat
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from histopilot.application.features import FeatureService, _stamp
from histopilot.schemas.feature_packs import FeaturePackSpec
from histopilot.schemas.features import FeatureSpec
from histopilot.storage.filesystem import LocalFilesystem
from histopilot.storage.lifecycle import LifecycleStore, lifecycle_guard
from histopilot.storage.project_lock import (
    StorageError,
    _reject_symlink_components,
    ensure_managed_directory,
    writer_lock,
)
from histopilot.storage.scientific import ScientificStore
from histopilot.workers.packing_process import (
    TmuxPackingExecutor,
    live_process,
    output_key,
    output_lock,
    registry_lock,
    write_json,
)

ACTIVE = {"starting", "running", "cancelling"}
JOB_ID = re.compile(r"^packing-[a-f0-9]{32}$")
MAX_JSON = 64 * 1024 * 1024
_UNREAD = object()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _read(path: Path) -> dict:
    try:
        value = json.loads(ScientificStore._read_file(path, MAX_JSON))
        if not isinstance(value, dict):
            raise ValueError("Expected an object")
        return value
    except (ValueError, UnicodeError) as error:
        raise StorageError("Packing metadata is invalid.", "PACKING_CORRUPT") from error


def _overlaps(first: str | Path, second: str | Path) -> bool:
    first, second = Path(first), Path(second)
    return first.is_relative_to(second) or second.is_relative_to(first)


def _compact(value: dict | None) -> dict | None:
    if value is None:
        return None
    result = {
        key: item
        for key, item in value.items()
        if key
        not in {"files", "slides", "semanticIdentity", "sourceStamps", "provenance", "packStamps"}
    }
    if isinstance(result.get("validation"), dict):
        result["validation"] = _compact(result["validation"])
    return result


class FeaturePackService:
    def __init__(self, store: ScientificStore, filesystem: LocalFilesystem, executor=None):
        self.store = store
        self.filesystem = filesystem
        self.outputs = LocalFilesystem((store.folder, *filesystem.roots))
        self.executor = executor or TmuxPackingExecutor()
        self.folder = store.folder / "packing"

    @staticmethod
    def format_available() -> bool:
        return all(importlib.util.find_spec(module) is not None for module in ("numpy", "pyarrow"))

    def _jobs(self) -> list[dict]:
        if not self.folder.exists():
            return []
        _reject_symlink_components(self.folder)
        paths = sorted(self.folder.glob("packing-*/job.json"))
        if len(paths) > 10000:
            raise StorageError("Too many packing jobs.", "PACKING_LIMIT", 413)
        return [self._job_record(path.parent.name) for path in paths]

    def _job_record(self, identity: str) -> dict:
        if not JOB_ID.fullmatch(identity):
            raise StorageError("Packing job not found.", "PACKING_NOT_FOUND", 404)
        path = self.folder / identity / "job.json"
        if not path.exists():
            raise StorageError("Packing job not found.", "PACKING_NOT_FOUND", 404)
        job = _read(path)
        if job.get("id") != identity or job.get("projectId") != self.store.project_id:
            raise StorageError("Packing job metadata is inconsistent.", "PACKING_CORRUPT")
        return job

    def _path(self, value: str, configuration: dict) -> Path:
        path = Path(value)
        if not path.is_absolute() or "\x00" in value or ".." in path.parts:
            raise StorageError("Use an absolute path without traversal.", "INVALID_PATH", 422)
        _reject_symlink_components(path)
        path = path.resolve()
        if not self.outputs._contains(path):
            raise StorageError("Output is outside configured data roots.", "INVALID_PATH", 403)
        reserved = (
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
            "project.sqlite3",
            "histopilot.sqlite3",
        )
        if (
            path == self.store.folder
            or path in self.outputs.roots
            or any(_overlaps(path, self.store.folder / name) for name in reserved)
        ):
            raise StorageError("Choose a dedicated feature pack directory.", "INVALID_OUTPUT", 422)
        manifest = configuration["manifest"]
        source_paths = [manifest.get("spec", {}).get(key) for key in ("path", "coordinatesPath")]
        source_paths += [
            item.get(key) for item in manifest["files"] for key in ("path", "coordinatePath")
        ]
        source_paths += [item["path"] for item in manifest.get("provenance", [])]
        if any(source and _overlaps(path, source) for source in source_paths):
            raise StorageError(
                "Output cannot overlap source features or provenance.", "INVALID_OUTPUT", 422
            )
        # Recognize published packs even after temporary lock claims have been cleaned.
        # Creating a child would change the exact membership of the existing artifact.
        for parent in path.parents:
            if all(
                (parent / name).is_file()
                for name in ("manifest.json", "features.bin", "index.parquet")
            ):
                raise StorageError(
                    "A new output cannot be inside an existing feature pack.",
                    "OUTPUT_IMMUTABLE",
                    422,
                )
        return path

    def _auto_output(self, spec: FeaturePackSpec) -> str:
        base = self.store.folder / "feature-packs" / f"{spec.featureSetId[-16:]}-{spec.dtype}"
        claimed = {item["outputPath"] for item in self._jobs() if item.get("outputPath")}
        path, number = base, 1
        while path.exists() or str(path) in claimed:
            number += 1
            path = base.with_name(f"{base.name}-{number}")
        return str(path)

    def _claim_busy(self, claim: dict) -> bool:
        folder = Path(claim["jobPath"]).parent
        # Check the actual process even when a service wrote a failed launch status.
        if live_process(folder) or self.executor.running(claim["sessionName"]):
            return True
        if (folder / "result.json").exists() or (folder / "cancelled").exists():
            return False
        if not (folder / "job.json").exists():
            return False
        job = _read(folder / "job.json")
        age = (datetime.now(UTC) - datetime.fromisoformat(job["createdAt"])).total_seconds()
        return job.get("state") == "starting" and age < 30

    def _claims(self, output: Path, folder: Path) -> list[dict]:
        claims = []
        for path in folder.glob("*.claim.json"):
            claim = _read(path)
            if _overlaps(output, claim["outputPath"]):
                result_path = Path(claim["jobPath"]).parent / "result.json"
                result = _read(result_path) if result_path.exists() else {}
                published = (
                    result.get("state") == "succeeded"
                    and result.get("artifact")
                    and Path(claim["outputPath"]).exists()
                )
                if published or self._claim_busy(claim):
                    claims.append(claim)
        return claims

    def _existing_path(self, value: str) -> Path:
        path = Path(value)
        if not path.is_absolute() or "\x00" in value or ".." in path.parts:
            raise StorageError(
                "Use an absolute pack folder without traversal.", "INVALID_PATH", 422
            )
        _reject_symlink_components(path)
        try:
            path = path.resolve(strict=True)
        except OSError as error:
            raise StorageError(
                "The existing pack folder is unavailable.", "PACK_NOT_FOUND", 404
            ) from error
        if not self.outputs._contains(path) or not path.is_dir():
            raise StorageError(
                "Select a pack folder within configured data roots.", "INVALID_PATH", 403
            )
        return path

    def _prepare(self, spec: FeaturePackSpec) -> tuple[dict, dict, dict | None]:
        configuration = self.store.get_configuration(spec.featureSetId)
        manifest = configuration["manifest"]
        if manifest.get("kind") != "feature":
            raise StorageError("Select a saved feature version.", "INVALID_FEATURE_SET", 422)
        files = manifest.get("files", [])
        findings = FeatureService(self.store, self.outputs).verify_binding(configuration)

        def finding(code, message, severity="error"):
            findings.append({"severity": severity, "code": code, "message": message})

        dimensions = {item["dimensions"] for item in files}
        dtypes = {item["dtype"] for item in files}
        if not files or len(dimensions) != 1 or len(dtypes) != 1:
            finding(
                "INVALID_FEATURE_SET", "The saved inventory must contain consistent feature arrays."
            )
        source_dtype = next(iter(dtypes)) if len(dtypes) == 1 else None
        output_dtype = (
            "float16" if spec.action == "pack" and spec.dtype == "float16" else source_dtype
        )
        if (
            spec.action == "pack"
            and spec.dtype == "preserve"
            and source_dtype not in {"float16", "float32"}
        ):
            finding(
                "UNSUPPORTED_PACK_DTYPE",
                "Preserving precision supports float16 or float32 features. Native validation supports other floating types.",
            )
        if spec.action == "pack" and spec.dtype == "float16" and source_dtype != "float16":
            finding(
                "LOSSY_FLOAT16",
                "Float16 changes numerical precision. Source files are retained and overflow will stop the job.",
                "warning",
            )
        if manifest.get("summary", {}).get("missingSlides", 0):
            finding(
                "PARTIAL_FEATURE_COVERAGE",
                "Only slides in this saved feature version will be included; missing dataset slides remain missing.",
                "warning",
            )
        tmux_available, format_available = self.executor.available(), self.format_available()
        if not tmux_available:
            finding("TMUX_UNAVAILABLE", "Install tmux to run persistent feature jobs.")
        if not format_available:
            finding(
                "PACK_FORMAT_UNAVAILABLE",
                "Install HistoPilot's NumPy and PyArrow dependencies to run feature jobs.",
            )
        output = (
            self._path(spec.outputPath or self._auto_output(spec), configuration)
            if spec.action == "pack"
            else None
        )
        existing_path, inspection, pack_stamps = None, None, None
        matches_features = True
        if spec.action == "attach":
            matches_features = False
            if not spec.existingPath:
                finding("PACK_FOLDER_REQUIRED", "Choose the existing pack folder to verify.")
            elif format_available:
                from histopilot.storage.pack_import import inspect_existing_pack
                from histopilot.storage.packed import PackedStoreError

                existing_path = self._existing_path(spec.existingPath)
                try:
                    inspected = inspect_existing_pack(configuration, existing_path)
                    inspection, pack_stamps = inspected["summary"], inspected["packStamps"]
                    matches_features = inspected["matchesFeatures"]
                    findings.extend(inspected["findings"])
                    if matches_features:
                        finding(
                            "PACK_CONTENT_COMPARISON_PENDING",
                            "Structure matches. Verification will compare every packed feature value and coordinate with this saved feature version.",
                            "warning",
                        )
                except PackedStoreError as error:
                    finding("INVALID_EXISTING_PACK", str(error))
            if spec.outputPath:
                finding(
                    "ATTACH_OUTPUT_UNUSED",
                    "An existing pack is verified in place; no output folder is created.",
                    "warning",
                )
        elif spec.existingPath:
            finding(
                "EXISTING_PACK_UNUSED",
                "The existing pack folder is only used when attaching a pack.",
                "warning",
            )
        if spec.action == "validate" and spec.outputPath:
            finding(
                "VALIDATION_OUTPUT_UNUSED",
                "Validation saves its report in the project; an output folder is only used when packing.",
                "warning",
            )
        patch_count = sum(item["patchCount"] for item in files)
        dimension = next(iter(dimensions)) if len(dimensions) == 1 else None
        estimated = (
            (
                patch_count * ((dimension or 0) * (2 if output_dtype == "float16" else 4) + 8)
                + len(files) * 2048
                + 1024 * 1024
            )
            if output
            else 0
        )
        available = None
        if output:
            if output.exists() and (not output.is_dir() or any(output.iterdir())):
                finding(
                    "OUTPUT_OCCUPIED",
                    "Choose a new or empty output folder. Completed packs are immutable.",
                )
            ancestor = output.parent
            while not ancestor.exists():
                ancestor = ancestor.parent
            if not ancestor.is_dir():
                finding("INVALID_OUTPUT", "An output parent is not a directory.")
            else:
                available = shutil.disk_usage(ancestor).free
                if available < estimated:
                    finding(
                        "INSUFFICIENT_SPACE",
                        "The output filesystem has less free space than the estimated pack size.",
                    )
            with registry_lock() as registry:
                if self._claims(output, registry):
                    finding("OUTPUT_BUSY", "A feature job is already using this output folder.")
        for job in self._jobs():
            if (
                job["featureSetId"] == spec.featureSetId
                and job["spec"]["action"] == spec.action
                and self.get(job["id"], include_inactive=True)["state"] in ACTIVE
            ):
                finding(
                    "FEATURE_JOB_BUSY",
                    "This feature version already has an active job for this action.",
                )
                break
        normalized = spec.model_dump(mode="json")
        normalized["outputPath"] = str(output) if output else None
        normalized["existingPath"] = str(existing_path) if existing_path else None
        result = {
            "spec": normalized,
            "findings": findings,
            "canRun": matches_features
            and not any(item["severity"] == "error" for item in findings),
            "slideCount": len(files),
            "patchCount": patch_count,
            "dimensions": dimension,
            "sourceDtype": source_dtype,
            "outputDtype": output_dtype,
            "estimatedBytes": estimated,
            "availableBytes": available,
            "outputPath": str(output) if output else None,
            "tmuxAvailable": tmux_available,
            "formatAvailable": format_available,
            "existingPath": str(existing_path) if existing_path else None,
            "packInspection": inspection,
            "matchesFeatures": matches_features,
            "verification": "headers",
        }
        # Free disk space may vary between two HTTP requests without invalidating user intent.
        result["previewHash"] = _hash(
            {
                "preview": {key: value for key, value in result.items() if key != "availableBytes"},
                "manifest": manifest,
                "packStamps": pack_stamps,
            }
        )
        return result, configuration, pack_stamps

    def preview(self, spec: FeaturePackSpec) -> dict:
        return self._prepare(spec)[0]

    def _existing(self, spec: FeaturePackSpec, preview_hash: str, operation_id: str) -> dict | None:
        for job in self._jobs():
            if job["operationId"] == operation_id:
                requested = spec.model_dump(mode="json")
                request_hashes = {_hash(requested)}
                if spec.existingPath is None and "existingPath" not in job["spec"]:
                    request_hashes.add(
                        _hash(
                            {
                                key: value
                                for key, value in requested.items()
                                if key != "existingPath"
                            }
                        )
                    )
                if job["requestHash"] not in request_hashes or job["previewHash"] != preview_hash:
                    raise StorageError(
                        "Operation ID was used for a different feature job.", "OPERATION_CONFLICT"
                    )
                return self.get(job["id"])
        return None

    def submit(self, spec: FeaturePackSpec, preview_hash: str, operation_id: str) -> dict:
        with lifecycle_guard(self.store.folder, timeout=5):
            lifecycle = LifecycleStore(self.store.folder, self.store.project_id)
            lifecycle.assert_document_usable(spec.model_dump(mode="json"))
            lifecycle.assert_document_usable(self.store.get_configuration(spec.featureSetId))
            return self._submit(spec, preview_hash, operation_id)

    def _submit(self, spec: FeaturePackSpec, preview_hash: str, operation_id: str) -> dict:
        self.store.initialize()
        existing = self._existing(spec, preview_hash, operation_id)
        if existing:
            return existing
        preview, configuration, pack_stamps = self._prepare(spec)
        if preview["previewHash"] != preview_hash:
            raise StorageError("Feature inputs or output changed. Preview again.", "PREVIEW_STALE")
        if not preview["canRun"]:
            raise StorageError("Resolve preview findings before starting.", "PACKING_INVALID", 422)
        with writer_lock(self.store.folder):
            existing = self._existing(spec, preview_hash, operation_id)
            if existing:
                return existing
            if any(
                job["featureSetId"] == spec.featureSetId
                and job["spec"]["action"] == spec.action
                and self.get(job["id"], include_inactive=True)["state"] in ACTIVE
                for job in self._jobs()
            ):
                raise StorageError(
                    "This feature version already has an active job for this action.",
                    "FEATURE_JOB_BUSY",
                )
            with registry_lock() as registry:
                output = Path(preview["outputPath"]) if preview["outputPath"] else None
                if output:
                    self._path(str(output), configuration)
                    if self._claims(output, registry):
                        raise StorageError("Another job owns this output.", "OUTPUT_BUSY")
                    with output_lock(output):
                        if output.exists() and (not output.is_dir() or any(output.iterdir())):
                            raise StorageError(
                                "Output contents changed. Preview again.", "PREVIEW_STALE"
                            )
                identity = f"packing-{uuid4().hex}"
                folder = self.folder / identity
                ensure_managed_directory(folder)
                job = {
                    "id": identity,
                    "projectId": self.store.project_id,
                    "state": "starting",
                    "operationId": operation_id,
                    "requestHash": _hash(spec.model_dump(mode="json")),
                    "previewHash": preview_hash,
                    "spec": preview["spec"],
                    "featureSetId": spec.featureSetId,
                    "outputPath": preview["outputPath"],
                    "sessionName": f"histopilot-pack-{identity.removeprefix('packing-')}",
                    "logPath": str(folder / "worker.log"),
                    "createdAt": _now(),
                    "updatedAt": _now(),
                }
                write_json(folder / "job.json", job)
                claim_path = registry / f"{output_key(output)}.claim.json" if output else None
                if claim_path:
                    write_json(
                        claim_path,
                        {
                            "jobId": identity,
                            "projectId": self.store.project_id,
                            "jobPath": str(folder / "job.json"),
                            "sessionName": job["sessionName"],
                            "outputPath": str(output),
                        },
                    )
                write_json(
                    folder / "plan.json",
                    {
                        "jobId": identity,
                        "configuration": configuration,
                        "spec": preview["spec"],
                        "resultPath": str(folder / "result.json"),
                        "progressPath": str(folder / "progress.json"),
                        "processPath": str(folder / "process.json"),
                        "logPath": job["logPath"],
                        "cancelPath": str(folder / "cancelled"),
                        "claimPath": str(claim_path) if claim_path else None,
                        "sourceRoots": [str(root) for root in self.outputs.roots],
                        "existingPackStamps": pack_stamps,
                    },
                )
            try:
                self.executor.launch(
                    job["sessionName"],
                    Path(__file__).parents[1] / "workers" / "pack_features.py",
                    folder / "plan.json",
                )
                # Persist the process launch even for fast workers that have already completed.
                job["state"] = "running"
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                job["state"], job["error"] = "failed", f"Could not start feature worker: {error}"
            job["updatedAt"] = _now()
            write_json(folder / "job.json", job)
        return self.get(identity)

    def _result(self, job: dict) -> dict | None:
        path = self.folder / job["id"] / "result.json"
        if not path.exists():
            return None
        result = _read(path)
        if result.get("jobId") != job["id"]:
            raise StorageError("Packing result belongs to another job.", "PACKING_CORRUPT")
        if result.get("state") not in {"succeeded", "failed", "cancelled"}:
            raise StorageError("Packing result has an invalid terminal state.", "PACKING_CORRUPT")
        if result["state"] == "succeeded":
            validation = result.get("validation")
            if (
                not isinstance(validation, dict)
                or validation.get("featureSetId") != job["featureSetId"]
                or validation.get("valid") is not True
                or validation.get("tensorValidationComplete") is not True
            ):
                raise StorageError(
                    "Packing validation does not match the saved feature job.", "PACKING_CORRUPT"
                )
            artifact = result.get("artifact")
            if job["spec"]["action"] in {"pack", "attach"}:
                if (
                    not isinstance(artifact, dict)
                    or not isinstance(artifact.get("id"), str)
                    or not re.fullmatch(r"pack-[a-f0-9]{64}", artifact["id"])
                    or not isinstance(artifact.get("materializationId"), str)
                    or not re.fullmatch(r"pack-[a-f0-9]{64}", artifact["materializationId"])
                    or (
                        job["spec"]["action"] == "pack"
                        and artifact["materializationId"] != artifact["id"]
                    )
                    or artifact.get("jobId") != job["id"]
                    or artifact.get("featureSetId") != job["featureSetId"]
                    or artifact.get("outputPath")
                    != (
                        job["spec"].get("existingPath")
                        if job["spec"]["action"] == "attach"
                        else job["outputPath"]
                    )
                    or artifact.get("sourceContentHash") != validation.get("sourceContentHash")
                    or artifact.get("validation") != validation
                ):
                    raise StorageError(
                        "Feature pack receipt does not match its job or validation.",
                        "PACKING_CORRUPT",
                    )
            elif artifact is not None:
                raise StorageError(
                    "A validation job cannot publish a feature pack.", "PACKING_CORRUPT"
                )
        return result

    def get(self, identity: str, *, logs=False, include_inactive=False) -> dict:
        if not include_inactive:
            LifecycleStore(self.store.folder, self.store.project_id).assert_usable(
                [f"packing:{identity}"]
            )
        job = self._job_record(identity)
        return self._present(job, logs=logs)

    def _present(self, job: dict, *, logs=False, result=_UNREAD) -> dict:
        job = dict(job)
        identity = job["id"]
        folder = self.folder / identity
        if result is _UNREAD:
            result = self._result(job)
        if result is not None:
            job["result"] = {
                **result,
                "validation": _compact(result.get("validation")),
                "artifact": _compact(result.get("artifact")),
            }
            job["state"] = result.get("state", "failed")
            job["updatedAt"] = result.get("finishedAt", job["updatedAt"])
            if result.get("error"):
                job["error"] = result["error"]
        elif job["state"] in ACTIVE or (folder / "cancelled").exists():
            try:
                running = bool(live_process(folder)) or self.executor.running(job["sessionName"])
                if (folder / "cancelled").exists():
                    job["state"] = "cancelling" if running else "cancelled"
                elif not running:
                    age = (
                        datetime.now(UTC) - datetime.fromisoformat(job["createdAt"])
                    ).total_seconds()
                    if job["state"] != "starting" or age >= 30:
                        job["state"], job["error"] = (
                            "interrupted",
                            "The worker ended without a completion report. Preview a new job to retry.",
                        )
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                job["error"] = f"Cannot inspect worker status: {error}"
        job["progress"] = (
            _read(folder / "progress.json") if (folder / "progress.json").exists() else None
        )
        if logs:
            from histopilot.application.extractions import _log_tail

            try:
                job["logs"] = _log_tail(folder / "worker.log")[0]
            except (OSError, StorageError):
                job["logs"] = "Worker log cannot be read safely."
        return {
            key: value for key, value in job.items() if key not in {"operationId", "requestHash"}
        }

    def list(self, *, include_inactive=False) -> dict:
        records = sorted(self._jobs(), key=lambda job: job["createdAt"], reverse=True)
        states = LifecycleStore(self.store.folder, self.store.project_id).read()["records"]
        jobs, artifacts = [], {}
        usable_receipts = {}
        for record in records:
            result = self._result(record)
            job = self._present(record, result=result)
            visible = states.get(f"packing:{record['id']}", {}).get("state", "active") == "active"
            if include_inactive or visible or job["state"] in ACTIVE:
                jobs.append(job)
            if result and result["state"] == "succeeded" and result.get("artifact"):
                retained = states.get(f"packing:{record['id']}", {}).get("state") != "trashed"
                for identity in (record["id"], result["artifact"]["id"]):
                    usable_receipts[identity] = usable_receipts.get(identity, False) or retained
            if (
                (include_inactive or visible)
                and result
                and result["state"] == "succeeded"
                and result.get("artifact")
            ):
                artifact = result["artifact"]
                representation = (
                    artifact["featureSetId"],
                    artifact["materializationId"],
                    artifact["outputPath"],
                )
                # A new verification replaces the visible receipt for this folder.
                # Historical jobs and IDs remain addressable by frozen protocols.
                artifacts.setdefault(representation, artifact)
        source_findings = {}
        fresh_artifacts = []
        for artifact in artifacts.values():
            feature_id = artifact["featureSetId"]
            if feature_id not in source_findings:
                source_findings[feature_id] = self._source_findings(
                    self.store.get_configuration(feature_id, include_inactive=include_inactive)
                )
            findings = self._artifact_findings(artifact, source_findings[feature_id])
            fresh_artifacts.append(
                {**_compact(artifact), "current": not findings, "findings": findings}
            )
        selections = {}
        selection_folder = self.folder / "selections"
        if selection_folder.exists():
            _reject_symlink_components(selection_folder)
            for path in selection_folder.glob("*.json"):
                selection = _read(path)
                artifact_id = selection.get("artifactId")
                if not include_inactive and usable_receipts.get(artifact_id) is False:
                    artifact_id = None
                selections[selection["featureSetId"]] = artifact_id
        return {
            "jobs": jobs,
            "artifacts": fresh_artifacts,
            "selections": selections,
            "tmuxAvailable": self.executor.available(),
            "formatAvailable": self.format_available(),
            "defaultOutputRoot": str(self.store.folder / "feature-packs"),
        }

    def cancel(self, identity: str) -> dict:
        with lifecycle_guard(self.store.folder, timeout=5):
            return self._cancel(identity)

    def _cancel(self, identity: str) -> dict:
        with writer_lock(self.store.folder):
            job = self.get(identity, include_inactive=True)
            if (
                job["state"] in ACTIVE
                or live_process(self.folder / identity)
                or self.executor.running(job["sessionName"])
            ):
                write_json(self.folder / identity / "cancelled", {"requestedAt": _now()})
        # Cancellation is cooperative at each bounded read/copy chunk, preserving atomic publication.
        return self.get(identity, include_inactive=True)

    def validation_for(self, feature_id: str) -> dict | None:
        configuration = self.store.get_configuration(feature_id)
        states = LifecycleStore(self.store.folder, self.store.project_id).read()["records"]
        successful = [
            job
            for job in sorted(
                (
                    self.get(record["id"])
                    for record in self._jobs()
                    if states.get(f"packing:{record['id']}", {}).get("state") != "trashed"
                ),
                key=lambda item: item["createdAt"],
                reverse=True,
            )
            if job["featureSetId"] == feature_id
            and job["state"] == "succeeded"
            and job.get("result", {}).get("validation", {}).get("tensorValidationComplete")
        ]
        if not successful:
            return None
        job = successful[0]
        findings = self._source_findings(configuration)
        return {
            **job["result"]["validation"],
            "jobId": job["id"],
            "validatedAt": job["updatedAt"],
            "current": not findings,
            "findings": findings,
        }

    def _source_findings(self, configuration: dict) -> list[dict]:
        findings = []
        manifest = configuration["manifest"]
        if manifest.get("sourceExtraction"):
            try:
                current = FeatureService(self.store, self.outputs)._source_extraction(
                    FeatureSpec.model_validate(manifest["spec"]), manifest["layout"]
                )
                if current != manifest["sourceExtraction"]:
                    raise ValueError("Recorded extraction evidence changed")
            except (ValueError, KeyError, OSError, StorageError) as error:
                findings.append(
                    {
                        "severity": "error",
                        "code": "FEATURE_SOURCE_CHANGED",
                        "message": f"Extraction provenance: {error}",
                    }
                )
        inputs = [(item["path"], item) for item in manifest["files"]]
        inputs += [
            (item["coordinatePath"], item["coordinateFile"])
            for item in manifest["files"]
            if "coordinateFile" in item
        ]
        inputs += [(item["path"], item) for item in manifest.get("provenance", [])]
        for value, expected in inputs:
            try:
                path = Path(value)
                _reject_symlink_components(path)
                if not self.outputs._contains(path.resolve(strict=True)):
                    raise ValueError("Source moved outside configured roots")
                info = path.stat()
                if not stat.S_ISREG(info.st_mode) or any(
                    actual != expected.get(key) for key, actual in _stamp(info).items()
                ):
                    raise ValueError("Source metadata changed")
                if "sha256" in expected:
                    content = ScientificStore._read_file(path, 2 * 1024 * 1024)
                    if hashlib.sha256(content).hexdigest() != expected["sha256"]:
                        raise ValueError("Provenance contents changed")
            except (OSError, ValueError, StorageError) as error:
                findings.append(
                    {
                        "severity": "error",
                        "code": "FEATURE_SOURCE_CHANGED",
                        "message": f"{Path(value).name}: {error}",
                    }
                )
        return findings

    def _artifact_findings(self, artifact: dict, source_findings: list[dict]) -> list[dict]:
        findings = list(source_findings)
        stamps = artifact.get("packStamps")
        if not stamps:
            findings.append(
                {
                    "severity": "warning",
                    "code": "PACK_VERIFICATION_REFRESH_REQUIRED",
                    "message": "Verify this existing pack once to record file freshness before selecting it.",
                }
            )
            return findings
        try:
            from histopilot.storage.pack_import import pack_file_stamps

            path = self._existing_path(artifact["outputPath"])
            if pack_file_stamps(path) != stamps:
                raise ValueError("Pack files changed since content verification")
        except (ValueError, OSError, StorageError) as error:
            findings.append(
                {
                    "severity": "warning",
                    "code": "PACK_SOURCE_CHANGED",
                    "message": f"{error}. Verify the existing pack again before selecting it.",
                }
            )
        return findings

    def resolve_artifact(self, feature_id: str, artifact_id: str) -> dict:
        configuration = self.store.get_configuration(feature_id)
        if configuration["manifest"].get("kind") != "feature":
            raise StorageError("Select a saved feature version.", "INVALID_FEATURE_SET", 422)
        artifact = self.artifact(artifact_id)
        if artifact["featureSetId"] != feature_id:
            raise StorageError(
                "This pack was verified for a different feature version.",
                "PACK_FEATURE_MISMATCH",
                422,
            )
        findings = self._artifact_findings(artifact, self._source_findings(configuration))
        return {"artifact": _compact(artifact), "current": not findings, "findings": findings}

    def _selection_path(self, feature_id: str) -> Path:
        return self.folder / "selections" / f"{_hash(feature_id)}.json"

    def selection_for(self, feature_id: str) -> dict:
        configuration = self.store.get_configuration(feature_id)
        if configuration["manifest"].get("kind") != "feature":
            raise StorageError("Select a saved feature version.", "INVALID_FEATURE_SET", 422)
        path = self._selection_path(feature_id)
        selected = _read(path) if path.exists() else {}
        if selected and selected.get("featureSetId") != feature_id:
            raise StorageError(
                "Saved pack selection belongs to another feature version.", "PACKING_CORRUPT"
            )
        artifact_id = selected.get("artifactId")
        result = {
            "featureSetId": feature_id,
            "artifactId": artifact_id,
            "artifact": None,
            "current": True,
            "findings": [],
        }
        if artifact_id:
            try:
                result.update(self.resolve_artifact(feature_id, artifact_id))
            except StorageError as error:
                result.update(
                    current=False,
                    findings=[
                        {
                            "severity": "warning",
                            "code": "PACK_SELECTION_UNAVAILABLE",
                            "message": str(error),
                        }
                    ],
                )
        return result

    def select(self, feature_id: str, artifact_id: str | None) -> dict:
        with lifecycle_guard(self.store.folder, timeout=5):
            return self._select(feature_id, artifact_id)

    def _select(self, feature_id: str, artifact_id: str | None) -> dict:
        configuration = self.store.get_configuration(feature_id)
        if configuration["manifest"].get("kind") != "feature":
            raise StorageError("Select a saved feature version.", "INVALID_FEATURE_SET", 422)
        artifact = self.artifact(artifact_id) if artifact_id else None
        if artifact and artifact["featureSetId"] != feature_id:
            raise StorageError(
                "This pack was verified for a different feature version.",
                "PACK_FEATURE_MISMATCH",
                422,
            )
        with writer_lock(self.store.folder):
            if artifact:
                if self._artifact_findings(artifact, self._source_findings(configuration)):
                    raise StorageError(
                        "The pack or source files changed. Verify this existing pack before selecting it.",
                        "PACK_NOT_CURRENT",
                        409,
                    )
            path = self._selection_path(feature_id)
            ensure_managed_directory(path.parent)
            write_json(
                path, {"featureSetId": feature_id, "artifactId": artifact_id, "selectedAt": _now()}
            )
        return self.selection_for(feature_id)

    def artifact(self, identity: str) -> dict:
        trashed = None
        for job in sorted(self._jobs(), key=lambda item: item["createdAt"], reverse=True):
            result = self._result(job)
            if result is None:
                continue
            artifact = result.get("artifact")
            if (
                result.get("state") == "succeeded"
                and artifact
                and identity in {artifact.get("id"), job["id"]}
            ):
                try:
                    LifecycleStore(self.store.folder, self.store.project_id).assert_usable(
                        [f"packing:{job['id']}"]
                    )
                except StorageError as error:
                    if error.code != "RECORD_TRASHED":
                        raise
                    trashed = error
                    continue
                return artifact
        if trashed is not None:
            raise trashed
        raise StorageError("Feature pack not found.", "PACK_NOT_FOUND", 404)
