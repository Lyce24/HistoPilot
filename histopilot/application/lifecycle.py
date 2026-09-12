"""Dependency-aware, recoverable workspace cleanup over immutable records.

Lifecycle metadata changes visibility, never scientific identities or source/output
bytes. A project-wide gate serializes review/commit with new references and launches.
"""

import hashlib
import json
from pathlib import Path

from histopilot.application.bulk_evaluations import BulkEvaluationService
from histopilot.application.compute_jobs import ComputeJobService
from histopilot.application.extractions import ExtractionService
from histopilot.application.feature_packs import FeaturePackService
from histopilot.application.training import TrainingService
from histopilot.schemas.lifecycle import ApplyCleanup, CancelCleanupJob, CleanupSelection
from histopilot.storage.lifecycle import LifecycleStore, lifecycle_guard
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import ScientificStore

ACTIVE = {"queued", "starting", "running", "cancelling"}
TERMINAL = {"completed", "succeeded", "failed", "cancelled", "interrupted"}
NOTE = (
    "Archive hides records from normal lists while preserving saved references. "
    "Delete moves records to recoverable Trash. Source files, slides, features, packs, "
    "logs and checkpoints are retained; these actions do not reclaim disk space."
)


def _hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _strings(value):
    """Match exact known IDs, including nested validation and pack receipts."""
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, str):
            yield item
        elif isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, (list, tuple)):
            pending.extend(item)


def _read_optional(path: Path):
    if not path.exists() and not path.is_symlink():
        return None
    try:
        result = json.loads(ScientificStore._read_file(path, 64 * 1024 * 1024))
        if not isinstance(result, dict):
            raise ValueError("Expected an object")
        return result
    except (ValueError, OSError) as error:
        raise StorageError(
            "Job evidence cannot be read safely; cleanup is blocked until it is repaired.",
            "CLEANUP_EVIDENCE_INVALID",
        ) from error


def _closure(keys, items, field):
    found, pending = set(keys), list(keys)
    while pending:
        for key in items[pending.pop()][field]:
            if key not in found:
                found.add(key)
                pending.append(key)
    return found


def _confirmed_live(process):
    """A missing process is stopped; unreadable evidence is never proof of that."""
    if process is None:
        return False
    if not isinstance(process, dict) or type(process.get("pid")) is not int or process["pid"] <= 1:
        raise StorageError(
            "Job process identity is invalid; cleanup is blocked.", "CLEANUP_PROCESS_UNKNOWN"
        )
    try:
        fields = Path(f"/proc/{process['pid']}/stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return False
    except (OSError, IndexError) as error:
        raise StorageError(
            "Cannot confirm whether a job process stopped.", "CLEANUP_PROCESS_UNKNOWN"
        ) from error
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        if not isinstance(process.get("bootId"), str) or type(process.get("startTicks")) is not int:
            raise ValueError("Incomplete process identity")
        return (
            fields[0] != "Z"
            and process["bootId"] == boot
            and process["startTicks"] == int(fields[19])
        )
    except (OSError, ValueError, IndexError) as error:
        raise StorageError(
            "Cannot verify the job process identity.", "CLEANUP_PROCESS_UNKNOWN"
        ) from error


class CleanupService:
    def __init__(
        self,
        store,
        filesystem,
        project_name="Project",
        *,
        training=None,
        packs=None,
        extractions=None,
        compute=None,
        evaluation_batches=None,
    ):
        self.store, self.filesystem = store, filesystem
        self.metadata = LifecycleStore(store.folder, store.project_id)
        self.project_name = project_name
        self.training = training or TrainingService(store, filesystem)
        self.packs = packs or FeaturePackService(store, filesystem)
        self.extractions = extractions or ExtractionService(store, filesystem)
        self.compute = compute or ComputeJobService(store, filesystem)
        self.evaluation_batches = evaluation_batches or BulkEvaluationService(store, filesystem)

    def _catalog(self):
        snapshot = self.metadata.read()
        items, documents, aliases = {}, {}, {}

        def add(resource_type, record, kind, name, document=None):
            identity = record["id"]
            key = f"{resource_type}:{identity}"
            if key in items:
                raise StorageError("Duplicate cleanup record.", "CLEANUP_EVIDENCE_INVALID")
            items[key] = {
                "key": key,
                "type": resource_type,
                "id": identity,
                "kind": kind,
                "name": name or identity,
                "state": snapshot["records"].get(key, {}).get("state", "active"),
                "createdAt": record.get("createdAt"),
                "recordVersion": record.get(
                    "contentHash", record.get("revision", record.get("updatedAt"))
                ),
                "dependsOn": [],
                "usedBy": [],
            }
            documents[key] = document if document is not None else record
            aliases.setdefault(identity, set()).add(key)
            return key

        descriptor = _read_optional(self.store.folder / "histopilot-project.json")
        project_key = add(
            "project",
            {"id": self.store.project_id, "contentHash": _hash(descriptor)},
            "project",
            (descriptor or {}).get("name", self.project_name),
            {},
        )
        for record in self.store.list_datasets(include_inactive=True):
            add("dataset", record, "dataset", record["manifest"].get("name", "Dataset"))
        for record in self.store.list_drafts(include_inactive=True):
            key = add(
                "draft", record, record["payload"].get("type", record["kind"]), record["name"]
            )
            if record["payload"].get("type") == "model-experiment":
                items[key]["configurationLocked"] = bool(record["payload"].get("submission"))
                submission = record["payload"].get("submission") or {}
                from histopilot.application.experiment_policy import predictor_work_expected

                if predictor_work_expected(submission):
                    from histopilot.application.experiment_predictors import (
                        ExperimentPredictorService,
                    )

                    coordinator = ExperimentPredictorService(self.store, self.filesystem)
                    execution = coordinator.status(record["id"], summary=True)
                    if execution:
                        _plan, coordinator_state = coordinator._read(record["id"])
                        session = coordinator_state.get("sessionName")
                        alive = _confirmed_live(coordinator_state.get("process")) or bool(
                            session and coordinator.executor.running(session)
                        )
                        status = {
                            "waiting": "queued",
                            "cancelling": "running",
                            "attention": "failed",
                        }.get(execution["status"], execution["status"])
                        self._job(items[key], status, alive, execution["status"] == "cancelling")
                        # Cancellation here would also need to choose which fold
                        # batches to stop. The experiment owns that explicit UI.
                        items[key]["job"]["cancellable"] = False
            else:
                spec = record["payload"].get("spec")
                owner = record["payload"].get("experimentId") or (
                    spec.get("experimentId") if isinstance(spec, dict) else None
                )
                if owner:
                    items[key]["experimentKey"] = f"draft:{owner}"
        for record in self.store.list_configurations(include_inactive=True):
            manifest = record["manifest"]
            spec = manifest.get("spec", {})
            name = (
                record.get("versionLabel", {}).get("tag")
                or spec.get("batchName")
                or spec.get("name")
                or manifest.get("name")
                or manifest["kind"].replace("-", " ").title()
            )
            key = add("configuration", record, manifest["kind"], name)
            if manifest["kind"] == "mil-batch":
                owner = spec.get("experimentId")
                if owner:
                    items[key]["experimentKey"] = f"draft:{owner}"
                execution = self.training.execution(record["id"], include_inactive=True)
                if execution:
                    if owner and f"draft:{owner}" in items:
                        items[f"draft:{owner}"]["configurationLocked"] = True
                    folder = self.store.folder / "training" / record["id"]
                    # A terminal state can precede final process cleanup. Confirm
                    # that neither scheduler nor any child still owns work.
                    alive = _confirmed_live(execution.get("process")) or any(
                        _confirmed_live(run.get("process")) for run in execution.get("runs", [])
                    )
                    self._job(
                        items[key],
                        execution["status"],
                        alive,
                        execution.get("cancelRequested", False),
                    )
                    documents[key] = [record, _read_optional(folder / "plan.json")]
            elif manifest["kind"] in {
                "predictor-refit",
                "model-evaluation",
                "model-interpretation",
            }:
                execution = self.compute.status(record["id"], include_inactive=True)
                if execution["status"] != "not_started":
                    self._job(
                        items[key],
                        execution["status"],
                        _confirmed_live(execution.get("process")),
                        execution.get("cancellationRequested", False),
                    )
                folder = self.store.folder / "compute-jobs" / record["id"]
                documents[key] = [record, _read_optional(folder / "plan.json")]
            elif manifest["kind"] == "evaluation-batch":
                execution = self.evaluation_batches.get(record["id"], include_inactive=True)
                status = execution["status"]
                # A partial batch contains terminal child failures; running or
                # not-yet-submitted members always keep the parent busy.
                self._job(
                    items[key],
                    "failed" if status == "partial" else status,
                    False,
                    execution.get("cancelRequested", False),
                )

        for resource_type, service in (("extraction", self.extractions), ("packing", self.packs)):
            for record in service._jobs():
                presented = service.get(record["id"], include_inactive=True)
                kind = (
                    "extraction"
                    if resource_type == "extraction"
                    else f"feature-{record.get('spec', {}).get('action', 'packing')}"
                )
                key = add(
                    resource_type,
                    record,
                    kind,
                    record.get("name") or f"{kind.replace('-', ' ').title()} · {record['id'][-8:]}",
                )
                folder = service.folder / record["id"]
                result = _read_optional(folder / "result.json")
                documents[key] = [record, result]
                artifact = (result or {}).get("artifact")
                if artifact and isinstance(artifact.get("id"), str):
                    aliases.setdefault(artifact["id"], set()).add(key)
                # Probe live identity even after a terminal result was written.
                alive = _confirmed_live(_read_optional(folder / "process.json"))
                self._job(items[key], presented["state"], alive, (folder / "cancelled").exists())

        if len(items) > 20000:
            raise StorageError("Too many records to review cleanup safely.", "CLEANUP_LIMIT", 413)
        for key, document in documents.items():
            references = set()
            for identity in _strings(document):
                owners = aliases.get(identity, set())
                if identity.startswith("pack-"):
                    if key in owners:
                        # A receipt defines its own artifact ID; that definition
                        # is an output, not a dependency on other receipts of it.
                        continue
                    # Identical artifact IDs can have several verification
                    # receipts. A removed receipt is not required by a bare
                    # artifact reference when a retained receipt still exists.
                    # An explicit job ID below still binds that exact receipt.
                    retained = {owner for owner in owners if items[owner]["state"] != "trashed"}
                    owners = retained or owners
                references.update(owners)
            # Every envelope has projectId. The project is a visibility container,
            # not an input to delete recursively; its child states are retained.
            references.difference_update((key, project_key))
            items[key]["dependsOn"] = sorted(references)
            for parent in references:
                items[parent]["usedBy"].append(key)
        for item in items.values():
            item["usedBy"].sort()
        return {
            "projectId": self.store.project_id,
            "revision": snapshot["revision"],
            "projectState": items[project_key]["state"],
            "items": sorted(
                items.values(),
                key=lambda item: (
                    item["type"] != "project",
                    item["kind"],
                    item["name"],
                    item["key"],
                ),
            ),
            "audit": list(reversed(snapshot.get("audit", [])[-100:])),
            "note": NOTE,
        }

    @staticmethod
    def _job(item, status, alive, cancel_requested):
        unknown = status not in ACTIVE | TERMINAL
        busy = status in ACTIVE or alive
        item["job"] = {
            "status": "cancelling"
            if busy and cancel_requested
            else "running"
            if alive and status not in ACTIVE
            else status,
            "cancellable": busy and not cancel_requested and not unknown,
            "busy": busy or unknown,
        }

    def catalog(self):
        with lifecycle_guard(self.store.folder):
            return self._catalog()

    def _preview(self, selection, catalog):
        items = {item["key"]: item for item in catalog["items"]}
        selected = set(selection.keys)
        project_key = f"project:{self.store.project_id}"
        blockers, required = [], set()

        def block(code, message, keys):
            blockers.append({"code": code, "message": message, "keys": sorted(keys)})

        missing = selected - items.keys()
        if missing:
            block(
                "RECORD_NOT_FOUND",
                "Some selected records no longer exist in this project.",
                missing,
            )
        else:
            locked_members = [
                key
                for key in selected
                if items[key].get("experimentKey") not in selected
                and items.get(items[key].get("experimentKey"), {}).get("configurationLocked")
            ]
            if locked_members and project_key not in selected:
                block(
                    "EXPERIMENT_CONFIGURATION_LOCKED",
                    "Submitted batches stay with their experiment. Manage the whole experiment together, or copy it to adjust its batches.",
                    locked_members,
                )
            if project_key in selected and len(selected) > 1:
                block(
                    "PROJECT_SEPARATE",
                    "Manage the whole project separately; its records keep their existing states.",
                    [project_key],
                )
            if catalog["projectState"] == "trashed" and selected != {project_key}:
                block(
                    "PROJECT_TRASHED",
                    "Restore the project before changing individual records.",
                    [project_key],
                )
            if selection.action in {"archive", "trash"}:
                affected = (
                    set(items) if project_key in selected else _closure(selected, items, "usedBy")
                )
                running = [key for key in affected if items[key].get("job", {}).get("busy")]
                if running:
                    block(
                        "JOBS_ACTIVE",
                        "Cancel these jobs and wait until their processes stop before cleanup.",
                        running,
                    )
            if selection.action == "archive":
                trashed = [key for key in selected if items[key]["state"] == "trashed"]
                if trashed:
                    block(
                        "RECORD_TRASHED",
                        "Restore records from Trash before archiving them.",
                        trashed,
                    )
            elif selection.action == "trash" and project_key not in selected:
                # Trashed consumers retain immutable bytes but no longer require
                # visible inputs. Archived consumers still protect their inputs.
                dependents = _closure(selected, items, "usedBy") - selected
                required.update(key for key in dependents if items[key]["state"] != "trashed")
                if required:
                    block(
                        "RECORD_IN_USE",
                        "Retained records use this selection. Include them explicitly or keep these inputs.",
                        required,
                    )
            elif selection.action == "restore" and project_key not in selected:
                parents = _closure(selected, items, "dependsOn") - selected
                required.update(key for key in parents if items[key]["state"] == "trashed")
                if required:
                    block(
                        "INPUTS_TRASHED",
                        "Restore the required inputs together with these records.",
                        required,
                    )
        # Include graph and live status, not volatile epoch progress/telemetry.
        # New consumers, edits, relabeling, lifecycle changes or job transitions
        # invalidate the reviewed operation before any metadata is written.
        return {
            "action": selection.action,
            "keys": selection.keys,
            "revision": catalog["revision"],
            "previewHash": _hash(
                {
                    "selection": selection.model_dump(),
                    "catalog": {key: value for key, value in catalog.items() if key != "audit"},
                }
            ),
            "canApply": not blockers,
            "blockers": blockers,
            "requiredKeys": sorted(required),
            "recordCount": len(selected),
            "note": NOTE,
        }

    def preview(self, selection: CleanupSelection):
        with lifecycle_guard(self.store.folder):
            return self._preview(selection, self._catalog())

    def apply(self, request: ApplyCleanup):
        with lifecycle_guard(self.store.folder):
            request_hash = _hash(request.model_dump(exclude={"operationId"}))
            prior = self.metadata.read().get("operations", {}).get(request.operationId)
            if prior:
                if prior["requestHash"] != request_hash:
                    raise StorageError(
                        "This operation ID belongs to a different cleanup request.",
                        "OPERATION_CONFLICT",
                    )
                return {
                    "action": request.action,
                    "revision": prior["revision"],
                    "changed": sorted(prior["changes"]),
                }
            selection = CleanupSelection(action=request.action, keys=request.keys)
            preview = self._preview(selection, self._catalog())
            if preview["previewHash"] != request.previewHash:
                raise StorageError(
                    "Workspace records or job status changed. Review cleanup again.",
                    "CLEANUP_PREVIEW_STALE",
                )
            if not preview["canApply"]:
                raise StorageError(
                    "Cleanup is blocked. Resolve the listed dependencies or active jobs first.",
                    "CLEANUP_BLOCKED",
                )
            state = {"archive": "archived", "trash": "trashed", "restore": "active"}[request.action]
            changes = dict.fromkeys(request.keys, state)
            result = self.metadata.apply(
                changes,
                operation_id=request.operationId,
                request_hash=request_hash,
                expected_revision=preview["revision"],
                action=request.action,
            )
            return {
                "action": request.action,
                "revision": result["revision"],
                "changed": request.keys,
            }

    def cancel(self, request: CancelCleanupJob):
        with lifecycle_guard(self.store.folder):
            catalog = self._catalog()
            item = next((item for item in catalog["items"] if item["key"] == request.key), None)
            if item is None or not item.get("job"):
                raise StorageError(
                    "Select an existing compute job or launched training batch.",
                    "JOB_NOT_FOUND",
                    404,
                )
            request_hash = _hash({"action": "cancel", "key": request.key})
            prior = self.metadata.read().get("operations", {}).get(request.operationId)
            if prior and prior["requestHash"] != request_hash:
                raise StorageError(
                    "This operation ID belongs to a different request.", "OPERATION_CONFLICT"
                )
            if prior:
                return {"key": request.key, "job": item["job"]}
            if item["type"] == "configuration":
                service = (
                    self.evaluation_batches
                    if item["kind"] == "evaluation-batch"
                    else self.compute
                    if item["kind"]
                    in {"predictor-refit", "model-evaluation", "model-interpretation"}
                    else self.training
                )
                result = service.cancel(item["id"], request.operationId)
            elif item["type"] == "extraction":
                result = self.extractions.cancel(item["id"])
            else:
                result = self.packs.cancel(item["id"])
            self.metadata.apply(
                {},
                operation_id=request.operationId,
                request_hash=request_hash,
                expected_revision=catalog["revision"],
                action="cancel",
                targets=[request.key],
            )
            return {"key": request.key, "job": result}
