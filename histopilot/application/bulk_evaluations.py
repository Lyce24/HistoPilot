"""Reviewed cohort evaluation batches with durable, independently tracked jobs."""

import hashlib

from histopilot.application.evaluation_runs import EvaluationRunService
from histopilot.application.feature_bundles import _hash
from histopilot.application.predictors import finding, lifecycle_document, reference
from histopilot.schemas.bulk_evaluations import BulkEvaluationSelection
from histopilot.schemas.predictors import EvaluationRunSelection
from histopilot.storage.lifecycle import lifecycle_guard
from histopilot.storage.project_lock import (
    StorageError,
    _reject_symlink_components,
    ensure_managed_directory,
    writer_lock,
)
from histopilot.storage.scientific import MAX_CONFIGURATION_BYTES, _json
from histopilot.workers.packing_process import write_json
from histopilot.workers.training_process import now, read_json


class BulkEvaluationService:
    def __init__(self, store, filesystem, evaluations=None):
        self.store, self.filesystem = store, filesystem
        self.evaluations = evaluations or EvaluationRunService(store, filesystem)

    def _record(self, identity, *, include_inactive=False):
        value = self.store.get_configuration(identity, include_inactive=include_inactive)
        if value["manifest"].get("kind") != "evaluation-batch":
            raise StorageError("Evaluation batch not found.", "EVALUATION_BATCH_NOT_FOUND", 404)
        return lifecycle_document(self.store, value)

    def _folder(self, identity):
        # identity originates from a validated scientific record, never a path.
        folder = self.store.folder / "evaluation-batches" / identity
        _reject_symlink_components(folder)
        return folder

    def _prepare(self, selection, *, reviewed=None):
        cohort = self.evaluations.cohorts.get(selection.cohortId)
        if reviewed is not None:
            identities = sorted(reviewed)
            if selection.scope == "selected" and identities != sorted(selection.predictorIds):
                raise StorageError("The reviewed predictor scope changed.", "PREVIEW_STALE")
        elif selection.scope == "all":
            identities = sorted(
                row["id"] for row in self.store.list_configurations("frozen-predictor")
            )
        else:
            identities = sorted(selection.predictorIds)
        if len(identities) > 256:
            raise StorageError(
                "Select at most 256 predictors in one evaluation batch.",
                "EVALUATION_BATCH_LIMIT",
                422,
            )
        items = []
        deleted_selection = False
        for identity in identities:
            row = {
                "predictorId": identity,
                "predictorName": identity,
                "method": None,
                "eligible": False,
                "findings": [],
                "selection": None,
                "evaluationPreviewHash": None,
                "evaluationManifest": None,
            }
            try:
                predictor = self.store.get_configuration(identity, include_inactive=True)
                model = predictor["manifest"]
                if model.get("kind") != "frozen-predictor":
                    raise StorageError("Select a frozen predictor.", "INVALID_PREDICTOR", 422)
                row.update(
                    predictor=reference(predictor),
                    predictorName=model["name"],
                    method=model.get("method", "ensemble"),
                    **{
                        key: model.get(key)
                        for key in (
                            "experimentId",
                            "batchId",
                            "candidateId",
                            "trainingSeed",
                            "splitSeed",
                        )
                    },
                )
                state = lifecycle_document(self.store, predictor)["lifecycleState"]
                deleted_selection |= state == "trashed"
                if state != "active":
                    raise StorageError(
                        "Restore this predictor to Active before adding it to a new evaluation batch.",
                        "PREDICTOR_INACTIVE",
                    )
                choice = EvaluationRunSelection(
                    cohortId=cohort["id"],
                    predictorId=identity,
                    name=f"{selection.namePrefix} · {model['name']}"[:120],
                    featureBundleId=selection.featureBundleId,
                    inference=selection.inference,
                    patientIdentifiers=selection.patientIdentifiers,
                )
                manifest = self.evaluations._prepare(choice)
                row.update(
                    eligible=True,
                    selection=choice.model_dump(exclude_none=True),
                    evaluationPreviewHash=_hash(manifest),
                    evaluationManifest=manifest,
                )
            except StorageError as error:
                row["findings"] = [finding(error.code, str(error))]
            items.append(row)
        count = sum(row["eligible"] for row in items)
        payload = {
            "cohortId": cohort["id"],
            "cohort": reference(cohort),
            "scope": selection.scope,
            "namePrefix": selection.namePrefix,
            **({"featureBundleId": selection.featureBundleId} if selection.featureBundleId else {}),
            **({"inference": selection.inference.model_dump()} if selection.inference else {}),
            **(
                {"patientIdentifiers": selection.patientIdentifiers}
                if selection.patientIdentifiers
                else {}
            ),
            "reviewedPredictorIds": identities,
            "items": items,
            "eligibleCount": count,
            "blockedCount": len(items) - count,
            "canRun": count > 0 and not deleted_selection,
        }
        return {**payload, "previewHash": _hash(payload)}

    def preview(self, selection):
        with lifecycle_guard(self.store.folder):
            return self._prepare(selection)

    def run(self, request):
        selection = BulkEvaluationSelection.model_validate(
            request.model_dump(include=set(BulkEvaluationSelection.model_fields))
        )
        submitted_request = request.model_dump(
            exclude={
                key
                for key in ("featureBundleId", "inference", "patientIdentifiers")
                if getattr(request, key) is None
            }
        )
        with lifecycle_guard(self.store.folder):
            prior = self.store.configuration_publication(request.operationId)
            if prior:
                if (
                    prior["manifest"].get("kind") != "evaluation-batch"
                    or prior["manifest"].get("request") != submitted_request
                ):
                    raise StorageError(
                        "This operation belongs to another evaluation batch.", "OPERATION_CONFLICT"
                    )
                batch = prior
            else:
                preview = self._prepare(selection, reviewed=request.reviewedPredictorIds)
                if preview["previewHash"] != request.previewHash:
                    raise StorageError(
                        "Reviewed predictor or cohort inputs changed. Review this batch again.",
                        "PREVIEW_STALE",
                    )
                if not preview["canRun"]:
                    raise StorageError(
                        "Choose compatible active predictors and remove deleted selections before running this batch.",
                        "EVALUATION_BATCH_BLOCKED",
                        422,
                    )
                token = _hash(request.operationId)
                members = []
                for index, item in enumerate(preview["items"]):
                    member = {
                        key: value for key, value in item.items() if key != "evaluationManifest"
                    }
                    if item["eligible"]:
                        manifest = {
                            **item["evaluationManifest"],
                            "previewHash": item["evaluationPreviewHash"],
                            "batchRequestId": token,
                        }
                        digest = hashlib.sha256(
                            _json(manifest, MAX_CONFIGURATION_BYTES)
                        ).hexdigest()
                        member.update(
                            evaluationId=f"configuration-{digest}",
                            evaluationContentHash=digest,
                            evaluationManifest=manifest,
                            saveOperation=f"bulk-eval-save-{token}-{index}",
                            launchOperation=f"bulk-eval-launch-{token}-{index}",
                        )
                    members.append(member)
                cohort = self.evaluations.cohorts.get(selection.cohortId)
                batch = self.store.publish_configuration(
                    manifest={
                        "kind": "evaluation-batch",
                        "schemaVersion": 1,
                        "datasetId": cohort["manifest"]["datasetId"],
                        **(
                            {"datasetIds": cohort["manifest"]["spec"]["datasetIds"]}
                            if cohort["manifest"]["spec"].get("datasetIds")
                            else {}
                        ),
                        "name": selection.namePrefix,
                        "cohortId": selection.cohortId,
                        "cohort": reference(cohort),
                        "request": submitted_request,
                        "items": members,
                    },
                    operation_id=request.operationId,
                )
        self._submit(batch)
        return self.get(batch["id"])

    def _submit(self, batch):
        folder = self._folder(batch["id"])
        ensure_managed_directory(folder)
        for member in batch["manifest"]["items"]:
            if member["eligible"] and not self._submit_member(batch, member):
                return
        with lifecycle_guard(self.store.folder), writer_lock(folder):
            path = folder / "state.json"
            state = (
                read_json(path)
                if path.exists()
                else {"items": {}, "cancelRequested": False, "submitted": False}
            )
            if not state["cancelRequested"]:
                state.update(submitted=True, updatedAt=now())
                write_json(path, state)

    def _submit_member(self, batch, member):
        # Release both locks between members so a separate cancellation request
        # can stop remaining submissions without waiting for the whole batch.
        folder = self._folder(batch["id"])
        with lifecycle_guard(self.store.folder), writer_lock(folder):
            path = folder / "state.json"
            state = (
                read_json(path)
                if path.exists()
                else {"items": {}, "cancelRequested": False, "submitted": False}
            )
            if state["cancelRequested"]:
                return False
            self.store.lifecycle.assert_document_usable(batch)
            identity = member["predictorId"]
            try:
                document = self.store.publish_configuration(
                    manifest=member["evaluationManifest"], operation_id=member["saveOperation"]
                )
                if document["id"] != member["evaluationId"]:
                    raise StorageError(
                        "A batch member identity changed.", "EVALUATION_BATCH_CHANGED"
                    )
                current = self.evaluations.jobs.status(document["id"])
                if current["status"] == "not_started":
                    self.evaluations.launch(document["id"], member["launchOperation"])
                # A batch replay submits missing jobs, but never implicitly
                # restarts completed, failed, cancelled or interrupted compute.
                state["items"][identity] = {"submitted": True, "error": None}
            except Exception as error:
                state["items"][identity] = {
                    "submitted": False,
                    "error": str(error),
                    "code": getattr(error, "code", "EVALUATION_SUBMISSION_FAILED"),
                }
            state["updatedAt"] = now()
            write_json(path, state)
            return True

    def get(self, identity, *, include_inactive=False):
        record = self._record(identity, include_inactive=include_inactive)
        folder = self._folder(identity)
        state = (
            read_json(folder / "state.json")
            if (folder / "state.json").exists()
            else {"items": {}, "cancelRequested": False, "submitted": False}
        )
        items = []
        for member in record["manifest"]["items"]:
            row = {
                key: value
                for key, value in member.items()
                if key
                not in {
                    "evaluationManifest",
                    "saveOperation",
                    "launchOperation",
                    "evaluationPreviewHash",
                    "selection",
                }
            }
            submission = state["items"].get(member["predictorId"], {})
            execution = None
            if not member["eligible"]:
                status = "skipped"
            else:
                try:
                    execution = self.evaluations.jobs.status(
                        member["evaluationId"], include_inactive=True
                    )
                    status = execution["status"]
                except StorageError as error:
                    if error.code == "CONFIGURATION_NOT_FOUND":
                        status = "not_started"
                    else:
                        status = "failed"
                        submission = {"error": str(error)}
                if status == "not_started":
                    status = (
                        "cancelled"
                        if state["cancelRequested"]
                        else "failed"
                        if submission.get("error")
                        else "planned"
                    )
            items.append(
                {
                    **row,
                    "status": status,
                    "execution": execution,
                    "error": submission.get("error") or (execution or {}).get("error"),
                }
            )
        statuses = [row["status"] for row in items if row["eligible"]]
        if "running" in statuses:
            status = "running"
        elif any(value in {"queued", "planned"} for value in statuses):
            status = "queued"
        elif statuses and all(value == "completed" for value in statuses):
            status = "completed"
        elif statuses and all(value == "cancelled" for value in statuses):
            status = "cancelled"
        elif "completed" in statuses:
            status = "partial"
        else:
            status = "failed"
        return {
            "id": identity,
            "name": record["manifest"]["name"],
            "cohortId": record["manifest"]["cohortId"],
            "createdAt": record["createdAt"],
            "lifecycleState": record["lifecycleState"],
            "status": status,
            "cancelRequested": state["cancelRequested"],
            "submitted": state["submitted"],
            "counts": {
                key: sum(row["status"] == key for row in items)
                for key in (
                    "planned",
                    "queued",
                    "running",
                    "completed",
                    "failed",
                    "cancelled",
                    "interrupted",
                    "skipped",
                )
            },
            "items": items,
        }

    def list(self, *, include_inactive=False):
        return {
            "items": [
                self.get(row["id"], include_inactive=include_inactive)
                for row in self.store.list_configurations(
                    "evaluation-batch", include_inactive=include_inactive
                )
            ]
        }

    def cancel(self, identity, operation_id):
        with lifecycle_guard(self.store.folder):
            record = self._record(identity, include_inactive=True)
            folder = self._folder(identity)
            ensure_managed_directory(folder)
            with writer_lock(folder):
                path = folder / "state.json"
                state = read_json(path) if path.exists() else {"items": {}, "submitted": False}
                state.update(cancelRequested=True, updatedAt=now())
                write_json(path, state)
                for member in record["manifest"]["items"]:
                    if not member["eligible"]:
                        continue
                    try:
                        self.evaluations.jobs.cancel(
                            member["evaluationId"], f"bulk-cancel-{_hash(operation_id)}"
                        )
                    except StorageError as error:
                        if error.code != "CONFIGURATION_NOT_FOUND":
                            state["items"][member["predictorId"]] = {"error": str(error)}
                            write_json(path, state)
            return self.get(identity, include_inactive=True)
