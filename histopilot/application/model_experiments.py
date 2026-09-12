"""Editable experiment plans become irrevocable submissions before work starts.

Submission pins inputs, batch recipes and a common execution environment. Worker
evidence determines progress; idempotent receipts recover partial dispatch. Older
batches and saved plans remain readable without rewriting their evidence.
"""

from copy import deepcopy
from datetime import UTC, datetime
from uuid import uuid4

from pydantic import ValidationError

from histopilot.application.feature_bundles import _hash
from histopilot.schemas.development import DevelopmentBatchSpec
from histopilot.schemas.mil import MILInputSpec
from histopilot.schemas.model_experiments import ExperimentPredictorPolicy
from histopilot.storage.lifecycle import lifecycle_guard
from histopilot.storage.project_lock import StorageError

EXPERIMENT_TYPE = "model-experiment"
LEGACY_DRAFT_TYPES = ("mil-experiment", "development-batch")


def _mapping(value):
    return value if isinstance(value, dict) else {}


def legacy_experiment_id(identity):
    return f"legacy-{identity}"


def experiment_id_for_batch(record):
    return record["manifest"].get("spec", {}).get("experimentId") or legacy_experiment_id(
        record["id"]
    )


def input_snapshot(store, inputs, *, include_inactive=False):
    """Small exact specifications and immutable hashes, never feature tensors."""
    values = inputs.model_dump() if isinstance(inputs, MILInputSpec) else inputs
    protocol = store.get_configuration(values["protocolId"], include_inactive=include_inactive)
    bundle = store.get_configuration(values["featureBundleId"], include_inactive=include_inactive)
    if protocol["manifest"].get("kind") != "protocol":
        raise StorageError("Choose a target and split protocol.", "INVALID_PROTOCOL", 422)
    if bundle["manifest"].get("kind") != "feature-bundle":
        raise StorageError("Choose a frozen feature bundle.", "INVALID_FEATURE_BUNDLE", 422)
    if protocol["manifest"]["datasetId"] != bundle["manifest"]["datasetId"]:
        raise StorageError(
            "The protocol and features belong to different dataset versions.",
            "BUNDLE_DATASET_MISMATCH",
            422,
        )
    dataset = store.get_dataset(
        protocol["manifest"]["datasetId"], include_inactive=include_inactive
    )
    features = store.get_configuration(
        bundle["manifest"]["spec"]["featureSetId"], include_inactive=include_inactive
    )

    def frozen(record):
        manifest = record["manifest"]
        return {
            "id": record["id"],
            "contentHash": record["contentHash"],
            "spec": manifest.get("spec", {}),
            "summary": manifest.get("summary", {}),
        }

    return {
        "dataset": {
            "id": dataset["id"],
            "contentHash": dataset["contentHash"],
            "name": dataset["manifest"].get("name", "Dataset"),
            "source": dataset["manifest"].get("source", {}),
        },
        "protocol": frozen(protocol),
        "featureBundle": frozen(bundle),
        "features": frozen(features),
        "inputs": values,
    }


def aggregate_status(batches, has_plan=False):
    statuses = {batch["status"] for batch in batches if batch["state"] != "trashed"}
    if not statuses:
        return "planned" if has_plan else "created"
    for status in ("cancelling", "running", "queued", "interrupted", "failed", "unknown"):
        if status in statuses:
            return status
    if len(statuses) == 1:
        return next(iter(statuses))
    return "partial"


def experiment_stage(batches, submission=None, *, legacy=False):
    # Hidden evidence must never reopen a submitted experiment. Unknown evidence
    # also locks fail-closed; it is not proof that training never started.
    locked = bool(submission) or legacy or any(row["status"] != "planned" for row in batches)
    if not locked:
        return "planning", False
    expected = set((submission or {}).get("batchIds", []))
    publications = (submission or {}).get("publications", [])
    complete_intent = all(row.get("batchId") for row in publications)
    actual = {row["id"] for row in batches}
    stage_batches = [row for row in batches if row["id"] in expected] if submission else batches
    finished = (
        bool(stage_batches)
        and complete_intent
        and expected <= actual
        and all(row["status"] in {"completed", "cancelled"} for row in stage_batches)
    )
    return ("finished" if finished else "running"), True


def execution_contract(plan):
    """Comparable worker code and interpreter identity, excluding live telemetry."""
    runtime = plan.get("runtime", {})
    return {
        "code": deepcopy(plan.get("code")),
        "runtime": {
            key: deepcopy(runtime.get(key))
            for key in ("python", "pythonVersion", "cudaVersion", "versions")
        },
    }


def require_execution_contract(expected, actual):
    if expected != actual:
        raise StorageError(
            "Training code or the Python environment changed after experiment review. Restore the submitted environment to continue, or copy this experiment for a new comparison.",
            "EXPERIMENT_RUNTIME_CHANGED",
            409,
        )


class ModelExperimentService:
    def __init__(self, store, filesystem, *, training=None, predictor_execution=None):
        self.store, self.filesystem = store, filesystem
        # Training imports batch planning. Keep this dependency lazy.
        if training is None:
            from histopilot.application.training import TrainingService

            training = TrainingService(store, filesystem)
        self.training = training
        self.predictor_execution = predictor_execution

    def _predictors(self):
        if self.predictor_execution is None:
            from histopilot.application.experiment_predictors import ExperimentPredictorService

            self.predictor_execution = ExperimentPredictorService(
                self.store, self.filesystem, training=self.training
            )
        return self.predictor_execution

    def _start_predictors(self, identity, submission):
        if (
            not submission.get("predictorPolicy")
            or submission["predictorPolicy"]["method"] == "skip"
        ):
            return
        try:
            self._predictors().launch(
                identity, "experiment-predictors-" + _hash(submission["operationId"])
            )
        except (StorageError, OSError, ValueError, RuntimeError) as error:
            # Fold submission is already durable. Keep its receipt intact and
            # make coordinator startup failures explicitly recoverable.
            submission["predictorStartupError"] = {
                "code": getattr(error, "code", "EXPERIMENT_PREDICTORS_LAUNCH_FAILED"),
                "message": str(error),
            }
            self._save_submission(identity, submission)

    def require_editable(self, identity, expected_revision=None, *, metadata_only=False):
        if identity.startswith("legacy-"):
            raise StorageError(
                "Historical experiments are read-only. Create a new experiment to continue.",
                "LEGACY_EXPERIMENT_READ_ONLY",
                409,
            )
        record = self.store.get_draft(identity)
        if record["kind"] != "experiment" or record["payload"].get("type") != EXPERIMENT_TYPE:
            raise StorageError("Choose a model development experiment.", "INVALID_EXPERIMENT", 422)
        state = self.store.lifecycle.read()["records"].get(f"draft:{identity}", {}).get("state")
        if state == "archived":
            raise StorageError(
                "Restore this archived experiment to Active before editing it or planning batches.",
                "EXPERIMENT_ARCHIVED",
                409,
            )
        if record["status"] != "editable":
            raise StorageError("This experiment record cannot be edited.", "DRAFT_FROZEN", 409)
        if expected_revision is not None and expected_revision != record["revision"]:
            raise StorageError(
                "The experiment changed. Reload it before reviewing or saving.",
                "REVISION_CONFLICT",
                409,
            )
        if not metadata_only and self._configuration_locked(record):
            raise StorageError(
                "This experiment has been submitted. Copy it to adjust inputs or batches.",
                "EXPERIMENT_CONFIGURATION_LOCKED",
                409,
            )
        return record

    def _owned_batches(self, identity):
        return [
            batch
            for batch in self.store.list_configurations("mil-batch", include_inactive=True)
            if batch["manifest"].get("spec", {}).get("experimentId") == identity
        ]

    def _configuration_locked(self, record):
        if record["payload"].get("submission"):
            return True
        for batch in self._owned_batches(record["id"]):
            try:
                if self.training.execution(
                    batch["id"], include_inactive=True, include_progress=False
                ):
                    return True
            except (StorageError, ValueError, OSError, KeyError, TypeError):
                return True
        return False

    @staticmethod
    def _normalize_plans(plans, *, identity, revision, name, inputs, deduplicate=False):
        if plans and not inputs:
            raise StorageError(
                "Choose inputs before saving batch plans.", "EXPERIMENT_INPUTS_REQUIRED", 422
            )
        normalized = [
            {
                "id": plan["id"],
                "spec": DevelopmentBatchSpec.model_validate(
                    {
                        **plan["spec"],
                        "experimentId": identity,
                        "experimentRevision": revision,
                        "experimentName": name,
                        "inputs": inputs,
                    }
                ).model_dump(),
            }
            for plan in plans
        ]
        unique = {}
        for plan in normalized:
            unique.setdefault(_hash(plan["spec"]), plan)
        if len(unique) != len(normalized) and not deduplicate:
            raise StorageError(
                "Two batch plans are identical. Remove the duplicate or give the new batch its own name and settings.",
                "EXPERIMENT_DUPLICATE_BATCH",
                422,
            )
        return list(unique.values())

    def create(self, request):
        values = request.model_dump(exclude={"operationId"})
        if values.get("sourceExperimentId") is None:
            values.pop("sourceExperimentId", None)
        digest = _hash(values)
        with lifecycle_guard(self.store.folder):
            for record in self.store.list_drafts(include_inactive=True):
                payload = record["payload"]
                if (
                    payload.get("type") == EXPERIMENT_TYPE
                    and payload.get("creationOperationId") == request.operationId
                ):
                    if payload.get("creationRequestHash") != digest:
                        raise StorageError(
                            "This operation ID belongs to another experiment request.",
                            "OPERATION_CONFLICT",
                            409,
                        )
                    return self.get(record["id"])
            source = None
            if request.sourceExperimentId:
                source = self.get(request.sourceExperimentId)
                if source["state"] == "trashed":
                    raise StorageError(
                        "Restore the source experiment before copying it.", "RECORD_TRASHED", 409
                    )
                values["inputs"] = deepcopy(source["inputs"])
                values["predictorPolicy"] = deepcopy(
                    source["predictorPolicy"] or ExperimentPredictorPolicy().model_dump()
                )
            plans = []
            if source:
                plans = deepcopy(source["batchPlans"])
                source_payload = (
                    {}
                    if source["legacy"]
                    else self.store.get_draft(source["id"], include_inactive=True)["payload"]
                )
                published = {
                    row.get("batchId")
                    for row in (source_payload.get("submission") or {}).get("publications", [])
                }
                for batch in source["batches"]:
                    if batch["state"] != "trashed" and batch["id"] not in published:
                        plans.append(
                            {
                                "id": "copy-" + uuid4().hex,
                                "spec": deepcopy(batch["manifest"]["spec"]),
                            }
                        )
                for draft in source["drafts"]:
                    if (
                        draft["state"] != "trashed"
                        and draft.get("payload", {}).get("type") == "development-batch"
                    ):
                        try:
                            spec = DevelopmentBatchSpec.model_validate(
                                draft["payload"]["spec"]
                            ).model_dump()
                        except (ValidationError, KeyError) as error:
                            raise StorageError(
                                "An old batch draft is incomplete. Complete its recipe before copying.",
                                "EXPERIMENT_COPY_INVALID",
                                422,
                            ) from error
                        plans.append({"id": "copy-" + uuid4().hex, "spec": spec})
                if len(plans) > 100:
                    raise StorageError(
                        "Limit copied experiments to 100 batch plans.", "EXPERIMENT_TOO_LARGE", 422
                    )
                plans = self._normalize_plans(
                    plans,
                    identity=None,
                    revision=None,
                    name=request.name,
                    inputs=values.get("inputs"),
                    deduplicate=True,
                )
            if values.get("inputs"):
                input_snapshot(self.store, values["inputs"], include_inactive=source is not None)
            # Inputs and recipes are copied in the same transaction. A replay can
            # never observe a partially copied experiment or re-read a changed source.
            record = self.store.create_draft(
                "experiment",
                request.name,
                {
                    "type": EXPERIMENT_TYPE,
                    "version": 2,
                    **{
                        key: value
                        for key, value in values.items()
                        if key not in {"name", "sourceExperimentId"}
                    },
                    "batchPlans": plans,
                    "creationOperationId": request.operationId,
                    "creationRequestHash": digest,
                },
            )
            return self.get(record["id"])

    def update(self, identity, request):
        with lifecycle_guard(self.store.folder):
            record = self.require_editable(identity, request.expectedRevision, metadata_only=True)
            locked = self._configuration_locked(record)
            if locked and any(
                key in request.model_fields_set
                and request.model_dump()[key]
                != record["payload"].get(key, [] if key == "batchPlans" else None)
                for key in ("inputs", "batchPlans", "predictorPolicy")
            ):
                raise StorageError(
                    "Submitted inputs, batch plans and predictor choices cannot change. Copy this experiment to adjust them.",
                    "EXPERIMENT_CONFIGURATION_LOCKED",
                    409,
                )
            if request.inputs:
                input_snapshot(self.store, request.inputs)
            # PATCH omissions retain saved inputs and metadata. Explicit null
            # still clears inputs, and explicit empty notes/tags clear those fields.
            values = {
                key: value
                for key, value in request.model_dump(exclude={"expectedRevision", "name"}).items()
                if key in request.model_fields_set
            }
            if not locked:
                plans = values.get("batchPlans", record["payload"].get("batchPlans", []))
                values["batchPlans"] = self._normalize_plans(
                    plans,
                    identity=identity,
                    revision=record["revision"] + 1,
                    name=request.name,
                    inputs=values.get("inputs", record["payload"].get("inputs")),
                )
            self.store.update_draft(
                identity,
                expected_revision=request.expectedRevision,
                name=request.name,
                payload={**record["payload"], **values},
            )
            return self.get(identity)

    def require_training_action(self, identity, batch_id, *, resume=False):
        record = self.require_editable(identity, metadata_only=True)
        presented = self.get(identity)
        if presented["stage"] == "finished":
            raise StorageError(
                "This experiment is finished. Copy it to start new runs.",
                "EXPERIMENT_FINISHED",
                409,
            )
        submission = record["payload"].get("submission")
        if submission:
            if batch_id not in submission["batchIds"]:
                raise StorageError(
                    "This batch is outside the submitted experiment.",
                    "EXPERIMENT_CONFIGURATION_LOCKED",
                    409,
                )
        elif not resume or not self.training.execution(batch_id, include_inactive=True):
            raise StorageError(
                "Submit this experiment to freeze all inputs and batches before training.",
                "EXPERIMENT_SUBMISSION_REQUIRED",
                409,
            )
        return record

    def _save_submission(self, identity, submission):
        record = self.store.get_draft(identity)
        return self.store.update_draft(
            identity,
            expected_revision=record["revision"],
            name=record["name"],
            payload={**record["payload"], "submission": deepcopy(submission)},
        )

    def submit(self, identity, request):
        from histopilot.application.development import DevelopmentService

        with lifecycle_guard(self.store.folder, timeout=5):
            record = self.require_editable(identity, metadata_only=True)
            submission = deepcopy(record["payload"].get("submission"))
            if submission:
                if submission["operationId"] != request.operationId:
                    raise StorageError(
                        "This experiment is already submitted. Retry its original submission or resume an unfinished batch.",
                        "EXPERIMENT_ALREADY_SUBMITTED",
                        409,
                    )
                if submission["status"] == "submitted":
                    self._start_predictors(identity, submission)
                    return self.get(identity)
            else:
                record = self.require_editable(identity, request.expectedRevision)
                if not record["payload"].get("inputs"):
                    raise StorageError(
                        "Choose experiment inputs before submission.",
                        "EXPERIMENT_INPUTS_REQUIRED",
                        422,
                    )
                plans = self._normalize_plans(
                    record["payload"].get("batchPlans", []),
                    identity=identity,
                    revision=record["revision"],
                    name=record["name"],
                    inputs=record["payload"]["inputs"],
                )
                states = self.store.lifecycle.read()["records"]
                batches = [
                    batch
                    for batch in self._owned_batches(identity)
                    if states.get(f"configuration:{batch['id']}", {}).get("state", "active")
                    == "active"
                ]
                if any(
                    batch["manifest"].get("spec", {}).get("inputs") != record["payload"]["inputs"]
                    for batch in batches
                ):
                    raise StorageError(
                        "An existing frozen batch uses different inputs from this experiment. Archive that batch during planning, then add its settings as an editable batch with the current inputs, or copy the experiment.",
                        "EXPERIMENT_BATCH_INPUTS_MISMATCH",
                        409,
                    )
                if not plans and not batches:
                    raise StorageError(
                        "Save at least one batch plan before submission.",
                        "EXPERIMENT_BATCHES_REQUIRED",
                        422,
                    )
                development = DevelopmentService(self.store, self.filesystem)
                publications, freshness_checks, contracts = [], [], []
                for plan in plans:
                    spec = DevelopmentBatchSpec.model_validate(plan["spec"])
                    preview = development.preview(spec)
                    if not preview["canFreeze"]:
                        raise StorageError(
                            "Resolve batch findings before submission: "
                            + "; ".join(
                                row["message"]
                                for row in preview["findings"]
                                if row["severity"] == "error"
                            ),
                            "BATCH_PREFLIGHT_BLOCKED",
                            409,
                        )
                    manifest = {
                        key: value
                        for key, value in preview.items()
                        if key not in {"canFreeze", "findings"}
                    }
                    _plan, freshness = self.training._prepare(
                        {
                            "id": "submission-preflight",
                            "contentHash": _hash(manifest),
                            "manifest": manifest,
                        }
                    )
                    freshness_checks.append(freshness)
                    contracts.append(execution_contract(_plan))
                    publications.append(
                        {
                            "planId": plan["id"],
                            "spec": spec.model_dump(),
                            "previewHash": preview["previewHash"],
                            "operationId": "experiment-batch-"
                            + _hash(
                                {
                                    "experiment": identity,
                                    "operation": request.operationId,
                                    "plan": plan["id"],
                                }
                            ),
                            "batchId": None,
                        }
                    )
                for batch in batches:
                    _plan, freshness = self.training._prepare(batch)
                    freshness_checks.append(freshness)
                    contracts.append(execution_contract(_plan))
                for contract in contracts[1:]:
                    require_execution_contract(contracts[0], contract)
                for freshness in freshness_checks:
                    freshness()
                submission = {
                    "operationId": request.operationId,
                    "expectedRevision": request.expectedRevision,
                    "submittedAt": datetime.now(UTC).isoformat(),
                    "status": "launching",
                    "batchIds": [batch["id"] for batch in batches],
                    "publications": publications,
                    "launchedBatchIds": [],
                    "executionContract": contracts[0],
                    "predictorPolicy": ExperimentPredictorPolicy.model_validate(
                        record["payload"].get("predictorPolicy", {})
                    ).model_dump(),
                    "error": None,
                    "experiment": {
                        "id": identity,
                        "revision": record["revision"],
                        "name": record["name"],
                        "payload": {
                            key: record["payload"].get(key, [] if key == "tags" else "")
                            for key in ("notes", "tags")
                        },
                    },
                }
                # All scientific/runtime checks precede the irreversible receipt.
                # It is durable before any publication or external worker launch.
                self._save_submission(identity, submission)
            try:
                development = DevelopmentService(self.store, self.filesystem)
                for publication in submission["publications"]:
                    if publication["batchId"]:
                        continue
                    frozen = development._freeze(
                        DevelopmentBatchSpec.model_validate(publication["spec"]),
                        publication["previewHash"],
                        publication["operationId"],
                        {
                            "tag": "Batch " + publication["operationId"][-12:],
                            "note": "Submitted with " + submission["experiment"]["name"],
                        },
                        experiment_record=submission["experiment"],
                    )
                    publication["batchId"] = frozen["id"]
                    if frozen["id"] not in submission["batchIds"]:
                        submission["batchIds"].append(frozen["id"])
                    self._save_submission(identity, submission)
                for batch_id in submission["batchIds"]:
                    if batch_id in submission["launchedBatchIds"]:
                        continue
                    execution = self.training.execution(
                        batch_id, include_inactive=True, include_progress=False
                    )
                    failed_start = (
                        execution
                        and execution["status"] == "failed"
                        and any(
                            item.get("code") == "TRAINING_LAUNCH_FAILED"
                            for item in execution.get("findings", [])
                        )
                    )
                    if execution and not failed_start:
                        # The worker may have accepted or finished before either
                        # launch acknowledgement was saved. Existing execution
                        # evidence owns this batch; never dispatch it again.
                        submission["launchedBatchIds"].append(batch_id)
                        self._save_submission(identity, submission)
                        continue
                    operation = "experiment-launch-" + _hash(
                        {
                            "experiment": identity,
                            "operation": request.operationId,
                            "batch": batch_id,
                        }
                    )
                    self.training.launch(batch_id, operation)
                    submission["launchedBatchIds"].append(batch_id)
                    self._save_submission(identity, submission)
                submission.update(status="submitted", error=None)
            except (StorageError, OSError, ValueError, RuntimeError) as error:
                submission.update(
                    status="attention",
                    error={
                        "code": getattr(error, "code", "EXPERIMENT_SUBMISSION_FAILED"),
                        "message": str(error),
                    },
                )
            self._save_submission(identity, submission)
            if submission["status"] == "submitted":
                self._start_predictors(identity, submission)
            return self.get(identity)

    @staticmethod
    def _inputs(record):
        payload = record.get("payload", {})
        value = payload.get("inputs")
        if payload.get("type") in LEGACY_DRAFT_TYPES:
            spec = _mapping(payload.get("spec"))
            value = spec.get("inputs") if payload["type"] == "development-batch" else spec
        if not value:
            return None
        try:
            return MILInputSpec.model_validate(value).model_dump()
        except ValidationError:
            # Keep incomplete historical drafts visible, without inventing inputs.
            return None

    def _batch(self, record, states, *, summary=False):
        key = f"configuration:{record['id']}"
        manifest = record["manifest"]
        execution_error = None
        try:
            execution = self.training.execution(
                record["id"], include_inactive=True, include_progress=not summary
            )
        except (StorageError, ValueError, OSError, KeyError, TypeError) as error:
            # One damaged execution receipt must not hide every experiment. This
            # is not a stopped-job assertion: cleanup and launch retain their
            # own fail-closed evidence checks.
            execution = None
            execution_error = {
                "code": getattr(error, "code", "TRAINING_STATE_INVALID"),
                "message": "Execution status is unavailable. Inspect this batch's training evidence before continuing.",
            }
        snapshot = manifest.get("inputSnapshot")
        if not summary and snapshot is None and manifest.get("spec", {}).get("inputs"):
            try:
                snapshot = input_snapshot(
                    self.store, manifest["spec"]["inputs"], include_inactive=True
                )
                snapshot.update(
                    resolvedInputs=manifest.get("resolvedInputs", {}),
                    configurations=manifest.get("configurations", []),
                    trainingSeeds=manifest["spec"].get("trainingSeeds", []),
                    resources=manifest["spec"].get("resources", {}),
                )
            except StorageError:
                # A historical record remains discoverable even if its old inputs
                # cannot be resolved. Its original manifest is still returned.
                snapshot = None
        result = {
            **record,
            "key": key,
            "name": manifest.get("spec", {}).get("batchName", record["id"]),
            "state": states.get(key, {}).get("state", "active"),
            "status": (
                "unknown"
                if execution_error
                else "cancelling"
                if execution
                and execution.get("cancelRequested")
                and execution["status"] in {"queued", "running"}
                else execution["status"]
                if execution
                else "planned"
            ),
            "inputSnapshot": snapshot,
            "execution": execution,
            "executionError": execution_error,
        }
        if summary:
            result.pop("execution")
            result.pop("inputSnapshot")
            result["manifest"] = {
                "kind": manifest["kind"],
                "version": manifest.get("version", 1),
                "summary": manifest.get("summary", {}),
                "spec": {
                    name: manifest["spec"][name]
                    for name in (
                        "experimentId",
                        "experimentRevision",
                        "experimentName",
                        "batchName",
                        "inputs",
                    )
                    if name in manifest.get("spec", {})
                },
            }
        return result

    def _record(self, record, batches, drafts, states, predictors, *, legacy=False, summary=False):
        resource_type = "configuration" if "manifest" in record else "draft"
        key = f"{resource_type}:{record['id']}"
        payload = record.get("payload", {})
        spec = record.get("manifest", {}).get("spec", {})
        inputs = spec.get("inputs") if resource_type == "configuration" else self._inputs(record)
        identity = legacy_experiment_id(record["id"]) if legacy else record["id"]
        snapshot = None
        if inputs and not summary:
            try:
                snapshot = input_snapshot(self.store, inputs, include_inactive=True)
            except StorageError:
                pass
        predictor_records = sorted(predictors.get(identity, []), key=lambda row: row["id"])
        predictor_ids = {}
        for item in predictor_records:
            predictor_ids.setdefault(item["method"], item["id"])
        submission = payload.get("submission")
        stage, locked = experiment_stage(batches, submission, legacy=legacy)
        predictor_policy = (
            submission.get("predictorPolicy")
            if submission
            else None
            if legacy
            else payload.get("predictorPolicy", ExperimentPredictorPolicy().model_dump())
        )
        predictor_execution = None
        if submission and predictor_policy and predictor_policy["method"] != "skip":
            try:
                predictor_execution = self._predictors().status(identity, summary=summary)
                if predictor_execution is None:
                    predictor_execution = self._predictors().pending(
                        identity,
                        submission.get("predictorStartupError")
                        or {
                            "code": "EXPERIMENT_PREDICTORS_NOT_STARTED",
                            "message": "Predictor coordination has not started. Finish submission, then resume predictor creation.",
                        },
                        summary=summary,
                    )
            except (StorageError, OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
                predictor_execution = self._predictors().public(
                    {
                        "status": "attention",
                        "items": [],
                        "error": {
                            "code": getattr(error, "code", "EXPERIMENT_PREDICTORS_INVALID"),
                            "message": str(error),
                        },
                    },
                    summary=summary,
                )
            if stage == "finished" and predictor_execution["status"] not in {
                "completed",
                "cancelled",
            }:
                stage = "running"
        status_rows = batches
        if predictor_execution:
            predictor_status = {"waiting": "queued", "attention": "failed"}.get(
                predictor_execution["status"], predictor_execution["status"]
            )
            status_rows = [*batches, {"state": "active", "status": predictor_status}]
        public_submission = (
            None
            if not submission
            else {
                **{
                    key: submission[key]
                    for key in (
                        "operationId",
                        "expectedRevision",
                        "submittedAt",
                        "status",
                        "batchIds",
                        "error",
                    )
                },
                "retryable": submission["status"] != "submitted" and stage != "finished",
            }
        )
        return {
            "id": identity,
            "projectId": self.store.project_id,
            "key": key,
            "name": record.get("name")
            or spec.get("experimentName")
            or spec.get("batchName")
            or record["id"],
            "notes": payload.get("notes", spec.get("notes", "")),
            "tags": payload.get("tags", []),
            "revision": record.get("revision", 1),
            "state": states.get(key, {}).get("state", "active"),
            "status": aggregate_status(status_rows, inputs is not None or bool(drafts)),
            "stage": stage,
            "configurationLocked": locked,
            "predictorPolicy": predictor_policy,
            "predictorExecution": predictor_execution,
            "batchPlans": payload.get("batchPlans", []),
            "submission": public_submission,
            "legacy": legacy,
            "createdAt": record["createdAt"],
            "updatedAt": record.get("updatedAt", record["createdAt"]),
            "inputs": inputs,
            "inputSnapshot": snapshot,
            "batches": batches,
            "drafts": drafts,
            # Keep the legacy singular field for older clients; expose both methods.
            "predictorId": predictor_ids.get("ensemble") or predictor_ids.get("refit"),
            "predictorIds": predictor_ids,
            "predictors": predictor_records,
            "summary": summary,
            # Current service capability is separate from historical manifests,
            # which may predate the execution adapter and contain false here.
            "executionImplemented": True,
        }

    def list(self, *, state="all", summary=False, _identity=None):
        with lifecycle_guard(self.store.folder):
            states = self.store.lifecycle.read()["records"]
            drafts = self.store.list_drafts(include_inactive=True)
            configurations = self.store.list_configurations(include_inactive=True)
            records = [
                record for record in drafts if record["payload"].get("type") == EXPERIMENT_TYPE
            ]
            known = {record["id"] for record in records}
            batches, saved, legacy = {}, {}, []
            predictors = {}
            for record in configurations:
                manifest = record["manifest"]
                if manifest.get("kind") == "frozen-predictor" and manifest.get("experimentId"):
                    predictors.setdefault(manifest["experimentId"], []).append(
                        {
                            "id": record["id"],
                            "method": manifest.get("method", "ensemble"),
                            "batchId": manifest.get("batchId"),
                            "candidateId": manifest.get("candidateId"),
                            "trainingSeed": manifest.get("trainingSeed"),
                            "splitSeed": manifest.get("splitSeed"),
                            "lifecycleState": states.get(f"configuration:{record['id']}", {}).get(
                                "state", "active"
                            ),
                        }
                    )
            for record in configurations:
                if record["manifest"].get("kind") != "mil-batch":
                    continue
                owner = record["manifest"].get("spec", {}).get("experimentId")
                owned = isinstance(owner, str) and owner in known
                row_id = owner if owned else legacy_experiment_id(record["id"])
                if _identity is not None and row_id != _identity:
                    continue
                presented = self._batch(record, states, summary=summary)
                if owned:
                    batches.setdefault(owner, []).append(presented)
                else:
                    legacy.append(
                        self._record(
                            record,
                            [presented],
                            [],
                            states,
                            predictors,
                            legacy=True,
                            summary=summary,
                        )
                    )
            for record in drafts:
                payload = record["payload"]
                if payload.get("type") not in LEGACY_DRAFT_TYPES:
                    continue
                owner = payload.get("experimentId") or _mapping(payload.get("spec")).get(
                    "experimentId"
                )
                owned = isinstance(owner, str) and owner in known
                row_id = owner if owned else legacy_experiment_id(record["id"])
                if _identity is not None and row_id != _identity:
                    continue
                presented = {
                    **record,
                    "state": states.get(f"draft:{record['id']}", {}).get("state", "active"),
                }
                if summary:
                    presented = {key: value for key, value in presented.items() if key != "payload"}
                if owned:
                    saved.setdefault(owner, []).append(presented)
                else:
                    legacy.append(
                        self._record(
                            record,
                            [],
                            [presented],
                            states,
                            predictors,
                            legacy=True,
                            summary=summary,
                        )
                    )
            items = [
                self._record(
                    record,
                    batches.get(record["id"], []),
                    saved.get(record["id"], []),
                    states,
                    predictors,
                    summary=summary,
                )
                for record in records
                if _identity is None or record["id"] == _identity
            ] + legacy
            return {
                "items": sorted(
                    [item for item in items if state == "all" or item["state"] == state],
                    key=lambda item: (item["createdAt"], item["id"]),
                    reverse=True,
                )
            }

    def get(self, identity):
        for record in self.list(_identity=identity)["items"]:
            if record["id"] == identity:
                return record
        raise StorageError(
            "This experiment does not exist in this project.", "EXPERIMENT_NOT_FOUND", 404
        )
