"""Editable experiment plans become irrevocable submissions before work starts.

Submission pins inputs, batch recipes and a common execution environment. Worker
evidence determines progress; idempotent receipts recover partial dispatch. Older
batches and saved plans remain readable without rewriting their evidence.
"""

import sqlite3
from copy import deepcopy
from datetime import UTC, datetime
from uuid import uuid4

from pydantic import ValidationError

from histopilot.application.experiment_policy import (
    predictor_work_expected,
    resolve_batch_policy,
    submission_policies,
)
from histopilot.application.feature_bundles import FeatureBundleService, _hash
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


def presented_batch_plans(payload):
    """Use the submitted recipe when its publication resolved an old default."""
    submission = payload.get("submission") or {}
    frozen = {row["planId"]: row["spec"] for row in submission.get("publications", [])}
    return [
        {"id": row["id"], "spec": deepcopy(frozen.get(row["id"], row["spec"]))}
        for row in payload.get("batchPlans", [])
    ]


def matching_published_batch(publication, batches):
    """Reuse exact reviewed contents already pinned by this submission.

    An older planning workflow may have frozen a batch while retaining its
    editable recipe. Its tag is metadata and must not trigger a second freeze.
    Verify the complete preview digest as well as the saved specification.
    """
    for batch in batches:
        manifest = batch["manifest"]
        if (
            manifest.get("spec") == publication["spec"]
            and manifest.get("previewHash") == publication["previewHash"]
            and _hash({key: value for key, value in manifest.items() if key != "previewHash"})
            == publication["previewHash"]
        ):
            return batch
    return None


def input_snapshot(store, inputs, *, include_inactive=False):
    """Small exact specifications and immutable hashes, never feature tensors."""
    values = inputs.model_dump() if isinstance(inputs, MILInputSpec) else inputs
    protocol = store.get_configuration(values["protocolId"], include_inactive=include_inactive)
    bundle = store.get_configuration(values["featureBundleId"], include_inactive=include_inactive)
    if protocol["manifest"].get("kind") != "protocol":
        raise StorageError("Choose a target and split protocol.", "INVALID_PROTOCOL", 422)
    if bundle["manifest"].get("kind") != "feature-bundle":
        raise StorageError("Choose a frozen feature bundle.", "INVALID_FEATURE_BUNDLE", 422)
    from histopilot.application.mil_inputs import protocol_bundle_findings

    findings = protocol_bundle_findings(protocol["manifest"], bundle)
    if findings:
        raise StorageError(findings[0]["message"], findings[0]["code"], 422)
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


# One execution status for a submitted experiment, shown by the library, the detail
# header and the setup list. Live work first, then anything that needs a person.
EXECUTION_STATUSES = (
    "running",
    "queued",
    "held",
    "waiting",
    "needs-attention",
    "cancelled",
    "completed",
)
LIVE_EXECUTION = frozenset({"running", "queued", "held", "waiting"})
# Batch, submission, predictor-coordinator and Task Center states in that vocabulary.
_EXECUTION_STATUS = {
    "running": "running",
    "cancelling": "running",
    "starting": "running",
    "stopping": "running",
    "queued": "queued",
    "scheduled": "queued",
    "launching": "queued",
    "held": "held",
    "waiting": "waiting",
    "blocked": "waiting",
    "needs-attention": "needs-attention",
    "attention": "needs-attention",
    "failed": "needs-attention",
    "interrupted": "needs-attention",
    "unknown": "needs-attention",
    "cancelled": "cancelled",
    "completed": "completed",
    "succeeded": "completed",
}


def execution_status(statuses):
    """The most urgent execution status among parts; ``None`` if none has run."""
    found = {_EXECUTION_STATUS.get(status) for status in statuses} - {None}
    return next((status for status in EXECUTION_STATUSES if status in found), None)


def _task_batch(task):
    """The training batch of a task, as the Task Center's rollups read it."""
    group = task.get("group") or {}
    return (
        (group.get("id") if group.get("kind") == "mil-batch" else None)
        or (task.get("adapterData") or {}).get("batchId")
        or (task.get("labels") or {}).get("batchId")
    )


def _task_counts(task, batch_ids):
    """Whether a task still speaks for its experiment (mirrors the Task Center rollups).

    Live work always counts. A progress collection that stopped short is superseded by
    its batch's final collection, and a finished task of a batch outside ``batch_ids``
    (trashed, or left behind) no longer describes the experiment.
    """
    from histopilot.taskcenter.model import LIVE, awaiting_requeue

    if task["state"] in LIVE or awaiting_requeue(task):
        return True
    if (
        task.get("kind") == "mil-collect"
        and (task.get("adapterData") or {}).get("final") in (False, 0)
        and task["state"] in {"failed", "cancelled", "interrupted"}
    ):
        return False
    batch = _task_batch(task)
    return batch_ids is None or batch is None or batch in batch_ids


def current_batch_ids(batches, submission):
    """The batches whose Task Center tasks still speak for an experiment: those its
    training may still resume (``require_training_action``), never a trashed one.

    ``batches`` are rows with ``id`` and lifecycle ``state``. A submitted experiment has
    only ever run its submitted batches (their records exist before they are submitted);
    one without a submission keeps its own batches.
    """
    trashed = {batch["id"] for batch in batches if batch["state"] == "trashed"}
    if submission:
        return set(submission.get("batchIds") or []) - trashed
    return {batch["id"] for batch in batches} - trashed


def task_execution(tasks, *, held=False, runner_alive=True, batch_ids=None):
    """``(status, reason)`` for one Task Center owner's tasks; ``None`` without tasks.

    A failed coordinator reads as needing attention, not as a failed experiment, and
    work the runner will requeue (a busy project, an OOM retry) reads as waiting.
    ``batch_ids`` (the experiment's current batches) leaves out finished tasks of other
    batches; see ``_task_counts``.
    """
    from histopilot.taskcenter.model import ACTIVE, PENDING, awaiting_requeue

    tasks = [task for task in tasks if _task_counts(task, batch_ids)]
    if not tasks:
        return None
    pending = [task for task in tasks if task["state"] in PENDING]
    awaiting = [task for task in tasks if awaiting_requeue(task)]
    if any(task["state"] in ACTIVE for task in tasks):
        return "running", None
    if pending or awaiting:
        if held:
            return "held", "Held in the Task Center. It starts after it is released there."
        if not runner_alive:
            return "waiting", "The Task Center runner is stopped; queued work starts once it runs."
        reason = next(
            (task["waitingReason"] for task in pending if task.get("waitingReason")), None
        )
        if any(task["state"] == "queued" for task in pending):
            return "queued", reason
        return "waiting", reason or (
            "Waiting for earlier tasks to finish."
            if pending
            else "Retrying automatically after a temporary problem."
        )
    failed = [task for task in tasks if task["state"] in {"failed", "interrupted"}]
    if failed:
        error = failed[0].get("error")
        return "needs-attention", error if isinstance(error, str) and error else None
    if any(task["state"] == "cancelled" for task in tasks):
        return "cancelled", None
    return "completed", None


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
    # Task Center submissions stay resumable: a cancelled batch keeps the stage running.
    settled = (
        {"completed"}
        if (submission or {}).get("executionMode") == "task-center"
        else {"completed", "cancelled"}
    )
    finished = (
        bool(stage_batches)
        and (not submission or submission.get("status") == "submitted")
        and complete_intent
        and expected <= actual
        and all(row["status"] in settled for row in stage_batches)
    )
    return ("finished" if finished else "running"), True


def predictors_settled(submission, execution) -> bool:
    """Whether predictor creation lets the experiment finish.

    Task Center experiments stay resumable after their predictors are cancelled.
    """
    managed = (submission or {}).get("executionMode") == "task-center" or (
        execution.get("executor") == "task-center"
    )
    return execution["status"] in ({"completed"} if managed else {"completed", "cancelled"})


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


def pinned_compute(store, submission):
    """The submitted worker code and a verified archive to copy it from, or None.

    Work added after submission, such as a refit launched later or a batch whose launch is
    retried, must run the code the experiment was reviewed with rather than whatever the
    live checkout holds now. Every launched batch archived that code; None before any has.
    """
    from histopilot.workers.compute_archive import prepare_compute_archive

    code = ((submission or {}).get("executionContract") or {}).get("code")
    if not code:
        return None
    for batch_id in submission.get("batchIds") or []:
        folder = store.folder / "training" / batch_id
        if (folder / "compute").is_dir():
            return code, prepare_compute_archive(folder, code) / "histopilot"
    return None


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

    def task_batch_ids(self, identity):
        """``current_batch_ids`` of one experiment, read without the project lock (a Task
        Center owner retry resumes only these batches)."""
        record = self.store.get_draft(identity, include_inactive=True)
        submission = record["payload"].get("submission")
        states = self.store.lifecycle.read()["records"]
        ids = (
            submission.get("batchIds") or []
            if submission
            else [batch["id"] for batch in self._owned_batches(identity)]
        )
        return current_batch_ids(
            [
                {"id": batch, "state": states.get(f"configuration:{batch}", {}).get("state")}
                for batch in ids
            ],
            submission,
        )

    def _task_execution(self, identity, batch_ids=None):
        """The experiment owner's Task Center status, read without any project lock.

        ``None`` when the owner has no tasks or the task store cannot be read; the
        batch and coordinator evidence then decide the status alone. ``batch_ids``: the
        batches whose finished tasks still count (``task_execution``).
        """
        from histopilot.taskcenter import ids

        try:
            client = getattr(self.training, "task_center", None)
            if client is None:
                from histopilot.taskcenter.client import default_client

                client = default_client()
            key = ids.owner_key("experiment", identity, str(self.store.folder))
            tasks = client.store.list(owner_key=key, limit=None)
            if not tasks:
                return None
            owner = client.store.owner(key) or {}
            return task_execution(
                tasks,
                held=bool(owner.get("held")),
                runner_alive=bool(client.runner_alive()),
                batch_ids=batch_ids,
            )
        except (StorageError, OSError, ValueError, sqlite3.Error):
            return None

    def _start_predictors(self, identity, submission):
        if not predictor_work_expected(submission):
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
                "This experimental setup is frozen or submitted. Copy it to adjust inputs or batches.",
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
        if record["payload"].get("submission") or record["payload"].get("frozenSetupId"):
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
    def _normalize_plans(
        plans, *, identity, revision, name, inputs, deduplicate=False, predictor_policy=None
    ):
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
                        "predictorPolicy": resolve_batch_policy(plan["spec"], predictor_policy),
                    },
                    context={"legacy": True},
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
        if values.get("setupVersion") is None:
            values.pop("setupVersion", None)
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
                if source.get("setupVersion") or source.get("setupDesign"):
                    values["setupVersion"] = 1
                    values["setupDesign"] = deepcopy(source.get("setupDesign"))
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
                                "spec": {
                                    **deepcopy(batch["manifest"]["spec"]),
                                    "predictorPolicy": resolve_batch_policy(
                                        batch["manifest"]["spec"],
                                        (source.get("predictorPolicies") or {}).get(
                                            batch["id"], source.get("predictorPolicy")
                                        ),
                                    ),
                                },
                            }
                        )
                for draft in source["drafts"]:
                    if (
                        draft["state"] != "trashed"
                        and draft.get("payload", {}).get("type") == "development-batch"
                    ):
                        try:
                            spec = DevelopmentBatchSpec.model_validate(
                                draft["payload"]["spec"], context={"legacy": True}
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
                    predictor_policy=values.get("predictorPolicy"),
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
                    "version": 3 if values.get("setupVersion") else 2,
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
                    "Frozen inputs, batch plans and predictor choices cannot change. Copy this experiment to adjust them.",
                    "EXPERIMENT_CONFIGURATION_LOCKED",
                    409,
                )
            if (record["payload"].get("setupVersion") or record["payload"].get("setupDesign")) and (
                "inputs" in request.model_fields_set
                and request.model_dump()["inputs"] != record["payload"].get("inputs")
            ):
                raise StorageError(
                    "Choose dataset, targets, features and training design through Experimental Setup.",
                    "EXPERIMENT_SETUP_INPUTS_REQUIRED",
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
                    predictor_policy=values.get(
                        "predictorPolicy", record["payload"].get("predictorPolicy")
                    ),
                )
            self.store.update_draft(
                identity,
                expected_revision=request.expectedRevision,
                name=request.name,
                payload={**record["payload"], **values},
            )
            return self.get(identity)

    def setup_inputs(self, identity, request):
        from histopilot.application.mil_inputs import MILInputService
        from histopilot.application.target_splits import TargetSplitService

        with lifecycle_guard(self.store.folder):
            record = self.require_editable(identity, request.expectedRevision)
            if request.trainingSplit.mode != "kfold":
                raise StorageError(
                    "Experimental Setup currently supports k-fold training. Choose k-fold before verifying inputs.",
                    "TRAINING_SPLIT_UNSUPPORTED",
                    422,
                )
            target_split = self.store.get_configuration(request.targetSplitId)
            manifest = target_split["manifest"]
            if manifest.get("kind") != "target-split":
                raise StorageError(
                    "Choose frozen targets and train/test populations.", "INVALID_TARGET_SPLIT", 422
                )
            if manifest.get("datasetId") != request.datasetId:
                raise StorageError(
                    "The target split belongs to a different dataset.",
                    "EXPERIMENT_DATASET_MISMATCH",
                    422,
                )
            self.store.get_dataset(request.datasetId)
            protocol = TargetSplitService(self.store, self.filesystem).derive_protocol(
                request.targetSplitId, request.trainingSplit
            )
            inputs = MILInputSpec(
                protocolId=protocol["id"],
                featureBundleId=request.featureBundleId,
                loadingPolicy=request.loadingPolicy,
                packArtifactId=request.packArtifactId,
            )
            self._require_input_compatibility(
                MILInputService(self.store, self.filesystem).preview(inputs)
            )
            design = {
                "datasetId": request.datasetId,
                "targetSplitId": request.targetSplitId,
                "splitUnit": manifest["spec"].get("splitUnit", "patient"),
                "trainingSplit": request.trainingSplit.model_dump(mode="json"),
            }
            plans = self._normalize_plans(
                record["payload"].get("batchPlans", []),
                identity=identity,
                revision=record["revision"] + 1,
                name=record["name"],
                inputs=inputs.model_dump(),
                predictor_policy=record["payload"].get("predictorPolicy"),
            )
            self.store.update_draft(
                identity,
                expected_revision=request.expectedRevision,
                name=record["name"],
                payload={
                    **record["payload"],
                    "version": 3,
                    "setupVersion": 1,
                    "setupDesign": design,
                    "inputs": inputs.model_dump(),
                    "batchPlans": plans,
                },
            )
            return self.get(identity)

    @staticmethod
    def _require_input_compatibility(review):
        if not review["canPlan"]:
            raise StorageError(
                "Resolve experiment input compatibility: "
                + "; ".join(
                    row["message"] for row in review["findings"] if row["severity"] == "error"
                ),
                "EXPERIMENT_INPUTS_INVALID",
                422,
            )

    def _frozen_setup(self, record):
        identity = record["payload"].get("frozenSetupId")
        if not identity:
            raise StorageError(
                "Freeze Experimental Setup before starting this experiment.",
                "EXPERIMENT_SETUP_REQUIRED",
                409,
            )
        document = self.store.get_configuration(identity)
        manifest = document["manifest"]
        if (
            manifest.get("kind") != "experiment-setup"
            or manifest.get("experimentId") != record["id"]
        ):
            raise StorageError(
                "The frozen setup does not belong to this experiment.",
                "EXPERIMENT_SETUP_CHANGED",
                409,
            )
        for key in ("inputs", "setupDesign", "batchPlans", "predictorPolicy"):
            if record["payload"].get(key) != manifest.get(key):
                raise StorageError(
                    "The experimental setup no longer matches its frozen configuration.",
                    "EXPERIMENT_SETUP_CHANGED",
                    409,
                )
        return document

    def freeze_setup(self, identity, request):
        from histopilot.application.development import DevelopmentService
        from histopilot.application.mil_inputs import MILInputService

        with lifecycle_guard(self.store.folder):
            record = self.require_editable(identity, metadata_only=True)
            previous = record["payload"].get("setupFreeze")
            if record["payload"].get("frozenSetupId"):
                if previous != request.model_dump():
                    raise StorageError(
                        "This setup is already frozen. Copy it to make changes.",
                        "EXPERIMENT_SETUP_FROZEN",
                        409,
                    )
                self._frozen_setup(record)
                return self.get(identity)
            record = self.require_editable(identity, request.expectedRevision)
            payload = record["payload"]
            if not payload.get("setupDesign") or not payload.get("inputs"):
                raise StorageError(
                    "Save the dataset, target split, features and training design first.",
                    "EXPERIMENT_SETUP_INPUTS_REQUIRED",
                    422,
                )
            operation = "experiment-setup-" + _hash(
                {"experimentId": identity, "operationId": request.operationId}
            )
            prior = self.store.configuration_publication(operation)
            if prior:
                manifest = prior["manifest"]
                if (
                    manifest.get("experimentId") != identity
                    or manifest.get("setupFreeze") != request.model_dump()
                ):
                    raise StorageError(
                        "This operation belongs to another setup request.",
                        "OPERATION_CONFLICT",
                        409,
                    )
                for key in ("setupDesign", "inputs", "predictorPolicy"):
                    if manifest.get(key) != payload.get(key):
                        raise StorageError(
                            "The setup changed after its publication.",
                            "EXPERIMENT_SETUP_CHANGED",
                            409,
                        )
                frozen = prior
            else:
                self._require_input_compatibility(
                    MILInputService(self.store, self.filesystem).preview(
                        MILInputSpec.model_validate(payload["inputs"])
                    )
                )
                target_split = self.store.get_configuration(payload["setupDesign"]["targetSplitId"])
                if target_split["manifest"].get("datasetId") != payload["setupDesign"]["datasetId"]:
                    raise StorageError(
                        "The target split belongs to a different dataset.",
                        "EXPERIMENT_DATASET_MISMATCH",
                        422,
                    )
                plans = self._normalize_plans(
                    payload.get("batchPlans", []),
                    identity=identity,
                    revision=record["revision"],
                    name=record["name"],
                    inputs=payload["inputs"],
                    predictor_policy=payload.get("predictorPolicy"),
                )
                states = self.store.lifecycle.read()["records"]
                batches = [
                    batch
                    for batch in self._owned_batches(identity)
                    if states.get(f"configuration:{batch['id']}", {}).get("state", "active")
                    == "active"
                ]
                if not plans and not batches:
                    raise StorageError(
                        "Save at least one batch plan before freezing setup.",
                        "EXPERIMENT_BATCHES_REQUIRED",
                        422,
                    )
                development = DevelopmentService(self.store, self.filesystem)
                previews = []
                for plan in plans:
                    preview = development._preview(
                        DevelopmentBatchSpec.model_validate(plan["spec"], context={"legacy": True}),
                        experiment_record=record,
                    )
                    if not preview["canFreeze"]:
                        raise StorageError(
                            "Resolve batch findings before freezing setup: "
                            + "; ".join(
                                row["message"]
                                for row in preview["findings"]
                                if row["severity"] == "error"
                            ),
                            "BATCH_PREFLIGHT_BLOCKED",
                            422,
                        )
                    previews.append({"planId": plan["id"], "previewHash": preview["previewHash"]})
                if any(
                    batch["manifest"].get("spec", {}).get("inputs") != payload["inputs"]
                    for batch in batches
                ):
                    raise StorageError(
                        "Existing batches use different experiment inputs.",
                        "EXPERIMENT_BATCH_INPUTS_MISMATCH",
                        409,
                    )
                manifest = {
                    "kind": "experiment-setup",
                    "version": 1,
                    "experimentId": identity,
                    "datasetId": payload["setupDesign"]["datasetId"],
                    "setupDesign": deepcopy(payload["setupDesign"]),
                    "inputs": deepcopy(payload["inputs"]),
                    "inputSnapshot": input_snapshot(self.store, payload["inputs"]),
                    "targetSplit": {key: target_split[key] for key in ("id", "contentHash")},
                    "batchPlans": plans,
                    "batchPreviews": previews,
                    "batches": [
                        {key: batch[key] for key in ("id", "contentHash")} for batch in batches
                    ],
                    "predictorPolicy": deepcopy(payload.get("predictorPolicy")),
                    "setupFreeze": request.model_dump(),
                    "experiment": {
                        "id": identity,
                        "revision": record["revision"],
                        "name": record["name"],
                        "payload": {
                            key: deepcopy(payload.get(key, [] if key == "tags" else ""))
                            for key in ("notes", "tags")
                        },
                    },
                }
                bundle_service = FeatureBundleService(self.store, self.filesystem)
                bundle = bundle_service.get(payload["inputs"]["featureBundleId"])
                feature = self.store.get_configuration(bundle["manifest"]["spec"]["featureSetId"])
                freshness = bundle_service._freshness_guard(feature, bundle["manifest"])
                frozen = self.store.publish_configuration(
                    manifest=manifest, operation_id=operation, before_publish=freshness
                )
            self.store.update_draft(
                identity,
                expected_revision=record["revision"],
                name=record["name"],
                payload={
                    **payload,
                    "setupVersion": 1,
                    "batchPlans": frozen["manifest"]["batchPlans"],
                    "frozenSetupId": frozen["id"],
                    "setupFreeze": request.model_dump(),
                },
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
                pipeline = bool(
                    record["payload"].get("setupVersion") or record["payload"].get("setupDesign")
                )
                frozen_setup = self._frozen_setup(record)["manifest"] if pipeline else None
                record = self.require_editable(
                    identity, request.expectedRevision, metadata_only=pipeline
                )
                experiment_record = frozen_setup["experiment"] if frozen_setup else record
                if (
                    frozen_setup
                    and input_snapshot(self.store, frozen_setup["inputs"])
                    != frozen_setup["inputSnapshot"]
                ):
                    raise StorageError(
                        "The frozen setup inputs changed. Copy the setup and review the current inputs.",
                        "EXPERIMENT_SETUP_CHANGED",
                        409,
                    )
                if not record["payload"].get("inputs"):
                    raise StorageError(
                        "Choose experiment inputs before submission.",
                        "EXPERIMENT_INPUTS_REQUIRED",
                        422,
                    )
                plans = (
                    deepcopy(frozen_setup["batchPlans"])
                    if frozen_setup
                    else self._normalize_plans(
                        record["payload"].get("batchPlans", []),
                        identity=identity,
                        revision=record["revision"],
                        name=record["name"],
                        inputs=record["payload"]["inputs"],
                        predictor_policy=record["payload"].get("predictorPolicy"),
                    )
                )
                states = self.store.lifecycle.read()["records"]
                owned_batches = self._owned_batches(identity)
                batches = [
                    batch
                    for batch in owned_batches
                    if states.get(f"configuration:{batch['id']}", {}).get("state", "active")
                    == "active"
                ]
                if frozen_setup:
                    batches = []
                    for expected in frozen_setup["batches"]:
                        batch = self.store.get_configuration(expected["id"])
                        if (
                            batch["contentHash"] != expected["contentHash"]
                            or states.get(f"configuration:{batch['id']}", {}).get("state", "active")
                            != "active"
                        ):
                            raise StorageError(
                                "A batch in this frozen setup changed or is inactive.",
                                "EXPERIMENT_SETUP_CHANGED",
                                409,
                            )
                        batches.append(batch)
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
                    spec = DevelopmentBatchSpec.model_validate(
                        plan["spec"], context={"legacy": True}
                    )
                    preview = (
                        development._preview(spec, experiment_record=experiment_record)
                        if frozen_setup
                        else development.preview(spec)
                    )
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
                    if (
                        frozen_setup
                        and {"planId": plan["id"], "previewHash": preview["previewHash"]}
                        not in frozen_setup["batchPreviews"]
                    ):
                        raise StorageError(
                            "The batch no longer matches its frozen experimental setup.",
                            "EXPERIMENT_SETUP_CHANGED",
                            409,
                        )
                    manifest = {
                        key: value
                        for key, value in preview.items()
                        if key not in {"canFreeze", "findings"}
                    }
                    for existing in owned_batches:
                        existing_state = states.get(f"configuration:{existing['id']}", {}).get(
                            "state", "active"
                        )
                        if existing_state != "active" and existing["manifest"] == manifest:
                            raise StorageError(
                                f"Batch '{spec.batchName}' already exists in the {existing_state} "
                                "state with these exact settings. Restore that batch to Active "
                                "or change this plan before submitting the experiment.",
                                "EXPERIMENT_BATCH_INACTIVE",
                                409,
                            )
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
                    "predictorPolicyVersion": 2,
                    "predictorPolicies": {
                        batch["id"]: resolve_batch_policy(
                            batch["manifest"]["spec"], record["payload"].get("predictorPolicy")
                        )
                        for batch in batches
                    },
                    "error": None,
                    "executionMode": "task-center",
                    "experiment": {
                        "id": identity,
                        "revision": experiment_record["revision"],
                        "name": experiment_record["name"],
                        "payload": {
                            key: experiment_record["payload"].get(key, [] if key == "tags" else "")
                            for key in ("notes", "tags")
                        },
                    },
                    **(
                        {"frozenSetupId": record["payload"]["frozenSetupId"]}
                        if frozen_setup
                        else {}
                    ),
                }
                # All scientific/runtime checks precede the irreversible receipt.
                # It is durable before any publication or external worker launch.
                self._save_submission(identity, submission)
            try:
                development = DevelopmentService(self.store, self.filesystem)
                pinned_batches = [
                    self.store.get_configuration(batch_id) for batch_id in submission["batchIds"]
                ]
                for publication in submission["publications"]:
                    if publication["batchId"]:
                        continue
                    frozen = matching_published_batch(publication, pinned_batches)
                    if frozen is None:
                        frozen = development._freeze(
                            DevelopmentBatchSpec.model_validate(
                                publication["spec"], context={"legacy": True}
                            ),
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
                    if "predictorPolicies" in submission:
                        submission["predictorPolicies"][frozen["id"]] = resolve_batch_policy(
                            publication["spec"]
                        )
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
                )
                # New specs leave parallelism to the Task Center and carry no resources.
                if "resources" in manifest["spec"]:
                    snapshot["resources"] = manifest["spec"]["resources"]
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
                        "predictorPolicy",
                    )
                    if name in manifest.get("spec", {})
                },
            }
        return result

    def _execution_status(
        self, identity, record, stage, batches, drafts, inputs, submission, predictor_execution
    ):
        """``(status, reason)``: planning statuses before submission, then one execution status.

        Task Center experiments take live states (running, queued, held, waiting) from
        the owner's tasks; batch, submission and coordinator evidence add what needs a
        person. Other experiments derive everything from that evidence.
        """
        retained = [batch for batch in batches if batch["state"] != "trashed"]
        payload = record.get("payload", {})
        if stage == "planning":
            if payload.get("frozenSetupId") and not submission:
                return "ready", None
            return aggregate_status(retained, inputs is not None or bool(drafts)), None
        parts = [batch["status"] for batch in retained]
        reasons = {}
        if submission and submission.get("status") != "submitted":
            # A completed batch cannot hide an unresolved submission receipt. Keep
            # acknowledgement recovery visible even if all runs finished while the
            # final launch response was being lost.
            attention = submission.get("status") == "attention"
            parts.append("attention" if attention else "launching")
            if attention:
                reasons["needs-attention"] = (submission.get("error") or {}).get("message")
        if predictor_execution:
            parts.append(predictor_execution["status"])
            if predictor_execution["status"] in {"attention", "interrupted"}:
                reasons.setdefault(
                    "needs-attention", (predictor_execution.get("error") or {}).get("message")
                )
            elif predictor_execution["status"] == "waiting":
                reasons["waiting"] = (
                    predictor_execution.get("waitingReason")
                    or "Waiting for the experiment's fold batches to finish."
                )
        # Tasks of trashed batches, like the batches themselves, no longer count.
        tasks = (
            self._task_execution(identity, current_batch_ids(batches, submission))
            if (submission or {}).get("executionMode") == "task-center"
            else None
        )
        if tasks is not None:
            if tasks[0] in LIVE_EXECUTION:
                return tasks
            # No live task: saved evidence cannot claim work is still moving.
            parts = [part for part in parts if _EXECUTION_STATUS.get(part) not in LIVE_EXECUTION]
            parts.append(tasks[0])
            if tasks[1]:
                reasons.setdefault(tasks[0], tasks[1])
        status = execution_status(parts)
        if status is None:
            return aggregate_status(retained, inputs is not None or bool(drafts)), None
        if status == "needs-attention" and not reasons.get(status):
            unreadable = next(
                (batch["executionError"] for batch in retained if batch.get("executionError")), None
            )
            reasons[status] = (unreadable or {}).get(
                "message", "A run stopped before finishing. Review it, then resume."
            )
        return status, reasons.get(status)

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
        frozen_setup = None
        if payload.get("frozenSetupId"):
            locked = True
            try:
                frozen_setup = self.store.get_configuration(
                    payload["frozenSetupId"], include_inactive=True
                )
            except StorageError:
                pass
        predictor_policy = (
            submission.get("predictorPolicy")
            if submission
            else None
            if legacy
            else payload.get("predictorPolicy", ExperimentPredictorPolicy().model_dump())
        )
        predictor_execution = None
        public_policies = None
        if submission and "predictorPolicies" in submission:
            try:
                public_policies = submission_policies(submission)
            except (StorageError, ValueError, KeyError, TypeError):
                # Retain invalid evidence in storage, but keep the read API's
                # map shape safe for list/count rendering alongside its error.
                pass
        try:
            expected_predictors = submission and predictor_work_expected(submission)
        except (StorageError, ValueError, KeyError, TypeError):
            # Invalid frozen intent must stay visible and locked. The status
            # boundary below presents the actionable error for this record.
            expected_predictors = True
        if expected_predictors:
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
            if stage == "finished" and not predictors_settled(submission, predictor_execution):
                stage = "running"
        status, status_reason = self._execution_status(
            identity, record, stage, batches, drafts, inputs, submission, predictor_execution
        )
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
            "status": status,
            # Why the experiment waits or needs attention, when one reason is known.
            "statusReason": status_reason,
            "stage": stage,
            "setupVersion": payload.get("setupVersion"),
            "setupDesign": deepcopy(payload.get("setupDesign")),
            "setupStatus": "frozen" if payload.get("frozenSetupId") else "draft",
            "frozenSetupId": payload.get("frozenSetupId"),
            "frozenSetup": frozen_setup,
            "configurationLocked": locked,
            "predictorPolicy": predictor_policy,
            "predictorPolicies": public_policies,
            "predictorExecution": predictor_execution,
            "batchPlans": presented_batch_plans(payload),
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
        """Read-only: parallel page reads never wait for, or block, project writers."""
        states = self.store.lifecycle.read()["records"]
        drafts = self.store.list_drafts(include_inactive=True)
        configurations = [
            *self.store.list_configurations("frozen-predictor", include_inactive=True),
            *self.store.list_configurations("mil-batch", include_inactive=True),
        ]
        records = [record for record in drafts if record["payload"].get("type") == EXPERIMENT_TYPE]
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
            owner = payload.get("experimentId") or _mapping(payload.get("spec")).get("experimentId")
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
