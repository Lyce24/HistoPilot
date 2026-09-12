"""Stable experiment records and immutable batch history over scientific storage.

The record's input choices are editable defaults. A frozen batch keeps its own
owner revision, input hashes, recipes and memberships. Older batches and saved
plans are projected as read-only records without rewriting their evidence.
"""

from pydantic import ValidationError

from histopilot.application.feature_bundles import _hash
from histopilot.schemas.mil import MILInputSpec
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


class ModelExperimentService:
    def __init__(self, store, filesystem, *, training=None):
        self.store, self.filesystem = store, filesystem
        # Training imports batch planning. Keep this dependency lazy.
        if training is None:
            from histopilot.application.training import TrainingService

            training = TrainingService(store, filesystem)
        self.training = training

    def require_editable(self, identity, expected_revision=None):
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
        return record

    def create(self, request):
        values = request.model_dump(exclude={"operationId"})
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
            if request.inputs:
                input_snapshot(self.store, request.inputs)
            record = self.store.create_draft(
                "experiment",
                request.name,
                {
                    "type": EXPERIMENT_TYPE,
                    "version": 1,
                    **{key: value for key, value in values.items() if key != "name"},
                    "creationOperationId": request.operationId,
                    "creationRequestHash": digest,
                },
            )
            return self.get(record["id"])

    def update(self, identity, request):
        with lifecycle_guard(self.store.folder):
            record = self.require_editable(identity, request.expectedRevision)
            if request.inputs:
                input_snapshot(self.store, request.inputs)
            # PATCH omissions retain saved inputs and metadata. Explicit null
            # still clears inputs, and explicit empty notes/tags clear those fields.
            values = {
                key: value
                for key, value in request.model_dump(exclude={"expectedRevision", "name"}).items()
                if key in request.model_fields_set
            }
            self.store.update_draft(
                identity,
                expected_revision=request.expectedRevision,
                name=request.name,
                payload={**record["payload"], **values},
            )
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
            "status": aggregate_status(batches, inputs is not None or bool(drafts)),
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
