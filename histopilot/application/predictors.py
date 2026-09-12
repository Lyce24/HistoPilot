"""Publish one immutable predictor per experiment, configuration, seeds and method.

The control process verifies bounded receipts and streams checkpoint hashes. It never
unpickles checkpoints, imports Torch, or rewrites training outputs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections import defaultdict
from pathlib import Path

from histopilot.application.development import development_plans
from histopilot.application.feature_bundles import _hash
from histopilot.application.training import membership_plan_id
from histopilot.schemas.predictors import PredictorSelection
from histopilot.storage.lifecycle import lifecycle_guard
from histopilot.storage.project_lock import StorageError, _reject_symlink_components
from histopilot.storage.scientific import ScientificStore


def reference(document):
    return {key: document[key] for key in ("id", "contentHash")}


def finding(code, message):
    return {"severity": "error", "code": code, "message": message}


def lifecycle_document(store, document):
    state = store.lifecycle.read()["records"].get(f"configuration:{document['id']}", {})
    return {**document, "lifecycleState": state.get("state", "active")}


def _file_path(path, root):
    path, root = Path(path), Path(root)
    try:
        path.absolute().relative_to(root.absolute())
        _reject_symlink_components(path)
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
        if not stat.S_ISREG(path.stat().st_mode):
            raise ValueError
    except (OSError, ValueError) as error:
        raise StorageError(
            "Training evidence must be a regular file inside its own run directory.",
            "PREDICTOR_EVIDENCE_INVALID",
            409,
        ) from error
    return path


def read_evidence(path, root):
    path = _file_path(path, root)
    try:
        value = json.loads(ScientificStore._read_file(path, 32 * 1024 * 1024))
        if not isinstance(value, dict):
            raise ValueError
        _hash(value)  # Reject NaN/Infinity in evidence as well as in scientific records.
        return value
    except (OSError, ValueError, TypeError) as error:
        raise StorageError(
            "Training evidence is unreadable or invalid.", "PREDICTOR_EVIDENCE_INVALID", 409
        ) from error


def checkpoint_snapshot(path, root):
    path = _file_path(path, root)
    if path.suffix != ".ckpt":
        raise StorageError("Select a training checkpoint.", "PREDICTOR_CHECKPOINT_INVALID", 409)
    try:
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
            before = os.fstat(stream.fileno())
            if before.st_size <= 0 or before.st_size > 8 * 1024**3:
                raise ValueError
            digest = hashlib.sha256()
            while block := stream.read(1024 * 1024):
                digest.update(block)
            after = os.fstat(stream.fileno())
            current = path.stat()

            def stamp(item):
                return (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)

            if stamp(before) != stamp(after) or stamp(after) != stamp(current):
                raise ValueError
    except (OSError, ValueError) as error:
        raise StorageError(
            "A selected checkpoint is missing, empty, too large, or changed while being read.",
            "PREDICTOR_CHECKPOINT_CHANGED",
            409,
        ) from error
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": after.st_size}


def feature_contract(document, bundle):
    manifest = document["manifest"]
    dimensions = {row["dimensions"] for row in manifest["files"]}
    dtypes = {row["dtype"] for row in manifest["files"]}
    return {
        "feature": reference(document),
        "bundle": reference(bundle),
        "dimensions": next(iter(dimensions)) if len(dimensions) == 1 else None,
        "dtype": next(iter(dtypes)) if len(dtypes) == 1 else None,
        "encoderId": manifest.get("layout", {}).get("encoderId")
        or manifest.get("spec", {}).get("encoderId"),
        "sourceContentHash": bundle["manifest"]["feature"].get("sourceContentHash"),
        "extraction": manifest.get("sourceExtraction"),
        "layout": manifest.get("layout", {}),
    }


# Display-only fields live inside the evidence manifest for provenance, but they
# must not decide whether reviewed evidence is still current: renaming an
# experiment does not change any checkpoint, membership or recipe.
def evidence_hash(manifest):
    experiment = manifest.get("experiment")
    if isinstance(experiment, dict) and "name" in experiment:
        manifest = {
            **manifest,
            "experiment": {key: value for key, value in experiment.items() if key != "name"},
        }
    return _hash({key: value for key, value in manifest.items() if key != "previewHash"})


def evidence_current(manifest, stored):
    """Compare live evidence with a saved manifest.

    Accepts the stable hash and, for records saved before v2, the hash that also
    covered the live experiment name and a name-dependent refit provenance hash.
    """
    stored_hash = stored.get("previewHash")
    if stored_hash is None:
        return False
    if stored_hash == evidence_hash(stored) == evidence_hash(manifest):
        return True
    # Validate the original saved evidence before comparing its scientific
    # content. Old plans include both the historical experiment name and the
    # provenance hash derived from that name; keep those values for this legacy
    # comparison only. Every other field must still match the live evidence.
    if stored_hash != _hash({key: value for key, value in stored.items() if key != "previewHash"}):
        return False
    legacy = {key: value for key, value in manifest.items() if key != "previewHash"}
    experiment, saved_experiment = legacy.get("experiment"), stored.get("experiment")
    if (
        isinstance(experiment, dict)
        and isinstance(saved_experiment, dict)
        and "name" in experiment
        and "name" in saved_experiment
    ):
        legacy["experiment"] = {**experiment, "name": saved_experiment["name"]}
    if "planTemplate" in legacy and "provenanceHash" in stored.get("planTemplate", {}):
        legacy["planTemplate"] = {
            **legacy["planTemplate"],
            "provenanceHash": stored["planTemplate"]["provenanceHash"],
        }
    return stored_hash == _hash(legacy)


class PredictorService:
    def __init__(self, store, filesystem):
        self.store, self.filesystem = store, filesystem

    def list(self, *, include_inactive=False):
        return {
            "items": [
                lifecycle_document(self.store, item)
                for item in self.store.list_configurations(
                    "frozen-predictor", include_inactive=include_inactive
                )
            ],
            "executionEnabled": True,
        }

    def get(self, identity):
        document = self.store.get_configuration(identity)
        if document["manifest"].get("kind") != "frozen-predictor":
            raise StorageError("Predictor not found.", "PREDICTOR_NOT_FOUND", 404)
        return lifecycle_document(self.store, document)

    @staticmethod
    def source_key(value, method=None):
        values = value.model_dump() if hasattr(value, "model_dump") else value
        return tuple(
            values.get(key)
            for key in ("experimentId", "batchId", "candidateId", "trainingSeed", "splitSeed")
        ) + (method or values.get("method", "ensemble"),)

    def _existing(self, selection, method=None):
        key = self.source_key(selection, method)
        return next(
            (
                item
                for item in self.store.list_configurations(
                    "frozen-predictor", include_inactive=True
                )
                if self.source_key(item["manifest"]) == key
            ),
            None,
        )

    def _experiment(self, experiment_id, batch):
        owner = batch["manifest"]["spec"].get("experimentId")
        if owner:
            if owner != experiment_id:
                raise StorageError(
                    "This batch belongs to another experiment.",
                    "PREDICTOR_EXPERIMENT_MISMATCH",
                    409,
                )
            experiment = self.store.get_draft(owner)
            if experiment["payload"].get("type") != "model-experiment":
                raise StorageError("Select a model experiment.", "INVALID_EXPERIMENT", 422)
            return {
                "id": owner,
                "revision": batch["manifest"]["spec"].get("experimentRevision"),
                "name": experiment["name"],
            }
        if experiment_id != f"legacy-{batch['id']}":
            raise StorageError(
                "This historical batch has a different experiment identity.",
                "PREDICTOR_EXPERIMENT_MISMATCH",
                409,
            )
        return {
            "id": experiment_id,
            "legacy": True,
            "name": batch["manifest"]["spec"].get("experimentName", "Historical experiment"),
        }

    @staticmethod
    def _groups(manifest):
        splits = {row["id"]: row for row in manifest["splitPlans"]}
        groups = defaultdict(list)
        for run in manifest["runs"]:
            split = splits[run["splitPlanId"]]
            groups[(run["candidateId"], run["trainingSeed"], split["seed"])].append(run)
        return groups

    def choices(self):
        items = []
        lifecycle = self.store.lifecycle.read()["records"]
        predictors = {
            self.source_key(item["manifest"]): item
            for item in self.store.list_configurations("frozen-predictor", include_inactive=True)
        }
        refits = {}
        for item in self.store.list_configurations("predictor-refit", include_inactive=True):
            key = self.source_key(item["manifest"])
            active = (
                lifecycle.get(f"configuration:{item['id']}", {}).get("state", "active") == "active"
            )
            if key not in refits or active:
                refits[key] = item["id"]
        for batch in self.store.list_configurations("mil-batch"):
            manifest = batch["manifest"]
            experiment_id = manifest["spec"].get("experimentId") or f"legacy-{batch['id']}"
            if lifecycle.get(f"draft:{experiment_id}", {}).get("state", "active") != "active":
                continue
            try:
                experiment = self._experiment(experiment_id, batch)
                folder = self.store.folder / "training" / batch["id"]
                state_path = folder / "state.json"
                state = read_evidence(state_path, folder) if state_path.exists() else {}
                states = {row["id"]: row for row in state.get("runs", [])}
                error = None
            except StorageError as problem:
                experiment, states, error = (
                    {"name": "Unavailable experiment"},
                    {},
                    str(problem),
                )
            candidates = {row["id"]: row for row in manifest["configurations"]}
            for (candidate, training_seed, split_seed), runs in self._groups(manifest).items():
                group = (experiment_id, batch["id"], candidate, training_seed, split_seed)
                existing = {
                    method: predictors[(*group, method)]["id"]
                    for method in ("ensemble", "refit")
                    if (*group, method) in predictors
                }
                completed = sum(
                    states.get(row["id"], {}).get("status") == "completed" for row in runs
                )
                reason = (
                    error
                    or (
                        "This configuration and seed group already has both predictor methods. Restore existing predictors to reuse their identities."
                        if len(existing) == 2
                        else None
                    )
                    or (
                        "Every fold in this candidate must finish before building a predictor."
                        if completed != len(runs)
                        else None
                    )
                )
                items.append(
                    {
                        "experimentId": experiment_id,
                        "experimentName": experiment["name"],
                        "batchId": batch["id"],
                        "batchName": manifest["spec"]["batchName"],
                        "candidateId": candidate,
                        "candidateNumber": candidates[candidate]["number"],
                        "trainingSeed": training_seed,
                        "splitSeed": split_seed,
                        "completedRuns": completed,
                        "totalRuns": len(runs),
                        "eligible": reason is None,
                        "reason": reason,
                        "existingPredictorId": existing.get("ensemble"),
                        "existingPredictorIds": existing,
                        "existingRefitId": refits.get((*group, "refit")),
                        "eligibleMethods": [
                            method
                            for method in ("ensemble", "refit")
                            if method not in existing and completed == len(runs) and not error
                        ],
                        "recipe": candidates[candidate]["recipe"],
                    }
                )
        return {"items": items, "executionEnabled": True}

    def _prepare(self, selection, *, allow_existing=False):
        try:
            return self._prepare_evidence(selection, allow_existing=allow_existing)
        except StorageError:
            raise
        except (KeyError, TypeError, ValueError, StopIteration) as error:
            raise StorageError(
                "Training receipts are incomplete or malformed. Repair the execution evidence before building a predictor.",
                "PREDICTOR_EVIDENCE_INVALID",
                409,
            ) from error

    def _prepare_evidence(self, selection, *, allow_existing=False):
        batch = self.store.get_configuration(selection.batchId)
        manifest = batch["manifest"]
        if manifest.get("kind") != "mil-batch":
            raise StorageError("Select a development batch.", "INVALID_BATCH", 422)
        experiment = self._experiment(selection.experimentId, batch)
        existing = self._existing(selection)
        if existing and not allow_existing:
            raise StorageError(
                "This configuration and seed group already has a predictor for this build method. Restore it to reuse its identity, or select another group.",
                "EXPERIMENT_ALREADY_FROZEN",
                409,
            )
        selected = self._groups(manifest).get(
            (selection.candidateId, selection.trainingSeed, selection.splitSeed), []
        )
        if not selected:
            raise StorageError(
                "Select a candidate and its frozen seeds.", "PREDICTOR_CANDIDATE_MISSING", 422
            )
        split_ids = {
            row["id"] for row in manifest["splitPlans"] if row["seed"] == selection.splitSeed
        }
        if len(selected) != len(split_ids) or {row["splitPlanId"] for row in selected} != split_ids:
            raise StorageError(
                "The candidate does not contain every frozen fold.",
                "PREDICTOR_FOLDS_INCOMPLETE",
                409,
            )
        folder = self.store.folder / "training" / batch["id"]
        plan = read_evidence(folder / "plan.json", folder)
        state = read_evidence(folder / "state.json", folder)
        inputs = manifest["spec"]["inputs"]
        protocol = self.store.get_configuration(inputs["protocolId"])
        bundle = self.store.get_configuration(inputs["featureBundleId"])
        feature = self.store.get_configuration(bundle["manifest"]["feature"]["id"])
        candidate = next(
            row for row in manifest["configurations"] if row["id"] == selection.candidateId
        )
        target = protocol["manifest"]["spec"]["target"]
        memberships = defaultdict(list)
        for row in protocol["manifest"]["memberships"]:
            memberships[membership_plan_id(row)].append(row)
        required = {row["slideId"] for row in protocol["manifest"]["memberships"]}
        files = {
            row["slideId"]: row
            for row in feature["manifest"]["files"]
            if row["slideId"] in required
        }
        if (
            _hash(plan) != state.get("planHash")
            or state.get("batchId") != batch["id"]
            or plan.get("batchId") != batch["id"]
            or plan.get("batchContentHash") != batch["contentHash"]
            or plan.get("protocolContentHash") != protocol["contentHash"]
            or plan.get("featureBundleContentHash") != bundle["contentHash"]
            or plan.get("target") != target
            or plan.get("configurations") != manifest["configurations"]
            or plan.get("splitPlans") != manifest["splitPlans"]
            or plan.get("runs") != manifest["runs"]
            or plan.get("memberships") != dict(memberships)
            or development_plans(protocol["manifest"]) != manifest["splitPlans"]
            or plan.get("data", {}).get("featureFiles") != files
            or plan.get("data", {}).get("sourceStamps")
            != {row["path"]: row for row in files.values()}
        ):
            raise StorageError(
                "Training provenance differs from the frozen batch.",
                "PREDICTOR_PROVENANCE_CHANGED",
                409,
            )
        if (
            protocol["manifest"]["spec"]["split"].get("mode") != "kfold"
            or protocol["manifest"]["spec"]["split"].get("version") != 4
            or candidate["recipe"]["model"].lower() != "abmil"
        ):
            raise StorageError(
                "Predictor promotion currently supports k-fold ABMIL.",
                "PREDICTOR_MODEL_UNSUPPORTED",
                422,
            )
        states = {row["id"]: row for row in state.get("runs", [])}
        checkpoints = []
        for run in sorted(selected, key=lambda item: item["id"]):
            status = states.get(run["id"], {})
            from histopilot.application.lifecycle import _confirmed_live

            try:
                live = _confirmed_live(status.get("process"))
            except StorageError as error:
                raise StorageError(
                    "Cannot confirm whether a selected training process has stopped.",
                    "PREDICTOR_PROCESS_UNKNOWN",
                    409,
                ) from error
            if status.get("status") != "completed" or live:
                raise StorageError(
                    "Every selected fold must have finished and stopped writing.",
                    "PREDICTOR_RUN_INCOMPLETE",
                    409,
                )
            run_folder = folder / "runs" / run["id"]
            receipt = read_evidence(run_folder / "result.json", run_folder)
            run_plan = read_evidence(run_folder / "plan.json", run_folder)
            if (
                receipt.get("state") != "succeeded"
                or receipt.get("runId") != run["id"]
                or receipt != status.get("result")
                or run_plan.get("runId") != run["id"]
                or run_plan.get("batchId") != batch["id"]
                or run_plan.get("batchContentHash") != batch["contentHash"]
                or run_plan.get("candidateId") != selection.candidateId
                or run_plan.get("trainingSeed") != selection.trainingSeed
                or run_plan.get("recipe") != candidate["recipe"]
                or run_plan.get("target") != target
                or run_plan.get("splitPlan")
                != next(row for row in plan["splitPlans"] if row["id"] == run["splitPlanId"])
                or run_plan.get("data")
                != {**plan["data"], "memberships": plan["memberships"][run["splitPlanId"]]}
                or run_plan.get("runtime") != plan.get("runtime")
                or run_plan.get("code") != plan.get("code")
                or receipt.get("checkpointMetric") != candidate["recipe"]["checkpointMetric"]
                or not isinstance(receipt.get("bestValidationScore"), (int, float))
                or not math.isfinite(receipt["bestValidationScore"])
            ):
                raise StorageError(
                    "A completed fold receipt differs from its frozen training plan.",
                    "PREDICTOR_PROVENANCE_CHANGED",
                    409,
                )
            snapshot = checkpoint_snapshot(receipt["bestCheckpointPath"], run_folder)
            checkpoints.append(
                {
                    "runId": run["id"],
                    "splitPlanId": run["splitPlanId"],
                    **snapshot,
                    "receiptHash": _hash(receipt),
                    "runPlanHash": _hash(run_plan),
                    "bestValidationScore": receipt["bestValidationScore"],
                    "epochsCompleted": receipt.get("epochsCompleted"),
                }
            )
        contract = feature_contract(feature, bundle)
        if (
            not contract["dimensions"]
            or not contract["dtype"]
            or not contract["sourceContentHash"]
            or plan["data"]["featureDim"] != contract["dimensions"]
        ):
            raise StorageError(
                "Feature representation provenance is incomplete.",
                "PREDICTOR_FEATURES_INVALID",
                409,
            )
        result = {
            "kind": "frozen-predictor",
            "schemaVersion": 2,
            "method": selection.method,
            "datasetId": manifest["datasetId"],
            "name": selection.name,
            "selection": selection.model_dump(),
            "experimentId": selection.experimentId,
            "experiment": experiment,
            "batchId": batch["id"],
            "batch": reference(batch),
            "candidateId": selection.candidateId,
            "trainingSeed": selection.trainingSeed,
            "splitSeed": selection.splitSeed,
            "runIds": [row["runId"] for row in checkpoints],
            "checkpoints": checkpoints,
            "target": target,
            "recipe": candidate["recipe"],
            "inputs": {
                "protocol": reference(protocol),
                "features": contract,
                "loading": inputs,
                "resolvedLoading": {
                    "policy": plan["data"].get("loadingPolicy"),
                    "packArtifactId": manifest.get("resolvedInputs", {}).get("packArtifactId"),
                    "packPath": plan["data"].get("packPath"),
                    "packStamps": plan["data"].get("packStamps"),
                },
            },
            "trainingPlanHash": _hash(plan),
            "compute": plan.get("code"),
            "runtime": plan.get("runtime"),
            "aggregation": "mean_probability",
            "patientAggregation": "mean",
            "executionEnabled": True,
        }

        if selection.method == "refit":
            from histopilot.application.refits import prepare_refit

            result = prepare_refit(result, plan, folder)
        return result

    def preview(self, selection):
        try:
            manifest = self._prepare(selection)
            return {
                "canFreeze": True,
                "previewHash": evidence_hash(manifest),
                "manifest": manifest,
                "findings": [],
            }
        except StorageError as error:
            return {
                "canFreeze": False,
                "previewHash": None,
                "manifest": None,
                "findings": [finding(error.code, str(error))],
            }

    def freeze(self, request):
        if request.method != "ensemble":
            raise StorageError(
                "Create and run a refit plan, then publish its completed model.",
                "PREDICTOR_REFIT_REQUIRED",
                422,
            )
        selection = PredictorSelection.model_validate(
            request.model_dump(include=set(PredictorSelection.model_fields))
        )
        with lifecycle_guard(self.store.folder):
            prior = self.store.configuration_publication(request.operationId)
            if prior:
                manifest = prior["manifest"]
                if (
                    manifest.get("kind") != "frozen-predictor"
                    or PredictorSelection.model_validate(manifest.get("selection", {})) != selection
                    or manifest.get("previewHash") != request.previewHash
                ):
                    raise StorageError(
                        "This operation belongs to another predictor.", "OPERATION_CONFLICT", 409
                    )
                return lifecycle_document(self.store, prior)
            manifest = self._prepare(selection)
            if evidence_hash(manifest) != request.previewHash:
                raise StorageError(
                    "Checkpoint or training evidence changed. Review again.", "PREVIEW_STALE", 409
                )
            published = self.store.publish_configuration(
                manifest={**manifest, "previewHash": request.previewHash},
                operation_id=request.operationId,
            )
            return lifecycle_document(self.store, published)

    def verify_checkpoints(self, predictor):
        manifest = predictor["manifest"]
        for expected in manifest["checkpoints"]:
            if manifest.get("method", "ensemble") == "refit":
                root = self.store.folder / "compute-jobs" / manifest["refitId"]
            else:
                root = (
                    self.store.folder
                    / "training"
                    / manifest["batchId"]
                    / "runs"
                    / expected["runId"]
                )
            current = checkpoint_snapshot(expected["path"], root)
            if any(current[key] != expected[key] for key in ("path", "bytes", "sha256")):
                raise StorageError(
                    "A frozen predictor checkpoint changed or is missing.",
                    "PREDICTOR_CHECKPOINT_CHANGED",
                    409,
                )
