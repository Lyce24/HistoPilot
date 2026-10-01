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
from histopilot.application.development_splits import training_split_issue
from histopilot.application.experiment_policy import has_predictor_intent, policy_for_batch
from histopilot.application.training import membership_plan_id
from histopilot.domain.features import representation_kind
from histopilot.models import catalog
from histopilot.schemas.predictors import PredictorSelection, SeedEnsembleSelection
from histopilot.storage.io import content_hash, read_file_bounded
from histopilot.storage.lifecycle import lifecycle_guard
from histopilot.storage.project_lock import StorageError, reject_symlink_components

SEED_ENSEMBLE = "seed_ensemble"
# Each member adds about a kilobyte to the 16 MiB manifest and one model pass to every
# evaluation and attention job.
MAX_SEED_ENSEMBLE_MEMBERS = 256
SINGLE_GROUP = (
    "A seed ensemble pools two or more seed groups; this configuration has one. "
    "Use its fold ensemble instead."
)
TOO_MANY_MEMBERS = f"A seed ensemble holds at most {MAX_SEED_ENSEMBLE_MEMBERS} fold models."
# Fields that every seed group of one configuration shares, being read from one batch plan.
SHARED_GROUP_KEYS = (
    "datasetId",
    "experiment",
    "batch",
    "candidateId",
    "target",
    "recipe",
    "inputs",
    "trainingPlanHash",
    "compute",
    "runtime",
    "aggregation",
    "patientAggregation",
)


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
        reject_symlink_components(path)
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
        value = json.loads(read_file_bounded(path, 32 * 1024 * 1024))
        if not isinstance(value, dict):
            raise ValueError
        content_hash(value)  # Reject NaN/Infinity in evidence as well as in scientific records.
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
        **({"featureKind": "slide"} if representation_kind(manifest) == "slide" else {}),
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
    return content_hash({key: value for key, value in manifest.items() if key != "previewHash"})


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
    if stored_hash != content_hash(
        {key: value for key, value in stored.items() if key != "previewHash"}
    ):
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
    return stored_hash == content_hash(legacy)


class PredictorService:
    def __init__(self, store, filesystem):
        self.store, self.filesystem = store, filesystem

    def require_work_open(self, experiment_id):
        """Older manual commands cannot reopen closed experiment predictor work.

        This guard belongs to mutations/creation previews, not evidence reads:
        existing predictors must remain verifiable and usable for evaluation.
        Publication retries return their accepted receipt before this check.
        """
        if experiment_id.startswith("legacy-"):
            return
        submission = self.store.get_draft(experiment_id)["payload"].get("submission") or {}
        if not has_predictor_intent(submission):
            return
        from histopilot.application.experiment_predictors import ExperimentPredictorService

        coordinator = ExperimentPredictorService(self.store, self.filesystem)
        folder = coordinator.folder(experiment_id)
        closed = (folder / "cancel.requested").exists()
        if (folder / "state.json").exists():
            _plan, state = coordinator._read(experiment_id)
            closed = closed or state["status"] in {"completed", "cancelled", "cancelling"}
        if closed:
            raise StorageError(
                "This experiment's predictor work is closed. Copy the experiment to create or retrain predictors; its existing predictors remain available for evaluation.",
                "EXPERIMENT_PREDICTORS_LOCKED",
                409,
            )

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
            selected_candidate = None
            select_best = manifest["spec"].get("candidateSelection") == "best_validation"
            try:
                experiment = self._experiment(experiment_id, batch)
                folder = self.store.folder / "training" / batch["id"]
                state_path = folder / "state.json"
                state = read_evidence(state_path, folder) if state_path.exists() else {}
                states = {row["id"]: row for row in state.get("runs", [])}
                if select_best and (folder / "plan.json").exists():
                    from histopilot.candidate_selection import validation_selection

                    selection_report = validation_selection(
                        read_evidence(folder / "plan.json", folder), state
                    )
                    selected_candidate = (selection_report or {}).get("selectedCandidateId")
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
                selection_reason = (
                    "Waiting for validation scores from every configuration."
                    if select_best and selected_candidate is None
                    else "Another configuration was selected by validation performance."
                    if select_best and selected_candidate != candidate
                    else None
                )
                incomplete_reason = (
                    "Every fold in this candidate must finish before building a predictor."
                    if completed != len(runs)
                    else None
                )
                reason = (
                    error
                    or selection_reason
                    or (
                        "This configuration and seed group already has both predictor methods. Restore existing predictors to reuse their identities."
                        if len(existing) == 2
                        else None
                    )
                    or incomplete_reason
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
                        "complete": completed == len(runs),
                        # What stops any predictor from this group, whichever methods exist.
                        "blocker": error or selection_reason or incomplete_reason,
                        "eligible": reason is None,
                        "reason": reason,
                        "existingPredictorId": existing.get("ensemble"),
                        "existingPredictorIds": existing,
                        "existingRefitId": refits.get((*group, "refit")),
                        "eligibleMethods": [
                            method
                            for method in ("ensemble", "refit")
                            if method not in existing and reason is None
                        ],
                        "recipe": candidates[candidate]["recipe"],
                    }
                )
        return {"items": items, "executionEnabled": True}

    def seed_ensemble_choices(self, experiment_id=None):
        """Configurations whose seed groups can pool into one seed-ensemble predictor."""
        grouped = defaultdict(list)
        for item in self.choices()["items"]:
            if experiment_id is None or item["experimentId"] == experiment_id:
                grouped[(item["experimentId"], item["batchId"], item["candidateId"])].append(item)
        existing = {
            self.source_key(row["manifest"]): row["id"]
            for row in self.store.list_configurations("frozen-predictor", include_inactive=True)
            if row["manifest"].get("method") == SEED_ENSEMBLE
        }
        items = []
        for (experiment, batch, candidate), rows in grouped.items():
            rows.sort(key=lambda row: (row["trainingSeed"], row["splitSeed"]))
            members = sum(row["totalRuns"] for row in rows)
            identity = existing.get((experiment, batch, candidate, None, None, SEED_ENSEMBLE))
            reason = (
                next((row["blocker"] for row in rows if row["blocker"]), None)
                or (SINGLE_GROUP if len(rows) < 2 else None)
                or (TOO_MANY_MEMBERS if members > MAX_SEED_ENSEMBLE_MEMBERS else None)
                or ("This configuration already has a seed ensemble." if identity else None)
            )
            first = rows[0]
            items.append(
                {
                    "experimentId": experiment,
                    "experimentName": first["experimentName"],
                    "batchId": batch,
                    "batchName": first["batchName"],
                    "candidateId": candidate,
                    "candidateNumber": first["candidateNumber"],
                    "trainingSeeds": sorted({row["trainingSeed"] for row in rows}),
                    "splitSeeds": sorted({row["splitSeed"] for row in rows}),
                    "seedGroups": len(rows),
                    "members": members,
                    "completedRuns": sum(row["completedRuns"] for row in rows),
                    "eligible": reason is None,
                    "reason": reason,
                    "existingPredictorId": identity,
                }
            )
        return {"items": items}

    def _prepare_seed_ensemble(self, selection, *, allow_existing=False):
        """Every seed group's verified fold checkpoints of one configuration, as one predictor.

        Each group is verified exactly as its own fold ensemble would be. Members average
        with equal weight under the recipe's ensemble rule; with equal fold counts, that
        equals averaging the per-seed fold ensembles, and it is the deployable form of the
        seed ensemble that cross-validated results report.
        """
        batch = self.store.get_configuration(selection.batchId)
        if batch["manifest"].get("kind") != "mil-batch":
            raise StorageError("Select a development batch.", "INVALID_BATCH", 422)
        seeds = sorted(
            (training_seed, split_seed)
            for candidate, training_seed, split_seed in self._groups(batch["manifest"])
            if candidate == selection.candidateId
        )
        if not seeds:
            raise StorageError(
                "Select a candidate and its frozen seeds.", "PREDICTOR_CANDIDATE_MISSING", 422
            )
        if len(seeds) < 2:
            raise StorageError(SINGLE_GROUP, "SEED_ENSEMBLE_SINGLE_GROUP", 422)
        if self._existing(selection, SEED_ENSEMBLE) and not allow_existing:
            raise StorageError(
                "This configuration already has a seed ensemble. Restore it to reuse its identity.",
                "EXPERIMENT_ALREADY_FROZEN",
                409,
            )
        groups = [
            self._prepare(
                PredictorSelection(
                    experimentId=selection.experimentId,
                    batchId=selection.batchId,
                    candidateId=selection.candidateId,
                    trainingSeed=training_seed,
                    splitSeed=split_seed,
                    name=selection.name,
                ),
                allow_existing=True,
                derived=True,
            )
            for training_seed, split_seed in seeds
        ]
        first = groups[0]
        if any(group.get(key) != first.get(key) for group in groups for key in SHARED_GROUP_KEYS):
            raise StorageError(
                "Seed groups of one configuration differ in their frozen training provenance.",
                "PREDICTOR_PROVENANCE_CHANGED",
                409,
            )
        checkpoints = [
            {**checkpoint, "trainingSeed": group["trainingSeed"], "splitSeed": group["splitSeed"]}
            for group in groups
            for checkpoint in group["checkpoints"]
        ]
        if len(checkpoints) > MAX_SEED_ENSEMBLE_MEMBERS:
            raise StorageError(TOO_MANY_MEMBERS, "SEED_ENSEMBLE_TOO_LARGE", 422)
        return {
            "kind": "frozen-predictor",
            "schemaVersion": 2,
            "method": SEED_ENSEMBLE,
            "datasetId": first["datasetId"],
            "name": selection.name,
            "selection": {**selection.model_dump(), "method": SEED_ENSEMBLE},
            "experimentId": selection.experimentId,
            "experiment": first["experiment"],
            "batchId": first["batchId"],
            "batch": first["batch"],
            "candidateId": selection.candidateId,
            **({"candidateNumber": first["candidateNumber"]} if "candidateNumber" in first else {}),
            "trainingSeeds": sorted({training_seed for training_seed, _ in seeds}),
            "splitSeeds": sorted({split_seed for _, split_seed in seeds}),
            "seedGroups": [
                {
                    "trainingSeed": group["trainingSeed"],
                    "splitSeed": group["splitSeed"],
                    "runIds": group["runIds"],
                }
                for group in groups
            ],
            "runIds": [checkpoint["runId"] for checkpoint in checkpoints],
            "checkpoints": checkpoints,
            "target": first["target"],
            "recipe": first["recipe"],
            "inputs": first["inputs"],
            "trainingPlanHash": first["trainingPlanHash"],
            **(
                {"selectionEvidence": first["selectionEvidence"]}
                if "selectionEvidence" in first
                else {}
            ),
            "compute": first["compute"],
            "runtime": first["runtime"],
            "aggregation": first["aggregation"],
            "patientAggregation": first["patientAggregation"],
            "executionEnabled": True,
        }

    def preview_seed_ensemble(self, selection):
        try:
            manifest = self._prepare_seed_ensemble(selection)
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

    def freeze_seed_ensemble(self, request):
        selection = SeedEnsembleSelection.model_validate(
            request.model_dump(include=set(SeedEnsembleSelection.model_fields))
        )
        with lifecycle_guard(self.store.folder):
            prior = self.store.configuration_publication(request.operationId)
            if prior:
                manifest = prior["manifest"]
                expected = {**selection.model_dump(), "method": SEED_ENSEMBLE}
                if (
                    manifest.get("kind") != "frozen-predictor"
                    or manifest.get("selection") != expected
                    or manifest.get("previewHash") != request.previewHash
                ):
                    raise StorageError(
                        "This operation belongs to another predictor.", "OPERATION_CONFLICT", 409
                    )
                return lifecycle_document(self.store, prior)
            manifest = self._prepare_seed_ensemble(selection)
            if evidence_hash(manifest) != request.previewHash:
                raise StorageError(
                    "Checkpoint or training evidence changed. Review again.", "PREVIEW_STALE", 409
                )
            published = self.store.publish_configuration(
                manifest={**manifest, "previewHash": request.previewHash},
                operation_id=request.operationId,
            )
            return lifecycle_document(self.store, published)

    def _prepare(self, selection, *, allow_existing=False, derived=False):
        try:
            return self._prepare_evidence(selection, allow_existing=allow_existing, derived=derived)
        except StorageError:
            raise
        except (KeyError, TypeError, ValueError, StopIteration) as error:
            raise StorageError(
                "Training receipts are incomplete or malformed. Repair the execution evidence before building a predictor.",
                "PREDICTOR_EVIDENCE_INVALID",
                409,
            ) from error

    def _prepare_evidence(self, selection, *, allow_existing=False, derived=False):
        """One seed group's verified fold evidence as a predictor manifest.

        ``derived`` verifies the same evidence for a predictor composed from finished
        groups (a seed ensemble). It trains nothing and changes no submitted artifact, so
        the submitted per-seed method policy does not govern it.
        """
        batch = self.store.get_configuration(selection.batchId)
        manifest = batch["manifest"]
        if manifest.get("kind") != "mil-batch":
            raise StorageError("Select a development batch.", "INVALID_BATCH", 422)
        experiment = self._experiment(selection.experimentId, batch)
        policy = None
        if not experiment.get("legacy"):
            submission = self.store.get_draft(selection.experimentId)["payload"].get("submission")
            policy = policy_for_batch(submission or {}, selection.batchId, spec=manifest["spec"])
            if (
                not derived
                and policy
                and (
                    selection.batchId not in submission["batchIds"]
                    or policy["method"] not in {selection.method, "both"}
                    or selection.method == "refit"
                    and policy["refitPercentile"] != selection.refitPercentile
                )
            ):
                raise StorageError(
                    "Predictor creation must preserve this experiment's frozen method and refit epoch policy. Copy the experiment to change them.",
                    "EXPERIMENT_PREDICTOR_POLICY_LOCKED",
                    409,
                )
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
        from histopilot.candidate_selection import validation_selection

        for key in ("selectionMetric", "candidateSelection"):
            if plan.get(key) != manifest["spec"].get(key):
                raise StorageError(
                    "Configuration selection differs from the frozen batch.",
                    "PREDICTOR_SELECTION_CHANGED",
                    409,
                )
        selection_evidence = validation_selection(plan, state)
        if selection_evidence and selection_evidence["ready"]:
            # Every competing configuration influences promotion. Verify all
            # selection receipts, not only the winning model's checkpoints.
            selection_states = {row["id"]: row for row in state["runs"]}
            for selection_run in plan["runs"]:
                evidence_folder = folder / "runs" / selection_run["id"]
                receipt = read_evidence(evidence_folder / "result.json", evidence_folder)
                if (
                    receipt != selection_states[selection_run["id"]].get("result")
                    or receipt.get("runId") != selection_run["id"]
                    or receipt.get("state") != "succeeded"
                ):
                    raise StorageError(
                        "A configuration-selection receipt changed.",
                        "PREDICTOR_SELECTION_CHANGED",
                        409,
                    )
        if plan.get("candidateSelection") == "best_validation" and (
            not selection_evidence
            or not selection_evidence["ready"]
            or selection_evidence["selectedCandidateId"] != selection.candidateId
        ):
            raise StorageError(
                "This batch promotes the configuration selected by its frozen validation metric after all configurations finish.",
                "PREDICTOR_VALIDATION_SELECTION_REQUIRED",
                409,
            )
        memberships = defaultdict(list)
        for row in protocol["manifest"]["memberships"]:
            memberships[membership_plan_id(row)].append(row)
        required = {row["slideId"] for row in protocol["manifest"]["memberships"]}
        from histopilot.application.clinical_inputs import development_clinical_values

        expected_clinical = development_clinical_values(
            self.store,
            self.filesystem,
            protocol["manifest"],
            [item["recipe"] for item in manifest["configurations"]],
        )
        files = {
            row["slideId"]: row
            for row in feature["manifest"]["files"]
            if row["slideId"] in required
        }
        if (
            content_hash(plan) != state.get("planHash")
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
            or plan.get("data", {}).get("clinicalValues", {}) != expected_clinical
            or plan.get("data", {}).get("sourceStamps")
            != {row["path"]: row for row in files.values()}
        ):
            raise StorageError(
                "Training provenance differs from the frozen batch.",
                "PREDICTOR_PROVENANCE_CHANGED",
                409,
            )
        if (
            training_split_issue(protocol["manifest"]["spec"]["split"])
            or protocol["manifest"]["spec"]["split"].get("version") != 4
            or not catalog.is_supported(candidate["recipe"]["model"])
        ):
            raise StorageError(
                "Predictors are built from models of a trainable development design, "
                f"from: {catalog.choices()}.",
                "PREDICTOR_MODEL_UNSUPPORTED",
                422,
            )
        feature_kind = representation_kind(feature["manifest"])
        if (
            candidate["recipe"].get("inputMode", "image") != "clinical"
            and catalog.feature_kind(candidate["recipe"].get("model")) != feature_kind
        ):
            raise StorageError(
                f"The trained model requires {catalog.feature_kind(candidate['recipe'].get('model'))} "
                f"features, but its frozen bundle holds {feature_kind} features. "
                "Train a compatible model in a new batch.",
                "PREDICTOR_FEATURE_KIND_MISMATCH",
                422,
            )
        states = {row["id"]: row for row in state.get("runs", [])}
        checkpoints = []
        for run in sorted(selected, key=lambda item: item["id"]):
            status = states.get(run["id"], {})
            from histopilot.application.lifecycle import confirmed_live

            try:
                live = confirmed_live(status.get("process"))
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
            from histopilot.schemas.training_controls import resolve_stopping, sampling_memberships

            expected_memberships = sampling_memberships(
                plan["memberships"][run["splitPlanId"]],
                candidate["recipe"],
                plan["data"].get("cohortValues", {}),
            )
            from histopilot.schemas.nnmil import resolve_nnmil_recipe

            try:
                effective_recipe, nnmil_planning = resolve_nnmil_recipe(
                    candidate["recipe"], expected_memberships, files
                )
            except ValueError as error:
                raise StorageError(str(error), "PREDICTOR_PROVENANCE_CHANGED", 409) from error
            resolved_recipe, stopping_decision = resolve_stopping(
                effective_recipe,
                target,
                expected_memberships,
                split_unit=run_plan.get("splitUnit"),
            )
            if nnmil_planning and (
                run_plan.get("effectiveRecipe") != effective_recipe
                or run_plan.get("nnmilPlanning") != nnmil_planning
                or receipt.get("effectiveRecipe") != effective_recipe
                or receipt.get("nnmilPlanning") != nnmil_planning
            ):
                raise StorageError(
                    "Resolved MIL settings differ from the fitting evidence.",
                    "PREDICTOR_PROVENANCE_CHANGED",
                    409,
                )
            is_nnmil = candidate["recipe"].get("model", "abmil").lower() == "nnmil"
            if is_nnmil:
                checkpoint_policy = (
                    "final_epoch"
                    if stopping_decision
                    else candidate["recipe"].get("nnmilCheckpointSelection", "best_validation")
                )
                epoch, completed = receipt.get("selectedEpoch"), receipt.get("epochsCompleted")
                expected_epoch = (
                    completed
                    if checkpoint_policy != "best_validation"
                    else receipt.get("bestEpoch")
                )
                if (
                    receipt.get("checkpointSelection") != checkpoint_policy
                    or type(epoch) is not int
                    or type(completed) is not int
                    or not 1 <= epoch <= completed <= resolved_recipe["maxEpochs"]
                    or epoch != expected_epoch
                    or (
                        checkpoint_policy != "best_validation"
                        and receipt.get("bestCheckpointPath") != receipt.get("lastCheckpointPath")
                    )
                ):
                    raise StorageError(
                        "nnMIL checkpoint selection differs from its frozen policy.",
                        "PREDICTOR_PROVENANCE_CHANGED",
                        409,
                    )
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
                or run_plan.get("data") != {**plan["data"], "memberships": expected_memberships}
                or run_plan.get("runtime") != plan.get("runtime")
                or run_plan.get("code") != plan.get("code")
                or receipt.get("checkpointMetric") != resolved_recipe["checkpointMetric"]
                or receipt.get("stoppingDecision") != stopping_decision
                or (
                    stopping_decision
                    and (
                        receipt.get("selectedEpoch" if is_nnmil else "bestEpoch")
                        != stopping_decision["epochs"]
                        or receipt.get("epochsCompleted") != stopping_decision["epochs"]
                        or receipt.get("bestCheckpointPath") != receipt.get("lastCheckpointPath")
                    )
                )
                or not isinstance(receipt.get("bestValidationScore"), (int, float))
                or not math.isfinite(receipt["bestValidationScore"])
            ):
                raise StorageError(
                    "A completed fold receipt differs from its frozen training plan.",
                    "PREDICTOR_PROVENANCE_CHANGED",
                    409,
                )
            from histopilot.clinical_features import clinical_fields, fit_clinical_preprocessor

            fields = clinical_fields(candidate["recipe"])
            clinical_preprocessing = None
            if fields:
                clinical_preprocessing = fit_clinical_preprocessor(
                    [row for row in expected_memberships if row["partition"] == "train"],
                    expected_clinical,
                    fields,
                    unit=run_plan.get("splitUnit", "patient"),
                )
                if receipt.get("clinicalPreprocessing") != clinical_preprocessing:
                    raise StorageError(
                        "Clinical preprocessing differs from its training patients.",
                        "PREDICTOR_CLINICAL_PROVENANCE_CHANGED",
                        409,
                    )
            snapshot = checkpoint_snapshot(receipt["bestCheckpointPath"], run_folder)
            checkpoints.append(
                {
                    "runId": run["id"],
                    "splitPlanId": run["splitPlanId"],
                    **snapshot,
                    "receiptHash": content_hash(receipt),
                    "runPlanHash": content_hash(run_plan),
                    "bestValidationScore": receipt["bestValidationScore"],
                    "epochsCompleted": receipt.get("epochsCompleted"),
                    **(
                        {"clinicalPreprocessing": clinical_preprocessing}
                        if clinical_preprocessing
                        else {}
                    ),
                    **(
                        {"effectiveRecipe": effective_recipe, "nnmilPlanning": nnmil_planning}
                        if nnmil_planning
                        else {}
                    ),
                    **(
                        {
                            "selectedEpoch": receipt["selectedEpoch"],
                            "checkpointSelection": receipt["checkpointSelection"],
                            "bestEpoch": receipt.get("bestEpoch"),
                        }
                        if is_nnmil
                        else {}
                    ),
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
            **({"candidateNumber": candidate["number"]} if policy else {}),
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
            "trainingPlanHash": content_hash(plan),
            **({"selectionEvidence": selection_evidence} if selection_evidence else {}),
            "compute": plan.get("code"),
            "runtime": plan.get("runtime"),
            "aggregation": candidate["recipe"].get("ensembleAggregation", "mean_probability"),
            "patientAggregation": "mean_logits"
            if candidate["recipe"].get("patientAggregation") == "mean_logits"
            else "mean",
            "executionEnabled": True,
        }

        if selection.method == "refit":
            from histopilot.application.refits import prepare_refit

            result = prepare_refit(result, plan, folder)
        return result

    def preview(self, selection):
        try:
            self.require_work_open(selection.experimentId)
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
            self.require_work_open(selection.experimentId)
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
