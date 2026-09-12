"""Reviewed full-development refits, derived exclusively from completed CV evidence."""

from __future__ import annotations

import json
import math
import struct
from copy import deepcopy

from histopilot.application.feature_bundles import _hash
from histopilot.application.predictors import (
    PredictorService,
    checkpoint_snapshot,
    evidence_current,
    evidence_hash,
    lifecycle_document,
    read_evidence,
    reference,
)
from histopilot.schemas.development import ResourcePolicy
from histopilot.schemas.predictors import PredictorSelection
from histopilot.storage.lifecycle import lifecycle_guard
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import ScientificStore


def _best_epoch(checkpoint, folder, recipe):
    """Read a one-based checkpoint epoch; legacy histories are resolved without pickle."""
    run_folder = folder / "runs" / checkpoint["runId"]
    receipt = read_evidence(run_folder / "result.json", run_folder)
    epoch = receipt.get("bestEpoch")
    if epoch is not None:
        if type(epoch) is not int or not 1 <= epoch <= recipe["maxEpochs"]:
            raise StorageError("The best checkpoint epoch is invalid.", "REFIT_EPOCH_INVALID", 409)
        if type(receipt.get("epochsCompleted")) is not int or epoch > receipt["epochsCompleted"]:
            raise StorageError(
                "The best epoch exceeds completed training.", "REFIT_EPOCH_INVALID", 409
            )
        return {"runId": checkpoint["runId"], "bestEpoch": epoch, "source": "checkpoint_receipt"}
    path = run_folder / "history.json"
    # Use the same bounded, regular, in-run file policy as all checkpoint evidence.
    from histopilot.application.predictors import _file_path

    try:
        history = json.loads(ScientificStore._read_file(_file_path(path, run_folder), 32 * 1024**2))
        if not isinstance(history, list) or not history:
            raise ValueError
        metric = recipe["checkpointMetric"].removeprefix("validation_")
        scored = []
        for index, row in enumerate(history):
            value = row["validation"][metric]
            if (
                type(row["epoch"]) is not int
                or row["epoch"] != index
                or not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                raise ValueError
            # Lightning logs Python scalar metrics as float32 tensors. Preserve
            # its first strict optimum even when two double-precision history
            # values become a tie after that conversion.
            scored.append(struct.unpack("f", struct.pack("f", value))[0])
        if len(history) != receipt.get("epochsCompleted") or len(history) > recipe["maxEpochs"]:
            raise ValueError
        best = min(scored) if metric == "loss" else max(scored)
        if not math.isclose(best, receipt["bestValidationScore"], rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError
        # ModelCheckpoint retains the first strict optimum when later epochs tie.
        epoch = scored.index(best) + 1
    except (OSError, ValueError, TypeError, KeyError, OverflowError, StorageError) as error:
        raise StorageError(
            "This fold has no verifiable best checkpoint epoch. Refit requires its bestEpoch receipt or complete validation history; stopped epochs cannot substitute.",
            "REFIT_BEST_EPOCH_UNAVAILABLE",
            409,
        ) from error
    return {
        "runId": checkpoint["runId"],
        "bestEpoch": epoch,
        "source": "validation_history",
        "historyHash": _hash(history),
    }


def epoch_budget(epochs, percentile):
    """Linear empirical quantile followed by ceil; all epochs are one based."""
    if not epochs or any(type(epoch) is not int or epoch < 1 for epoch in epochs):
        raise ValueError("Refit needs positive, one-based best checkpoint epochs.")
    if not math.isfinite(percentile) or not 1 <= percentile <= 100:
        raise ValueError("Choose a percentile from 1 to 100.")
    ordered = sorted(epochs)
    index = (len(ordered) - 1) * percentile / 100
    low, high = math.floor(index), math.ceil(index)
    return math.ceil(ordered[low] + (ordered[high] - ordered[low]) * (index - low))


def prepare_refit(evidence, plan, folder):
    selected = evidence["runIds"]
    best_epochs = [_best_epoch(row, folder, evidence["recipe"]) for row in evidence["checkpoints"]]
    percentile = evidence["selection"]["refitPercentile"]
    epochs = epoch_budget([row["bestEpoch"] for row in best_epochs], percentile)
    members = {}
    split_ids = {row["splitPlanId"] for row in evidence["checkpoints"]}
    for split in split_ids:
        for row in plan["memberships"][split]:
            if row.get("phase") == "final" or row.get("pool") == "external_test":
                raise StorageError("Test rows cannot enter a refit.", "REFIT_TEST_LEAKAGE", 409)
            if row["partition"] not in {"train", "val", "test"}:
                raise StorageError(
                    "Unknown development partition.", "REFIT_MEMBERSHIP_INVALID", 409
                )
            normalized = {key: row[key] for key in ("slideId", "patientId", "label")}
            normalized.update(partition="train", phase="refit", pool="development")
            prior = members.get(row["slideId"])
            if prior is not None and prior != normalized:
                raise StorageError(
                    "Development folds disagree on a slide label or patient.",
                    "REFIT_MEMBERSHIP_INVALID",
                    409,
                )
            members[row["slideId"]] = normalized
    if not members or not selected:
        raise StorageError("A refit requires development slides.", "REFIT_MEMBERSHIP_INVALID", 409)
    source_recipe = deepcopy(evidence["recipe"])
    recipe = {
        **source_recipe,
        "maxEpochs": epochs,
        "minEpochs": epochs,
        "earlyStopping": False,
        "warmupEpochs": min(source_recipe.get("warmupEpochs", 0), epochs - 1),
    }
    budget = {
        "percentile": percentile,
        "epochs": epochs,
        "foldBestEpochs": best_epochs,
        "interpolation": "linear",
        "rounding": "ceil",
        "epochIndexing": "one_based",
    }
    rows = sorted(members.values(), key=lambda row: row["slideId"])
    return {
        **evidence,
        "kind": "predictor-refit",
        "method": "refit",
        "sourceCheckpoints": evidence["checkpoints"],
        "checkpoints": [],
        "sourceRecipe": source_recipe,
        "recipe": recipe,
        "epochBudget": budget,
        "trainingSlideCount": len(rows),
        "trainingPatientCount": len({row["patientId"] for row in rows}),
        "resources": ResourcePolicy.model_validate(
            {
                "dataLoaderWorkers": 0,
                **plan.get("resources", {}),
                "gpuIds": plan.get("resources", {}).get("gpuIds", [])[:1],
                "maxConcurrentRuns": 1,
                "runsPerGpu": 1,
            }
        ).model_dump(),
        "aggregation": "single_model",
        "status": "planned",
        "planTemplate": {
            "kind": "refit",
            "references": [
                evidence["batch"],
                evidence["inputs"]["protocol"],
                evidence["inputs"]["features"]["feature"],
                evidence["inputs"]["features"]["bundle"],
            ],
            "checkpoints": evidence["checkpoints"],
            "target": evidence["target"],
            "recipe": recipe,
            "trainingSeed": evidence["trainingSeed"],
            "data": {**deepcopy(plan["data"]), "memberships": rows},
            "epochBudget": budget,
            "provenanceHash": evidence_hash(evidence),
        },
    }


class RefitService:
    def __init__(self, store, filesystem, jobs=None):
        self.store, self.filesystem = store, filesystem
        self.predictors = PredictorService(store, filesystem)
        if jobs is None:
            from histopilot.application.compute_jobs import ComputeJobService

            jobs = ComputeJobService(store, filesystem)
        self.jobs = jobs

    def list(self, *, include_inactive=False):
        return {
            "items": [
                {
                    **lifecycle_document(self.store, row),
                    "execution": self.jobs.status(row["id"], include_inactive=include_inactive),
                }
                for row in self.store.list_configurations(
                    "predictor-refit", include_inactive=include_inactive
                )
            ],
            "executionEnabled": True,
        }

    def get(self, identity):
        row = self.store.get_configuration(identity)
        if row["manifest"].get("kind") != "predictor-refit":
            raise StorageError("Refit plan not found.", "REFIT_NOT_FOUND", 404)
        return lifecycle_document(self.store, row)

    def create(self, request):
        selection = PredictorSelection.model_validate(
            request.model_dump(include=set(PredictorSelection.model_fields))
        )
        if selection.method != "refit":
            raise StorageError("Select the refit build method.", "REFIT_METHOD_REQUIRED", 422)
        with lifecycle_guard(self.store.folder):
            prior = self.store.configuration_publication(request.operationId)
            if prior:
                if (
                    prior["manifest"].get("kind") != "predictor-refit"
                    or prior["manifest"].get("selection") != selection.model_dump()
                    or prior["manifest"].get("previewHash") != request.previewHash
                ):
                    raise StorageError(
                        "This operation belongs to another refit plan.", "OPERATION_CONFLICT", 409
                    )
                return lifecycle_document(self.store, prior)
            manifest = self.predictors._prepare(selection)
            if evidence_hash(manifest) != request.previewHash:
                raise StorageError(
                    "Development evidence changed. Review again.", "PREVIEW_STALE", 409
                )
            return lifecycle_document(
                self.store,
                self.store.publish_configuration(
                    manifest={**manifest, "previewHash": request.previewHash},
                    operation_id=request.operationId,
                ),
            )

    def _verify_sources(self, record):
        manifest = record["manifest"]
        selection = PredictorSelection.model_validate(manifest["selection"])
        current = self.predictors._prepare(selection)
        if not evidence_current(current, manifest):
            raise StorageError(
                "The refit's reviewed development evidence changed.", "REFIT_EVIDENCE_CHANGED", 409
            )

    def launch(self, identity, request, *, resume=False):
        with lifecycle_guard(self.store.folder):
            record = self.get(identity)
            self._verify_sources(record)
            resources = (
                request.resources.model_dump()
                if request.resources is not None
                else record["manifest"]["resources"]
            )
            if resources["maxConcurrentRuns"] != 1 or len(resources["gpuIds"]) > 1:
                raise StorageError(
                    "A refit trains one model on at most one GPU.", "REFIT_RESOURCES_INVALID", 422
                )
            plan = {
                **record["manifest"]["planTemplate"],
                "runId": record["id"],
                "recordId": record["id"],
                "recordContentHash": record["contentHash"],
                "resources": resources,
                "device": "cuda" if resources["gpuIds"] else "cpu",
            }
            return self.jobs.launch(identity, plan, request.operationId, resume=resume)

    def execution(self, identity):
        self.get(identity)
        return self.jobs.status(identity)

    def cancel(self, identity, operation_id):
        self.get(identity)
        return self.jobs.cancel(identity, operation_id=operation_id)

    def publish(self, identity, operation_id):
        with lifecycle_guard(self.store.folder):
            prior = self.store.configuration_publication(operation_id)
            if prior:
                if (
                    prior["manifest"].get("kind") != "frozen-predictor"
                    or prior["manifest"].get("refitId") != identity
                ):
                    raise StorageError(
                        "This operation belongs to another model.", "OPERATION_CONFLICT", 409
                    )
                return lifecycle_document(self.store, prior)
            record = self.get(identity)
            self._verify_sources(record)
            status = self.jobs.status(identity)
            from histopilot.application.lifecycle import _confirmed_live

            if status.get("status") != "completed" or _confirmed_live(status.get("process")):
                raise StorageError(
                    "The refit must complete and stop before publication.", "REFIT_INCOMPLETE", 409
                )
            folder = self.jobs.folder(identity)
            receipt = read_evidence(folder / "result.json", folder)
            plan = read_evidence(folder / "plan.json", folder)
            manifest = record["manifest"]
            if (
                receipt != status.get("result")
                or receipt.get("state") != "succeeded"
                or receipt.get("runId") != identity
                or receipt.get("epochsCompleted") != manifest["epochBudget"]["epochs"]
                or plan.get("recordContentHash") != record["contentHash"]
                or _hash(plan) != status.get("planHash")
                or any(plan.get(key) != value for key, value in manifest["planTemplate"].items())
            ):
                raise StorageError(
                    "Refit output does not match the reviewed plan.",
                    "REFIT_PROVENANCE_CHANGED",
                    409,
                )
            checkpoint = checkpoint_snapshot(receipt["bestCheckpointPath"], folder)
            published = {
                key: value
                for key, value in manifest.items()
                if key not in {"kind", "planTemplate", "status", "previewHash"}
            }
            published.update(
                kind="frozen-predictor",
                refitId=identity,
                refit=reference(record),
                checkpoints=[
                    {
                        "runId": identity,
                        **checkpoint,
                        "receiptHash": _hash(receipt),
                        "runPlanHash": _hash(plan),
                        "bestEpoch": receipt["epochsCompleted"],
                        "epochsCompleted": receipt["epochsCompleted"],
                    }
                ],
                executionPlanHash=_hash(plan),
            )
            return lifecycle_document(
                self.store,
                self.store.publish_configuration(manifest=published, operation_id=operation_id),
            )
