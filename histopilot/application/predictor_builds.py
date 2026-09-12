"""Reviewed per-seed predictor batches with durable, independently retryable items."""

from __future__ import annotations

import hashlib

from histopilot.application.feature_bundles import _hash
from histopilot.application.predictors import (
    PredictorService,
    evidence_current,
    finding,
    lifecycle_document,
)
from histopilot.schemas.predictors import PredictorBuildSelection, PredictorSelection
from histopilot.storage.lifecycle import lifecycle_guard
from histopilot.storage.project_lock import (
    StorageError,
    _reject_symlink_components,
    ensure_managed_directory,
)
from histopilot.workers.packing_process import write_json
from histopilot.workers.training_process import now, read_json


class PredictorBuildService:
    def __init__(self, store, filesystem):
        self.store, self.filesystem = store, filesystem
        self.predictors = PredictorService(store, filesystem)

    def _folder(self, operation_id):
        folder = (
            self.store.folder
            / "predictor-builds"
            / hashlib.sha256(operation_id.encode()).hexdigest()
        )
        _reject_symlink_components(folder)
        return folder

    def _existing_plan(self, selection):
        matches = [
            lifecycle_document(self.store, row)
            for row in self.store.list_configurations("predictor-refit", include_inactive=True)
            if self.predictors.source_key(row["manifest"]) == self.predictors.source_key(selection)
        ]
        matches.sort(
            key=lambda row: (
                row["lifecycleState"] != "active",
                row["manifest"].get("epochBudget", {}).get("percentile")
                != selection.refitPercentile,
                row["id"],
            )
        )
        return matches[0] if matches else None

    def _item(self, selection):
        key = _hash(self.predictors.source_key(selection))
        kind = "frozen-predictor" if selection.method == "ensemble" else "predictor-refit"
        result = {
            "key": key,
            "selection": selection.model_dump(),
            "recordKind": kind,
            "action": "blocked",
            "findings": [],
        }
        existing = self.predictors._existing(selection)
        if existing:
            existing = lifecycle_document(self.store, existing)
        elif selection.method == "refit":
            existing = self._existing_plan(selection)
        if existing and existing["lifecycleState"] != "active":
            return {
                **result,
                "recordId": existing["id"],
                "recordKind": existing["manifest"]["kind"],
                "findings": [
                    finding(
                        "PREDICTOR_RECORD_INACTIVE",
                        "This exact configuration, seed group and method already exists in Archive or Trash. Restore that record explicitly before building.",
                    )
                ],
            }
        try:
            selected = selection
            if existing:
                selected = PredictorSelection.model_validate(existing["manifest"]["selection"])
            # Reuse still requires intact completed source evidence. It is never
            # permission to substitute stopped/failed folds or changed inputs.
            manifest = self.predictors._prepare(selected, allow_existing=True)
            if existing and existing["manifest"]["kind"] == "frozen-predictor":
                self.predictors.verify_checkpoints(existing)
            if (
                existing
                and existing["manifest"]["kind"] == "predictor-refit"
                and not evidence_current(manifest, existing["manifest"])
            ):
                raise StorageError(
                    "The saved refit plan's reviewed development evidence changed.",
                    "REFIT_EVIDENCE_CHANGED",
                    409,
                )
            if existing:
                evidence = existing["manifest"]
                result.update(
                    action="reuse",
                    recordId=existing["id"],
                    recordKind=evidence["kind"],
                    manifestHash=existing["contentHash"],
                )
                if (
                    selected.method == "refit"
                    and selected.refitPercentile != selection.refitPercentile
                ):
                    result["findings"].append(
                        {
                            "severity": "warning",
                            "code": "REFIT_POLICY_ALREADY_EXISTS",
                            "message": f"Reusing the existing P{selected.refitPercentile:g} refit for this seed group; the requested percentile does not change an existing model or plan.",
                        }
                    )
            else:
                evidence = manifest
                result.update(action="create", manifestHash=_hash(manifest), _manifest=manifest)
            result.update(
                epochBudget=evidence.get("epochBudget"),
                trainingSlideCount=evidence.get("trainingSlideCount"),
                name=evidence["name"],
            )
        except StorageError as error:
            result["findings"].append(finding(error.code, str(error)))
        return result

    def _review(self, request):
        items = []
        methods = ("ensemble", "refit") if request.method == "both" else (request.method,)
        for source in request.selections:
            try:
                batch = self.store.get_configuration(source.batchId)
                experiment = self.predictors._experiment(source.experimentId, batch)
                number = next(
                    row["number"]
                    for row in batch["manifest"]["configurations"]
                    if row["id"] == source.candidateId
                )
                prefix = request.namePrefix or experiment["name"]
            except (StorageError, KeyError, StopIteration, TypeError):
                prefix, number = request.namePrefix or "Predictor", source.candidateId[-8:]
            for method in methods:
                suffix = f" · config {number} · seed {source.trainingSeed} / split {source.splitSeed} · {method}"
                name = prefix[: 120 - len(suffix)] + suffix
                selection = PredictorSelection(
                    **source.model_dump(),
                    name=name,
                    method=method,
                    refitPercentile=request.refitPercentile,
                )
                items.append(self._item(selection))
        public = [{key: value for key, value in row.items() if key != "_manifest"} for row in items]
        can_build = all(row["action"] != "blocked" for row in items)
        counts = {
            "total": len(items),
            **{
                state: sum(row["action"] == state for row in items)
                for state in ("create", "reuse", "blocked")
            },
        }
        return {
            "canBuild": can_build,
            "previewHash": _hash({"request": request.model_dump(), "items": public})
            if can_build
            else None,
            "items": public,
            "findings": [],
            "counts": counts,
        }, items

    def preview(self, request):
        with lifecycle_guard(self.store.folder):
            return self._review(request)[0]

    @staticmethod
    def _public(receipt):
        return {
            key: receipt[key]
            for key in ("operationId", "status", "items", "createdAt", "updatedAt")
        }

    @staticmethod
    def _validate_receipt(receipt, operation_id):
        try:
            valid = (
                receipt["version"] == 1
                and receipt["operationId"] == operation_id
                and _hash({"selection": receipt["request"], "previewHash": receipt["previewHash"]})
                == receipt["requestHash"]
                and _hash({"request": receipt["request"], "items": receipt["review"]})
                == receipt["previewHash"]
                and receipt["status"] in {"partial", "completed"}
                and _hash(receipt["items"]) == receipt["resultsHash"]
            )
            expected = {row["key"]: row for row in receipt["review"]}
            result_keys = {row["key"] for row in receipt["items"]}
            valid = (
                valid
                and len(expected) == len(receipt["review"])
                and len(result_keys) == len(receipt["items"])
                and result_keys <= set(expected)
            )
            valid = valid and all(
                row["selection"] == expected[row["key"]]["selection"]
                and row["status"] in {"created", "reused", "failed"}
                for row in receipt["items"]
            )
            completed = len(result_keys) == len(expected) and all(
                row["status"] != "failed" for row in receipt["items"]
            )
            valid = valid and (receipt["status"] == "completed") == completed
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            raise StorageError(
                "The durable predictor build receipt changed or is malformed.",
                "PREDICTOR_BUILD_RECEIPT_INVALID",
                409,
            )

    def get(self, operation_id):
        self.store.lifecycle.assert_usable([])
        path = self._folder(operation_id) / "receipt.json"
        if not path.exists():
            raise StorageError(
                "Predictor build receipt not found.", "PREDICTOR_BUILD_NOT_FOUND", 404
            )
        receipt = read_json(path)
        self._validate_receipt(receipt, operation_id)
        return self._public(receipt)

    def apply(self, request):
        selection = PredictorBuildSelection.model_validate(
            request.model_dump(include=set(PredictorBuildSelection.model_fields))
        )
        request_hash = _hash(
            {"selection": selection.model_dump(), "previewHash": request.previewHash}
        )
        with lifecycle_guard(self.store.folder):
            folder = self._folder(request.operationId)
            path = folder / "receipt.json"
            if path.exists():
                receipt = read_json(path)
                self._validate_receipt(receipt, request.operationId)
                if (
                    receipt.get("operationId") != request.operationId
                    or receipt.get("requestHash") != request_hash
                ):
                    raise StorageError(
                        "This operation belongs to another predictor build request.",
                        "OPERATION_CONFLICT",
                        409,
                    )
                if receipt["status"] == "completed":
                    return self._public(receipt)
            else:
                preview, _ = self._review(selection)
                if not preview["canBuild"] or preview["previewHash"] != request.previewHash:
                    raise StorageError(
                        "Build inputs, evidence, or record visibility changed. Review again.",
                        "PREVIEW_STALE",
                        409,
                    )
                self.store.lifecycle.assert_usable([])
                ensure_managed_directory(folder)
                receipt = {
                    "version": 1,
                    "operationId": request.operationId,
                    "requestHash": request_hash,
                    "request": selection.model_dump(),
                    "previewHash": request.previewHash,
                    "status": "partial",
                    "createdAt": now(),
                    "updatedAt": now(),
                    "review": preview["items"],
                    "items": [],
                    "resultsHash": _hash([]),
                }
                write_json(path, receipt)
            results = {row["key"]: row for row in receipt["items"]}
            for expected in receipt["review"]:
                key = expected["key"]
                if results.get(key, {}).get("status") in {"created", "reused"}:
                    continue
                result = {
                    "key": key,
                    "method": expected["selection"]["method"],
                    "selection": expected["selection"],
                    "recordKind": expected["recordKind"],
                }
                try:
                    self.store.lifecycle.assert_usable([])
                    operation = "predictor-build-item-" + _hash([request.operationId, key])
                    prior = self.store.configuration_publication(operation)
                    if prior:
                        original = {
                            k: v for k, v in prior["manifest"].items() if k != "previewHash"
                        }
                        if (
                            prior["manifest"].get("previewHash") != expected["manifestHash"]
                            or _hash(original) != expected["manifestHash"]
                        ):
                            raise StorageError(
                                "A build item publication differs from its reviewed inputs.",
                                "OPERATION_CONFLICT",
                                409,
                            )
                        result.update(status="created", recordId=prior["id"])
                    else:
                        current = self._item(
                            PredictorSelection.model_validate(expected["selection"])
                        )
                        if any(
                            current.get(field) != expected.get(field)
                            for field in ("action", "manifestHash", "recordId", "recordKind")
                        ):
                            raise StorageError(
                                "This build item's source evidence or visibility changed. Review a new build request.",
                                "PREVIEW_STALE",
                                409,
                            )
                        if current["action"] == "reuse":
                            result.update(status="reused", recordId=current["recordId"])
                        elif current["action"] == "create":
                            record = self.store.publish_configuration(
                                manifest={
                                    **current["_manifest"],
                                    "previewHash": current["manifestHash"],
                                },
                                operation_id=operation,
                            )
                            result.update(status="created", recordId=record["id"])
                        else:
                            raise StorageError(
                                "This build item is blocked.", "PREDICTOR_BUILD_BLOCKED", 409
                            )
                except StorageError as error:
                    result.update(
                        status="failed", error={"code": error.code, "message": str(error)}
                    )
                results[key] = result
                receipt["items"] = [
                    results[item["key"]] for item in receipt["review"] if item["key"] in results
                ]
                receipt["status"] = (
                    "completed"
                    if len(receipt["items"]) == len(receipt["review"])
                    and all(row["status"] != "failed" for row in receipt["items"])
                    else "partial"
                )
                receipt["updatedAt"] = now()
                receipt["resultsHash"] = _hash(receipt["items"])
                write_json(path, receipt)
            return self._public(receipt)
