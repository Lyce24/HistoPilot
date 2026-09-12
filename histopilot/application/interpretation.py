"""Immutable attention studies connecting predictors, evaluation and slide evidence."""

import hashlib
import heapq
import json
import math
import re
import time
from collections import OrderedDict
from pathlib import Path
from threading import Lock

from histopilot.application.compute_jobs import ComputeJobService
from histopilot.application.feature_bundles import _hash
from histopilot.application.interpretation_gallery import (
    InterpretationGalleryService,
    allowed_folder,
)
from histopilot.application.predictors import (
    PredictorService,
    finding,
    lifecycle_document,
    reference,
)
from histopilot.schemas.interpretation import InterpretationSelection, SaveInterpretation
from histopilot.storage.attention_inputs import (
    as_input_error,
    file_stamp,
    inspect_inputs,
    verify_sources,
)
from histopilot.storage.filesystem import LocalFilesystem
from histopilot.storage.lifecycle import lifecycle_guard
from histopilot.storage.project_lock import StorageError, ensure_managed_directory, writer_lock
from histopilot.storage.scientific import ScientificStore
from histopilot.viewer.attention_arrays import attention_page, attention_top, coordinate_bounds
from histopilot.viewer.slide_images import allowed_file, inspect_slide, render_slide
from histopilot.workers.packing_process import write_json

EXECUTION_NOTE = (
    "ABMIL pooling attention is class-independent and describes relative patch weighting within a slide. "
    "It does not establish a class-specific explanation, causality, or diagnostic correctness. "
    "Ensembles average normalized attention and probabilities across every frozen member. "
    "Feature vectors and coordinates must describe the same patches in the same row order."
)
MAX_ARTIFACT_BYTES = 512 * 1024 * 1024
# Historical JSON-only writers were bounded to 64 MiB. Larger current studies
# use the verified mmap sidecar; raw JSON exports retain their 512 MiB limit.
MAX_LEGACY_VIEW_BYTES = 64 * 1024 * 1024
_LEGACY_RANKS = OrderedDict()
_LEGACY_RANKS_LOCK = Lock()
MAX_BATCH_SECONDS = 45


def _attention_scientific_inputs(manifest):
    """Scheduling reservations and display names do not change an existing attention map."""
    return {
        key: manifest.get(key)
        for key in (
            "kind",
            "schemaVersion",
            "datasetId",
            "experimentId",
            "predictorId",
            "predictor",
            "evaluationId",
            "clinicalAnalysisId",
            "featureBundleId",
            "packArtifactId",
            "slideFolder",
            "references",
            "method",
            "memberCount",
            "target",
            "encoderId",
            "featureContract",
            "slides",
            "attentionKind",
        )
    }


class InterpretationService:
    def __init__(self, store, filesystem):
        self.store, self.filesystem = store, filesystem
        self.predictors = PredictorService(store, filesystem)
        self.jobs = ComputeJobService(store, filesystem)
        self.gallery = InterpretationGalleryService(store, filesystem)
        self.pack_filesystem = LocalFilesystem((store.folder, *filesystem.roots))

    def gallery_thumbnail(self, value, *, max_size=256):
        self.store.lifecycle.assert_usable([f"project:{self.store.project_id}"])
        path = allowed_file(self.filesystem, value)
        stamp = file_stamp(path)
        content = render_slide(path, max_size=max_size)
        if file_stamp(path) != stamp:
            raise StorageError(
                "The slide changed while its thumbnail was rendered.",
                "INTERPRETATION_INPUTS_CHANGED",
                409,
            )
        return content

    def visualize(self, request):
        """Freeze and queue each selected slide independently, with durable retry intent."""
        operation_folder = (
            self.store.folder / "interpretation-requests" / _hash(request.operationId)
        )
        request_hash = _hash(request.model_dump())
        deadline = time.monotonic() + MAX_BATCH_SECONDS
        with lifecycle_guard(self.store.folder):
            self.store.lifecycle.assert_usable([f"project:{self.store.project_id}"])
            ensure_managed_directory(operation_folder)
        with writer_lock(operation_folder):
            receipt_path = operation_folder / "request.json"
            if receipt_path.exists():
                receipt = json.loads(ScientificStore._read_file(receipt_path, 2 * 1024 * 1024))
                if receipt.get("requestHash") != request_hash:
                    raise StorageError(
                        "This visualization operation belongs to a different selection. Start a new request after changing inputs.",
                        "OPERATION_CONFLICT",
                        409,
                    )
            else:
                write_json(
                    receipt_path, {"requestHash": request_hash, "request": request.model_dump()}
                )
            resolved, folder, rows, _ = self.gallery.rows(request, require_current=True)
            context = (resolved, folder, rows, deadline)
            by_path = {row["slidePath"]: row for row in rows}
            candidates = self.store.list_configurations("model-interpretation")
            result_path = operation_folder / "result.json"
            receipts = {}
            if result_path.exists():
                stored = json.loads(ScientificStore._read_file(result_path, 64 * 1024 * 1024))
                if stored.get("requestHash") != request_hash:
                    raise StorageError(
                        "Visualization receipt is inconsistent.",
                        "INTERPRETATION_RECEIPT_INVALID",
                        409,
                    )
                receipts = {row["slidePath"]: row for row in stored["items"]}

            def persist(result):
                receipts[result["slidePath"]] = {
                    key: value for key, value in result.items() if key != "interpretation"
                }
                write_json(
                    result_path, {"requestHash": request_hash, "items": list(receipts.values())}
                )

            results, documents = [], []
            for value in request.slidePaths:
                result = {
                    "slidePath": value,
                    "slideId": by_path.get(value, {}).get("slideId", Path(value).stem),
                    "status": "error",
                    "reused": False,
                }
                document = None
                try:
                    prior = receipts.get(value)
                    if prior and prior.get("_launchOperation"):
                        # An acknowledged or uncertain launch belongs to this exact operation.
                        # Replaying its original compute operation cannot start another attempt.
                        result.update(prior)
                        result.pop("error", None)
                        result["reused"] = True
                        document = self.get(prior["interpretationId"])
                        if not prior.get("_launched"):
                            self.launch(
                                document["id"],
                                prior["_launchOperation"],
                                resume=prior["_launchResume"],
                                gallery_context=context,
                            )
                            document = self.get(document["id"])
                            result["_launched"] = True
                        result.update(
                            interpretation=document, status=document["execution"]["status"]
                        )
                        documents.append(document)
                        persist(result)
                        results.append(
                            {key: value for key, value in result.items() if not key.startswith("_")}
                        )
                        continue
                    if time.monotonic() >= deadline:
                        raise StorageError(
                            "Batch review reached its 45-second limit. Completed submissions are retained; retry the remaining slides in a smaller batch.",
                            "INTERPRETATION_BATCH_LIMIT",
                            422,
                        )
                    path = str(allowed_file(self.filesystem, value))
                    if path not in by_path:
                        raise StorageError(
                            "The slide is not in the selected slide folder.",
                            "INTERPRETATION_SLIDE_UNAVAILABLE",
                            422,
                        )
                    row = by_path[path]
                    result["slideId"] = row["slideId"]
                    selected = self.gallery.input(
                        resolved,
                        row,
                        width=request.patchWidthLevel0,
                        height=request.patchHeightLevel0,
                    )
                    selection = InterpretationSelection(
                        name=f"Attention · {row['slideId']}"[:120],
                        predictorId=request.predictorId,
                        evaluationId=request.evaluationId,
                        clinicalAnalysisId=request.clinicalAnalysisId,
                        featureBundleId=request.featureBundleId,
                        packArtifactId=request.packArtifactId,
                        slideFolder=str(folder),
                        encoderId=resolved["contract"]["encoderId"],
                        slides=[selected],
                        resources=request.resources,
                    )
                    # Existing completed/running attention needs no new reservation.
                    # A fresh publication below still enforces its requested memory budget.
                    preview = self.preview(
                        selection, gallery_context=context, check_resources=False
                    )
                    if not preview["canSave"]:
                        failure = preview["findings"][0]
                        raise StorageError(failure["message"], failure["code"], 422)
                    previous = next(
                        (
                            item
                            for item in candidates
                            if item["manifest"].get("previewHash") == preview["previewHash"]
                        ),
                        None,
                    )
                    same_inputs = _attention_scientific_inputs(preview["manifest"])
                    reusable = []
                    for candidate in candidates:
                        if _attention_scientific_inputs(candidate["manifest"]) != same_inputs:
                            continue
                        current = self.get(candidate["id"])
                        status = current["execution"]["status"]
                        if status in {"completed", "queued", "running"}:
                            reusable.append(current)
                    if reusable:
                        previous = min(
                            reusable,
                            key=lambda item: (
                                item["execution"]["status"] != "completed",
                                item["id"],
                            ),
                        )
                    if previous:
                        # A reusable snapshot was already verified above. If its worker
                        # finishes meanwhile, polling observes that result; this request
                        # must not restart it under another reservation by accident.
                        document = previous if reusable else self.get(previous["id"])
                        result["reused"] = True
                    else:
                        document = self.save(
                            SaveInterpretation(
                                **selection.model_dump(),
                                previewHash=preview["previewHash"],
                                operationId=f"visualize-{_hash([request.operationId, path])}",
                            ),
                            gallery_context=context,
                        )
                        candidates.append(document)
                        document = self.get(document["id"])
                    execution = document["execution"]
                    if execution["status"] not in {"queued", "running", "completed"}:
                        resume = execution["status"] != "not_started"
                        launch_operation = f"visualize-{_hash([request.operationId, path, resume, execution.get('attempt', 0)])}"
                        result.update(
                            interpretationId=document["id"],
                            _launchOperation=launch_operation,
                            _launchResume=resume,
                            _launched=False,
                        )
                        persist(result)
                        self.launch(
                            document["id"],
                            launch_operation,
                            resume=resume,
                            gallery_context=context,
                        )
                        result["_launched"] = True
                        document = self.get(document["id"])
                    else:
                        # Reusing another request's study is also fixed to its current attempt.
                        result.update(
                            interpretationId=document["id"],
                            _launchOperation="reused",
                            _launchResume=False,
                            _launched=True,
                        )
                    result.update(
                        interpretationId=document["id"],
                        interpretation=document,
                        status=document["execution"]["status"],
                    )
                except StorageError as error:
                    result["status"] = "error"
                    result["error"] = {"code": error.code, "message": str(error)}
                    if document:
                        result["interpretationId"] = document["id"]
                        try:
                            document = self.get(document["id"])
                        except StorageError:
                            document = None
                        if document:
                            result["interpretation"] = document
                        if (
                            document
                            and error.code == "COMPUTE_ACTIVE"
                            and document["execution"]["status"]
                            in {
                                "queued",
                                "running",
                                "completed",
                            }
                        ):
                            result.pop("error", None)
                            result.update(
                                status=document["execution"]["status"], reused=True, _launched=True
                            )
                if document:
                    documents.append(document)
                persist(result)
                results.append(
                    {key: value for key, value in result.items() if not key.startswith("_")}
                )
            return {"items": results, "interpretations": documents}

    def list(self, *, include_inactive=False):
        return {
            "items": [
                {
                    **lifecycle_document(self.store, item),
                    "execution": self.jobs.status(item["id"], include_inactive=include_inactive),
                }
                for item in self.store.list_configurations(
                    "model-interpretation", include_inactive=include_inactive
                )
            ],
            "executionEnabled": True,
            "executionNote": EXECUTION_NOTE,
        }

    def get(self, identity):
        document = self.store.get_configuration(identity)
        if document["manifest"].get("kind") != "model-interpretation":
            raise StorageError("Interpretation not found.", "INTERPRETATION_NOT_FOUND", 404)
        return {**lifecycle_document(self.store, document), "execution": self.jobs.status(identity)}

    def inspect_slide(self, value):
        self.store.lifecycle.assert_usable([f"project:{self.store.project_id}"])
        path = allowed_file(self.filesystem, value)
        return {"path": str(path), **inspect_slide(path)}

    def _lineage(self, selection, predictor):
        evaluation_id = selection.evaluationId
        references = [reference(predictor)]
        clinical = None
        if selection.clinicalAnalysisId:
            clinical = self.store.get_configuration(selection.clinicalAnalysisId)
            manifest = clinical["manifest"]
            if (
                manifest.get("kind") != "clinical-analysis"
                or manifest.get("predictorId") != predictor["id"]
            ):
                raise StorageError(
                    "Clinical analysis must belong to the selected predictor.",
                    "INTERPRETATION_LINEAGE_MISMATCH",
                    409,
                )
            if evaluation_id and evaluation_id != manifest.get("evaluationId"):
                raise StorageError(
                    "Clinical analysis and evaluation must describe the same evaluation.",
                    "INTERPRETATION_LINEAGE_MISMATCH",
                    409,
                )
            evaluation_id = manifest.get("evaluationId")
            references.append(reference(clinical))
        if evaluation_id:
            evaluation = self.store.get_configuration(evaluation_id)
            if (
                evaluation["manifest"].get("kind") != "model-evaluation"
                or evaluation["manifest"].get("predictorId") != predictor["id"]
            ):
                raise StorageError(
                    "Evaluation must belong to the selected predictor.",
                    "INTERPRETATION_LINEAGE_MISMATCH",
                    409,
                )
            references.append(reference(evaluation))
        return evaluation_id, references

    def _prepare(self, selection, *, gallery_context=None, check_resources=True):
        predictor = self.predictors.get(selection.predictorId)
        model = predictor["manifest"]
        if model.get("recipe", {}).get("model", "abmil").lower() != "abmil":
            raise StorageError(
                "Select a native ABMIL predictor for attention overlay.",
                "INTERPRETATION_MODEL_UNSUPPORTED",
                422,
            )
        checkpoints = model.get("checkpoints", [])
        method = model.get("method", "ensemble")
        if (
            method not in {"ensemble", "refit"}
            or not checkpoints
            or (method == "refit" and len(checkpoints) != 1)
        ):
            raise StorageError(
                "Predictor membership is incomplete or unsupported.",
                "INTERPRETATION_PREDICTOR_INVALID",
                422,
            )
        contract = model["inputs"]["features"]
        if not contract.get("encoderId") or selection.encoderId != contract["encoderId"]:
            raise StorageError(
                "Features require an explicit encoder identity matching the frozen predictor.",
                "INTERPRETATION_ENCODER_MISMATCH",
                422,
            )
        evaluation_id, references = self._lineage(selection, predictor)
        self.predictors.verify_checkpoints(predictor)
        if selection.featureBundleId:
            requests, source_references, slide_folder = self.gallery.bind(
                selection, context=gallery_context
            )
            references.extend(source_references)
        else:
            requests = [request.model_dump() for request in selection.slides]
            slide_folder = None
        slides = []
        deadline = (
            min(time.monotonic() + 45, gallery_context[3])
            if gallery_context
            else time.monotonic() + 45
        )
        try:
            for row in requests:
                if row.get("sourceFormat") == "h5":
                    # Preserve scientific fingerprints of previously saved manual studies.
                    for key in ("sourceFormat", "packPath", "packSlideId"):
                        row.pop(key, None)
                elif row.get("sourceFormat") == "packed":
                    row["packPath"] = str(allowed_folder(self.pack_filesystem, row["packPath"]))
                for key in ("slidePath", "featurePath", "coordinatesPath"):
                    if row.get(key):
                        row[key] = str(allowed_file(self.filesystem, row[key]))
                stamp = file_stamp(row["slidePath"])
                dimensions = inspect_slide(row["slidePath"])
                evidence = inspect_inputs(row, dimensions, contract, deadline=deadline)
                slides.append({**row, **dimensions, **evidence, "slideSource": stamp})
            verify_sources(slides)
        except StorageError:
            raise
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise as_input_error(error) from error
        if len({row["slidePath"] for row in slides}) != len(slides):
            raise StorageError("Select each slide once.", "INTERPRETATION_DUPLICATE_SLIDE", 422)
        # Whole-bag attention retains exact normalization. Refuse clearly undersized
        # host reservations rather than quietly sampling patches to fit memory.
        peak = max(row["patchCount"] for row in slides)
        recipe = model.get("recipe", {})
        estimated = (
            peak
            * (
                contract["dimensions"] * 12
                + 512
                + (recipe.get("embedDim", 512) + recipe.get("attentionDim", 384) * 3) * 8
            )
            + 256 * 1024**2
        )
        if check_resources and estimated > selection.resources.ramGbPerRun * 1024**3:
            raise StorageError(
                "Whole-bag attention exceeds this RAM reservation. Increase RAM per run; patches are never silently sampled.",
                "INTERPRETATION_MEMORY_INSUFFICIENT",
                422,
            )
        return {
            "kind": "model-interpretation",
            "schemaVersion": 1,
            "datasetId": model["datasetId"],
            "name": selection.name,
            "selection": selection.model_dump(),
            "experimentId": model.get("experimentId"),
            "predictorId": predictor["id"],
            "predictor": reference(predictor),
            "evaluationId": evaluation_id,
            "clinicalAnalysisId": selection.clinicalAnalysisId,
            "featureBundleId": selection.featureBundleId,
            "packArtifactId": selection.packArtifactId,
            "slideFolder": slide_folder,
            "references": references,
            "method": method,
            "memberCount": len(checkpoints),
            "target": model["target"],
            "encoderId": selection.encoderId,
            "featureContract": {key: contract[key] for key in ("encoderId", "dimensions", "dtype")},
            "slides": slides,
            "attentionKind": "class_independent_pooling",
            "resources": selection.resources.model_dump(),
            "estimatedPeakRamGb": estimated / 1024**3,
            "executionEnabled": True,
            "executionNote": EXECUTION_NOTE,
        }

    def preview(self, selection, *, gallery_context=None, check_resources=True):
        try:
            manifest = self._prepare(
                selection, gallery_context=gallery_context, check_resources=check_resources
            )
            findings = []
            if any(slide["alignment"] == "user_confirmed" for slide in manifest["slides"]):
                findings.append(
                    {
                        "severity": "warning",
                        "code": "INTERPRETATION_ALIGNMENT_ATTESTED",
                        "message": "Separate coordinates have no embedded counterpart in the feature file. Row order relies on your explicit confirmation; equal counts cannot prove alignment.",
                    }
                )
            return {
                "canSave": True,
                "previewHash": _hash(manifest),
                "manifest": manifest,
                "findings": findings,
                "executionNote": EXECUTION_NOTE,
            }
        except StorageError as error:
            return {
                "canSave": False,
                "previewHash": None,
                "manifest": None,
                "findings": [finding(error.code, str(error))],
                "executionNote": EXECUTION_NOTE,
            }

    def save(self, request, *, gallery_context=None):
        selection = InterpretationSelection.model_validate(
            request.model_dump(include=set(InterpretationSelection.model_fields))
        )
        with lifecycle_guard(self.store.folder):
            prior = self.store.configuration_publication(request.operationId)
            if prior:
                manifest = prior["manifest"]
                if (
                    manifest.get("kind") != "model-interpretation"
                    or InterpretationSelection.model_validate(
                        manifest.get("selection")
                    ).model_dump()
                    != selection.model_dump()
                    or manifest.get("previewHash") != request.previewHash
                ):
                    raise StorageError(
                        "This operation belongs to another interpretation.",
                        "OPERATION_CONFLICT",
                        409,
                    )
                return lifecycle_document(self.store, prior)
        manifest = self._prepare(selection, gallery_context=gallery_context)
        if _hash(manifest) != request.previewHash:
            raise StorageError("Interpretation inputs changed. Review again.", "PREVIEW_STALE", 409)

        def verify_before_publish():
            try:
                verify_sources(manifest["slides"])
            except (OSError, ValueError) as error:
                raise as_input_error(error) from error

        return lifecycle_document(
            self.store,
            self.store.publish_configuration(
                manifest={**manifest, "previewHash": request.previewHash},
                operation_id=request.operationId,
                before_publish=verify_before_publish,
            ),
        )

    def execution(self, identity):
        return self.get(identity)["execution"]

    def _execution_plan(self, identity, *, gallery_context=None):
        document = self.get(identity)
        manifest = document["manifest"]
        reviewed = self._prepare(
            InterpretationSelection.model_validate(manifest["selection"]),
            gallery_context=gallery_context,
        )
        scientific_keys = (
            "predictor",
            "evaluationId",
            "clinicalAnalysisId",
            "references",
            "method",
            "target",
            "encoderId",
            "featureContract",
            "slides",
            "resources",
        )
        if any(reviewed[key] != manifest[key] for key in scientific_keys):
            raise StorageError(
                "Interpretation evidence changed. Create a new reviewed interpretation.",
                "INTERPRETATION_INPUTS_CHANGED",
                409,
            )
        predictor = self.predictors.get(manifest["predictorId"])
        return {
            "kind": "interpretation",
            "runId": identity,
            "method": manifest["method"],
            "target": manifest["target"],
            "resources": manifest["resources"],
            "checkpoints": predictor["manifest"]["checkpoints"],
            "references": manifest["references"],
            "slides": manifest["slides"],
            "featureContract": manifest["featureContract"],
            "data": {
                "sourceStamps": {
                    path: stamp
                    for slide in manifest["slides"]
                    for path, stamp in slide["sourceStamps"].items()
                }
            },
        }

    def launch(self, identity, operation_id, *, resume=False, gallery_context=None):
        plan = self._execution_plan(identity, gallery_context=gallery_context)
        with lifecycle_guard(self.store.folder):
            try:
                verify_sources(plan["slides"])
            except (OSError, ValueError) as error:
                raise as_input_error(error) from error
            return self.jobs.launch(identity, plan, operation_id, resume=resume)

    def cancel(self, identity, operation_id):
        self.get(identity)
        return self.jobs.cancel(identity, operation_id)

    def artifact(self, identity, filename, *, max_bytes=MAX_ARTIFACT_BYTES):
        if not re.fullmatch(
            r"(?:attention\.json|slide-\d+(?:-member-\d+)?\.(?:json|npy))", filename
        ):
            raise StorageError(
                "Attention artifact not found.", "INTERPRETATION_ARTIFACT_NOT_FOUND", 404
            )
        execution = self.execution(identity)
        if execution["status"] != "completed":
            raise StorageError(
                "Attention is available after execution completes.",
                "INTERPRETATION_NOT_COMPLETED",
                409,
            )
        expected = execution["result"].get("artifacts", {}).get(filename)
        path = self.jobs.folder(identity) / filename
        if not expected or expected.get("path") != str(path):
            raise StorageError(
                "Attention artifact not found.", "INTERPRETATION_ARTIFACT_NOT_FOUND", 404
            )
        if max_bytes < MAX_ARTIFACT_BYTES and expected.get("bytes", 0) > max_bytes:
            raise StorageError(
                "This legacy JSON attention map exceeds the 64 MiB viewer limit. Recompute attention to create a current array artifact; the original JSON export remains available.",
                "INTERPRETATION_LEGACY_VIEW_LIMIT",
                413,
            )
        content = ScientificStore._read_file(path, max_bytes)
        if (
            len(content) != expected["bytes"]
            or hashlib.sha256(content).hexdigest() != expected["sha256"]
        ):
            raise StorageError("Attention output changed.", "INTERPRETATION_RESULT_CHANGED", 409)
        return content

    def _slide(self, identity, slide_id):
        document = self.get(identity)
        for index, row in enumerate(document["manifest"]["slides"]):
            if row["slideId"] == slide_id:
                return document, index, row
        raise StorageError("Selected slide not found.", "INTERPRETATION_SLIDE_NOT_FOUND", 404)

    def image(self, identity, slide_id, *, max_size=1024, region=None):
        _, _, row = self._slide(identity, slide_id)
        path = allowed_file(self.filesystem, row["slidePath"])
        try:
            if file_stamp(path) != row["slideSource"]:
                raise ValueError("Slide image changed after review; create a new interpretation.")
            image = render_slide(path, max_size=max_size, region=region)
            if file_stamp(path) != row["slideSource"]:
                raise ValueError("Slide image changed during rendering.")
            return image
        except (OSError, ValueError) as error:
            raise as_input_error(error) from error

    def _legacy_attention(self, identity, filename, expected, row, *, mode=None):
        """Retain only twenty ranked rows and one requested crop per verified file.

        A full stat fingerprint and receipt bind cache reuse. Serialize historical
        JSON parsing so concurrent crop requests cannot decode twenty full maps.
        """
        path = self.jobs.folder(identity) / filename

        def compact(patch):
            return {key: patch[key] for key in ("index", "x", "y", "weight", "percentile")}

        try:
            if (
                not expected
                or expected.get("path") != str(path)
                or expected.get("bytes", 0) > MAX_LEGACY_VIEW_BYTES
            ):
                self.artifact(identity, filename, max_bytes=MAX_LEGACY_VIEW_BYTES)
            with _LEGACY_RANKS_LOCK:
                stamp = file_stamp(path)
                signature = (
                    str(path),
                    expected["sha256"],
                    expected["bytes"],
                    tuple(sorted(stamp.items())),
                )
                cached = _LEGACY_RANKS.get(signature)
                if cached is not None and mode is not None:
                    selected = (
                        cached["top"]
                        if mode[0] == "top"
                        else [cached["rows"][mode[1]]]
                        if mode[1] in cached["rows"]
                        else []
                    )
                    if selected or (mode[0] == "patch" and mode[1] >= row["patchCount"]):
                        if file_stamp(path) != stamp:
                            raise ValueError("Attention JSON changed while selecting its patch.")
                        _LEGACY_RANKS.move_to_end(signature)
                        return {
                            **cached["metadata"],
                            "coordinateBounds": dict(cached["metadata"]["coordinateBounds"]),
                            "probabilities": list(cached["metadata"]["probabilities"]),
                            "patches": [dict(patch) for patch in selected],
                        }
                source = json.loads(
                    self.artifact(identity, filename, max_bytes=MAX_LEGACY_VIEW_BYTES)
                )
                legacy = source["patches"]
                if (
                    source.get("slideId") != row["slideId"]
                    or source.get("patchCount") != row["patchCount"]
                    or not isinstance(legacy, list)
                    or len(legacy) != row["patchCount"]
                ):
                    raise ValueError("Attention slide identity or patch count changed.")
                minimum_x, minimum_y = row["width"], row["height"]
                maximum_x = maximum_y = 0
                for patch_index, patch in enumerate(legacy):
                    numbers = [patch[key] for key in ("x", "y", "weight", "percentile")]
                    if (
                        type(patch["index"]) is not int
                        or patch["index"] != patch_index
                        or any(type(value) not in (int, float) for value in numbers)
                        or any(not math.isfinite(value) or value < 0 for value in numbers)
                        or patch["x"] != int(patch["x"])
                        or patch["y"] != int(patch["y"])
                        or patch["x"] >= row["width"]
                        or patch["y"] >= row["height"]
                        or patch["weight"] > 1
                        or patch["percentile"] > 1
                    ):
                        raise ValueError(
                            "Attention patch coordinates, index or weights are invalid."
                        )
                    minimum_x, minimum_y = min(minimum_x, patch["x"]), min(minimum_y, patch["y"])
                    maximum_x, maximum_y = max(maximum_x, patch["x"]), max(maximum_y, patch["y"])
                source["coordinateBounds"] = coordinate_bounds(
                    row, minimum_x, minimum_y, maximum_x, maximum_y
                )
                if file_stamp(path) != stamp:
                    raise ValueError("Attention JSON changed while reading its patches.")
                top = heapq.nsmallest(
                    20, legacy, key=lambda patch: (-patch["weight"], patch["index"])
                )
                rows = {patch["index"]: compact(patch) for patch in top}
                if mode and mode[0] == "patch" and mode[1] < len(legacy):
                    rows[mode[1]] = compact(legacy[mode[1]])
                _LEGACY_RANKS[signature] = {
                    "metadata": {
                        "slideId": source["slideId"],
                        "patchCount": source["patchCount"],
                        "probabilities": list(source["probabilities"]),
                        "coordinateBounds": dict(source["coordinateBounds"]),
                    },
                    "top": [compact(patch) for patch in top],
                    "rows": rows,
                }
                _LEGACY_RANKS.move_to_end(signature)
                while len(_LEGACY_RANKS) > 128:
                    _LEGACY_RANKS.popitem(last=False)
                if mode:
                    selected = top if mode[0] == "top" else legacy[mode[1] : mode[1] + 1]
                    return {
                        **_LEGACY_RANKS[signature]["metadata"],
                        "coordinateBounds": dict(source["coordinateBounds"]),
                        "probabilities": list(source["probabilities"]),
                        "patches": [compact(patch) for patch in selected],
                    }
                return source
        except StorageError:
            raise
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise StorageError(
                f"Attention output changed or is invalid: {error}",
                "INTERPRETATION_RESULT_CHANGED",
                409,
            ) from error

    def _attention_context(self, identity, slide_id, member, *, legacy_mode=None):
        document, index, row = self._slide(identity, slide_id)
        if member != "mean" and (
            not isinstance(member, str)
            or not re.fullmatch(r"[0-9]{1,8}", member)
            or int(member) >= document["manifest"]["memberCount"]
        ):
            raise StorageError(
                "Choose the ensemble mean or an existing model member.",
                "INTERPRETATION_MEMBER_INVALID",
                422,
            )
        execution = document["execution"]
        if execution["status"] != "completed":
            raise StorageError(
                "Attention is available after execution completes.",
                "INTERPRETATION_NOT_COMPLETED",
                409,
            )
        stem = f"slide-{index}" if member == "mean" else f"slide-{index}-member-{int(member)}"
        array_name = stem + ".npy"
        expected = execution["result"].get("artifacts", {}).get(array_name)
        legacy = None
        if expected:
            summaries = execution["result"].get("slides", [])
            summary = next((item for item in summaries if item.get("slideId") == slide_id), None)
            if summary is None or summary.get("patchCount") != row["patchCount"]:
                raise StorageError(
                    "Attention slide provenance changed.", "INTERPRETATION_RESULT_CHANGED", 409
                )
            source = (
                summary
                if member == "mean"
                else next(
                    (
                        item
                        for item in summary.get("members", [])
                        if item.get("index") == int(member)
                    ),
                    None,
                )
            )
            if source is None or source.get("attentionArray") != array_name:
                raise StorageError(
                    "Attention member provenance changed.", "INTERPRETATION_RESULT_CHANGED", 409
                )
        else:
            source = self._legacy_attention(
                identity,
                stem + ".json",
                execution["result"].get("artifacts", {}).get(stem + ".json"),
                row,
                mode=legacy_mode,
            )
            legacy = source["patches"]
        metadata = {
            "slideId": slide_id,
            "patchCount": row["patchCount"],
            "width": row["width"],
            "height": row["height"],
            "patchWidthLevel0": row["patchWidthLevel0"],
            "patchHeightLevel0": row["patchHeightLevel0"],
            "coordinateSpace": "level0",
            "attentionKind": "class_independent_pooling",
            "attentionNote": EXECUTION_NOTE,
            "classOrder": document["manifest"]["target"]["classes"],
            "member": member,
            "probabilities": source["probabilities"],
            **({"coordinateBounds": dict(source["coordinateBounds"])} if not expected else {}),
        }
        return row, metadata, self.jobs.folder(identity) / array_name, expected, legacy

    def attention(self, identity, slide_id, *, member="mean", offset=0, limit=10000, region=None):
        if (
            type(offset) is not int
            or type(limit) is not int
            or offset < 0
            or not 1 <= limit <= 50000
        ):
            raise StorageError(
                "Attention pagination is outside its bounds.", "INTERPRETATION_VIEW_INVALID", 422
            )
        row, metadata, path, expected, patches = self._attention_context(identity, slide_id, member)
        if region:
            x, y, width, height = region
            if (
                any(not math.isfinite(v) for v in region)
                or x < 0
                or y < 0
                or width <= 0
                or height <= 0
                or x + width > row["width"]
                or y + height > row["height"]
            ):
                raise StorageError(
                    "Choose a viewport inside the slide.", "INTERPRETATION_VIEW_INVALID", 422
                )
        if expected:
            return {
                **metadata,
                **attention_page(path, expected, row, offset=offset, limit=limit, region=region),
            }
        if region:
            patches = [
                patch
                for patch in patches
                if patch["x"] < x + width
                and patch["x"] + row["patchWidthLevel0"] > x
                and patch["y"] < y + height
                and patch["y"] + row["patchHeightLevel0"] > y
            ]
        return {
            **metadata,
            "total": len(patches),
            "offset": offset,
            "limit": limit,
            "patches": patches[offset : offset + limit],
        }

    def top_attention(self, identity, slide_id, *, member="mean", limit=10):
        if type(limit) is not int or not 1 <= limit <= 20:
            raise StorageError(
                "Choose 1–20 attention locations.", "INTERPRETATION_VIEW_INVALID", 422
            )
        row, metadata, path, expected, patches = self._attention_context(
            identity, slide_id, member, legacy_mode=("top", limit)
        )
        if expected:
            ranking = attention_top(path, expected, row, limit=limit)
        else:
            selected = heapq.nsmallest(
                limit, patches, key=lambda patch: (-patch["weight"], patch["index"])
            )
            ranking = {
                "scope": "whole_slide",
                "total": row["patchCount"],
                "returned": len(selected),
                "offset": 0,
                "limit": limit,
                "patches": [{**patch, "rank": rank} for rank, patch in enumerate(selected, 1)],
            }
        return {**metadata, **ranking}

    def patch_image(self, identity, slide_id, patch_index, *, member="mean", max_size=512):
        if type(patch_index) is not int or patch_index < 0:
            raise StorageError(
                "Choose an existing patch index.", "INTERPRETATION_PATCH_INVALID", 422
            )
        if type(max_size) is not int or not 64 <= max_size <= 1024:
            raise StorageError(
                "Patch image size must be 64–1024 pixels.", "SLIDE_VIEW_INVALID", 422
            )
        row, _, path, expected, patches = self._attention_context(
            identity, slide_id, member, legacy_mode=("patch", patch_index)
        )
        if expected:
            patches = attention_page(path, expected, row, offset=patch_index, limit=1)["patches"]
        if not patches:
            raise StorageError(
                "Patch not found on this slide.", "INTERPRETATION_PATCH_NOT_FOUND", 404
            )
        patch = patches[0]
        # Coordinates are authenticated artifact rows, never caller-provided boxes.
        # Patches at slide edges show only their actual in-slide footprint.
        bounds = (
            patch["x"],
            patch["y"],
            min(row["patchWidthLevel0"], row["width"] - patch["x"]),
            min(row["patchHeightLevel0"], row["height"] - patch["y"]),
        )
        return self.image(identity, slide_id, max_size=max_size, region=bounds)
