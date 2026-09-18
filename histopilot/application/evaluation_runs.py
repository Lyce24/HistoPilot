"""Reviewed predictor/cohort plans with durable refit and ensemble inference."""

import hashlib
import json
from pathlib import Path

from histopilot.adapters.native.runtime import training_runtime
from histopilot.application.compute_jobs import ComputeJobService
from histopilot.application.evaluations import EvaluationService
from histopilot.application.feature_bundles import _hash
from histopilot.application.predictors import (
    PredictorService,
    feature_contract,
    finding,
    lifecycle_document,
    reference,
)
from histopilot.domain.features import representation_kind
from histopilot.schemas.analysis import PatientAnalysisSettings
from histopilot.schemas.development import ResourcePolicy
from histopilot.schemas.evaluations import EvaluationSpec, InferenceSettings
from histopilot.schemas.predictors import EvaluationRunSelection
from histopilot.schemas.protocols import TargetSpec
from histopilot.storage.lifecycle import lifecycle_guard
from histopilot.storage.pack_import import _layout
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import ScientificStore
from histopilot.workers.training_process import read_json

EXECUTION_NOTE = (
    "Evaluate the selected predictor on its frozen test cohort using the recorded bag, "
    "aggregation and decision-threshold settings. Patient predictions combine eligible "
    "slides; metrics use labeled test records."
)


class EvaluationRunService:
    def __init__(self, store, filesystem):
        self.store, self.filesystem = store, filesystem
        self.predictors = PredictorService(store, filesystem)
        self.cohorts = EvaluationService(store, filesystem)
        self.jobs = ComputeJobService(store, filesystem)

    def list(self, *, include_inactive=False):
        return {
            "items": [
                {
                    **lifecycle_document(self.store, item),
                    "execution": self.jobs.status(item["id"], include_inactive=include_inactive),
                }
                for item in self.store.list_configurations(
                    "model-evaluation", include_inactive=include_inactive
                )
            ],
            "executionEnabled": True,
            "executionNote": EXECUTION_NOTE,
        }

    def get(self, identity):
        document = self.store.get_configuration(identity)
        if document["manifest"].get("kind") != "model-evaluation":
            raise StorageError("Evaluation record not found.", "MODEL_EVALUATION_NOT_FOUND", 404)
        return {**lifecycle_document(self.store, document), "execution": self.jobs.status(identity)}

    def patient_evidence(self, identity):
        """Read checksummed predictions and validate their frozen patient membership."""
        from histopilot.application.clinical import _patient_records, _validate_records

        document = self.get(identity)
        manifest = document["manifest"]
        cohort = self.store.get_configuration(manifest["cohortId"])
        predictor = self.store.get_configuration(manifest["predictorId"])
        if (manifest["cohort"] != reference(cohort)
                or manifest["predictor"] != reference(predictor)
                or manifest["target"] != predictor["manifest"]["target"]):
            raise StorageError("Evaluation references or targets changed.", "COMPARISON_EVIDENCE_CHANGED", 409)
        content = self.artifact(identity, "predictions.json")
        try:
            predictions = json.loads(content)
            classes = manifest["target"]["classes"]
            if predictions["classOrder"] != classes:
                raise ValueError("Prediction class order differs from the frozen evaluation.")
            slides = _validate_records(predictions["records"], classes)
            memberships = {row["slideId"]: row for row in cohort["manifest"]["memberships"]}
            observed = {row["slideId"]: (row.get("patientId"), row.get("label")) for row in slides}
            expected = {key: (row.get("patientId"), row.get("label"))
                        for key, row in memberships.items()}
            if observed != expected:
                raise ValueError("Predictions must cover exactly the frozen cohort membership and labels.")
            for row in slides:
                source = memberships[row["slideId"]].get("patientIdSource")
                if "patientIdSource" in row and row["patientIdSource"] != source:
                    raise ValueError("Patient identity provenance differs from the frozen cohort.")
                if not row.get("patientId") or source == "slide_fallback":
                    raise ValueError("Patient comparisons require verified patient identities.")
                row["patientIdSource"] = source
            patients = _patient_records(slides, predictions.get("patientRecords"), classes,
                                         manifest["inference"]["patientAggregation"])
        except (ValueError, KeyError, TypeError, OverflowError) as error:
            raise StorageError(str(error), "COMPARISON_EVIDENCE_INVALID", 409) from error
        return document, slides, patients, hashlib.sha256(content).hexdigest()

    def compare(self, request):
        from histopilot.statistics import patient_bootstrap

        left, _left_slides, left_patients, left_hash = self.patient_evidence(request.leftEvaluationId)
        right, _right_slides, right_patients, right_hash = self.patient_evidence(request.rightEvaluationId)
        a, b = left["manifest"], right["manifest"]
        if (a["cohort"] != b["cohort"] or a["target"] != b["target"]
                or a["inference"]["patientAggregation"] != b["inference"]["patientAggregation"]):
            raise StorageError("Choose the same frozen cohort, target, and patient aggregation for paired comparison.",
                               "COMPARISON_CONTEXT_MISMATCH", 409)
        try:
            result = patient_bootstrap(left_patients, a["target"], request.analysis.model_dump(),
                                       other=right_patients)
        except ValueError as error:
            raise StorageError(str(error), "COMPARISON_PATIENT_MISMATCH", 409) from error
        return {
            "leftEvaluationId": left["id"], "rightEvaluationId": right["id"],
            "leftName": a.get("name", left["id"]), "rightName": b.get("name", right["id"]),
            "cohortId": a["cohortId"], "target": a["target"],
            "patientAggregation": a["inference"]["patientAggregation"],
            "analysis": request.analysis.model_dump(), "difference": "left_minus_right",
            "predictionsSha256": {"left": left_hash, "right": right_hash},
            "statistics": result,
        }

    def _test_bundle(self, selection, model, test, inference):
        """Resolve once at review; execution also checks the reviewed feature references."""
        if selection.featureBundleId:
            return selection.featureBundleId
        if test["spec"].get("featureBundleId"):
            return test["spec"]["featureBundleId"]
        selected = {row["slideId"] for row in test["memberships"]}
        development = model["inputs"]["features"]
        candidates = {}
        for document in self.store.list_configurations("feature-bundle"):
            try:
                bundle = self.cohorts.bundles.get(document["id"])
                if not bundle["current"]:
                    continue
                if inference.packArtifactId and not any(
                    pack["id"] == inference.packArtifactId for pack in bundle["manifest"]["packs"]
                ):
                    continue
                feature = self.store.get_configuration(bundle["manifest"]["feature"]["id"])
                available = {row["slideId"] for row in feature["manifest"]["files"]}
                contract = feature_contract(feature, bundle)
                if not selected <= available or any(
                    contract[key] != development[key]
                    for key in ("dimensions", "dtype", "encoderId")
                ):
                    continue
                if representation_kind(contract) != representation_kind(development):
                    continue
                # Several bundle revisions may verify the same feature inventory.
                # They do not make the underlying feature choice ambiguous.
                previous = candidates.get(feature["id"])
                if previous is None or bundle["createdAt"] > previous["createdAt"]:
                    candidates[feature["id"]] = bundle
            except StorageError:
                continue
        if not candidates:
            raise StorageError(
                "No current feature bundle covers every selected test slide with this model's representation, encoder, dimensions, and dtype. Extract and verify the missing test features, then select a bundle here.",
                "EVALUATION_FEATURE_BUNDLE_REQUIRED",
                409,
            )
        if len(candidates) > 1:
            raise StorageError(
                "More than one compatible test feature inventory is available. Select the test feature bundle to evaluate.",
                "EVALUATION_FEATURE_BUNDLE_AMBIGUOUS",
                409,
            )
        return next(iter(candidates.values()))["id"]

    def _review_cohort(self, selection, model, test):
        if test["spec"].get("target"):
            target = TargetSpec.model_validate(test["spec"]["target"])
            expected = TargetSpec.model_validate(model["target"])
            if any(
                getattr(target, key) != getattr(expected, key)
                for key in ("task", "unit", "classes", "positiveClass")
            ):
                raise StorageError(
                    "Test targets must match the selected model's task, prediction unit, class order, and positive class.",
                    "TARGET_CONTRACT_MISMATCH",
                    409,
                )
        inference = selection.inference or InferenceSettings.model_validate(
            test["spec"].get("inference", {})
        )
        if inference.patientAggregation == "predictor" or (
            selection.inference is None and not test["spec"].get("protocolId")
        ):
            inference = inference.model_copy(
                update={"patientAggregation": model["patientAggregation"]}
            )
        frozen_threshold = model.get("recipe", {}).get("decisionThreshold")
        if inference.decisionThreshold == "predictor" or (
            frozen_threshold is not None and selection.inference is None
        ):
            inference = inference.model_copy(
                update={"decisionThreshold": frozen_threshold if frozen_threshold is not None else 0.5}
            )
        bundle_id = self._test_bundle(selection, model, test, inference)
        spec = EvaluationSpec.model_validate(
            {
                **test["spec"],
                "protocolId": model["inputs"]["protocol"]["id"],
                "developmentFeatureBundleId": model["inputs"]["features"]["bundle"]["id"],
                "featureBundleId": bundle_id,
                "inference": inference.model_dump(),
                "patientIdentifiers": selection.patientIdentifiers
                or test["spec"]["patientIdentifiers"],
            }
        )
        reviewed, _guards = self.cohorts._prepare_bound(spec)
        errors = [item for item in reviewed["findings"] if item["severity"] == "error"]
        if errors:
            error = StorageError(errors[0]["message"], errors[0]["code"], 409)
            error.findings = reviewed["findings"]
            raise error
        if reviewed["memberships"] != test["memberships"]:
            raise StorageError(
                "The test dataset or target membership changed. Review and freeze the cohort again.",
                "EVALUATION_COHORT_STALE",
                409,
            )
        return {**test, **reviewed}

    def _prepare(self, selection):
        predictor = self.predictors.get(selection.predictorId)
        cohort = self.cohorts.get(selection.cohortId)
        if not cohort["current"]:
            raise StorageError(
                "The test cohort or its feature verification changed. Review its inputs first.",
                "EVALUATION_COHORT_STALE",
                409,
            )
        model, test = predictor["manifest"], cohort["manifest"]
        reviewed_at_evaluation = (
            not test["spec"].get("protocolId")
            or selection.featureBundleId is not None
            or selection.inference is not None
            or selection.patientIdentifiers is not None
        )
        if reviewed_at_evaluation:
            test = self._review_cohort(selection, model, test)
        if (
            test["bindings"]["protocol"] != model["inputs"]["protocol"]
            or test["bindings"]["development"]["bundle"] != model["inputs"]["features"]["bundle"]
            or test["bindings"]["development"]["feature"] != model["inputs"]["features"]["feature"]
            or TargetSpec.model_validate(test["target"]).model_dump()
            != TargetSpec.model_validate(model["target"]).model_dump()
        ):
            raise StorageError(
                "The test cohort must use this predictor's exact development protocol, target encoding, and development feature bundle.",
                "EVALUATION_PREDICTOR_MISMATCH",
                409,
            )
        if test["overlap"]["slideIds"] or test["overlap"]["patientIds"]:
            raise StorageError(
                "Test membership overlaps this predictor's development data.",
                "EVALUATION_DEVELOPMENT_OVERLAP",
                409,
            )
        evaluation_binding = test["bindings"]["evaluation"]
        feature = self.store.get_configuration(evaluation_binding["feature"]["id"])
        bundle = self.store.get_configuration(evaluation_binding["bundle"]["id"])
        features = feature_contract(feature, bundle)
        development = model["inputs"]["features"]
        if (
            not features["sourceContentHash"]
            or features["dimensions"] != development["dimensions"]
            or features["dtype"] != development["dtype"]
            or features["encoderId"] != development["encoderId"]
            or representation_kind(features) != representation_kind(development)
        ):
            raise StorageError(
                "Test features must preserve the verified representation, encoder, dimensions, and dtype used by the predictor.",
                "EVALUATION_FEATURE_CONTRACT_MISMATCH",
                409,
            )
        if features["feature"] != development["feature"]:
            if not features["encoderId"] or not development["encoderId"]:
                raise StorageError(
                    "Distinct feature inventories require explicit matching encoder identities.",
                    "EVALUATION_ENCODER_UNVERIFIABLE",
                    409,
                )
            left, right = development.get("extraction"), features.get("extraction")
            if bool(left) != bool(right):
                raise StorageError(
                    "Extraction provenance is available for only one feature source. Attach matching extraction provenance to both sources.",
                    "EVALUATION_EXTRACTION_UNVERIFIABLE",
                    409,
                )
            if left and right:
                keys = (
                    "patch_encoder",
                    "patch_encoder_ckpt_path",
                    "patch_encoder_img_size",
                    "mag",
                    "patch_size",
                    "overlap",
                    "custom_mpp_keys",
                )
                if (representation_kind(features) == "slide"
                        or model.get("recipe", {}).get("analysis") is not None):
                    keys += ("slide_encoder",)
                if model.get("recipe", {}).get("analysis") is not None:
                    keys += (
                        "reader_type", "segmenter", "seg_conf_thresh", "remove_holes",
                        "remove_artifacts", "remove_penmarks", "min_tissue_proportion",
                    )
                left_options, right_options = left["spec"]["options"], right["spec"]["options"]
                if any(left_options.get(key) != right_options.get(key) for key in keys):
                    raise StorageError(
                        "Development and test extraction settings differ.",
                        "EVALUATION_EXTRACTION_MISMATCH",
                        409,
                    )
        if test["spec"]["inference"]["patientAggregation"] != model["patientAggregation"]:
            raise StorageError(
                "Patient aggregation must preserve the frozen predictor's scoring rule.",
                "EVALUATION_AGGREGATION_MISMATCH",
                409,
            )
        frozen_threshold = model.get("recipe", {}).get("decisionThreshold")
        if frozen_threshold is not None and (
            test["spec"]["inference"]["decisionThreshold"] != frozen_threshold
        ):
            raise StorageError(
                "Use the decision threshold frozen in the training recipe for every external cohort.",
                "EVALUATION_THRESHOLD_MISMATCH", 409,
            )
        self.predictors.verify_checkpoints(predictor)
        clinical_contract = self._clinical_contract(model, test)
        return {
            "kind": "model-evaluation",
            "schemaVersion": 1,
            "datasetId": test["datasetId"],
            **(
                {"datasetIds": test["spec"]["datasetIds"]} if test["spec"].get("datasetIds") else {}
            ),
            "name": selection.name,
            "selection": selection.model_dump(exclude_none=True),
            "experimentId": model["experimentId"],
            "predictorId": predictor["id"],
            "predictor": reference(predictor),
            "cohortId": cohort["id"],
            "cohort": reference(cohort),
            "target": model["target"],
            **({"clinical": clinical_contract} if clinical_contract else {}),
            "inference": test["spec"]["inference"],
            "analysis": model.get("recipe", {}).get("analysis")
            or PatientAnalysisSettings().model_dump(),
            "bagPolicy": {"evalBagSize": model.get("recipe", {}).get("evalBagSize"),
                          "trainingSeed": model["trainingSeed"]},
            "features": features,
            "summary": test["summary"],
            "status": "planned",
            "results": None,
            "executionEnabled": True,
            "executionNote": EXECUTION_NOTE,
            **(
                {
                    "coverage": test["coverage"],
                    "overlap": test["overlap"],
                    "findings": test["findings"],
                }
                if reviewed_at_evaluation
                else {}
            ),
        }

    def _clinical_contract(self, model, test):
        from histopilot.application.clinical_inputs import frozen_clinical_values
        from histopilot.clinical_features import clinical_fields, clinical_rows

        recipe = model.get("recipe", {})
        fields = clinical_fields(recipe)
        if not fields:
            return None
        values = frozen_clinical_values(
            self.store, self.filesystem, test["spec"].get("datasetIds") or [test["datasetId"]],
            test["memberships"], fields,
        )
        try:
            clinical_rows(test["memberships"], values, fields)
        except ValueError as error:
            raise StorageError(str(error), "CLINICAL_VALUES_INVALID", 422) from error
        return {"inputMode": recipe["inputMode"], "fields": fields, "valuesSha256": _hash(values)}

    def preview(self, selection):
        try:
            manifest = self._prepare(selection)
            return {
                "canSave": True,
                "previewHash": _hash(manifest),
                "manifest": manifest,
                "findings": manifest.get("findings", []),
                "executionEnabled": True,
                "executionNote": EXECUTION_NOTE,
            }
        except StorageError as error:
            return {
                "canSave": False,
                "previewHash": None,
                "manifest": None,
                "findings": getattr(error, "findings", [finding(error.code, str(error))]),
                "executionEnabled": True,
                "executionNote": EXECUTION_NOTE,
            }

    def save(self, request):
        selection = EvaluationRunSelection.model_validate(
            request.model_dump(include=set(EvaluationRunSelection.model_fields))
        )
        with lifecycle_guard(self.store.folder):
            prior = self.store.configuration_publication(request.operationId)
            if prior:
                manifest = prior["manifest"]
                if (
                    manifest.get("kind") != "model-evaluation"
                    or manifest.get("selection") != selection.model_dump(exclude_none=True)
                    or manifest.get("previewHash") != request.previewHash
                ):
                    raise StorageError(
                        "This operation belongs to another evaluation.", "OPERATION_CONFLICT", 409
                    )
                return lifecycle_document(self.store, prior)
            manifest = self._prepare(selection)
            if _hash(manifest) != request.previewHash:
                raise StorageError("Evaluation inputs changed. Review again.", "PREVIEW_STALE", 409)
            published = self.store.publish_configuration(
                manifest={**manifest, "previewHash": request.previewHash},
                operation_id=request.operationId,
            )
            return lifecycle_document(self.store, published)

    def execution(self, identity):
        self.get(identity)
        return self.jobs.status(identity)

    def _execution_plan(self, identity):
        document = self.get(identity)
        manifest = document["manifest"]
        selection = EvaluationRunSelection.model_validate(manifest["selection"])
        if "coverage" in manifest:
            # Auto selection is resolved by the saved review. New inventories
            # must not silently replace it or make an existing plan ambiguous.
            selection = selection.model_copy(
                update={
                    "featureBundleId": manifest["features"]["bundle"]["id"],
                    "inference": InferenceSettings.model_validate(manifest["inference"]),
                }
            )
        reviewed = self._prepare(selection)
        for key in ("predictor", "cohort", "target", "features", "inference"):
            if reviewed[key] != manifest[key]:
                raise StorageError(
                    "Evaluation inputs changed. Create and review a new evaluation.",
                    "EVALUATION_INPUTS_CHANGED",
                )
        if "analysis" in manifest and reviewed["analysis"] != manifest["analysis"]:
            raise StorageError("The frozen analysis policy changed.", "EVALUATION_INPUTS_CHANGED", 409)
        if "bagPolicy" in manifest and reviewed["bagPolicy"] != manifest["bagPolicy"]:
            raise StorageError("The frozen evaluation bag policy changed.", "EVALUATION_INPUTS_CHANGED", 409)
        predictor = self.predictors.get(manifest["predictorId"])
        cohort = self.cohorts.get(manifest["cohortId"])
        feature = self.store.get_configuration(manifest["features"]["feature"]["id"])
        bundle = self.store.get_configuration(manifest["features"]["bundle"]["id"])
        memberships = cohort["manifest"]["memberships"]
        clinical_values = {}
        if manifest.get("clinical"):
            from histopilot.application.clinical_inputs import frozen_clinical_values

            current_clinical = self._clinical_contract(predictor["manifest"], cohort["manifest"])
            if current_clinical != manifest["clinical"]:
                raise StorageError("Clinical schema or frozen values changed.", "CLINICAL_INPUTS_CHANGED", 409)
            clinical_values = frozen_clinical_values(
                self.store, self.filesystem,
                cohort["manifest"]["spec"].get("datasetIds") or [cohort["manifest"]["datasetId"]],
                memberships, manifest["clinical"]["fields"],
            )
        selected = {row["slideId"] for row in memberships}
        files = {
            row["slideId"]: row
            for row in feature["manifest"]["files"]
            if row["slideId"] in selected
        }
        if set(files) != selected:
            raise StorageError(
                "The exact test feature membership is incomplete.", "EVALUATION_COVERAGE_CHANGED"
            )
        inference = manifest["inference"]
        pack_path, pack_stamps = None, None
        if inference["loadingPolicy"] == "packed":
            resolved = self.cohorts.bundles.packing.resolve_artifact(
                feature["id"], inference["packArtifactId"]
            )
            if not resolved["current"]:
                raise StorageError("The selected test pack changed.", "EVALUATION_PACK_CHANGED")
            pack_path = resolved["artifact"]["outputPath"]
            pack_stamps = _layout(Path(pack_path))["packStamps"]
        if document["execution"]["status"] != "not_started":
            # Device selection belongs to the first launch. Resume and request
            # replay must preserve it even if CUDA availability has changed.
            saved = read_json(self.jobs.folder(identity) / "plan.json")
            if _hash(saved) != document["execution"].get("planHash"):
                raise StorageError("The saved execution plan changed.", "COMPUTE_PLAN_CHANGED")
            defaults = ResourcePolicy().model_dump()
            resources = {
                # Compute validation serializes numeric defaults such as 8 as
                # 8.0. Restore the original request representation for retries.
                key: defaults[key] if key in defaults and value == defaults[key] else value
                for key, value in saved["resources"].items()
            }
        else:
            runtime = training_runtime()
            use_cuda = inference["device"] == "cuda" or (
                inference["device"] == "auto" and runtime.get("cudaAvailable")
            )
            resources = ResourcePolicy(
                gpuIds=[0] if use_cuda else [], dataLoaderWorkers=inference["numWorkers"]
            ).model_dump()
        return {
            "kind": "evaluation",
            "runId": identity,
            "method": predictor["manifest"].get("method", "ensemble"),
            **(
                {"aggregation": "mean_logit"}
                if predictor["manifest"].get("aggregation") == "mean_logit"
                else {}
            ),
            "target": manifest["target"],
            "inference": inference,
            **({"analysis": manifest["analysis"]} if manifest.get("analysis") else {}),
            **({"bagPolicy": manifest["bagPolicy"]} if "bagPolicy" in manifest else {}),
            "resources": resources,
            "checkpoints": predictor["manifest"]["checkpoints"],
            "references": [reference(value) for value in (predictor, cohort, feature, bundle)],
            "data": {
                **({"clinicalValues": clinical_values,
                    "inputMode": manifest["clinical"]["inputMode"],
                    "clinicalFields": manifest["clinical"]["fields"]} if clinical_values else {}),
                "memberships": memberships,
                "featureDim": manifest["features"]["dimensions"],
                "featureFiles": files,
                "sourceStamps": {row["path"]: row for row in files.values()},
                "loadingPolicy": "mmap" if pack_path else "native",
                "packPath": pack_path,
                "packStamps": pack_stamps,
            },
        }

    def launch(self, identity, operation_id, *, resume=False):
        with lifecycle_guard(self.store.folder):
            replay = self.jobs.replay_launch(identity, operation_id, resume=resume, record_kind="model-evaluation")
            if replay is not None:
                return replay
            plan = self._execution_plan(identity)
            return self.jobs.launch(identity, plan, operation_id, resume=resume)

    def cancel(self, identity, operation_id):
        self.get(identity)
        return self.jobs.cancel(identity, operation_id)

    def artifact(self, identity, filename):
        if filename not in {
            "predictions.json",
            "metrics.json",
            "slide-predictions.csv",
            "patient-predictions.csv",
        }:
            raise StorageError(
                "Evaluation artifact not found.", "EVALUATION_ARTIFACT_NOT_FOUND", 404
            )
        execution = self.execution(identity)
        if execution["status"] != "completed":
            raise StorageError(
                "Evaluation results are available after the job finishes.",
                "EVALUATION_NOT_COMPLETED",
            )
        expected = execution["result"]["artifacts"].get(filename)
        path = self.jobs.folder(identity) / filename
        if not expected or expected["path"] != str(path):
            raise StorageError("Evaluation result provenance changed.", "EVALUATION_RESULT_CHANGED")
        content = ScientificStore._read_file(path, 64 * 1024 * 1024)
        if (
            len(content) != expected["bytes"]
            or hashlib.sha256(content).hexdigest() != expected["sha256"]
        ):
            raise StorageError("The evaluation output changed.", "EVALUATION_RESULT_CHANGED")
        return content
