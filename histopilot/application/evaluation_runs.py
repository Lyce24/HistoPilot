"""Reviewed predictor/cohort plans with durable refit and ensemble inference."""

import copy
import hashlib
import json
import threading
from collections import OrderedDict
from pathlib import Path

from histopilot.adapters.native.runtime import training_runtime
from histopilot.application.compute_jobs import ComputeJobService
from histopilot.application.evaluations import EvaluationService, patient_overlap_allowed
from histopilot.application.predictors import (
    PredictorService,
    feature_contract,
    finding,
    lifecycle_document,
    reference,
)
from histopilot.application.run_evidence import (
    LABELS_FROM_COHORT,
    development_flag,
    development_patients,
    join_labels,
    join_reference,
    labels_withheld,
    patient_evidence,
    prediction_table,
    strip_labels,
)
from histopilot.domain.features import representation_kind
from histopilot.schemas.analysis import PatientAnalysisSettings
from histopilot.schemas.development import ResourcePolicy
from histopilot.schemas.evaluations import (
    STORED,
    EvaluationSpec,
    InferenceSettings,
    is_inference_purpose,
)
from histopilot.schemas.predictors import EvaluationRunSelection
from histopilot.schemas.protocols import TargetSpec
from histopilot.storage.io import (
    content_hash,
    read_file_bounded,
    read_json_bounded,
    write_json_atomic,
)
from histopilot.storage.lifecycle import lifecycle_guard
from histopilot.storage.pack_import import pack_layout
from histopilot.storage.project_lock import StorageError

EXECUTION_NOTE = (
    "Predict the frozen labeled cohort with the recorded bag, aggregation and decision-threshold "
    "settings, without reading its labels. The control service then scores the labeled "
    "records against the cohort's frozen labels; unlabeled records keep their predictions."
)
INFERENCE_NOTE = (
    "Predict the frozen unlabeled cohort with the recorded bag, aggregation and "
    "decision-threshold settings. No labels are read and no performance metrics are "
    "computed. Slides from development patients are flagged in every analysis and export."
)


# Service-scored metrics per (project, run, scoring key). Every input is immutable, so an
# entry never goes stale; the bound only limits memory. The same documents are kept on disk
# beside each run's outputs, so a restart reads them instead of re-running bootstraps.
_SCORES = OrderedDict()
_SCORE_LIMIT = 256
# Requests run on a thread pool; the lock keeps look-up, reorder and eviction consistent.
_SCORES_LOCK = threading.Lock()
SCORES_FILE = "service-metrics.json"
# Scores against reference standards, one file per scoring key.
REFERENCE_SCORES = "reference-scores"
REFERENCE_SCORED_BY = "control_service_reference_join_v1"
# Recalibration reports kept beside a run, and the rule that fits and applies their maps.
RECALIBRATION = "recalibration"
RECALIBRATED_BY = "control_service_development_oof_recalibration_v1"


def _remembered(memo):
    """A copy of a scored document kept in memory, or None."""
    with _SCORES_LOCK:
        cached = _SCORES.get(memo)
        if cached is not None:
            _SCORES.move_to_end(memo)
    return None if cached is None else copy.deepcopy(cached)


def _remember(memo, scored) -> None:
    with _SCORES_LOCK:
        _SCORES[memo] = scored
        while len(_SCORES) > _SCORE_LIMIT:
            _SCORES.popitem(last=False)


def _kept_scores(path, key):
    """A kept scored document for exactly ``key``, or None to score again."""
    try:
        kept = read_json_bounded(path)
    except (OSError, ValueError, StorageError):
        return None
    metrics = kept.get("metrics") if isinstance(kept, dict) else None
    if (
        kept.get("key") != key
        or not isinstance(metrics, dict)
        or kept.get("sha256") != content_hash(metrics)
    ):
        return None
    return metrics


def run_purpose(manifest):
    """Inference and legacy review runs are prediction-only; everything else is evaluation."""
    return "inference" if is_inference_purpose(manifest.get("purpose")) else "evaluation"


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
                    "execution": self.scored_execution(
                        item,
                        self.jobs.status(item["id"], include_inactive=include_inactive),
                        analysis=False,
                    ),
                }
                for item in self.store.list_configurations(
                    "model-evaluation", include_inactive=include_inactive
                )
            ],
            "executionEnabled": True,
            "executionNote": EXECUTION_NOTE,
        }

    def get(self, identity):
        record = self.record(identity)
        return {**record, "execution": self.scored_execution(record, record["execution"])}

    def record(self, identity):
        """The run with its worker's execution status, without service scoring."""
        document = self._document(identity)
        return {**lifecycle_document(self.store, document), "execution": self.jobs.status(identity)}

    def _document(self, identity):
        document = self.store.get_configuration(identity)
        if document["manifest"].get("kind") != "model-evaluation":
            raise StorageError("Evaluation record not found.", "MODEL_EVALUATION_NOT_FOUND", 404)
        return document

    def scored_execution(self, document, execution, *, analysis=True):
        """A label-blind run's completed execution, with the metrics the service scored.

        Earlier labeled runs carry their worker's metrics in the result already. A run whose
        evidence cannot be scored keeps its predictions and reports why in ``metricsError``.
        """
        if (
            not labels_withheld(document["manifest"])
            or (execution or {}).get("status") != "completed"
        ):
            return execution
        result = execution.get("result") or {}
        try:
            metrics = self.metrics(document, result, analysis=analysis)
        except StorageError as error:
            return {
                **execution,
                "result": {**result, "metricsError": {"code": error.code, "message": str(error)}},
            }
        return {**execution, "result": {**result, "metrics": metrics}}

    def metrics(self, document, result, *, analysis=True):
        """Score a completed label-blind run against its frozen cohort labels.

        The scored document is kept beside the run's outputs under a key of everything that
        determines it, so its bootstrap runs once per run, not once per service start.
        ``analysis=False`` serves list views: until a full document is kept, it returns the
        point metrics without the bootstrap (``analysisPending``) and keeps nothing.
        """
        from histopilot.application.run_metrics import SCORED_BY

        manifest = document["manifest"]
        recorded = (result.get("artifacts") or {}).get("predictions.json", {}).get("sha256")
        cohort = self.store.get_configuration(manifest["cohortId"])
        predictor = self.store.get_configuration(manifest["predictorId"])
        if manifest["cohort"] != reference(cohort) or manifest["predictor"] != reference(predictor):
            raise StorageError(
                "The evaluation no longer matches its frozen predictor and cohort.",
                "EVALUATION_EVIDENCE_CHANGED",
                409,
            )
        key = {"method": SCORED_BY, "predictionsSha256": recorded, "cohort": reference(cohort)}
        bootstrap = "analysis" in manifest and manifest.get("splitUnit") != "slide"
        return self._kept(
            document,
            key,
            self.jobs.folder(document["id"]) / SCORES_FILE,
            lambda: self._score(document, cohort, predictor, key),
            quick=(lambda: self._score(document, cohort, predictor, key, bootstrap=False))
            if not analysis and bootstrap
            else None,
        )

    def _kept(self, document, key, path, score, *, quick=None):
        """A scored document for ``key``: from memory, from disk, or scored and kept.

        ``quick`` serves a view that must not wait for a bootstrap: it answers only while
        no full document is kept, and its answer is kept in memory only, so lists that poll
        do not score the run again.
        """
        memo = (str(self.store.folder), document["id"], content_hash(key))
        cached = _remembered(memo)
        if cached is not None:
            return cached
        kept = _kept_scores(path, key)
        if kept is None:
            if quick is not None:
                answer = _remembered((*memo, "quick"))
                if answer is None:
                    answer = quick()
                    _remember((*memo, "quick"), answer)
                return copy.deepcopy(answer)
            kept = score()
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                write_json_atomic(path, {"key": key, "metrics": kept, "sha256": content_hash(kept)})
            except (OSError, StorageError):
                pass  # A read-only project still scores; it only cannot keep the result.
        _remember(memo, kept)
        return copy.deepcopy(kept)

    def _completed(self, identity):
        """A completed run with its frozen predictor, cohort and verified prediction bytes."""
        document = self.record(identity)
        manifest = document["manifest"]
        if (document["execution"] or {}).get("status") != "completed":
            raise StorageError(
                "Only a completed run has predictions to score.", "RUN_NOT_COMPLETED", 409
            )
        cohort = self.store.get_configuration(manifest["cohortId"])
        predictor = self.store.get_configuration(manifest["predictorId"])
        if manifest["cohort"] != reference(cohort) or manifest["predictor"] != reference(predictor):
            raise StorageError(
                "The evaluation no longer matches its frozen predictor and cohort.",
                "EVALUATION_EVIDENCE_CHANGED",
                409,
            )
        return document, cohort, predictor, self.artifact(identity, "predictions.json")

    def _predictions(self, manifest, content):
        from histopilot.application.clinical import validate_records

        predictions = json.loads(content)
        classes = manifest["target"]["classes"]
        if predictions.get("classOrder") != classes:
            raise ValueError("Saved class order differs from the evaluation.")
        validate_records(predictions.get("records"), classes)
        return predictions

    def reference_metrics(self, identity, reference_id):
        """Score a completed run against a reference standard attached to its cohort.

        The same functions and document shape as the cohort's own scores, from the same
        verified predictions; nothing is predicted again. Kept on disk by scoring key.
        """
        from histopilot.application.references import ReferenceService

        document, cohort, predictor, content = self._completed(identity)
        manifest = document["manifest"]
        standard = ReferenceService(self.store, self.filesystem).for_run(manifest, reference_id)
        if "analysis" not in manifest and manifest.get("splitUnit") != "slide":
            # Runs on unlabeled cohorts record no bootstrap settings. Score with those a run
            # of the same predictor on a labeled cohort would record.
            manifest = {
                **manifest,
                "analysis": predictor["manifest"].get("recipe", {}).get("analysis")
                or PatientAnalysisSettings().model_dump(),
            }
        key = {
            "method": REFERENCE_SCORED_BY,
            "predictionsSha256": hashlib.sha256(content).hexdigest(),
            "reference": reference(standard),
            "analysis": manifest.get("analysis"),
        }

        def score():
            from histopilot.application.run_metrics import run_metrics

            try:
                predictions = self._predictions(manifest, content)
                slides, conflicting = join_reference(
                    manifest, standard["manifest"], predictions["records"]
                )
                metrics = run_metrics(
                    manifest, slides, ensemble_aggregation=predictor["manifest"].get("aggregation")
                )
            except (ValueError, KeyError, TypeError, OverflowError, RecursionError) as error:
                raise StorageError(
                    str(error) or "The saved prediction evidence is invalid.",
                    "EVALUATION_EVIDENCE_INVALID",
                    409,
                ) from error
            return {
                **metrics,
                "scoredBy": key,
                "reference": {"id": standard["id"], "name": standard["manifest"]["name"]},
                **({"conflictingPatients": conflicting} if conflicting else {}),
            }

        folder = self.jobs.folder(document["id"]) / REFERENCE_SCORES
        return self._kept(document, key, folder / f"{content_hash(key)}.json", score)

    def reference_download(self, identity, reference_id, filename):
        """The metrics or a scored table of a run against a reference standard, in the form
        the run's own downloads take, with the reference's labels joined."""
        from histopilot.application.references import ReferenceService

        if filename == "metrics.json":
            metrics = self.reference_metrics(identity, reference_id)
            return (json.dumps(metrics, indent=2, sort_keys=True) + "\n").encode("utf-8")
        if filename not in {"slide-predictions.csv", "patient-predictions.csv"}:
            raise StorageError(
                "Evaluation artifact not found.", "EVALUATION_ARTIFACT_NOT_FOUND", 404
            )
        document, _cohort, _predictor, content = self._completed(identity)
        manifest = document["manifest"]
        if filename == "patient-predictions.csv" and manifest.get("splitUnit") == "slide":
            raise StorageError(
                "Slide-split evaluations score each slide on its own; they have no patient table.",
                "EVALUATION_ARTIFACT_NOT_FOUND",
                404,
            )
        standard = ReferenceService(self.store, self.filesystem).for_run(manifest, reference_id)
        try:
            predictions = self._predictions(manifest, content)
            slides, _conflicting = join_reference(
                manifest, standard["manifest"], predictions["records"]
            )
            return prediction_table(
                manifest,
                slides,
                None,
                patient=filename == "patient-predictions.csv",
                rebuild=True,
            )
        except (ValueError, KeyError, TypeError, OverflowError, RecursionError) as error:
            raise StorageError(
                str(error) or "The saved prediction evidence is invalid.",
                "EVALUATION_EVIDENCE_INVALID",
                409,
            ) from error

    def label_sources(self, document, cohort, records):
        """Every label source of a run's cohort, each as slides joined to its labels.

        The cohort's own labels come first when it has any; then each usable reference
        standard with the run's classes. A source that cannot label these predictions is
        listed with the reason instead.
        """
        from histopilot.application.references import ReferenceService

        manifest = document["manifest"]
        sources = []
        if run_purpose(manifest) != "inference":
            try:
                sources.append(
                    {
                        "id": "cohort",
                        "name": "Cohort labels",
                        "kind": "cohort",
                        "slides": join_labels(manifest, cohort["manifest"], records),
                    }
                )
            except ValueError as error:
                sources.append(
                    {
                        "id": "cohort",
                        "name": "Cohort labels",
                        "kind": "cohort",
                        "reason": str(error),
                    }
                )
        references = ReferenceService(self.store, self.filesystem)
        listed = references.list(cohort_id=cohort["id"])["items"]
        for item in sorted(listed, key=lambda item: (item["manifest"]["name"], item["id"])):
            if item["lifecycleState"] == "trashed" or sorted(item["manifest"]["classes"]) != sorted(
                manifest["target"]["classes"]
            ):
                continue
            entry = {"id": item["id"], "name": item["manifest"]["name"], "kind": "reference"}
            try:
                standard = references.for_run(manifest, item["id"])
                entry["slides"], entry["conflictingPatients"] = join_reference(
                    manifest, standard["manifest"], records
                )
            except (ValueError, StorageError) as error:
                entry["reason"] = str(error)
            sources.append(entry)
        return sources

    def recalibration(self, identity, unit="selected", reference_id=None):
        """Calibration of a run's probabilities as predicted and after recalibration.

        The map is fitted on the predictor's out-of-fold development predictions (a seed
        ensemble averages its seed groups' OOF predictions as it averages their models) at
        the scoring unit, then applied to this run: Platt scaling for binary targets,
        temperature scaling for multiclass ones. It is never fitted on the cohort it is
        scored on. Decisions and ranking metrics keep the original probabilities. Labels
        come from the cohort or a reference standard; development patients are left out.
        """
        from histopilot.application.references import ReferenceService

        document, cohort, predictor, content = self._completed(identity)
        manifest = document["manifest"]
        target = manifest["target"]
        classes = target["classes"]
        actual = target["unit"] if unit == "selected" else unit
        if actual == "patient" and manifest.get("splitUnit") == "slide":
            raise StorageError(
                "Patient analysis is disabled for slide-level experiments.",
                "PATIENT_ANALYSIS_DISABLED",
                409,
            )
        standard = (
            ReferenceService(self.store, self.filesystem).for_run(manifest, reference_id)
            if reference_id
            else None
        )
        if standard is None and run_purpose(manifest) == "inference":
            raise StorageError(
                "Recalibration is checked against labels. Add a reference standard to this "
                "run's cohort first.",
                "RECALIBRATION_REQUIRES_LABELS",
                409,
            )
        source = predictor["manifest"]
        groups = source.get("seedGroups") or (
            [{"trainingSeed": source["trainingSeed"], "splitSeed": source["splitSeed"]}]
            if source.get("batchId") and "trainingSeed" in source
            else []
        )
        if not groups:
            raise StorageError(
                "This predictor records no development predictions to fit a map on.",
                "RECALIBRATION_UNAVAILABLE",
                409,
            )
        key = {
            "method": RECALIBRATED_BY,
            "predictionsSha256": hashlib.sha256(content).hexdigest(),
            "reference": reference(standard) if standard else None,
            "unit": actual,
            "predictor": manifest["predictor"],
        }

        def fit():
            from histopilot import cv_summary
            from histopilot.application.training_exports import verified_oof
            from histopilot.calibration import (
                apply_platt,
                apply_temperature,
                calibration_metrics,
                fit_platt,
                fit_temperature,
            )
            from histopilot.scoring import patient_predictions

            aggregation = (
                "mean_logits"
                if manifest["inference"]["patientAggregation"] == "mean_logits"
                else "mean"
            )

            def units(slides):
                labeled = slides if actual == "slide" else patient_predictions(slides, aggregation)
                return [row for row in labeled if row["labelIndex"] is not None]

            try:
                predictions = self._predictions(manifest, content)
                if standard:
                    slides, _conflicting = join_reference(
                        manifest, standard["manifest"], predictions["records"]
                    )
                else:
                    slides = join_labels(manifest, cohort["manifest"], predictions["records"])
                shared = development_patients(manifest)
                cohort_units = units(
                    [row for row in slides if development_flag(row, shared) is not True]
                )
                oofs = [
                    verified_oof(
                        self.store,
                        source["batchId"],
                        source["candidateId"],
                        group["trainingSeed"],
                        group["splitSeed"],
                    )
                    for group in groups
                ]
                if any(item["target"]["classes"] != classes for item in oofs):
                    raise ValueError("Development predictions use another class order.")
                development = (
                    cv_summary.seed_ensemble(
                        [item["slides"] for item in oofs],
                        "mean_logit"
                        if source.get("aggregation") == "mean_logit"
                        else "mean_probability",
                    )
                    if len(oofs) > 1
                    else oofs[0]["slides"]
                )
                if development is None:
                    raise ValueError("The seed groups assessed different development units.")
                development_units = units(development)
            except ValueError as error:
                raise StorageError(
                    str(error) or "The saved prediction evidence is invalid.",
                    "RECALIBRATION_UNAVAILABLE",
                    409,
                ) from error
            if not cohort_units:
                raise StorageError(
                    "No labeled units remain to check calibration on.",
                    "RECALIBRATION_UNAVAILABLE",
                    409,
                )
            development_probabilities = [row["probabilities"] for row in development_units]
            development_labels = [row["labelIndex"] for row in development_units]
            probabilities = [row["probabilities"] for row in cohort_units]
            labels = [row["labelIndex"] for row in cohort_units]
            binary = target["task"] == "binary_classification"
            positive = classes.index(target["positiveClass"]) if binary else None
            try:
                if binary:
                    parameters = fit_platt(
                        [row[positive] for row in development_probabilities],
                        [label == positive for label in development_labels],
                    )

                    def recalibrate(values):
                        risk = apply_platt([row[positive] for row in values], **parameters)
                        other = 1 - positive
                        rows = [[0.0, 0.0] for _ in values]
                        for row, value in zip(rows, risk, strict=True):
                            row[positive], row[other] = float(value), float(1 - value)
                        return rows
                else:
                    parameters = fit_temperature(development_probabilities, development_labels)

                    def recalibrate(values):
                        return apply_temperature(values, **parameters).tolist()
            except ValueError as error:
                raise StorageError(str(error), "RECALIBRATION_UNAVAILABLE", 409) from error
            return {
                "evaluationId": document["id"],
                "unit": actual,
                "classOrder": classes,
                "positiveClass": target.get("positiveClass") if binary else None,
                "method": "platt" if binary else "temperature",
                "parameters": parameters,
                "development": {
                    "units": len(development_units),
                    "seedGroups": len(groups),
                    "original": calibration_metrics(
                        development_probabilities, development_labels, positive
                    ),
                    # In-sample: the map is fitted on these same units.
                    "recalibrated": calibration_metrics(
                        recalibrate(development_probabilities), development_labels, positive
                    ),
                },
                "cohort": {
                    "units": len(cohort_units),
                    "original": calibration_metrics(probabilities, labels, positive),
                    "recalibrated": calibration_metrics(
                        recalibrate(probabilities), labels, positive
                    ),
                },
                "developmentExcluded": sum(development_flag(row, shared) is True for row in slides),
                "reference": (
                    {"id": standard["id"], "name": standard["manifest"]["name"]}
                    if standard
                    else None
                ),
                "recalibratedBy": key,
                "developmentSources": [
                    {
                        "batchId": source["batchId"],
                        "candidateId": source["candidateId"],
                        "trainingSeed": group["trainingSeed"],
                        "splitSeed": group["splitSeed"],
                        "analysisInputHash": item["analysisInputHash"],
                    }
                    for group, item in zip(groups, oofs, strict=True)
                ],
            }

        path = self.jobs.folder(document["id"]) / RECALIBRATION / f"{content_hash(key)}.json"
        return self._kept(document, key, path, fit)

    def agreement(self, identity, unit):
        """How often this run's decisions and every label source of its cohort agree.

        Pairs cover the units both sides label. Units from development patients are left out,
        as they are from the metrics. With three or more classes, ``weightedKappa`` reads the
        class order as an ordered scale.
        """
        from histopilot.inference_summary import agreement, predicted_index
        from histopilot.scoring import patient_predictions

        document, cohort, _predictor, content = self._completed(identity)
        manifest = document["manifest"]
        target = manifest["target"]
        classes = target["classes"]
        actual = target["unit"] if unit == "selected" else unit
        if actual == "patient" and manifest.get("splitUnit") == "slide":
            raise StorageError(
                "Patient analysis is disabled for slide-level experiments.",
                "PATIENT_ANALYSIS_DISABLED",
                409,
            )
        try:
            predictions = self._predictions(manifest, content)
        except (ValueError, KeyError, TypeError, OverflowError, RecursionError) as error:
            raise StorageError(
                str(error) or "The saved prediction evidence is invalid.",
                "EVALUATION_EVIDENCE_INVALID",
                409,
            ) from error
        records = predictions["records"]
        shared = development_patients(manifest)
        independent = {
            row["slideId"] for row in records if development_flag(row, shared) is not True
        }
        aggregation = (
            "mean_logits"
            if manifest["inference"]["patientAggregation"] == "mean_logits"
            else "mean"
        )
        threshold = manifest["inference"]["decisionThreshold"]

        def units(slides):
            kept = [row for row in slides if row["slideId"] in independent]
            if actual == "slide":
                return {row["slideId"]: row for row in kept}
            return {row["patientId"]: row for row in patient_predictions(kept, aggregation)}

        unlabeled = [{**row, "label": None, "labelIndex": None} for row in records]
        try:
            decisions = {
                key: predicted_index(row["probabilities"], target, threshold)
                for key, row in units(unlabeled).items()
            }
        except ValueError as error:
            raise StorageError(str(error), "AGREEMENT_UNAVAILABLE", 409) from error
        sources = [{"id": "run", "name": manifest["name"], "kind": "run", "labels": decisions}]
        for source in self.label_sources(document, cohort, records):
            entry = {key: source[key] for key in ("id", "name", "kind")}
            if "slides" in source:
                try:
                    entry["labels"] = {
                        key: row["labelIndex"]
                        for key, row in units(source["slides"]).items()
                        if row["labelIndex"] is not None
                    }
                except ValueError as error:
                    entry["reason"] = str(error)
            else:
                entry["reason"] = source["reason"]
            sources.append(entry)
        pairs = []
        usable = [item for item in sources if "labels" in item]
        for index, left in enumerate(usable):
            for right in usable[index + 1 :]:
                common = sorted(set(left["labels"]) & set(right["labels"]))
                pairs.append(
                    {
                        "left": left["id"],
                        "right": right["id"],
                        **agreement(
                            [left["labels"][key] for key in common],
                            [right["labels"][key] for key in common],
                            classes,
                        ),
                    }
                )
        return {
            "evaluationId": document["id"],
            "unit": actual,
            "classOrder": classes,
            "sources": [
                {
                    **{key: item[key] for key in ("id", "name", "kind")},
                    **(
                        {"labeled": len(item["labels"])}
                        if "labels" in item
                        else {"reason": item["reason"]}
                    ),
                }
                for item in sources
            ],
            "pairs": pairs,
            "developmentExcluded": len(records) - len(independent),
            "source": {"predictionsSha256": hashlib.sha256(content).hexdigest()},
        }

    def _score(self, document, cohort, predictor, key, *, bootstrap=True):
        from histopilot.application.clinical import validate_records
        from histopilot.application.run_metrics import run_metrics

        manifest = document["manifest"]
        content = self.artifact(document["id"], "predictions.json")
        try:
            predictions = json.loads(content)
            classes = manifest["target"]["classes"]
            if predictions.get("classOrder") != classes:
                raise ValueError("Saved class order differs from the evaluation.")
            validate_records(predictions.get("records"), classes)
            slides = join_labels(manifest, cohort["manifest"], predictions["records"])
            scored = (
                manifest
                if bootstrap
                else {name: value for name, value in manifest.items() if name != "analysis"}
            )
            metrics = run_metrics(
                scored, slides, ensemble_aggregation=predictor["manifest"].get("aggregation")
            )
        except StorageError:
            raise
        except (ValueError, KeyError, TypeError, OverflowError, RecursionError) as error:
            raise StorageError(
                str(error) or "The saved prediction evidence is invalid.",
                "EVALUATION_EVIDENCE_INVALID",
                409,
            ) from error
        return {**metrics, "scoredBy": key, **({} if bootstrap else {"analysisPending": True})}

    def patient_evidence(self, identity):
        """Read checksummed predictions and validate their frozen patient membership."""
        from histopilot.application.clinical import patient_records, validate_records

        document = self.record(identity)
        manifest = document["manifest"]
        if run_purpose(manifest) == "inference":
            raise StorageError(
                "A run on an unlabeled cohort has no labels to compare patients by. "
                "Compare its predictions case by case instead.",
                "COMPARISON_REQUIRES_LABELS",
                409,
            )
        if manifest.get("splitUnit") == "slide":
            raise StorageError(
                "Patient analysis is disabled for slide-level experiments.",
                "PATIENT_ANALYSIS_DISABLED",
                409,
            )
        cohort = self.store.get_configuration(manifest["cohortId"])
        predictor = self.store.get_configuration(manifest["predictorId"])
        if (
            manifest["cohort"] != reference(cohort)
            or manifest["predictor"] != reference(predictor)
            or manifest["target"] != predictor["manifest"]["target"]
        ):
            raise StorageError(
                "Evaluation references or targets changed.", "COMPARISON_EVIDENCE_CHANGED", 409
            )
        content = self.artifact(identity, "predictions.json")
        try:
            predictions = json.loads(content)
            classes = manifest["target"]["classes"]
            if predictions["classOrder"] != classes:
                raise ValueError("Prediction class order differs from the frozen evaluation.")
            validate_records(predictions["records"], classes)
            memberships = {row["slideId"]: row for row in cohort["manifest"]["memberships"]}
            try:
                slides = join_labels(manifest, cohort["manifest"], predictions["records"])
            except ValueError as error:
                raise ValueError(
                    "Predictions must cover exactly the frozen cohort membership and labels."
                ) from error
            for row in slides:
                source = memberships[row["slideId"]].get("patientIdSource")
                if "patientIdSource" in row and row["patientIdSource"] != source:
                    raise ValueError("Patient identity provenance differs from the frozen cohort.")
                if not row.get("patientId") or source == "slide_fallback":
                    raise ValueError("Patient comparisons require verified patient identities.")
                row["patientIdSource"] = source
            saved = predictions.get("patientRecords")
            if labels_withheld(manifest):
                # Compare only the units its metrics score: development patients are left out.
                shared = development_patients(manifest)
                slides = [row for row in slides if development_flag(row, shared) is not True]
                saved = patient_evidence(manifest, slides, saved)
            patients = patient_records(
                slides, saved, classes, manifest["inference"]["patientAggregation"]
            )
        except (ValueError, KeyError, TypeError, OverflowError) as error:
            raise StorageError(str(error), "COMPARISON_EVIDENCE_INVALID", 409) from error
        return document, slides, patients, hashlib.sha256(content).hexdigest()

    def compare(self, request):
        from histopilot.statistics import patient_bootstrap

        left, _left_slides, left_patients, left_hash = self.patient_evidence(
            request.leftEvaluationId
        )
        right, _right_slides, right_patients, right_hash = self.patient_evidence(
            request.rightEvaluationId
        )
        a, b = left["manifest"], right["manifest"]
        if (
            a["cohort"] != b["cohort"]
            or a["target"] != b["target"]
            or a["inference"]["patientAggregation"] != b["inference"]["patientAggregation"]
        ):
            raise StorageError(
                "Choose the same frozen cohort, target, and patient aggregation for paired comparison.",
                "COMPARISON_CONTEXT_MISMATCH",
                409,
            )
        try:
            result = patient_bootstrap(
                left_patients, a["target"], request.analysis.model_dump(), other=right_patients
            )
        except ValueError as error:
            raise StorageError(str(error), "COMPARISON_PATIENT_MISMATCH", 409) from error
        return {
            "leftEvaluationId": left["id"],
            "rightEvaluationId": right["id"],
            "leftName": a.get("name", left["id"]),
            "rightName": b.get("name", right["id"]),
            "cohortId": a["cohortId"],
            "target": a["target"],
            "patientAggregation": a["inference"]["patientAggregation"],
            "analysis": request.analysis.model_dump(),
            "difference": "left_minus_right",
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
        if (
            test["spec"].get("purpose") == "review"
            and selection.patientIdentifiers == "independent"
        ):
            raise StorageError(
                "Slide review predictions require shared patient identifiers.",
                "INVALID_REVIEW_COHORT",
                409,
            )
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
            test["spec"].get("inference", {}), context=STORED
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
                update={
                    "decisionThreshold": frozen_threshold if frozen_threshold is not None else 0.5
                }
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
            },
            context=STORED,
        )
        reviewed, _guards = self.cohorts._prepare_bound(spec)
        errors = [item for item in reviewed["findings"] if item["severity"] == "error"]
        if errors:
            raise StorageError(
                errors[0]["message"], errors[0]["code"], 409, findings=reviewed["findings"]
            )
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
                "The cohort or its feature verification changed. Review its inputs first.",
                "EVALUATION_COHORT_STALE",
                409,
            )
        model, test = predictor["manifest"], cohort["manifest"]
        slide_unit = test["spec"].get("splitUnit") == "slide"
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
                "The cohort must use this predictor's exact development protocol, target encoding, and development feature bundle.",
                "EVALUATION_PREDICTOR_MISMATCH",
                409,
            )
        purpose = test["spec"].get("purpose", "independent")
        inference = is_inference_purpose(purpose)
        if inference and (
            test["spec"].get("target") is not None
            or any(row.get("label") is not None for row in test["memberships"])
        ):
            raise StorageError(
                "Inference cohorts must be unlabeled.", "INVALID_INFERENCE_COHORT", 409
            )
        if purpose == "review" and (
            test["spec"].get("patientIdentifiers") != "shared" or model["target"]["unit"] != "slide"
        ):
            raise StorageError(
                "Review predictions require unlabeled slide outcomes and shared patient IDs.",
                "INVALID_REVIEW_COHORT",
                409,
            )
        if (
            test["overlap"]["slideIds"]
            or test["overlap"].get("sourceSlideIds")
            or (
                not slide_unit
                and test["overlap"]["patientIds"]
                and not patient_overlap_allowed(model["target"]["unit"])
            )
        ):
            raise StorageError(
                "Inference would predict slides or patients used in this predictor's development."
                if inference
                else "Test membership overlaps this predictor's development data.",
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
                if (
                    representation_kind(features) == "slide"
                    or model.get("recipe", {}).get("analysis") is not None
                ):
                    keys += ("slide_encoder",)
                if model.get("recipe", {}).get("analysis") is not None:
                    keys += (
                        "reader_type",
                        "segmenter",
                        "seg_conf_thresh",
                        "remove_holes",
                        "remove_artifacts",
                        "remove_penmarks",
                        "min_tissue_proportion",
                    )
                left_options, right_options = left["spec"]["options"], right["spec"]["options"]
                if any(left_options.get(key) != right_options.get(key) for key in keys):
                    raise StorageError(
                        "Development and test extraction settings differ.",
                        "EVALUATION_EXTRACTION_MISMATCH",
                        409,
                    )
        if (
            not slide_unit
            and test["spec"]["inference"]["patientAggregation"] != model["patientAggregation"]
        ):
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
                "EVALUATION_THRESHOLD_MISMATCH",
                409,
            )
        self.predictors.verify_checkpoints(predictor)
        clinical_contract = self._clinical_contract(model, test)
        if reviewed_at_evaluation or inference:
            review = {
                "coverage": test["coverage"],
                "overlap": test["overlap"],
                "findings": test["findings"],
            }
        elif (test.get("overlap") or {}).get("patientIds"):
            # A labeled run predicts the new slides of development patients but scores none
            # of them (`run_evidence.development_patients`), so it names them. Runs without
            # shared patients keep the manifest, and so the ID, they had.
            review = {"overlap": test["overlap"]}
        else:
            review = {}
        return {
            "kind": "model-evaluation",
            **({"splitUnit": test["spec"]["splitUnit"]} if "splitUnit" in test["spec"] else {}),
            # Review cohorts are the earlier name of inference cohorts; new runs of
            # either execute as inference. Earlier "review" runs keep their manifest.
            **({"purpose": "inference"} if inference else {}),
            # Labeled runs are predicted label-blind and scored from the frozen cohort.
            # Runs saved before this key existed keep their worker-scored metrics.
            **({} if inference else {"labelSource": LABELS_FROM_COHORT}),
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
            # Patient bootstrap analysis needs observed outcomes; inference has none.
            **(
                {}
                if inference or slide_unit
                else {
                    "analysis": model.get("recipe", {}).get("analysis")
                    or PatientAnalysisSettings().model_dump()
                }
            ),
            "bagPolicy": {
                "evalBagSize": model.get("recipe", {}).get("evalBagSize"),
                # A seed ensemble has no single seed: this is only the loader default,
                # and each member applies its own recorded training seed.
                "trainingSeed": model["trainingSeed"]
                if "trainingSeed" in model
                else model["trainingSeeds"][0],
            },
            "features": features,
            "summary": test["summary"],
            "status": "planned",
            "results": None,
            "executionEnabled": True,
            "executionNote": INFERENCE_NOTE if inference else EXECUTION_NOTE,
            **review,
        }

    def _clinical_contract(self, model, test):
        from histopilot.application.clinical_inputs import frozen_clinical_values
        from histopilot.clinical_features import clinical_fields, clinical_rows

        recipe = model.get("recipe", {})
        fields = clinical_fields(recipe)
        if not fields:
            return None
        values = frozen_clinical_values(
            self.store,
            self.filesystem,
            test["spec"].get("datasetIds") or [test["datasetId"]],
            test["memberships"],
            fields,
        )
        try:
            clinical_rows(
                test["memberships"], values, fields, unit=test["spec"].get("splitUnit", "patient")
            )
        except ValueError as error:
            raise StorageError(str(error), "CLINICAL_VALUES_INVALID", 422) from error
        return {
            "inputMode": recipe["inputMode"],
            "fields": fields,
            "valuesSha256": content_hash(values),
        }

    def preview(self, selection):
        try:
            manifest = self._prepare(selection)
            return {
                "canSave": True,
                "previewHash": content_hash(manifest),
                "manifest": manifest,
                "findings": manifest.get("findings", []),
                "executionEnabled": True,
                "executionNote": manifest["executionNote"],
            }
        except StorageError as error:
            return {
                "canSave": False,
                "previewHash": None,
                "manifest": None,
                "findings": error.findings or [finding(error.code, str(error))],
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
            if content_hash(manifest) != request.previewHash:
                raise StorageError("Evaluation inputs changed. Review again.", "PREVIEW_STALE", 409)
            published = self.store.publish_configuration(
                manifest={**manifest, "previewHash": request.previewHash},
                operation_id=request.operationId,
            )
            return lifecycle_document(self.store, published)

    def execution(self, identity):
        return self.get(identity)["execution"]

    def _execution_plan(self, identity):
        document = self.record(identity)
        manifest = document["manifest"]
        selection = EvaluationRunSelection.model_validate(manifest["selection"], context=STORED)
        if "coverage" in manifest:
            # Auto selection is resolved by the saved review. New inventories
            # must not silently replace it or make an existing plan ambiguous.
            selection = selection.model_copy(
                update={
                    "featureBundleId": manifest["features"]["bundle"]["id"],
                    "inference": InferenceSettings.model_validate(
                        manifest["inference"], context=STORED
                    ),
                }
            )
        reviewed = self._prepare(selection)
        if reviewed.get("splitUnit", "patient") != manifest.get("splitUnit", "patient"):
            raise StorageError("The frozen split unit changed.", "EVALUATION_INPUTS_CHANGED", 409)
        for key in ("predictor", "cohort", "target", "features", "inference"):
            if reviewed[key] != manifest[key]:
                raise StorageError(
                    "Evaluation inputs changed. Create and review a new evaluation.",
                    "EVALUATION_INPUTS_CHANGED",
                )
        # Earlier "review" runs froze a label analysis policy that inference omits;
        # their saved plan keeps it, so only current policies are compared.
        if (
            "analysis" in manifest
            and manifest.get("purpose") != "review"
            and reviewed.get("analysis") != manifest["analysis"]
        ):
            raise StorageError(
                "The frozen analysis policy changed.", "EVALUATION_INPUTS_CHANGED", 409
            )
        if "bagPolicy" in manifest and reviewed["bagPolicy"] != manifest["bagPolicy"]:
            raise StorageError(
                "The frozen evaluation bag policy changed.", "EVALUATION_INPUTS_CHANGED", 409
            )
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
                raise StorageError(
                    "Clinical schema or frozen values changed.", "CLINICAL_INPUTS_CHANGED", 409
                )
            clinical_values = frozen_clinical_values(
                self.store,
                self.filesystem,
                cohort["manifest"]["spec"].get("datasetIds") or [cohort["manifest"]["datasetId"]],
                memberships,
                manifest["clinical"]["fields"],
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
            pack_stamps = pack_layout(Path(pack_path))["packStamps"]
        if document["execution"]["status"] != "not_started":
            # Device selection belongs to the first launch. Resume and request
            # replay must preserve it even if CUDA availability has changed.
            saved = read_json_bounded(self.jobs.folder(identity) / "plan.json")
            if content_hash(saved) != document["execution"].get("planHash"):
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
        withheld = labels_withheld(manifest)
        if withheld:
            memberships = strip_labels(memberships)
        return {
            "kind": "evaluation",
            **({"splitUnit": manifest["splitUnit"]} if "splitUnit" in manifest else {}),
            # Only runs created as inference carry this key; earlier plans must keep
            # their exact keys so interrupted jobs remain resumable.
            **({"purpose": "inference"} if manifest.get("purpose") == "inference" else {}),
            # The saved record, not the current review, decides label blindness.
            **({"labelsWithheld": True} if withheld else {}),
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
                **(
                    {
                        "clinicalValues": clinical_values,
                        "inputMode": manifest["clinical"]["inputMode"],
                        "clinicalFields": manifest["clinical"]["fields"],
                    }
                    if clinical_values
                    else {}
                ),
                "memberships": memberships,
                "featureDim": manifest["features"]["dimensions"],
                "featureFiles": files,
                "sourceStamps": {row["path"]: row for row in files.values()},
                "loadingPolicy": "mmap" if pack_path else "native",
                "packPath": pack_path,
                "packStamps": pack_stamps,
            },
        }

    def launch(self, identity, operation_id, *, resume=False, task_owner=None, task_title=None):
        """``task_owner`` queues the job under another Task Center owner (a bulk batch)."""
        with lifecycle_guard(self.store.folder):
            replay = self.jobs.replay_launch(
                identity, operation_id, resume=resume, record_kind="model-evaluation"
            )
            if replay is not None:
                return replay
            plan = self._execution_plan(identity)
            return self.jobs.launch(
                identity,
                plan,
                operation_id,
                resume=resume,
                task_owner=task_owner,
                task_title=task_title,
            )

    def cancel(self, identity, operation_id):
        self._document(identity)
        return self.jobs.cancel(identity, operation_id)

    def download(self, identity, filename):
        """The file a download serves: the worker's own, or for a label-blind run, its
        service-scored metrics and prediction tables with the cohort labels joined."""
        record = self.record(identity)
        manifest = record["manifest"]
        derived = {"metrics.json", "slide-predictions.csv", "patient-predictions.csv"}
        if not labels_withheld(manifest) or filename not in derived:
            return self.artifact(identity, filename)
        content = self.artifact(identity, "predictions.json")
        if filename == "metrics.json":
            metrics = self.metrics(record, record["execution"]["result"])
            return (json.dumps(metrics, indent=2, sort_keys=True) + "\n").encode("utf-8")
        if filename == "patient-predictions.csv" and manifest.get("splitUnit") == "slide":
            raise StorageError(
                "Slide-split evaluations score each slide on its own; they have no patient table.",
                "EVALUATION_ARTIFACT_NOT_FOUND",
                404,
            )
        from histopilot.application.clinical import validate_records

        cohort = self.store.get_configuration(manifest["cohortId"])
        if manifest["cohort"] != reference(cohort):
            raise StorageError(
                "The evaluation no longer matches its frozen cohort.",
                "EVALUATION_EVIDENCE_CHANGED",
                409,
            )
        try:
            predictions = json.loads(content)
            validate_records(predictions.get("records"), manifest["target"]["classes"])
            slides = join_labels(manifest, cohort["manifest"], predictions["records"])
            return prediction_table(
                manifest,
                slides,
                predictions.get("patientRecords"),
                patient=filename == "patient-predictions.csv",
            )
        except StorageError:
            raise
        except (ValueError, KeyError, TypeError) as error:
            raise StorageError(
                str(error) or "The saved prediction evidence is invalid.",
                "EVALUATION_EVIDENCE_INVALID",
                409,
            ) from error

    def artifact(self, identity, filename):
        if filename not in {
            "predictions.json",
            "metrics.json",
            "summary.json",
            "slide-predictions.csv",
            "patient-predictions.csv",
        }:
            raise StorageError(
                "Evaluation artifact not found.", "EVALUATION_ARTIFACT_NOT_FOUND", 404
            )
        execution = self.record(identity)["execution"]
        if execution["status"] != "completed":
            raise StorageError(
                "Evaluation results are available after the job finishes.",
                "EVALUATION_NOT_COMPLETED",
            )
        artifacts = execution["result"]["artifacts"]
        if filename not in artifacts and filename in {"metrics.json", "summary.json"}:
            # Inference writes a label-free summary; evaluations write metrics.
            raise StorageError(
                f"This run did not produce {filename}.", "EVALUATION_ARTIFACT_NOT_FOUND", 404
            )
        expected = artifacts.get(filename)
        path = self.jobs.folder(identity) / filename
        if not expected or expected["path"] != str(path):
            raise StorageError("Evaluation result provenance changed.", "EVALUATION_RESULT_CHANGED")
        content = read_file_bounded(path, 64 * 1024 * 1024)
        if (
            len(content) != expected["bytes"]
            or hashlib.sha256(content).hexdigest() != expected["sha256"]
        ):
            raise StorageError("The evaluation output changed.", "EVALUATION_RESULT_CHANGED")
        return content
