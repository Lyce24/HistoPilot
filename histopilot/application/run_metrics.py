"""Metrics of a label-blind run, computed by the control service.

A labeled evaluation used to be scored inside its worker, which wrote ``metrics.json``.
A label-blind run's worker writes predictions only. This module scores those predictions
against the labels joined from the frozen cohort, with the same torch-free functions and
in the same document shape as that worker, so both generations report identical numbers.

One difference is deliberate: a slide-level predictor may predict new slides of patients
seen in development (they are flagged, never refused). Those units are not independent of
development, so every metric leaves them out and ``developmentExcluded`` counts them.
"""

from __future__ import annotations

from histopilot.application.run_evidence import development_flag, development_patients
from histopilot.cv_summary import evaluation_metrics
from histopilot.scoring import patient_predictions

SCORED_BY = "control_service_label_join_v1"


def run_metrics(manifest, slides, *, ensemble_aggregation=None):
    """The worker's ``metrics.json`` document for labeled ``slides`` of one run.

    ``slides`` carry the joined labels (``run_evidence.join_labels``); unlabeled slides are
    predicted but not scored, as in the worker.
    """
    target = manifest["target"]
    threshold = manifest["inference"]["decisionThreshold"]
    aggregation = manifest["inference"]["patientAggregation"]
    slide_unit = manifest.get("splitUnit") == "slide"
    shared = development_patients(manifest)
    scored = [row for row in slides if development_flag(row, shared) is not True]
    if slide_unit:
        patients, patient_metrics = (
            [],
            {
                "available": False,
                "count": 0,
                "reason": "Patient analysis is disabled for slide-level experiments.",
            },
        )
    else:
        try:
            patients = patient_predictions(scored, aggregation)
            patient_metrics = evaluation_metrics(patients, target, threshold)
        except ValueError as error:
            if target["unit"] == "patient":
                raise
            patients, patient_metrics = [], {"available": False, "count": 0, "reason": str(error)}
    slide_metrics = evaluation_metrics(scored, target, threshold)
    metrics = {
        "unit": target["unit"],
        "classOrder": target["classes"],
        "positiveClass": target.get("positiveClass"),
        "decisionThreshold": threshold,
        "patientAggregation": None
        if slide_unit
        else "mean_logits"
        if aggregation == "mean_logits"
        else "mean_probabilities",
        **({"splitUnit": manifest["splitUnit"]} if "splitUnit" in manifest else {}),
        **({"ensembleAggregation": "mean_logit"} if ensemble_aggregation == "mean_logit" else {}),
        "slide": slide_metrics,
        "patient": patient_metrics,
        "selected": patient_metrics if target["unit"] == "patient" else slide_metrics,
    }
    if not slide_unit and manifest.get("analysis") is not None:
        from histopilot.statistics import patient_analysis

        metrics["patientAnalysis"] = patient_analysis(
            scored, patients, target, manifest["analysis"]
        )
        intervals = metrics["patientAnalysis"]["uncertainty"].get("intervals")
        if intervals:
            patient_metrics["confidenceIntervals"] = intervals
    excluded = [row for row in slides if development_flag(row, shared) is True]
    if excluded:
        metrics["developmentExcluded"] = {
            "slides": len(excluded),
            "patients": len({row["patientId"] for row in excluded}),
            "reason": "These slides come from patients used in this predictor's development. "
            "They were predicted but are left out of every metric.",
        }
    return metrics
