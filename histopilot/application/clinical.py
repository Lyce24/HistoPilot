"""Descriptive clinical utility from immutable evaluation evidence.

Only standard-library statistics run in the control service. Test predictions are
never used to fit calibration, choose an optimal cutoff, or alter a predictor.
"""

import csv
import hashlib
import io
import json
import math
from bisect import bisect_left
from collections import defaultdict

from histopilot.application.evaluation_runs import EvaluationRunService
from histopilot.application.feature_bundles import _hash
from histopilot.application.predictors import finding, lifecycle_document, reference
from histopilot.schemas.clinical import ClinicalSelection
from histopilot.storage.lifecycle import lifecycle_guard
from histopilot.storage.project_lock import StorageError

SOURCES = [
    {
        "title": "Scikit-learn: probability calibration",
        "url": "https://scikit-learn.org/stable/modules/calibration.html",
    },
    {
        "title": "Scikit-learn: classification metrics",
        "url": "https://scikit-learn.org/stable/modules/model_evaluation.html",
    },
    {
        "title": "Decision curve analysis tutorial",
        "url": "https://www.danieldsjoberg.com/dca-tutorial/dca-tutorial-stata.html",
    },
    {
        "title": "Reporting and Interpreting Decision Curve Analysis: A Guide for Investigators",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6261531/",
    },
]


def _invalid(message):
    return StorageError(message, "CLINICAL_EVIDENCE_INVALID", 409)


def _divide(numerator, denominator):
    if not denominator:
        return None
    result = numerator / denominator
    return result if math.isfinite(result) else None


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _logsumexp(values):
    maximum = max(values)
    return maximum + math.log(math.fsum(math.exp(value - maximum) for value in values))


def _validate_records(rows, classes, *, patient=False):
    if not isinstance(rows, list):
        raise _invalid("Saved predictions must contain a list of records.")
    key = "patientId" if patient else "slideId"
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise _invalid("A saved prediction record is invalid.")
        identity = row.get(key)
        if not isinstance(identity, str) or not identity or identity in seen:
            raise _invalid(f"Saved predictions require one unique {key} per record.")
        seen.add(identity)
        label, index = row.get("label"), row.get("labelIndex")
        if (index is None and label is not None) or (
            index is not None
            and (type(index) is not int or not 0 <= index < len(classes) or label != classes[index])
        ):
            raise _invalid("Saved prediction labels do not match the frozen class encoding.")
        values = row.get("probabilities")
        if (
            not isinstance(values, list)
            or len(values) != len(classes)
            or any(not _number(value) or not 0 <= value <= 1 for value in values)
            or not math.isclose(math.fsum(values), 1, abs_tol=1e-6)
        ):
            raise _invalid("Saved class probabilities must be finite, normalized, and ordered.")
        logs = row.get("logProbabilities")
        if logs is not None and (
            not isinstance(logs, list)
            or len(logs) != len(classes)
            or any(not _number(value) or value > 1e-6 for value in logs)
            or not math.isclose(_logsumexp(logs), 0, abs_tol=1e-6)
            or any(
                not math.isclose(math.exp(log), value, abs_tol=1e-6)
                for log, value in zip(logs, values, strict=True)
            )
        ):
            raise _invalid("Saved log probabilities do not match the probability evidence.")
    return rows


def _patient_records(slides, patients, classes):
    _validate_records(patients, classes, patient=True)
    groups = defaultdict(list)
    for row in slides:
        if row.get("patientId"):
            groups[row["patientId"]].append(row)
    if not groups:
        raise StorageError(
            "This evaluation has no patient grouping identities. Select slide-level analysis.",
            "CLINICAL_PATIENT_UNAVAILABLE",
            409,
        )
    if {row["patientId"] for row in patients} != set(groups):
        raise StorageError(
            "Patient predictions are unavailable or incomplete. Review patient labels and grouping in the evaluation.",
            "CLINICAL_PATIENT_UNAVAILABLE",
            409,
        )
    for patient in patients:
        group = groups[patient["patientId"]]
        labels = {row["labelIndex"] for row in group if row["labelIndex"] is not None}
        identities = patient.get("slideIds")
        if (
            not isinstance(identities, list)
            or any(not isinstance(value, str) for value in identities)
            or len(identities) != len(group)
            or set(identities) != {row["slideId"] for row in group}
            or len(labels) > 1
            or patient["labelIndex"] != next(iter(labels), None)
            or any(
                not math.isclose(
                    patient["probabilities"][index],
                    math.fsum(row["probabilities"][index] for row in group) / len(group),
                    abs_tol=1e-6,
                )
                for index in range(len(classes))
            )
        ):
            raise _invalid(
                "Saved patient predictions must preserve labels, slide groups, and mean probabilities."
            )
        if all(row.get("logProbabilities") is not None for row in group) and (
            patient.get("logProbabilities") is None
            or any(
                not math.isclose(
                    patient["logProbabilities"][index],
                    _logsumexp([row["logProbabilities"][index] for row in group])
                    - math.log(len(group)),
                    abs_tol=1e-6,
                )
                for index in range(len(classes))
            )
        ):
            raise _invalid("Saved patient log probabilities differ from the mean of their slides.")
    return patients


def _operating_point(positive_scores, negative_scores, threshold):
    positive = len(positive_scores)
    n = positive + len(negative_scores)
    tp = positive - bisect_left(positive_scores, threshold)
    fp = len(negative_scores) - bisect_left(negative_scores, threshold)
    fn, tn = positive - tp, n - positive - fp
    sensitivity, specificity = _divide(tp, positive), _divide(tn, n - positive)
    weight = threshold / (1 - threshold)
    net_benefit = (tp - fp * weight) / n
    treat_all = (positive - (n - positive) * weight) / n
    weighted_missed = _divide(fn / n * 100, weight)
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": _divide(tp, tp + fp),
        "npv": _divide(tn, tn + fn),
        "accuracy": (tp + tn) / n,
        "balancedAccuracy": (
            (sensitivity + specificity) / 2
            if sensitivity is not None and specificity is not None
            else None
        ),
        "f1": _divide(2 * tp, 2 * tp + fp + fn),
        "positiveLikelihoodRatio": (
            _divide(sensitivity, 1 - specificity)
            if sensitivity is not None and specificity is not None
            else None
        ),
        "negativeLikelihoodRatio": (
            _divide(1 - sensitivity, specificity)
            if sensitivity is not None and specificity is not None
            else None
        ),
        "predictedPositive": tp + fp,
        "predictedNegative": tn + fn,
        "netBenefit": net_benefit,
        "treatAllNetBenefit": treat_all,
        "treatNoneNetBenefit": 0.0,
        "standardizedNetBenefit": _divide(net_benefit, positive / n),
        "netInterventionsAvoidedPer100": tn / n * 100 - weighted_missed
        if weighted_missed is not None
        else None,
        "highRiskPer100": (tp + fp) / n * 100,
        "truePositivePer100": tp / n * 100,
        "falsePositivePer100": fp / n * 100,
        "missedPositivePer100": fn / n * 100,
    }


def _wilson(successes, denominator):
    """Two-sided 95% Wilson interval conditional on this observed denominator."""
    if not denominator:
        return None
    z = 1.959963984540054
    proportion = successes / denominator
    scale = 1 + z * z / denominator
    center = (proportion + z * z / (2 * denominator)) / scale
    half = (
        z
        * math.sqrt(proportion * (1 - proportion) / denominator + z * z / (4 * denominator**2))
        / scale
    )
    return {"lower": max(0.0, center - half), "upper": min(1.0, center + half)}


def _ranking_curves(labels, scores):
    """Exact tied-score ROC AUC and non-interpolated average precision."""
    positive, negative = sum(labels), len(labels) - sum(labels)
    roc = [
        {
            "threshold": None,
            "falsePositiveRate": 0.0 if negative else None,
            "truePositiveRate": 0.0 if positive else None,
        }
    ]
    pr = [{"threshold": None, "recall": 0.0 if positive else None, "precision": 1.0}]
    groups = defaultdict(lambda: [0, 0])
    for label, score in zip(labels, scores, strict=True):
        groups[score][0 if label else 1] += 1
    tp = fp = 0
    auc = ap = 0.0
    for threshold in sorted(groups, reverse=True):
        new_tp, new_fp = groups[threshold]
        if positive and negative:
            auc += new_fp / negative * (2 * tp + new_tp) / (2 * positive)
        tp, fp = tp + new_tp, fp + new_fp
        if positive:
            ap += new_tp / positive * tp / (tp + fp)
        roc.append(
            {
                "threshold": threshold,
                "falsePositiveRate": _divide(fp, negative),
                "truePositiveRate": _divide(tp, positive),
            }
        )
        pr.append(
            {"threshold": threshold, "recall": _divide(tp, positive), "precision": tp / (tp + fp)}
        )
    # Bound report size without approximating the summary statistics.
    if len(roc) > 2001:
        indices = [round(index * (len(roc) - 1) / 2000) for index in range(2001)]
        roc, pr = [roc[index] for index in indices], [pr[index] for index in indices]
    return roc, pr, auc if positive and negative else None, ap if positive else None


def _calibration(labels, scores, bins):
    groups = [[] for _ in range(bins)]
    for label, score in zip(labels, scores, strict=True):
        groups[min(int(score * bins), bins - 1)].append((label, score))
    result = []
    for index, group in enumerate(groups):
        predicted = math.fsum(score for _, score in group) / len(group) if group else None
        observed = sum(label for label, _ in group) / len(group) if group else None
        result.append(
            {
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "count": len(group),
                "meanPredicted": predicted,
                "observedFraction": observed,
                "absoluteError": abs(predicted - observed) if group else None,
            }
        )
    return result


def clinical_report(predictions, target, inference, selection):
    """Calculate a finite JSON report; the service separately verifies provenance."""
    classes = target["classes"]
    if not isinstance(predictions, dict) or predictions.get("classOrder") != classes:
        raise _invalid("Prediction class order differs from the frozen evaluation target.")
    slides = _validate_records(predictions.get("records"), classes)
    for row in slides:
        if row.get("patientId") is not None and (
            not isinstance(row["patientId"], str) or not row["patientId"]
        ):
            raise _invalid("Patient grouping identities must be nonempty strings or null.")
    unit = target["unit"] if selection.unit == "selected" else selection.unit
    fallback_ids = sum(row.get("patientIdSource") == "slide_fallback" for row in slides)
    if unit == "patient" and fallback_ids:
        raise StorageError(
            "Patient-level clinical analysis requires verified patient identities. Some slides use Slide_ID fallback groups; map their patients or select slide-level analysis.",
            "CLINICAL_PATIENT_IDENTITIES_UNVERIFIED",
            409,
        )
    multiclass = target["task"] == "multiclass_classification"
    positive_class = selection.positiveClass if multiclass else target["positiveClass"]
    if positive_class not in classes or (
        not multiclass and selection.positiveClass not in (None, positive_class)
    ):
        raise StorageError(
            "Choose a class for one-versus-rest analysis."
            if multiclass
            else "Binary clinical analysis must preserve the frozen positive class.",
            "CLINICAL_POSITIVE_CLASS_INVALID",
            422,
        )
    positive_index = classes.index(positive_class)
    rows = (
        _patient_records(slides, predictions.get("patientRecords"), classes)
        if unit == "patient"
        else slides
    )
    if inference.get("patientAggregation") != "mean":
        raise _invalid("Clinical reports require the frozen mean-probability patient rule.")
    labeled = [row for row in rows if row["labelIndex"] is not None]
    if not labeled:
        raise StorageError(
            "No labeled predictions are available for this unit. Clinical statistics require observed outcomes.",
            "CLINICAL_NO_LABELED_OUTCOMES",
            409,
        )
    labels = [row["labelIndex"] == positive_index for row in labeled]
    scores = [row["probabilities"][positive_index] for row in labeled]
    count, positive = len(labels), sum(labels)
    prevalence = positive / count
    threshold = (
        inference["decisionThreshold"] if selection.threshold is None else selection.threshold
    )
    brier = (
        math.fsum((score - label) ** 2 for label, score in zip(labels, scores, strict=True)) / count
    )
    baseline = prevalence * (1 - prevalence)
    losses, multiclass_losses = [], []
    clipped = False
    for row, label, score in zip(labeled, labels, scores, strict=True):
        logs = row.get("logProbabilities")
        if logs is not None:
            loss = (
                -logs[positive_index]
                if label
                else -_logsumexp(
                    [value for index, value in enumerate(logs) if index != positive_index]
                )
            )
            multiclass_losses.append(max(0.0, -logs[row["labelIndex"]]))
        else:
            probability = (
                score
                if label
                else math.fsum(
                    value
                    for index, value in enumerate(row["probabilities"])
                    if index != positive_index
                )
            )
            clipped = clipped or probability == 0 or row["probabilities"][row["labelIndex"]] == 0
            loss = -math.log(max(probability, 1e-300))
            multiclass_losses.append(
                -math.log(max(row["probabilities"][row["labelIndex"]], 1e-300))
            )
        losses.append(max(0.0, loss))
    roc, pr, auc, ap = _ranking_curves(labels, scores)
    calibration = _calibration(labels, scores, selection.bins)
    delta = (selection.thresholdMax - selection.thresholdMin) / (selection.thresholdSteps - 1)
    thresholds = [
        selection.thresholdMin + index * delta for index in range(selection.thresholdSteps)
    ]
    thresholds[-1] = selection.thresholdMax
    patient_ids = {
        row["patientId"]
        for row in slides
        if row.get("patientId") and row.get("patientIdSource") != "slide_fallback"
    }
    missing_ids = sum(not row.get("patientId") for row in slides)
    positive_scores = sorted(score for label, score in zip(labels, scores, strict=True) if label)
    negative_scores = sorted(
        score for label, score in zip(labels, scores, strict=True) if not label
    )
    operating = _operating_point(positive_scores, negative_scores, threshold)
    independent = not fallback_ids and (
        unit == "patient" or (not missing_ids and len(patient_ids) == len(slides))
    )
    uncertainty = {
        "method": "wilson_95" if independent else "unavailable",
        "reason": (
            "95% Wilson intervals assume independent analysis units and condition on each metric's observed denominator. They do not account for model fitting or threshold selection. Brier and AUC uncertainty are not estimated."
            if independent
            else "Intervals are unavailable for slide analyses with repeated, missing, or Slide_ID fallback patient identities. Verify patient identities and use patient-level analysis; independent-slide intervals would misrepresent uncertainty."
        ),
        "operatingPoint": {
            "sensitivity": _wilson(operating["tp"], positive) if independent else None,
            "specificity": _wilson(operating["tn"], count - positive) if independent else None,
            "ppv": _wilson(operating["tp"], operating["predictedPositive"])
            if independent
            else None,
            "npv": _wilson(operating["tn"], operating["predictedNegative"])
            if independent
            else None,
        },
    }
    warnings = [
        "These are descriptive test-cohort estimates; no threshold is optimized and no probability recalibration is fitted.",
        "Decision curves assume a representative cohort and that the selected class is the outcome for the proposed intervention. Threshold odds encode the relative harm of a false positive; these curves do not establish patient benefit.",
        uncertainty["reason"],
    ]
    if unit == "slide":
        warnings.append(
            "Clinical impact counts are per 100 slides. Multiple slides from one patient are correlated and can overweight patients; use patient-level analysis for patient decisions."
        )
    if missing_ids:
        warnings.append(
            f"{missing_ids} slides have no patient identity; they are excluded from saved patient predictions."
        )
    if fallback_ids:
        warnings.append(
            f"{fallback_ids} slides use Slide_ID fallback groups. These are not verified patients, and patient independence cannot be established."
        )
    if len(rows) != count:
        warnings.append(
            f"{len(rows) - count} unlabeled {unit} records are excluded from every statistic and clinical impact denominator."
        )
    if not positive or positive == count:
        warnings.append(
            "Only one outcome class is observed. ROC AUC, Brier skill, and some operating metrics are undefined; interpret decision curves with caution."
        )
    if multiclass:
        warnings.append(
            "Operating and decision curves are one-versus-rest for the selected class. Their thresholds are descriptive and do not replace the evaluation's multiclass argmax decisions."
        )
    if clipped:
        warnings.append(
            "Legacy predictions without log probabilities contain a zero-probability observed outcome. Log loss uses a documented floor of 1e-300."
        )
    if len(set(scores)) > 2000:
        warnings.append(
            "Ranking curves and their CSV exports are sampled to at most 2,001 points. ROC AUC and average precision use every distinct prediction score."
        )
    return {
        "unit": unit,
        "frozenUnit": target["unit"],
        "positiveClass": positive_class,
        "classOrder": classes,
        "multiclass": multiclass,
        "decisionThreshold": threshold,
        "frozenDecisionThreshold": inference["decisionThreshold"],
        "thresholdSource": "frozen_evaluation"
        if selection.threshold is None and not multiclass
        else "descriptive_override",
        "counts": {
            "total": len(rows),
            "labeled": count,
            "unlabeled": len(rows) - count,
            "positive": positive,
            "negative": count - positive,
            "patients": len(patient_ids) if patient_ids else None,
            "slides": len(slides),
            "missingPatientIds": missing_ids,
            "fallbackPatientIds": fallback_ids,
        },
        "metrics": {
            "prevalence": prevalence,
            "meanPredictedRisk": math.fsum(scores) / count,
            "observedExpectedRatio": _divide(positive, math.fsum(scores)),
            "calibrationGap": prevalence - math.fsum(scores) / count,
            "brierScore": brier,
            "brierReference": baseline,
            "brierSkillScore": 1 - brier / baseline if baseline else None,
            "logLoss": math.fsum(loss / count for loss in losses),
            "multiclassBrierScore": math.fsum(
                math.fsum(
                    (value - (row["labelIndex"] == index)) ** 2
                    for index, value in enumerate(row["probabilities"])
                )
                for row in labeled
            )
            / count
            if multiclass
            else None,
            "multiclassLogLoss": math.fsum(loss / count for loss in multiclass_losses)
            if multiclass
            else None,
            "rocAuc": auc,
            "averagePrecision": ap,
            "ece": math.fsum(
                row["count"] * row["absoluteError"] for row in calibration if row["count"]
            )
            / count,
            "mce": max(row["absoluteError"] for row in calibration if row["count"]),
        },
        "operatingPoint": operating,
        "uncertainty": uncertainty,
        "operatingCurve": [
            _operating_point(positive_scores, negative_scores, value) for value in thresholds
        ],
        "rocCurve": roc,
        "precisionRecallCurve": pr,
        "curveSampling": {
            "distinctScores": len(set(scores)),
            "maximumPoints": 2001,
            "returnedPoints": len(roc),
            "downsampled": len(set(scores)) > 2000,
            "summaryStatistics": "exact",
            "csv": "same_points_as_report",
        },
        "calibration": calibration,
        "warnings": warnings,
        "definitions": {
            "brierScore": "Mean (probability of selected class - observed binary outcome)^2; range 0 to 1; lower is better.",
            "brierReference": "Observed cohort prevalence × (1 - prevalence), the constant-risk reference on this cohort; not a development-fitted comparator.",
            "brierSkillScore": "1 - Brier / reference Brier; null when the reference is zero.",
            "multiclassBrierScore": "Mean sum over all frozen classes of squared probability error, without dividing by two; range 0 to 2.",
            "logLoss": "Mean negative log probability of the observed one-versus-rest outcome; saved log probabilities preserve extreme finite losses. Legacy zero probabilities use a 1e-300 floor.",
            "calibration": "Equal-width probability bins [lower, upper), with probability 1 in the final bin. ECE weights absolute bin error by bin count; MCE is the largest nonempty-bin error. Both depend on binning.",
            "observedExpectedRatio": "Observed positive count / sum of predicted risks; null if expected positives are zero. Calibration gap = observed prevalence - mean predicted risk.",
            "rocAuc": "Exact trapezoidal ROC area with tied scores grouped; undefined when either outcome class is absent.",
            "averagePrecision": "Non-interpolated sum of precision × increase in recall across distinct descending scores; not trapezoidal PR area.",
            "operatingPoint": "Probability ≥ threshold predicts the selected class. Null denotes an undefined denominator, including unbounded likelihood ratios.",
            "netBenefit": "TP / N - FP / N × threshold / (1 - threshold). Treat-all = prevalence - (1 - prevalence) × threshold odds; treat-none = 0.",
            "standardizedNetBenefit": "Model net benefit divided by prevalence; null when prevalence is zero.",
            "netInterventionsAvoidedPer100": "100 × (model net benefit - treat-all net benefit) / threshold odds; a weighted net reduction relative to treat-all, not a raw number of avoided interventions.",
            "impact": "High-risk, true-positive, false-positive, and missed-positive counts per 100 labeled analysis units.",
            "scope": "Fixed binary or multiclass outcomes only. This analysis does not estimate survival, censoring-adjusted utility, monetary cost-effectiveness, or causal treatment effects.",
        },
        "sources": SOURCES,
    }


class ClinicalService:
    def __init__(self, store, filesystem):
        self.store, self.filesystem = store, filesystem
        self.evaluations = EvaluationRunService(store, filesystem)

    def list(self, *, include_inactive=False):
        return {
            "items": [
                lifecycle_document(self.store, item)
                for item in self.store.list_configurations(
                    "clinical-analysis", include_inactive=include_inactive
                )
            ]
        }

    def get(self, identity):
        document = self.store.get_configuration(identity)
        if document["manifest"].get("kind") != "clinical-analysis":
            raise StorageError("Clinical analysis not found.", "CLINICAL_ANALYSIS_NOT_FOUND", 404)
        return lifecycle_document(self.store, document)

    def _prepare(self, selection):
        evaluation = self.evaluations.get(selection.evaluationId)
        manifest = evaluation["manifest"]
        self.store.lifecycle.assert_document_usable(manifest)
        predictor = self.store.get_configuration(manifest["predictorId"])
        cohort = self.store.get_configuration(manifest["cohortId"])
        standalone = cohort["manifest"].get("overlap", {}).get("deferred", False)
        cohort_target = cohort["manifest"].get("target")
        target_matches = cohort_target == manifest["target"]
        if standalone:
            target_matches = cohort_target is None or all(
                cohort_target.get(key) == manifest["target"].get(key)
                for key in ("task", "unit", "classes", "positiveClass")
            )
        if (
            manifest.get("predictor") != reference(predictor)
            or manifest.get("cohort") != reference(cohort)
            or predictor["manifest"].get("kind") != "frozen-predictor"
            or cohort["manifest"].get("kind") != "evaluation-cohort"
            or predictor["manifest"].get("target") != manifest["target"]
            or not target_matches
        ):
            raise _invalid(
                "Evaluation, predictor, and cohort references must preserve the frozen target."
            )
        overlap = manifest.get("overlap", cohort["manifest"].get("overlap", {}))
        if standalone and (
            overlap.get("deferred")
            or not {"slideIds", "patientIds", "patientsComparable"} <= overlap.keys()
        ):
            raise _invalid(
                "Clinical utility requires the model evaluation's reviewed overlap evidence."
            )
        if overlap.get("slideIds") or overlap.get("patientIds") or overlap.get("sourceSlideIds"):
            raise _invalid(
                "Clinical utility requires an evaluation cohort without development overlap."
            )
        content = self.evaluations.artifact(selection.evaluationId, "predictions.json")
        metrics_content = self.evaluations.artifact(selection.evaluationId, "metrics.json")
        try:
            predictions, metrics = json.loads(content), json.loads(metrics_content)
            if not isinstance(metrics, dict):
                raise ValueError
            if any(
                metrics.get(key) != value
                for key, value in {
                    "classOrder": manifest["target"]["classes"],
                    "unit": manifest["target"]["unit"],
                    "positiveClass": manifest["target"].get("positiveClass"),
                    "decisionThreshold": manifest["inference"]["decisionThreshold"],
                    "patientAggregation": "mean_probabilities",
                }.items()
            ):
                raise _invalid("Saved metrics differ from the frozen evaluation settings.")
            if not isinstance(predictions, dict):
                raise ValueError
            records = _validate_records(predictions.get("records"), manifest["target"]["classes"])
            expected = {
                row["slideId"]: (row.get("patientId"), row.get("label"))
                for row in cohort["manifest"]["memberships"]
            }
            observed = {
                row["slideId"]: (row.get("patientId"), row.get("label"))
                for row in predictions["records"]
            }
            if expected != observed:
                raise _invalid(
                    "Saved predictions differ from the frozen evaluation cohort membership or labels."
                )
            memberships = {row["slideId"]: row for row in cohort["manifest"]["memberships"]}
            for row in records:
                source = memberships[row["slideId"]].get("patientIdSource")
                if "patientIdSource" in row and row["patientIdSource"] != source:
                    raise _invalid(
                        "Saved patient identity provenance differs from the frozen evaluation cohort."
                    )
                # Older predictions omitted this field. The frozen membership
                # remains authoritative, including acknowledged slide fallbacks.
                row["patientIdSource"] = source
            report = clinical_report(
                predictions, manifest["target"], manifest["inference"], selection
            )
            if overlap.get("patientsComparable") is False:
                report["warnings"].append(
                    "Development and evaluation patient identifier namespaces were declared independent. Cross-dataset patient overlap could not be verified; confirm the cohorts contain different patients before interpreting external performance."
                )
        except StorageError:
            # StorageError is a ValueError subclass. Preserve actionable clinical
            # findings instead of mislabeling valid-but-ineligible evidence as corrupt.
            raise
        except (ValueError, TypeError, KeyError, OverflowError, RecursionError) as error:
            raise _invalid("The saved evaluation predictions or metrics are malformed.") from error
        return {
            "kind": "clinical-analysis",
            "schemaVersion": 1,
            "name": selection.name,
            "datasetId": manifest["datasetId"],
            **({"datasetIds": manifest["datasetIds"]} if manifest.get("datasetIds") else {}),
            "experimentId": manifest["experimentId"],
            "predictorId": predictor["id"],
            "predictor": reference(predictor),
            "evaluationId": evaluation["id"],
            "evaluation": reference(evaluation),
            "cohortId": cohort["id"],
            "cohort": reference(cohort),
            "target": manifest["target"],
            "selection": selection.model_dump(),
            "source": {
                "predictionsSha256": hashlib.sha256(content).hexdigest(),
                "metricsSha256": hashlib.sha256(metrics_content).hexdigest(),
                "planHash": evaluation["execution"].get("planHash"),
                "inputHash": evaluation["execution"].get("result", {}).get("inputHash"),
            },
            "report": report,
        }

    def preview(self, selection):
        try:
            manifest = self._prepare(selection)
            return {
                "canSave": True,
                "previewHash": _hash(manifest),
                "manifest": manifest,
                "findings": [],
            }
        except StorageError as error:
            return {
                "canSave": False,
                "previewHash": None,
                "manifest": None,
                "findings": [finding(error.code, str(error))],
            }

    def save(self, request):
        selection = ClinicalSelection.model_validate(
            request.model_dump(include=set(ClinicalSelection.model_fields))
        )
        with lifecycle_guard(self.store.folder):
            prior = self.store.configuration_publication(request.operationId)
            if prior:
                manifest = prior["manifest"]
                if (
                    manifest.get("kind") != "clinical-analysis"
                    or manifest.get("selection") != selection.model_dump()
                    or manifest.get("previewHash") != request.previewHash
                ):
                    raise StorageError(
                        "This operation belongs to another analysis.", "OPERATION_CONFLICT", 409
                    )
                return lifecycle_document(self.store, prior)
            manifest = self._prepare(selection)
            if _hash(manifest) != request.previewHash:
                raise StorageError(
                    "Clinical analysis inputs changed. Preview again.", "PREVIEW_STALE", 409
                )
            return lifecycle_document(
                self.store,
                self.store.publish_configuration(
                    manifest={**manifest, "previewHash": request.previewHash},
                    operation_id=request.operationId,
                ),
            )

    def artifact(self, identity, filename):
        tables = {
            "operating-curves.csv": "operatingCurve",
            "calibration.csv": "calibration",
            "roc.csv": "rocCurve",
            "precision-recall.csv": "precisionRecallCurve",
        }
        if filename not in {*tables, "report.json"}:
            raise StorageError("Clinical artifact not found.", "CLINICAL_ARTIFACT_NOT_FOUND", 404)
        document = self.get(identity)
        if filename == "report.json":
            return json.dumps(
                document, allow_nan=False, ensure_ascii=False, indent=2
            ).encode(), "application/json"
        rows = document["manifest"]["report"][tables[filename]]
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        return stream.getvalue().encode("utf-8"), "text/csv"
