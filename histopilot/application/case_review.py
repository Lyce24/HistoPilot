"""Connect verified evaluation predictions to frozen metadata and human review."""

import csv
import hashlib
import io
import json
import math
from collections import Counter

from histopilot.application.clinical import _patient_records, _validate_records
from histopilot.application.evaluation_runs import EvaluationRunService, run_purpose
from histopilot.application.predictors import reference
from histopilot.application.slide_reviews import SlideReviewService, dataset_rows
from histopilot.inference_summary import describe, patient_member_probabilities
from histopilot.inference_summary import predicted_index as decision_index
from histopilot.models import catalog
from histopilot.schemas.case_review import CaseReviewQuery
from histopilot.storage.project_lock import StorageError

MAX_REVIEW_SLIDES = 10000
MAX_REVIEW_RESPONSE_BYTES = 32 * 1024 * 1024


def _invalid(message):
    return StorageError(message, "CASE_REVIEW_EVIDENCE_INVALID", 409)


def predicted_index(probabilities, target, inference):
    return decision_index(probabilities, target, inference["decisionThreshold"])


def development_patients(manifest):
    """Patients shared with development, or None when overlap cannot be compared."""
    overlap = manifest.get("overlap") or {}
    if not overlap.get("patientsComparable"):
        return None
    return set(overlap.get("patientIds") or [])


def development_flag(row, shared):
    """True/False for verified patient IDs; None when overlap cannot be established.

    A slide-ID fallback group can never match a development patient, so it is
    unknown rather than evidence of a new patient.
    """
    if shared is None or not row.get("patientId") or row.get("patientIdSource") == "slide_fallback":
        return None
    return row["patientId"] in shared


def cohort_metadata(store, cohort, selected_slides):
    """Frozen dataset rows and attribute labels for evaluated slides."""
    dataset_ids = cohort["manifest"].get("spec", {}).get("datasetIds") or [cohort["manifest"]["datasetId"]]
    lookup, dictionary = {}, {}
    for dataset_id in dataset_ids:
        dataset, rows = dataset_rows(store, dataset_id)
        for field in dataset["manifest"].get("dictionary", []):
            dictionary[field["key"]] = field.get("label") or field.get("sourceColumn") or field["key"]
        for slide_id in selected_slides.intersection(rows):
            if slide_id in lookup:
                raise _invalid("A reviewed slide resolves to more than one frozen dataset.")
            lookup[slide_id] = {**rows[slide_id], "datasetId": dataset_id}
    if set(lookup) != selected_slides:
        raise _invalid("Some evaluated slides cannot be resolved to their frozen dataset.")
    return lookup, dictionary


def _validate_members(row, classes):
    members = row.get("memberProbabilities")
    if members is None:
        if row.get("memberLogProbabilities") is not None:
            raise ValueError("Saved member log probabilities require member probability evidence.")
        return
    if (
        not isinstance(members, list)
        or len(members) < 2
        or any(
            not isinstance(values, list)
            or len(values) != classes
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not 0 <= value <= 1 for value in values
            )
            or abs(sum(values) - 1) > 1e-6
            for values in members
        )
    ):
        raise ValueError("Saved member probabilities must be normalized for every class.")
    logs = row.get("memberLogProbabilities")
    if logs is not None:
        if not isinstance(logs, list) or len(logs) != len(members):
            raise ValueError("Saved member log probabilities must match every ensemble member.")
        for probabilities, values in zip(members, logs, strict=True):
            if (
                not isinstance(values, list) or len(values) != classes
                or any(type(value) not in (int, float) or not math.isfinite(value) or value > 1e-6 for value in values)
            ):
                raise ValueError("Saved member log probabilities must be finite and ordered.")
            peak = max(values)
            normalizer = peak + math.log(math.fsum(math.exp(value - peak) for value in values))
            if abs(normalizer) > 1e-6 or any(
                abs(math.exp(value) - probability) > 1e-6
                for probability, value in zip(probabilities, values, strict=True)
            ):
                raise ValueError("Saved member log probabilities must match the probability evidence.")


def _sort_key(order):
    if order == "confidence_asc":
        return lambda row: (row["confidence"], row["id"])
    if order == "margin_asc":
        return lambda row: (row["margin"], row["id"])
    if order == "agreement_asc":
        def agreement(row):
            value = row["memberAgreement"]
            if value is None:
                return (1, 1.0, 0.0, row["confidence"], row["id"])
            return (0, value["agree"] / value["total"], -value["spread"], row["confidence"], row["id"])
        return agreement
    return lambda row: (-row["confidence"], row["id"])


def _outcome(actual, predicted, target):
    if actual is None:
        return "unlabeled"
    if actual == predicted:
        return "correct"
    if len(target["classes"]) == 2 and target.get("positiveClass") in target["classes"]:
        return "false_positive" if target["classes"][predicted] == target["positiveClass"] else "false_negative"
    return "error"


def _csv_text(value):
    value = str(value if value is not None else "")
    return "'" + value if value.lstrip().startswith(("=", "+", "-", "@", "\t", "\r")) else value


class CaseReviewService:
    def __init__(self, store, filesystem):
        self.store = store
        self.evaluations = EvaluationRunService(store, filesystem)
        self.reviews = SlideReviewService(store)

    def _supports_attention(self, evaluation):
        predictor = self.store.get_configuration(evaluation["manifest"]["predictorId"])
        recipe = predictor["manifest"].get("recipe", {})
        return catalog.supports_attention(recipe.get("model"), recipe.get("inputMode", "image"))

    def _evidence(self, identity, unit):
        evaluation = self.evaluations.get(identity)
        manifest = evaluation["manifest"]
        cohort = self.store.get_configuration(manifest["cohortId"])
        predictor = self.store.get_configuration(manifest["predictorId"])
        if (manifest["cohort"] != reference(cohort) or manifest["predictor"] != reference(predictor)
                or manifest["target"] != predictor["manifest"]["target"]):
            raise _invalid("The evaluation no longer matches its frozen predictor and cohort.")
        content = self.evaluations.artifact(identity, "predictions.json")
        try:
            source = json.loads(content)
            target = manifest["target"]
            if source["classOrder"] != target["classes"]:
                raise ValueError("Saved class order differs from the evaluation.")
            slides = _validate_records(source["records"], target["classes"])
            membership = {row["slideId"]: row for row in cohort["manifest"]["memberships"]}
            if len(membership) != len(cohort["manifest"]["memberships"]):
                raise ValueError("The cohort contains duplicate slide membership.")
            observed = {row["slideId"]: (row.get("patientId"), row.get("label")) for row in slides}
            expected = {key: (row.get("patientId"), row.get("label")) for key, row in membership.items()}
            if expected != observed:
                raise ValueError("Predictions differ from frozen cohort membership or labels.")
            for row in slides:
                _validate_members(row, len(target["classes"]))
            member_counts = {len(row.get("memberProbabilities") or []) for row in slides}
            checkpoint_count = len(predictor["manifest"].get("checkpoints") or [])
            if len(member_counts) > 1 or (
                checkpoint_count and member_counts != {0} and member_counts != {checkpoint_count}
            ):
                raise ValueError("Saved member probabilities must match the same frozen ensemble for every slide.")
            actual_unit = target["unit"] if unit == "selected" else unit
            if actual_unit == "patient":
                if any(not row.get("patientId") or row.get("patientIdSource") == "slide_fallback" for row in membership.values()):
                    raise ValueError("Patient review requires verified patient identities. Choose slide review for this cohort.")
                aggregation = manifest["inference"]["patientAggregation"]
                records = _patient_records(slides, source["patientRecords"], target["classes"], aggregation)
                groups = {}
                for row in slides:
                    groups.setdefault(row.get("patientId"), []).append(row)
                patients = []
                for row in records:
                    # Recompute from slide evidence; stored patient copies are not trusted.
                    members = patient_member_probabilities(groups[row["patientId"]], aggregation)
                    patient = {key: value for key, value in row.items() if key != "memberProbabilities"}
                    patients.append({**patient, **({"memberProbabilities": members} if members else {})})
                records = patients
            else:
                records = [{**row, "slideIds": [row["slideId"]]} for row in slides]
        except (ValueError, KeyError, TypeError, OverflowError, RecursionError) as error:
            if isinstance(error, StorageError):
                raise
            raise _invalid(str(error) or "The saved prediction evidence is invalid.") from error
        return evaluation, cohort, records, actual_unit, hashlib.sha256(content).hexdigest()

    def _query(self, identity, query, *, export=False):
        query = CaseReviewQuery.model_validate(query)
        evaluation, cohort, source, unit, checksum = self._evidence(identity, query.unit)
        manifest = evaluation["manifest"]
        target = manifest["target"]
        if any(index is not None and index >= len(target["classes"]) for index in (query.actualClass, query.predictedClass)):
            raise StorageError("Choose a class in this evaluation.", "CASE_REVIEW_FILTER_INVALID", 422)
        comparison = None
        other_rows = {}
        other_checksum = None
        key = "patientId" if unit == "patient" else "slideId"
        if query.comparisonId:
            comparison, other_cohort, other, other_unit, other_checksum = self._evidence(query.comparisonId, unit)
            if (cohort["id"] != other_cohort["id"] or target != comparison["manifest"]["target"]
                    or unit != other_unit or manifest["inference"]["patientAggregation"] != comparison["manifest"]["inference"]["patientAggregation"]):
                raise StorageError("Compare evaluations of the same frozen cohort, target, and patient aggregation.", "CASE_COMPARISON_MISMATCH", 409)
            other_rows = {row[key]: row for row in other}
            if set(other_rows) != {row[key] for row in source}:
                raise _invalid("Compared evaluations have different case membership.")
        elif query.outcome == "disagreement":
            raise StorageError("Select a second evaluation to review disagreements.", "CASE_COMPARISON_REQUIRED", 422)

        selected_slides = {slide for row in source for slide in row["slideIds"]}
        lookup, dictionary = cohort_metadata(self.store, cohort, selected_slides)
        if query.attribute and query.attribute not in dictionary:
            raise StorageError("Choose an attribute from the frozen data dictionary.", "CASE_REVIEW_FILTER_INVALID", 422)
        shared = development_patients(manifest)
        counts = Counter()
        items = []
        attributes = {field: set() for field in dictionary}
        for row in source:
            described = describe(row, target, manifest["inference"]["decisionThreshold"])
            predicted = described["predictedIndex"]
            agreement = described.get("memberAgreement")
            development = development_flag(row, shared)
            outcome = _outcome(row["labelIndex"], predicted, target)
            counts[outcome] += 1
            metadata = {}
            for slide_id in row["slideIds"]:
                for field, value in lookup[slide_id].get("attributes", {}).items():
                    if field not in metadata:
                        metadata[field] = []
                    if value not in metadata[field]:
                        metadata[field].append(value)
                    if field in attributes and value is not None and len(attributes[field]) < 200:
                        attributes[field].add(str(value))
            other = other_rows.get(row[key])
            comparison_value = None
            if other:
                predicted_other = predicted_index(other["probabilities"], target, comparison["manifest"]["inference"])
                comparison_value = {"probabilities": other["probabilities"], "predictedIndex": predicted_other,
                                    "predictedLabel": target["classes"][predicted_other], "disagrees": predicted_other != predicted}
                counts["disagreement"] += predicted_other != predicted
            if query.outcome == "error" and outcome in {"correct", "unlabeled"}:
                continue
            if query.outcome == "disagreement" and not comparison_value["disagrees"]:
                continue
            if query.outcome not in {"all", "error", "disagreement"} and outcome != query.outcome:
                continue
            if query.actualClass is not None and row["labelIndex"] != query.actualClass:
                continue
            if query.predictedClass is not None and predicted != query.predictedClass:
                continue
            confidence = described["confidence"]
            if confidence < query.minConfidence or confidence > query.maxConfidence:
                continue
            if query.maxMargin is not None and described["margin"] >= query.maxMargin:
                continue
            if query.developmentPatients != "all" and development is not (query.developmentPatients == "shared"):
                continue
            if query.memberDisagreement and not (agreement and agreement["agree"] < agreement["total"]):
                continue
            if (query.attribute and query.attributeValue is not None
                    and query.attributeValue not in {str(value) for value in metadata.get(query.attribute, []) if value is not None}):
                continue
            if query.search.strip() and query.search.strip().casefold() not in " ".join([str(row[key]), *row["slideIds"]]).casefold():
                continue
            items.append({"id": row[key], "patientId": row.get("patientId"), "slideIds": row["slideIds"],
                          "label": row["label"], "labelIndex": row["labelIndex"], "probabilities": row["probabilities"],
                          "predictedIndex": predicted, "predictedLabel": target["classes"][predicted],
                          "confidence": confidence, "margin": described["margin"],
                          "memberAgreement": agreement, "developmentPatient": development,
                          "outcome": outcome, "comparison": comparison_value, "attributes": metadata})
        items.sort(key=_sort_key(query.sort))
        total = len(items)
        page = items if export else items[query.offset:query.offset + query.limit]
        if sum(len(row["slideIds"]) for row in page) > MAX_REVIEW_SLIDES:
            raise StorageError("This selection contains too many slides to review at once. Narrow the filters or choose a smaller slide-level page.", "CASE_REVIEW_LIMIT", 413)
        review_bytes = 0
        for row in page:
            row["slides"] = []
            for slide_id in row["slideIds"]:
                slide = lookup[slide_id]
                review = self.reviews._read(slide["datasetId"], slide_id)
                review.pop("history")
                review_bytes += len(json.dumps(review, ensure_ascii=False).encode("utf-8"))
                if review_bytes > MAX_REVIEW_RESPONSE_BYTES:
                    raise StorageError("This selection contains too much review text. Narrow the filters before exporting or reviewing it.", "CASE_REVIEW_LIMIT", 413)
                row["slides"].append({"datasetId": slide["datasetId"], "slideId": slide_id, "slidePath": slide.get("slidePath"),
                                      "hasImage": bool(slide.get("slidePath")), "attributes": slide.get("attributes", {}), "review": review})
        member_count = max((len(row["memberProbabilities"]) for row in source if row.get("memberProbabilities")), default=None)
        return {"evaluationId": identity, "name": manifest["name"], "predictorId": manifest["predictorId"],
                "purpose": run_purpose(manifest), "memberCount": member_count,
                "developmentComparable": shared is not None,
                "supportsAttention": self._supports_attention(evaluation),
                "featureBundleId": manifest.get("features", {}).get("bundle", {}).get("id"),
                "packArtifactId": (
                    manifest["inference"].get("packArtifactId")
                    if manifest["inference"].get("loadingPolicy") == "packed" else None
                ),
                "cohortId": cohort["id"], "classOrder": target["classes"], "positiveClass": target.get("positiveClass"),
                "decisionThreshold": manifest["inference"]["decisionThreshold"], "unit": unit,
                "comparison": {"id": comparison["id"], "name": comparison["manifest"]["name"], "predictorId": comparison["manifest"]["predictorId"],
                               "supportsAttention": self._supports_attention(comparison),
                               "featureBundleId": comparison["manifest"].get("features", {}).get("bundle", {}).get("id"),
                               "packArtifactId": (
                                   comparison["manifest"]["inference"].get("packArtifactId")
                                   if comparison["manifest"]["inference"].get("loadingPolicy") == "packed" else None
                               ),
                               "decisionThreshold": comparison["manifest"]["inference"]["decisionThreshold"]} if comparison else None,
                "source": {"predictionsSha256": checksum, "comparisonSha256": other_checksum},
                "attributes": [{"key": field, "label": name, "values": sorted(attributes[field]), "valuesLimited": len(attributes[field]) >= 200} for field, name in dictionary.items()],
                "summary": {"total": len(source), **dict(counts)}, "items": page, "total": total, "offset": query.offset,
                "hasMore": not export and query.offset + query.limit < total}

    def query(self, identity, query):
        return self._query(identity, query)

    def export(self, identity, query):
        result = self._query(identity, query, export=True)
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        inference = result["purpose"] == "inference"
        writer.writerow([
            "Evaluation", "Unit", "Case", "Patient", *([] if inference else ["Actual"]),
            "Predicted", "Predicted_probability", *([] if inference else ["Outcome"]),
            "Comparison_predicted", "Dataset", "Slide", "Review", "Reviewer", "Reasons",
            "Notes", "Review_revision", "Predictions_SHA256",
        ])
        for row in result["items"]:
            for slide in row["slides"]:
                review = slide["review"]
                writer.writerow([_csv_text(value) for value in [
                    identity, result["unit"], row["id"], row["patientId"],
                    *([] if inference else [row["label"]]), row["predictedLabel"], row["confidence"],
                    *([] if inference else [row["outcome"]]), (row["comparison"] or {}).get("predictedLabel"),
                    slide["datasetId"], slide["slideId"], review["status"], review["reviewer"],
                    "; ".join(review["reasons"]), review["notes"], review["revision"],
                    result["source"]["predictionsSha256"],
                ]])
        return output.getvalue().encode("utf-8-sig")
