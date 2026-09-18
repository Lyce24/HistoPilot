"""Connect verified evaluation predictions to frozen metadata and human review."""

import csv
import hashlib
import io
import json
from collections import Counter

from histopilot.application.clinical import _patient_records, _validate_records
from histopilot.application.evaluation_runs import EvaluationRunService
from histopilot.application.predictors import reference
from histopilot.application.slide_reviews import SlideReviewService, dataset_rows
from histopilot.models import catalog
from histopilot.schemas.case_review import CaseReviewQuery
from histopilot.storage.project_lock import StorageError

MAX_REVIEW_SLIDES = 10000
MAX_REVIEW_RESPONSE_BYTES = 32 * 1024 * 1024


def _invalid(message):
    return StorageError(message, "CASE_REVIEW_EVIDENCE_INVALID", 409)


def predicted_index(probabilities, target, inference):
    classes = target["classes"]
    if len(classes) == 2 and target.get("positiveClass") in classes:
        positive = classes.index(target["positiveClass"])
        return positive if probabilities[positive] >= inference["decisionThreshold"] else 1 - positive
    return max(range(len(classes)), key=lambda index: probabilities[index])


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
            actual_unit = target["unit"] if unit == "selected" else unit
            if actual_unit == "patient":
                if any(not row.get("patientId") or row.get("patientIdSource") == "slide_fallback" for row in membership.values()):
                    raise ValueError("Patient review requires verified patient identities. Choose slide review for this cohort.")
                records = _patient_records(slides, source["patientRecords"], target["classes"], manifest["inference"]["patientAggregation"])
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

        dataset_ids = cohort["manifest"].get("spec", {}).get("datasetIds") or [cohort["manifest"]["datasetId"]]
        selected_slides = {slide for row in source for slide in row["slideIds"]}
        lookup, dictionary = {}, {}
        for dataset_id in dataset_ids:
            dataset, rows = dataset_rows(self.store, dataset_id)
            for field in dataset["manifest"].get("dictionary", []):
                dictionary[field["key"]] = field.get("label") or field.get("sourceColumn") or field["key"]
            for slide_id in selected_slides.intersection(rows):
                if slide_id in lookup:
                    raise _invalid("A reviewed slide resolves to more than one frozen dataset.")
                lookup[slide_id] = {**rows[slide_id], "datasetId": dataset_id}
        if set(lookup) != selected_slides:
            raise _invalid("Some evaluated slides cannot be resolved to their frozen dataset.")
        if query.attribute and query.attribute not in dictionary:
            raise StorageError("Choose an attribute from the frozen data dictionary.", "CASE_REVIEW_FILTER_INVALID", 422)
        counts = Counter()
        items = []
        attributes = {field: set() for field in dictionary}
        for row in source:
            predicted = predicted_index(row["probabilities"], target, manifest["inference"])
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
            confidence = row["probabilities"][predicted]
            if confidence < query.minConfidence:
                continue
            if (query.attribute and query.attributeValue is not None
                    and query.attributeValue not in {str(value) for value in metadata.get(query.attribute, []) if value is not None}):
                continue
            if query.search.strip() and query.search.strip().casefold() not in " ".join([str(row[key]), *row["slideIds"]]).casefold():
                continue
            items.append({"id": row[key], "patientId": row.get("patientId"), "slideIds": row["slideIds"],
                          "label": row["label"], "labelIndex": row["labelIndex"], "probabilities": row["probabilities"],
                          "predictedIndex": predicted, "predictedLabel": target["classes"][predicted],
                          "confidence": confidence, "outcome": outcome, "comparison": comparison_value, "attributes": metadata})
        items.sort(key=lambda row: (-row["confidence"], row["id"]))
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
        return {"evaluationId": identity, "name": manifest["name"], "predictorId": manifest["predictorId"],
                "supportsAttention": self._supports_attention(evaluation),
                "featureBundleId": manifest.get("features", {}).get("bundle", {}).get("id"),
                "cohortId": cohort["id"], "classOrder": target["classes"], "positiveClass": target.get("positiveClass"),
                "decisionThreshold": manifest["inference"]["decisionThreshold"], "unit": unit,
                "comparison": {"id": comparison["id"], "name": comparison["manifest"]["name"], "predictorId": comparison["manifest"]["predictorId"],
                               "supportsAttention": self._supports_attention(comparison),
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
        writer.writerow(["Evaluation", "Unit", "Case", "Patient", "Actual", "Predicted", "Predicted_probability", "Outcome", "Comparison_predicted", "Dataset", "Slide", "Review", "Reviewer", "Reasons", "Notes", "Review_revision", "Predictions_SHA256"])
        for row in result["items"]:
            for slide in row["slides"]:
                review = slide["review"]
                writer.writerow([_csv_text(value) for value in [identity, result["unit"], row["id"], row["patientId"], row["label"], row["predictedLabel"], row["confidence"], row["outcome"], (row["comparison"] or {}).get("predictedLabel"), slide["datasetId"], slide["slideId"], review["status"], review["reviewer"], "; ".join(review["reasons"]), review["notes"], review["revision"], result["source"]["predictionsSha256"]]])
        return output.getvalue().encode("utf-8-sig")
