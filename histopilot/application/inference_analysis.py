"""Label-free views, exports and attention requests over verified inference predictions."""

import csv
import io
from collections import defaultdict

from histopilot.application.case_review import (
    CaseReviewService,
    _csv_text,
    cohort_metadata,
    development_flag,
    development_patients,
)
from histopilot.application.evaluation_runs import run_purpose
from histopilot.inference_summary import agreement, class_counts, cross_tab, describe, summarize
from histopilot.schemas.interpretation import VisualizeInterpretation
from histopilot.storage.io import content_hash
from histopilot.storage.project_lock import StorageError

MISSING = "Missing"
MIXED = "Mixed"
MAX_EXPORT_ATTRIBUTES = 200
# Exporting every frozen attribute has a wider, fixed ceiling.
MAX_DEFAULT_EXPORT_ATTRIBUTES = 1000


def _attribute_value(values):
    present = [value for value in values if value is not None]
    if not present:
        return MISSING
    distinct = {str(value) for value in present}
    return next(iter(distinct)) if len(distinct) == 1 else MIXED


class InferenceAnalysisService:
    def __init__(self, store, filesystem):
        self.store, self.filesystem = store, filesystem
        self.cases = CaseReviewService(store, filesystem)
        self.evaluations = self.cases.evaluations

    def _evidence(self, identity, unit):
        evaluation, cohort, records, actual_unit, checksum = self.cases._evidence(identity, unit)
        manifest = evaluation["manifest"]
        target = manifest["target"]
        threshold = manifest["inference"]["decisionThreshold"]
        described = [describe(row, target, threshold) for row in records]
        return evaluation, cohort, records, described, actual_unit, checksum

    @staticmethod
    def _attributes(records, lookup, field):
        return [
            _attribute_value(
                [lookup[slide].get("attributes", {}).get(field) for slide in row["slideIds"]]
            )
            for row in records
        ]

    def summary(self, identity, query):
        evaluation, cohort, records, described, unit, checksum = self._evidence(
            identity, query.unit
        )
        manifest = evaluation["manifest"]
        target = manifest["target"]
        classes = target["classes"]
        threshold = manifest["inference"]["decisionThreshold"]
        result = {
            "evaluationId": identity,
            "name": manifest["name"],
            "purpose": run_purpose(manifest),
            "predictorId": manifest["predictorId"],
            "cohortId": cohort["id"],
            "unit": unit,
            "task": target["task"],
            "classOrder": classes,
            "positiveClass": target.get("positiveClass"),
            "decisionThreshold": threshold,
            "patientAggregation": manifest["inference"]["patientAggregation"],
            "source": {"predictionsSha256": checksum},
            **summarize(records, target, threshold, described=described),
            "patients": 0
            if manifest.get("splitUnit") == "slide"
            else len({row["patientId"] for row in records if row.get("patientId")}),
        }
        shared = development_patients(manifest)
        if shared is None:
            result["development"] = {"comparable": False}
        else:
            flags = [development_flag(row, shared) for row in records]
            pairs = list(zip(described, flags, strict=True))
            result["development"] = {
                "comparable": True,
                "patients": len(shared),
                "records": sum(flag is True for flag in flags),
                "unknown": sum(flag is None for flag in flags),
                "shared": class_counts([item for item, flag in pairs if flag is True], classes),
                "new": class_counts([item for item, flag in pairs if flag is False], classes),
            }
        selected = {slide for row in records for slide in row["slideIds"]}
        lookup, dictionary = cohort_metadata(self.store, cohort, selected)
        result["attributes"] = [{"key": key, "label": label} for key, label in dictionary.items()]
        if query.attribute:
            if query.attribute not in dictionary:
                raise StorageError(
                    "Choose an attribute from the frozen data dictionary.",
                    "INFERENCE_ATTRIBUTE_INVALID",
                    422,
                )
            result["breakdown"] = {
                "attribute": query.attribute,
                "label": dictionary[query.attribute],
                **cross_tab(described, self._attributes(records, lookup, query.attribute), classes),
            }
        if query.comparisonId:
            result["comparison"] = self._compare(
                evaluation, cohort, records, described, unit, query.comparisonId
            )
        return result

    def _compare(self, evaluation, cohort, records, described, unit, other_id):
        if other_id == evaluation["id"]:
            raise StorageError(
                "Choose a different run to compare.", "INFERENCE_COMPARISON_INVALID", 422
            )
        other, other_cohort, other_records, other_described, other_unit, checksum = (
            self._evidence(other_id, unit)
        )
        manifest, second = evaluation["manifest"], other["manifest"]
        # Match the case review's pairing rule so disagreements can be opened there.
        if (
            other_cohort["id"] != cohort["id"]
            or other_unit != unit
            or second["target"] != manifest["target"]
            or second["inference"]["patientAggregation"] != manifest["inference"]["patientAggregation"]
        ):
            raise StorageError(
                "Compare runs on the same frozen cohort, target, prediction unit and patient aggregation.",
                "INFERENCE_COMPARISON_MISMATCH",
                409,
            )
        key = "patientId" if unit == "patient" else "slideId"
        decisions = {
            row[key]: item["predictedIndex"]
            for row, item in zip(other_records, other_described, strict=True)
        }
        if set(decisions) != {row[key] for row in records}:
            raise StorageError(
                "Compared runs predict different cases.", "INFERENCE_COMPARISON_MISMATCH", 409
            )
        left = [item["predictedIndex"] for item in described]
        right = [decisions[row[key]] for row in records]
        return {
            "evaluationId": other["id"],
            "name": second["name"],
            "predictorId": second["predictorId"],
            "decisionThreshold": second["inference"]["decisionThreshold"],
            "predictionsSha256": checksum,
            **agreement(left, right, manifest["target"]["classes"]),
        }

    def export(self, identity, query):
        evaluation, cohort, records, described, unit, checksum = self._evidence(
            identity, query.unit
        )
        manifest = evaluation["manifest"]
        classes = manifest["target"]["classes"]
        selected = {slide for row in records for slide in row["slideIds"]}
        lookup, dictionary = cohort_metadata(self.store, cohort, selected)
        fields = list(dictionary) if query.attributes is None else query.attributes
        unknown = [field for field in fields if field not in dictionary]
        limit = MAX_DEFAULT_EXPORT_ATTRIBUTES if query.attributes is None else MAX_EXPORT_ATTRIBUTES
        if unknown or len(fields) > limit:
            raise StorageError(
                "Choose attributes from the frozen data dictionary.",
                "INFERENCE_ATTRIBUTE_INVALID",
                422,
            )
        shared = development_patients(manifest)
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        headers = [
            "Run", "Unit", "Case", "Patient_ID", "Slide_IDs", "Predicted", "Confidence", "Margin",
            *[f"P({label})" for label in classes],
            "Members_agreeing", "Members", "Development_patient",
        ]
        used = {_csv_text(value) for value in [*headers, "Predictions_SHA256"]}
        for field in fields:
            label = dictionary[field]
            if _csv_text(label) in used:
                label = f"Attribute: {label} [{field}]"
            while _csv_text(label) in used:
                label = f"Attribute: {label}"
            headers.append(label)
            used.add(_csv_text(label))
        writer.writerow([_csv_text(value) for value in [*headers, "Predictions_SHA256"]])
        values = {field: self._attributes(records, lookup, field) for field in fields}
        for index, (row, item) in enumerate(zip(records, described, strict=True)):
            members = item.get("memberAgreement")
            flag = development_flag(row, shared)
            development = "" if flag is None else "yes" if flag else "no"
            writer.writerow([_csv_text(value) for value in [
                identity, unit, row["patientId"] if unit == "patient" else row["slideId"],
                row.get("patientId"), ";".join(row["slideIds"]), item["predictedLabel"],
                item["confidence"], item["margin"], *row["probabilities"],
                members["agree"] if members else "", members["total"] if members else "",
                development, *[values[field][index] for field in fields], checksum,
            ]])
        return output.getvalue().encode("utf-8-sig")

    def attention(self, identity, request):
        """Queue attention for run slides through the reusable visualization flow."""
        from histopilot.application.interpretation import InterpretationService

        evaluation = self.evaluations.get(identity)
        manifest = evaluation["manifest"]
        cohort = self.store.get_configuration(manifest["cohortId"])
        members = {row["slideId"] for row in cohort["manifest"]["memberships"]}
        missing = [slide for slide in request.slideIds if slide not in members]
        if missing:
            raise StorageError(
                "Choose slides predicted by this run.", "INFERENCE_SLIDE_UNAVAILABLE", 422
            )
        lookup, _dictionary = cohort_metadata(self.store, cohort, set(request.slideIds))
        interpretation = InterpretationService(self.store, self.filesystem)
        folders = defaultdict(list)
        sources = {}
        for slide in request.slideIds:
            row = lookup[slide]
            if not row.get("slidePath"):
                raise StorageError(
                    f"Slide {slide} has no linked image in its frozen dataset.",
                    "INFERENCE_SLIDE_IMAGE_MISSING",
                    422,
                )
            dataset = row["datasetId"]
            if dataset not in sources:
                sources[dataset] = interpretation.gallery.dataset_source(dataset)
            source = sources[dataset]
            if source["slideFolder"] is None:
                finding = source["slideFolderFinding"]
                raise StorageError(finding["message"], finding["code"], 422)
            folders[source["slideFolder"]].append(row["slidePath"])
        packed = manifest["inference"].get("loadingPolicy") == "packed"
        items, documents = [], []
        for folder, paths in sorted(folders.items()):
            result = interpretation.visualize(
                VisualizeInterpretation(
                    slideFolder=folder,
                    featureBundleId=manifest["features"]["bundle"]["id"],
                    packArtifactId=manifest["inference"].get("packArtifactId") if packed else None,
                    predictorId=manifest["predictorId"],
                    evaluationId=identity,
                    slidePaths=paths,
                    patchWidthLevel0=request.patchWidthLevel0,
                    patchHeightLevel0=request.patchHeightLevel0,
                    resources=request.resources,
                    operationId=f"inference-attention-{content_hash([request.operationId, folder])}"[:128],
                )
            )
            items.extend(result["items"])
            documents.extend(result["interpretations"])
        return {"evaluationId": identity, "items": items, "interpretations": documents}
