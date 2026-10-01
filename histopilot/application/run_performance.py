"""Performance of a labeled run within each value of a frozen dataset attribute.

Subgroups score exactly the records the run's metrics score: labeled units, without the
development patients a slide-level predictor may predict in a label-blind run. The rows
describe performance; they are not tests. Small groups give unstable estimates, and
comparing many groups finds gaps by chance.
"""

from histopilot.application.case_review import CaseReviewService, cohort_metadata
from histopilot.application.evaluation_runs import run_purpose
from histopilot.application.inference_analysis import attribute_values
from histopilot.application.run_evidence import (
    development_flag,
    development_patients,
    labels_withheld,
)
from histopilot.cv_summary import evaluation_metrics
from histopilot.storage.project_lock import StorageError

# Distinct values shown before the rest are pooled, as in the prediction breakdown.
MAX_GROUPS = 50


class RunPerformanceService:
    def __init__(self, store, filesystem):
        self.store, self.filesystem = store, filesystem
        self.cases = CaseReviewService(store, filesystem)

    def breakdown(self, identity, query):
        if (
            not query.referenceId
            and run_purpose(self.cases.evaluations.record(identity)["manifest"]) == "inference"
        ):
            raise StorageError(
                "This run predicts an unlabeled cohort. Score it against a reference standard "
                "to break its performance down.",
                "PERFORMANCE_REQUIRES_LABELS",
                409,
            )
        evaluation, cohort, records, unit, checksum = self.cases._evidence(
            identity, query.unit, query.referenceId
        )
        manifest = evaluation["manifest"]
        target = manifest["target"]
        threshold = manifest["inference"]["decisionThreshold"]
        excluded = 0
        # Leave out exactly the units the matching metrics leave out: service scoring (every
        # label-blind run, and every score against a reference standard) excludes them.
        if labels_withheld(manifest) or query.referenceId:
            shared = development_patients(manifest)
            scored = [row for row in records if development_flag(row, shared) is not True]
            excluded = len(records) - len(scored)
            records = scored
        selected = {slide for row in records for slide in row["slideIds"]}
        lookup, dictionary = cohort_metadata(self.store, cohort, selected)
        if query.attribute not in dictionary:
            raise StorageError(
                "Choose an attribute from the frozen data dictionary.",
                "PERFORMANCE_ATTRIBUTE_INVALID",
                422,
            )
        groups = {}
        for row, value in zip(
            records, attribute_values(records, lookup, query.attribute), strict=True
        ):
            groups.setdefault(value, []).append(row)
        ordered = sorted(
            groups.items(),
            key=lambda entry: (-sum(row["labelIndex"] is not None for row in entry[1]), entry[0]),
        )
        remainder = [row for _, rows in ordered[MAX_GROUPS:] for row in rows]
        return {
            "evaluationId": identity,
            "unit": unit,
            "attribute": query.attribute,
            "label": dictionary[query.attribute],
            "classOrder": target["classes"],
            "positiveClass": target.get("positiveClass"),
            "decisionThreshold": threshold,
            "overall": evaluation_metrics(records, target, threshold),
            "rows": [
                {"value": value, **evaluation_metrics(rows, target, threshold)}
                for value, rows in ordered[:MAX_GROUPS]
            ],
            "otherValues": max(0, len(ordered) - MAX_GROUPS),
            **({"other": evaluation_metrics(remainder, target, threshold)} if remainder else {}),
            "developmentExcluded": excluded,
            "source": {"predictionsSha256": checksum},
        }
