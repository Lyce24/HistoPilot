"""Reference standards: labels attached to a frozen cohort after it was frozen.

A cohort's own labels are fixed when it is frozen, and an unlabeled cohort has none. A
reference standard adds labels afterwards, without changing the cohort or rerunning a
model: one column of frozen datasets (the cohort's own, or newer versions matched by slide
ID) mapped to a class set. Every completed run on the cohort with the same classes can then
be scored against it, and references can be compared with each other and with a run's
predictions. Missing values, and values the mapping leaves out, stay unlabeled.

A reference is a frozen, content-addressed record, like the cohort it labels.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from histopilot.application.predictors import lifecycle_document, reference
from histopilot.application.protocols import ProtocolService
from histopilot.schemas.references import ReferenceSelection
from histopilot.storage.io import content_hash
from histopilot.storage.lifecycle import lifecycle_guard
from histopilot.storage.project_lock import StorageError

KIND = "reference-standard"
# The column's values over the cohort's slides, so a mapping can be made and completed.
VALUES_LISTED = 200


def source_value(raw):
    """A dataset value as the data browser and the mapping show it."""
    return None if raw is None else str(raw)


class ReferenceService:
    def __init__(self, store, filesystem):
        self.store, self.filesystem = store, filesystem

    def list(self, *, cohort_id=None, include_inactive=False):
        """References without their per-slide memberships, which can be large."""
        return {
            "items": [
                {
                    **document,
                    "manifest": {
                        key: value
                        for key, value in document["manifest"].items()
                        if key != "memberships"
                    },
                }
                for document in (
                    lifecycle_document(self.store, item)
                    for item in self.store.list_configurations(
                        KIND, include_inactive=include_inactive
                    )
                )
                if cohort_id is None or document["manifest"]["cohortId"] == cohort_id
            ]
        }

    def get(self, identity):
        document = self.store.get_configuration(identity)
        if document["manifest"].get("kind") != KIND:
            raise StorageError("Reference standard not found.", "REFERENCE_NOT_FOUND", 404)
        return lifecycle_document(self.store, document)

    def _prepare(self, selection):
        cohort = self.store.get_configuration(selection.cohortId)
        if cohort["manifest"].get("kind") != "evaluation-cohort":
            raise StorageError("Choose a frozen cohort.", "REFERENCE_COHORT_INVALID", 422)
        self.store.lifecycle.assert_usable([f"configuration:{cohort['id']}"])
        protocols = ProtocolService(self.store, self.filesystem)
        datasets = [protocols._load_dataset(identity) for identity in selection.datasetIds]
        findings = []

        def finding(code, message, severity="error"):
            findings.append({"severity": severity, "code": code, "message": message})

        if any(selection.field not in dictionary for _dataset, dictionary, _records in datasets):
            finding(
                "REFERENCE_FIELD_UNKNOWN",
                f"Column '{selection.field}' is not in every chosen dataset's data dictionary.",
            )
        rows = defaultdict(list)
        for _dataset, _dictionary, records in datasets:
            for row in records:
                rows[row["slideId"]].append(row)
        memberships = cohort["manifest"]["memberships"]
        labeled, counts, unmapped, values = [], Counter(), Counter(), Counter()
        missing = unmatched = 0
        for member in memberships:
            source = rows.get(member["slideId"])
            label = None
            if not source:
                unmatched += 1
            elif len(source) > 1:
                finding(
                    "REFERENCE_DUPLICATE_SLIDES",
                    "The chosen datasets list a cohort slide more than once, so its label is ambiguous.",
                )
            else:
                raw = source_value(source[0]["attributes"].get(selection.field))
                values[raw] += 1
                if raw is None:
                    missing += 1
                else:
                    label = selection.labels.get(raw)
                    if label is None:
                        unmapped[raw] += 1
            if label is not None:
                counts[label] += 1
            labeled.append({"slideId": member["slideId"], "label": label})
        if not counts:
            finding(
                "REFERENCE_NO_LABELS",
                "No cohort slide receives a label. Map the column's values to the classes.",
            )
        elif set(counts) != set(selection.classes):
            finding(
                "REFERENCE_CLASSES_ABSENT",
                "Some classes label no cohort slide, so some metrics will be unavailable.",
                "warning",
            )
        if unmatched:
            finding(
                "REFERENCE_UNMATCHED_SLIDES",
                f"{unmatched:,} cohort slides are not in the chosen datasets; they stay unlabeled.",
                "warning",
            )
        if unmapped:
            finding(
                "REFERENCE_UNMAPPED_VALUES",
                f"{sum(unmapped.values()):,} slides have values the mapping leaves out; "
                "they stay unlabeled.",
                "warning",
            )
        patients = defaultdict(set)
        for member, row in zip(memberships, labeled, strict=True):
            if member.get("patientId") and row["label"] is not None:
                patients[member["patientId"]].add(row["label"])
        conflicting = sum(len(values) > 1 for values in patients.values())
        if conflicting and cohort["manifest"]["spec"].get("splitUnit") != "slide":
            finding(
                "REFERENCE_CONFLICTING_PATIENTS",
                f"{conflicting:,} patients have slides with different labels. Patient-level "
                "scoring leaves them unlabeled.",
                "warning",
            )
        manifest = {
            "kind": KIND,
            "name": selection.name,
            "cohortId": cohort["id"],
            "cohort": reference(cohort),
            # The store requires one dataset per record; every source is listed below it.
            "datasetId": datasets[0][0]["id"],
            "datasets": [reference(dataset) for dataset, _dictionary, _records in datasets],
            "field": selection.field,
            "matchedBy": "slideId",
            "classes": list(selection.classes),
            "labels": dict(sorted(selection.labels.items())),
            "memberships": labeled,
            "summary": {
                "slides": len(memberships),
                "labeledSlides": sum(counts.values()),
                "missingSlides": missing,
                "unmappedSlides": sum(unmapped.values()),
                "unmatchedSlides": unmatched,
                "classCounts": {name: counts[name] for name in selection.classes},
                "conflictingPatients": conflicting,
                # Missing values are counted above; these are the values present.
                "values": [
                    {"value": value, "count": count, "label": selection.labels.get(value)}
                    for value, count in sorted(
                        ((value, count) for value, count in values.items() if value is not None),
                        key=lambda item: (-item[1], item[0]),
                    )[:VALUES_LISTED]
                ],
                "valuesTruncated": sum(value is not None for value in values) > VALUES_LISTED,
            },
            "findings": [item for item in _unique(findings) if item["severity"] != "error"],
        }
        errors = [item for item in _unique(findings) if item["severity"] == "error"]
        return manifest, errors

    def preview(self, selection):
        try:
            manifest, errors = self._prepare(selection)
        except StorageError as error:
            return {
                "canSave": False,
                "previewHash": None,
                "manifest": None,
                "findings": [{"severity": "error", "code": error.code, "message": str(error)}],
            }
        return {
            "canSave": not errors,
            "previewHash": None if errors else content_hash(manifest),
            "manifest": manifest,
            "findings": errors + manifest["findings"],
        }

    def save(self, request):
        selection = ReferenceSelection.model_validate(
            request.model_dump(include=set(ReferenceSelection.model_fields))
        )
        with lifecycle_guard(self.store.folder):
            prior = self.store.configuration_publication(request.operationId)
            if prior:
                manifest = prior["manifest"]
                if (
                    manifest.get("kind") != KIND
                    or manifest.get("previewHash") != request.previewHash
                ):
                    raise StorageError(
                        "This operation belongs to another record.", "OPERATION_CONFLICT", 409
                    )
                return lifecycle_document(self.store, prior)
            manifest, errors = self._prepare(selection)
            if errors:
                raise StorageError(errors[0]["message"], errors[0]["code"], 422, findings=errors)
            if content_hash(manifest) != request.previewHash:
                raise StorageError(
                    "The reference standard's inputs changed. Preview it again.",
                    "PREVIEW_STALE",
                    409,
                )
            return lifecycle_document(
                self.store,
                self.store.publish_configuration(
                    manifest={**manifest, "previewHash": request.previewHash},
                    operation_id=request.operationId,
                ),
            )

    def for_run(self, manifest, identity):
        """A reference usable for the run ``manifest``: same frozen cohort and classes."""
        document = self.get(identity)
        self.store.lifecycle.assert_usable([f"configuration:{identity}"])
        reference_manifest = document["manifest"]
        if (
            reference_manifest["cohortId"] != manifest["cohortId"]
            or reference_manifest["cohort"] != manifest["cohort"]
        ):
            raise StorageError(
                "This reference standard labels another cohort.", "REFERENCE_COHORT_MISMATCH", 409
            )
        if sorted(reference_manifest["classes"]) != sorted(manifest["target"]["classes"]):
            raise StorageError(
                "This reference standard's classes differ from the run's classes.",
                "REFERENCE_CLASSES_MISMATCH",
                409,
            )
        return document


def _unique(findings):
    seen, unique = set(), []
    for item in findings:
        key = (item["code"], item["message"])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique
