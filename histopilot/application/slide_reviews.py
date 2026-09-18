"""Versioned review notes and QC decisions; never edits frozen dataset artifacts."""

import hashlib
import json
from datetime import UTC, datetime

from pydantic import ValidationError

from histopilot.schemas.slide_reviews import (
    SaveSlideReview,
    SlideReviewDocument,
    SlideReviewValues,
)
from histopilot.storage.lifecycle import lifecycle_guard
from histopilot.storage.project_lock import (
    StorageError,
    _reject_symlink_components,
    ensure_managed_directory,
    writer_lock,
)
from histopilot.storage.scientific import ScientificStore
from histopilot.workers.packing_process import write_json

MAX_REVIEW_BYTES = 8 * 1024 * 1024


def dataset_rows(store, dataset_id):
    """Read the checksummed, frozen metadata rather than the original spreadsheet."""
    dataset = store.get_dataset(dataset_id)
    try:
        rows = json.loads(store.read_artifact(dataset_id, "records.json"))
        if not isinstance(rows, list) or len(rows) > 50000:
            raise ValueError
        index = {}
        for row in rows:
            if (not isinstance(row, dict) or not isinstance(row.get("slideId"), str)
                    or not row["slideId"] or row["slideId"] in index
                    or not isinstance(row.get("attributes", {}), dict)):
                raise ValueError
            index[row["slideId"]] = row
    except (ValueError, UnicodeError, TypeError, RecursionError) as error:
        if isinstance(error, StorageError):
            raise
        raise StorageError("Frozen dataset records are invalid.", "REVIEW_DATASET_INVALID") from error
    return dataset, index


class SlideReviewService:
    def __init__(self, store):
        self.store = store

    def _folder(self, dataset_id):
        # Dataset lookup validates the identity; both components are still hashed.
        digest = hashlib.sha256(dataset_id.encode()).hexdigest()
        path = self.store.folder / "slide-reviews" / digest
        _reject_symlink_components(path)
        return path

    def _path(self, dataset_id, slide_id):
        return self._folder(dataset_id) / (hashlib.sha256(slide_id.encode()).hexdigest() + ".json")

    def _read(self, dataset_id, slide_id):
        path = self._path(dataset_id, slide_id)
        _reject_symlink_components(path)
        if not path.exists():
            return SlideReviewDocument(datasetId=dataset_id, slideId=slide_id).model_dump()
        try:
            value = SlideReviewDocument.model_validate_json(
                ScientificStore._read_file(path, MAX_REVIEW_BYTES)
            ).model_dump()
            if value["datasetId"] != dataset_id or value["slideId"] != slide_id:
                raise ValueError
            return value
        except (ValueError, ValidationError, UnicodeError) as error:
            if isinstance(error, StorageError):
                raise
            raise StorageError("Saved slide review failed validation.", "SLIDE_REVIEW_CORRUPT") from error

    def get(self, dataset_id, slide_id):
        _, rows = dataset_rows(self.store, dataset_id)
        if slide_id not in rows:
            raise StorageError("This slide is not in the frozen dataset.", "REVIEW_SLIDE_NOT_FOUND", 404)
        return self._read(dataset_id, slide_id)

    def list(self, dataset_id, *, offset=0, limit=200):
        _, rows = dataset_rows(self.store, dataset_id)
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 500:
            raise StorageError("Choose a valid review page.", "INVALID_REVIEW_PAGE", 422)
        folder = self._folder(dataset_id)
        if not folder.exists():
            return {"items": [], "total": 0, "offset": offset, "hasMore": False}
        paths = sorted(folder.glob("*.json"))
        if len(paths) > len(rows):
            raise StorageError("Review inventory exceeds the frozen dataset.", "SLIDE_REVIEW_CORRUPT")
        items = []
        for path in paths[offset:offset + limit]:
            try:
                saved = SlideReviewDocument.model_validate_json(
                    ScientificStore._read_file(path, MAX_REVIEW_BYTES)
                ).model_dump()
                if (saved["datasetId"] != dataset_id or saved["slideId"] not in rows
                        or path != self._path(dataset_id, saved["slideId"])):
                    raise ValueError
            except (ValueError, ValidationError) as error:
                if isinstance(error, StorageError):
                    raise
                raise StorageError("Saved slide review failed validation.", "SLIDE_REVIEW_CORRUPT") from error
            saved.pop("history")
            items.append(saved)
        return {"items": items, "total": len(paths), "offset": offset,
                "hasMore": offset + limit < len(paths)}

    def save(self, dataset_id, slide_id, request):
        request = SaveSlideReview.model_validate(request)
        values = request.model_dump(include=set(SlideReviewValues.model_fields))
        with lifecycle_guard(self.store.folder):
            _, rows = dataset_rows(self.store, dataset_id)
            if slide_id not in rows:
                raise StorageError("This slide is not in the frozen dataset.", "REVIEW_SLIDE_NOT_FOUND", 404)
            if request.evaluationId:
                evaluation = self.store.get_configuration(request.evaluationId)["manifest"]
                if evaluation.get("kind") != "model-evaluation":
                    raise StorageError("Choose a saved model evaluation.", "REVIEW_CONTEXT_INVALID", 422)
                cohort = self.store.get_configuration(evaluation["cohortId"])["manifest"]
                dataset_ids = cohort.get("spec", {}).get("datasetIds") or [cohort["datasetId"]]
                if dataset_id not in dataset_ids or not any(
                    row.get("slideId") == slide_id for row in cohort["memberships"]
                ):
                    raise StorageError("Review context must include this dataset and slide.", "REVIEW_CONTEXT_INVALID", 422)
            self.store.lifecycle.assert_usable([f"dataset:{dataset_id}"])
            with writer_lock(self.store.folder):
                current = self._read(dataset_id, slide_id)
                existing = {key: current[key] for key in SlideReviewValues.model_fields}
                if request.expectedRevision != current["revision"]:
                    # A lost-response retry is safe only for the immediately preceding save.
                    if request.expectedRevision + 1 == current["revision"] and existing == values:
                        return current
                    raise StorageError("This review changed in another tab. Your edits are preserved; reload the saved review before merging.",
                                       "SLIDE_REVIEW_CONFLICT", 409)
                if current["revision"] and existing == values:
                    return current
                event = {**values, "revision": current["revision"] + 1,
                         "updatedAt": datetime.now(UTC).isoformat()}
                document = {**current, **event, "history": [*current["history"], event]}
                encoded = json.dumps(document, indent=2, allow_nan=False).encode() + b"\n"
                if len(encoded) > MAX_REVIEW_BYTES:
                    raise StorageError("This review has reached its history storage limit.", "SLIDE_REVIEW_LIMIT", 413)
                ensure_managed_directory(self._folder(dataset_id))
                write_json(self._path(dataset_id, slide_id), document)
                return document
