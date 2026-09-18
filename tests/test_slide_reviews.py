import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import ValidationError

from histopilot.application.slide_reviews import SlideReviewService
from histopilot.schemas.slide_reviews import SaveSlideReview
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import ScientificStore


@pytest.fixture
def reviews(tmp_path):
    store = ScientificStore(tmp_path, "review-project")
    draft = store.create_draft("import", "Review dataset", {})
    records = [{"slideId": "slide / 01", "patientId": "patient-1", "attributes": {"grade": "high"}}]
    dataset = store.publish_dataset(draft["id"], expected_revision=1,
                                    manifest={"kind": "dataset", "name": "Review dataset"},
                                    artifacts={"records.json": json.dumps(records).encode()}, operation_id="dataset")
    return SlideReviewService(store), dataset["id"], records[0]["slideId"]


def test_review_history_and_retry_preserve_frozen_labels(reviews):
    service, dataset, slide = reviews
    before = service.store.read_artifact(dataset, "records.json")
    assert service.get(dataset, slide)["revision"] == 0
    request = SaveSlideReview(expectedRevision=0, status="exclude", notes="Low tissue", reviewer="YL")
    first = service.save(dataset, slide, request)
    assert first["revision"] == 1 and first["history"][0]["status"] == "exclude"
    assert service.save(dataset, slide, request) == first
    with pytest.raises(StorageError, match="another tab") as error:
        service.save(dataset, slide, SaveSlideReview(expectedRevision=0, notes="stale"))
    assert error.value.code == "SLIDE_REVIEW_CONFLICT"
    second = service.save(dataset, slide, SaveSlideReview(expectedRevision=1, status="review", notes="Needs adjudication"))
    assert [entry["revision"] for entry in second["history"]] == [1, 2]
    assert service.store.read_artifact(dataset, "records.json") == before
    assert service.get(dataset, slide) == second
    assert service.list(dataset)["items"][0]["revision"] == 2
    assert "history" not in service.list(dataset)["items"][0]


def test_concurrent_review_writers_have_one_winner(reviews):
    service, dataset, slide = reviews
    def save(note):
        try:
            return service.save(dataset, slide, SaveSlideReview(expectedRevision=0, notes=note))["revision"]
        except StorageError as error:
            return error.code
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(save, ["first", "second"]))
    assert sorted(map(str, results)) == ["1", "SLIDE_REVIEW_CONFLICT"]
    assert service.get(dataset, slide)["revision"] == 1


def test_unknown_slide_and_invalid_page_are_rejected(reviews):
    service, dataset, _ = reviews
    with pytest.raises(StorageError) as error:
        service.save(dataset, "not-present", SaveSlideReview(expectedRevision=0, status="accept"))
    assert error.value.code == "REVIEW_SLIDE_NOT_FOUND"
    with pytest.raises(StorageError):
        service.list(dataset, limit=10000)


@pytest.mark.parametrize("field", ["revision", "notes"])
def test_review_audit_corruption_is_not_silently_overwritten(reviews, field):
    service, dataset, slide = reviews
    record = service.save(dataset, slide, SaveSlideReview(expectedRevision=0, notes="original"))
    record[field] = 8 if field == "revision" else "tampered"
    service._path(dataset, slide).write_text(json.dumps(record))
    with pytest.raises(StorageError) as error:
        service.save(dataset, slide, SaveSlideReview(expectedRevision=1, notes="new"))
    assert error.value.code == "SLIDE_REVIEW_CORRUPT"


def test_review_storage_rejects_symlink_and_hardlink(reviews, tmp_path):
    service, dataset, slide = reviews
    service.save(dataset, slide, SaveSlideReview(expectedRevision=0, status="review"))
    path = service._path(dataset, slide)
    copy = tmp_path / "another-review.json"
    path.rename(copy)
    path.symlink_to(copy)
    with pytest.raises(StorageError) as error:
        service.get(dataset, slide)
    assert error.value.code == "STORAGE_UNSAFE_PATH"
    path.unlink()
    path.hardlink_to(copy)
    with pytest.raises(StorageError):
        service.get(dataset, slide)


@pytest.mark.parametrize("values", [
    {"expectedRevision": True}, {"expectedRevision": -1}, {"status": "pass"},
    {"regions": [{"id": "roi", "x": float("nan"), "y": 0, "width": 10, "height": 10}]},
    {"reasons": ["artifact", "artifact"]}, {"notes": "x" * 8001},
])
def test_review_schema_blocks_ambiguous_or_unbounded_edits(values):
    with pytest.raises(ValidationError):
        SaveSlideReview.model_validate({"expectedRevision": 0, **values})
