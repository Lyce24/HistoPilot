"""Authenticated, project-owned human review API."""

from fastapi import APIRouter, Query

from histopilot.application.slide_reviews import SlideReviewService
from histopilot.schemas.slide_reviews import SaveSlideReview


def slide_review_router(projects, filesystem):
    router = APIRouter(prefix="/api/v1/projects/{identity}/datasets/{dataset_id}/slide-reviews")

    def service(identity):
        return SlideReviewService(projects.scientific_store(identity))

    @router.get("")
    def list_reviews(identity: str, dataset_id: str, offset: int = Query(0, ge=0), limit: int = Query(200, ge=1, le=500)):
        return service(identity).list(dataset_id, offset=offset, limit=limit)

    @router.get("/{slide_id:path}")
    def get_review(identity: str, dataset_id: str, slide_id: str):
        return service(identity).get(dataset_id, slide_id)

    @router.put("/{slide_id:path}")
    def save_review(identity: str, dataset_id: str, slide_id: str, payload: SaveSlideReview):
        return service(identity).save(dataset_id, slide_id, payload)

    return router
