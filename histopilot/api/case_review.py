from fastapi import APIRouter, Response

from histopilot.application.case_review import CaseReviewService
from histopilot.schemas.case_review import CaseReviewQuery


def case_review_router(projects, filesystem):
    router = APIRouter(prefix="/api/v1/projects/{identity}/evaluation-runs/{evaluation_id}/cases")

    @router.post("/query")
    def query(identity: str, evaluation_id: str, payload: CaseReviewQuery):
        return CaseReviewService(projects.scientific_store(identity), filesystem).query(evaluation_id, payload)

    @router.post("/export")
    def export(identity: str, evaluation_id: str, payload: CaseReviewQuery):
        content = CaseReviewService(projects.scientific_store(identity), filesystem).export(evaluation_id, payload)
        return Response(content, media_type="text/csv", headers={"Content-Disposition": 'attachment; filename="case-reviews.csv"', "Cache-Control": "no-store"})

    return router
