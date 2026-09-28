"""Label-free analysis of inference runs; attention needs no observed outcomes."""

from fastapi import APIRouter, Response

from histopilot.application.inference_analysis import InferenceAnalysisService
from histopilot.schemas.inference import (
    InferenceAttentionRequest,
    InferenceExportQuery,
    InferenceSummaryQuery,
)


def inference_router(projects, filesystem):
    router = APIRouter(prefix="/api/v1/projects/{identity}/evaluation-runs/{evaluation_id}")

    def service(identity):
        return InferenceAnalysisService(projects.scientific_store(identity), filesystem)

    @router.post("/inference/summary")
    def summary(identity: str, evaluation_id: str, payload: InferenceSummaryQuery):
        return service(identity).summary(evaluation_id, payload)

    @router.post("/inference/export")
    def export(identity: str, evaluation_id: str, payload: InferenceExportQuery):
        content = service(identity).export(evaluation_id, payload)
        return Response(
            content,
            media_type="text/csv",
            headers={
                "Content-Disposition": 'attachment; filename="inference-predictions.csv"',
                "Cache-Control": "no-store",
            },
        )

    @router.post("/attention", status_code=202)
    def attention(identity: str, evaluation_id: str, payload: InferenceAttentionRequest):
        return service(identity).attention(evaluation_id, payload)

    return router
