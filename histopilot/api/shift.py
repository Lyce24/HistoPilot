"""Dataset shift of a run's cohort against its predictor's development slides, and the
model card of a frozen predictor."""

from fastapi import APIRouter, Response

from histopilot.application.model_cards import model_card, render_markdown
from histopilot.application.shift import ShiftService
from histopilot.schemas.shift import ShiftPerformanceQuery, StartShiftAnalysis


def shift_router(projects, filesystem):
    router = APIRouter(prefix="/api/v1/projects/{identity}")

    def service(identity):
        return ShiftService(projects.scientific_store(identity), filesystem)

    @router.get("/evaluation-runs/{evaluation_id}/shift")
    def shift_status(identity: str, evaluation_id: str):
        return service(identity).status(evaluation_id)

    @router.post("/evaluation-runs/{evaluation_id}/shift/preview")
    def shift_preview(identity: str, evaluation_id: str):
        return service(identity).preview(evaluation_id)

    @router.post("/evaluation-runs/{evaluation_id}/shift", status_code=202)
    def shift_start(identity: str, evaluation_id: str, payload: StartShiftAnalysis):
        return service(identity).start(evaluation_id, payload.previewHash, payload.operationId)

    @router.post("/evaluation-runs/{evaluation_id}/shift/performance")
    def shift_performance(identity: str, evaluation_id: str, payload: ShiftPerformanceQuery):
        return service(identity).performance(
            evaluation_id, unit=payload.unit, reference_id=payload.referenceId
        )

    @router.get("/predictors/{predictor_id}/card")
    def predictor_card(identity: str, predictor_id: str, format: str = "json"):
        card = model_card(projects.scientific_store(identity), filesystem, predictor_id)
        if format == "markdown":
            return Response(
                content=render_markdown(card),
                media_type="text/markdown; charset=utf-8",
                headers={"Content-Disposition": f'attachment; filename="{predictor_id}.md"'},
            )
        return card

    return router
