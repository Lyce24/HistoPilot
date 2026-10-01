"""Reference standards of cohorts, and runs scored, compared and recalibrated against labels."""

from fastapi import APIRouter, Response

from histopilot.application.evaluation_runs import EvaluationRunService
from histopilot.application.references import ReferenceService
from histopilot.schemas.references import (
    AgreementQuery,
    RecalibrationQuery,
    ReferenceScoreQuery,
    ReferenceSelection,
    SaveReference,
)
from histopilot.storage.project_lock import StorageError


def references_router(projects, filesystem):
    router = APIRouter(prefix="/api/v1/projects/{identity}")

    def references(identity):
        return ReferenceService(projects.scientific_store(identity), filesystem)

    def runs(identity):
        return EvaluationRunService(projects.scientific_store(identity), filesystem)

    @router.get("/reference-standards")
    def list_references(
        identity: str, cohort_id: str | None = None, include_inactive: bool = False
    ):
        return references(identity).list(cohort_id=cohort_id, include_inactive=include_inactive)

    @router.post("/reference-standards/preview")
    def preview(identity: str, payload: ReferenceSelection):
        return references(identity).preview(payload)

    @router.post("/reference-standards", status_code=201)
    def save(identity: str, payload: SaveReference):
        return references(identity).save(payload)

    @router.get("/reference-standards/{reference_id}")
    def get(identity: str, reference_id: str):
        return references(identity).get(reference_id)

    @router.post("/evaluation-runs/{evaluation_id}/scores")
    def scores(identity: str, evaluation_id: str, payload: ReferenceScoreQuery):
        service = runs(identity)
        if payload.referenceId:
            return service.reference_metrics(evaluation_id, payload.referenceId)
        result = (service.get(evaluation_id)["execution"] or {}).get("result") or {}
        if "metrics" not in result:
            error = result.get("metricsError") or {}
            raise StorageError(
                error.get("message") or "This run has no scores against its cohort's labels.",
                error.get("code") or "RUN_NOT_SCORED",
                409,
            )
        return result["metrics"]

    @router.get("/evaluation-runs/{evaluation_id}/reference-standards/{reference_id}/{filename}")
    def reference_artifact(identity: str, evaluation_id: str, reference_id: str, filename: str):
        content = runs(identity).reference_download(evaluation_id, reference_id, filename)
        return Response(
            content=content,
            media_type="text/csv" if filename.endswith(".csv") else "application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @router.post("/evaluation-runs/{evaluation_id}/recalibration")
    def recalibration(identity: str, evaluation_id: str, payload: RecalibrationQuery):
        return runs(identity).recalibration(evaluation_id, payload.unit, payload.referenceId)

    @router.post("/evaluation-runs/{evaluation_id}/agreement")
    def agreement(identity: str, evaluation_id: str, payload: AgreementQuery):
        return runs(identity).agreement(evaluation_id, payload.unit)

    return router
