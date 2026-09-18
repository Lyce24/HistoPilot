"""Frozen predictor registry and explicit predictor/cohort evaluation chains."""

from fastapi import APIRouter, Response

from histopilot.application.bulk_evaluations import BulkEvaluationService
from histopilot.application.evaluation_runs import EvaluationRunService
from histopilot.application.predictor_builds import PredictorBuildService
from histopilot.application.predictors import PredictorService
from histopilot.application.refits import RefitService
from histopilot.schemas.bulk_evaluations import BulkEvaluationSelection, RunBulkEvaluation
from histopilot.schemas.predictors import (
    ApplyPredictorBuilds,
    CompareEvaluations,
    EvaluationRunSelection,
    FreezePredictor,
    LaunchRefit,
    PredictorAction,
    PredictorBuildSelection,
    PredictorSelection,
    SaveEvaluationRun,
)


def predictor_router(projects, filesystem):
    router = APIRouter(prefix="/api/v1/projects/{identity}")

    def service(identity):
        return PredictorService(projects.scientific_store(identity), filesystem)

    @router.get("/predictors")
    def list_predictors(identity: str, include_inactive: bool = False):
        return service(identity).list(include_inactive=include_inactive)

    @router.get("/predictors/choices")
    def choices(identity: str):
        return service(identity).choices()

    def builds(identity):
        return PredictorBuildService(projects.scientific_store(identity), filesystem)

    @router.post("/predictors/builds/preview")
    def preview_builds(identity: str, payload: PredictorBuildSelection):
        return builds(identity).preview(payload)

    @router.post("/predictors/builds", status_code=201)
    def apply_builds(identity: str, payload: ApplyPredictorBuilds):
        return builds(identity).apply(payload)

    @router.get("/predictors/builds/{operation_id}")
    def build_receipt(identity: str, operation_id: str):
        return builds(identity).get(operation_id)

    def refits(identity):
        return RefitService(projects.scientific_store(identity), filesystem)

    @router.get("/predictors/refits")
    def list_refits(identity: str, include_inactive: bool = False):
        return refits(identity).list(include_inactive=include_inactive)

    @router.post("/predictors/refits", status_code=201)
    def create_refit(identity: str, payload: FreezePredictor):
        return refits(identity).create(payload)

    @router.get("/predictors/refits/{refit_id}/execution")
    def refit_execution(identity: str, refit_id: str):
        return refits(identity).execution(refit_id)

    @router.post("/predictors/refits/{refit_id}/launch", status_code=202)
    def launch_refit(identity: str, refit_id: str, payload: LaunchRefit):
        return refits(identity).launch(refit_id, payload)

    @router.post("/predictors/refits/{refit_id}/resume", status_code=202)
    def resume_refit(identity: str, refit_id: str, payload: LaunchRefit):
        return refits(identity).launch(refit_id, payload, resume=True)

    @router.post("/predictors/refits/{refit_id}/cancel", status_code=202)
    def cancel_refit(identity: str, refit_id: str, payload: PredictorAction):
        return refits(identity).cancel(refit_id, payload.operationId)

    @router.post("/predictors/refits/{refit_id}/publish", status_code=201)
    def publish_refit(identity: str, refit_id: str, payload: PredictorAction):
        return refits(identity).publish(refit_id, payload.operationId)

    @router.get("/predictors/{predictor_id}")
    def get_predictor(identity: str, predictor_id: str):
        return service(identity).get(predictor_id)

    @router.post("/predictors/preview")
    def preview(identity: str, payload: PredictorSelection):
        return service(identity).preview(payload)

    @router.post("/predictors/freeze", status_code=201)
    def freeze(identity: str, payload: FreezePredictor):
        return service(identity).freeze(payload)

    return router


def evaluation_run_router(projects, filesystem):
    router = APIRouter(prefix="/api/v1/projects/{identity}")

    def service(identity):
        return EvaluationRunService(projects.scientific_store(identity), filesystem)

    def bulk(identity):
        return BulkEvaluationService(projects.scientific_store(identity), filesystem)

    @router.get("/evaluation-runs/bulk")
    def list_batches(identity: str, include_inactive: bool = False):
        return bulk(identity).list(include_inactive=include_inactive)

    @router.get("/evaluation-runs/bulk/{batch_id}")
    def get_batch(identity: str, batch_id: str):
        return bulk(identity).get(batch_id)

    @router.post("/evaluation-runs/bulk/preview")
    def preview_batch(identity: str, payload: BulkEvaluationSelection):
        return bulk(identity).preview(payload)

    @router.post("/evaluation-runs/bulk", status_code=202)
    def run_batch(identity: str, payload: RunBulkEvaluation):
        return bulk(identity).run(payload)

    @router.post("/evaluation-runs/bulk/{batch_id}/cancel", status_code=202)
    def cancel_batch(identity: str, batch_id: str, payload: PredictorAction):
        return bulk(identity).cancel(batch_id, payload.operationId)

    @router.get("/evaluation-runs")
    def list_evaluations(identity: str, include_inactive: bool = False):
        return service(identity).list(include_inactive=include_inactive)

    @router.post("/evaluation-runs/compare")
    def compare(identity: str, payload: CompareEvaluations):
        return service(identity).compare(payload)

    @router.get("/evaluation-runs/{evaluation_id}")
    def get_evaluation(identity: str, evaluation_id: str):
        return service(identity).get(evaluation_id)

    @router.post("/evaluation-runs/preview")
    def preview(identity: str, payload: EvaluationRunSelection):
        return service(identity).preview(payload)

    @router.post("/evaluation-runs", status_code=201)
    def save(identity: str, payload: SaveEvaluationRun):
        return service(identity).save(payload)

    @router.get("/evaluation-runs/{evaluation_id}/execution")
    def execution(identity: str, evaluation_id: str):
        return service(identity).execution(evaluation_id)

    @router.post("/evaluation-runs/{evaluation_id}/launch", status_code=202)
    def launch(identity: str, evaluation_id: str, payload: PredictorAction):
        return service(identity).launch(evaluation_id, payload.operationId)

    @router.post("/evaluation-runs/{evaluation_id}/resume", status_code=202)
    def resume(identity: str, evaluation_id: str, payload: PredictorAction):
        return service(identity).launch(evaluation_id, payload.operationId, resume=True)

    @router.post("/evaluation-runs/{evaluation_id}/cancel", status_code=202)
    def cancel(identity: str, evaluation_id: str, payload: PredictorAction):
        return service(identity).cancel(evaluation_id, payload.operationId)

    @router.get("/evaluation-runs/{evaluation_id}/artifacts/{filename}")
    def artifact(identity: str, evaluation_id: str, filename: str):
        content = service(identity).artifact(evaluation_id, filename)
        return Response(
            content=content,
            media_type="text/csv" if filename.endswith(".csv") else "application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    return router
