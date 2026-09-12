"""Persistent model-development records, including retained historical records."""

from typing import Literal

from fastapi import APIRouter

from histopilot.application.model_experiments import ModelExperimentService
from histopilot.schemas.model_experiments import (
    CreateModelExperiment,
    SubmitModelExperiment,
    UpdateModelExperiment,
)
from histopilot.schemas.predictors import PredictorAction


def model_experiments_router(projects, filesystem):
    router = APIRouter(prefix="/api/v1/projects/{identity}/model-experiments")

    def service(identity):
        return ModelExperimentService(projects.scientific_store(identity), filesystem)

    @router.get("")
    def experiments(
        identity: str,
        state: Literal["all", "active", "archived", "trashed"] = "all",
        summary: bool = False,
    ):
        return service(identity).list(state=state, summary=summary)

    @router.post("", status_code=201)
    def create(identity: str, payload: CreateModelExperiment):
        return service(identity).create(payload)

    @router.get("/{experiment_id}")
    def detail(identity: str, experiment_id: str):
        return service(identity).get(experiment_id)

    @router.patch("/{experiment_id}")
    def update(identity: str, experiment_id: str, payload: UpdateModelExperiment):
        return service(identity).update(experiment_id, payload)

    @router.post("/{experiment_id}/submit", status_code=202)
    def submit(identity: str, experiment_id: str, payload: SubmitModelExperiment):
        return service(identity).submit(experiment_id, payload)

    @router.post("/{experiment_id}/predictors/resume", status_code=202)
    def resume_predictors(identity: str, experiment_id: str, payload: PredictorAction):
        return (
            service(identity)._predictors().launch(experiment_id, payload.operationId, resume=True)
        )

    @router.post("/{experiment_id}/predictors/cancel", status_code=202)
    def cancel_predictors(identity: str, experiment_id: str, payload: PredictorAction):
        return service(identity)._predictors().cancel(experiment_id, payload.operationId)

    return router
