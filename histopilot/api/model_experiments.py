"""Persistent model-development records, including retained historical records."""

from typing import Literal

from fastapi import APIRouter

from histopilot.application.experiment_results import experiment_headlines, experiment_results
from histopilot.application.model_experiments import ModelExperimentService
from histopilot.schemas.model_experiments import (
    ConfigureModelExperimentSetup,
    CreateModelExperiment,
    FreezeModelExperimentSetup,
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

    @router.get("/headlines")
    def headlines(identity: str):
        """One seed-mean OOF headline per batch, for the experiment list. Read-only."""
        return experiment_headlines(projects.scientific_store(identity), filesystem)

    @router.get("/{experiment_id}")
    def detail(identity: str, experiment_id: str):
        return service(identity).get(experiment_id)

    @router.get("/{experiment_id}/results")
    def results(identity: str, experiment_id: str):
        """Per-fold, per-seed and seed-averaged results of every batch, read-only."""
        return experiment_results(projects.scientific_store(identity), filesystem, experiment_id)

    @router.patch("/{experiment_id}")
    def update(identity: str, experiment_id: str, payload: UpdateModelExperiment):
        return service(identity).update(experiment_id, payload)

    @router.post("/{experiment_id}/setup-inputs")
    def setup_inputs(identity: str, experiment_id: str, payload: ConfigureModelExperimentSetup):
        return service(identity).setup_inputs(experiment_id, payload)

    @router.post("/{experiment_id}/freeze-setup", status_code=201)
    def freeze_setup(identity: str, experiment_id: str, payload: FreezeModelExperimentSetup):
        return service(identity).freeze_setup(experiment_id, payload)

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
