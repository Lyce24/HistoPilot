"""Persistent model-development records, including retained historical records."""

from typing import Literal

from fastapi import APIRouter

from histopilot.application.model_experiments import ModelExperimentService
from histopilot.schemas.model_experiments import CreateModelExperiment, UpdateModelExperiment


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

    return router
