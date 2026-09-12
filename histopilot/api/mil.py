"""MIL input planning and explicit execution of frozen ABMIL development batches."""

from fastapi import APIRouter

from histopilot.adapters.native.runtime import training_runtime
from histopilot.application.development import DevelopmentService
from histopilot.application.mil_inputs import MILInputService
from histopilot.application.project_workspace import ProjectWorkspace
from histopilot.application.training import TrainingService
from histopilot.application.training_history import training_history
from histopilot.schemas.development import (
    DevelopmentBatchSpec,
    FreezeDevelopmentBatch,
    TrainingAction,
)
from histopilot.schemas.mil import MILInputSpec
from histopilot.storage.filesystem import LocalFilesystem


def mil_router(projects: ProjectWorkspace, filesystem: LocalFilesystem) -> APIRouter:
    router = APIRouter(prefix="/api/v1/projects/{identity}/mil-experiments")

    @router.post("/preview")
    def preview(identity: str, payload: MILInputSpec):
        return MILInputService(projects.scientific_store(identity), filesystem).preview(payload)

    @router.get("/batches")
    def batches(identity: str):
        store = projects.scientific_store(identity)
        return {
            **DevelopmentService(store, filesystem).list(),
            "executionImplemented": True,
            "executions": TrainingService(store, filesystem).list()["items"],
        }

    @router.get("/runtime")
    def runtime(identity: str):
        projects.scientific_store(identity)
        return training_runtime()

    @router.get("/batches/{batch_id}/execution")
    def execution(identity: str, batch_id: str):
        return TrainingService(projects.scientific_store(identity), filesystem).execution(batch_id)

    @router.get("/batches/{batch_id}/results")
    def results(identity: str, batch_id: str):
        return TrainingService(projects.scientific_store(identity), filesystem).results(batch_id)

    @router.get("/batches/{batch_id}/runs/{run_id}/history")
    def run_history(identity: str, batch_id: str, run_id: str):
        return training_history(projects.scientific_store(identity), batch_id, run_id)

    @router.post("/batches/{batch_id}/launch", status_code=202)
    def launch(identity: str, batch_id: str, payload: TrainingAction):
        return TrainingService(projects.scientific_store(identity), filesystem).launch(
            batch_id, payload.operationId
        )

    @router.post("/batches/{batch_id}/resume", status_code=202)
    def resume(identity: str, batch_id: str, payload: TrainingAction):
        return TrainingService(projects.scientific_store(identity), filesystem).launch(
            batch_id, payload.operationId, resume=True
        )

    @router.post("/batches/{batch_id}/cancel", status_code=202)
    def cancel(identity: str, batch_id: str, payload: TrainingAction):
        return TrainingService(projects.scientific_store(identity), filesystem).cancel(
            batch_id, payload.operationId
        )

    @router.post("/batches/preview")
    def preview_batch(identity: str, payload: DevelopmentBatchSpec):
        return DevelopmentService(projects.scientific_store(identity), filesystem).preview(payload)

    @router.post("/batches/freeze", status_code=201)
    def freeze_batch(identity: str, payload: FreezeDevelopmentBatch):
        return DevelopmentService(projects.scientific_store(identity), filesystem).freeze(
            payload.spec,
            payload.previewHash,
            payload.operationId,
            payload.versionLabel.model_dump(),
        )

    return router
