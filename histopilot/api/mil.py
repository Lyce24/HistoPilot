"""MIL input planning and explicit execution of frozen development batches."""

from typing import Literal

from fastapi import APIRouter, Query, Response

from histopilot.adapters.native.runtime import training_runtime
from histopilot.application.clinical_inputs import clinical_field_choices
from histopilot.application.development import DevelopmentService
from histopilot.application.mil_inputs import MILInputService
from histopilot.application.project_workspace import ProjectWorkspace
from histopilot.application.training import TrainingService
from histopilot.application.training_exports import training_oof_csv
from histopilot.application.training_history import training_history
from histopilot.application.training_resources import training_resources
from histopilot.schemas.development import (
    DevelopmentBatchSpec,
    FreezeDevelopmentBatch,
    TrainingAction,
)
from histopilot.schemas.mil import MILInputSpec
from histopilot.storage.filesystem import LocalFilesystem
from histopilot.storage.project_lock import StorageError


def mil_router(projects: ProjectWorkspace, filesystem: LocalFilesystem) -> APIRouter:
    router = APIRouter(prefix="/api/v1/projects/{identity}/mil-experiments")

    @router.post("/preview")
    def preview(identity: str, payload: MILInputSpec):
        return MILInputService(projects.scientific_store(identity), filesystem).preview(payload)

    @router.get("/clinical-fields")
    def clinical_fields(identity: str, protocolId: str = Query(min_length=1, max_length=128)):  # noqa: N803
        store = projects.scientific_store(identity)
        protocol = store.get_configuration(protocolId)["manifest"]
        if protocol.get("kind") != "protocol":
            raise StorageError("Choose the experiment's frozen split.", "INVALID_PROTOCOL", 422)
        return {"fields": clinical_field_choices(store, filesystem, protocol)}

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

    @router.get("/batches/{batch_id}/oof/{candidate_id}/{training_seed}/{split_seed}/{unit}.csv")
    def oof_predictions(
        identity: str,
        batch_id: str,
        candidate_id: str,
        training_seed: int,
        split_seed: int,
        unit: Literal["slide", "patient"],
    ):
        content = training_oof_csv(
            projects.scientific_store(identity),
            batch_id,
            candidate_id,
            training_seed,
            split_seed,
            unit,
        )
        filename = f"oof-{unit}-training-{training_seed}-split-{split_seed}.csv"
        return Response(
            content=content,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @router.get("/batches/{batch_id}/runs/{run_id}/history")
    def run_history(identity: str, batch_id: str, run_id: str):
        return training_history(projects.scientific_store(identity), batch_id, run_id)

    @router.get("/batches/{batch_id}/resources/history")
    def resource_history(identity: str, batch_id: str):
        return training_resources(projects.scientific_store(identity), batch_id)

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
