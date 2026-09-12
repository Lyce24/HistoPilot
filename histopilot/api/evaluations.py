"""Test-cohort setup and review; no inference execution is exposed here."""

from fastapi import APIRouter

from histopilot.application.evaluations import EvaluationService
from histopilot.application.project_workspace import ProjectWorkspace
from histopilot.schemas.evaluations import EvaluationFreezeRequest, EvaluationPreviewRequest
from histopilot.storage.filesystem import LocalFilesystem


def evaluation_router(projects: ProjectWorkspace, filesystem: LocalFilesystem) -> APIRouter:
    router = APIRouter(prefix="/api/v1/projects/{identity}")

    def service(identity):
        return EvaluationService(projects.scientific_store(identity), filesystem)

    @router.get("/evaluation-cohorts")
    def list_cohorts(identity: str):
        return service(identity).list()

    @router.get("/evaluation-cohorts/{configuration_id}")
    def get_cohort(identity: str, configuration_id: str):
        return service(identity).get(configuration_id)

    @router.post("/drafts/{draft_id}/evaluation-preview")
    def preview(identity: str, draft_id: str, payload: EvaluationPreviewRequest):
        return service(identity).preview(draft_id, payload.expectedRevision)

    @router.post("/drafts/{draft_id}/evaluation-freeze", status_code=201)
    def freeze(identity: str, draft_id: str, payload: EvaluationFreezeRequest):
        return service(identity).freeze(
            draft_id,
            payload.expectedRevision,
            payload.previewHash,
            payload.operationId,
            version_label=payload.versionLabel.model_dump(),
        )

    return router
