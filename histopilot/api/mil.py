"""MIL input planning; execution remains a separate, unavailable capability."""

from fastapi import APIRouter

from histopilot.application.mil_inputs import MILInputService
from histopilot.application.project_workspace import ProjectWorkspace
from histopilot.schemas.mil import MILInputSpec
from histopilot.storage.filesystem import LocalFilesystem


def mil_router(projects: ProjectWorkspace, filesystem: LocalFilesystem) -> APIRouter:
    router = APIRouter(prefix="/api/v1/projects/{identity}/mil-experiments")

    @router.post("/preview")
    def preview(identity: str, payload: MILInputSpec):
        return MILInputService(projects.scientific_store(identity), filesystem).preview(payload)

    return router
