"""Workspace cleanup always resolves project-owned record IDs, never browser paths."""

from fastapi import APIRouter

from histopilot.application.lifecycle import CleanupService
from histopilot.schemas.lifecycle import ApplyCleanup, CancelCleanupJob, CleanupSelection


def lifecycle_router(projects, filesystem):
    router = APIRouter(prefix="/api/v1/projects/{identity}/cleanup")

    def service(identity):
        document, _ = projects._load(identity)
        return CleanupService(projects.scientific_store(identity), filesystem, document["name"])

    @router.get("")
    def catalog(identity: str):
        return service(identity).catalog()

    @router.post("/preview")
    def preview(identity: str, payload: CleanupSelection):
        return service(identity).preview(payload)

    @router.post("/apply")
    def apply(identity: str, payload: ApplyCleanup):
        return service(identity).apply(payload)

    @router.post("/cancel", status_code=202)
    def cancel(identity: str, payload: CancelCleanupJob):
        return service(identity).cancel(payload)

    return router
