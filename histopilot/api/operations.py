"""Project operations reuse the authenticated local control-service boundary."""

from fastapi import APIRouter

from histopilot.application.operations import operations_inventory, relink_source, source_inventory
from histopilot.application.portability_jobs import PortabilityJobs
from histopilot.schemas.operations import PortabilityRequest, RelinkSource


def operations_router(projects, filesystem):
    router = APIRouter(prefix="/api/v1/projects/{identity}/operations")
    jobs = PortabilityJobs(projects, filesystem)

    @router.get("")
    def inventory(identity: str):
        return operations_inventory(projects.scientific_store(identity), filesystem)

    @router.get("/sources")
    def sources(identity: str):
        return source_inventory(projects.scientific_store(identity), projects.storage)

    @router.post("/sources/relink")
    def relink(identity: str, payload: RelinkSource):
        return relink_source(projects, identity, payload)

    @router.get("/archives")
    def archives(identity: str):
        return jobs.list(identity)

    @router.post("/archives", status_code=202)
    def submit_archive(identity: str, payload: PortabilityRequest):
        return jobs.submit(identity, payload)

    @router.get("/archives/{job_id}")
    def archive_status(identity: str, job_id: str):
        return jobs.get(identity, job_id)

    @router.post("/archives/{job_id}/cancel", status_code=202)
    def cancel_archive(identity: str, job_id: str):
        return jobs.cancel(identity, job_id)

    @router.post("/archives/{job_id}/retry", status_code=202)
    def retry_archive(identity: str, job_id: str):
        return jobs.retry(identity, job_id)

    return router
