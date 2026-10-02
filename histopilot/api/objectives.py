"""A project's objectives: the scientific questions its experiments are filed under."""

from fastapi import APIRouter

from histopilot.application.objectives import ObjectiveService
from histopilot.schemas.objectives import CreateObjective, RemoveObjective, UpdateObjective


def objectives_router(projects, filesystem):
    router = APIRouter(prefix="/api/v1/projects/{identity}/objectives")

    def service(identity):
        return ObjectiveService(projects.scientific_store(identity))

    @router.get("")
    def objectives(identity: str):
        """Every objective, primary first, then secondary, then exploratory."""
        return service(identity).list()

    @router.post("", status_code=201)
    def create(identity: str, payload: CreateObjective):
        return service(identity).create(payload)

    @router.get("/{objective_id}")
    def detail(identity: str, objective_id: str):
        return service(identity).get(objective_id)

    @router.patch("/{objective_id}")
    def update(identity: str, objective_id: str, payload: UpdateObjective):
        """Rename, re-rank or describe an objective under `expectedRevision`."""
        return service(identity).update(objective_id, payload)

    @router.post("/{objective_id}/remove")
    def remove(identity: str, objective_id: str, payload: RemoveObjective):
        """Remove an objective; its experiments stay, unfiled from it."""
        return service(identity).remove(objective_id, payload)

    return router
