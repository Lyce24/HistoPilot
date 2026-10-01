"""Subgroup performance of labeled runs, from their scored predictions."""

from fastapi import APIRouter

from histopilot.application.run_performance import RunPerformanceService
from histopilot.schemas.performance import PerformanceBreakdownQuery


def performance_router(projects, filesystem):
    router = APIRouter(prefix="/api/v1/projects/{identity}/evaluation-runs/{evaluation_id}")

    @router.post("/performance/breakdown")
    def breakdown(identity: str, evaluation_id: str, payload: PerformanceBreakdownQuery):
        service = RunPerformanceService(projects.scientific_store(identity), filesystem)
        return service.breakdown(evaluation_id, payload)

    return router
