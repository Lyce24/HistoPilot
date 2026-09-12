"""Clinical utility reports from completed, checksum-verified evaluations."""

from fastapi import APIRouter, Response

from histopilot.application.clinical import ClinicalService
from histopilot.schemas.clinical import ClinicalSelection, SaveClinicalAnalysis


def clinical_router(projects, filesystem):
    router = APIRouter(prefix="/api/v1/projects/{identity}/clinical-analyses")

    def service(identity):
        return ClinicalService(projects.scientific_store(identity), filesystem)

    @router.get("")
    def list_analyses(identity: str, include_inactive: bool = False):
        return service(identity).list(include_inactive=include_inactive)

    @router.post("/preview")
    def preview(identity: str, payload: ClinicalSelection):
        return service(identity).preview(payload)

    @router.post("", status_code=201)
    def save(identity: str, payload: SaveClinicalAnalysis):
        return service(identity).save(payload)

    @router.get("/{analysis_id}")
    def get_analysis(identity: str, analysis_id: str):
        return service(identity).get(analysis_id)

    @router.get("/{analysis_id}/artifacts/{filename}")
    def artifact(identity: str, analysis_id: str, filename: str):
        content, media_type = service(identity).artifact(analysis_id, filename)
        return Response(
            content=content,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    return router
