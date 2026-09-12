"""Attention study review, durable execution, authenticated slide views and exports."""

from fastapi import APIRouter, Path, Query, Response

from histopilot.application.interpretation import InterpretationService
from histopilot.schemas.interpretation import (
    InterpretationGalleryQuery,
    InterpretationSelection,
    SaveInterpretation,
    SlideInspection,
    VisualizeInterpretation,
)
from histopilot.schemas.predictors import PredictorAction
from histopilot.storage.project_lock import StorageError


def interpretation_router(projects, filesystem):
    router = APIRouter(prefix="/api/v1/projects/{identity}/interpretations")

    def service(identity):
        return InterpretationService(projects.scientific_store(identity), filesystem)

    @router.get("")
    def list_interpretations(identity: str, include_inactive: bool = False):
        return service(identity).list(include_inactive=include_inactive)

    @router.get("/sources")
    def sources(identity: str):
        return service(identity).gallery.sources()

    @router.post("/gallery")
    def gallery(identity: str, payload: InterpretationGalleryQuery):
        return service(identity).gallery.query(payload)

    @router.get("/gallery/thumbnail")
    def gallery_thumbnail(
        identity: str,
        path: str = Query(min_length=1, max_length=4096),
        max_size: int = Query(default=256, ge=64, le=512),
    ):
        return Response(
            content=service(identity).gallery_thumbnail(path, max_size=max_size),
            media_type="image/png",
            headers={"Cache-Control": "private, no-store"},
        )

    @router.post("/visualize", status_code=202)
    def visualize(identity: str, payload: VisualizeInterpretation):
        return service(identity).visualize(payload)

    @router.post("/slide-inspection")
    def inspect_slide(identity: str, payload: SlideInspection):
        return service(identity).inspect_slide(payload.path)

    @router.post("/preview")
    def preview(identity: str, payload: InterpretationSelection):
        return service(identity).preview(payload)

    @router.post("", status_code=201)
    def save(identity: str, payload: SaveInterpretation):
        return service(identity).save(payload)

    @router.get("/{interpretation_id}")
    def get(identity: str, interpretation_id: str):
        return service(identity).get(interpretation_id)

    @router.get("/{interpretation_id}/execution")
    def execution(identity: str, interpretation_id: str):
        return service(identity).execution(interpretation_id)

    @router.post("/{interpretation_id}/launch", status_code=202)
    def launch(identity: str, interpretation_id: str, payload: PredictorAction):
        return service(identity).launch(interpretation_id, payload.operationId)

    @router.post("/{interpretation_id}/resume", status_code=202)
    def resume(identity: str, interpretation_id: str, payload: PredictorAction):
        return service(identity).launch(interpretation_id, payload.operationId, resume=True)

    @router.post("/{interpretation_id}/cancel", status_code=202)
    def cancel(identity: str, interpretation_id: str, payload: PredictorAction):
        return service(identity).cancel(interpretation_id, payload.operationId)

    @router.get("/{interpretation_id}/artifacts/{filename}")
    def artifact(identity: str, interpretation_id: str, filename: str):
        return Response(
            content=service(identity).artifact(interpretation_id, filename),
            media_type="application/octet-stream"
            if filename.endswith(".npy")
            else "application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @router.get("/{interpretation_id}/slides/{slide_id}/thumbnail")
    def thumbnail(
        identity: str,
        interpretation_id: str,
        slide_id: str,
        max_size: int = Query(default=1024, ge=64, le=2048),
    ):
        return Response(
            content=service(identity).image(interpretation_id, slide_id, max_size=max_size),
            media_type="image/png",
            headers={"Cache-Control": "private, no-store"},
        )

    @router.get("/{interpretation_id}/slides/{slide_id}/region")
    def region(
        identity: str,
        interpretation_id: str,
        slide_id: str,
        x: int = Query(ge=0),
        y: int = Query(ge=0),
        width: int = Query(gt=0),
        height: int = Query(gt=0),
        max_size: int = Query(default=1024, ge=64, le=2048),
    ):
        return Response(
            content=service(identity).image(
                interpretation_id, slide_id, max_size=max_size, region=(x, y, width, height)
            ),
            media_type="image/png",
            headers={"Cache-Control": "private, no-store"},
        )

    @router.get("/{interpretation_id}/slides/{slide_id}/attention")
    def attention(
        identity: str,
        interpretation_id: str,
        slide_id: str,
        member: str = "mean",
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=10000, ge=1, le=50000),
        x: int | None = Query(default=None, ge=0),
        y: int | None = Query(default=None, ge=0),
        width: int | None = Query(default=None, gt=0),
        height: int | None = Query(default=None, gt=0),
    ):
        bounds = (x, y, width, height)
        if any(value is not None for value in bounds) and not all(
            value is not None for value in bounds
        ):
            raise StorageError(
                "Provide all four viewport bounds.", "INTERPRETATION_VIEW_INVALID", 422
            )
        return service(identity).attention(
            interpretation_id,
            slide_id,
            member=member,
            offset=offset,
            limit=limit,
            region=bounds if x is not None else None,
        )

    @router.get("/{interpretation_id}/slides/{slide_id}/attention/top")
    def top_attention(
        identity: str,
        interpretation_id: str,
        slide_id: str,
        member: str = "mean",
        limit: int = Query(default=10, ge=1, le=20),
    ):
        return service(identity).top_attention(
            interpretation_id, slide_id, member=member, limit=limit
        )

    @router.get("/{interpretation_id}/slides/{slide_id}/patches/{patch_index}/image")
    def patch_image(
        identity: str,
        interpretation_id: str,
        slide_id: str,
        patch_index: int = Path(ge=0),
        member: str = "mean",
        max_size: int = Query(default=512, ge=64, le=1024),
    ):
        return Response(
            content=service(identity).patch_image(
                interpretation_id, slide_id, patch_index, member=member, max_size=max_size
            ),
            media_type="image/png",
            headers={"Cache-Control": "private, no-store"},
        )

    return router
