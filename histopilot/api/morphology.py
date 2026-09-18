"""Project-scoped morphology exploration and exact frozen-dataset images."""

from fastapi import APIRouter, Query, Response

from histopilot.application.morphology import MorphologyService
from histopilot.schemas.evaluations import ConfigurationId
from histopilot.schemas.morphology import (
    DatasetId,
    MorphologyIndexRequest,
    MorphologyNeighborsRequest,
)
from histopilot.storage.project_lock import StorageError


def morphology_router(projects, filesystem):
    router = APIRouter(prefix="/api/v1/projects/{identity}/morphology")

    def service(identity):
        return MorphologyService(projects.scientific_store(identity), filesystem)

    def image_response(content):
        return Response(
            content, media_type="image/png", headers={"Cache-Control": "private, no-store"}
        )

    @router.get("/slides")
    def slides(
        identity: str,
        datasetId: DatasetId,
        search: str = Query(default="", max_length=200),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=200),
    ):
        return service(identity).slides(datasetId, search=search, offset=offset, limit=limit)

    @router.post("/index")
    def index(identity: str, payload: MorphologyIndexRequest):
        return service(identity).build(payload)

    @router.post("/neighbors")
    def neighbors(identity: str, payload: MorphologyNeighborsRequest):
        return service(identity).neighbors(payload)

    @router.get("/quality")
    def quality(
        identity: str,
        datasetId: DatasetId,
        slideId: str,
        featureBundleId: ConfigurationId | None = None,
    ):
        return service(identity).quality(datasetId, slideId, featureBundleId)

    @router.get("/image")
    def image(
        identity: str,
        datasetId: DatasetId,
        slideId: str,
        max_size: int = Query(default=1024, ge=64, le=2048),
        x: float | None = Query(default=None, ge=0, allow_inf_nan=False),
        y: float | None = Query(default=None, ge=0, allow_inf_nan=False),
        width: float | None = Query(default=None, gt=0, allow_inf_nan=False),
        height: float | None = Query(default=None, gt=0, allow_inf_nan=False),
    ):
        values = (x, y, width, height)
        if any(value is not None for value in values) and any(value is None for value in values):
            raise StorageError(
                "Provide every viewport coordinate.", "MORPHOLOGY_REGION_INVALID", 422
            )
        region = values if x is not None else None
        return image_response(
            service(identity).image(datasetId, slideId, max_size=max_size, region=region)
        )

    @router.get("/patch-region")
    def patch_region(
        identity: str,
        datasetId: DatasetId,
        slideId: str,
        featureBundleId: ConfigurationId,
        patchIndex: int = Query(ge=0),
    ):
        return service(identity).patch_region(datasetId, slideId, featureBundleId, patchIndex)

    @router.get("/patch")
    def patch(
        identity: str,
        datasetId: DatasetId,
        slideId: str,
        featureBundleId: ConfigurationId,
        patchIndex: int = Query(ge=0),
    ):
        return image_response(
            service(identity).patch_image(datasetId, slideId, featureBundleId, patchIndex)
        )

    return router
