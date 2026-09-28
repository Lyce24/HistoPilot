"""Project-scoped import and protocol intent routes; all publication is server-derived."""

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from histopilot.application.exploration import ExploreRequest, explore
from histopilot.application.extractions import ExtractionService
from histopilot.application.feature_bundles import FeatureBundleService
from histopilot.application.feature_packs import FeaturePackService
from histopilot.application.features import FeatureService
from histopilot.application.imports import ImportService
from histopilot.application.project_workspace import ProjectWorkspace
from histopilot.application.protocols import ProtocolService
from histopilot.application.target_splits import TargetSplitService
from histopilot.schemas.extractions import ExtractionSpec, SubmitExtractionRequest
from histopilot.schemas.feature_bundles import FeatureBundleSpec, FreezeFeatureBundleRequest
from histopilot.schemas.feature_packs import (
    FeaturePackSpec,
    SelectFeaturePackRequest,
    SubmitFeaturePackRequest,
)
from histopilot.schemas.features import FeatureSpec, FreezeFeatureRequest
from histopilot.schemas.imports import FreezeRequest, InspectRequest, PreviewRequest
from histopilot.schemas.protocols import (
    ProtocolExploreRequest,
    ProtocolFreezeRequest,
    ProtocolPreviewRequest,
)
from histopilot.schemas.target_splits import TargetSplitPartitionPreviewRequest
from histopilot.schemas.version_labels import SetVersionLabelRequest
from histopilot.storage.filesystem import LocalFilesystem


def scientific_router(projects: ProjectWorkspace, filesystem: LocalFilesystem) -> APIRouter:
    router = APIRouter(prefix="/api/v1/projects/{identity}")

    def importer(identity):
        return ImportService(projects.scientific_store(identity), filesystem)

    def extraction(identity):
        return ExtractionService(projects.scientific_store(identity), filesystem)

    def feature_service(identity):
        store = projects.scientific_store(identity)
        return FeatureService(store, LocalFilesystem((store.folder, *filesystem.roots)))

    def packing(identity):
        return FeaturePackService(projects.scientific_store(identity), filesystem)

    def bundles(identity):
        return FeatureBundleService(projects.scientific_store(identity), filesystem)

    def protocols(identity):
        return ProtocolService(projects.scientific_store(identity), filesystem)

    @router.get("/feature-bundles")
    def list_feature_bundles(identity: str):
        return bundles(identity).list()

    @router.post("/feature-bundles/preview")
    def preview_feature_bundle(identity: str, payload: FeatureBundleSpec):
        return bundles(identity).preview(payload)

    @router.post("/feature-bundles/freeze", status_code=201)
    def freeze_feature_bundle(identity: str, payload: FreezeFeatureBundleRequest):
        spec = FeatureBundleSpec.model_validate(
            payload.model_dump(exclude={"previewHash", "operationId", "versionLabel"})
        )
        return bundles(identity).freeze(
            spec,
            payload.previewHash,
            payload.operationId,
            version_label=payload.versionLabel.model_dump(),
        )

    @router.get("/feature-bundles/{bundle_id}")
    def feature_bundle(identity: str, bundle_id: str):
        return bundles(identity).get(bundle_id)

    @router.get("/feature-packs")
    def list_feature_packs(identity: str):
        return packing(identity).list()

    @router.post("/feature-packs/preview")
    def preview_feature_pack(identity: str, payload: FeaturePackSpec):
        return packing(identity).preview(payload)

    @router.post("/feature-packs", status_code=201)
    def submit_feature_pack(identity: str, payload: SubmitFeaturePackRequest):
        spec = FeaturePackSpec.model_validate(
            payload.model_dump(exclude={"previewHash", "operationId"})
        )
        return packing(identity).submit(spec, payload.previewHash, payload.operationId)

    @router.get("/feature-packs/artifacts/{artifact_id}")
    def feature_pack_artifact(identity: str, artifact_id: str):
        return packing(identity).artifact(artifact_id)

    @router.get("/feature-packs/{job_id}")
    def feature_pack_job(identity: str, job_id: str):
        return packing(identity).get(job_id, logs=True)

    @router.post("/feature-packs/{job_id}/cancel")
    def cancel_feature_pack(identity: str, job_id: str):
        return packing(identity).cancel(job_id)

    @router.get("/features/{feature_id}/validation")
    def feature_validation(identity: str, feature_id: str):
        return packing(identity).validation_for(feature_id)

    @router.get("/features/{feature_id}/pack-selection")
    def feature_pack_selection(identity: str, feature_id: str):
        return packing(identity).selection_for(feature_id)

    @router.put("/features/{feature_id}/pack-selection")
    def select_feature_pack(identity: str, feature_id: str, payload: SelectFeaturePackRequest):
        return packing(identity).select(feature_id, payload.artifactId)

    @router.put("/datasets/{dataset_id}/label")
    def label_dataset(identity: str, dataset_id: str, payload: SetVersionLabelRequest):
        return projects.scientific_store(identity).set_version_label(
            "dataset",
            dataset_id,
            tag=payload.tag,
            note=payload.note,
            expected_revision=payload.expectedRevision,
        )

    @router.put("/configurations/{configuration_id}/label")
    def label_configuration(identity: str, configuration_id: str, payload: SetVersionLabelRequest):
        return projects.scientific_store(identity).set_version_label(
            "configuration",
            configuration_id,
            tag=payload.tag,
            note=payload.note,
            expected_revision=payload.expectedRevision,
        )

    @router.get("/extractions/catalog")
    def extraction_catalog(identity: str):
        return extraction(identity).catalog()

    @router.post("/extractions/preview")
    def preview_extraction(identity: str, payload: ExtractionSpec):
        return extraction(identity).preview(payload)

    @router.post("/extractions", status_code=201)
    def submit_extraction(identity: str, payload: SubmitExtractionRequest):
        spec = ExtractionSpec.model_validate(
            payload.model_dump(exclude={"previewHash", "operationId"})
        )
        return extraction(identity).submit(spec, payload.previewHash, payload.operationId)

    @router.get("/extractions")
    def extractions(identity: str):
        return extraction(identity).list()

    @router.get("/extractions/{job_id}")
    def get_extraction(identity: str, job_id: str):
        return extraction(identity).get(job_id, logs=True)

    @router.post("/extractions/{job_id}/cancel")
    def cancel_extraction(identity: str, job_id: str):
        return extraction(identity).cancel(job_id)

    @router.post("/extractions/{job_id}/resume")
    def resume_extraction(identity: str, job_id: str):
        return extraction(identity).resume(job_id)

    @router.post("/imports/inspect")
    def inspect(identity: str, payload: InspectRequest):
        return importer(identity).inspect(payload.source)

    @router.post("/imports/{draft_id}/preview")
    def preview_import(identity: str, draft_id: str, payload: PreviewRequest):
        return importer(identity).preview(draft_id, payload.expectedRevision)

    @router.post("/imports/{draft_id}/freeze", status_code=201)
    def freeze_import(identity: str, draft_id: str, payload: FreezeRequest):
        return importer(identity).freeze(
            draft_id,
            payload.expectedRevision,
            payload.previewHash,
            payload.operationId,
            version_label=payload.versionLabel.model_dump(),
        )

    @router.get("/datasets/{dataset_id}/records")
    def records(
        identity: str,
        dataset_id: str,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=200, ge=1, le=1000),
    ):
        values = importer(identity).records(dataset_id)
        return {
            "records": values[offset : offset + limit],
            "total": len(values),
            "offset": offset,
            "limit": limit,
        }

    @router.post("/datasets/{dataset_id}/query")
    def query_dataset(identity: str, dataset_id: str, payload: ExploreRequest):
        store = projects.scientific_store(identity)
        dataset = store.get_dataset(dataset_id)
        return explore(
            importer(identity).records(dataset_id),
            dataset["manifest"].get("dictionary", []),
            payload,
        )

    @router.post("/protocols/explore")
    def explore_protocol(identity: str, payload: ProtocolExploreRequest):
        return protocols(identity).explore(payload)

    @router.post("/target-splits/partition-preview")
    def preview_target_partition(identity: str, payload: TargetSplitPartitionPreviewRequest):
        return TargetSplitService(
            projects.scientific_store(identity), filesystem
        ).partition_preview(payload)

    @router.post("/target-splits/{draft_id}/preview")
    def preview_target_split(identity: str, draft_id: str, payload: ProtocolPreviewRequest):
        return TargetSplitService(projects.scientific_store(identity), filesystem).preview(
            draft_id, payload.expectedRevision
        )

    @router.post("/target-splits/{draft_id}/freeze", status_code=201)
    def freeze_target_split(identity: str, draft_id: str, payload: ProtocolFreezeRequest):
        return TargetSplitService(projects.scientific_store(identity), filesystem).freeze(
            draft_id,
            payload.expectedRevision,
            payload.previewHash,
            payload.operationId,
            version_label=payload.versionLabel.model_dump(),
        )

    @router.get("/target-splits/{configuration_id}")
    def target_split(identity: str, configuration_id: str):
        return TargetSplitService(projects.scientific_store(identity), filesystem).get(
            configuration_id
        )

    @router.post("/target-splits/{configuration_id}/test-cohort")
    def target_split_test_cohort(identity: str, configuration_id: str):
        cohort = TargetSplitService(
            projects.scientific_store(identity), filesystem
        ).create_test_cohort(configuration_id)
        return {"evaluationCohortId": cohort["id"] if cohort else None, "cohort": cohort}

    # Stored configurations are verified JSON documents: skip FastAPI's recursive
    # re-encoding, which costs more than the read itself for multi-MB manifests.
    @router.get("/configurations")
    def configurations(identity: str, kind: str | None = None):
        store = projects.scientific_store(identity)
        return JSONResponse({"configurations": store.list_configurations(kind)})

    @router.get("/configurations/{configuration_id}")
    def configuration(identity: str, configuration_id: str):
        return JSONResponse(projects.scientific_store(identity).get_configuration(configuration_id))

    @router.post("/features/preview")
    def preview_feature(identity: str, payload: FeatureSpec):
        return feature_service(identity).preview(payload)

    @router.post("/features/freeze", status_code=201)
    def freeze_feature(identity: str, payload: FreezeFeatureRequest):
        spec = FeatureSpec.model_validate(
            payload.model_dump(exclude={"previewHash", "operationId", "versionLabel"})
        )
        return feature_service(identity).freeze(
            spec,
            payload.previewHash,
            payload.operationId,
            version_label=payload.versionLabel.model_dump(),
        )

    return router
