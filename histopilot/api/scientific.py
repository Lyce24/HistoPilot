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
from histopilot.application.protocols import ProtocolService, pack_binding_snapshot
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
from histopilot.schemas.version_labels import SetVersionLabelRequest
from histopilot.schemas.workspace import RequestModel
from histopilot.storage.filesystem import LocalFilesystem
from histopilot.storage.project_lock import StorageError


class JobIntent(RequestModel):
    protocolId: str


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

    @router.post("/protocols/{draft_id}/preview")
    def preview_protocol(identity: str, draft_id: str, payload: ProtocolPreviewRequest):
        return protocols(identity).preview(draft_id, payload.expectedRevision)

    @router.post("/protocols/{draft_id}/freeze", status_code=201)
    def freeze_protocol(identity: str, draft_id: str, payload: ProtocolFreezeRequest):
        return protocols(identity).freeze(
            draft_id,
            payload.expectedRevision,
            payload.previewHash,
            payload.operationId,
            version_label=payload.versionLabel.model_dump(),
        )

    @router.get("/configurations")
    def configurations(identity: str, kind: str | None = None):
        return {"configurations": projects.scientific_store(identity).list_configurations(kind)}

    @router.get("/configurations/{configuration_id}")
    def configuration(identity: str, configuration_id: str):
        return projects.scientific_store(identity).get_configuration(configuration_id)

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

    @router.get("/protocols/{configuration_id}/preflight")
    def preflight(identity: str, configuration_id: str):
        store = projects.scientific_store(identity)
        protocol = store.get_configuration(configuration_id)["manifest"]
        if protocol.get("kind") != "protocol":
            raise StorageError("Select a frozen target/split protocol.", "INVALID_PROTOCOL", 422)
        store.get_dataset(protocol["datasetId"])
        feature_id = protocol["spec"].get("featureSetId")
        findings = []
        if feature_id:
            feature = store.get_configuration(feature_id)
            if (
                feature["manifest"].get("kind") != "feature"
                or feature["manifest"].get("datasetId") != protocol["datasetId"]
            ):
                raise StorageError(
                    "The feature binding belongs to a different dataset.",
                    "FEATURE_DATASET_MISMATCH",
                    422,
                )
            eligible = {row["slideId"] for row in protocol["memberships"]}
            selected_files = [
                row for row in feature["manifest"]["files"] if row["slideId"] in eligible
            ]
            if eligible - {row["slideId"] for row in selected_files}:
                findings.append(
                    {
                        "severity": "error",
                        "code": "MISSING_FEATURES",
                        "message": "Some eligible slides have no assigned features.",
                    }
                )
            findings.extend(
                FeatureService(
                    store, LocalFilesystem((store.folder, *filesystem.roots))
                ).verify_binding(
                    {**feature, "manifest": {**feature["manifest"], "files": selected_files}}
                )
            )
        else:
            findings.append(
                {
                    "severity": "error",
                    "code": "FEATURES_UNASSIGNED",
                    "message": "Attach features and select them in a new protocol revision.",
                }
            )
        pack_id = protocol["spec"].get("featurePackId")
        pack_status = None
        if pack_id:
            try:
                if not feature_id:
                    raise StorageError(
                        "The selected pack has no feature binding.", "FEATURES_UNASSIGNED", 422
                    )
                pack_status = packing(identity).resolve_artifact(feature_id, pack_id)
                if not pack_status["current"]:
                    findings.append(
                        {
                            "severity": "error",
                            "code": "FEATURE_PACK_UNAVAILABLE",
                            "message": "The selected pack changed or needs verification in PFM & features.",
                        }
                    )
                    findings.extend(
                        {**item, "severity": "error"} for item in pack_status["findings"]
                    )
                if pack_binding_snapshot(pack_status["artifact"]) != protocol.get("featurePack"):
                    findings.append(
                        {
                            "severity": "error",
                            "code": "FEATURE_PACK_BINDING_CHANGED",
                            "message": "The pack no longer matches the representation frozen in this protocol.",
                        }
                    )
                    pack_status = None
            except StorageError as error:
                findings.append({"severity": "error", "code": error.code, "message": str(error)})
        validation = (
            {**pack_status["artifact"]["validation"], "current": pack_status["current"]}
            if pack_status
            else None
            if pack_id
            else packing(identity).validation_for(feature_id)
            if feature_id
            else None
        )
        tensor_complete = bool(
            validation
            and validation.get("valid")
            and validation.get("current")
            and validation.get("tensorValidationComplete")
        )
        provenance_complete = bool(tensor_complete and validation.get("provenanceComplete"))
        if validation:
            findings.extend(validation.get("findings", []))
        if not provenance_complete:
            findings.append(
                {
                    "severity": "warning",
                    "code": "ENCODER_PROVENANCE_UNVERIFIED"
                    if tensor_complete
                    else "FULL_FEATURE_VALIDATION_PENDING",
                    "message": (
                        "Feature contents were validated. Complete encoder/checkpoint provenance has not been authenticated."
                        if tensor_complete
                        else "Validate feature contents in PFM & features before preparing training."
                    ),
                }
            )
        return {
            "protocolId": configuration_id,
            "featurePackId": pack_id,
            "featureSource": {"type": "pack", **protocol.get("featurePack", {})}
            if pack_id
            else {"type": "native", "featureSetId": feature_id},
            "scope": "protocol-and-feature-contents"
            if tensor_complete
            else "protocol-and-feature-headers",
            "protocolReady": True,
            "headerInputsReady": not any(row["severity"] == "error" for row in findings),
            "tensorValidationComplete": tensor_complete,
            "provenanceComplete": provenance_complete,
            "fullFeatureValidationComplete": tensor_complete and provenance_complete,
            "scientificReady": tensor_complete
            and provenance_complete
            and not any(row["severity"] == "error" for row in findings),
            "executionEnabled": False,
            "executionReady": False,
            "findings": findings,
        }

    @router.post("/jobs")
    def submit(identity: str, payload: JobIntent):
        report = preflight(identity, payload.protocolId)
        if not report["scientificReady"]:
            return JSONResponse(
                {
                    "detail": "Preflight blocked this experiment.",
                    "code": "PREFLIGHT_BLOCKED",
                    **report,
                },
                status_code=422,
            )
        return JSONResponse(
            {
                "detail": "Compute execution is not implemented. No job was submitted.",
                "code": "EXECUTION_UNAVAILABLE",
                **report,
            },
            status_code=501,
        )

    return router
