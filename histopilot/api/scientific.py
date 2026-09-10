"""Project-scoped import and protocol intent routes; all publication is server-derived."""

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from histopilot.application.exploration import ExploreRequest, explore
from histopilot.application.features import FeatureService
from histopilot.application.imports import ImportService
from histopilot.application.project_workspace import ProjectWorkspace
from histopilot.application.protocols import ProtocolService
from histopilot.schemas.features import FeatureSpec, FreezeFeatureRequest
from histopilot.schemas.imports import FreezeRequest, InspectRequest, PreviewRequest
from histopilot.schemas.protocols import (
    ProtocolExploreRequest,
    ProtocolFreezeRequest,
    ProtocolPreviewRequest,
)
from histopilot.schemas.workspace import RequestModel
from histopilot.storage.filesystem import LocalFilesystem
from histopilot.storage.project_lock import StorageError


class JobIntent(RequestModel):
    protocolId: str


def scientific_router(projects: ProjectWorkspace, filesystem: LocalFilesystem) -> APIRouter:
    router = APIRouter(prefix="/api/v1/projects/{identity}")

    def importer(identity):
        return ImportService(projects.scientific_store(identity), filesystem)

    @router.post("/imports/inspect")
    def inspect(identity: str, payload: InspectRequest):
        return importer(identity).inspect(payload.source)

    @router.post("/imports/{draft_id}/preview")
    def preview_import(identity: str, draft_id: str, payload: PreviewRequest):
        return importer(identity).preview(draft_id, payload.expectedRevision)

    @router.post("/imports/{draft_id}/freeze", status_code=201)
    def freeze_import(identity: str, draft_id: str, payload: FreezeRequest):
        return importer(identity).freeze(
            draft_id, payload.expectedRevision, payload.previewHash, payload.operationId
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
        return ProtocolService(projects.scientific_store(identity)).explore(payload)

    @router.post("/protocols/{draft_id}/preview")
    def preview_protocol(identity: str, draft_id: str, payload: ProtocolPreviewRequest):
        return ProtocolService(projects.scientific_store(identity)).preview(
            draft_id, payload.expectedRevision
        )

    @router.post("/protocols/{draft_id}/freeze", status_code=201)
    def freeze_protocol(identity: str, draft_id: str, payload: ProtocolFreezeRequest):
        return ProtocolService(projects.scientific_store(identity)).freeze(
            draft_id, payload.expectedRevision, payload.previewHash, payload.operationId
        )

    @router.get("/configurations")
    def configurations(identity: str, kind: str | None = None):
        return {"configurations": projects.scientific_store(identity).list_configurations(kind)}

    @router.get("/configurations/{configuration_id}")
    def configuration(identity: str, configuration_id: str):
        return projects.scientific_store(identity).get_configuration(configuration_id)

    @router.post("/features/preview")
    def preview_feature(identity: str, payload: FeatureSpec):
        return FeatureService(projects.scientific_store(identity), filesystem).preview(payload)

    @router.post("/features/freeze", status_code=201)
    def freeze_feature(identity: str, payload: FreezeFeatureRequest):
        spec = FeatureSpec.model_validate(
            payload.model_dump(exclude={"previewHash", "operationId"})
        )
        return FeatureService(projects.scientific_store(identity), filesystem).freeze(
            spec, payload.previewHash, payload.operationId
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
                FeatureService(store, filesystem).verify_binding(
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
        findings.append(
            {
                "severity": "warning",
                "code": "FULL_FEATURE_VALIDATION_PENDING",
                "message": "Full tensor integrity and encoder provenance checks are pending.",
            }
        )
        return {
            "protocolId": configuration_id,
            "scope": "protocol-and-feature-headers",
            "protocolReady": True,
            "headerInputsReady": not any(row["severity"] == "error" for row in findings),
            "fullFeatureValidationComplete": False,
            "scientificReady": not any(row["severity"] == "error" for row in findings),
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
