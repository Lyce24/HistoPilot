"""Single-process local metadata service and packaged React static assets."""

from contextlib import asynccontextmanager
from pathlib import Path
from secrets import token_urlsafe
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse

from histopilot import __version__
from histopilot.application.local_workspace import LocalWorkspace, WorkspaceError
from histopilot.application.project_workspace import ProjectWorkspace
from histopilot.config import Settings, load_settings
from histopilot.doctor import system_report
from histopilot.schemas.scientific import CreateDraftRequest, UpdateDraftRequest
from histopilot.schemas.workspace import (
    CohortRequest,
    ExperimentRequest,
    OpenProjectRequest,
    ProjectRequest,
    ProjectSourceRequest,
    ProjectUpdateRequest,
    SourceRequest,
)
from histopilot.storage.database import SCHEMA_VERSION, Database
from histopilot.storage.filesystem import FilesystemError, LocalFilesystem
from histopilot.storage.project_lock import StorageError

from .scientific import scientific_router
from .security import configure_browser_boundary


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    database = Database(settings.workspace)
    filesystem = LocalFilesystem(settings.data_roots)
    workspace = LocalWorkspace(database, filesystem)
    storage = LocalFilesystem((settings.workspace, *settings.data_roots))
    projects = ProjectWorkspace(database, workspace, storage)
    token = token_urlsafe(32)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            database.initialize()
            workspace.initialize()
            yield
        finally:
            database.close()

    app = FastAPI(
        title="HistoPilot local control service",
        version=__version__,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.settings = settings
    app.state.projects = projects
    configure_browser_boundary(app, settings, token)

    @app.exception_handler(WorkspaceError)
    @app.exception_handler(FilesystemError)
    async def invalid_request(_request: Request, error: WorkspaceError | FilesystemError):
        return JSONResponse({"detail": str(error)}, status_code=error.status_code)

    @app.exception_handler(StorageError)
    async def invalid_storage(_request: Request, error: StorageError):
        return JSONResponse(
            {"detail": str(error), "code": error.code}, status_code=error.status_code
        )

    @app.get("/api/v1/health")
    def health():
        return {"status": "ok", "version": __version__, "executionEnabled": False}

    @app.get("/api/v1/session")
    def session():
        return {"token": token}

    @app.get("/api/v1/workspace")
    def get_workspace():
        return workspace.workspace()

    @app.get("/api/v1/projects")
    def list_projects():
        return projects.list_projects()

    @app.post("/api/v1/projects", status_code=201)
    def create_project(payload: ProjectRequest):
        return projects.create(payload)

    @app.post("/api/v1/projects/open")
    def open_project(payload: OpenProjectRequest):
        return projects.open(payload.path)

    @app.get("/api/v1/projects/{identity}/workspace")
    def project_workspace(identity: str):
        return projects.workspace(identity)

    @app.patch("/api/v1/projects/{identity}")
    def update_project(identity: str, payload: ProjectUpdateRequest):
        return projects.update_config(identity, payload.config)

    @app.post("/api/v1/projects/{identity}/sources", status_code=201)
    def project_source(identity: str, payload: ProjectSourceRequest):
        return projects.add_source(identity, payload.path, payload.role)

    @app.get("/api/v1/projects/{identity}/storage")
    def project_storage(identity: str):
        return projects.scientific_store(identity).status()

    @app.get("/api/v1/projects/{identity}/drafts")
    def project_drafts(identity: str):
        return {"drafts": projects.scientific_store(identity).list_drafts()}

    @app.post("/api/v1/projects/{identity}/drafts", status_code=201)
    def create_project_draft(identity: str, payload: CreateDraftRequest):
        return projects.scientific_store(identity).create_draft(
            kind=payload.kind, name=payload.name, payload=payload.payload
        )

    @app.get("/api/v1/projects/{identity}/drafts/{draft_id}")
    def project_draft(identity: str, draft_id: str):
        return projects.scientific_store(identity).get_draft(draft_id)

    @app.patch("/api/v1/projects/{identity}/drafts/{draft_id}")
    def update_project_draft(identity: str, draft_id: str, payload: UpdateDraftRequest):
        return projects.scientific_store(identity).update_draft(
            draft_id,
            expected_revision=payload.expectedRevision,
            name=payload.name,
            payload=payload.payload,
        )

    @app.get("/api/v1/projects/{identity}/datasets")
    def project_datasets(identity: str):
        return {"datasets": projects.scientific_store(identity).list_datasets()}

    @app.get("/api/v1/projects/{identity}/datasets/{dataset_id}")
    def project_dataset(identity: str, dataset_id: str):
        return projects.scientific_store(identity).get_dataset(dataset_id)

    @app.get("/api/v1/workspace/export")
    def export_workspace():
        return JSONResponse(
            {
                "schema_version": 1,
                "mode": "synthetic-demo",
                "executable": False,
                **workspace.workspace(),
            },
            headers={"Content-Disposition": 'attachment; filename="histopilot-workspace.json"'},
        )

    @app.get("/api/v1/models/encoders")
    def encoders():
        return {"encoders": workspace.models("encoders")}

    @app.get("/api/v1/models/mil")
    def mil_models():
        return {"milModels": workspace.models("milModels")}

    @app.post("/api/v1/cohorts", status_code=201)
    def save_cohort(payload: CohortRequest):
        return workspace.save_cohort(payload)

    @app.post("/api/v1/experiments", status_code=201)
    def save_experiments(payload: ExperimentRequest):
        return {"drafts": workspace.save_experiments(payload)}

    @app.delete("/api/v1/experiments/{identity}", status_code=204)
    def delete_experiment(identity: str):
        workspace.delete_experiment(identity)
        return Response(status_code=204)

    @app.get("/api/v1/experiments/{identity}/manifest")
    def experiment_manifest(identity: str):
        return workspace.experiment_manifest(identity)

    @app.get("/api/v1/filesystem/roots")
    def roots(purpose: Literal["source", "storage"] = "source"):
        return (storage if purpose == "storage" else filesystem).root_listing()

    @app.get("/api/v1/filesystem/list")
    def list_directory(
        path: str = Query(min_length=1, max_length=4096),
        purpose: Literal["source", "storage"] = "source",
    ):
        return (storage if purpose == "storage" else filesystem).list_directory(path)

    @app.post("/api/v1/sources", status_code=201)
    def register_source(payload: SourceRequest):
        return workspace.add_source(payload.path)

    @app.get("/api/v1/system")
    def system():
        return {
            "mode": "local-first",
            "workspace": str(settings.workspace),
            "storage": {"engine": "sqlite", "journalMode": "wal", "schemaVersion": SCHEMA_VERSION},
            "control": {"cudaModelsLoaded": False, "process": "control-service"},
            "workers": {"executionEnabled": False, "status": "not-implemented"},
            "sourcesReadOnly": True,
            "diagnostics": system_report(),
        }

    @app.get("/api/v1/jobs")
    def jobs():
        return {"jobs": [], "executionEnabled": False}

    @app.get("/api/v1/jobs/events")
    def job_events():
        raise HTTPException(501, "SSE worker progress is not implemented.")

    @app.post("/api/v1/jobs")
    def submit_job():
        raise HTTPException(501, "Compute execution is not implemented. No job was submitted.")

    app.include_router(scientific_router(projects, filesystem))

    @app.get("/{path:path}")
    def frontend(path: str):
        if path == "api" or path.startswith("api/"):
            raise HTTPException(404, "Unknown API endpoint.")
        root = settings.static_dir.resolve()
        try:
            candidate = (root / path).resolve()
        except (OSError, RuntimeError, ValueError):
            raise HTTPException(404, "Static asset not found.") from None
        if not candidate.is_relative_to(root) or any(
            part.startswith(".") for part in Path(path).parts
        ):
            raise HTTPException(404, "Static asset not found.")
        suffixes = {
            ".html",
            ".js",
            ".css",
            ".svg",
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".ico",
            ".woff",
            ".woff2",
        }
        if candidate.is_file() and candidate.suffix.lower() in suffixes:
            return FileResponse(candidate)
        if Path(path).suffix or path.startswith("assets/"):
            raise HTTPException(404, "Static asset not found.")
        index = root / "index.html"
        if not index.is_file() or not index.resolve().is_relative_to(root):
            raise HTTPException(
                503,
                "The frontend bundle is missing. Build web/ or use the Vite development server.",
            )
        return FileResponse(index)

    return app
