"""Single-process local metadata service and packaged React static assets."""

import shutil
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from secrets import token_urlsafe
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from histopilot import __version__
from histopilot.adapters.trident import discover_runtime
from histopilot.application import creations
from histopilot.application.project_workspace import ProjectWorkspace, WorkspaceError
from histopilot.application.system_compute import ComputeSampler
from histopilot.config import Settings, load_settings
from histopilot.doctor import system_report
from histopilot.schemas.scientific import CreateDraftRequest, UpdateDraftRequest
from histopilot.schemas.workspace import (
    CreateDirectoryRequest,
    OpenProjectRequest,
    ProjectRequest,
    ProjectSourceRequest,
    ProjectUpdateRequest,
)
from histopilot.storage.database import SCHEMA_VERSION, Database
from histopilot.storage.filesystem import FilesystemError, LocalFilesystem
from histopilot.storage.lifecycle import lifecycle_guard
from histopilot.storage.project_lock import StorageError

from . import login
from .access import access_router, token_project
from .lifecycle import lifecycle_router
from .responses import coded_response
from .scientific import scientific_router
from .scopes import ScopeGate, audit_session
from .security import configure_browser_boundary


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    database = Database(settings.workspace)
    filesystem = LocalFilesystem(settings.data_roots)
    storage = LocalFilesystem((settings.workspace, *settings.data_roots))
    projects = ProjectWorkspace(database, filesystem, storage)
    compute_sampler = ComputeSampler(settings.workspace, settings.data_roots)
    token = token_urlsafe(32)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            database.initialize()
            yield
        finally:
            from histopilot.viewer.image_cache import SLIDE_IMAGES
            from histopilot.viewer.reader_cache import OPENSLIDE_READERS
            from histopilot.viewer.sdpc import close_readers

            try:
                SLIDE_IMAGES.clear()
                OPENSLIDE_READERS.close()
                close_readers()
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
    from starlette.concurrency import run_in_threadpool

    from histopilot.application.access_tokens import AccessTokens

    tokens = AccessTokens(database)

    async def audited(request: Request, response) -> None:
        # Commit and admin requests made with the session token join the agents' audit log.
        await run_in_threadpool(
            audit_session,
            projects,
            request.method,
            request.url.path,
            response.status_code,
            request.headers.get("x-histopilot-client"),
            getattr(request.state, "audit_project", None),
        )

    def work_project(what: str, key: str) -> str | None:
        # The Task Center is built below; a scoped token's request arrives after both.
        center = app.state.task_center
        try:
            found = center.task(key).get("owner") if what == "task" else center.owner(key)
        except StorageError:
            return None
        return (found or {}).get("projectId")

    configure_browser_boundary(
        app,
        settings,
        token,
        scoped=ScopeGate(projects, tokens, database, work_project=work_project),
        after_session=audited,
        login_secret=login.ensure_secret(settings.port) if settings.login else None,
    )

    @app.exception_handler(WorkspaceError)
    @app.exception_handler(FilesystemError)
    @app.exception_handler(StorageError)
    async def coded_error(
        _request: Request, error: WorkspaceError | FilesystemError | StorageError
    ):
        return coded_response(error)

    @app.exception_handler(RequestValidationError)
    async def invalid_schema(_request: Request, error: RequestValidationError):
        # detail stays FastAPI's list of rejected fields, which the browser formats.
        return JSONResponse(
            {"detail": jsonable_encoder(error.errors()), "code": "REQUEST_INVALID"},
            status_code=422,
        )

    @app.exception_handler(StarletteHTTPException)
    async def unknown_route(request: Request, error: StarletteHTTPException):
        # Static assets keep FastAPI's plain body; only API routes carry codes.
        if request.url.path.startswith("/api/") and error.status_code == 404:
            return JSONResponse(
                {"detail": error.detail, "code": "API_ENDPOINT_UNKNOWN"}, status_code=404
            )
        if request.url.path.startswith("/api/") and error.status_code == 405:
            return JSONResponse(
                {"detail": error.detail, "code": "API_METHOD_NOT_ALLOWED"},
                status_code=405,
                headers=error.headers,
            )
        return await http_exception_handler(request, error)

    @app.exception_handler(Exception)
    async def unexpected_error(_request: Request, _error: Exception):
        # Starlette raises the error again after sending this, so the server logs it.
        return JSONResponse(
            {
                "detail": "HistoPilot hit an unexpected error. The server log has the details.",
                "code": "INTERNAL_ERROR",
            },
            status_code=500,
        )

    @app.get("/api/v1/health")
    def health():
        # Legacy generic/demo job submission remains disabled. Project execution
        # capabilities and runtime readiness are reported by authenticated routes.
        return {"status": "ok", "version": __version__, "executionEnabled": False}

    from histopilot.version_info import FEATURES, git_revision

    # Read at startup: a checkout moved on afterwards must not change what this process reports.
    revision = git_revision()

    @app.get("/api/v1/version")
    def version():
        """Which code serves this API, so clients can notice a mismatched checkout."""
        from histopilot import templates

        return {
            "version": __version__,
            "apiVersion": 1,
            "contractVersion": 1,
            "gitRevision": revision,
            "workspaceSchemaVersion": SCHEMA_VERSION,
            "templatesVersion": templates.VERSION,
            "features": list(FEATURES),
        }

    @app.get("/api/v1/templates")
    def service_templates():
        """The server-owned starting specs and presets, as the browser reads them."""
        from histopilot import templates

        return templates.describe()

    @app.get("/api/v1/session")
    def session():
        return {
            "token": token,
            "scientificCapabilities": {"versionLabels": True, "taggedFreeze": True},
        }

    @app.get("/api/v1/projects")
    def list_projects(request: Request):
        listed = projects.list_projects()
        own = token_project(request)
        if own is not None:
            # A scoped token sees its own project and nothing about the workspace.
            return {"projects": [item for item in listed["projects"] if item["id"] == own]}
        return listed

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
        return projects.update_config(identity, payload.config, payload.expectedConfig)

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
        store = projects.scientific_store(identity)
        values = payload.model_dump(mode="json", exclude={"operationId"})

        def create():
            guard_experiment_draft(store, payload.payload)
            return store.create_draft(kind=payload.kind, name=payload.name, payload=payload.payload)

        with lifecycle_guard(store.folder):
            return creations.once(
                database,
                f"draft:{identity}",
                payload.operationId,
                values,
                create,
                lambda draft: store.get_draft(draft, include_inactive=True),
            )

    @app.get("/api/v1/projects/{identity}/drafts/{draft_id}")
    def project_draft(identity: str, draft_id: str):
        return projects.scientific_store(identity).get_draft(draft_id)

    @app.patch("/api/v1/projects/{identity}/drafts/{draft_id}")
    def update_project_draft(identity: str, draft_id: str, payload: UpdateDraftRequest):
        store = projects.scientific_store(identity)
        with lifecycle_guard(store.folder):
            current = store.get_draft(draft_id)
            guard_experiment_draft(store, payload.payload, current["payload"])
            return store.update_draft(
                draft_id,
                expected_revision=payload.expectedRevision,
                name=payload.name,
                payload=payload.payload,
            )

    def guard_experiment_draft(store, payload, previous=None):
        from histopilot.application.model_experiments import ModelExperimentService

        if (
            payload.get("type") == "model-experiment"
            or (previous or {}).get("type") == "model-experiment"
        ):
            raise StorageError(
                "Manage model experiments through their experiment record.",
                "EXPERIMENT_TYPED_ENDPOINT_REQUIRED",
                409,
            )
        spec = payload.get("spec") or {}
        old_spec = (previous or {}).get("spec") or {}
        spec = spec if isinstance(spec, dict) else {}
        old_spec = old_spec if isinstance(old_spec, dict) else {}
        nested_owner, direct_owner = spec.get("experimentId"), payload.get("experimentId")
        if nested_owner is not None and direct_owner is not None and nested_owner != direct_owner:
            raise StorageError(
                "The draft names conflicting experiment owners.", "EXPERIMENT_OWNER_CHANGED", 409
            )
        owner = nested_owner if nested_owner is not None else direct_owner
        old_owner = old_spec.get("experimentId") or (previous or {}).get("experimentId")
        if old_owner and (owner != old_owner or payload.get("type") != previous.get("type")):
            raise StorageError(
                "A batch draft cannot change its experiment owner.", "EXPERIMENT_OWNER_CHANGED", 409
            )
        if owner is not None:
            revision = spec.get("experimentRevision", payload.get("experimentRevision"))
            if (
                payload.get("experimentRevision") is not None
                and spec.get("experimentRevision") is not None
                and payload["experimentRevision"] != spec["experimentRevision"]
            ):
                raise StorageError(
                    "The draft names conflicting experiment revisions.",
                    "INVALID_EXPERIMENT_REFERENCE",
                    422,
                )
            if not isinstance(owner, str) or not owner or type(revision) is not int or revision < 1:
                raise StorageError(
                    "A batch draft requires its experiment ID and revision.",
                    "INVALID_EXPERIMENT_REFERENCE",
                    422,
                )
            if payload.get("type") not in ("development-batch", "mil-experiment"):
                raise StorageError(
                    "Invalid experiment draft type.", "INVALID_EXPERIMENT_DRAFT", 422
                )
            ModelExperimentService(store, filesystem).require_editable(owner, revision)

    @app.get("/api/v1/projects/{identity}/datasets")
    def project_datasets(identity: str):
        return {"datasets": projects.scientific_store(identity).list_datasets()}

    @app.get("/api/v1/projects/{identity}/datasets/{dataset_id}")
    def project_dataset(identity: str, dataset_id: str):
        return projects.scientific_store(identity).get_dataset(dataset_id)

    @app.get("/api/v1/filesystem/roots")
    def roots(purpose: Literal["source", "storage"] = "source"):
        return (storage if purpose == "storage" else filesystem).root_listing()

    @app.get("/api/v1/filesystem/list")
    def list_directory(
        path: str = Query(min_length=1, max_length=4096),
        purpose: Literal["source", "storage"] = "source",
    ):
        return (storage if purpose == "storage" else filesystem).list_directory(path)

    @app.post("/api/v1/filesystem/directories", status_code=201)
    def create_directory(payload: CreateDirectoryRequest):
        selected_filesystem = storage if payload.purpose == "storage" else filesystem
        return selected_filesystem.create_directory(payload.parentPath, payload.name)

    @app.get("/api/v1/system")
    def system():
        trident = discover_runtime()
        # tmux hosts the Task Center runner, which runs every job.
        tmux_available = shutil.which("tmux") is not None
        extraction_ready = trident["available"] and tmux_available
        return {
            "mode": "local-first",
            "workspace": str(settings.workspace),
            "storage": {"engine": "sqlite", "journalMode": "wal", "schemaVersion": SCHEMA_VERSION},
            # Measured, not assumed: the service never imports Torch; workers load models.
            "control": {"process": "control-service", "torchImported": "torch" in sys.modules},
            "workers": {
                # Retained for old clients: this flag describes TRIDENT only.
                "executionEnabled": extraction_ready,
                "tmuxAvailable": tmux_available,
                "nativeExecutionImplemented": True,
                "status": (
                    "TRIDENT extraction available. "
                    if extraction_ready
                    else "TRIDENT extraction requires runtime setup or tmux. "
                )
                + "ABMIL and nnMIL training, refitting, evaluation and attention are implemented; "
                "check runtime readiness in their modules.",
                "trident": trident,
            },
            "sourcesReadOnly": True,
            "diagnostics": system_report(),
        }

    @app.get("/api/v1/system/compute")
    def system_compute():
        return compute_sampler.snapshot()

    app.include_router(scientific_router(projects, filesystem))
    app.include_router(lifecycle_router(projects, filesystem))
    from histopilot.api.case_review import case_review_router
    from histopilot.api.clinical import clinical_router
    from histopilot.api.evaluations import evaluation_router
    from histopilot.api.inference import inference_router
    from histopilot.api.interpretation import interpretation_router
    from histopilot.api.mil import mil_router
    from histopilot.api.model_experiments import model_experiments_router
    from histopilot.api.morphology import morphology_router
    from histopilot.api.operations import operations_router
    from histopilot.api.performance import performance_router
    from histopilot.api.predictors import evaluation_run_router, predictor_router
    from histopilot.api.references import references_router
    from histopilot.api.slide_reviews import slide_review_router
    from histopilot.api.task_center import task_center_router
    from histopilot.taskcenter.service import TaskCenterService

    # Machine-wide queue: the store is opened lazily on the first request.
    task_center = TaskCenterService(projects, filesystem, settings.workspace)
    app.state.task_center = task_center
    app.include_router(task_center_router(task_center))
    app.include_router(mil_router(projects, filesystem))
    app.include_router(evaluation_router(projects, filesystem))
    app.include_router(model_experiments_router(projects, filesystem))
    app.include_router(predictor_router(projects, filesystem))
    app.include_router(evaluation_run_router(projects, filesystem))
    app.include_router(clinical_router(projects, filesystem))
    app.include_router(interpretation_router(projects, filesystem))
    app.include_router(slide_review_router(projects, filesystem))
    app.include_router(morphology_router(projects, filesystem))
    app.include_router(case_review_router(projects, filesystem))
    app.include_router(inference_router(projects, filesystem))
    app.include_router(performance_router(projects, filesystem))
    app.include_router(references_router(projects, filesystem))
    app.include_router(operations_router(projects, filesystem))
    app.include_router(access_router(projects, tokens, database))

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
            headers = {"Cache-Control": "no-cache"} if candidate.suffix.lower() == ".html" else None
            return FileResponse(candidate, headers=headers)
        if Path(path).suffix or path.startswith("assets/"):
            raise HTTPException(404, "Static asset not found.")
        index = root / "index.html"
        if not index.is_file() or not index.resolve().is_relative_to(root):
            raise HTTPException(
                503,
                "The frontend bundle is missing. Build web/ or use the Vite development server.",
            )
        # Revalidate the application shell so a rebuilt bundle replaces its old asset links.
        return FileResponse(index, headers={"Cache-Control": "no-cache"})

    return app
