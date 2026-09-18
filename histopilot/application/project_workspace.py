"""Folder-backed project setup and a central recent-project registry.

The selected folder owns its descriptor; SQLite is only a discovery index.
Registering a source path does not import a dataset or authorize execution.
"""

import json
import os
import stat
import tempfile
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Literal
from uuid import uuid4

from pydantic import Field, ValidationError
from sqlalchemy import select

from histopilot.application.blca_demo import DEMO_ID, demo_summary, load_demo
from histopilot.application.local_workspace import LocalWorkspace, WorkspaceError, _identity, _now
from histopilot.schemas.workspace import (
    ProjectConfig,
    ProjectRequest,
    RequestModel,
)
from histopilot.storage.database import Database, Record
from histopilot.storage.filesystem import FilesystemError, LocalFilesystem
from histopilot.storage.lifecycle import LifecycleStore, lifecycle_guard
from histopilot.storage.project_lock import StorageError, fsync_directory, writer_lock
from histopilot.storage.scientific import ScientificStore

DESCRIPTOR = "histopilot-project.json"
DESCRIPTOR_LIMIT = 1024 * 1024


def _timestamp() -> str:
    return _now().replace("+00:00", "Z")


class StoredSource(RequestModel):
    id: str = Field(pattern=r"^source-[a-f0-9]{20}$")
    path: str = Field(min_length=1, max_length=4096)
    name: str = Field(min_length=1, max_length=4096)
    kind: Literal["directory"] = "directory"
    role: Literal["data", "slides", "features"]
    readOnly: Literal[True] = True
    importStatus: Literal["not-imported"] = "not-imported"
    createdAt: datetime


class ProjectDocument(RequestModel):
    format: Literal["histopilot-project"]
    schemaVersion: Literal[1]
    id: str = Field(pattern=r"^project-[a-f0-9]{32}$")
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=2000)
    createdAt: datetime
    updatedAt: datetime
    config: ProjectConfig = Field(default_factory=ProjectConfig)
    sources: list[StoredSource] = Field(default_factory=list, max_length=1000)


class ProjectWorkspace:
    def __init__(
        self,
        database: Database,
        legacy: LocalWorkspace,
        storage: LocalFilesystem,
    ):
        self.database = database
        self.legacy = legacy
        self.storage = storage
        self.lock = RLock()

    def _demo(self) -> dict:
        return {
            "id": "synthetic-v1",
            "name": "CRC KRAS · synthetic demo",
            "description": "Synthetic example data for exploring the workflow.",
            "storagePath": str(self.database.workspace),
            "mode": "synthetic-demo",
            "createdAt": "",
            "updatedAt": "",
            "config": {},
            "sources": [],
            "available": True,
        }

    @staticmethod
    def _summary(document: dict, path: Path) -> dict:
        return {
            **{
                key: deepcopy(value)
                for key, value in document.items()
                if key not in {"format", "schemaVersion"}
            },
            "storagePath": str(path),
            "mode": "local",
            "available": True,
            "lifecycleState": LifecycleStore(path, document["id"])
            .read()["records"]
            .get(f"project:{document['id']}", {})
            .get("state", "active"),
        }

    @staticmethod
    def _read(path: Path) -> dict:
        descriptor = path / DESCRIPTOR
        try:
            if descriptor.is_symlink():
                raise WorkspaceError("The project descriptor must not be a symbolic link.", 403)
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
            descriptor_fd = os.open(descriptor, flags)
            try:
                metadata = os.fstat(descriptor_fd)
                if not stat.S_ISREG(metadata.st_mode):
                    raise WorkspaceError("The project descriptor must be a regular file.")
                if metadata.st_size > DESCRIPTOR_LIMIT:
                    raise WorkspaceError("The project descriptor exceeds the supported size.")
                with os.fdopen(descriptor_fd, "rb", closefd=False) as handle:
                    content = handle.read(DESCRIPTOR_LIMIT + 1)
                if len(content) > DESCRIPTOR_LIMIT:
                    raise WorkspaceError("The project descriptor exceeds the supported size.")
            finally:
                os.close(descriptor_fd)
            document = ProjectDocument.model_validate_json(content)
            return document.model_dump(mode="json", exclude_none=True)
        except FileNotFoundError:
            raise WorkspaceError(
                f"This folder has no {DESCRIPTOR}. Select an existing HistoPilot project.", 404
            ) from None
        except (OSError, RuntimeError):
            raise WorkspaceError("The project descriptor cannot be read.", 403) from None
        except ValidationError:
            raise WorkspaceError(
                "The project descriptor is invalid or uses an unsupported format."
            ) from None

    @staticmethod
    def _write(path: Path, document: dict, *, create: bool = False) -> None:
        """Atomic durable replacement, with an exclusive initial publication."""
        temporary: str | None = None
        content = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
        if len(content.encode("utf-8")) > DESCRIPTOR_LIMIT:
            raise WorkspaceError("The project descriptor exceeds the supported size.")
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix=".histopilot-",
                suffix=".tmp",
                dir=path,
                delete=False,
            ) as handle:
                temporary = handle.name
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            if create:
                os.link(temporary, path / DESCRIPTOR)
            else:
                os.replace(temporary, path / DESCRIPTOR)
                temporary = None
            fsync_directory(path)
        except FileExistsError:
            raise WorkspaceError(
                "A project already exists in this folder. Load it instead.", 409
            ) from None
        except OSError:
            raise WorkspaceError(
                "The project folder cannot be written. Choose a writable location.", 403
            ) from None
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)
                fsync_directory(path)

    def _register(self, summary: dict) -> None:
        with self.database.sessions.begin() as session:
            record = session.get(Record, ("project", summary["id"]))
            if record is None:
                session.add(Record(kind="project", id=summary["id"], payload=summary))
            else:
                record.payload = summary

    def list_projects(self) -> dict:
        with self.database.sessions.begin() as session:
            summaries = [
                deepcopy(record.payload)
                for record in session.scalars(select(Record).where(Record.kind == "project"))
            ]
        projects = []
        for summary in summaries:
            try:
                path = self.storage.directory(summary["storagePath"])
                document = self._read(path)
                if document["id"] != summary["id"]:
                    raise WorkspaceError("The folder now belongs to a different project.", 409)
                projects.append(self._summary(document, path))
            except (FilesystemError, WorkspaceError, StorageError) as error:
                projects.append({**summary, "available": False, "unavailableReason": str(error)})
        projects.sort(key=lambda project: project["updatedAt"], reverse=True)
        return {
            "projects": projects + [demo_summary()],
            "defaultStoragePath": str(self.database.workspace),
        }

    def _validate_config(self, config: ProjectConfig) -> dict:
        for value, kind in ((config.encoderId, "encoders"), (config.milId, "milModels")):
            if value is not None and value not in {
                model["id"] for model in self.legacy.models(kind)
            }:
                raise WorkspaceError("Select an encoder and MIL model from the service registry.")
        return config.model_dump(exclude_none=True)

    def _source(self, identity: str, value: str, role: str) -> dict:
        path = self.legacy.filesystem.directory(value)
        return {
            "id": _identity("source", [identity, str(path), role]),
            "path": str(path),
            "name": path.name or str(path),
            "kind": "directory",
            "role": role,
            "readOnly": True,
            "importStatus": "not-imported",
            "createdAt": _timestamp(),
        }

    def create(self, request: ProjectRequest) -> dict:
        config = self._validate_config(request.config)
        identity = f"project-{uuid4().hex}"
        sources = [
            self._source(identity, value, role)
            for role, value in (
                ("data", request.dataPath),
                ("slides", request.slidePath),
                ("features", request.featurePath),
            )
            if value is not None
        ]
        candidate = Path(request.storagePath)
        if not candidate.is_absolute() or "\x00" in request.storagePath:
            raise FilesystemError(
                "Select an absolute project folder within permitted storage roots."
            )
        # Requiring an existing parent avoids implicitly creating an arbitrary tree.
        if candidate.exists() or candidate.is_symlink():
            path = self.storage.directory(str(candidate))
        else:
            parent = self.storage.directory(str(candidate.parent))
            path = parent / candidate.name
        with self.lock:
            created = False
            try:
                if path.exists() or path.is_symlink():
                    path = self.storage.directory(str(path))
                    if next(path.iterdir(), None) is not None:
                        raise WorkspaceError(
                            "The selected folder is not empty. Choose a new or empty folder, "
                            "or load its existing project.",
                            409,
                        )
                else:
                    path.mkdir()
                    fsync_directory(path.parent)
                    created = True
                now = _timestamp()
                document = {
                    "format": "histopilot-project",
                    "schemaVersion": 1,
                    "id": identity,
                    "name": request.name,
                    "description": request.description,
                    "createdAt": now,
                    "updatedAt": now,
                    "config": config,
                    "sources": sources,
                }
                # Use one canonical JSON representation on both create and reload.
                document = ProjectDocument.model_validate(document).model_dump(
                    mode="json", exclude_none=True
                )
                self._write(path, document, create=True)
            except OSError:
                raise WorkspaceError(
                    "The project folder cannot be created or inspected.", 403
                ) from None
            finally:
                if created and not (path / DESCRIPTOR).exists():
                    try:
                        path.rmdir()
                    except OSError:
                        pass
            # Descriptor publication is independent of scientific initialization.
            # If initialization fails, the valid folder can be reopened and retried.
            ScientificStore(path, identity).initialize()
            summary = self._summary(document, path)
            self._register(summary)
            return summary

    def open(self, value: str) -> dict:
        with self.lock:
            path = self.storage.directory(value)
            document = self._read(path)
            with self.database.sessions.begin() as session:
                record = session.get(Record, ("project", document["id"]))
                if record is not None:
                    previous = Path(record.payload["storagePath"])
                    if previous != path and previous.exists():
                        raise WorkspaceError(
                            "This project ID is already registered at another folder.", 409
                        )
            ScientificStore(path, document["id"]).initialize()
            summary = self._summary(document, path)
            self._register(summary)
            return summary

    def _load(self, identity: str) -> tuple[dict, Path]:
        if identity == DEMO_ID:
            raise WorkspaceError("The BLCA demo is read only and has no local project folder.", 409)
        with self.database.sessions.begin() as session:
            record = session.get(Record, ("project", identity))
            if record is None:
                raise WorkspaceError("The project does not exist. Load its folder first.", 404)
            path = self.storage.directory(record.payload["storagePath"])
        document = self._read(path)
        if document["id"] != identity:
            raise WorkspaceError("The folder now belongs to a different project.", 409)
        return document, path

    def scientific_store(self, identity: str) -> ScientificStore:
        """Resolve storage from the folder; the central registry holds no scientific state."""
        if identity in ("synthetic-v1", DEMO_ID):
            raise StorageError(
                "The synthetic demo has no local scientific store.", "DEMO_STORE_UNAVAILABLE"
            )
        document, path = self._load(identity)
        # Each store operation initializes/reconciles under its own writer lock.
        return ScientificStore(path, document["id"])

    def workspace(self, identity: str) -> dict:
        if identity == DEMO_ID:
            return load_demo()
        if identity == "synthetic-v1":
            return {**self.legacy.workspace(), "project": self._demo()}
        document, path = self._load(identity)
        scientific = ScientificStore(path, document["id"])
        datasets = [
            value
            for value in scientific.list_datasets()
            if value["manifest"].get("kind") == "dataset"
        ]
        latest = max(datasets, key=lambda value: value["createdAt"], default=None)
        summary = latest["manifest"].get("summary", {}) if latest else {}
        configurations = scientific.list_configurations()
        return {
            "mode": "local",
            "executionEnabled": False,
            "scientificStorage": scientific.status(),
            "project": self._summary(document, path),
            "dataset": {
                "id": latest["id"] if latest else "",
                **(
                    {
                        "name": latest["manifest"].get("name", "Dataset version"),
                        "versionLabel": latest.get("versionLabel"),
                    }
                    if latest
                    else {}
                ),
                "patientCount": summary.get(
                    "verifiedPatientCount", summary.get("mappedPatientCount", 0)
                ),
                "fallbackSlideCount": summary.get("fallbackSlideCount", 0),
                "groupCount": summary.get("mappedPatientCount", 0),
                "unlinkedSlideCount": summary.get("unlinkedSlideCount", 0),
                "specimenCount": 0,
                "slideCount": summary.get("slideCount", 0),
            },
            "scientificSummary": {
                "datasetCount": len(datasets),
                "protocolCount": sum(
                    item["manifest"].get("kind") == "protocol" for item in configurations
                ),
                "featureCount": sum(
                    item["manifest"].get("kind") == "feature" for item in configurations
                ),
            },
            "patients": [],
            "slides": [],
            "featureSets": [],
            "results": [],
            "cohortSnapshots": [],
            "drafts": [],
            "exampleManifests": [],
            "sources": deepcopy(document["sources"]),
            "encoders": self.legacy.models("encoders"),
            "milModels": self.legacy.models("milModels"),
            "split": {
                "id": "",
                "seed": document["config"].get("seed", 42),
                "groupBy": "patient_id",
            },
        }

    def update_config(
        self,
        identity: str,
        config: ProjectConfig,
        expected_config: ProjectConfig | None = None,
    ) -> dict:
        choices = self._validate_config(config)
        with self.lock:
            _, path = self._load(identity)
            with lifecycle_guard(path), writer_lock(path):
                LifecycleStore(path, identity).assert_usable([f"project:{identity}"])
                document = self._read(path)
                if document["id"] != identity:
                    raise WorkspaceError("The folder now belongs to a different project.", 409)
                if (
                    expected_config is not None
                    and document["config"] != expected_config.model_dump(exclude_none=True)
                    and document["config"] != choices
                ):
                    raise StorageError(
                        "Project settings changed in another tab. Your edits have not been saved. "
                        "Reload the saved settings before making further changes.",
                        "PROJECT_CONFIG_CONFLICT",
                        409,
                    )
                # Retrying a save whose response was lost is a no-op. Source
                # additions do not invalidate the independent settings baseline.
                if document["config"] == choices:
                    return self._summary(document, path)
                document["config"] = choices
                document["updatedAt"] = _timestamp()
                self._write(path, document)
            summary = self._summary(document, path)
            self._register(summary)
            return summary

    def add_source(self, identity: str, value: str, role: str) -> dict:
        with self.lock:
            _, path = self._load(identity)
            with lifecycle_guard(path), writer_lock(path):
                LifecycleStore(path, identity).assert_usable([f"project:{identity}"])
                document = self._read(path)
                if document["id"] != identity:
                    raise WorkspaceError("The folder now belongs to a different project.", 409)
                source = self._source(identity, value, role)
                for existing in document["sources"]:
                    if existing["id"] == source["id"]:
                        return existing
                if len(document["sources"]) >= 1000:
                    raise WorkspaceError(
                        "The project has reached the maximum of 1000 source folders."
                    )
                document["sources"].append(source)
                document["updatedAt"] = _timestamp()
                self._write(path, document)
            self._register(self._summary(document, path))
            return source
