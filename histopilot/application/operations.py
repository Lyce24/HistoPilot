"""Verified project archives and a unified view of existing job ownership.

External inputs are references, never implicitly copied or rewritten. Immutable
scientific records retain their original identities, hashes and source paths.
Archive work is run by a persistent worker; the control API only submits plans.
"""

import hashlib
import json
import os
import re
import shutil
import sqlite3
import stat
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from histopilot.application.lifecycle import CleanupService
from histopilot.application.project_workspace import (
    DESCRIPTOR,
    DESCRIPTOR_LIMIT,
    ProjectDocument,
    ProjectWorkspace,
)
from histopilot.storage.lifecycle import LifecycleStore, lifecycle_guard
from histopilot.storage.project_lock import (
    LOCK_FILE,
    StorageError,
    _reject_symlink_components,
    fsync_directory,
    writer_lock,
)
from histopilot.storage.scientific import DATABASE_FILE, ScientificStore

MANIFEST = "histopilot-archive.json"
MAX_MEMBERS = 200_000
MAX_MANIFEST_BYTES = 64 * 1024 * 1024
MAX_RESTORE_BYTES = 16 * 1024**4
TRANSIENT = {LOCK_FILE, ".histopilot-lifecycle.lock", ".histopilot-portability.lock"}
ACTIVE = {"queued", "starting", "running", "cancelling", "unknown"}


class PortabilityCancelled(ValueError):
    pass


def _now():
    return datetime.now(UTC).isoformat()


def _error(message, code="PORTABILITY_INVALID", status=409):
    return StorageError(message, code, status)


def permitted_path(filesystem, value, *, existing=False):
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or "\x00" in value:
        raise _error("Choose an absolute path without parent traversal.", status=422)
    _reject_symlink_components(path)
    if not filesystem._contains(path.resolve()):
        raise _error("The path is outside configured storage roots.", status=403)
    if existing and not path.exists():
        raise _error("The selected path does not exist.", status=404)
    return path


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def source_inventory(store, filesystem):
    """Check references without opening external files or decoding slide tensors."""
    document = ProjectWorkspace._read(store.folder)
    sources = []
    for source in document["sources"]:
        path = Path(source["path"])
        permitted = filesystem._contains(path.resolve())
        sources.append({**source, "available": permitted and path.is_dir(), "permitted": permitted})
    paths = set()
    documents = [
        *store.list_datasets(include_inactive=True),
        *store.list_configurations(include_inactive=True),
    ]
    for document in documents:
        paths.update(
            value
            for value in _strings(document.get("manifest", {}))
            if value.startswith("/") and "\x00" not in value
        )
        if document.get("manifest", {}).get("kind") == "dataset":
            for artifact in ("records.json", "inventory.json"):
                if artifact in document.get("artifacts", {}):
                    rows = json.loads(
                        store.read_artifact(document["id"], artifact, include_inactive=True)
                    )
                    paths.update(
                        value
                        for value in _strings(rows)
                        if value.startswith("/") and "\x00" not in value
                    )
        if len(paths) > 100_000:
            raise _error("The source inventory exceeds 100,000 references.", status=413)
    missing = []
    for value in sorted(paths):
        path = Path(value)
        if path.is_relative_to(store.folder):
            continue
        allowed = filesystem._contains(path.resolve())
        if not allowed or not path.exists():
            missing.append({"path": value, "reason": "missing" if allowed else "outside-roots"})
    return {
        "sources": sources,
        "referenceCount": len(paths),
        "missingReferences": missing,
        "note": "Source registrations apply to new imports. Frozen versions retain their original source paths and must be re-imported and validated after a move.",
    }


def operations_inventory(store, filesystem):
    catalog = CleanupService(store, filesystem).catalog()
    jobs = []
    for item in catalog["items"]:
        if item.get("job"):
            row = {key: item[key] for key in ("key", "id", "kind", "name", "job")}
            if item["type"] in {"packing", "extraction"} and row["job"]["status"] == "queued":
                row["job"]["waitingReason"] = "Waiting for shared CPU, RAM or GPU capacity."
            jobs.append(row)
    jobs.sort(key=lambda item: (item["job"]["status"] not in ACTIVE, item["name"]))
    from histopilot.workers.train_batch import _capacity, _leases

    with _leases() as (_, active):
        reservations = [
            {
                key: value.get(key)
                for key in ("batchId", "runId", "kind", "cpus", "ramGb", "gpu", "runsPerGpu")
            }
            for value in active
        ]
    cpus, ram = _capacity()
    return {
        "projectId": store.project_id,
        "jobs": jobs,
        "reservations": reservations,
        "capacity": {"cpus": cpus, "availableRamGb": round(ram, 2)},
        "note": "Reservations coordinate this user's HistoPilot workers across projects. Running jobs keep their original ownership and cancellation controls.",
    }


class StudyPortability:
    def __init__(self, store, filesystem):
        self.store = store
        self.filesystem = filesystem

    def _idle(self):
        # Caller holds the launch/lifecycle gate. Existing workers must be terminal
        # before we hold the writer lock and snapshot their durable files.
        catalog = CleanupService(self.store, self.filesystem)._catalog()
        if any(
            item.get("job", {}).get("busy") or item.get("job", {}).get("status") in ACTIVE
            for item in catalog["items"]
        ):
            raise _error(
                "Finish or cancel active jobs before exporting a consistent project.",
                "PORTABILITY_ACTIVE_JOBS",
            )

    def export(self, value, *, progress=None, operation_id=None):
        archive = permitted_path(self.filesystem, value)
        folder = self.store.folder
        if archive.is_relative_to(folder):
            raise _error("Save the archive outside the project folder.", status=422)
        if archive.exists():
            if operation_id:
                previous = verify_archive(archive, progress=progress)
                if (
                    previous.get("operationId") == operation_id
                    and previous["projectId"] == self.store.project_id
                ):
                    return {**previous, "archivePath": str(archive), "recovered": True}
            raise _error("The archive already exists; choose a new filename.")
        if not archive.parent.is_dir():
            raise _error("Choose an existing destination directory.", status=422)
        self.store.initialize()
        with lifecycle_guard(folder, timeout=5):
            self._idle()
            sources = source_inventory(self.store, self.filesystem)
            with (
                writer_lock(folder, timeout=5),
                tempfile.TemporaryDirectory(
                    prefix=".histopilot-export-", dir=archive.parent
                ) as temporary,
            ):
                temporary = Path(temporary)
                snapshot = temporary / DATABASE_FILE
                with self.store._connection() as source, sqlite3.connect(snapshot) as target:
                    source.backup(
                        target,
                        pages=1024,
                        progress=(
                            lambda _status, remaining, total: progress(
                                {
                                    "stage": "snapshot",
                                    "completed": total - remaining,
                                    "total": total,
                                    "file": DATABASE_FILE,
                                }
                            )
                        )
                        if progress
                        else None,
                    )
                    if target.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                        raise _error("The database snapshot failed its integrity check.")
                files = []
                for current, directories, names in os.walk(folder, followlinks=False):
                    for name in directories:
                        _reject_symlink_components(Path(current) / name)
                    for name in sorted(names):
                        path = Path(current) / name
                        relative = path.relative_to(folder).as_posix()
                        if name in TRANSIENT or relative in {
                            DATABASE_FILE + suffix for suffix in ("-wal", "-shm", "-journal")
                        }:
                            continue
                        ScientificStore._regular(path)
                        files.append((relative, snapshot if relative == DATABASE_FILE else path))
                        if len(files) > MAX_MEMBERS:
                            raise _error("This project exceeds the archive file limit.", status=413)
                staged = temporary / "archive.zip"
                entries = []
                with zipfile.ZipFile(
                    staged, "w", compression=zipfile.ZIP_STORED, allowZip64=True
                ) as zipped:
                    for index, (relative, path) in enumerate(sorted(files)):
                        if progress:
                            progress(
                                {
                                    "stage": "copying",
                                    "completed": index,
                                    "total": len(files),
                                    "file": relative,
                                }
                            )
                        before = path.stat()
                        digest = hashlib.sha256()
                        size = 0
                        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                        with (
                            os.fdopen(fd, "rb") as source,
                            zipped.open(f"project/{relative}", "w", force_zip64=True) as target,
                        ):
                            opened = os.fstat(source.fileno())
                            if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
                                raise _error("A file changed during export; export again.")
                            while chunk := source.read(4 * 1024 * 1024):
                                if progress:
                                    progress(
                                        {
                                            "stage": "copying",
                                            "completed": index,
                                            "total": len(files),
                                            "file": relative,
                                        }
                                    )
                                target.write(chunk)
                                digest.update(chunk)
                                size += len(chunk)
                            after = os.fstat(source.fileno())
                        if (before.st_size, before.st_mtime_ns) != (
                            after.st_size,
                            after.st_mtime_ns,
                        ):
                            raise _error("A file changed during export; export again.")
                        entries.append(
                            {"path": relative, "size": size, "sha256": digest.hexdigest()}
                        )
                    manifest = {
                        "format": "histopilot-study-archive",
                        "schemaVersion": 1,
                        "createdAt": _now(),
                        "projectId": self.store.project_id,
                        "originalPath": str(folder),
                        "files": entries,
                        "externalSources": sources,
                        "externalSourcePolicy": "references-only",
                        "operationId": operation_id,
                    }
                    content = json.dumps(manifest, allow_nan=False).encode()
                    if len(content) > MAX_MANIFEST_BYTES:
                        raise _error("The archive manifest exceeds its limit.", status=413)
                    zipped.writestr(MANIFEST, content)
                result = verify_archive(staged, progress=progress)
                with staged.open("rb") as handle:
                    os.fsync(handle.fileno())
                # Exclusive link prevents a concurrent export overwriting any destination.
                if progress:
                    progress(
                        {
                            "stage": "publishing",
                            "completed": len(files),
                            "total": len(files),
                            "file": str(archive),
                        }
                    )
                _reject_symlink_components(archive)
                try:
                    os.link(staged, archive)
                except FileExistsError:
                    raise _error("The destination was created by another operation.") from None
                staged.unlink()
                fsync_directory(archive.parent)
        return {**result, "archivePath": str(archive)}


def _archive_manifest(zipped):
    members = zipped.infolist()
    names = [item.filename for item in members]
    name_set = set(names)
    if len(names) > MAX_MEMBERS + 1 or len(names) != len(name_set):
        raise _error("The archive has duplicate or excessive entries.")
    if MANIFEST not in name_set:
        raise _error("This file is not a HistoPilot study archive.")
    descriptor_name = f"project/{DESCRIPTOR}"
    if descriptor_name in name_set and zipped.getinfo(descriptor_name).file_size > DESCRIPTOR_LIMIT:
        raise _error("The archive project descriptor exceeds its size limit.", status=413)
    info = zipped.getinfo(MANIFEST)
    if info.file_size > MAX_MANIFEST_BYTES:
        raise _error("The archive manifest exceeds its size limit.", status=413)
    manifest = json.loads(zipped.read(MANIFEST))
    if (
        not isinstance(manifest, dict)
        or manifest.get("format") != "histopilot-study-archive"
        or manifest.get("schemaVersion") != 1
        or not isinstance(manifest.get("files"), list)
        or not re.fullmatch(r"project-[a-f0-9]{32}", str(manifest.get("projectId", "")))
    ):
        raise _error("The archive manifest has an unsupported format.")
    expected = {MANIFEST}
    total = 0
    for entry in manifest["files"]:
        if not isinstance(entry, dict):
            raise _error("An archive entry is invalid.")
        value = entry.get("path")
        if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
            raise _error("An archive entry has an unsafe path.")
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or ".." in path.parts
            or str(path) != value
            or path.name in TRANSIENT
            or path.name.endswith(("-wal", "-shm", "-journal"))
        ):
            raise _error("An archive entry has an unsafe path.")
        name = f"project/{value}"
        if name in expected or name not in name_set:
            raise _error("An archive entry is duplicated or missing.")
        expected.add(name)
        info = zipped.getinfo(name)
        mode = info.external_attr >> 16
        if (
            info.is_dir()
            or stat.S_IFMT(mode) not in (0, stat.S_IFREG)
            or type(entry.get("size")) is not int
            or entry["size"] < 0
            or info.file_size != entry["size"]
            or not re.fullmatch(r"[a-f0-9]{64}", str(entry.get("sha256", "")))
        ):
            raise _error("An archive member has invalid type, size or checksum metadata.")
        total += entry["size"]
        if total > MAX_RESTORE_BYTES:
            raise _error("The archive exceeds the maximum expanded size.", status=413)
    if (
        expected != name_set
        or not {f"project/{DESCRIPTOR}", f"project/{DATABASE_FILE}"} <= expected
    ):
        raise _error("The archive contains unexpected files or lacks required project files.")
    return manifest


def verify_archive(path, *, destination=None, progress=None):
    """Validate every byte; optionally write into a caller-owned empty staging folder."""
    ScientificStore._regular(Path(path))
    try:
        with zipfile.ZipFile(path) as zipped:
            manifest = _archive_manifest(zipped)
            for index, entry in enumerate(manifest["files"]):
                if progress:
                    progress(
                        {
                            "stage": "restoring" if destination else "verifying",
                            "completed": index,
                            "total": len(manifest["files"]),
                            "file": entry["path"],
                        }
                    )
                digest = hashlib.sha256()
                size = 0
                target = None
                try:
                    if destination:
                        target_path = destination / entry["path"]
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        target = target_path.open("xb")
                    with zipped.open(f"project/{entry['path']}") as source:
                        while chunk := source.read(4 * 1024 * 1024):
                            if progress:
                                progress(
                                    {
                                        "stage": "restoring" if destination else "verifying",
                                        "completed": index,
                                        "total": len(manifest["files"]),
                                        "file": entry["path"],
                                    }
                                )
                            size += len(chunk)
                            if size > entry["size"]:
                                raise _error("An archive member exceeds its declared size.")
                            digest.update(chunk)
                            if target:
                                target.write(chunk)
                    if size != entry["size"] or digest.hexdigest() != entry["sha256"]:
                        raise _error(
                            f"Checksum verification failed for {entry['path']}.",
                            "PORTABILITY_CHECKSUM_MISMATCH",
                        )
                    if target:
                        target.flush()
                        os.fsync(target.fileno())
                finally:
                    if target:
                        target.close()
            descriptor = ProjectDocument.model_validate_json(zipped.read(f"project/{DESCRIPTOR}"))
            if descriptor.id != manifest["projectId"]:
                raise _error("The archive descriptor belongs to a different project.")
    except (
        zipfile.BadZipFile,
        json.JSONDecodeError,
        UnicodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        if isinstance(error, (StorageError, PortabilityCancelled)):
            raise
        raise _error("The archive is corrupt or has invalid metadata.") from error
    return {
        "verified": True,
        "projectId": manifest["projectId"],
        "fileCount": len(manifest["files"]),
        "totalBytes": sum(item["size"] for item in manifest["files"]),
        "originalPath": manifest.get("originalPath"),
        "externalSources": manifest.get("externalSources", {}),
        "externalSourcePolicy": "references-only",
        "operationId": manifest.get("operationId"),
        "manifestSha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
    }


def restore_archive(archive, destination, filesystem, *, progress=None, operation_id=None):
    archive = permitted_path(filesystem, archive, existing=True)
    destination = permitted_path(filesystem, destination)
    receipt_name = ".histopilot-restore.json"
    if destination.is_dir() and operation_id and (destination / receipt_name).exists():
        receipt = json.loads(ScientificStore._read_file(destination / receipt_name, 65536))
        if receipt.get("operationId") == operation_id:
            result = verify_archive(archive, progress=progress)
            if receipt.get("manifestSha256") == result["manifestSha256"]:
                return {
                    **result,
                    "destinationPath": str(destination),
                    "recovered": True,
                    "relocated": str(destination) != result.get("originalPath"),
                    "note": "This operation already published the restored folder. The archive was verified again; the restored project was not modified.",
                }
    if destination.exists() or not destination.parent.is_dir():
        raise _error("Restore into a new folder inside an existing storage directory.", status=422)
    staging = Path(tempfile.mkdtemp(prefix=".histopilot-restore-", dir=destination.parent))
    try:
        result = verify_archive(archive, destination=staging, progress=progress)
        # Validate the snapshot before publishing a project descriptor at its final path.
        database = staging / DATABASE_FILE
        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
            connection.execute("PRAGMA trusted_schema=OFF")
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise _error("The restored SQLite snapshot failed its integrity check.")
        ScientificStore(staging, result["projectId"]).initialize()
        if operation_id:
            from histopilot.workers.packing_process import write_json

            write_json(
                staging / receipt_name,
                {
                    "operationId": operation_id,
                    "manifestSha256": result["manifestSha256"],
                    "restoredAt": _now(),
                },
            )
        for current, _directories, _files in os.walk(staging, topdown=False):
            fsync_directory(Path(current))
        # A parent lock coordinates our restores. mkdir reserves the final target
        # without overwriting any unrelated existing folder, including an empty one.
        with writer_lock(destination.parent, timeout=5):
            if progress:
                progress(
                    {
                        "stage": "publishing",
                        "completed": result["fileCount"],
                        "total": result["fileCount"],
                        "file": str(destination),
                    }
                )
            _reject_symlink_components(destination)
            destination.mkdir(mode=0o700)
            try:
                os.rename(staging, destination)
            except BaseException:
                destination.rmdir()
                raise
            fsync_directory(destination.parent)
        return {
            **result,
            "destinationPath": str(destination),
            "relocated": str(destination) != result.get("originalPath"),
            "note": "Project identity and frozen scientific records were preserved. External sources are not included. Restoring at another path may require new validated imports and execution plans. Open the restored folder when this project ID is no longer registered at its original location.",
        }
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def relink_source(projects, identity, payload):
    """Compare-and-swap a source registration, never editing immutable versions."""
    replacement = projects.legacy.filesystem.directory(payload.replacementPath)
    _reject_symlink_components(replacement)
    with projects.lock:
        _, folder = projects._load(identity)
        with lifecycle_guard(folder), writer_lock(folder):
            LifecycleStore(folder, identity).assert_usable([f"project:{identity}"])
            document = projects._read(folder)
            source = next(
                (item for item in document["sources"] if item["id"] == payload.sourceId), None
            )
            if not source:
                raise _error("The source registration does not exist.", status=404)
            if source["path"] != payload.expectedPath:
                raise _error(
                    "The source changed in another tab. Refresh before relinking.",
                    "SOURCE_RELINK_CONFLICT",
                )
            source["path"] = str(replacement)
            source["name"] = replacement.name
            document["updatedAt"] = _now()
            projects._write(folder, document)
        summary = projects._summary(document, folder)
        projects._register(summary)
    return {
        "project": summary,
        "sourceId": payload.sourceId,
        "note": "Source registration updated. Existing frozen versions retain their recorded paths; import and validate a new version to use the moved source.",
    }
