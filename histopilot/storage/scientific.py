"""Folder-owned drafts and immutable, recoverable scientific publication.

This is a storage boundary, not an importer or a scientific validator. Callers
publish already-reviewed manifests; browser draft payloads confer no readiness.
"""

import hashlib
import json
import os
import re
import sqlite3
import stat
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from threading import RLock
from uuid import uuid4

from pydantic import ValidationError

from histopilot.schemas.version_labels import (
    FreezeVersionLabel,
    SetVersionLabelRequest,
    VersionLabelValues,
)
from histopilot.storage.project_lock import (
    StorageError,
    ensure_managed_directory,
    fsync_directory,
    writer_lock,
)

SCHEMA_VERSION = 4
DATASET_FORMAT_VERSION = 1
APPLICATION_ID = 0x48535054
DATABASE_FILE = "histopilot-state.sqlite"
MAX_DOCUMENT_BYTES = 1024 * 1024
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_PUBLICATION_BYTES = 256 * 1024 * 1024
MAX_ARTIFACTS = 128
MAX_CONFIGURATION_BYTES = 16 * 1024 * 1024
_DATASET_ID = re.compile(r"dataset-[a-f0-9]{64}\Z")
_SHA256 = re.compile(r"[a-f0-9]{64}\Z")
_STAGE_NAME = re.compile(r"operation-[a-f0-9]{32}\Z")
# Concurrent SQLite opening/closing can deadlock in supported native builds.
# Closing the final connection also unlinks WAL/SHM files: path validation must
# share this guard. Queries and transactions run outside the lifecycle lock.
_SQLITE_LIFECYCLE_LOCK = RLock()

_SCHEMA_V1 = (
    "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    """CREATE TABLE drafts (
        id TEXT PRIMARY KEY, kind TEXT NOT NULL CHECK(kind IN ('import','experiment')),
        name TEXT NOT NULL, payload TEXT NOT NULL, revision INTEGER NOT NULL CHECK(revision>0),
        status TEXT NOT NULL CHECK(status IN ('editable','frozen')),
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL
    )""",
    """CREATE TABLE datasets (
        id TEXT PRIMARY KEY, content_hash TEXT NOT NULL UNIQUE, document TEXT NOT NULL
    )""",
    """CREATE TABLE publications (
        id TEXT PRIMARY KEY, request_hash TEXT NOT NULL, draft_id TEXT NOT NULL,
        expected_revision INTEGER NOT NULL, dataset_id TEXT NOT NULL,
        stage_name TEXT NOT NULL UNIQUE, document TEXT NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('preparing','published','interrupted')),
        error TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
        FOREIGN KEY(draft_id) REFERENCES drafts(id)
    )""",
)
_SCHEMA_V2 = (
    *_SCHEMA_V1,
    """CREATE TABLE configurations (
        id TEXT PRIMARY KEY, content_hash TEXT NOT NULL UNIQUE, document TEXT NOT NULL
    )""",
    """CREATE TABLE configuration_publications (
        id TEXT PRIMARY KEY, request_hash TEXT NOT NULL, configuration_id TEXT NOT NULL,
        FOREIGN KEY(configuration_id) REFERENCES configurations(id)
    )""",
)
_SCHEMA_V3 = (
    *_SCHEMA_V2,
    """CREATE TABLE version_labels (
        resource_type TEXT NOT NULL CHECK(resource_type IN ('dataset','configuration')),
        resource_id TEXT NOT NULL, kind TEXT NOT NULL,
        tag TEXT NOT NULL, normalized_tag TEXT, note TEXT NOT NULL,
        revision INTEGER NOT NULL CHECK(revision>0),
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
        PRIMARY KEY(resource_type,resource_id), UNIQUE(kind,normalized_tag)
    )""",
)
_SCHEMA = (
    *_SCHEMA_V3,
    """CREATE TABLE publication_labels (
        operation_id TEXT PRIMARY KEY, tag TEXT NOT NULL, normalized_tag TEXT NOT NULL,
        note TEXT NOT NULL, FOREIGN KEY(operation_id) REFERENCES publications(id)
    )""",
)
_COLUMNS_V1 = {
    "metadata": ("key", "value"),
    "drafts": ("id", "kind", "name", "payload", "revision", "status", "created_at", "updated_at"),
    "datasets": ("id", "content_hash", "document"),
    "publications": (
        "id",
        "request_hash",
        "draft_id",
        "expected_revision",
        "dataset_id",
        "stage_name",
        "document",
        "status",
        "error",
        "created_at",
        "updated_at",
    ),
}
_COLUMNS_V2 = {
    **_COLUMNS_V1,
    "configurations": ("id", "content_hash", "document"),
    "configuration_publications": ("id", "request_hash", "configuration_id"),
}
_COLUMNS_V3 = {
    **_COLUMNS_V2,
    "version_labels": (
        "resource_type",
        "resource_id",
        "kind",
        "tag",
        "normalized_tag",
        "note",
        "revision",
        "created_at",
        "updated_at",
    ),
}
_COLUMNS = {
    **_COLUMNS_V3,
    "publication_labels": ("operation_id", "tag", "normalized_tag", "note"),
}


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _error(message: str, code: str = "STORAGE_INVALID", status: int = 409) -> StorageError:
    return StorageError(message, code, status)


def _json(value: dict, maximum: int = MAX_DOCUMENT_BYTES) -> bytes:
    def check(item: object) -> None:
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("JSON object keys must be strings")
            for child in item.values():
                check(child)
        elif isinstance(item, list):
            for child in item:
                check(child)
        elif item is not None and not isinstance(item, (str, int, float, bool)):
            raise ValueError("Only JSON values are supported")

    try:
        if not isinstance(value, dict):
            raise ValueError("Expected an object")
        check(value)
        encoded = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (ValueError, TypeError, RecursionError, UnicodeError) as error:
        raise _error(
            "Supply a JSON object containing finite JSON values.", "INVALID_DOCUMENT", 422
        ) from error
    if len(encoded) > maximum:
        raise _error("The metadata document exceeds the supported size.", "DOCUMENT_TOO_LARGE", 413)
    return encoded


def _load_json(value: str | bytes) -> dict:
    try:
        document = json.loads(value)
        _json(document)
        return document
    except (ValueError, TypeError, RecursionError) as error:
        raise _error("Stored metadata is invalid.", "STORAGE_CORRUPT") from error


def _label(value: str, label: str, maximum: int = 200) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum or "\x00" in value:
        raise _error(
            f"Supply a nonempty {label} of at most {maximum} characters.", "INVALID_INPUT", 422
        )
    return value


def _revision(value: int) -> None:
    if type(value) is not int or not 1 <= value < 2**63 - 1:
        raise _error("expected_revision must be a positive integer.", "INVALID_REVISION", 422)


def _artifact_name(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 1024
        or any(c in value for c in "\\\x00:")
    ):
        raise _error("Artifact names must be safe relative paths.", "ARTIFACT_PATH_INVALID", 422)
    path = PurePosixPath(value)
    if (
        not path.parts
        or path.is_absolute()
        or path.as_posix() != value
        or any(
            part in {".", ".."} or part.startswith(".") or len(part.encode("utf-8")) > 255
            for part in path.parts
        )
    ):
        raise _error("Artifact names must be safe relative paths.", "ARTIFACT_PATH_INVALID", 422)
    if path.parts[0] == "manifest.json":
        raise _error(
            "manifest.json is reserved for the published dataset.", "ARTIFACT_PATH_INVALID", 422
        )
    return value


def _artifact_metadata(artifacts: dict[str, bytes]) -> dict:
    if not isinstance(artifacts, dict) or len(artifacts) > MAX_ARTIFACTS:
        raise _error("Too many artifacts in one publication.", "ARTIFACT_LIMIT", 413)
    total = 0
    for name, content in artifacts.items():
        _artifact_name(name)
        if not isinstance(content, bytes):
            raise _error("Materialized artifacts must be bytes.", "INVALID_ARTIFACT", 422)
        if len(content) > MAX_ARTIFACT_BYTES:
            raise _error("An artifact exceeds the supported size.", "ARTIFACT_LIMIT", 413)
        total += len(content)
        if total > MAX_PUBLICATION_BYTES:
            raise _error("The publication exceeds the supported size.", "ARTIFACT_LIMIT", 413)
        if any(
            parent.as_posix() in artifacts
            for parent in PurePosixPath(name).parents
            if parent.as_posix() != "."
        ):
            raise _error(
                "An artifact file cannot also be an artifact directory.",
                "ARTIFACT_PATH_INVALID",
                422,
            )
    return {
        name: {"sha256": hashlib.sha256(content).hexdigest(), "sizeBytes": len(content)}
        for name, content in sorted(artifacts.items())
    }


def _hash_content(manifest: dict, artifacts: dict) -> str:
    # Generated timestamps, project IDs and storage locations are envelope fields,
    # deliberately excluded. Every caller-supplied scientific manifest value is hashed.
    return hashlib.sha256(_json({"manifest": manifest, "artifacts": artifacts})).hexdigest()


class ScientificStore:
    def __init__(self, folder: Path, project_id: str):
        self.folder = Path(folder).absolute()
        self.project_id = _label(project_id, "project ID", 128)
        self.path = self.folder / DATABASE_FILE

    def _checkpoint(self, phase: str) -> None:
        """Failure-injection seam; production publication has no callback actions."""

    @staticmethod
    def _regular(path: Path, *, missing_ok: bool = False) -> os.stat_result | None:
        if any(parent.is_symlink() for parent in path.parents):
            raise _error(
                "Managed storage cannot traverse symbolic links.", "STORAGE_UNSAFE_PATH", 403
            )
        try:
            info = path.lstat()
        except FileNotFoundError:
            if missing_ok:
                return None
            raise _error("A required storage file is missing.", "STORAGE_CORRUPT") from None
        except OSError as error:
            raise _error(
                "A managed storage path cannot be inspected.", "STORAGE_READ_FAILED", 403
            ) from error
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise _error(
                "Managed files must be regular files without aliases.", "STORAGE_UNSAFE_PATH", 403
            )
        return info

    def _database_paths(self) -> None:
        with _SQLITE_LIFECYCLE_LOCK:
            for suffix in ("", "-wal", "-shm", "-journal"):
                self._regular(Path(str(self.path) + suffix), missing_ok=True)

    @staticmethod
    def _read_file(path: Path, maximum: int) -> bytes:
        ScientificStore._regular(path)
        descriptor = None
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise _error(
                    "Managed files must be regular files without aliases.",
                    "STORAGE_UNSAFE_PATH",
                    403,
                )
            if info.st_size > maximum:
                raise _error("A stored artifact exceeds its declared bounds.", "STORAGE_CORRUPT")
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                content = handle.read(maximum + 1)
            if len(content) > maximum:
                raise _error("A stored artifact exceeds its declared bounds.", "STORAGE_CORRUPT")
            return content
        except OSError as error:
            raise _error("A stored artifact cannot be read.", "STORAGE_READ_FAILED", 403) from error
        finally:
            if descriptor is not None:
                os.close(descriptor)

    @contextmanager
    def _connection(self, *, create: bool = False) -> Iterator[sqlite3.Connection]:
        connection = None
        try:
            uri = self.path.as_uri() + ("?mode=rwc" if create else "?mode=rw")
            with _SQLITE_LIFECYCLE_LOCK:
                self._database_paths()
                connection = sqlite3.connect(uri, uri=True, timeout=2, isolation_level=None)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA trusted_schema=OFF")
            connection.execute("PRAGMA synchronous=FULL")
            yield connection
        except sqlite3.Error as error:
            raise _error(
                "The scientific database is unavailable or invalid.", "STORAGE_DATABASE_INVALID"
            ) from error
        finally:
            if connection is not None:
                with _SQLITE_LIFECYCLE_LOCK:
                    connection.close()

    @staticmethod
    @contextmanager
    def _transaction(connection: sqlite3.Connection) -> Iterator[None]:
        connection.execute("BEGIN IMMEDIATE")
        try:
            yield
            connection.execute("COMMIT")
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise

    def initialize(self) -> None:
        # Independent browser queries initialize/recover the same folder. Let
        # these short checks serialize instead of surfacing spurious busy errors.
        with writer_lock(self.folder, timeout=5):
            self._initialize_locked()

    def _initialize_locked(self) -> None:
        self._database_paths()
        info = self._regular(self.path, missing_ok=True)
        existing_state = any(
            path.exists() or path.is_symlink()
            for path in (
                self.folder / "datasets",
                self.folder / ".staging",
                *(Path(str(self.path) + suffix) for suffix in ("-wal", "-shm", "-journal")),
            )
        )
        if (info is None or info.st_size == 0) and existing_state:
            raise _error(
                "Scientific files exist but their database is missing or empty. Restore the project database.",
                "STORAGE_DATABASE_MISSING",
            )
        if info is not None and info.st_size:
            descriptor = None
            try:
                descriptor = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW)
                if os.read(descriptor, 16) != b"SQLite format 3\x00":
                    raise _error(
                        "The scientific database has an unknown file format.",
                        "STORAGE_DATABASE_INVALID",
                    )
            except OSError as error:
                raise _error(
                    "The scientific database cannot be read.", "STORAGE_READ_FAILED", 403
                ) from error
            finally:
                if descriptor is not None:
                    os.close(descriptor)
        with self._connection(create=True) as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            app_id = connection.execute("PRAGMA application_id").fetchone()[0]
            objects = list(
                connection.execute(
                    "SELECT type, name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
                )
            )
            if version == 0 and not objects and app_id == 0:
                if existing_state:
                    raise _error(
                        "An empty database cannot replace existing scientific state.",
                        "STORAGE_DATABASE_MISSING",
                    )
                mode = connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
                if mode.lower() != "wal":
                    raise _error(
                        "The experiment filesystem does not support SQLite WAL.",
                        "STORAGE_UNSUPPORTED",
                    )
                with self._transaction(connection):
                    for statement in _SCHEMA:
                        connection.execute(statement)
                    connection.executemany(
                        "INSERT INTO metadata VALUES (?,?)",
                        (
                            ("project_id", self.project_id),
                            ("schema_version", str(SCHEMA_VERSION)),
                        ),
                    )
                    connection.execute(f"PRAGMA application_id={APPLICATION_ID}")
                    connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
                fsync_directory(self.folder)
            elif version in {1, 2, 3} and app_id == APPLICATION_ID:
                # Validate the complete old schema before making an atomic additive migration.
                self._validate_schema(connection, version=version)
                self._validate_integrity(connection)
                with self._transaction(connection):
                    previous_schema = {1: _SCHEMA_V1, 2: _SCHEMA_V2, 3: _SCHEMA_V3}[version]
                    for statement in _SCHEMA[len(previous_schema) :]:
                        connection.execute(statement)
                    connection.execute(
                        "UPDATE metadata SET value=? WHERE key='schema_version'",
                        (str(SCHEMA_VERSION),),
                    )
                    connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            elif version != SCHEMA_VERSION or app_id != APPLICATION_ID:
                raise _error(
                    "The scientific database version or format is unsupported.",
                    "STORAGE_SCHEMA_UNSUPPORTED",
                )
            self._validate_schema(connection)
            self._validate_integrity(connection)
            mode = connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            if mode.lower() != "wal":
                raise _error(
                    "The experiment filesystem does not support SQLite WAL.", "STORAGE_UNSUPPORTED"
                )
        ensure_managed_directory(self.folder / "datasets")
        ensure_managed_directory(self.folder / ".staging")
        self._recover_locked()

    @staticmethod
    def _validate_integrity(connection: sqlite3.Connection) -> None:
        if connection.execute("PRAGMA quick_check").fetchall()[0][0] != "ok":
            raise _error("The scientific database failed its integrity check.", "STORAGE_CORRUPT")
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise _error("The scientific database contains invalid references.", "STORAGE_CORRUPT")

    def _validate_schema(
        self, connection: sqlite3.Connection, version: int = SCHEMA_VERSION
    ) -> None:
        columns_by_name = {1: _COLUMNS_V1, 2: _COLUMNS_V2, 3: _COLUMNS_V3, 4: _COLUMNS}[version]
        schema = {1: _SCHEMA_V1, 2: _SCHEMA_V2, 3: _SCHEMA_V3, 4: _SCHEMA}[version]
        objects = list(
            connection.execute(
                "SELECT type, name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
            )
        )
        if {(row["type"], row["name"]) for row in objects} != {
            ("table", name) for name in columns_by_name
        }:
            raise _error("The scientific database schema is unknown.", "STORAGE_SCHEMA_UNSUPPORTED")
        expected_ddl = {" ".join(statement.split()) for statement in schema}
        if {" ".join(row["sql"].split()) for row in objects} != expected_ddl:
            raise _error(
                "The scientific database constraints are unknown.", "STORAGE_SCHEMA_UNSUPPORTED"
            )
        for name, columns in columns_by_name.items():
            found = tuple(row["name"] for row in connection.execute(f"PRAGMA table_info({name})"))
            if found != columns:
                raise _error(
                    "The scientific database schema is unknown.", "STORAGE_SCHEMA_UNSUPPORTED"
                )
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        if metadata.get("project_id") != self.project_id:
            raise _error(
                "This scientific database belongs to a different experiment.", "PROJECT_ID_MISMATCH"
            )
        if metadata != {"project_id": self.project_id, "schema_version": str(version)}:
            raise _error(
                "The scientific database metadata version is unsupported.",
                "STORAGE_SCHEMA_UNSUPPORTED",
            )

    def status(self) -> dict:
        self.initialize()
        with self._connection() as connection:
            operations = []
            for row in connection.execute("SELECT * FROM publications ORDER BY created_at,id"):
                operation = {
                    "id": row["id"],
                    "draftId": row["draft_id"],
                    "status": row["status"],
                    "datasetId": row["dataset_id"],
                }
                if row["error"]:
                    operation["error"] = row["error"]
                operations.append(operation)
            return {
                "projectId": self.project_id,
                "schemaVersion": SCHEMA_VERSION,
                "database": DATABASE_FILE,
                "journalMode": connection.execute("PRAGMA journal_mode").fetchone()[0],
                "draftCount": connection.execute("SELECT COUNT(*) FROM drafts").fetchone()[0],
                "datasetCount": connection.execute("SELECT COUNT(*) FROM datasets").fetchone()[0],
                "configurationCount": connection.execute(
                    "SELECT COUNT(*) FROM configurations"
                ).fetchone()[0],
                "operations": operations,
            }

    def _draft(self, row: sqlite3.Row | None) -> dict:
        if row is None:
            raise _error("The scientific draft does not exist.", "DRAFT_NOT_FOUND", 404)
        return {
            "id": row["id"],
            "projectId": self.project_id,
            "kind": row["kind"],
            "name": row["name"],
            "payload": _load_json(row["payload"]),
            "revision": row["revision"],
            "status": row["status"],
            "createdAt": row["created_at"],
            "updatedAt": row["updated_at"],
        }

    def create_draft(self, kind: str, name: str, payload: dict) -> dict:
        if not isinstance(kind, str) or kind not in {"import", "experiment"}:
            raise _error("Draft kind must be import or experiment.", "INVALID_DRAFT_KIND", 422)
        _label(name, "draft name")
        content = _json(payload).decode("utf-8")
        identity, now = f"draft-{uuid4().hex}", _now()
        with writer_lock(self.folder):
            self._initialize_locked()
            with self._connection() as connection, self._transaction(connection):
                connection.execute(
                    "INSERT INTO drafts VALUES (?,?,?,?,?,?,?,?)",
                    (
                        identity,
                        kind,
                        name,
                        content,
                        1,
                        "editable",
                        now,
                        now,
                    ),
                )
                return self._draft(
                    connection.execute("SELECT * FROM drafts WHERE id=?", (identity,)).fetchone()
                )

    def get_draft(self, identity: str) -> dict:
        self.initialize()
        with self._connection() as connection:
            return self._draft(
                connection.execute("SELECT * FROM drafts WHERE id=?", (identity,)).fetchone()
            )

    def list_drafts(self) -> list[dict]:
        self.initialize()
        with self._connection() as connection:
            return [
                self._draft(row)
                for row in connection.execute("SELECT * FROM drafts ORDER BY created_at,id")
            ]

    def update_draft(
        self, identity: str, *, expected_revision: int, name: str, payload: dict
    ) -> dict:
        _revision(expected_revision)
        _label(name, "draft name")
        content = _json(payload).decode("utf-8")
        with writer_lock(self.folder):
            self._initialize_locked()
            with self._connection() as connection, self._transaction(connection):
                draft = self._draft(
                    connection.execute("SELECT * FROM drafts WHERE id=?", (identity,)).fetchone()
                )
                self._editable(draft, expected_revision)
                connection.execute(
                    "UPDATE drafts SET name=?,payload=?,revision=revision+1,updated_at=? WHERE id=? AND revision=?",
                    (
                        name,
                        content,
                        _now(),
                        identity,
                        expected_revision,
                    ),
                )
                return self._draft(
                    connection.execute("SELECT * FROM drafts WHERE id=?", (identity,)).fetchone()
                )

    @staticmethod
    def _editable(draft: dict, revision: int) -> None:
        if draft["revision"] != revision:
            raise _error(
                "The draft changed. Reload before saving or publishing.", "REVISION_CONFLICT"
            )
        if draft["status"] != "editable":
            raise _error("A frozen draft cannot be modified.", "DRAFT_FROZEN")

    def _version_target(
        self, connection: sqlite3.Connection, resource_type: str, resource_id: str
    ) -> tuple[str, dict]:
        if resource_type == "dataset":
            return "dataset", self._get_dataset(connection, resource_id)
        if resource_type == "configuration":
            document = self._configuration(connection, resource_id)
            kind = document["manifest"].get("kind")
            if not isinstance(kind, str) or not kind:
                raise _error("The configuration kind is invalid.", "STORAGE_CORRUPT")
            return kind, document
        raise _error(
            "Labels belong to a frozen dataset or configuration.", "INVALID_RESOURCE_TYPE", 422
        )

    @staticmethod
    def _version_label_record(row: sqlite3.Row | None, kind: str) -> dict | None:
        if row is None:
            return None
        try:
            values = VersionLabelValues.model_validate({"tag": row["tag"], "note": row["note"]})
            if (
                row["kind"] != kind
                or values.tag != row["tag"]
                or values.note != row["note"]
                or row["normalized_tag"] != (values.tag.casefold() or None)
                or type(row["revision"]) is not int
                or not 1 <= row["revision"] <= 2**63 - 1
            ):
                raise ValueError
            for key in ("created_at", "updated_at"):
                if datetime.fromisoformat(row[key].replace("Z", "+00:00")).tzinfo is None:
                    raise ValueError
            return {
                "tag": row["tag"],
                "note": row["note"],
                "revision": row["revision"],
                "createdAt": row["created_at"],
                "updatedAt": row["updated_at"],
            }
        except (ValueError, TypeError, AttributeError, KeyError) as error:
            raise _error("Version label metadata is invalid.", "STORAGE_CORRUPT") from error

    def _version_label(
        self, connection: sqlite3.Connection, resource_type: str, resource_id: str, kind: str
    ) -> dict | None:
        row = connection.execute(
            "SELECT * FROM version_labels WHERE resource_type=? AND resource_id=?",
            (resource_type, resource_id),
        ).fetchone()
        return self._version_label_record(row, kind)

    def _with_version_label(
        self, connection: sqlite3.Connection, resource_type: str, document: dict
    ) -> dict:
        kind = "dataset" if resource_type == "dataset" else document["manifest"].get("kind")
        label = self._version_label(connection, resource_type, document["id"], kind)
        return {**document, "versionLabel": label} if label is not None else document

    def get_version_label(self, resource_type: str, resource_id: str) -> dict | None:
        self.initialize()
        with self._connection() as connection:
            kind, _document = self._version_target(connection, resource_type, resource_id)
            return self._version_label(connection, resource_type, resource_id, kind)

    @staticmethod
    def _publication_label_values(value: dict | None) -> VersionLabelValues | None:
        if value is None:
            return None
        try:
            return FreezeVersionLabel.model_validate(value)
        except ValidationError as error:
            raise _error(
                "Supply a nonempty version tag of at most 80 characters and a commit "
                "note of at most 2000 characters.",
                "INVALID_VERSION_LABEL",
                422,
            ) from error

    def _check_version_tag(
        self,
        connection: sqlite3.Connection,
        resource_type: str,
        resource_id: str,
        kind: str,
        tag: str,
    ) -> None:
        normalized = tag.casefold() or None
        if not normalized:
            return
        used = connection.execute(
            "SELECT 1 FROM version_labels WHERE kind=? AND normalized_tag=? "
            "AND NOT (resource_type=? AND resource_id=?)",
            (kind, normalized, resource_type, resource_id),
        ).fetchone()
        # A recoverable dataset publication owns its tag before the dataset becomes
        # visible. Presentation edits and other publishers must respect that claim.
        reserved = (
            kind == "dataset"
            and connection.execute(
                "SELECT 1 FROM publication_labels labels JOIN publications publication "
                "ON publication.id=labels.operation_id WHERE publication.status='preparing' "
                "AND labels.normalized_tag=? AND publication.dataset_id!=?",
                (normalized, resource_id if resource_type == "dataset" else ""),
            ).fetchone()
        )
        if used or reserved:
            raise _error(
                f"This tag is already used or reserved for another {kind} version in this "
                "project. Choose a different tag.",
                "VERSION_TAG_CONFLICT",
            )

    def _check_publication_label(
        self,
        connection: sqlite3.Connection,
        resource_type: str,
        resource_id: str,
        kind: str,
        values: VersionLabelValues | None,
    ) -> dict | None:
        current = self._version_label(connection, resource_type, resource_id, kind)
        if values is None:
            return current
        if current and (current["tag"] or current["note"]):
            if (current["tag"], current["note"]) != (values.tag, values.note):
                raise _error(
                    f"These scientific contents are already saved as {resource_id} with "
                    f"the tag {current['tag']!r}. Reuse that version, or edit its existing "
                    "tag and commit note before freezing identical contents.",
                    "VERSION_LABEL_MISMATCH",
                )
        self._check_version_tag(connection, resource_type, resource_id, kind, values.tag)
        return current

    @staticmethod
    def _write_version_label(
        connection: sqlite3.Connection,
        resource_type: str,
        resource_id: str,
        kind: str,
        values: VersionLabelValues,
        current: dict | None,
    ) -> None:
        revision = current["revision"] if current else 0
        if revision >= 2**63 - 1:
            raise _error("The version label revision limit was reached.", "REVISION_LIMIT")
        now = _now()
        connection.execute(
            "INSERT INTO version_labels VALUES (?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(resource_type,resource_id) DO UPDATE SET "
            "tag=excluded.tag,normalized_tag=excluded.normalized_tag,note=excluded.note,"
            "revision=excluded.revision,updated_at=excluded.updated_at",
            (
                resource_type,
                resource_id,
                kind,
                values.tag,
                values.tag.casefold() or None,
                values.note,
                revision + 1,
                current["createdAt"] if current else now,
                now,
            ),
        )

    def _apply_publication_label(
        self,
        connection: sqlite3.Connection,
        resource_type: str,
        resource_id: str,
        kind: str,
        values: VersionLabelValues | None,
    ) -> None:
        current = self._check_publication_label(
            connection, resource_type, resource_id, kind, values
        )
        if values is not None and not (current and (current["tag"] or current["note"])):
            self._write_version_label(connection, resource_type, resource_id, kind, values, current)
            self._checkpoint("publication_label_written")

    def _journal_label(
        self, connection: sqlite3.Connection, operation_id: str
    ) -> VersionLabelValues | None:
        row = connection.execute(
            "SELECT * FROM publication_labels WHERE operation_id=?", (operation_id,)
        ).fetchone()
        if row is None:
            return None
        try:
            values = self._publication_label_values({"tag": row["tag"], "note": row["note"]})
            if (
                row["tag"] != values.tag
                or row["note"] != values.note
                or row["normalized_tag"] != values.tag.casefold()
            ):
                raise ValueError
            return values
        except (StorageError, ValueError) as error:
            raise _error("The publication label journal is invalid.", "STORAGE_CORRUPT") from error

    def set_version_label(
        self,
        resource_type: str,
        resource_id: str,
        *,
        tag: str,
        note: str = "",
        expected_revision: int,
    ) -> dict:
        """Edit only presentation metadata; version IDs and scientific inputs remain immutable.

        Revision zero creates the first label. Empty tags release their uniqueness claim,
        retaining a revision tombstone so another editor cannot overwrite a cleared label.
        """
        try:
            values = SetVersionLabelRequest.model_validate(
                {
                    "tag": tag,
                    "note": note,
                    "expectedRevision": expected_revision,
                }
            )
        except ValidationError as error:
            raise _error(
                "Supply a version tag of at most 80 characters, a note of at most "
                "2000 characters, and the current label revision.",
                "INVALID_VERSION_LABEL",
                422,
            ) from error
        with writer_lock(self.folder):
            self._initialize_locked()
            with self._connection() as connection, self._transaction(connection):
                kind, _document = self._version_target(connection, resource_type, resource_id)
                current = self._version_label(connection, resource_type, resource_id, kind)
                revision = current["revision"] if current else 0
                if revision != values.expectedRevision:
                    raise _error(
                        "The version label changed. Reload it before saving your edits.",
                        "REVISION_CONFLICT",
                    )
                self._check_version_tag(connection, resource_type, resource_id, kind, values.tag)
                self._write_version_label(
                    connection, resource_type, resource_id, kind, values, current
                )
                self._checkpoint("version_label_before_commit")
                result = self._version_label(connection, resource_type, resource_id, kind)
            self._checkpoint("version_label_committed")
            return result

    def _configuration(self, connection: sqlite3.Connection, identity: str) -> dict:
        row = connection.execute("SELECT * FROM configurations WHERE id=?", (identity,)).fetchone()
        if row is None:
            raise _error("The frozen configuration does not exist.", "CONFIGURATION_NOT_FOUND", 404)
        try:
            document = json.loads(row["document"])
            _json(document, MAX_CONFIGURATION_BYTES)
            digest = hashlib.sha256(
                _json(document["manifest"], MAX_CONFIGURATION_BYTES)
            ).hexdigest()
            if (
                document["projectId"] != self.project_id
                or document["contentHash"] != digest
                or row["content_hash"] != digest
                or document["id"] != identity
                or identity != f"configuration-{digest}"
            ):
                raise ValueError
            return document
        except (ValueError, TypeError, KeyError) as error:
            raise _error(
                "The frozen configuration failed its checksum.", "STORAGE_CORRUPT"
            ) from error

    def get_configuration(self, identity: str) -> dict:
        self.initialize()
        with self._connection() as connection:
            return self._with_version_label(
                connection, "configuration", self._configuration(connection, identity)
            )

    def list_configurations(self, kind: str | None = None) -> list[dict]:
        self.initialize()
        with self._connection() as connection:
            documents = [
                self._with_version_label(
                    connection, "configuration", self._configuration(connection, row[0])
                )
                for row in connection.execute(
                    "SELECT id FROM configurations ORDER BY id"
                ).fetchall()
            ]
        return [item for item in documents if kind is None or item["manifest"].get("kind") == kind]

    def configuration_publication(self, operation_id: str) -> dict | None:
        """Return a completed publication for caller-side immutable intent replay."""
        _label(operation_id, "operation ID")
        self.initialize()
        with self._connection() as connection:
            operation = connection.execute(
                "SELECT configuration_id FROM configuration_publications WHERE id=?",
                (operation_id,),
            ).fetchone()
            return (
                self._with_version_label(
                    connection, "configuration", self._configuration(connection, operation[0])
                )
                if operation
                else None
            )

    def publish_configuration(
        self,
        draft_id: str | None = None,
        *,
        expected_revision: int | None = None,
        manifest: dict,
        operation_id: str,
        version_label: dict | None = None,
        before_publish: Callable[[], None] | None = None,
    ) -> dict:
        """Atomically pin a server-validated scientific binding and freeze its draft.

        Memberships and feature headers are bounded JSON in SQLite, so one FULL-sync
        transaction commits the complete snapshot, retry receipt and draft transition.
        Large embedding arrays remain in the read-only source directory.
        """
        _label(operation_id, "operation ID")
        label_values = self._publication_label_values(version_label)
        if draft_id is not None:
            _revision(expected_revision)
        if manifest.get("kind") not in {"protocol", "feature", "feature-bundle"}:
            raise _error("Unsupported scientific configuration kind.", "INVALID_CONFIGURATION", 422)
        content = _json(manifest, MAX_CONFIGURATION_BYTES)
        digest = hashlib.sha256(content).hexdigest()
        identity = f"configuration-{digest}"
        request_hash = hashlib.sha256(
            _json(
                {
                    "draftId": draft_id,
                    "revision": expected_revision,
                    "contentHash": digest,
                    **({"versionLabel": label_values.model_dump()} if label_values else {}),
                }
            )
        ).hexdigest()
        with writer_lock(self.folder):
            self._initialize_locked()
            with self._connection() as connection, self._transaction(connection):
                prior = connection.execute(
                    "SELECT * FROM configuration_publications WHERE id=?", (operation_id,)
                ).fetchone()
                if prior is not None:
                    if prior["request_hash"] != request_hash:
                        raise _error(
                            "This operation ID belongs to a different request.",
                            "OPERATION_CONFLICT",
                        )
                    return self._with_version_label(
                        connection,
                        "configuration",
                        self._configuration(connection, prior["configuration_id"]),
                    )
                if before_publish is not None:
                    before_publish()
                self._get_dataset(connection, manifest.get("datasetId", ""))
                if draft_id is not None:
                    draft = self._draft(
                        connection.execute(
                            "SELECT * FROM drafts WHERE id=?", (draft_id,)
                        ).fetchone()
                    )
                    self._editable(draft, expected_revision)
                    if draft["kind"] != "experiment":
                        raise _error(
                            "A configuration requires an experiment draft.",
                            "INVALID_DRAFT_KIND",
                            422,
                        )
                self._check_publication_label(
                    connection, "configuration", identity, manifest["kind"], label_values
                )
                document = {
                    "id": identity,
                    "projectId": self.project_id,
                    "contentHash": digest,
                    "manifest": manifest,
                    "createdAt": _now(),
                }
                connection.execute(
                    "INSERT INTO configurations VALUES (?,?,?) ON CONFLICT(id) DO NOTHING",
                    (identity, digest, _json(document, MAX_CONFIGURATION_BYTES).decode()),
                )
                self._apply_publication_label(
                    connection, "configuration", identity, manifest["kind"], label_values
                )
                connection.execute(
                    "INSERT INTO configuration_publications VALUES (?,?,?)",
                    (operation_id, request_hash, identity),
                )
                if draft_id is not None:
                    connection.execute(
                        "UPDATE drafts SET status='frozen',revision=revision+1,updated_at=? WHERE id=?",
                        (_now(), draft_id),
                    )
                self._checkpoint("configuration_before_commit")
                result = self._with_version_label(
                    connection, "configuration", self._configuration(connection, identity)
                )
            self._checkpoint("configuration_committed")
            return result

    def _dataset_document(self, value: str | bytes | dict) -> dict:
        document = _load_json(value) if isinstance(value, (str, bytes)) else value
        try:
            if not isinstance(document, dict) or set(document) - {
                "id",
                "projectId",
                "contentHash",
                "manifest",
                "artifacts",
                "createdAt",
                "parentId",
            }:
                raise ValueError
            if document["projectId"] != self.project_id or not isinstance(
                document["createdAt"], str
            ):
                raise ValueError
            if not isinstance(document["manifest"], dict):
                raise ValueError
            artifacts = document["artifacts"]
            if not isinstance(artifacts, dict) or len(artifacts) > MAX_ARTIFACTS:
                raise ValueError
            total = 0
            for name, item in artifacts.items():
                _artifact_name(name)
                if (
                    not isinstance(item, dict)
                    or set(item) != {"sha256", "sizeBytes"}
                    or not isinstance(item["sha256"], str)
                    or not _SHA256.fullmatch(item["sha256"])
                    or type(item["sizeBytes"]) is not int
                    or not 0 <= item["sizeBytes"] <= MAX_ARTIFACT_BYTES
                ):
                    raise ValueError
                total += item["sizeBytes"]
            if total > MAX_PUBLICATION_BYTES:
                raise ValueError
            content_hash = _hash_content(document["manifest"], artifacts)
            if (
                document["contentHash"] != content_hash
                or document["id"] != f"dataset-{content_hash}"
            ):
                raise ValueError
            parent = document["manifest"].get("parentId")
            if parent is not None and (
                not isinstance(parent, str) or not _DATASET_ID.fullmatch(parent)
            ):
                raise ValueError
            if (parent is None and "parentId" in document) or (
                parent is not None and document.get("parentId") != parent
            ):
                raise ValueError
            return document
        except (KeyError, TypeError, ValueError) as error:
            raise _error(
                "The published dataset metadata is inconsistent.", "STORAGE_CORRUPT"
            ) from error

    @staticmethod
    def _disk_manifest(document: dict) -> bytes:
        return (
            _json(
                {
                    "format": "histopilot-dataset",
                    "schemaVersion": DATASET_FORMAT_VERSION,
                    **document,
                }
            )
            + b"\n"
        )

    def _verify_directory(self, folder: Path, document: dict, *, checksums: bool = True) -> None:
        if folder.is_symlink() or not folder.is_dir():
            raise _error("The published dataset directory is missing or unsafe.", "STORAGE_CORRUPT")
        expected = set(document["artifacts"]) | {"manifest.json"}
        expected_directories = {
            parent.as_posix()
            for name in expected
            for parent in PurePosixPath(name).parents
            if parent.as_posix() != "."
        }
        found = set()
        pending = [folder]
        try:
            while pending:
                current = pending.pop()
                for entry in current.iterdir():
                    relative = entry.relative_to(folder).as_posix()
                    info = entry.lstat()
                    if stat.S_ISDIR(info.st_mode):
                        if relative not in expected_directories:
                            raise _error(
                                "The dataset contains an undeclared directory.", "STORAGE_CORRUPT"
                            )
                        pending.append(entry)
                    else:
                        self._regular(entry)
                        if relative not in expected:
                            raise _error(
                                "The dataset contains an undeclared artifact.", "STORAGE_CORRUPT"
                            )
                        found.add(relative)
            if found != expected:
                raise _error("The dataset has missing or undeclared artifacts.", "STORAGE_CORRUPT")
            disk = self._read_file(folder / "manifest.json", MAX_DOCUMENT_BYTES + 1)
            if disk != self._disk_manifest(document):
                raise _error("The dataset manifest changed.", "STORAGE_CORRUPT")
            for name, item in document["artifacts"].items():
                if checksums:
                    content = self._read_file(folder / name, item["sizeBytes"])
                    if (
                        len(content) != item["sizeBytes"]
                        or hashlib.sha256(content).hexdigest() != item["sha256"]
                    ):
                        raise _error(
                            "A dataset artifact failed checksum validation.", "ARTIFACT_CORRUPT"
                        )
                elif self._regular(folder / name).st_size != item["sizeBytes"]:
                    raise _error("A dataset artifact changed size.", "ARTIFACT_CORRUPT")
        except OSError as error:
            raise _error(
                "The published dataset cannot be inspected.", "STORAGE_READ_FAILED", 403
            ) from error

    def _get_dataset(
        self, connection: sqlite3.Connection, identity: str, *, checksums: bool = True
    ) -> dict:
        row = connection.execute("SELECT document FROM datasets WHERE id=?", (identity,)).fetchone()
        if row is None:
            raise _error("The frozen dataset does not exist.", "DATASET_NOT_FOUND", 404)
        document = self._dataset_document(row["document"])
        if document["id"] != identity:
            raise _error("The dataset index is inconsistent.", "STORAGE_CORRUPT")
        self._verify_directory(self.folder / "datasets" / identity, document, checksums=checksums)
        return document

    def get_dataset(self, identity: str) -> dict:
        self.initialize()
        with self._connection() as connection:
            return self._with_version_label(
                connection, "dataset", self._get_dataset(connection, identity)
            )

    def list_datasets(self) -> list[dict]:
        self.initialize()
        with self._connection() as connection:
            return [
                self._with_version_label(
                    connection, "dataset", self._get_dataset(connection, row["id"])
                )
                for row in connection.execute("SELECT id FROM datasets ORDER BY id").fetchall()
            ]

    def read_artifact(self, dataset_id: str, name: str) -> bytes:
        _artifact_name(name)
        self.initialize()
        with self._connection() as connection:
            document = self._get_dataset(connection, dataset_id)
        if name not in document["artifacts"]:
            raise _error(
                "The requested artifact is not declared in this dataset.", "ARTIFACT_NOT_FOUND", 404
            )
        metadata = document["artifacts"][name]
        content = self._read_file(
            self.folder / "datasets" / dataset_id / name, metadata["sizeBytes"]
        )
        if (
            len(content) != metadata["sizeBytes"]
            or hashlib.sha256(content).hexdigest() != metadata["sha256"]
        ):
            raise _error("A dataset artifact failed checksum validation.", "ARTIFACT_CORRUPT")
        return content

    @staticmethod
    def _dataset_request_hash(
        draft_id: str,
        expected_revision: int,
        manifest: dict,
        metadata: dict,
        label_values: VersionLabelValues | None,
    ) -> str:
        return hashlib.sha256(
            _json(
                {
                    "draftId": draft_id,
                    "revision": expected_revision,
                    "manifest": manifest,
                    "artifacts": metadata,
                    **({"versionLabel": label_values.model_dump()} if label_values else {}),
                }
            )
        ).hexdigest()

    def replay_dataset_publication(
        self,
        operation_id: str,
        *,
        draft_id: str,
        expected_revision: int,
        preview_hash: str,
        version_label: dict | None = None,
    ) -> dict | None:
        """Replay a completed import without reopening its original external source files."""
        _revision(expected_revision)
        _label(operation_id, "operation ID")
        label_values = self._publication_label_values(version_label)
        self.initialize()
        with self._connection() as connection:
            operation = connection.execute(
                "SELECT * FROM publications WHERE id=?", (operation_id,)
            ).fetchone()
            if operation is None or operation["status"] != "published":
                return None
            document = self._get_dataset(connection, operation["dataset_id"])
            request_hash = self._dataset_request_hash(
                draft_id,
                expected_revision,
                document["manifest"],
                document["artifacts"],
                label_values,
            )
            if (
                operation["draft_id"] != draft_id
                or operation["expected_revision"] != expected_revision
                or document["manifest"].get("previewHash") != preview_hash
                or operation["request_hash"] != request_hash
            ):
                raise _error(
                    "The publication operation was used with different inputs or a "
                    "different version tag or commit note.",
                    "OPERATION_CONFLICT",
                )
            draft = self._draft(
                connection.execute("SELECT * FROM drafts WHERE id=?", (draft_id,)).fetchone()
            )
            if draft["status"] != "frozen" or draft["revision"] != expected_revision + 1:
                raise _error("The publication draft is inconsistent.", "STORAGE_CORRUPT")
            return self._with_version_label(connection, "dataset", document)

    def publish_dataset(
        self,
        draft_id: str,
        *,
        expected_revision: int,
        manifest: dict,
        artifacts: dict[str, bytes],
        operation_id: str,
        version_label: dict | None = None,
    ) -> dict:
        _revision(expected_revision)
        _label(operation_id, "operation ID")
        label_values = self._publication_label_values(version_label)
        manifest = _load_json(_json(manifest))
        metadata = _artifact_metadata(artifacts)
        content_hash = _hash_content(manifest, metadata)
        request_hash = self._dataset_request_hash(
            draft_id, expected_revision, manifest, metadata, label_values
        )
        with writer_lock(self.folder):
            self._initialize_locked()
            with self._connection() as connection, self._transaction(connection):
                previous = connection.execute(
                    "SELECT * FROM publications WHERE id=?", (operation_id,)
                ).fetchone()
                if previous is not None:
                    if previous["request_hash"] != request_hash:
                        raise _error(
                            "This operation ID was used for different publication inputs.",
                            "OPERATION_CONFLICT",
                        )
                    if previous["status"] == "published":
                        return self._with_version_label(
                            connection,
                            "dataset",
                            self._get_dataset(connection, previous["dataset_id"], checksums=True),
                        )
                draft = self._draft(
                    connection.execute("SELECT * FROM drafts WHERE id=?", (draft_id,)).fetchone()
                )
                self._editable(draft, expected_revision)
                if draft["kind"] != "import":
                    raise _error(
                        "Only an import draft can publish a dataset.", "INVALID_DRAFT_KIND", 422
                    )
                parent = manifest.get("parentId")
                if parent is not None:
                    if not isinstance(parent, str) or not _DATASET_ID.fullmatch(parent):
                        raise _error(
                            "A parentId must identify an existing frozen dataset.",
                            "INVALID_PARENT",
                            422,
                        )
                    self._get_dataset(connection, parent)
                now = _now()
                document = {
                    "id": f"dataset-{content_hash}",
                    "projectId": self.project_id,
                    "contentHash": content_hash,
                    "manifest": manifest,
                    "artifacts": metadata,
                    "createdAt": now,
                }
                if parent is not None:
                    document["parentId"] = parent
                existing = connection.execute(
                    "SELECT document FROM datasets WHERE id=?", (document["id"],)
                ).fetchone()
                final = self.folder / "datasets" / document["id"]
                if not existing and (final.exists() or final.is_symlink()):
                    raise _error(
                        "The dataset destination is occupied by unregistered artifacts.",
                        "PUBLICATION_CONFLICT",
                    )
                if existing:
                    document = self._get_dataset(connection, document["id"], checksums=True)
                elif previous is not None:
                    document = self._dataset_document(previous["document"])
                self._check_publication_label(
                    connection, "dataset", document["id"], "dataset", label_values
                )
                self._disk_manifest(document)  # Bound the complete envelope before journaling.
                stage_name = f"operation-{uuid4().hex}"
                if previous is None:
                    connection.execute(
                        "INSERT INTO publications VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            operation_id,
                            request_hash,
                            draft_id,
                            expected_revision,
                            document["id"],
                            stage_name,
                            _json(document).decode(),
                            "preparing",
                            None,
                            now,
                            now,
                        ),
                    )
                    if label_values is not None:
                        connection.execute(
                            "INSERT INTO publication_labels VALUES (?,?,?,?)",
                            (
                                operation_id,
                                label_values.tag,
                                label_values.tag.casefold(),
                                label_values.note,
                            ),
                        )
                else:
                    connection.execute(
                        "UPDATE publications SET stage_name=?,status='preparing',error=NULL,updated_at=? WHERE id=?",
                        (
                            stage_name,
                            now,
                            operation_id,
                        ),
                    )
            self._checkpoint("journal_committed")
            if existing:
                self._finish_locked(operation_id, document)
                with self._connection() as connection:
                    return self._with_version_label(connection, "dataset", document)
            stage = self.folder / ".staging" / stage_name
            ensure_managed_directory(stage)
            try:
                for name, content in sorted(artifacts.items()):
                    destination = stage / name
                    ensure_managed_directory(destination.parent)
                    self._write_file(destination, content)
                    fsync_directory(destination.parent)
                    self._checkpoint("artifact_written")
                self._write_file(stage / "manifest.json", self._disk_manifest(document))
                fsync_directory(stage)
                self._checkpoint("staging_complete")
                self._publish_directory(stage, document)
                self._checkpoint("directory_published")
                self._finish_locked(operation_id, document)
                self._checkpoint("database_committed")
                with self._connection() as connection:
                    return self._with_version_label(connection, "dataset", document)
            except OSError as error:
                raise _error(
                    "Dataset publication was interrupted; reopen to recover.",
                    "PUBLICATION_INTERRUPTED",
                    403,
                ) from error

    @staticmethod
    def _write_file(path: Path, content: bytes) -> None:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())

    def _publish_directory(self, stage: Path, document: dict) -> None:
        final = self.folder / "datasets" / document["id"]
        self._verify_directory(stage, document)
        if final.exists() or final.is_symlink():
            # Unindexed outputs are never overwritten, even if their contents look similar.
            raise _error("The dataset destination is already occupied.", "PUBLICATION_CONFLICT")
        self._sync_artifacts(stage, document)
        stage.rename(final)
        fsync_directory(self.folder / "datasets")
        fsync_directory(self.folder / ".staging")

    def _finish_locked(self, operation_id: str, document: dict) -> None:
        final = self.folder / "datasets" / document["id"]
        self._verify_directory(final, document)
        # Recovery may observe complete files before the interrupted writer had
        # synchronized them, or a rename before its parent-directory sync.
        self._sync_artifacts(final, document)
        fsync_directory(self.folder / "datasets")
        fsync_directory(self.folder / ".staging")
        with self._connection() as connection, self._transaction(connection):
            operation = connection.execute(
                "SELECT * FROM publications WHERE id=?", (operation_id,)
            ).fetchone()
            if operation is None or operation["dataset_id"] != document["id"]:
                raise _error("The publication journal is inconsistent.", "STORAGE_CORRUPT")
            if operation["status"] == "published":
                return
            draft = self._draft(
                connection.execute(
                    "SELECT * FROM drafts WHERE id=?", (operation["draft_id"],)
                ).fetchone()
            )
            self._editable(draft, operation["expected_revision"])
            label_values = self._journal_label(connection, operation_id)
            if operation["request_hash"] != self._dataset_request_hash(
                operation["draft_id"],
                operation["expected_revision"],
                document["manifest"],
                document["artifacts"],
                label_values,
            ):
                raise _error("The publication label journal is inconsistent.", "STORAGE_CORRUPT")
            connection.execute(
                "INSERT INTO datasets VALUES (?,?,?) ON CONFLICT(id) DO NOTHING",
                (
                    document["id"],
                    document["contentHash"],
                    _json(document).decode(),
                ),
            )
            self._apply_publication_label(
                connection, "dataset", document["id"], "dataset", label_values
            )
            connection.execute(
                "UPDATE drafts SET status='frozen',revision=revision+1,updated_at=? WHERE id=?",
                (_now(), draft["id"]),
            )
            connection.execute(
                "UPDATE publications SET status='published',error=NULL,updated_at=? WHERE id=?",
                (_now(), operation_id),
            )

    def _sync_artifacts(self, folder: Path, document: dict) -> None:
        directories = {folder}
        for name in (*document["artifacts"], "manifest.json"):
            path = folder / name
            self._regular(path)
            descriptor = None
            try:
                descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                os.fsync(descriptor)
            except OSError as error:
                raise _error(
                    "A published artifact cannot be synchronized.", "STORAGE_SYNC_FAILED"
                ) from error
            finally:
                if descriptor is not None:
                    os.close(descriptor)
            directories.update(parent for parent in path.parents if parent.is_relative_to(folder))
        for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
            fsync_directory(directory)

    def _recover_locked(self) -> None:
        with self._connection() as connection:
            operations = connection.execute(
                "SELECT * FROM publications WHERE status='preparing' ORDER BY created_at,id"
            ).fetchall()
        for operation in operations:
            document = self._dataset_document(operation["document"])
            if operation["dataset_id"] != document["id"] or not _STAGE_NAME.fullmatch(
                operation["stage_name"]
            ):
                raise _error("The publication journal contains unsafe metadata.", "STORAGE_CORRUPT")
            stage = self.folder / ".staging" / operation["stage_name"]
            final = self.folder / "datasets" / document["id"]
            try:
                if final.exists() or final.is_symlink():
                    self._verify_directory(final, document)
                else:
                    self._publish_directory(stage, document)
                self._finish_locked(operation["id"], document)
            except StorageError as error:
                if error.code not in {
                    "STORAGE_CORRUPT",
                    "ARTIFACT_CORRUPT",
                    "STORAGE_UNSAFE_PATH",
                    "PUBLICATION_CONFLICT",
                    "REVISION_CONFLICT",
                    "DRAFT_FROZEN",
                    "VERSION_TAG_CONFLICT",
                    "VERSION_LABEL_MISMATCH",
                }:
                    # Complete publications remain recoverable if synchronization
                    # or database access is temporarily unavailable.
                    raise
                # Keep questionable data quarantined for inspection; never follow it,
                # recursively delete it, or expose it as a published dataset.
                if stage.exists() or stage.is_symlink():
                    quarantine = self.folder / ".staging" / f"interrupted-{uuid4().hex}"
                    stage.rename(quarantine)
                    fsync_directory(stage.parent)
                # A conflicting final path is not proof of journal ownership.
                # Leave it untouched and unexposed for explicit repair.
                with self._connection() as connection, self._transaction(connection):
                    connection.execute(
                        "UPDATE publications SET status='interrupted',error=?,updated_at=? WHERE id=?",
                        (
                            error.code,
                            _now(),
                            operation["id"],
                        ),
                    )
