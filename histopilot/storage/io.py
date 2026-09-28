"""Canonical JSON, content hashes, timestamps and bounded, durable JSON files.

Stored identities hash one of three canonical encodings. Each call site keeps the one it
has always used, because configuration ids, content hashes, job ids and preview hashes
are persisted in project stores and records:

- compact ASCII (``ascii=True, compact=True``), the default of ``content_hash``: most
  content hashes, evidence hashes and plan hashes;
- compact UTF-8 (``ascii=False, compact=True``): scientific-store documents, and so every
  configuration id, and import previews and artifacts;
- default separators (``ascii=True, compact=False``): extraction and feature-pack job
  identities and preview hashes.

``tests/test_storage_io.py`` pins the digest each encoding gives for fixed inputs.

This module is part of the training code fingerprint
(``workers/training_process.compute_snapshot``); keep it free of heavy imports.
"""

import hashlib
import json
import math
import os
import stat
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from histopilot.storage.project_lock import (
    StorageError,
    fsync_directory,
    reject_symlink_components,
)

MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_JSON_DEPTH = 64


def canonical_json(value, *, ascii: bool, compact: bool) -> bytes:
    """Sorted-key, finite JSON; see the module docstring for which encoding to use."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":") if compact else (", ", ": "),
        ensure_ascii=ascii,
        allow_nan=False,
    ).encode("utf-8")


def content_hash(value, *, ascii: bool = True, compact: bool = True) -> str:
    """SHA-256 hex digest of ``canonical_json(value)``; compact ASCII unless told otherwise."""
    return hashlib.sha256(canonical_json(value, ascii=ascii, compact=compact)).hexdigest()


def utc_now(*, zulu: bool = False) -> str:
    """The current UTC time in ISO 8601, ending ``+00:00`` or, with ``zulu``, ``Z``.

    The scientific store, project descriptors and lifecycle metadata write ``Z``; worker and
    Task Center records write ``+00:00``. Keep each record's existing form.
    """
    value = datetime.now(UTC).isoformat()
    return value.replace("+00:00", "Z") if zulu else value


def write_json_atomic(
    path: Path, value, *, limit: int | None = MAX_JSON_BYTES, sync: bool = True
) -> None:
    """Replace ``path`` with indented JSON through a temporary file and a rename.

    With ``sync`` (the default) the file and its folder are fsynced, so the new content
    survives a power loss once this returns. Per-epoch telemetry passes ``sync=False``.
    """
    path = Path(path)
    reject_symlink_components(path)
    content = json.dumps(value, indent=2, allow_nan=False).encode() + b"\n"
    if limit is not None and len(content) > limit:
        raise StorageError("A metadata file exceeds its size limit.", "METADATA_LIMIT", 413)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            if sync:
                stream.flush()
                os.fsync(stream.fileno())
        os.replace(temporary, path)
        if sync:
            fsync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def regular_file(path: Path, *, missing_ok: bool = False, parents: bool = True):
    """``lstat`` of a managed regular file with one link, reached without symbolic links.

    Returns None for a missing file when ``missing_ok``.
    """
    if parents and any(parent.is_symlink() for parent in path.parents):
        raise StorageError(
            "Managed storage cannot traverse symbolic links.", "STORAGE_UNSAFE_PATH", 403
        )
    try:
        info = path.lstat()
    except FileNotFoundError:
        if missing_ok:
            return None
        raise StorageError("A required storage file is missing.", "STORAGE_CORRUPT") from None
    except OSError as error:
        raise StorageError(
            "A managed storage path cannot be inspected.", "STORAGE_READ_FAILED", 403
        ) from error
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise StorageError(
            "Managed files must be regular files without aliases.", "STORAGE_UNSAFE_PATH", 403
        )
    return info


def read_file_bounded(path: Path, limit: int) -> bytes:
    """The bytes of a managed regular file, refusing aliases and files over ``limit``."""
    regular_file(path)
    descriptor = None
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise StorageError(
                "Managed files must be regular files without aliases.",
                "STORAGE_UNSAFE_PATH",
                403,
            )
        if info.st_size > limit:
            raise StorageError("A stored artifact exceeds its declared bounds.", "STORAGE_CORRUPT")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            content = handle.read(limit + 1)
        if len(content) > limit:
            raise StorageError("A stored artifact exceeds its declared bounds.", "STORAGE_CORRUPT")
        return content
    except OSError as error:
        raise StorageError(
            "A stored artifact cannot be read.", "STORAGE_READ_FAILED", 403
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _finite(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Metadata numbers must be finite.")
    return number


def decode_json_object(content: bytes | str) -> dict:
    """A JSON object with finite numbers, nested at most ``MAX_JSON_DEPTH`` levels.

    Raises ValueError (or UnicodeError) otherwise. The decoder may accept nesting that
    later exhausts FastAPI's recursive response encoder, so the depth is checked
    iteratively, keeping one iterator per level so wide arrays add no large work list.
    """
    value = json.loads(content, parse_float=_finite, parse_constant=_finite)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object.")
    pending = [(iter(value.values()), 1)]
    while pending:
        children, depth = pending[-1]
        try:
            child = next(children)
        except StopIteration:
            pending.pop()
            continue
        if isinstance(child, (dict, list)):
            if depth >= MAX_JSON_DEPTH:
                raise ValueError("Metadata exceeds its nesting limit.")
            pending.append((iter(child.values() if isinstance(child, dict) else child), depth + 1))
    return value


def read_json_bounded(path: Path, limit: int = MAX_JSON_BYTES) -> dict:
    """A managed JSON object file (see ``read_file_bounded`` and ``decode_json_object``).

    Unreadable or unsafe files raise their storage error; content that is not a finite,
    bounded JSON object raises ``TRAINING_STATE_INVALID``, the code callers and the UI
    already handle for worker metadata.
    """
    content = read_file_bounded(Path(path), limit)
    try:
        return decode_json_object(content)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise StorageError("Invalid training metadata.", "TRAINING_STATE_INVALID") from error
