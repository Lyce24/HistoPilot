"""Pretrained weights a recipe starts from, pinned by SHA-256.

An architecture can start from a file of pretrained weights (OceanPath Mamba-2 MIL from an
OceanPath aggregator checkpoint). Reviewing a batch resolves each such path inside the
configured data roots, without symbolic links, and hashes the file; the frozen batch
keeps those pins. Launching rechecks them, and a worker loads only bytes whose size and
SHA-256 still match. A changed, moved or missing file refuses the work with
MODEL_CHECKPOINT_CHANGED rather than training from other weights.

Standard library only: the service, the Task Center and the workers all import it.
"""

import hashlib
import os
import stat
from pathlib import Path

from histopilot.models import catalog
from histopilot.storage.project_lock import StorageError, reject_symlink_components

MAX_BYTES = 4 * 1024**3
PIN_KEYS = ("path", "bytes", "sha256")


def _read(path, *, keep):
    """Size, SHA-256 and optionally the bytes of one regular file, read without links."""
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= MAX_BYTES:
            raise ValueError("not a nonempty regular file of at most 4 GiB")
        digest, blocks = hashlib.sha256(), []
        while block := stream.read(1024 * 1024):
            digest.update(block)
            if keep:
                blocks.append(block)
        after = os.fstat(stream.fileno())
    if any(
        getattr(before, key) != getattr(after, key)
        for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    ):
        raise ValueError("it changed while being read")
    pin = {"path": str(path), "bytes": after.st_size, "sha256": digest.hexdigest()}
    return pin, b"".join(blocks) if keep else None


def checkpoint_key(value: str) -> str:
    """The form of a recipe's path that its pin records."""
    return str(Path(value))


def pin(filesystem, value: str) -> dict:
    """Resolve and hash one checkpoint path a recipe names, inside the data roots."""
    path = Path(value)
    try:
        if not path.is_absolute() or ".." in path.parts or "\x00" in value:
            raise ValueError("not an absolute path without '..'")
        reject_symlink_components(path)
        resolved = path.resolve(strict=True)
        if resolved != path:
            raise ValueError("not a direct path")
        if not filesystem._contains(resolved):
            raise ValueError("outside the configured data roots")
        return _read(resolved, keep=False)[0]
    except (OSError, ValueError, RuntimeError, StorageError) as error:
        raise StorageError(
            f"The aggregator checkpoint {value} cannot be used: {error}. Choose a regular "
            "file inside a configured data root, without symbolic links.",
            "MODEL_CHECKPOINT_INVALID",
            422,
        ) from error


def pin_recipes(filesystem, recipes) -> list[dict]:
    """One pin per distinct checkpoint the recipes start from, in path order."""
    paths = {
        checkpoint_key(path)
        for path in (catalog.initial_checkpoint(recipe) for recipe in recipes)
        if path
    }
    return [pin(filesystem, path) for path in sorted(paths)]


def verify(pins, filesystem=None) -> None:
    """Refuse when a pinned checkpoint is missing or its bytes changed since review."""
    for expected in pins:
        try:
            current = (
                pin(filesystem, expected["path"])
                if filesystem is not None
                else _read(Path(expected["path"]), keep=False)[0]
            )
        except (OSError, ValueError, StorageError) as error:
            raise StorageError(
                f"The pinned aggregator checkpoint {expected['path']} is missing or can no "
                f"longer be read ({error}). Restore it, or review a new batch.",
                "MODEL_CHECKPOINT_CHANGED",
            ) from error
        if any(current[key] != expected[key] for key in PIN_KEYS):
            raise StorageError(
                f"The aggregator checkpoint {expected['path']} changed after the batch was "
                f"frozen (SHA-256 {current['sha256'][:12]}…, pinned {expected['sha256'][:12]}…). "
                "Restore the pinned file, or review a new batch.",
                "MODEL_CHECKPOINT_CHANGED",
            )


def read_pinned(pins, value: str) -> tuple[dict, bytes]:
    """The bytes of the checkpoint a recipe names, only if they match its pin."""
    key = checkpoint_key(value)
    expected = next((item for item in pins or () if item["path"] == key), None)
    if expected is None:
        raise StorageError(
            f"The aggregator checkpoint {value} was not pinned when the batch was frozen.",
            "MODEL_CHECKPOINT_CHANGED",
        )
    try:
        reject_symlink_components(Path(key))
        current, content = _read(Path(key), keep=True)
    except (OSError, ValueError, StorageError) as error:
        raise StorageError(
            f"The pinned aggregator checkpoint {key} is missing or can no longer be read "
            f"({error}).",
            "MODEL_CHECKPOINT_CHANGED",
        ) from error
    if any(current[name] != expected[name] for name in PIN_KEYS):
        raise StorageError(
            f"The aggregator checkpoint {key} changed after the batch was frozen.",
            "MODEL_CHECKPOINT_CHANGED",
        )
    return expected, content
