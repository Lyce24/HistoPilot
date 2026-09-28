"""Interop with the legacy per-user resource lease registry.

Legacy batch schedulers, compute workers and preparation workers publish leases in
``$TMPDIR/histopilot-training-<uid>``. The runner reads them without side effects and
publishes one lease per task it starts, so legacy schedulers see its load too.
"""

import json
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path

from histopilot.storage.project_lock import StorageError, ensure_managed_directory, writer_lock
from histopilot.workers.packing_process import write_json
from histopilot.workers.training_process import owned_processes

LEASE_NAME = re.compile(r"^lease-(\d+)(?:-preparation-(\d+))?\.json$")
MAX_LEASE_BYTES = 1024 * 1024


def registry() -> Path:
    # Pinned legacy workers hard-code this location; it must not move.
    return Path(tempfile.gettempdir()) / f"histopilot-training-{os.getuid()}"


def _check_registry(folder: Path) -> None:
    info = folder.stat()
    if info.st_uid != os.getuid() or info.st_mode & 0o022:
        raise StorageError("Training resource registry is unsafe.", "TRAINING_REGISTRY_UNSAFE", 403)


def _writable_registry() -> Path:
    folder = registry()
    ensure_managed_directory(folder)
    _check_registry(folder)
    return folder


def _read(path: Path) -> dict:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        content = stream.read(MAX_LEASE_BYTES + 1)
    if len(content) > MAX_LEASE_BYTES:
        raise ValueError("Lease exceeds its size limit.")
    value = json.loads(content)
    identity = value.get("process") if isinstance(value, dict) else None
    if (
        not isinstance(identity, dict)
        or type(identity.get("pid")) is not int
        or type(identity.get("startTicks")) is not int
        or not isinstance(identity.get("bootId"), str)
    ):
        raise ValueError("Lease has no valid process identity.")
    return value


def _live(value: dict) -> bool:
    identity = value["process"]
    group = value.get("processGroupId")
    if group is None:
        group = identity.get("pid")
    try:
        return bool(owned_processes(identity, group)) or bool(
            owned_processes(value.get("supervisor"))
        )
    except (StorageError, OSError):
        return True  # Unreadable ownership evidence never frees capacity.


def read_leases() -> list[dict]:
    """Every lease in the registry, never deleting files or taking the registry lock.

    Safe to call while holding ``registry_lock()``. Unreadable or malformed files are
    reported as ``{"invalid": True, "file": name}``. Raises ``StorageError`` when the
    registry itself is unsafe.
    """
    folder = registry()
    if not folder.is_dir():
        return []
    _check_registry(folder)
    leases = []
    for path in sorted(folder.glob("lease-*.json")):
        if not LEASE_NAME.match(path.name):
            leases.append({"invalid": True, "file": path.name})
            continue
        try:
            value = _read(path)
            gpu = value.get("gpu")
            if gpu is not None and type(gpu) is not int:
                raise ValueError("Lease GPU is invalid.")
            group = value.get("processGroupId")
            leases.append(
                {
                    "file": path.name,
                    "taskId": value.get("taskId"),
                    "kind": value.get("kind"),
                    "batchId": value.get("batchId"),
                    "runId": value.get("runId"),
                    "gpu": gpu,
                    "cpus": max(0, int(value.get("cpus") or 0)),
                    "ramGb": max(0.0, float(value.get("ramGb") or 0)),
                    "runsPerGpu": int(value.get("runsPerGpu") or 1),
                    "process": value["process"],
                    "processGroupId": group if type(group) is int else None,
                    "supervisor": value.get("supervisor")
                    if isinstance(value.get("supervisor"), dict)
                    else None,
                    "live": _live(value),
                }
            )
        except FileNotFoundError:
            continue  # Released while the directory was being listed.
        except (OSError, ValueError, TypeError, AttributeError):
            leases.append({"invalid": True, "file": path.name})
    return leases


@contextmanager
def registry_lock(timeout: float = 5):
    """Hold the registry writer lock that legacy schedulers take around read, spawn, write.

    Raises ``StorageError`` (``PROJECT_BUSY``) when the lock stays taken for ``timeout``.
    """
    folder = _writable_registry()
    with writer_lock(folder, timeout=timeout):
        yield folder


def write_task_lease(
    task: dict,
    identity: dict,
    gpu: int | None,
    *,
    cpus: int,
    ram_gb: float,
    runs_per_gpu: int,
    supervisor: dict,
    locked: bool = False,
) -> str:
    """Publish a legacy-compatible lease for a runner task and return its file name.

    ``locked`` means the caller already holds ``registry_lock()``.
    """
    group = task.get("group") or {}
    value = {
        "process": identity,
        "processGroupId": identity["pid"],
        "gpu": gpu,
        "cpus": max(1, int(cpus)),
        "ramGb": max(0.1, float(ram_gb)),
        "runsPerGpu": min(16, max(1, int(runs_per_gpu))),
        "batchId": group.get("id") or task.get("ownerKey") or task["id"],
        "runId": task["id"],
        "kind": "task-center",
        "taskId": task["id"],
        "supervisor": supervisor,
    }
    name = f"lease-{identity['pid']}.json"
    folder = _writable_registry()
    if locked:
        write_json(folder / name, value)
    else:
        with writer_lock(folder, timeout=5):
            write_json(folder / name, value)
    return name


def remove_task_lease(file_name: str, *, locked: bool = False) -> None:
    if not LEASE_NAME.match(file_name):
        raise ValueError(f"Not a lease file name: {file_name}")
    folder = registry()
    if not folder.is_dir():
        return
    _check_registry(folder)
    if locked:
        (folder / file_name).unlink(missing_ok=True)
        return
    with writer_lock(folder, timeout=5):
        (folder / file_name).unlink(missing_ok=True)


def foreign(leases: list[dict], own_task_ids) -> list[dict]:
    """Live leases that do not belong to tasks this runner manages."""
    own = set(own_task_ids)
    return [
        lease
        for lease in leases
        if not lease.get("invalid") and lease.get("live") and lease.get("taskId") not in own
    ]
