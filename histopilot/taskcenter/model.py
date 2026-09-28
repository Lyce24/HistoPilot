"""Task Center vocabulary and validation of task specifications, requests and settings."""

import copy
import math
import os
from datetime import UTC, datetime

from histopilot.storage.project_lock import StorageError

PROTOCOL_VERSION = 1
SCHEMA_VERSION = 1
STATES = (
    "blocked",
    "queued",
    "starting",
    "running",
    "stopping",
    "succeeded",
    "failed",
    "cancelled",
    "interrupted",
)
PENDING = frozenset({"blocked", "queued"})
ACTIVE = frozenset({"starting", "running", "stopping"})
LIVE = PENDING | ACTIVE
TERMINAL = frozenset({"succeeded", "failed", "cancelled", "interrupted"})
LANES = ("gpu", "cpu")
PRIORITIES = ("interactive", "normal")  # interactive is admitted first
STOP_REASONS = ("cancel", "pause")
EXIT_REASONS = (
    "ok",
    "error",
    "oom",
    "cuda_failure",
    "cancelled",
    "paused",
    "interrupted",
    "lost",
    "already-complete",
    "busy",
)
CONDITIONS = ("succeeded", "terminal")
# Bookkeeping hooks of a concluded task that may still run again: an automatic requeue
# still being decided ("requeue_intent") or decided but not yet carried out ("on_requeue").
REQUEUE_HOOKS = frozenset({"requeue_intent", "on_requeue"})
PLAN_ORDER_SPAN = 1_000_000

DEFAULT_SETTINGS = {
    "gpuSlots": {},  # {"0": 5}; a missing index uses defaultGpuSlots
    "defaultGpuSlots": 4,
    "cpuTaskSlots": None,  # None -> max(1, logical_cpus // 4)
    # None -> ram: min(16, max(2, 5% of total)); vram: max(1.0, 5% of the GPU's total)
    "reserves": {"cpuThreads": 2, "ramGb": None, "vramGb": None},
    "defaults": {"cpuThreadsPerRun": 2, "dataLoaderWorkers": 2},
    "paused": False,
    "autoResume": True,
    "cancelGraceSeconds": 30,
    "stallMinutes": 30,
}

REQUEST_DEFAULTS = {
    "lane": "cpu",
    "cpuThreads": 2,
    "dataWorkers": 0,
    "ramGb": 4.0,
    "vramGb": 0.0,
    "sharedFiles": [],
    "exclusiveGpu": False,
    "service": False,
    "graceSeconds": None,
    "workloadKey": None,
    "workload": None,
}


def awaiting_requeue(task: dict) -> bool:
    """A concluded task the runner may still requeue; not finished for its dependents."""
    return (
        task.get("state") in TERMINAL
        and (task.get("bookkeeping") or {}).get("hook") in REQUEUE_HOOKS
    )


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def parse_iso(value) -> datetime | None:
    """Parse a stored timestamp; naive values are UTC. Invalid input reads as unknown."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def seconds_between(start, end) -> float | None:
    first, last = parse_iso(start), parse_iso(end)
    return (last - first).total_seconds() if first and last else None


def _invalid(message: str, code: str = "TASK_INVALID") -> StorageError:
    return StorageError(message, code, 422)


def _text(value, name: str, *, optional=False, limit=2000) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value or len(value) > limit or "\0" in value:
        raise _invalid(f"Task field {name} must be a non-empty string.")
    return value


def _integer(value, name: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise _invalid(f"Task field {name} must be an integer from {low} to {high}.")
    return value


def _number(value, name: str, low: float, high: float) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise _invalid(f"Task field {name} must be a number from {low} to {high}.")
    return float(value)


def _boolean(value, name: str) -> bool:
    if type(value) is not bool:
        raise _invalid(f"Task field {name} must be true or false.")
    return value


def _absolute(value, name: str, *, optional=False) -> str | None:
    text = _text(value, name, optional=optional, limit=4096)
    if text is not None and not os.path.isabs(text):
        raise _invalid(f"Task field {name} must be an absolute path.")
    return text


def _mapping(value, name: str) -> dict:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise _invalid(f"Task field {name} must be an object.")
    return copy.deepcopy(value)


def normalize_request(value: dict) -> dict:
    """Fill request defaults and validate ranges; unknown keys are kept for forward use."""
    if not isinstance(value, dict):
        raise _invalid("A task resource request must be an object.")
    request = {**copy.deepcopy(REQUEST_DEFAULTS), **copy.deepcopy(value)}
    if request["lane"] not in LANES:
        raise _invalid("Task lane must be gpu or cpu.")
    request["cpuThreads"] = _integer(request["cpuThreads"], "request.cpuThreads", 1, 1024)
    request["dataWorkers"] = _integer(request["dataWorkers"], "request.dataWorkers", 0, 256)
    request["ramGb"] = _number(request["ramGb"], "request.ramGb", 0, 65536)
    request["vramGb"] = _number(request["vramGb"], "request.vramGb", 0, 4096)
    shared = request["sharedFiles"]
    if not isinstance(shared, list) or any(not isinstance(item, str) for item in shared):
        raise _invalid("Task field request.sharedFiles must be a list of paths.")
    request["exclusiveGpu"] = _boolean(request["exclusiveGpu"], "request.exclusiveGpu")
    request["service"] = _boolean(request["service"], "request.service")
    if request["graceSeconds"] is not None:
        request["graceSeconds"] = _number(request["graceSeconds"], "request.graceSeconds", 0, 86400)
    if request["workloadKey"] is not None:
        _text(request["workloadKey"], "request.workloadKey")
    if request["workload"] is not None:
        request["workload"] = _mapping(request["workload"], "request.workload")
    return request


def normalize_command(value: dict) -> dict:
    if not isinstance(value, dict):
        raise _invalid("A task command must be an object.")
    argv = value.get("argv")
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(item, str) or "\0" in item for item in argv)
    ):
        raise _invalid("Task field command.argv must be a non-empty list of strings.")
    env = value.get("env") or {}
    if not isinstance(env, dict):
        raise _invalid("Task field command.env must be an object.")
    environment = {}
    for key, item in env.items():
        if not isinstance(key, str) or not key or "=" in key or "\0" in key:
            raise _invalid("Task environment names must be non-empty strings without '='.")
        if type(item) is int:
            item = str(item)
        if not isinstance(item, str) or "\0" in item:
            raise _invalid(f"Task environment value {key} must be a string.")
        environment[key] = item
    return {
        "argv": list(argv),
        "cwd": _absolute(value.get("cwd"), "command.cwd"),
        "env": environment,
        "log": _absolute(value.get("log"), "command.log"),
        "progress": _absolute(value.get("progress"), "command.progress", optional=True),
        "result": _absolute(value.get("result"), "command.result", optional=True),
    }


def normalize_owner(value: dict) -> dict:
    if not isinstance(value, dict):
        raise _invalid("A task owner must be an object.")
    workspace = value.get("workspace")
    if workspace is not None:
        workspace = _text(workspace, "owner.workspace", limit=4096)
    return {
        "kind": _text(value.get("kind"), "owner.kind", limit=100),
        "id": _text(value.get("id"), "owner.id", limit=500),
        "projectId": _text(value.get("projectId"), "owner.projectId", limit=500),
        "projectFolder": _absolute(value.get("projectFolder"), "owner.projectFolder"),
        "workspace": workspace,
        "title": _text(value.get("title"), "owner.title"),
        "labels": _mapping(value.get("labels") or {}, "owner.labels"),
    }


def normalize_task(value: dict) -> dict:
    """Validate an enqueue task specification (section 2.4 of the spec)."""
    if not isinstance(value, dict):
        raise _invalid("A task specification must be an object.")
    group = value.get("group")
    if group is not None:
        if not isinstance(group, dict):
            raise _invalid("Task field group must be an object.")
        group = {
            "kind": _text(group.get("kind"), "group.kind", limit=100),
            "id": _text(group.get("id"), "group.id", limit=500),
        }
    priority = value.get("priority") or "normal"
    if priority not in PRIORITIES:
        raise _invalid("Task priority must be interactive or normal.")
    dependencies = []
    seen = set()
    for item in value.get("dependsOn") or []:
        if not isinstance(item, dict):
            raise _invalid("Task dependencies must be objects.")
        condition = item.get("condition") or "succeeded"
        if condition not in CONDITIONS:
            raise _invalid("Task dependency conditions are succeeded or terminal.")
        target = _text(item.get("task"), "dependsOn.task", limit=200)
        if target not in seen:
            seen.add(target)
            dependencies.append({"task": target, "condition": condition})
    identity = _text(value.get("id"), "id", limit=200)
    if identity in seen:
        raise _invalid("A task cannot depend on itself.")
    return {
        "id": identity,
        "kind": _text(value.get("kind"), "kind", limit=100),
        "adapter": _text(value.get("adapter") or "generic", "adapter", limit=100),
        "title": _text(value.get("title") or identity, "title"),
        "group": group,
        "planOrder": _integer(value.get("planOrder", 0), "planOrder", 0, PLAN_ORDER_SPAN - 1),
        "priority": priority,
        "exclusiveKey": _text(value.get("exclusiveKey"), "exclusiveKey", optional=True, limit=4096),
        "sessionName": _text(value.get("sessionName"), "sessionName", optional=True, limit=500),
        "labels": _mapping(value.get("labels") or {}, "labels"),
        "request": normalize_request(value.get("request") or {}),
        "command": normalize_command(value.get("command")),
        "adapterData": _mapping(value.get("adapterData") or {}, "adapterData"),
        "dependsOn": dependencies,
    }


def _settings_error(message: str) -> StorageError:
    return StorageError(message, "TASK_CENTER_SETTINGS_INVALID", 422)


def merge_settings(stored: dict) -> dict:
    """Overlay stored top-level keys on the defaults; nested objects merge one level deep."""
    merged = copy.deepcopy(DEFAULT_SETTINGS)
    for key, value in stored.items():
        if key not in merged:
            continue  # Keys written by a newer protocol are ignored, not fatal.
        if key in ("reserves", "defaults") and isinstance(value, dict):
            merged[key].update({k: v for k, v in value.items() if k in merged[key]})
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def apply_settings_patch(current: dict, patch: dict) -> dict:
    """Deep-merge a settings patch and validate the result."""
    if not isinstance(patch, dict):
        raise _settings_error("Task Center settings must be an object.")
    unknown = set(patch) - set(DEFAULT_SETTINGS)
    if unknown:
        raise _settings_error(f"Unknown Task Center settings: {', '.join(sorted(unknown))}.")
    result = copy.deepcopy(current)
    for key, value in patch.items():
        if key == "gpuSlots":
            if not isinstance(value, dict):
                raise _settings_error("gpuSlots must map GPU indices to slot counts.")
            slots = dict(result.get("gpuSlots") or {})
            for index, count in value.items():
                if not str(index).isdecimal():
                    raise _settings_error("gpuSlots keys must be GPU indices.")
                if count is None:
                    slots.pop(str(int(index)), None)
                else:
                    slots[str(int(index))] = count
            result["gpuSlots"] = slots
        elif key in ("reserves", "defaults"):
            if not isinstance(value, dict):
                raise _settings_error(f"{key} must be an object.")
            unknown = set(value) - set(DEFAULT_SETTINGS[key])
            if unknown:
                raise _settings_error(f"Unknown {key} settings: {', '.join(sorted(unknown))}.")
            result[key] = {**result[key], **value}
        else:
            result[key] = value
    validate_settings(result)
    return result


def validate_settings(settings: dict) -> None:
    def integer(value, name, low, high):
        if type(value) is not int or not low <= value <= high:
            raise _settings_error(f"{name} must be an integer from {low} to {high}.")

    def number(value, name, low, high):
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise _settings_error(f"{name} must be a number from {low} to {high}.")

    for index, count in settings["gpuSlots"].items():
        integer(count, f"gpuSlots[{index}]", 1, 16)
    integer(settings["defaultGpuSlots"], "defaultGpuSlots", 1, 16)
    if settings["cpuTaskSlots"] is not None:
        integer(settings["cpuTaskSlots"], "cpuTaskSlots", 1, 128)
    reserves = settings["reserves"]
    integer(reserves["cpuThreads"], "reserves.cpuThreads", 0, 1024)
    for key in ("ramGb", "vramGb"):
        if reserves[key] is not None:
            number(reserves[key], f"reserves.{key}", 0, 65536)
    integer(settings["defaults"]["cpuThreadsPerRun"], "defaults.cpuThreadsPerRun", 1, 32)
    integer(settings["defaults"]["dataLoaderWorkers"], "defaults.dataLoaderWorkers", 0, 16)
    for key in ("paused", "autoResume"):
        if type(settings[key]) is not bool:
            raise _settings_error(f"{key} must be true or false.")
    number(settings["cancelGraceSeconds"], "cancelGraceSeconds", 0, 3600)
    number(settings["stallMinutes"], "stallMinutes", 1, 10080)
