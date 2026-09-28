"""Host capacity, effective Task Center limits and admission of one task."""

import math
from pathlib import Path

from histopilot.storage.io import utc_now
from histopilot.taskcenter import procs
from histopilot.taskcenter.model import seconds_between
from histopilot.workers.training_process import gpu_snapshot, host_snapshot

NO_GPU = "No GPU available"
WAITING_RAM = "Waiting for RAM"
WAITING_GPU_MEMORY = "Waiting for GPU memory"
WAITING_IDLE_GPU = "Waiting for an idle GPU"
# A started task allocates its VRAM within about a minute; until then live free memory
# does not show it, so its request counts as pending on its GPU.
ALLOCATION_SECONDS = 90.0
# A GPU without HistoPilot tasks counts as empty while no more than its reserve and this
# much memory is in use; only an empty GPU lets a task that does not fit run alone.
EMPTY_GPU_SLACK_GB = 0.5


def host(*, gpu_probe=None) -> dict:
    """Current host snapshot; never initializes CUDA. ``gpu_probe`` defaults to gpu_snapshot."""
    from histopilot.application.system_compute import parse_cpu_info

    value = host_snapshot()
    try:
        value["physicalCpuCount"] = parse_cpu_info(Path("/proc/cpuinfo").read_text())[1]
    except OSError:
        value["physicalCpuCount"] = None
    probe = (gpu_probe or gpu_snapshot)()
    value["gpus"] = list(probe.get("gpus") or [])
    if probe.get("gpuProbeError"):
        value["gpuProbeError"] = probe["gpuProbeError"]
    return value


def effective(settings: dict, host: dict) -> dict:
    """Resolve defaults against detected hardware. GPU keys are integer indices."""
    slots = settings.get("gpuSlots") or {}
    reserves = settings["reserves"]
    gpus = host.get("gpus") or []
    vram_reserves = {
        gpu["index"]: float(reserves["vramGb"])
        if reserves.get("vramGb") is not None
        else max(1.0, 0.05 * (gpu.get("totalMemoryGb") or 0))
        for gpu in gpus
    }
    total_ram = host.get("totalRamGb") or 0
    return {
        "gpuSlots": {
            gpu["index"]: int(slots.get(str(gpu["index"]), settings["defaultGpuSlots"]))
            for gpu in gpus
        },
        "cpuTaskSlots": settings["cpuTaskSlots"]
        if settings.get("cpuTaskSlots") is not None
        else max(1, int(host.get("cpuCount") or 1) // 4),
        "reserves": {
            "cpuThreads": int(reserves["cpuThreads"]),
            "ramGb": float(reserves["ramGb"])
            if reserves.get("ramGb") is not None
            else min(16.0, max(2.0, 0.05 * total_ram)),
            "vramGb": float(reserves["vramGb"])
            if reserves.get("vramGb") is not None
            else max(vram_reserves.values(), default=1.0),
        },
        "vramReserves": vram_reserves,
    }


def _gpu_usage() -> dict:
    return {
        "slots": 0,
        "tasks": 0,
        "foreign": 0,
        "committedVramGb": 0.0,
        "pendingVramGb": 0.0,
        "exclusive": False,
        "exclusiveForeign": False,
    }


def _allocating(task: dict, now: str) -> bool:
    elapsed = seconds_between(task.get("startedAt"), now)
    return elapsed is None or elapsed < ALLOCATION_SECONDS


def usage(
    running_tasks: list[dict],
    foreign_leases: list[dict],
    host: dict,
    *,
    now: str | None = None,
    resident=None,
) -> dict:
    """Committed and pending capacity of running tasks and live foreign leases.

    ``now`` (ISO time, default the current UTC time) decides which tasks may still be
    allocating VRAM; ``resident(lease)`` measures a foreign lease's resident GiB and
    defaults to ``procs.lease_resident_gb``.
    """
    now = now or utc_now()
    measure = resident or procs.lease_resident_gb
    value = {
        "gpus": {gpu["index"]: _gpu_usage() for gpu in host.get("gpus") or []},
        "cpu": {"committedThreads": 0, "cpuTasks": 0},
        "ram": {"committedGb": 0.0, "pendingGb": 0.0, "tasks": 0, "foreignGb": 0.0},
    }
    for task in running_tasks:
        claim(value, task, task.get("gpu"), allocating=_allocating(task, now))
    for lease in foreign_leases:
        value["cpu"]["committedThreads"] += int(lease.get("cpus") or 0)
        reserved = float(lease.get("ramGb") or 0.0)
        if reserved > 0:
            # As in train_batch.available_device: the part of a foreign reservation that is
            # not resident yet is invisible in MemAvailable. Unknown residency frees nothing.
            value["ram"]["pendingGb"] += max(0.0, reserved - (measure(lease) or 0.0))
            value["ram"]["foreignGb"] += reserved
        if lease.get("gpu") is None:
            continue
        gpu = value["gpus"].setdefault(lease["gpu"], _gpu_usage())
        gpu["slots"] += 1
        gpu["foreign"] += 1
        if lease.get("runsPerGpu") == 1:
            gpu["exclusive"] = True
            gpu["exclusiveForeign"] = True
    return value


def claim(usage: dict, task: dict, gpu: int | None, *, allocating: bool = True) -> None:
    """Account a running (or just admitted) task in ``usage`` in place.

    ``allocating`` counts its VRAM request as pending on its GPU: live free memory does not
    show a task that was just admitted or is still starting.
    """
    request = task["request"]
    if not request.get("service"):
        usage["cpu"]["committedThreads"] += request["cpuThreads"] + request["dataWorkers"]
        if request["lane"] == "cpu":
            usage["cpu"]["cpuTasks"] += 1
        usage["ram"]["tasks"] = usage["ram"].get("tasks", 0) + 1
    usage["ram"]["committedGb"] += request["ramGb"]
    # Memory a task has requested but not yet made resident is not visible in MemAvailable.
    resident = (task.get("resources") or {}).get("privateRamGb") or 0.0
    usage["ram"]["pendingGb"] += max(0.0, request["ramGb"] - resident)
    if request["lane"] == "gpu" and gpu is not None:
        item = usage["gpus"].setdefault(gpu, _gpu_usage())
        item["slots"] += 1
        item["tasks"] += 1
        item["committedVramGb"] += request["vramGb"]
        if allocating:
            item["pendingVramGb"] = item.get("pendingVramGb", 0.0) + request["vramGb"]
        if request.get("exclusiveGpu"):
            item["exclusive"] = True


def blocking_scope(reason: str | None) -> str | None:
    """Which later candidates a waiting reason holds back: "all", "cpu" or "gpu"."""
    if reason is None:
        return None
    if reason.startswith("Waiting for CPU threads") or reason.startswith(WAITING_RAM):
        return "all"
    if reason.startswith("Waiting for a CPU task slot"):
        return "cpu"
    return "gpu"


def _ram_reason(need: float, host: dict, reserve: float) -> str:
    total = host.get("totalRamGb")
    if total is None or total - reserve >= need:
        return WAITING_RAM
    return (
        f"{WAITING_RAM}: needs {need:.1f} GiB, more than this machine offers "
        f"({max(0.0, total - reserve):.1f} GiB after the reserve), so it will run alone once "
        "no other task holds RAM"
    )


def _in_use_outside(total, live_free, reserve) -> float | None:
    """GiB a GPU without HistoPilot tasks has in use beyond its reserve; None when empty.

    Without a live reading nothing is known to be in use.
    """
    if total is None or live_free is None or not math.isfinite(live_free):
        return None
    used = total - live_free
    return used if used - reserve > EMPTY_GPU_SLACK_GB else None


def admit(
    task: dict, usage: dict, host: dict, effective: dict
) -> tuple[bool, int | None, str | None]:
    """Decide whether ``task`` fits now; the first failing rule gives the waiting reason.

    Like CPU threads, a RAM or VRAM request that does not fit may still run alone rather
    than wait forever: RAM when no other task of this runner and no foreign lease holds
    RAM, VRAM on a GPU without tasks, leases or exclusivity that is really empty (or has
    no live reading). Memory a program outside HistoPilot holds is waited out instead:
    alone next to it, the task would only run out of memory.
    """
    request = task["request"]
    reserves = effective["reserves"]
    if not request.get("service"):
        need = request["cpuThreads"] + request["dataWorkers"]
        committed = usage["cpu"]["committedThreads"]
        budget = max(0, int(host.get("cpuCount") or 1) - reserves["cpuThreads"])
        # A task larger than the whole budget may still run alone rather than wait forever.
        if committed and committed + need > budget:
            return False, None, f"Waiting for CPU threads ({committed}/{budget})"
        if request["lane"] == "cpu" and usage["cpu"]["cpuTasks"] >= effective["cpuTaskSlots"]:
            return (
                False,
                None,
                f"Waiting for a CPU task slot ({usage['cpu']['cpuTasks']}/"
                f"{effective['cpuTaskSlots']})",
            )
    available = host.get("availableRamGb")
    if available is None:
        return False, None, WAITING_RAM
    ram = usage["ram"]
    alone = not ram.get("tasks") and not ram.get("foreignGb")
    if available - reserves["ramGb"] - ram.get("pendingGb", 0.0) < request["ramGb"] and not alone:
        return False, None, _ram_reason(request["ramGb"], host, reserves["ramGb"])
    if request["lane"] != "gpu":
        return True, None, None
    gpus = host.get("gpus") or []
    if not gpus:
        return False, None, NO_GPU

    def free(gpu):
        value = gpu.get("freeMemoryGb")
        return value if value is not None and math.isfinite(value) else -math.inf

    candidates = sorted(
        gpus,
        key=lambda gpu: (usage["gpus"].get(gpu["index"], _gpu_usage())["slots"], -free(gpu)),
    )
    need = request["vramGb"]
    first_reason, idle, budgets, outside = None, [], [], None
    for gpu in candidates:
        index = gpu["index"]
        used = usage["gpus"].get(index, _gpu_usage())
        cap = effective["gpuSlots"].get(index, 1)
        reserve = effective.get("vramReserves", {}).get(index, reserves["vramGb"])
        total = gpu.get("totalMemoryGb")
        live_free = gpu.get("freeMemoryGb")
        pending = used.get("pendingVramGb", 0.0)
        budget = total - reserve if total is not None else None
        if budget is not None and cap >= 1:
            budgets.append(budget)
        if used["exclusive"]:
            reason = (
                f"GPU {index} is reserved exclusively by another job"
                if used["exclusiveForeign"]
                else WAITING_IDLE_GPU
            )
        elif request.get("exclusiveGpu") and used["slots"]:
            reason = WAITING_IDLE_GPU
        elif used["slots"] + 1 > cap:
            reason = f"Waiting for a GPU slot ({used['slots']}/{cap})"
        elif budget is not None and used["committedVramGb"] + need > budget:
            reason = WAITING_GPU_MEMORY
        elif live_free is not None and live_free - reserve - pending < need:
            reason = WAITING_GPU_MEMORY
        else:
            return True, index, None
        if reason == WAITING_GPU_MEMORY and not used["slots"] and not pending:
            idle.append((index, budget, _in_use_outside(total, live_free, reserve)))
        first_reason = first_reason or reason
    largest = max(budgets, default=None)
    for index, budget, in_use in idle:
        # Never alone on a smaller GPU while a larger one could hold the task.
        if budget is None or largest is None or need <= budget or budget >= largest:
            if in_use is None:
                return True, index, None
            outside = outside or (
                f"{WAITING_GPU_MEMORY}: {in_use:.1f} GiB of GPU {index} is in use outside "
                "HistoPilot"
            )
    if outside:
        return False, None, outside
    if largest is not None and need > largest:
        return (
            False,
            None,
            f"{WAITING_IDLE_GPU}: needs {need:.1f} GiB of GPU memory, more than any GPU "
            f"offers ({largest:.1f} GiB after the reserve), so it will run alone",
        )
    return False, None, first_reason
