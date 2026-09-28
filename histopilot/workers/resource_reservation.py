"""Use the existing cross-project CPU/RAM/GPU leases for preparation workers.

Only workers launched outside the Task Center lease themselves. Under the Task Center
(``HISTOPILOT_TASK_MANAGED=1``) the runner admits the task and publishes its lease, so
``reserve_preparation`` does nothing.
"""

import os
import time
from contextlib import contextmanager
from pathlib import Path

from histopilot.adapters.trident.performance import (
    estimate_vram_gb,
    execution_device_count,
    resolve_max_workers,
)
from histopilot.workers.packing_process import write_json
from histopilot.workers.train_batch import _capacity, _leases, available_device
from histopilot.workers.training_process import (
    cpu_slots_per_run,
    now,
    owned_processes,
    process_identity,
    stop_owned_processes,
)

# An extraction shares its GPU like a Task Center task does (the default "parallel GPU
# tasks" setting), instead of claiming it exclusively: HEST segmentation, the largest
# stage, uses about 9 of 24 GiB. ``runsPerGpu`` 1 made training and extraction block
# each other in both directions (the legacy scheduler admits by the smallest runsPerGpu
# on a GPU, and the Task Center treats a foreign runsPerGpu 1 lease as exclusive).
SHARED_RUNS_PER_GPU = 4


def managed() -> bool:
    return os.environ.get("HISTOPILOT_TASK_MANAGED") == "1"


def preparation_resources(kind, options=None):
    """Conservative reservations; the worker runtime still controls actual memory."""
    options = options or {}
    gpus = options.get("gpus") or [options.get("gpu", 0)] if kind == "extraction" else []
    gpus = sorted(set(gpu for gpu in gpus if gpu >= 0))
    workers = (
        resolve_max_workers(options) if kind == "extraction" else (options.get("max_workers") or 0)
    )
    devices = execution_device_count(options) if kind == "extraction" else 1
    return {
        "maxConcurrentRuns": 1,
        "gpuIds": gpus,
        "runsPerGpu": SHARED_RUNS_PER_GPU,
        "vramGb": estimate_vram_gb(options) if gpus else 0.0,
        "cpuThreadsPerRun": devices,
        "dataLoaderWorkers": workers * devices,
        "ramGbPerRun": float(4 * devices if kind == "extraction" else 1),
    }


class Reservation:
    def __init__(self, paths, values):
        self.paths, self.values = paths, values
        self.child = None

    def attach(self, pid):
        """A subprocess group keeps the reservation if its supervisor disappears."""
        identity = process_identity(pid)
        with _leases():
            for path, value in zip(self.paths, self.values, strict=True):
                value.update(process=identity, processGroupId=pid)
                write_json(path, value)
        self.child = identity

    def finish_child(self):
        if self.child:
            stop_owned_processes(self.child)


@contextmanager
def reserve_preparation(folder, kind, resources, cancelled):
    folder = Path(folder)
    if managed():
        # The Task Center runner admitted this task and holds its lease.
        yield Reservation([], [])
        return
    requested = list(resources["gpuIds"])
    paths, values = [], []
    reservation = Reservation(paths, values)
    process = process_identity()
    cpus, _ram = _capacity()
    if cpu_slots_per_run(resources) > cpus:
        raise ValueError(
            "The preparation worker requests more CPU slots than this host provides. Reduce data-loader workers."
        )
    try:
        while True:
            if cancelled():
                raise ValueError("Cancelled while waiting for resources.")
            with _leases() as (registry, active):
                allocations = []
                for index, gpu in enumerate(requested or [None]):
                    policy = {**resources, "gpuIds": [] if gpu is None else [gpu]}
                    if index:
                        policy.update(cpuThreadsPerRun=0, dataLoaderWorkers=0, ramGbPerRun=0)
                    available, selected = available_device(
                        policy, [*active, *allocations], _capacity()
                    )
                    if not available:
                        break
                    allocations.append(
                        {
                            "process": process,
                            "supervisor": process,
                            "processGroupId": os.getpid(),
                            "gpu": selected,
                            "cpus": cpu_slots_per_run(policy),
                            "ramGb": policy["ramGbPerRun"],
                            # Historical plans lack the key; they keep their exclusive GPU.
                            "runsPerGpu": int(policy.get("runsPerGpu") or 1),
                            "vramGb": float(policy.get("vramGb") or 0.0),
                            "kind": kind,
                            "batchId": folder.name,
                            "runId": folder.name,
                        }
                    )
                else:
                    for index, value in enumerate(allocations):
                        path = registry / f"lease-{os.getpid()}-preparation-{index}.json"
                        write_json(path, value)
                        paths.append(path)
                        values.append(value)
                    write_json(
                        folder / "resources.json",
                        {
                            "status": "running",
                            "kind": kind,
                            "resources": resources,
                            "updatedAt": now(),
                        },
                    )
                    break
            write_json(
                folder / "resources.json",
                {
                    "status": "queued",
                    "kind": kind,
                    "resources": resources,
                    "updatedAt": now(),
                    "waitingReason": "Waiting for shared CPU, RAM or GPU capacity.",
                },
            )
            time.sleep(1)
        yield reservation
    finally:
        # Descendants can outlive their parent. Keep their leases until the normal
        # registry reaper confirms their isolated process group is gone.
        survivors = reservation.child and owned_processes(
            reservation.child, reservation.child["pid"]
        )
        if paths and not survivors:
            with _leases():
                for path in paths:
                    path.unlink(missing_ok=True)
        write_json(
            folder / "resources.json",
            {
                "status": "released" if not survivors else "retained",
                "kind": kind,
                "resources": resources,
                "updatedAt": now(),
            },
        )
