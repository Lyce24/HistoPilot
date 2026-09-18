"""Use the existing cross-project CPU/RAM/GPU leases for preparation workers."""

import os
import time
from contextlib import contextmanager
from pathlib import Path

from histopilot.workers.packing_process import write_json
from histopilot.workers.train_batch import _capacity, _leases, available_device
from histopilot.workers.training_process import (
    cpu_slots_per_run,
    now,
    owned_processes,
    process_identity,
    stop_owned_processes,
)


def preparation_resources(kind, options=None):
    """Conservative reservations; the worker runtime still controls actual memory."""
    options = options or {}
    gpus = options.get("gpus") or [options.get("gpu", 0)] if kind == "extraction" else []
    gpus = sorted(set(gpu for gpu in gpus if gpu >= 0))
    workers = options.get("max_workers")
    workers = 2 if workers is None and kind == "extraction" else (workers or 0)
    devices = max(1, len(gpus))
    return {
        "maxConcurrentRuns": 1,
        "gpuIds": gpus,
        "runsPerGpu": 1,
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
                            "runsPerGpu": 1,
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
