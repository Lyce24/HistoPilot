"""Attention, gallery and morphology helpers for tests on the Task Center path.

Services built here queue their work in the test's private Task Center (the
``task_center`` fixture) exactly as production does. Tests stand in for the worker:
they write its artifacts and receipts, or run it in this process, and then conclude
the task the way the runner records an exited worker.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from histopilot.application.compute_jobs import ComputeJobService
from histopilot.taskcenter.model import LIVE
from histopilot.workers.packing_process import write_json
from histopilot.workers.training_process import read_json


def runtime():
    return {
        "available": True,
        "python": sys.executable,
        "versions": {},
        "cudaAvailable": False,
        "gpuCount": 0,
        # Launch checks must not inherit the CI runner's resource limits.
        "host": {"cpuCount": 8, "totalRamGb": 16},
    }


def managed_jobs(store, center):
    """Compute jobs that queue in ``center``, as production launches them."""
    return ComputeJobService(store, runtime=runtime, task_center=center.client)


def compute_tasks(center, **filters):
    """Queued compute work only; gallery fixtures also leave their packing task behind."""
    return center.tasks(kind="compute-job", **filters)


def compute_task(service, identity, center):
    """The Task Center task of one interpretation's compute job."""
    return center.task(read_json(service.jobs.folder(identity) / "state.json")["taskId"])


def complete(service, identity, result, center):
    """Record a finished worker: its receipts, then its task concluded as succeeded.

    Rewriting the receipts of an already concluded job leaves the task as it is.
    """
    folder = service.jobs.folder(identity)
    state = read_json(folder / "state.json")
    write_json(folder / "state.json", {**state, "status": "completed", "result": result})
    write_json(folder / "result.json", result)
    if center.state(state["taskId"]) in LIVE:
        center.finish(state["taskId"], "succeeded")


def run_compute_worker(center, task_id, *, timeout=60):
    """Run a queued CPU-lane compute task's worker as the runner spawns it, then conclude
    the task with its exit status."""
    center.start(task_id)
    task = center.task(task_id)
    command = task["command"]
    finished = subprocess.run(
        command["argv"],
        cwd=command["cwd"],
        env={
            **os.environ,
            **command["env"],
            "CUDA_VISIBLE_DEVICES": "",
            "HISTOPILOT_TASK_ID": task_id,
            "HISTOPILOT_TASK_ATTEMPT": str(task["attempt"]),
            "HISTOPILOT_TASK_GPU": "",
        },
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    center.finish(
        task_id,
        "succeeded" if finished.returncode == 0 else "failed",
        returncode=finished.returncode,
    )
    return finished


def managed_packing(store, filesystem, center):
    """Feature validation and packing jobs that queue in ``center``."""
    from histopilot.application.feature_packs import FeaturePackService

    return FeaturePackService(store, filesystem, task_center=center.client)


def run_pack(packing, job, center):
    """Run a queued packing task's worker in this process as the runner would, then
    conclude the task with the worker's outcome."""
    from histopilot.workers.pack_features import run_job

    task_id = job["taskId"]
    center.start(task_id)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("HISTOPILOT_TASK_MANAGED", "1")
        patch.setenv("HISTOPILOT_TASK_ID", task_id)
        patch.setenv("HISTOPILOT_TASK_ATTEMPT", str(center.task(task_id)["attempt"]))
        result = run_job(Path(packing.folder) / job["id"] / "plan.json")
    succeeded = result["state"] == "succeeded"
    center.finish(task_id, "succeeded" if succeeded else "failed", returncode=int(not succeeded))
    return result


def launches(center):
    """How many compute workers were queued: new tasks plus requeued attempts of old ones.

    The Task Center analogue of counting a fake executor's launch calls; a resumed job
    requeues its task instead of starting another one.
    """
    return sum(task["attempt"] for task in compute_tasks(center))
