"""Run a queued task's worker the way the runner does, or leave the receipt it writes.

A service in Task Center mode only queues its work. ``run_task`` takes one queued task
through the runner's steps (``support.task_center.begin``), runs the worker's ``main()`` in
this process with the task's command line and environment, and concludes the task as the
adapter decides from the worker's receipt. In-process workers keep a test fast and
deterministic; ``run_compute_worker`` spawns the compute worker as a subprocess instead.

TRIDENT never runs here: ``finish_extraction`` writes the receipt its runner writes.
"""

import json
import os
import subprocess
import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from histopilot.application.extraction_artifacts import complete_coverage
from histopilot.taskcenter.adapters.archive import ArchiveAdapter
from histopilot.taskcenter.adapters.extraction import (
    ExtractionAdapter,
    ExtractionValidationAdapter,
)
from histopilot.taskcenter.adapters.packing import PackingAdapter
from histopilot.taskcenter.model import PENDING
from histopilot.workers import pack_features, portability, verify_extraction
from support.task_center import begin, conclude, task_environment, worker_environment


@contextmanager
def _argv(values):
    saved = sys.argv
    sys.argv = list(values)
    try:
        yield
    finally:
        sys.argv = saved


def run_task(store, task_id, adapter, module):
    """Run a queued task's worker ``module.main()`` in this process, as the runner would.

    The worker gets the task's command line and environment. Returns the adapter's
    outcome, or None when ``prepare`` concluded the task without running it.
    """
    if not begin(store, task_id, adapter):
        return None
    task = store.get(task_id)
    argv = task["command"]["argv"]
    # [python, -u, script, ...] or [python, -u, -m, module, ...]: what follows is its argv.
    arguments = argv[4:] if argv[2] == "-m" else argv[3:]
    with task_environment(task), _argv([module.__file__, *arguments]):
        returncode = module.main()
    return conclude(store, task_id, adapter, returncode)


def _receipt(path, task_id):
    """The receipt at ``path`` when an attempt of ``task_id`` wrote it, else None.

    That is the attempt just run, or an earlier one whose success made ``prepare`` skip it.
    """
    if not Path(path).exists():
        return None
    value = json.loads(Path(path).read_text())
    return value if value.get("taskId") == task_id else None


def run_pack(store, job):
    """Run a feature job's packing task to its conclusion; returns the worker's receipt.

    ``store`` is the Task Center store: ``task_center.store``, or ``service.tasks.client.store``
    of the feature job service that queued ``job``.
    """
    task_id = job["taskId"]
    run_task(store, task_id, PackingAdapter(), pack_features)
    return _receipt(store.get(task_id)["command"]["result"], task_id)


def run_archive(store, job):
    """Run an archive operation's task to its conclusion; returns the worker's state."""
    run_task(store, job["taskId"], ArchiveAdapter(), portability)
    return json.loads(Path(store.get(job["taskId"])["command"]["result"]).read_text())


def _now():
    return datetime.now(UTC).isoformat()


def finish_extraction(store, job, state="succeeded", **fields):
    """End TRIDENT's attempt with the receipt its managed runner writes.

    The attempt starts as the runner starts it (``prepare`` re-arms the validation of a
    resumed job) unless it already runs. ``fields`` override receipt values.
    """
    task_id = job["taskId"]
    adapter = ExtractionAdapter()
    if store.get(task_id)["state"] in PENDING:
        assert begin(store, task_id, adapter), f"{task_id} was concluded before it started"
    task = store.get(task_id)
    folder = Path(task["adapterData"]["extractionFolder"])
    exit_code = {"succeeded": 0, "failed": 1}.get(state, -15)
    receipt = {
        "state": state,
        "exitCode": exit_code,
        "startedAt": task["startedAt"],
        "managed": True,
        "taskId": task_id,
        "taskAttempt": task["attempt"],
        "tridentExitCode": exit_code,
        "cancelRequested": (folder / "cancelled").exists(),
        "finishedAt": _now(),
        **fields,
    }
    (folder / "result.json").write_text(json.dumps(receipt))
    return conclude(store, task_id, adapter, 0 if state in {"succeeded", "cancelled"} else 1)


def run_validation(store, job):
    """Run the artifact validation worker of an extracted job, as its task runs it."""
    return run_task(
        store, job["validationTaskId"], ExtractionValidationAdapter(), verify_extraction
    )


def finish_validation(store, job, **report):
    """End the validation attempt with a report of ``report`` counts.

    Unless the report says otherwise, it is ``complete`` exactly when the worker would
    say so; the stage checks the counts again itself.
    """
    task_id = job["validationTaskId"]
    adapter = ExtractionValidationAdapter()
    assert begin(store, task_id, adapter), f"{task_id} was concluded before it started"
    task = store.get(task_id)
    folder = Path(task["adapterData"]["extractionFolder"])
    value = {"jobId": job["id"], "taskId": task_id, "taskAttempt": task["attempt"], **report}
    value.setdefault("complete", complete_coverage(value, job["slideCount"]))
    (folder / "validation.json").write_text(json.dumps(value))
    return conclude(store, task_id, adapter, 0 if value["complete"] else 1)


def run_compute_worker(center, task_id, *, timeout=60):
    """Run a queued CPU-lane compute task's worker as the runner spawns it, then record the
    task's conclusion from its exit status."""
    center.start(task_id)
    task = center.task(task_id)
    command = task["command"]
    finished = subprocess.run(
        command["argv"],
        cwd=command["cwd"],
        env={**os.environ, **worker_environment(task), "CUDA_VISIBLE_DEVICES": ""},
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
