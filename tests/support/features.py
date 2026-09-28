"""Run feature preparation tasks (packing, extraction, archives) the way the runner does.

A service in Task Center mode only queues its work. These helpers take one queued task
through the steps the runner performs for it: the adapter's ``prepare`` hook, the
running state, the worker with the task's own environment, and the conclusion the
adapter's ``on_exit`` hook decides from the worker's receipt. Workers run in this
process, as the legacy tests ran them inline, so a test stays fast and deterministic.

TRIDENT never runs here: ``finish_extraction`` writes the receipt its runner writes.
"""

import json
import os
import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from histopilot.application.extraction_artifacts import complete_coverage
from histopilot.taskcenter.adapters.archive import ArchiveAdapter
from histopilot.taskcenter.adapters.base import RunnerContext
from histopilot.taskcenter.adapters.extraction import (
    ExtractionAdapter,
    ExtractionValidationAdapter,
)
from histopilot.taskcenter.adapters.packing import PackingAdapter
from histopilot.taskcenter.model import PENDING, TERMINAL, utc_now_iso
from histopilot.workers import pack_features, portability, verify_extraction
from support.task_center import fake_host


def runner_context(store):
    """The context an adapter hook receives from the runner."""
    return RunnerContext(
        store=store,
        now=utc_now_iso,
        settings=store.settings(),
        host=fake_host()(),
        log=lambda _message: None,
    )


def worker_environment(task):
    """What the runner adds to the environment of the worker of ``task`` (``procs.spawn``)."""
    return {
        **(task["command"].get("env") or {}),
        "HISTOPILOT_TASK_ID": task["id"],
        "HISTOPILOT_TASK_ATTEMPT": str(task["attempt"]),
        "HISTOPILOT_TASK_GPU": "",
    }


@contextmanager
def task_environment(task):
    """This process runs with the worker environment of ``task`` inside the block."""
    values = worker_environment(task)
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def begin(store, task_id, adapter):
    """Prepare a queued task and record it running; False when ``prepare`` concluded it."""
    task = store.get(task_id)
    assert task["state"] in PENDING, f"{task_id} is {task['state']}, not waiting to run"
    prepared = adapter.prepare(task, runner_context(store))
    if prepared and "skip" in prepared:
        skip = prepared["skip"]
        assert store.transition(
            task_id,
            from_states=("queued",),
            to_state=skip["state"] if skip["state"] in TERMINAL else "succeeded",
            exit={"reason": skip["exitReason"], "error": skip["error"]},
            error=skip["error"],
            waiting_reason=None,
            detail={"skipped": skip["exitReason"]},
        )
        return False
    started = store.transition(
        task_id,
        from_states=tuple(PENDING | {"starting"}),
        to_state="running",
        started_at=utc_now_iso(),
    )
    assert started, f"{task_id} could not start"
    return True


def conclude(store, task_id, adapter, returncode):
    """Conclude the running attempt as the adapter classifies its exit; returns the outcome.

    A stop the Task Center requested (a cancel) reaches the adapter as the runner passes it.
    Dependents whose dependencies are now met are queued, as the runner's next tick does.
    """
    task = store.get(task_id)
    stop = task["stopRequest"]
    exit = {
        "returncode": returncode,
        "lost": False,
        "signalled": False,
        "killed": False,
        "stopReason": stop,
    }
    decided = adapter.on_exit(task, exit, runner_context(store))
    assert decided["state"] in TERMINAL, (
        f"The runner would {decided['state']} {task_id}; drive a real runner for that"
    )
    status = store.conclude(
        task_id,
        to_state=decided["state"],
        stop_request=stop,
        exit={**exit, "reason": decided["exitReason"], "error": decided["error"]},
        error=decided["error"],
        detail={"exitReason": decided["exitReason"]},
    )
    assert status == "concluded", f"{task_id} could not be concluded: {status}"
    store.promote_ready()
    return decided


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
    """Run a feature job's packing task to its conclusion; returns the worker's receipt."""
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
