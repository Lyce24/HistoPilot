"""Drive the Task Center tasks of MIL batches and feature jobs the way the runner would."""

from histopilot.taskcenter.model import LIVE


def task_ids(center):
    """Every task in the store (fixtures may queue feature jobs), to show nothing was added."""
    return [task["id"] for task in center.tasks()]


def batch_tasks(center, identity, kind=None):
    """The tasks queued for one training batch: one fold per run, then the final collection."""
    return center.tasks(group=identity, kind=kind)


def attempts(center, identity):
    """Each task of the batch with its attempt, to tell a replay from another enqueue."""
    return sorted((task["id"], task["attempt"]) for task in batch_tasks(center, identity))


def lose_batch(center, identity, *, completed=(), state="interrupted"):
    """End every live task of the batch as ``state``, by default as a lost runner leaves it.

    Folds of the runs in ``completed`` finished first and end succeeded. Afterwards no task
    can move the batch any more, so it reads as its saved runs say.
    """
    for task in batch_tasks(center, identity):
        if task["state"] not in LIVE:
            continue
        run_id = (task.get("adapterData") or {}).get("runId")
        outcome = "succeeded" if run_id is not None and run_id in completed else state
        if outcome == "interrupted":
            center.finish(task["id"], outcome, returncode=None, reason="lost")
        else:
            center.finish(task["id"], outcome, returncode=0 if outcome == "succeeded" else 1)


def run_managed(center, task, monkeypatch, worker):
    """Run ``worker`` in this process with the environment the runner gives a task.

    The task is recorded as started first and left running: the caller concludes it.
    """
    center.start(task["id"])
    monkeypatch.setenv("HISTOPILOT_TASK_MANAGED", "1")
    monkeypatch.setenv("HISTOPILOT_TASK_ID", task["id"])
    monkeypatch.setenv("HISTOPILOT_TASK_ATTEMPT", str(center.task(task["id"])["attempt"]))
    try:
        return worker()
    finally:
        for name in ("HISTOPILOT_TASK_MANAGED", "HISTOPILOT_TASK_ID", "HISTOPILOT_TASK_ATTEMPT"):
            monkeypatch.delenv(name, raising=False)
