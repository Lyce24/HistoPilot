"""Facade used by application services (and pinned coordinator copies) to submit and steer tasks.

Import-light by design: no torch, no FastAPI.
"""

import threading

from histopilot.taskcenter import paths
from histopilot.taskcenter.model import ACTIVE, LIVE, PENDING, STATES, TERMINAL, awaiting_requeue
from histopilot.taskcenter.store import TaskStore


class TaskCenterClient:
    def __init__(self, store: TaskStore | None = None):
        self.store = store if store is not None else TaskStore()

    def enqueue(self, owner: dict, tasks: list[dict], *, operation_id=None) -> dict:
        return self.store.enqueue(owner, tasks, operation_id=operation_id)

    def task(self, task_id: str) -> dict | None:
        return self.store.get(task_id)

    def by_session(self, session_name: str) -> dict | None:
        return self.store.by_session(session_name)

    def _group_tasks(self, kind: str, identity: str, project_folder) -> list[dict]:
        return self.store.list(
            group=(kind, identity), project_folder=str(project_folder), limit=None
        )

    def group(self, kind: str, identity: str, project_folder) -> dict:
        tasks = self._group_tasks(kind, identity, project_folder)
        counts = dict.fromkeys(STATES, 0)
        for task in tasks:
            counts[task["state"]] += 1
        owner = self.store.owner(tasks[0]["ownerKey"]) if tasks else None
        waiting = next((task["waitingReason"] for task in tasks if task["state"] == "queued"), None)
        # A concluded task the runner is about to requeue (a pause, an OOM retry or an
        # auto-resume) still has work ahead of it.
        awaiting = sum(1 for task in tasks if awaiting_requeue(task))
        return {
            "tasks": tasks,
            "live": sum(counts[state] for state in LIVE) + awaiting,
            "awaitingRequeue": awaiting,
            "counts": counts,
            "owner": owner,
            "waitingReason": waiting,
            "held": bool(owner and owner["held"]),
        }

    def cancel_task(self, task_id: str) -> dict | None:
        """Pending tasks are cancelled at once; active ones are stopped by the runner."""
        task = self.store.get(task_id)
        if task is None:
            return None
        # Both steps are state-guarded; a task the runner starts in between is stopped instead.
        if not self.store.cancel_pending([task_id]):
            self.store.request_stop([task_id], "cancel")
        return self.store.get(task_id)

    def cancel_group(self, kind: str, identity: str, project_folder, *, exclude_kinds=()) -> dict:
        tasks = [
            task
            for task in self._group_tasks(kind, identity, project_folder)
            if task["kind"] not in set(exclude_kinds)
        ]
        # Offer every live task to both steps: they are state-guarded, so a task the runner
        # starts after the listing is stopped rather than missed. A concluded task awaiting
        # its requeue gets the cancel too, so the runner cancels it instead of requeueing.
        live = [
            task["id"]
            for task in tasks
            if task["state"] in PENDING or task["state"] in ACTIVE or awaiting_requeue(task)
        ]
        cancelled = self.store.cancel_pending(live)
        stopping = self.store.request_stop([i for i in live if i not in set(cancelled)], "cancel")
        return {"cancelled": cancelled, "stopping": stopping}

    def requeue_task(self, task_id: str, *, reason: str) -> bool:
        return bool(self.store.requeue([task_id], reason=reason))

    def requeue_group(
        self, kind: str, identity: str, project_folder, *, kinds=None, reason: str
    ) -> list[str]:
        """Requeue unsuccessful tasks of ``kinds`` and every finished task that depends on them.

        A dependent (for example a batch's final collection) re-runs even if it succeeded,
        because the work it summarized is about to change.
        """
        tasks = self._group_tasks(kind, identity, project_folder)
        wanted = None if kinds is None else set(kinds)
        selected = [
            task
            for task in tasks
            if (wanted is None or task["kind"] in wanted)
            and task["state"] in TERMINAL - {"succeeded"}
        ]
        requeued = self.store.requeue([task["id"] for task in selected], reason=reason)
        in_group = {task["id"]: task for task in tasks}
        frontier = list(requeued)
        while frontier:
            dependents = [
                task_id
                for task_id in self.store.dependents(frontier)
                if task_id in in_group
                and task_id not in requeued
                and (wanted is None or in_group[task_id]["kind"] in wanted)
            ]
            frontier = self.store.requeue(dependents, reason=reason, include_succeeded=True)
            requeued.extend(frontier)
        return requeued

    def runner_alive(self) -> bool:
        from histopilot.taskcenter.launcher import runner_alive

        return runner_alive(self.store.path.parent / "runner.lock")


_DEFAULT: dict[str, TaskCenterClient] = {}
_DEFAULT_LOCK = threading.Lock()


def default_client() -> TaskCenterClient:
    """One client per resolved store path (the state directory is read at call time)."""
    path = paths.store_path()
    with _DEFAULT_LOCK:
        client = _DEFAULT.get(str(path))
        if client is None:
            client = _DEFAULT[str(path)] = TaskCenterClient(TaskStore(path))
        return client


def _reset_default_client() -> None:
    with _DEFAULT_LOCK:
        _DEFAULT.clear()
