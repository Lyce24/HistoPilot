"""Task Center plumbing shared by every service that queues or reads tasks.

``TaskCenterAccess`` is the one way a service reaches the Task Center: it resolves the
machine's client lazily, or uses a client a test injects, and wakes the runner after a
submission.

A record created as a Task Center task stores ``executionMode: "task-center"`` and its
task ids; its execution state is read from the task store. Records created before the
Task Center ran in tmux sessions. They stay readable, but they can never run again:
``legacy_state`` gives an unfinished one its final reading and ``refuse_legacy`` rejects
launching, resuming, retrying or cancelling it.
"""

from __future__ import annotations

import os

from histopilot.storage.project_lock import StorageError
from histopilot.taskcenter import ids
from histopilot.taskcenter.model import LIVE, awaiting_requeue

TASK_CENTER = "task-center"
LEGACY_CODE = "CREATED_BEFORE_TASK_CENTER"
LEGACY_MESSAGE = (
    "Created before the Task Center, so it can no longer run. "
    "Clone it, or preview it again, to run it in the Task Center."
)


def managed_record(record: dict) -> bool:
    return record.get("executionMode") == TASK_CENTER


def legacy_error() -> StorageError:
    """The refusal for any action that would run, stop or retry a pre-Task Center record."""
    return StorageError(LEGACY_MESSAGE, LEGACY_CODE, 409)


def refuse_legacy() -> None:
    raise legacy_error()


def legacy_state(record: dict, active, *, status_key: str = "state") -> dict:
    """A pre-Task Center record as it now reads: an unfinished one is ``interrupted``.

    Nothing runs it any more, so its tmux session and processes are never inspected.
    """
    if record.get(status_key) not in active:
        return record
    return {**record, status_key: "interrupted", "error": LEGACY_MESSAGE}


class TaskCenterAccess:
    """A lazily resolved Task Center client; an injected one never starts a runner."""

    def __init__(self, client=None):
        self._client = client
        self.default = client is None

    @property
    def client(self):
        if self._client is None:
            from histopilot.taskcenter.client import default_client

            self._client = default_client()
        return self._client

    def wake(self) -> None:
        """The first submission starts the runner; failures only delay queued work.

        Injected clients (tests) never start one. Neither does code running inside a task
        (the coordinator's refits, bulk submission): it may be a pinned archive, and a
        runner started from there would run that old code.
        """
        if not self.default or os.environ.get("HISTOPILOT_TASK_ID"):
            return
        try:
            from histopilot.taskcenter.launcher import ensure_runner

            ensure_runner()
        except Exception:  # never fail an accepted submission because tmux misbehaved
            pass

    def views(self, task_ids: list[str | None]) -> list[dict | None]:
        """One view per id (None for a missing task), or ``{"unknown": True}`` views."""
        try:
            client = self.client
            tasks = [client.task(task_id) if task_id else None for task_id in task_ids]
            owners = {}
            for task in tasks:
                if task and task["ownerKey"] not in owners:
                    owners[task["ownerKey"]] = client.store.owner(task["ownerKey"])
        except (StorageError, OSError) as error:
            return [
                {"id": task_id, "state": None, "unknown": True, "error": str(error)}
                for task_id in task_ids
            ]
        try:
            runner_alive = bool(client.runner_alive())
        except (StorageError, OSError):
            runner_alive = None
        views = []
        for task in tasks:
            if task is None:
                views.append(None)
                continue
            owner = owners.get(task["ownerKey"])
            views.append(
                {
                    "id": task["id"],
                    "kind": task["kind"],
                    "state": task["state"],
                    "attempt": task["attempt"],
                    "waitingReason": task["waitingReason"],
                    "held": bool(owner and owner["held"]),
                    "ownerKey": task["ownerKey"],
                    "runnerAlive": runner_alive,
                    "stopRequest": task["stopRequest"],
                    "error": task["error"],
                    "awaitingRequeue": awaiting_requeue(task),
                }
            )
        return views


def public_view(view: dict | None) -> dict | None:
    """The compute-compatible task view published in record responses."""
    if view is None:
        return None
    keys = ("id", "state", "attempt", "waitingReason", "held", "ownerKey", "runnerAlive")
    if view.get("unknown"):
        return {"id": view.get("id"), "state": None, "unknown": True, "error": view["error"]}
    return {key: view.get(key) for key in keys}


def pending(view: dict | None) -> bool:
    """Queued, blocked or concluded-but-requeueing: work that has not started yet."""
    return bool(view) and (
        view.get("state") in {"blocked", "queued"} or bool(view.get("awaitingRequeue"))
    )


def live(view: dict | None) -> bool:
    """Whether the task may still run; an unknown store never reads as stopped."""
    return bool(view) and (
        bool(view.get("unknown")) or view["state"] in LIVE or bool(view.get("awaitingRequeue"))
    )


def record_state(view: dict | None, *, cancelled_state="cancelled") -> str | None:
    """A record's legacy status word for one task: queued/starting/running/cancelling/
    succeeded/failed/cancelled/interrupted; None when the store cannot answer."""
    if view is None or view.get("unknown"):
        return None
    state = view["state"]
    if pending(view):
        return "queued"
    if state == "starting":
        return "starting"
    if state == "running":
        return "running"
    if state == "stopping":
        return "cancelling" if view.get("stopRequest") == "cancel" else "running"
    if state == "cancelled":
        return cancelled_state
    return state


def waiting_reason(view: dict | None) -> str | None:
    if not view:
        return None
    if view.get("unknown"):
        return f"The Task Center is unavailable: {view.get('error')}"
    if view.get("held") and view["state"] in {"blocked", "queued"}:
        return "On hold in the Task Center"
    if view.get("awaitingRequeue"):
        return "Waiting for the Task Center to run it again"
    if view["state"] == "blocked":
        return "Waiting for the previous task"
    return view.get("waitingReason")


def owner(kind: str, identity: str, title: str, store, labels: dict | None = None) -> dict:
    return {
        "kind": kind,
        "id": identity,
        "title": str(title)[:200],
        "projectId": store.project_id,
        "projectFolder": str(store.folder),
        "workspace": None,
        "labels": labels or {},
    }


def owner_key(kind: str, identity: str, store) -> str:
    return ids.owner_key(kind, identity, str(store.folder))


def enqueue(access: TaskCenterAccess, owner_value: dict, specs: list[dict]) -> None:
    """Enqueue new tasks, or start a new attempt of finished ones with refreshed specs."""
    from histopilot.taskcenter.model import normalize_command

    client = access.client
    fresh = [spec for spec in specs if client.task(spec["id"]) is None]
    if fresh:
        client.enqueue(owner_value, fresh)
    new_ids = {spec["id"] for spec in fresh}
    again = []
    for spec in specs:
        if spec["id"] in new_ids:
            continue
        task = client.task(spec["id"])
        if task["state"] in LIVE:
            raise StorageError("This task is already queued or running.", "TASK_ACTIVE", 409)
        client.store.set_fields(
            spec["id"],
            title=spec["title"],
            labels=spec.get("labels") or {},
            request=spec["request"],
            command=normalize_command(spec["command"]),
        )
        again.append(spec["id"])
    if again:
        client.store.requeue(again, reason="launch", include_succeeded=True)
    access.wake()
