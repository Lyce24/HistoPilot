"""Persistent archive operations with immutable requests and retry receipts.

New operations are Task Center tasks (kind ``archive``, CPU lane, owner kind
``archive``); operations recorded before, or launched with an injected tmux executor,
keep their tmux session for status, cancel and retry.
"""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

from histopilot.application import task_records
from histopilot.application.operations import _now, permitted_path
from histopilot.storage.project_lock import (
    StorageError,
    _reject_symlink_components,
    ensure_managed_directory,
    writer_lock,
)
from histopilot.storage.scientific import ScientificStore
from histopilot.taskcenter import ids
from histopilot.workers.packing_process import TmuxScriptExecutor, live_process, write_json

WORKER = Path(__file__).parents[1] / "workers" / "portability.py"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_STATUSES = {"starting", "queued", "running", "cancelling"}


class PortabilityExecutor(TmuxScriptExecutor):
    label = "portability"
    append_output = True


class _ProjectRef:
    """What ``task_records.owner`` needs from a store: the project id and folder."""

    def __init__(self, project_id, folder):
        self.project_id, self.folder = project_id, Path(folder)


class PortabilityJobs:
    def __init__(
        self, projects, filesystem, executor=None, *, execution_mode=None, task_center=None
    ):
        self.projects, self.filesystem = projects, filesystem
        self.folder = projects.database.workspace / "portability-jobs"
        # An injected executor keeps its caller on the tmux path unless a mode is named.
        self._mode = execution_mode or ("tmux" if executor is not None else None)
        self.executor = executor or PortabilityExecutor()
        self.tasks = task_records.TaskCenterAccess(task_center)

    @property
    def mode(self) -> str:
        """How new records launch (resolved per call, so a long-lived service follows it)."""
        return self._mode or task_records.default_execution_mode()

    @property
    def managed(self):
        return self.mode == task_records.TASK_CENTER

    def _folder(self, job_id):
        if not re.fullmatch(r"portability-[a-f0-9]{64}", job_id):
            raise StorageError("Archive operation not found.", "PORTABILITY_NOT_FOUND", 404)
        folder = self.folder / job_id
        _reject_symlink_components(folder)
        return folder

    def _enqueue(self, folder, state, plan):
        """Queue the operation; a retry starts a new attempt of the same task."""
        request = plan["request"]
        project = _ProjectRef(plan["projectId"], plan["projectPath"])
        task = {
            "id": state["taskId"],
            "kind": "archive",
            "adapter": "archive",
            "title": f"Archive · {request['action']} · {Path(request['archivePath']).name}"[:200],
            "group": {"kind": "archive", "id": state["id"]},
            "labels": {
                "recordKind": "archive",
                "recordId": state["id"],
                "projectId": plan["projectId"],
                "action": request["action"],
            },
            "exclusiveKey": "archive-project:"
            + hashlib.sha256(str(plan["projectPath"]).encode()).hexdigest(),
            "request": {
                "lane": "cpu",
                "cpuThreads": 1,
                "dataWorkers": 0,
                "ramGb": 2.0,
                "graceSeconds": 30,
            },
            "command": {
                "argv": [sys.executable, "-u", str(WORKER), str(folder / "plan.json")],
                "cwd": str(REPOSITORY_ROOT),
                "env": {"HISTOPILOT_TASK_MANAGED": "1", "PYTHONDONTWRITEBYTECODE": "1"},
                "log": str(folder / "worker.log"),
                "progress": str(folder / "progress.json"),
                "result": str(folder / "state.json"),
            },
            "adapterData": {
                "portabilityFolder": str(folder),
                "jobId": state["id"],
                "action": request["action"],
            },
        }
        owner = task_records.owner(
            "archive",
            state["id"],
            task["title"],
            project,
            {"recordKind": "archive"},
        )
        task_records.enqueue(self.tasks, owner, [task])

    def _launch(self, folder, state):
        try:
            self.executor.launch(
                state["sessionName"],
                WORKER,
                folder / "plan.json",
            )
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            # The worker owns state.json after launch. An acknowledgement error
            # is a separate diagnostic and cannot overwrite its terminal receipt.
            write_json(
                folder / "launch-error.json",
                {
                    "at": _now(),
                    "error": f"Launch acknowledgement unavailable: {error}",
                },
            )

    def submit(self, identity, request):
        document, project = self.projects._load(identity)
        values = request.model_dump(mode="json")
        archive = permitted_path(
            self.projects.storage, request.archivePath, existing=request.action != "export"
        )
        if request.action == "restore":
            if not request.destinationPath:
                raise StorageError("Choose a restore destination.", "PORTABILITY_INVALID", 422)
            permitted_path(self.projects.storage, request.destinationPath)
        elif request.destinationPath is not None:
            raise StorageError(
                "Only restore accepts a destination folder.", "PORTABILITY_INVALID", 422
            )
        if request.action == "export" and archive.is_relative_to(project):
            raise StorageError(
                "Save the archive outside the project folder.", "PORTABILITY_INVALID", 422
            )
        if not self.managed and not self.executor.available():
            raise StorageError(
                "tmux is required for durable archive operations.", "TMUX_UNAVAILABLE", 422
            )
        identity_hash = hashlib.sha256(f"{identity}:{request.operationId}".encode()).hexdigest()
        folder = self._folder(f"portability-{identity_hash}")
        ensure_managed_directory(self.folder)
        with writer_lock(self.folder, timeout=5):
            if folder.exists():
                previous = json.loads(ScientificStore._read_file(folder / "plan.json", 1024 * 1024))
                if previous["request"] != values or previous["projectId"] != identity:
                    raise StorageError(
                        "Operation ID belongs to a different archive request.", "OPERATION_CONFLICT"
                    )
                return self.get(identity, folder.name)
            comparable = {key: value for key, value in values.items() if key != "operationId"}
            # Refresh/reconnect may generate a fresh operation ID. Coalesce the
            # exact same active request durably instead of launching another copy.
            for prior in self.folder.glob("portability-*/plan.json"):
                previous = json.loads(ScientificStore._read_file(prior, 1024 * 1024))
                if (
                    previous.get("projectId") == identity
                    and {
                        key: value
                        for key, value in previous["request"].items()
                        if key != "operationId"
                    }
                    == comparable
                ):
                    current = self.get(identity, prior.parent.name)
                    if current["status"] in ACTIVE_STATUSES:
                        return current
            ensure_managed_directory(folder)
            plan = {
                "jobId": folder.name,
                "projectId": identity,
                "projectPath": str(project),
                "storageRoots": [str(root) for root in self.projects.storage.roots],
                "sourceRoots": [str(root) for root in self.filesystem.roots],
                "request": values,
            }
            state = {
                "id": folder.name,
                "projectId": identity,
                "action": request.action,
                "status": "starting",
                "createdAt": _now(),
                "sessionName": f"histopilot-archive-{identity_hash[:20]}",
                "logPath": str(folder / "worker.log"),
                "result": None,
                "error": None,
            }
            if self.managed:
                state.update(
                    status="queued",
                    sessionName=None,
                    executionMode=task_records.TASK_CENTER,
                    taskId=ids.task_id("archive", str(folder)),
                    ownerKey=ids.owner_key("archive", folder.name, str(project)),
                )
            write_json(folder / "plan.json", plan)
            write_json(folder / "state.json", state)
            if self.managed:
                try:
                    self._enqueue(folder, state, plan)
                except (StorageError, OSError) as error:
                    state.update(
                        status="failed",
                        error=f"Could not queue the archive operation in the Task Center: {error}",
                    )
                    write_json(folder / "state.json", state)
            else:
                self._launch(folder, state)
        return self.get(identity, folder.name)

    def get(self, identity, job_id):
        self.projects._load(identity)
        folder = self._folder(job_id)
        if not (folder / "state.json").exists():
            raise StorageError("Archive operation not found.", "PORTABILITY_NOT_FOUND", 404)
        state = json.loads(ScientificStore._read_file(folder / "state.json", 64 * 1024 * 1024))
        if state.get("projectId") != identity:
            raise StorageError("Archive operation not found.", "PORTABILITY_NOT_FOUND", 404)
        if (folder / "progress.json").exists():
            state["progress"] = json.loads(
                ScientificStore._read_file(folder / "progress.json", 65536)
            )
        if task_records.managed_record(state):
            return self._task_state(state)
        if state["status"] in {"starting", "running"}:
            if not self.executor.running(state["sessionName"]) and not live_process(folder):
                state = {
                    **state,
                    "status": "interrupted",
                    "error": "The archive worker stopped without a completion record. Inspect its log; submit a new operation to retry.",
                }
            elif (folder / "cancel.requested").exists():
                state = {**state, "status": "cancelling"}
        return state

    def _task_state(self, state):
        """A Task Center operation's status comes from its task, not from processes."""
        (view,) = self.tasks.views([state.get("taskId")])
        state = {**state, "executor": "task-center", "task": task_records.public_view(view)}
        if view is None:
            if state["status"] in ACTIVE_STATUSES:
                state.update(
                    status="interrupted",
                    error="The Task Center has no task for this operation. Retry to run it again.",
                )
            return state
        if view.get("unknown"):
            state["waitingReason"] = task_records.waiting_reason(view)
            return state
        status = task_records.record_state(view)
        if status == "succeeded":
            status = "completed" if state["status"] == "completed" else state["status"]
        state["status"] = status
        if status == "queued":
            # The worker's own reason (the project is busy) says more than the busy backoff.
            state["waitingReason"] = (
                task_records.waiting_reason(view)
                if view.get("held")
                else state.get("waitingReason") or task_records.waiting_reason(view)
            )
        if status in {"failed", "cancelled", "interrupted"} and view.get("error"):
            state["error"] = state.get("error") or view["error"]
        return state

    def cancel(self, identity, job_id):
        folder = self._folder(job_id)
        with writer_lock(self.folder, timeout=5):
            state = self.get(identity, job_id)
            if task_records.managed_record(state):
                if state["status"] in ACTIVE_STATUSES:
                    task = state.get("task") or {}
                    write_json(
                        folder / "cancel.requested",
                        {
                            "requestedAt": _now(),
                            "attempts": {task["id"]: task["attempt"]}
                            if task.get("id") and task.get("attempt")
                            else {},
                        },
                    )
                    try:
                        self.tasks.client.cancel_task(state["taskId"])
                    except (StorageError, OSError):
                        pass  # the worker still stops on the marker
            elif state["status"] in {"starting", "running", "cancelling"}:
                write_json(folder / "cancel.requested", {"requestedAt": _now()})
        return self.get(identity, job_id)

    def retry(self, identity, job_id):
        folder = self._folder(job_id)
        with writer_lock(self.folder, timeout=5):
            state = self.get(identity, job_id)
            if state["status"] in ACTIVE_STATUSES | {"completed"}:
                return state
            if task_records.managed_record(state):
                # A new attempt of the same task; the adapter resets the record first.
                try:
                    self.tasks.client.requeue_task(state["taskId"], reason="retry")
                except (StorageError, OSError) as error:
                    raise StorageError(
                        f"The Task Center cannot retry this operation: {error}",
                        "TASK_CENTER_UNAVAILABLE",
                        503,
                    ) from error
                self.tasks.wake()
                return self.get(identity, job_id)
            if live_process(folder) or self.executor.running(state["sessionName"]):
                return state
            if not self.executor.available():
                raise StorageError(
                    "tmux is required to retry archive operations.", "TMUX_UNAVAILABLE", 422
                )
            (folder / "cancel.requested").unlink(missing_ok=True)
            (folder / "progress.json").unlink(missing_ok=True)
            state.pop("progress", None)
            state.update(
                status="starting",
                error=None,
                result=None,
                updatedAt=_now(),
                attempts=state.get("attempts", 1) + 1,
            )
            write_json(folder / "state.json", state)
            self._launch(folder, state)
        return self.get(identity, job_id)

    def list(self, identity):
        self.projects._load(identity)
        if not self.folder.exists():
            return {"jobs": []}
        _reject_symlink_components(self.folder)
        jobs = []
        for path in sorted(self.folder.glob("portability-*/state.json"), reverse=True):
            state = json.loads(ScientificStore._read_file(path, 64 * 1024 * 1024))
            if state.get("projectId") == identity:
                jobs.append(self.get(identity, path.parent.name))
        return {"jobs": sorted(jobs, key=lambda row: row["createdAt"], reverse=True)[:50]}
