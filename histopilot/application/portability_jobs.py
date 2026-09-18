"""Persistent archive operations with immutable requests and retry receipts."""

import hashlib
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

from histopilot.application.operations import _now, permitted_path
from histopilot.storage.project_lock import (
    StorageError,
    _reject_symlink_components,
    ensure_managed_directory,
    writer_lock,
)
from histopilot.storage.scientific import ScientificStore
from histopilot.workers.packing_process import TmuxPackingExecutor, live_process, write_json


class PortabilityExecutor(TmuxPackingExecutor):
    def launch(self, session, runner, plan):
        subprocess.run(["tmux", "ls"], capture_output=True, timeout=10)
        if self.running(session):
            raise RuntimeError("This portability session already exists.")
        command = shlex.join([sys.executable, "-u", str(runner), str(plan)])
        command += " >> " + shlex.quote(str(plan.parent / "worker.log")) + " 2>&1"
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", session, command],
            capture_output=True,
            check=True,
            timeout=15,
        )


class PortabilityJobs:
    def __init__(self, projects, filesystem, executor=None):
        self.projects, self.filesystem = projects, filesystem
        self.folder = projects.database.workspace / "portability-jobs"
        self.executor = executor or PortabilityExecutor()

    def _folder(self, job_id):
        if not re.fullmatch(r"portability-[a-f0-9]{64}", job_id):
            raise StorageError("Archive operation not found.", "PORTABILITY_NOT_FOUND", 404)
        folder = self.folder / job_id
        _reject_symlink_components(folder)
        return folder

    def _launch(self, folder, state):
        try:
            self.executor.launch(
                state["sessionName"],
                Path(__file__).parents[1] / "workers" / "portability.py",
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
        if not self.executor.available():
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
                    if current["status"] in {"starting", "running", "cancelling"}:
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
            write_json(folder / "plan.json", plan)
            write_json(folder / "state.json", state)
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

    def cancel(self, identity, job_id):
        folder = self._folder(job_id)
        with writer_lock(self.folder, timeout=5):
            state = self.get(identity, job_id)
            if state["status"] in {"starting", "running", "cancelling"}:
                write_json(folder / "cancel.requested", {"requestedAt": _now()})
        return self.get(identity, job_id)

    def retry(self, identity, job_id):
        folder = self._folder(job_id)
        with writer_lock(self.folder, timeout=5):
            state = self.get(identity, job_id)
            if state["status"] in {"starting", "running", "cancelling", "completed"}:
                return state
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
