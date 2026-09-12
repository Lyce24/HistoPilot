"""Durable, isolated predictor build and test evaluation jobs.

The API never imports Torch. Workers share the training resource lease registry
and keep a verified source archive so an interrupted job can resume safely.
"""

import os
import re
import shlex
import signal
import subprocess
from pathlib import Path

from histopilot.adapters.native.runtime import training_runtime
from histopilot.application.feature_bundles import _hash
from histopilot.schemas.development import ResourcePolicy
from histopilot.storage.lifecycle import lifecycle_guard
from histopilot.storage.project_lock import (
    StorageError,
    _reject_symlink_components,
    ensure_managed_directory,
    writer_lock,
)
from histopilot.workers.compute_archive import prepare_compute_archive
from histopilot.workers.packing_process import write_json
from histopilot.workers.training_process import (
    TmuxTrainingExecutor,
    compute_snapshot,
    cpu_slots_per_run,
    host_snapshot,
    now,
    process_identity,
    read_json,
    read_progress,
)


def job_processes(state):
    """Track an isolated worker group even if its parent died before its loaders."""
    from histopilot.application.lifecycle import _confirmed_live

    process = state.get("process")
    result = [process] if _confirmed_live(process) else []
    group = state.get("processGroupId")
    if not process or group != process.get("pid"):
        return result
    if process["bootId"] != Path("/proc/sys/kernel/random/boot_id").read_text().strip():
        return result
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal() or int(path.name) == process["pid"]:
            continue
        try:
            if path.stat().st_uid != os.getuid():
                continue
            fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
            if fields[0] != "Z" and int(fields[2]) == group and int(fields[3]) == group:
                identity = process_identity(int(path.name))
                if identity["startTicks"] >= process["startTicks"]:
                    result.append(identity)
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (OSError, ValueError, IndexError) as error:
            raise StorageError(
                "Cannot confirm whether compute worker children stopped.", "COMPUTE_PROCESS_UNKNOWN"
            ) from error
    return result


class TmuxComputeExecutor(TmuxTrainingExecutor):
    def launch(self, session, python, plan, log, *, package_root):
        subprocess.run(["tmux", "ls"], capture_output=True, timeout=10)
        if self.running(session):
            raise StorageError("This compute session already exists.", "COMPUTE_ACTIVE")
        _reject_symlink_components(log)
        command = "cd " + shlex.quote(str(package_root)) + " && "
        command += shlex.join(
            [
                "env",
                "PYTHONDONTWRITEBYTECODE=1",
                python,
                "-u",
                "-m",
                "histopilot.workers.compute_job",
                str(plan),
            ]
        )
        command += " >> " + shlex.quote(str(log)) + " 2>&1"
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", session, command],
            capture_output=True,
            check=True,
            timeout=15,
        )


class ComputeJobService:
    def __init__(self, store, filesystem=None, executor=None, runtime=None):
        self.store, self.filesystem = store, filesystem
        self.executor = executor or TmuxComputeExecutor()
        self.runtime = runtime or training_runtime

    def folder(self, identity):
        if not re.fullmatch(r"configuration-[a-f0-9]{64}", identity):
            raise StorageError("Invalid compute record identity.", "COMPUTE_NOT_FOUND", 404)
        folder = self.store.folder / "compute-jobs" / identity
        _reject_symlink_components(folder)
        return folder

    def _record(self, identity, *, include_inactive=False):
        document = self.store.get_configuration(identity, include_inactive=include_inactive)
        if document["manifest"].get("kind") not in {
            "model-evaluation",
            "predictor-refit",
            "model-interpretation",
        }:
            raise StorageError(
                "Select a predictor refit, model evaluation, or interpretation.",
                "COMPUTE_NOT_FOUND",
                404,
            )
        return document

    def status(self, identity, *, include_inactive=False):
        self._record(identity, include_inactive=include_inactive)
        folder = self.folder(identity)
        if not (folder / "state.json").exists():
            return {"status": "not_started", "result": None}
        state = read_json(folder / "state.json")
        if state.get("recordId") != identity or state.get("status") not in {
            "queued",
            "running",
            "completed",
            "failed",
            "cancelled",
            "interrupted",
        }:
            raise StorageError("Compute state is invalid.", "COMPUTE_STATE_INVALID")
        plan = read_json(folder / "plan.json")
        if _hash(plan) != state.get("planHash"):
            raise StorageError("The saved execution plan changed.", "COMPUTE_PLAN_CHANGED")
        cancellation_requested = (folder / "cancel.requested").exists()
        live_processes = job_processes(state)
        live = bool(live_processes)
        if state["status"] in {"queued", "running"} and not live:
            if not self.executor.running(state["sessionName"]):
                state = {
                    **state,
                    "status": "cancelled" if cancellation_requested else "interrupted",
                    "error": "The compute worker stopped after cancellation was requested."
                    if cancellation_requested
                    else "The compute worker stopped. Resume to continue from saved work.",
                }
        elif live and state["status"] not in {"queued", "running"}:
            # A receipt does not prove its writer has exited.
            state = {**state, "status": "running"}
        if state["status"] == "completed":
            result = read_json(folder / "result.json")
            if (
                result != state.get("result")
                or result.get("runId") != identity
                or result.get("state") != "succeeded"
            ):
                raise StorageError("Compute result evidence changed.", "COMPUTE_RESULT_CHANGED")
        progress, warning = read_progress(folder / "progress.json")
        return {
            **state,
            "liveProcesses": live_processes,
            "progress": progress,
            **({"progressWarning": warning} if warning else {}),
            "cancellationRequested": cancellation_requested,
        }

    def launch(self, identity, plan, operation_id, *, resume=False):
        with lifecycle_guard(self.store.folder):
            record = self._record(identity)
            self.store.lifecycle.assert_document_usable(record)
            folder = self.folder(identity)
            ensure_managed_directory(folder)
            with writer_lock(folder):
                prior = (
                    read_json(folder / "state.json") if (folder / "state.json").exists() else None
                )
                request_hash = _hash({"plan": plan, "resume": resume})
                operations = dict(prior.get("operations", {})) if prior else {}
                if operation_id in operations:
                    if operations[operation_id] != request_hash:
                        raise StorageError(
                            "This operation belongs to another compute request.",
                            "OPERATION_CONFLICT",
                        )
                    return self.status(identity)
                state = self.status(identity)
                if state["status"] in {"queued", "running", "completed"}:
                    raise StorageError("This job is active or already completed.", "COMPUTE_ACTIVE")
                if bool(prior) != resume:
                    raise StorageError(
                        "Resume an existing job; launch a new job only once.",
                        "COMPUTE_RESUME_REQUIRED",
                    )
                runtime = self.runtime()
                if not runtime.get("available"):
                    raise StorageError(
                        "The optional training runtime is unavailable.",
                        "TRAINING_RUNTIME_UNAVAILABLE",
                    )
                resources = ResourcePolicy.model_validate(plan.get("resources", {})).model_dump()
                host = runtime.get("host") or host_snapshot()
                if (
                    cpu_slots_per_run(resources) > host["cpuCount"]
                    or resources["ramGbPerRun"] > host["totalRamGb"]
                ):
                    raise StorageError(
                        "The requested CPU or RAM reservation exceeds this workstation's capacity.",
                        "COMPUTE_RESOURCES_UNAVAILABLE",
                    )
                if resources["gpuIds"] and (
                    not runtime.get("cudaAvailable")
                    or max(resources["gpuIds"]) >= runtime.get("gpuCount", 0)
                ):
                    raise StorageError(
                        "A requested GPU is unavailable.", "TRAINING_GPU_UNAVAILABLE"
                    )
                frozen = {
                    **plan,
                    "resources": resources,
                    "recordId": identity,
                    "recordContentHash": record["contentHash"],
                    "projectId": self.store.project_id,
                    "projectFolder": str(self.store.folder),
                    "runtime": runtime,
                    "code": compute_snapshot(),
                }
                if prior:
                    original = read_json(folder / "plan.json")
                    if _hash(original) != prior["planHash"]:
                        raise StorageError(
                            "The saved execution plan changed.", "COMPUTE_PLAN_CHANGED"
                        )
                    if original["runtime"]["versions"] != runtime["versions"]:
                        raise StorageError(
                            "The compute environment changed; restore its original package versions.",
                            "TRAINING_RUNTIME_CHANGED",
                        )
                    comparable = {key: original[key] for key in plan}
                    if comparable != plan:
                        raise StorageError(
                            "Resume must preserve the original inputs and resource settings.",
                            "COMPUTE_PLAN_CHANGED",
                        )
                    frozen = original
                archive = prepare_compute_archive(folder, frozen["code"])
                if not prior:
                    write_json(folder / "plan.json", frozen)
                operations[operation_id] = request_hash
                session = f"hp-{plan['kind']}-{identity.removeprefix('configuration-')[:16]}"
                state = {
                    "status": "queued",
                    "recordId": identity,
                    "planHash": _hash(frozen),
                    "process": None,
                    "sessionName": session,
                    "logPath": str(folder / "worker.log"),
                    "createdAt": prior["createdAt"] if prior else now(),
                    "updatedAt": now(),
                    "operations": operations,
                    "result": None,
                    "error": None,
                    "attempt": prior.get("attempt", 1) + 1 if prior else 1,
                }
                (folder / "cancel.requested").unlink(missing_ok=True)
                write_json(folder / "state.json", state)
                try:
                    self.executor.launch(
                        session,
                        runtime["python"],
                        folder / "plan.json",
                        folder / "worker.log",
                        package_root=archive,
                    )
                except Exception as error:
                    # A tmux timeout can lose the acknowledgement after creating
                    # the session. Do not overwrite a worker that already began.
                    try:
                        session_running = self.executor.running(session)
                        # A fast worker may publish and exit during this probe.
                        # Its recorded identity/outcome proves launch occurred
                        # even when no process is alive by the time we read it.
                        current = read_json(folder / "state.json")
                        if (
                            session_running
                            or current.get("process") is not None
                            or current["status"] not in {"queued", "running"}
                        ):
                            return current
                    except Exception as inspection_error:
                        raise StorageError(
                            "Launch acknowledgement was lost. Check execution status before retrying.",
                            "COMPUTE_LAUNCH_UNCERTAIN",
                        ) from inspection_error
                    state.update(status="failed", error=str(error), updatedAt=now())
                    write_json(folder / "state.json", state)
                    raise StorageError(
                        f"Cannot launch the compute worker: {error}", "COMPUTE_LAUNCH_FAILED"
                    ) from error
                return state

    def cancel(self, identity, operation_id=None):
        with lifecycle_guard(self.store.folder):
            state = self.status(identity, include_inactive=True)
            folder = self.folder(identity)
            receipt = folder / f"cancel-{_hash(operation_id)}.json" if operation_id else None
            if receipt and receipt.exists():
                return state
            if state["status"] not in {"queued", "running"}:
                return state
            write_json(folder / "cancel.requested", {"at": now(), "operationId": operation_id})
            if receipt:
                write_json(
                    receipt, {"at": now(), "operationId": operation_id, "attempt": state["attempt"]}
                )
            for process in state.get("liveProcesses", []):
                from histopilot.application.lifecycle import _confirmed_live

                if _confirmed_live(process):
                    try:
                        os.kill(process["pid"], signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            return self.status(identity, include_inactive=True)
