"""Durable, isolated predictor build and test evaluation jobs.

The API never imports Torch. Jobs run as Task Center tasks: the runner admits them and
holds their resource lease. Every job keeps a verified source archive so an interrupted
job can resume safely. Jobs recorded before the Task Center, and archives pinned before
it (their worker declares no ``TASK_CENTER_PROTOCOL``), can no longer run.
"""

import hashlib
import os
import re
import shutil
import signal
from pathlib import Path

from histopilot.adapters.native.runtime import training_runtime
from histopilot.application.task_records import LEGACY_MESSAGE, refuse_legacy
from histopilot.schemas.development import ResourcePolicy
from histopilot.storage.io import content_hash, read_json_bounded, utc_now, write_json_atomic
from histopilot.storage.lifecycle import lifecycle_guard
from histopilot.storage.project_lock import (
    StorageError,
    ensure_managed_directory,
    reject_symlink_components,
    writer_lock,
)
from histopilot.taskcenter import ids
from histopilot.taskcenter.model import LIVE, awaiting_requeue, normalize_command
from histopilot.workers.compute_archive import prepare_compute_archive
from histopilot.workers.training_process import (
    compute_snapshot,
    confirmed_process_alive,
    cpu_slots_per_run,
    host_snapshot,
    owned_processes,
    read_progress,
    stop_owned_processes,
)

ACTIVE_STATUSES = {"queued", "running"}
FINISHED_STATUSES = {"completed", "failed", "cancelled", "interrupted"}
PROTOCOL_PATTERN = re.compile(rb"^TASK_CENTER_PROTOCOL\s*=\s*(\d+)\s*$", re.MULTILINE)
KIND_TITLES = {"refit": "Refit", "evaluation": "Evaluation", "interpretation": "Attention"}
INTERACTIVE_KINDS = frozenset({"evaluation", "interpretation"})
DEFAULT_TASK_RAM_GB = 6.0
EVALUATION_TASK_VRAM_GB = 2.0
DEFAULT_REFIT_VRAM_GB = 4.0


def job_processes(state):
    """Track an isolated worker group even if its parent died before its loaders."""
    return owned_processes(state.get("process"), state.get("processGroupId"), descendants=True)


def archive_protocol(module_path) -> int | None:
    """The Task Center protocol a pinned archive module declares, read without importing it."""
    try:
        match = PROTOCOL_PATTERN.search(Path(module_path).read_bytes())
    except OSError:
        return None
    return int(match.group(1)) if match else None


def refit_vram_gb(plan) -> float:
    """A refit trains every development slide with its fold recipe; size it like a fold."""
    from histopilot.taskcenter import estimator

    try:
        data = plan["data"]
        files = dict(data["featureFiles"])
        rows = [{**row, "partition": "train"} for row in data["memberships"]]
        largest = max(rows, key=lambda row: files[row["slideId"]]["patchCount"])
        # The fold formula needs an evaluation stream; a copy of the largest slide
        # stands in for it, so the training stream is sized from every refit slide.
        proxy = "\0refit-evaluation-proxy"
        files[proxy] = files[largest["slideId"]]
        rows.append({**largest, "slideId": proxy, "partition": "val"})
        workload = estimator.fold_workloads(
            plan.get("effectiveRecipe") or plan["recipe"],
            rows,
            files,
            data.get("loadingPolicy") or "native",
        )
        return round(max(1.0, estimator.gpu_estimate(workload)), 2)
    except (KeyError, TypeError, ValueError, AttributeError):
        return DEFAULT_REFIT_VRAM_GB


def compute_request(plan) -> dict:
    """Task Center admission request for a frozen compute plan; plan resources stay untouched."""
    resources = plan["resources"]
    gpu = bool(resources.get("gpuIds"))
    kind = plan.get("kind")
    ram = float(resources["ramGbPerRun"]) if kind == "interpretation" else DEFAULT_TASK_RAM_GB
    vram = 0.0
    if gpu:
        vram = refit_vram_gb(plan) if kind == "refit" else EVALUATION_TASK_VRAM_GB
    return {
        "lane": "gpu" if gpu else "cpu",
        "cpuThreads": int(resources["cpuThreadsPerRun"]),
        "dataWorkers": int(resources["dataLoaderWorkers"]),
        "ramGb": ram,
        "vramGb": vram,
    }


def compute_priority(plan, owner) -> str:
    """A single evaluation or interpretation is short work someone is waiting for, so it goes
    ahead of queued training (never preempting it); refits and bulk members keep their turn."""
    if plan.get("kind") in INTERACTIVE_KINDS and owner.get("kind") != "evaluation-batch":
        return "interactive"
    return "normal"


def wake_runner(default_client: bool) -> None:
    """The first submission starts the runner; failures only delay queued work.

    Injected clients (tests) never start one. Neither does code running inside a task
    (the coordinator's refits, bulk submission): it may be a pinned archive, and a runner
    started from there would run that old code.
    """
    if not default_client or os.environ.get("HISTOPILOT_TASK_ID"):
        return
    try:
        from histopilot.taskcenter.launcher import ensure_runner

        ensure_runner()
    except Exception:  # never fail an accepted submission because tmux misbehaved
        pass


def submit_task(client, owner, spec, *, reason, active_message, active_code):
    """Enqueue a task, or requeue its finished predecessor with a refreshed command.

    A live task means another launch owns the record; the caller reconciles it.
    """
    existing = client.task(spec["id"])
    if existing is None:
        client.enqueue(owner, [spec])
        return
    if existing["state"] in LIVE:
        raise StorageError(active_message, active_code)
    client.store.set_fields(
        spec["id"],
        title=spec["title"],
        labels=spec.get("labels") or {},
        request=spec["request"],
        command=normalize_command(spec["command"]),
        **({"priority": spec["priority"]} if spec.get("priority") else {}),
    )
    client.store.requeue([spec["id"]], reason=reason, include_succeeded=True)


def task_view(client, task_id):
    """A managed job's task, or ``{"unknown": True}`` when the Task Center cannot answer.

    ``runnerAlive`` is None when the runner lock cannot be probed.
    """
    try:
        task = client.task(task_id) if task_id else None
        if task is None:
            return None
        owner = client.store.owner(task["ownerKey"])
    except (StorageError, OSError) as error:
        return {
            "id": task_id,
            "state": None,
            "unknown": True,
            "error": str(error),
            "runnerAlive": None,
        }
    try:
        runner_alive = bool(client.runner_alive())
    except (StorageError, OSError):
        runner_alive = None
    return {
        "id": task["id"],
        "state": task["state"],
        "attempt": task["attempt"],
        "waitingReason": task["waitingReason"],
        "held": bool(owner and owner["held"]),
        "ownerKey": task["ownerKey"],
        "runnerAlive": runner_alive,
    }


def task_pending(view) -> bool:
    """Whether a managed job may still start or is starting; unknown never reads as stopped."""
    return bool(view) and (view.get("unknown") or view["state"] in LIVE)


def host_gpu_argv(argv) -> list[str]:
    """Run a CPU-lane orchestration command with the host's GPU visibility restored.

    The runner hides every GPU from CPU-lane tasks (``CUDA_VISIBLE_DEVICES=""``). The
    predictor coordinator and bulk submission never use a GPU themselves, but they probe
    the training runtime to plan GPU work for other tasks; hidden GPUs would read as
    ``cudaAvailable: False`` and refuse or silently downgrade that work.
    """
    return [shutil.which("env") or "/usr/bin/env", "-u", "CUDA_VISIBLE_DEVICES", *argv]


class TaskCenterComputeExecutor:
    """Queue compute workers in the machine-wide Task Center."""

    def __init__(self, client=None):
        self._client = client
        self._default_client = client is None

    @property
    def client(self):
        if self._client is None:
            from histopilot.taskcenter.client import default_client

            self._client = default_client()
        return self._client

    def running(self, session):
        task = self.client.by_session(session)
        return bool(task and task["state"] in LIVE)

    def launch(self, session, python, plan, log, *, package_root, task_id, owner, title, frozen):
        folder = Path(plan).parent
        kind = frozen["kind"]
        existing = self.client.task(task_id)
        # A resume keeps the owner the task was queued under (for example a bulk batch,
        # whose members keep their turn), whoever resumes it.
        owner = (existing and self.client.store.owner(existing["ownerKey"])) or owner
        labels = {
            "recordId": frozen["recordId"],
            "computeKind": kind,
            **({"purpose": frozen["purpose"]} if frozen.get("purpose") else {}),
            **({"experimentId": owner["id"]} if owner["kind"] == "experiment" else {}),
        }
        spec = {
            "id": task_id,
            "kind": "compute-job",
            "adapter": "compute-job",
            "title": title,
            "group": {"kind": kind, "id": frozen["recordId"]},
            "sessionName": session,
            "labels": labels,
            "priority": compute_priority(frozen, owner),
            "request": compute_request(frozen),
            "command": {
                "argv": [python, "-u", "-m", "histopilot.workers.compute_job", str(plan)],
                "cwd": str(package_root),
                "env": {"PYTHONDONTWRITEBYTECODE": "1", "HISTOPILOT_TASK_MANAGED": "1"},
                "log": str(log),
                "progress": str(folder / "progress.json"),
                "result": str(folder / "result.json"),
            },
            "adapterData": {
                "computeFolder": str(folder),
                "recordId": frozen["recordId"],
                "kind": kind,
            },
        }
        submit_task(
            self.client,
            owner,
            spec,
            reason="launch",
            active_message="This compute task is already queued or running.",
            active_code="COMPUTE_ACTIVE",
        )
        wake_runner(self._default_client)


class ComputeJobService:
    def __init__(self, store, filesystem=None, *, runtime=None, task_center=None):
        self.store, self.filesystem = store, filesystem
        self.executor = TaskCenterComputeExecutor(task_center)
        self.runtime = runtime or training_runtime

    @property
    def task_center(self):
        return self.executor.client

    def _task_owner(self, record, owner):
        manifest = record["manifest"]
        owner = owner or {
            "kind": manifest["kind"],
            "id": record["id"],
            "title": manifest.get("name") or record["id"],
        }
        return {
            "workspace": None,
            "labels": {},
            **owner,
            "title": str(owner.get("title") or owner["id"])[:200],
            "projectId": self.store.project_id,
            "projectFolder": str(self.store.folder),
        }

    @staticmethod
    def _task_title(record, plan):
        manifest = record["manifest"]
        kind = (
            "Inference"
            if plan.get("purpose") == "inference"
            else KIND_TITLES.get(plan["kind"], "Compute")
        )
        return f"{kind} · {manifest.get('name') or record['id']}"[:200]

    def _task(self, state):
        """The Task Center view of a job (None for pre-Task Center jobs and missing tasks)."""
        if state.get("executor") != "task-center":
            return None
        try:
            client = self.task_center
        except (StorageError, OSError) as error:
            return {
                "id": state.get("taskId"),
                "state": None,
                "unknown": True,
                "error": str(error),
                "runnerAlive": None,
            }
        return task_view(client, state.get("taskId"))

    def folder(self, identity):
        if not re.fullmatch(r"configuration-[a-f0-9]{64}", identity):
            raise StorageError("Invalid compute record identity.", "COMPUTE_NOT_FOUND", 404)
        folder = self.store.folder / "compute-jobs" / identity
        reject_symlink_components(folder)
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
        state = read_json_bounded(folder / "state.json")
        if state.get("recordId") != identity or state.get("status") not in {
            "queued",
            "running",
            "completed",
            "failed",
            "cancelled",
            "interrupted",
        }:
            raise StorageError("Compute state is invalid.", "COMPUTE_STATE_INVALID")
        plan = read_json_bounded(folder / "plan.json")
        if content_hash(plan) != state.get("planHash"):
            raise StorageError("The saved execution plan changed.", "COMPUTE_PLAN_CHANGED")
        cancellation_requested = (folder / "cancel.requested").exists()
        managed = state.get("executor") == "task-center"
        # A job from before the Task Center is never probed: nothing runs it any more.
        live_processes = job_processes(state) if managed else []
        live = bool(live_processes)
        task = self._task(state)
        stopped = {
            "status": "cancelled" if cancellation_requested else "interrupted",
            "error": "The compute worker stopped after cancellation was requested."
            if cancellation_requested
            else "The compute worker stopped. Resume to continue from saved work.",
        }
        if not managed:
            if state["status"] in ACTIVE_STATUSES:
                state = {**state, "status": "interrupted", "error": LEGACY_MESSAGE}
        elif state["status"] in ACTIVE_STATUSES and not live:
            if not task_pending(task):
                # The worker may have recorded its outcome and exited since the first read.
                latest = read_json_bounded(folder / "state.json")
                if (
                    latest.get("recordId") == identity
                    and latest.get("planHash") == state["planHash"]
                    and latest.get("status") in FINISHED_STATUSES
                ):
                    state = latest
                    live_processes = job_processes(state)
                    live = bool(live_processes)
                    if live:
                        state = {**state, "status": "running"}
                else:
                    state = {**state, **stopped}
        elif live and state["status"] not in ACTIVE_STATUSES:
            # A receipt does not prove its writer has exited.
            state = {**state, "status": "running"}
        if state["status"] == "completed":
            result = read_json_bounded(folder / "result.json")
            if (
                result != state.get("result")
                or result.get("runId") != identity
                or result.get("state") != "succeeded"
            ):
                raise StorageError("Compute result evidence changed.", "COMPUTE_RESULT_CHANGED")
        progress, warning = read_progress(folder / "progress.json")
        queue = {}
        if managed:
            queue["task"] = task
            if task and state["status"] in ACTIVE_STATUSES and not live:
                queue["waitingReason"] = (
                    f"The Task Center is unavailable: {task['error']}"
                    if task.get("unknown")
                    else task["waitingReason"]
                )
        return {
            **state,
            "executor": "task-center" if managed else "tmux",
            **queue,
            "liveProcesses": live_processes,
            "progress": progress,
            **({"progressWarning": warning} if warning else {}),
            "cancellationRequested": cancellation_requested,
        }

    def replay_launch(
        self, identity, operation_id, *, resume=False, resources=None, record_kind=None
    ):
        """Return an accepted launch without rebuilding or revalidating its inputs.

        This performs no new execution. The immutable owning record, saved plan,
        lifecycle and requested action still have to match. New launches and
        resumes fall through to the caller's full scientific preparation.
        """
        with lifecycle_guard(self.store.folder):
            record = self._record(identity)
            if record_kind is not None and record["manifest"].get("kind") != record_kind:
                raise StorageError(
                    "Compute record type does not match this operation.", "COMPUTE_NOT_FOUND", 404
                )
            self.store.lifecycle.assert_document_usable(record)
            folder = self.folder(identity)
            if not (folder / "state.json").exists():
                return None
            with writer_lock(folder):
                prior = read_json_bounded(folder / "state.json")
                operations = prior.get("operations", {})
                if operation_id not in operations:
                    return None
                frozen = read_json_bounded(folder / "plan.json")
                if (
                    content_hash(frozen) != prior.get("planHash")
                    or frozen.get("recordId") != identity
                    or frozen.get("recordContentHash") != record["contentHash"]
                    or frozen.get("projectId") != self.store.project_id
                    or frozen.get("projectFolder") != str(self.store.folder)
                ):
                    raise StorageError("The saved execution plan changed.", "COMPUTE_PLAN_CHANGED")
                if (
                    resources is not None
                    and ResourcePolicy.model_validate(resources).model_dump() != frozen["resources"]
                ):
                    raise StorageError(
                        "This operation belongs to another compute request.", "OPERATION_CONFLICT"
                    )
                actions = prior.get("operationActions", {})
                if operation_id in actions:
                    if type(actions[operation_id]) is not bool or actions[operation_id] != resume:
                        raise StorageError(
                            "This operation belongs to another compute request.",
                            "OPERATION_CONFLICT",
                        )
                else:
                    # Historical receipts hash the unnormalized request plan.
                    # Reconstruct only its known envelope/resource representations;
                    # never edit the saved plan or reinterpret an unknown receipt.
                    base = {
                        key: value
                        for key, value in frozen.items()
                        if key
                        not in {
                            "recordId",
                            "recordContentHash",
                            "projectId",
                            "projectFolder",
                            "runtime",
                            "code",
                        }
                    }
                    defaults = ResourcePolicy().model_dump()
                    resource_versions = [
                        frozen["resources"],
                        {
                            key: defaults[key]
                            if key in defaults and value == defaults[key]
                            else value
                            for key, value in frozen["resources"].items()
                        },
                    ]
                    candidates = [
                        {**base, "resources": values, **envelope}
                        for values in resource_versions
                        for envelope in (
                            {},
                            {"recordId": identity, "recordContentHash": record["contentHash"]},
                        )
                    ]
                    if not any(
                        content_hash({"plan": plan, "resume": resume}) == operations[operation_id]
                        for plan in candidates
                    ):
                        if any(
                            content_hash({"plan": plan, "resume": not resume})
                            == operations[operation_id]
                            for plan in candidates
                        ):
                            raise StorageError(
                                "This operation belongs to another compute request.",
                                "OPERATION_CONFLICT",
                            )
                        return None  # Preserve full validation for other historical shapes.
                return self.status(identity)

    def launch(
        self,
        identity,
        plan,
        operation_id,
        *,
        resume=False,
        task_owner=None,
        task_title=None,
        pinned=None,
    ):
        """Freeze and start one job. ``task_owner`` groups its Task Center task (for example
        under an experiment or an evaluation batch); by default the record owns it.
        ``pinned`` is (code, source root) for work that must run a submitted experiment's
        archived code instead of the live checkout."""
        with lifecycle_guard(self.store.folder):
            record = self._record(identity)
            self.store.lifecycle.assert_document_usable(record)
            folder = self.folder(identity)
            ensure_managed_directory(folder)
            with writer_lock(folder):
                prior = (
                    read_json_bounded(folder / "state.json")
                    if (folder / "state.json").exists()
                    else None
                )
                request_hash = content_hash({"plan": plan, "resume": resume})
                operations = dict(prior.get("operations", {})) if prior else {}
                if operation_id in operations:
                    if operations[operation_id] != request_hash:
                        raise StorageError(
                            "This operation belongs to another compute request.",
                            "OPERATION_CONFLICT",
                        )
                    return self.status(identity)
                if prior and prior.get("executor") != "task-center":
                    refuse_legacy()
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
                    "code": pinned[0] if pinned else compute_snapshot(),
                }
                if prior:
                    original = read_json_bounded(folder / "plan.json")
                    if content_hash(original) != prior["planHash"]:
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
                archive = prepare_compute_archive(
                    folder, frozen["code"], source_root=pinned[1] if pinned else None
                )
                if archive_protocol(archive / "histopilot" / "workers" / "compute_job.py") is None:
                    # Code pinned before the Task Center has a worker that leases itself.
                    refuse_legacy()
                task_id = ids.compute_task_id(str(folder))
                existing = self.task_center.task(task_id)
                if existing is not None and existing["state"] in LIVE:
                    # The previous worker exited but the runner has not recorded it yet.
                    # Rewriting state.json now would hand this launch's state to that
                    # attempt's exit bookkeeping.
                    raise StorageError(
                        "The previous attempt is still stopping. Try again in a moment.",
                        "COMPUTE_ACTIVE",
                    )
                if not prior:
                    write_json_atomic(folder / "plan.json", frozen)
                operations[operation_id] = request_hash
                actions = {
                    **(prior.get("operationActions", {}) if prior else {}),
                    operation_id: resume,
                }
                session = (
                    f"tc-{plan['kind']}-{hashlib.sha256(str(folder).encode()).hexdigest()[:16]}"
                )
                state = {
                    "status": "queued",
                    "recordId": identity,
                    "planHash": content_hash(frozen),
                    "process": None,
                    "sessionName": session,
                    "logPath": str(folder / "worker.log"),
                    "createdAt": prior["createdAt"] if prior else utc_now(),
                    "updatedAt": utc_now(),
                    "operations": operations,
                    "operationActions": actions,
                    "result": None,
                    "error": None,
                    "attempt": prior.get("attempt", 1) + 1 if prior else 1,
                    "executor": "task-center",
                    "taskId": task_id,
                    "taskAttempt": existing["attempt"] + 1 if existing else 1,
                }
                (folder / "cancel.requested").unlink(missing_ok=True)
                write_json_atomic(folder / "state.json", state)
                try:
                    self.executor.launch(
                        session,
                        runtime["python"],
                        folder / "plan.json",
                        folder / "worker.log",
                        package_root=archive,
                        task_id=task_id,
                        owner=self._task_owner(record, task_owner),
                        title=task_title or self._task_title(record, frozen),
                        frozen=frozen,
                    )
                except Exception as error:
                    # A lost acknowledgement can follow a committed enqueue. Do not
                    # overwrite a worker that already began.
                    try:
                        session_running = self.executor.running(session)
                        # A fast worker may publish and exit during this probe.
                        # Its recorded identity/outcome proves launch occurred
                        # even when no process is alive by the time we read it.
                        current = read_json_bounded(folder / "state.json")
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
                    state.update(status="failed", error=str(error), updatedAt=utc_now())
                    write_json_atomic(folder / "state.json", state)
                    raise StorageError(
                        f"Cannot launch the compute worker: {error}", "COMPUTE_LAUNCH_FAILED"
                    ) from error
                return state

    def _awaiting_requeue(self, state) -> bool:
        """A paused or lost managed attempt whose requeue the runner has not carried out yet."""
        if state.get("executor") != "task-center" or not state.get("taskId"):
            return False
        try:
            task = self.task_center.task(state["taskId"])
        except (StorageError, OSError):
            return False
        return bool(task) and awaiting_requeue(task)

    def cancel(self, identity, operation_id=None):
        with lifecycle_guard(self.store.folder):
            state = self.status(identity, include_inactive=True)
            folder = self.folder(identity)
            receipt = folder / f"cancel-{content_hash(operation_id)}.json" if operation_id else None
            if receipt and receipt.exists():
                return state
            if state["status"] != "not_started" and state.get("executor") != "task-center":
                refuse_legacy()
            if state["status"] not in {"queued", "running"} and not self._awaiting_requeue(state):
                return state
            write_json_atomic(
                folder / "cancel.requested", {"at": utc_now(), "operationId": operation_id}
            )
            if receipt:
                write_json_atomic(
                    receipt,
                    {"at": utc_now(), "operationId": operation_id, "attempt": state["attempt"]},
                )
            # The runner drops a pending task, or stops a started worker's leader
            # (SIGTERM, then SIGKILL to its group after the grace period).
            try:
                task = self.task_center.cancel_task(state["taskId"])
            except (StorageError, OSError):
                task = None
            # A task that already finished cannot stop a worker that is still alive.
            signalled = bool(task) and task["state"] in {"stopping", "cancelled"}
            if not signalled:
                for process in state.get("liveProcesses", []):
                    if confirmed_process_alive(process):
                        try:
                            os.kill(process["pid"], signal.SIGTERM)
                        except ProcessLookupError:
                            pass
            process = state.get("process")
            if (
                process
                and state.get("processGroupId") == process["pid"]
                and not confirmed_process_alive(process)
            ):
                stop_owned_processes(process)
            return self.status(identity, include_inactive=True)
