"""Adapters for managed MIL training batches: fold runs and results collection.

Live code, not pinned: it maps task lifecycle events onto the batch's ``state.json``. Every
write holds the batch writer lock with a timeout; a busy lock (``PROJECT_BUSY``) is transient
and the runner retries the hook. Folds are classified from ``result.json``, never from the
exit code alone, because a fold stopped with SIGTERM during fitting exits 0; a receipt counts
only when its attempt exited cleanly and no later ``failure.json`` contradicts it.
"""

import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from histopilot.application.feature_bundles import _hash
from histopilot.storage.project_lock import StorageError, writer_lock
from histopilot.taskcenter import ids
from histopilot.taskcenter.adapters.base import Adapter, AdapterError, outcome
from histopilot.taskcenter.model import LIVE, PLAN_ORDER_SPAN, TERMINAL
from histopilot.workers.training_process import (
    ACTIVE as RUN_ACTIVE,
)
from histopilot.workers.training_process import (
    append_event,
    classify_training_failure,
    gpu_snapshot,
    host_snapshot,
    now,
    read_json,
    save_state,
)

BUSY_EXIT = 75  # managed_collect: the batch output is held by another collector
STATE_CACHE_ENTRIES = 16
RUNTIME_RETRY_SECONDS = 30.0  # first retry of a failed runtime probe, doubling per failure
RUNTIME_RETRY_MAX_SECONDS = 300.0
# The runtime probe runs in a thread (it can take 45 s after a reboot); the runner loop
# waits this long for it, then moves on and asks again after RUNTIME_PENDING_SECONDS.
RUNTIME_WAIT_SECONDS = 0.5
RUNTIME_PENDING_SECONDS = 2.0
FAILURE_REASONS = {"out_of_memory": "oom", "cuda_device_failure": "cuda_failure"}
# attempts.jsonl names a requeue by why the runner requeued the fold.
REQUEUE_ACTIONS = {
    "paused": "pause",
    "oom": "oom-backoff",
    "lost": "auto-resume",
    "interrupted": "auto-resume",
    "cuda_failure": "device-lost",
}
HOST_CHANGED = {
    "severity": "warning",
    "code": "TRAINING_HOST_CHANGED",
    "message": "Host boot or GPU driver/device provenance changed while this batch ran (for "
    "example after a restart or a driver update); every attempt's provenance is retained in "
    "attempts.jsonl.",
}
RUN_PROCESS_FIELDS = (
    "process",
    "processGroupId",
    "gpu",
    "startedAt",
    "finishedAt",
    "exitCode",
    "error",
    "failureCategory",
)


@contextmanager
def _translated():
    """Map storage failures to adapter errors: a busy lock is retried, a broken record is not."""
    try:
        yield
    except AdapterError:
        raise
    except StorageError as error:
        raise AdapterError(str(error), fatal=error.code != "PROJECT_BUSY") from error
    except FileNotFoundError as error:
        raise AdapterError(f"The training batch record is missing: {error}", fatal=True) from error
    except OSError as error:
        raise AdapterError(f"The training batch record is unavailable: {error}") from error
    except (ValueError, KeyError, TypeError) as error:
        raise AdapterError(
            f"The training batch record is invalid: {error!r}", fatal=True
        ) from error


@contextmanager
def _locked(folder: Path):
    with writer_lock(folder, timeout=5):
        yield


def _folder(task: dict) -> Path:
    return Path(task["adapterData"]["batchFolder"])


def _run(state: dict, run_id: str) -> dict:
    for run in state["runs"]:
        if run["id"] == run_id:
            return run
    raise KeyError(f"run {run_id} is not part of this batch")


def _result(folder: Path, run_id: str) -> dict | None:
    """The run's exact success receipt, verbatim, or None.

    The fit writes result.json before the worker's post-fit input check, so a failure.json
    written at or after it contradicts the receipt. Both mtimes come from the same filesystem
    clock, unlike the worker's timestamps on a Windows-backed mount.
    """
    path = folder / "runs" / run_id / "result.json"
    if not path.exists():
        return None
    result = read_json(path)
    if result.get("runId") != run_id or result.get("state") != "succeeded":
        return None
    try:
        failed = (path.parent / "failure.json").stat().st_mtime_ns >= path.stat().st_mtime_ns
    except FileNotFoundError:
        failed = False
    return None if failed else result


def _failure(folder: Path, run_id: str, since: str | None) -> dict:
    path = folder / "runs" / run_id / "failure.json"
    if not path.exists():
        return {}
    failure = read_json(path)
    # A prior attempt's failure is not this process's exit.
    return failure if str(failure.get("at", "")) >= (since or "") else {}


def _cached_state(ctx, folder: Path) -> dict:
    """state.json parsed once per change; admission may ask about many folds per tick."""
    path = folder / "state.json"
    info = os.stat(path)
    stamp = (info.st_mtime_ns, info.st_size, info.st_ino)
    cache = ctx.shared.setdefault("mil.states", {})
    cached = cache.get(str(path))
    if cached is None or cached[0] != stamp:
        cache.pop(str(path), None)
        cached = cache[str(path)] = (stamp, read_json(path))
        while len(cache) > STATE_CACHE_ENTRIES:
            cache.pop(next(iter(cache)))
    return cached[1]


def _requeue_action(task: dict) -> str:
    """Why the runner requeues this fold: its pending intent, else how the attempt ended."""
    reason = (task.get("bookkeeping") or {}).get("reason") or (task.get("exit") or {}).get("reason")
    return REQUEUE_ACTIONS.get(reason, reason or "requeue")


def _host_changed(previous: dict | None, current: dict) -> bool:
    """Boot or GPU driver/device changed since the batch's recorded provenance."""
    if not previous:
        return False
    if (previous.get("host") or {}).get("bootId") != current["host"].get("bootId"):
        return True
    if previous.get("gpuProbeError") or current.get("gpuProbeError"):
        return False  # an unreadable driver is unknown, not changed

    def drivers(provenance):
        return {(gpu.get("uuid"), gpu.get("driverVersion")) for gpu in provenance.get("gpus", [])}

    return drivers(previous) != drivers(current)


def _owner_spec(owner: dict) -> dict:
    return {
        key: owner[key]
        for key in ("kind", "id", "projectId", "projectFolder", "workspace", "title", "labels")
    }


def _live_folds(task: dict, ctx, *, limit=None) -> list[dict]:
    """The batch's fold tasks that can still run; an unreadable store is retried later."""
    try:
        return ctx.store.list(
            states=LIVE,
            group=(task["group"]["kind"], task["group"]["id"]),
            project_folder=task["projectFolder"],
            kinds=("mil-fold",),
            limit=limit,
        )
    except StorageError as error:
        raise AdapterError(f"The Task Center store is unavailable: {error}") from error


class MilFoldAdapter(Adapter):
    """One fold run of a managed batch (``histopilot.workers.managed_fold``)."""

    def __init__(self, *, runtime=None):
        self._runtime = runtime

    # -- admission ---------------------------------------------------------------------

    def prepare(self, task, ctx):
        folder, run_id = _folder(task), task["adapterData"]["runId"]
        with _translated():
            run = _run(_cached_state(ctx, folder), run_id)
            if run["status"] == "completed":
                return {"skip": outcome("succeeded", "already-complete")}
            if run["status"] == "cancelled":
                # Cancelled on its own (a single-run cancel) or with its batch.
                return {"skip": outcome("cancelled", "cancelled")}
            if (folder / "cancel.json").exists():
                self._update(folder, run_id, self._cancelled_before_start)
                return {"skip": outcome("cancelled", "cancelled")}
            if _result(folder, run_id) is not None:
                # A lost runner can leave an exact completed result; adopt only this run,
                # and only when no later failure contradicts it.
                self._update(folder, run_id, self._adopt)
                return {"skip": outcome("succeeded", "already-complete")}
        return self._refine(task, ctx)

    @staticmethod
    def _update(folder: Path, run_id: str, change) -> None:
        with _locked(folder):
            state = read_json(folder / "state.json")
            if change(folder, state, _run(state, run_id)) is not False:
                save_state(folder, state)

    @staticmethod
    def _cancelled_before_start(folder, state, run):
        if run["status"] in {"completed", "cancelled"}:
            return False
        run.update(status="cancelled", finishedAt=now(), error="Cancelled before start.")
        return True

    @staticmethod
    def _adopt(folder, state, run):
        result = _result(folder, run["id"])
        if result is None or run["status"] == "completed":
            return False
        run.update(
            status="completed",
            result=result,
            metrics=result.get("metrics"),
            checkpointPath=result.get("bestCheckpointPath"),
        )
        return True

    def _refine(self, task, ctx):
        """Replace the enqueue-time guess with the estimator's evidence-based request."""
        request = task["request"]
        workload = request.get("workload")
        if not isinstance(workload, dict):
            return None
        try:
            from histopilot.taskcenter import estimator

            key = ("mil.observations", task["projectFolder"])
            if key not in ctx.cache:
                if "mil.measurements" not in ctx.cache:
                    ctx.cache["mil.measurements"] = ctx.store.measurements()
                ctx.cache[key] = estimator.gather_observations(
                    [Path(task["projectFolder"])], measurements=ctx.cache["mil.measurements"]
                )
            value = estimator.estimate(
                workload,
                ctx.cache[key],
                ctx.host,
                cpu_threads=request["cpuThreads"],
                data_workers=request["dataWorkers"],
            )
        except Exception as error:  # an estimate is advisory; the enqueue request stands
            reported = ctx.shared.setdefault("mil.refineErrors", set())
            if task["id"] not in reported:
                reported.add(task["id"])
                ctx.log(f"cannot refine the request of {task['id']}: {error}")
            return None
        vram = float(value["vramGb"]) if request["lane"] == "gpu" else request["vramGb"]
        if task["adapterData"].get("oomRetries"):
            # After an OOM backoff the raised VRAM request is evidence: never lower it.
            vram = max(vram, request["vramGb"])
        refined = {**request, "vramGb": vram, "ramGb": float(value["ramGb"])}
        if (refined["vramGb"], refined["ramGb"]) == (request["vramGb"], request["ramGb"]):
            return None
        return {"request": refined}

    # -- lifecycle ---------------------------------------------------------------------

    def on_started(self, task, identity, gpu, ctx):
        folder, run_id = _folder(task), task["adapterData"]["runId"]
        with _translated(), _locked(folder):
            state = read_json(folder / "state.json")
            run = _run(state, run_id)
            cancelled = run["status"] == "cancelled" or (folder / "cancel.json").exists()
            run.update(
                status="running",
                process=identity,
                processGroupId=identity["pid"],
                gpu=gpu,
                startedAt=task["startedAt"],
                logPath=task["command"]["log"],
            )
            if state["status"] == "queued":
                state["status"] = "running"
            save_state(folder, state)
        if cancelled:
            # Cancelled between admission and start: stop it like any running fold.
            ctx.store.request_stop([task["id"]], "cancel")

    def on_exit(self, task, exit, ctx):
        folder, run_id = _folder(task), task["adapterData"]["runId"]
        code = exit.get("returncode")
        completed = False
        with _translated(), _locked(folder):
            # A cancel may have upgraded a pause while this hook waited for the batch lock.
            stop = self._stop_request(task, ctx) or exit.get("stopReason")
            state = read_json(folder / "state.json")
            run = _run(state, run_id)
            failure = _failure(folder, run_id, task.get("queuedAt") or task.get("startedAt"))
            # A result counts only if this attempt exited cleanly (an adopted or lost process
            # has no observable code) and did not fail afterwards, as the legacy scheduler
            # required; the post-fit input check fails after the fit wrote result.json.
            result = _result(folder, run_id) if code in (0, None) and not failure else None
            finished = {"finishedAt": now(), "exitCode": code}
            if result is not None:
                run.update(
                    status="completed",
                    result=result,
                    metrics=result.get("metrics"),
                    checkpointPath=result.get("bestCheckpointPath"),
                    **finished,
                )
                for key in ("error", "failureCategory"):
                    run.pop(key, None)
                peak = result.get("cudaPeakReservedBytes")
                epochs = result.get("epochsCompleted")
                decision = outcome(
                    "succeeded",
                    "ok",
                    measurement={
                        "peakVramGb": peak / 2**30
                        if type(peak) in (int, float) and peak > 0
                        else None,
                        "epochs": epochs if type(epochs) is int and epochs > 0 else None,
                    },
                )
                completed = True
            elif stop == "cancel" or (folder / "cancel.json").exists():
                # Before pause: a cancel that raced "Stop & hold" must not requeue the fold.
                run.update(status="cancelled", error="Cancelled while running.", **finished)
                decision = outcome("cancelled", "cancelled")
            elif stop == "pause":
                run["status"] = "queued"
                decision = outcome("requeue", "paused")
            elif exit.get("lost"):
                run.update(
                    status="interrupted",
                    error="The training process was lost (for example by a restart). "
                    "Resume to continue from its last checkpoint.",
                    **finished,
                )
                decision = outcome("interrupted", "lost")
            else:
                if not failure and code is None:
                    # An adopted process's exit status cannot be observed.
                    run.update(
                        status="interrupted",
                        error="The training process stopped without a result.",
                        **finished,
                    )
                    decision = outcome("interrupted", "interrupted")
                else:
                    error = (
                        failure.get("error")
                        or f"Training process exited with code {code}. See run.log."
                    )
                    category = failure.get("category") or classify_training_failure(
                        failure.get("error", "")
                    )
                    run.update(status="failed", error=error, failureCategory=category, **finished)
                    append_event(
                        folder / "events.jsonl",
                        {
                            "at": now(),
                            "runId": run_id,
                            "attempt": run.get("attempt"),
                            "exitCode": code,
                            "category": category,
                            "error": error,
                        },
                    )
                    decision = outcome("failed", FAILURE_REASONS.get(category, "error"), error)
            save_state(folder, state)
        if completed:
            try:
                self._schedule_progress(task, ctx)
            except Exception as error:  # partial results are optional; the fold result is not
                ctx.log(f"cannot schedule partial results for {task['id']}: {error}")
        return decision

    @staticmethod
    def _stop_request(task, ctx):
        """The task's current stop request; the runner's listing can predate a cancel."""
        try:
            current = ctx.store.get(task["id"])
        except Exception:  # the snapshot's request still applies
            return None
        return current["stopRequest"] if current else None

    def _schedule_progress(self, task, ctx):
        """Queue one partial-results collection unless one is pending or the final one follows."""
        store, folder = ctx.store, task["adapterData"]["batchFolder"]
        if not any(item["id"] != task["id"] for item in _live_folds(task, ctx)):
            return
        progress_id = ids.collect_task_id(folder, False)
        existing = store.get(progress_id)
        if existing is not None:
            if existing["state"] not in LIVE:
                store.requeue([progress_id], reason="progress", include_succeeded=True)
            return
        final = store.get(ids.collect_task_id(folder, True))
        owner = store.owner(task["ownerKey"])
        if final is None or owner is None:
            return
        command = {
            **final["command"],
            "argv": [item for item in final["command"]["argv"] if item != "--final"],
        }
        store.enqueue(
            _owner_spec(owner),
            [
                {
                    "id": progress_id,
                    "kind": "mil-collect",
                    "adapter": "mil-collect",
                    "title": final["title"].replace("Final results", "Partial results"),
                    "group": final["group"],
                    "planOrder": max(0, final["planOrder"] % PLAN_ORDER_SPAN - 1),
                    "priority": "interactive",
                    "exclusiveKey": final["exclusiveKey"],
                    "labels": final["labels"],
                    "request": final["request"],
                    "command": command,
                    "adapterData": {**final["adapterData"], "final": False},
                }
            ],
        )

    # -- recovery ----------------------------------------------------------------------

    def _versions(self, ctx, python):
        """Package versions of the fold's own interpreter; transient while they are unknown.

        A successful probe holds for the runner's lifetime. The probe runs in a thread, so a
        slow one (for example a cold import after a reboot) never blocks the runner loop for
        its whole timeout; meanwhile the runner keeps the requeue pending. A failed probe is
        retried after a short, growing pause instead of treating the fold as ineligible.
        """
        cache = ctx.shared.setdefault("mil.runtimeVersions", {})
        entry = cache.get(python)
        failures = entry["failures"] if entry else 0
        delay = min(RUNTIME_RETRY_MAX_SECONDS, RUNTIME_RETRY_SECONDS * 2 ** max(0, failures - 1))
        if entry is None or (
            entry["versions"] is None
            and entry["thread"] is None
            and time.monotonic() - entry["at"] > delay
        ):
            refresh = entry is not None
            entry = cache[python] = {
                "versions": None,
                "error": entry["error"] if entry else None,
                "failures": failures,
                "at": time.monotonic(),
                "thread": None,
            }
            thread = threading.Thread(
                target=self._probe,
                args=(ctx, python, refresh, entry),
                name="training-runtime-probe",
                daemon=True,
            )
            entry["thread"] = thread
            thread.start()
        thread = entry["thread"]
        if thread is not None:
            # Every fold of a batch asks while one probe runs: wait for it once per tick.
            waited = ctx.cache.setdefault("mil.runtimeWaits", set())
            if thread not in waited:
                waited.add(thread)
                thread.join(RUNTIME_WAIT_SECONDS)
            if thread.is_alive():
                raise AdapterError(
                    "Checking the training runtime.", retry_after=RUNTIME_PENDING_SECONDS
                )
        if entry["versions"] is None:
            raise AdapterError(f"Cannot check the training runtime yet: {entry['error']}")
        return entry["versions"]

    def _probe(self, ctx, python, refresh, entry) -> None:
        runtime = self._runtime
        if runtime is None:
            from histopilot.adapters.native.runtime import training_runtime as runtime
        versions, error = None, None
        try:
            probed = runtime(python=python, refresh=refresh)
            # Versions are known once the interpreter answered, even if a later
            # requirement marked the runtime unavailable.
            if probed.get("available") or probed.get("versions"):
                versions = probed.get("versions") or {}
            else:
                error = next(
                    (item["message"] for item in probed.get("findings") or []),
                    "the runtime probe failed.",
                )
        except Exception as failure:
            error = str(failure) or type(failure).__name__
        entry.update(
            versions=versions,
            error=error,
            failures=entry["failures"] + 1 if error else 0,
            at=time.monotonic(),
            thread=None,
        )
        if error:
            ctx.log(f"cannot probe the training runtime {python or '(default)'}: {error}")

    def can_requeue(self, task, ctx):
        folder, run_id = _folder(task), task["adapterData"]["runId"]
        try:
            plan = read_json(folder / "plan.json")
            state = read_json(folder / "state.json")
            if state.get("planHash") != _hash(plan) or (folder / "cancel.json").exists():
                return False
            if _run(state, run_id)["status"] in {"completed", "cancelled"}:
                return False
            runtime = plan["runtime"]
        except (StorageError, OSError, ValueError, KeyError, TypeError):
            return False
        # Probe the interpreter the fold runs, not the runner's: the runner's environment
        # need not carry the server's HISTOPILOT_TRAINING_PYTHON.
        return self._versions(ctx, runtime.get("python")) == runtime.get("versions")

    def on_requeue(self, task, ctx):
        folder, run_id = _folder(task), task["adapterData"]["runId"]
        with _translated():
            provenance = self._provenance(ctx)  # before the lock: nvidia-smi can take seconds
            with _locked(folder):
                state = read_json(folder / "state.json")
                run = _run(state, run_id)
                if run["status"] == "completed":
                    return
                if self._stop_request(task, ctx) == "cancel":
                    # A cancel that landed after the runner read this task: the store refuses
                    # the requeue and the runner concludes the task cancelled via on_exit.
                    return
                if run["status"] == "cancelled" or (folder / "cancel.json").exists():
                    # A cancel that raced a pause (or a single-run cancel) wins over a requeue.
                    if run["status"] in RUN_ACTIVE:
                        run.update(
                            status="cancelled", finishedAt=now(), error="Cancelled while running."
                        )
                        save_state(folder, state)
                    raise AdapterError(
                        "Cancellation was requested; this fold will not run again.", fatal=True
                    )
                task_attempt = int(task["attempt"]) + 1
                # A retried requeue (the store's requeue failed, or the runner died right after
                # this hook) finds its attempt already prepared and records it only once.
                if run["status"] != "queued" or run.get("taskAttempt") != task_attempt:
                    self._prepare_attempt(folder, state, run, task, task_attempt, provenance)
        self._rearm_final(task, ctx)

    @staticmethod
    def _prepare_attempt(folder, state, run, task, task_attempt, provenance) -> None:
        for key in RUN_PROCESS_FIELDS:
            run.pop(key, None)
        run.update(
            status="queued", attempt=int(run.get("attempt") or 0) + 1, taskAttempt=task_attempt
        )
        if state["status"] not in RUN_ACTIVE or not any(
            row["status"] == "running" for row in state["runs"]
        ):
            # Terminal after a finished attempt, or nothing left running (held work).
            state["status"] = "queued"
            state.pop("finishedAt", None)
        # Like a manual resume, every new attempt records where it runs.
        append_event(
            folder / "attempts.jsonl",
            {
                "action": _requeue_action(task),
                "runId": run["id"],
                "attempt": run["attempt"],
                "at": now(),
                "planHash": state.get("planHash"),
                "provenance": provenance,
            },
        )
        findings = state.setdefault("findings", [])
        if _host_changed(state.get("provenance"), provenance) and not any(
            item.get("code") == HOST_CHANGED["code"] for item in findings
        ):
            findings.append(dict(HOST_CHANGED))
        save_state(folder, state)

    @staticmethod
    def _provenance(ctx) -> dict:
        """Host and GPU provenance, probed once per tick for every fold requeued in it."""
        if "mil.provenance" not in ctx.cache:
            ctx.cache["mil.provenance"] = {"at": now(), "host": host_snapshot(), **gpu_snapshot()}
        return ctx.cache["mil.provenance"]

    @staticmethod
    def _rearm_final(task, ctx):
        """A fold requeued after its batch was finalized must be collected again.

        This happens when the requeue hook itself was deferred (for example by a busy batch
        lock) and the final collection ran meanwhile; without it the batch never finishes.
        """
        final = ids.collect_task_id(task["adapterData"]["batchFolder"], True)
        try:
            current = ctx.store.get(final)
            if current is None or current["state"] not in TERMINAL:
                return
            ctx.store.requeue([final], reason="fold-requeued", include_succeeded=True)
            # The runner requeues this fold right after this hook; the collection waits for it.
            ctx.store.transition(
                final,
                from_states=("queued",),
                to_state="blocked",
                waiting_reason=None,
                detail={"reason": "fold-requeued"},
            )
        except Exception as error:  # a later resume still re-runs the final collection
            ctx.log(f"cannot re-arm the final results for {task['id']}: {error}")


class MilCollectAdapter(Adapter):
    """Partial and final results collection (``histopilot.workers.managed_collect``)."""

    def prepare(self, task, ctx):
        if task["adapterData"].get("final"):
            self._record_cancelled_folds(task, ctx)
            return None
        # With no fold left the final collection follows and supersedes a partial one.
        if _live_folds(task, ctx, limit=1):
            return None
        return {"skip": outcome("succeeded", "already-complete")}

    @staticmethod
    def _record_cancelled_folds(task, ctx) -> None:
        """Folds cancelled in the Task Center alone keep that status in the batch record.

        Otherwise the collection reads their still-queued runs as interrupted.
        """

        def cancelled():
            try:
                folds = ctx.store.list(
                    states=("cancelled",),
                    group=(task["group"]["kind"], task["group"]["id"]),
                    project_folder=task["projectFolder"],
                    kinds=("mil-fold",),
                    limit=None,
                )
            except StorageError as error:
                raise AdapterError(f"The Task Center store is unavailable: {error}") from error
            return {fold["adapterData"].get("runId") for fold in folds}

        folder = _folder(task)
        with _translated():
            runs = cancelled()
            if not runs or not any(
                run["id"] in runs and run["status"] in RUN_ACTIVE
                for run in _cached_state(ctx, folder)["runs"]
            ):
                return
            with _locked(folder):
                # Again under the lock: a resume may have requeued those folds meanwhile.
                runs = cancelled()
                state = read_json(folder / "state.json")
                at, changed = now(), False
                for run in state["runs"]:
                    if run["id"] in runs and run["status"] in RUN_ACTIVE:
                        error = (
                            "Cancelled before start."
                            if run["status"] == "queued"
                            else "Cancelled while running."
                        )
                        run.update(status="cancelled", finishedAt=at, error=error)
                        changed = True
                if changed:
                    save_state(folder, state)

    @staticmethod
    def _superseded(task, ctx) -> bool:
        """Folds requeued (by a resume) while this final collection ran make it stale.

        Its receipt describes the previous attempt; the requeued collection waits for the
        folds again instead of finalizing a batch that is running.
        """
        if not _live_folds(task, ctx, limit=1):
            return False
        ctx.log(f"final results of {task['id']} were superseded by a resume")
        return True

    @staticmethod
    def _receipt(folder: Path, final: bool, since: str | None) -> dict | None:
        path = folder / "collect-result.json"
        if not path.exists():
            return None
        receipt = read_json(path)
        if bool(receipt.get("final")) != final or str(receipt.get("at") or "") < (since or ""):
            return None
        return receipt

    def on_exit(self, task, exit, ctx):
        folder = _folder(task)
        final = bool(task["adapterData"].get("final"))
        if exit.get("returncode") == BUSY_EXIT:
            return outcome("requeue", "busy")
        if exit.get("stopReason") == "pause":
            return outcome("requeue", "paused")
        if final and self._superseded(task, ctx):
            return outcome("requeue", "busy")
        with _translated():
            # The runner records startedAt only after spawning, so a quick collection can
            # finish earlier; this attempt began when the task was queued.
            receipt = self._receipt(folder, final, task.get("queuedAt") or task.get("startedAt"))
        if receipt is None and exit.get("stopReason") == "cancel":
            return outcome("cancelled", "cancelled")
        if receipt is None and exit.get("lost"):
            return outcome("interrupted", "lost")
        if receipt is None and exit.get("returncode") is None:
            return outcome("interrupted", "interrupted")
        error = (
            receipt.get("error")
            if receipt is not None
            else f"Results collection exited with code {exit.get('returncode')} without a "
            "receipt. See batch.log."
        )
        if not final:
            if error:
                ctx.log(f"partial results collection for {task['id']} failed: {error}")
                return outcome("failed", "error", error)
            return outcome("succeeded", "ok")
        with _translated(), _locked(folder):
            # Again under the batch lock that resume holds while it requeues folds.
            if self._superseded(task, ctx):
                return outcome("requeue", "busy")
            state = read_json(folder / "state.json")
            cancelled = (folder / "cancel.json").exists()
            for run in state["runs"]:
                # Folds whose tasks ended without an exit hook (for example a lost cancel
                # acknowledgement) must not stay active once the batch is final.
                if run["status"] in RUN_ACTIVE:
                    run.update(
                        status="cancelled" if cancelled else "interrupted",
                        error=run.get("error")
                        or ("Cancelled before start." if cancelled else "The run did not finish."),
                    )
            if error:
                state["status"] = "failed"
                state["findings"] = [
                    *[
                        item
                        for item in state.get("findings", [])
                        if item.get("code") != "TRAINING_RESULTS_FAILED"
                    ],
                    {
                        "severity": "error",
                        "code": "TRAINING_RESULTS_FAILED",
                        "message": f"Results could not be collected: {error}",
                    },
                ]
            else:
                state["status"] = receipt["status"]
            state["finishedAt"] = now()
            save_state(folder, state)
        return outcome("failed", "error", error) if error else outcome("succeeded", "ok")

    def can_requeue(self, task, ctx):
        if not task["adapterData"].get("final"):
            return False  # the next finished fold schedules another partial collection
        folder = _folder(task)
        try:
            return read_json(folder / "state.json").get("planHash") == _hash(
                read_json(folder / "plan.json")
            )
        except (StorageError, OSError, ValueError):
            return False
