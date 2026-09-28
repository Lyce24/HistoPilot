"""Persistent project archive worker; contains no server lifecycle operations.

As a Task Center task (``HISTOPILOT_TASK_MANAGED=1``) an export that finds the project
busy (active jobs, or its lifecycle lock held) records ``queued`` with the reason and
exits 75 (EX_TEMPFAIL); the Task Center requeues it with a backoff until the project is
idle. A signal without a cancel request is recorded as ``interrupted``, never
``cancelled``.
"""

import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from histopilot.application.operations import (
    PortabilityCancelled,
    StudyPortability,
    _now,
    permitted_path,
    restore_archive,
    verify_archive,
)
from histopilot.storage.filesystem import LocalFilesystem
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import ScientificStore
from histopilot.workers.packing_process import output_lock, process_metadata, write_json

BUSY_EXIT = 75  # EX_TEMPFAIL: the project is busy; the Task Center retries later
BUSY_CODES = frozenset({"PORTABILITY_ACTIVE_JOBS", "PROJECT_BUSY"})


def _task_identity() -> dict:
    attempt = os.environ.get("HISTOPILOT_TASK_ATTEMPT", "")
    if not os.environ.get("HISTOPILOT_TASK_ID"):
        return {}
    return {
        "taskId": os.environ["HISTOPILOT_TASK_ID"],
        "taskAttempt": int(attempt) if attempt.isdecimal() else None,
    }


def run(plan_path):
    folder = plan_path.parent
    managed = os.environ.get("HISTOPILOT_TASK_MANAGED") == "1"
    with output_lock(f"portability:{plan_path}"):
        plan = json.loads(ScientificStore._read_file(plan_path, 1024 * 1024))
        state = json.loads(ScientificStore._read_file(folder / "state.json", 64 * 1024 * 1024))
        if state["status"] in {"completed", "failed", "cancelled"}:
            return state
        write_json(folder / "process.json", process_metadata())
        state.update(status="running", updatedAt=_now(), **_task_identity())
        state.pop("waitingReason", None)
        write_json(folder / "state.json", state)
        last_update, last_stage = 0.0, None
        interrupted = False
        previous_handlers = {}

        def stop(_signum, _frame):
            nonlocal interrupted
            interrupted = True

        try:
            for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                previous_handlers[signum] = signal.signal(signum, stop)
        except ValueError:
            pass  # In-process tests may execute the worker on an API thread.

        def progress(value):
            nonlocal last_update, last_stage
            if interrupted or (folder / "cancel.requested").exists():
                raise PortabilityCancelled(
                    "Archive operation cancelled. No partial destination was published."
                )
            current = time.monotonic()
            if current - last_update >= 0.5 or value["stage"] != last_stage:
                write_json(folder / "progress.json", {**value, "updatedAt": _now()})
                print(f"{value['stage']}: {value['completed']}/{value['total']} files", flush=True)
                last_update, last_stage = current, value["stage"]

        try:
            progress({"stage": "preparing", "completed": 0, "total": 1, "file": ""})
            roots = LocalFilesystem(tuple(Path(root) for root in plan["storageRoots"]))
            request = plan["request"]
            path = permitted_path(
                roots, request["archivePath"], existing=request["action"] != "export"
            )
            if request["action"] == "export":
                store = ScientificStore(Path(plan["projectPath"]), plan["projectId"])
                result = StudyPortability(store, roots).export(
                    str(path), progress=progress, operation_id=request["operationId"]
                )
            elif request["action"] == "verify":
                result = {**verify_archive(path, progress=progress), "archivePath": str(path)}
            elif request["action"] == "restore":
                result = restore_archive(
                    str(path),
                    request["destinationPath"],
                    roots,
                    progress=progress,
                    operation_id=request["operationId"],
                )
            else:
                raise ValueError("Unsupported archive operation.")
            state.update(status="completed", result=result, error=None)
        except PortabilityCancelled as error:
            if (folder / "cancel.requested").exists():
                state.update(status="cancelled", error=str(error), result=None)
            else:
                # A signal from the host is not a cancel request; the operation can resume.
                state.update(
                    status="interrupted",
                    error="The archive worker was stopped before finishing. Retry to run it again.",
                    result=None,
                )
        except StorageError as error:
            if managed and error.code in BUSY_CODES:
                print(f"PROJECT_BUSY: {error}", flush=True)
                state.update(status="queued", waitingReason=str(error), error=None, result=None)
            else:
                traceback.print_exc()
                state.update(status="failed", error=str(error), result=None)
        except Exception as error:
            traceback.print_exc()
            state.update(status="failed", error=str(error), result=None)
        finally:
            state["updatedAt"] = _now()
            write_json(folder / "state.json", state)
            (folder / "process.json").unlink(missing_ok=True)
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
        return state


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("Usage: portability.py PLAN.json", file=sys.stderr)
        return 2
    managed = os.environ.get("HISTOPILOT_TASK_MANAGED") == "1"
    try:
        state = run(Path(argv[0]))
    except StorageError as error:
        if managed and error.code == "OUTPUT_BUSY":
            print(f"OUTPUT_BUSY: {error}", flush=True)
            return BUSY_EXIT
        raise
    if managed and state["status"] == "queued":
        return BUSY_EXIT
    return 0 if state["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
