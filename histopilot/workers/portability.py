"""Persistent project archive worker; contains no server lifecycle operations."""

import json
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
from histopilot.storage.scientific import ScientificStore
from histopilot.workers.packing_process import output_lock, process_metadata, write_json


def run(plan_path):
    folder = plan_path.parent
    with output_lock(f"portability:{plan_path}"):
        plan = json.loads(ScientificStore._read_file(plan_path, 1024 * 1024))
        state = json.loads(ScientificStore._read_file(folder / "state.json", 64 * 1024 * 1024))
        if state["status"] in {"completed", "failed", "cancelled"}:
            return state
        write_json(folder / "process.json", process_metadata())
        state.update(status="running", updatedAt=_now())
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
            state.update(status="cancelled", error=str(error), result=None)
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


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: portability.py PLAN.json")
    raise SystemExit(0 if run(Path(sys.argv[1]))["status"] == "completed" else 1)
