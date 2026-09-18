"""Persistent CPU-only feature validation/packing worker with atomic completion receipts.

Run as ``python -m histopilot.workers.pack_features /absolute/job/plan.json``.
The absolute script entry point also works from an unrelated working directory.
"""

import json
import os
import signal
import stat
import sys
import time
import traceback
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from datetime import UTC, datetime
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from histopilot.application.features import FeatureService
from histopilot.storage.filesystem import LocalFilesystem
from histopilot.storage.project_lock import _reject_symlink_components
from histopilot.storage.scientific import ScientificStore
from histopilot.workers.packing_process import (
    output_lock,
    process_metadata,
    registry_lock,
    write_json,
)
from histopilot.workers.resource_reservation import preparation_resources, reserve_preparation


def _now() -> str:
    return datetime.now(UTC).isoformat()


def run_job(plan_path: Path) -> dict:
    """A second launch of the same plan cannot race progress or completion receipts."""
    with output_lock(f"job:{Path(plan_path).resolve()}"):
        return _run_job(plan_path)


def _run_job(plan_path: Path) -> dict:
    """Execute an immutable plan; always save a durable terminal result on normal errors."""
    plan_path = Path(plan_path)
    _reject_symlink_components(plan_path)
    plan = json.loads(ScientificStore._read_file(plan_path, 64 * 1024 * 1024))
    folder = plan_path.parent.resolve(strict=True)
    for key, name in {
        "resultPath": "result.json",
        "progressPath": "progress.json",
        "processPath": "process.json",
        "logPath": "worker.log",
        "cancelPath": "cancelled",
    }.items():
        path = Path(plan[key])
        _reject_symlink_components(path)
        if path != folder / name:
            raise ValueError(f"{key} must be the managed file beside this plan.")
    result_path = Path(plan["resultPath"])
    if result_path.exists():
        # A repeated launch does not rebuild or overwrite a completed immutable artifact.
        return json.loads(ScientificStore._read_file(result_path, 64 * 1024 * 1024))
    write_json(Path(plan["processPath"]), process_metadata())
    log_descriptor = os.open(
        plan["logPath"],
        os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW | os.O_NONBLOCK,
        0o600,
    )
    info = os.fstat(log_descriptor)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        os.close(log_descriptor)
        raise ValueError("Worker log must be a regular file without aliases.")
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    previous_handlers = {}
    # Direct tests may call from another thread; marker-based cancellation still works there.
    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[signum] = signal.signal(signum, stop)
    except ValueError:
        pass

    def cancelled():
        return stopped or Path(plan["cancelPath"]).exists()

    result = {
        "jobId": plan["jobId"],
        "state": "failed",
        "startedAt": _now(),
        "validation": None,
        "artifact": None,
    }
    last_progress, last_stage = 0.0, None

    def progress(value):
        nonlocal last_progress, last_stage
        now = time.monotonic()
        if (
            now - last_progress >= 0.2
            or value.get("stage") != last_stage
            or value.get("percent") == 100
        ):
            write_json(Path(plan["progressPath"]), {**value, "updatedAt": _now()})
            if value.get("stage") != last_stage or value.get("percent") == 100:
                print(
                    f"[{value.get('stage', 'working')}] {value.get('completed', 0)}/{value.get('total', 0)} {value.get('unit', '')}",
                    flush=True,
                )
            last_progress, last_stage = now, value.get("stage")

    try:
        with (
            os.fdopen(log_descriptor, "a", buffering=1) as log,
            redirect_stdout(log),
            redirect_stderr(log),
        ):
            reservation_stack = ExitStack()
            try:
                reservation_stack.enter_context(reserve_preparation(
                    folder, "packing", preparation_resources("packing"), cancelled,
                ))
                from histopilot.storage.pack_import import pack_file_stamps, verify_existing_pack
                from histopilot.storage.packed import (
                    PackingCancelled,
                    build_pack,
                    validate_features,
                )

                print(f"Starting {plan['spec']['action']} job {plan['jobId']}", flush=True)
                if cancelled():
                    raise PackingCancelled("Cancelled before validation.")
                configuration = plan["configuration"]
                job = json.loads(ScientificStore._read_file(folder / "job.json", 8 * 1024 * 1024))
                store = ScientificStore(folder.parent.parent, job["projectId"])
                roots = LocalFilesystem(tuple(Path(root) for root in plan["sourceRoots"]))
                findings = FeatureService(store, roots).verify_binding(configuration)
                if findings:
                    raise ValueError(
                        "Saved feature sources changed: "
                        + "; ".join(item["message"] for item in findings[:10])
                    )
                with ExitStack() as stack:
                    if plan["spec"]["action"] == "pack":
                        output = Path(plan["spec"]["outputPath"])
                        _reject_symlink_components(output)
                        with registry_lock():
                            claim = json.loads(
                                ScientificStore._read_file(Path(plan["claimPath"]), 65536)
                            )
                            if claim.get("jobId") != plan["jobId"]:
                                raise ValueError("Output reservation belongs to another job.")
                            stack.enter_context(output_lock(output))
                        artifact = build_pack(
                            configuration,
                            output,
                            dtype=plan["spec"]["dtype"],
                            progress=progress,
                            cancelled=cancelled,
                        )
                        # build_pack performs payload checksum readback and index validation
                        # before atomically publishing the directory.
                        result["validation"] = artifact["validation"]
                        result["artifact"] = {
                            **artifact,
                            "outputPath": str(output),
                            "jobId": plan["jobId"],
                            "origin": "created",
                            "verification": "validated-source-copy",
                            "packStamps": pack_file_stamps(output),
                        }
                    elif plan["spec"]["action"] == "attach":
                        existing = Path(plan["spec"]["existingPath"])
                        _reject_symlink_components(existing)
                        if not roots._contains(existing.resolve(strict=True)):
                            raise ValueError("Existing pack is outside configured data roots.")
                        artifact = verify_existing_pack(
                            configuration,
                            existing,
                            expected_stamps=plan.get("existingPackStamps"),
                            progress=progress,
                            cancelled=cancelled,
                        )
                        result["validation"] = artifact["validation"]
                        result["artifact"] = {**artifact, "jobId": plan["jobId"]}
                    else:
                        result["validation"] = validate_features(
                            configuration, progress=progress, cancelled=cancelled
                        )
                    result["state"] = "succeeded"
                print("Completed successfully.", flush=True)
            except Exception as error:
                result["state"] = (
                    "cancelled"
                    if cancelled() or type(error).__name__ == "PackingCancelled"
                    else "failed"
                )
                result["error"] = str(error)
                traceback.print_exc()
            finally:
                reservation_stack.close()
                result["finishedAt"] = _now()
                write_json(result_path, result)
                print(f"Final state: {result['state']}", flush=True)
                log.flush()
                os.fsync(log.fileno())
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        # Terminal result is durable before releasing the live-process receipt.
        if result_path.exists():
            Path(plan["processPath"]).unlink(missing_ok=True)
    return result


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python -m histopilot.workers.pack_features PLAN", file=sys.stderr)
        return 2
    try:
        result = run_job(Path(sys.argv[1]))
        return 0 if result["state"] == "succeeded" else 1
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
