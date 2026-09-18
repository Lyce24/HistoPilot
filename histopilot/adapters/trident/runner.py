"""Standalone stdlib supervisor, compatible with TRIDENT's Python 3.10 environment.

Invoke this file by absolute path with a private JSON plan, never with `-m`:
TRIDENT's interpreter need not have HistoPilot or Pydantic installed. The parent
service owns plan validation, tmux persistence and artifact coverage checking.
"""

import json
import os
import signal
import subprocess
import sys
import traceback
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path


def _now():
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017 - worker supports Python 3.10


def _write_result(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _process_identity(pid):
    """Record Linux process identity so PID reuse never implies a live worker."""
    identity = {"pid": pid, "startTicks": None, "bootId": None}
    try:
        # comm (field 2) may contain spaces or parentheses. starttime is field 22.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        identity["startTicks"] = int(fields[19])
        identity["bootId"] = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except (OSError, ValueError, IndexError):
        # A very short command may already have exited; its completion record
        # follows immediately. Unavailable identity must not imply a live PID.
        pass
    return identity


def run_plan(path):
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    result_path = Path(plan["resultPath"])
    cancel_path = Path(plan["cancelPath"]) if plan.get("cancelPath") else None
    process = None
    interrupted = False
    started = _now()

    def stop(signum, _frame):
        nonlocal interrupted
        interrupted = True
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, stop)
    result = {"state": "failed", "exitCode": None, "startedAt": started}
    try:
        commands = [("TRIDENT", plan["command"])]
        if plan.get("validationCommand") is not None:
            commands.append(("Artifact validation", plan["validationCommand"]))
        for phase, command in commands:
            if (
                not isinstance(command, list)
                or not command
                or any(not isinstance(arg, str) or "\x00" in arg for arg in command)
            ):
                raise ValueError(f"{phase} command must be a nonempty argv list")
        if cancel_path and cancel_path.exists():
            result["state"] = "cancelled"
            return 0
        with ExitStack() as resource_stack, Path(plan["logPath"]).open("a", encoding="utf-8", buffering=1) as log:
            reservation = None
            if plan.get("resources"):
                # New plans use the service interpreter; historical standalone
                # plans still run without importing the HistoPilot package.
                sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
                from histopilot.workers.resource_reservation import reserve_preparation

                _write_result(Path(plan["processPath"]), _process_identity(os.getpid()))
                reservation = resource_stack.enter_context(reserve_preparation(
                    Path(path).parent, "extraction", plan["resources"],
                    lambda: interrupted or bool(cancel_path and cancel_path.exists()),
                ))
            for phase, command in commands:
                # A cancellation between stages prevents the verifier from starting.
                if interrupted or (cancel_path and cancel_path.exists()):
                    result["state"] = "cancelled"
                    break
                log.write(f"[{_now()}] Starting {phase} worker\n")
                process = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    shell=False,
                    start_new_session=True,
                    cwd=plan.get("cwd"),
                )
                if reservation is not None:
                    reservation.attach(process.pid)
                if plan.get("processPath"):
                    _write_result(
                        Path(plan["processPath"]),
                        {**_process_identity(process.pid), "phase": phase},
                    )
                # Each phase has its own process group and durable identity.
                while process.poll() is None:
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        pass
                    if interrupted or (cancel_path and cancel_path.exists()):
                        stop(signal.SIGTERM, None)
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            try:
                                os.killpg(process.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                            process.wait()
                        # Descendants can outlive a parent that accepted SIGTERM.
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        break
                if reservation is not None:
                    reservation.finish_child()
                result["exitCode"] = process.returncode
                if phase == "TRIDENT":
                    result["tridentExitCode"] = process.returncode
                result["state"] = (
                    "cancelled"
                    if interrupted or (cancel_path and cancel_path.exists())
                    else "succeeded"
                    if process.returncode == 0
                    else "failed"
                )
                if result["state"] == "failed":
                    result["error"] = (
                        f"{phase} exited with status {process.returncode}; see worker log."
                    )
                log.write(f"[{_now()}] {phase} {result['state']} (exit {process.returncode})\n")
                if result["state"] != "succeeded":
                    break
    except Exception as error:
        result["error"] = str(error)
        if interrupted or (cancel_path and cancel_path.exists()):
            result["state"] = "cancelled"
        traceback.print_exc()
        if process is not None and process.poll() is None:
            stop(signal.SIGTERM, None)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
    finally:
        result["finishedAt"] = _now()
        _write_result(result_path, result)
    return 0 if result["state"] in {"succeeded", "cancelled"} else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: runner.py PLAN.json")
    raise SystemExit(run_plan(sys.argv[1]))
