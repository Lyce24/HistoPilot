"""Task Center process wrapper: start one task and record how it ended.

The runner runs this file by path, isolated (``python -I wrap.py SPAWN EXIT FD``), so it
uses only the standard library and a pinned archive's ``histopilot`` package can never
shadow it. The task command arrives as JSON on stdin (``{"argv": [...], "env": {...}}``);
stdout, stderr and the working directory are the task's, and the task inherits them.

The task runs in its own session exactly as when the runner started it directly: its pid
is the tracked identity and leads its process group, so signals, telemetry and workers
that check their own session are unchanged. The wrapper stays outside that session:

1. it starts the task and writes SPAWN (``{"process", "wrapper"}`` identities), which lets
   a runner that died before recording the start adopt the task;
2. it reports the task pid on file descriptor FD and closes it;
3. it waits for the task and atomically writes EXIT
   (``{"returncode", "signal", "finishedAt", "bootId", "hostSignal"}``), so an exit status
   survives a runner restart. It exits with the task's code (128 + signal for a signalled
   task).

The wrapper survives SIGTERM, SIGINT and SIGHUP once the task runs and records the first
one it received as ``hostSignal``. The runner signals only the task's own session, so such
a signal came from the host (a shutdown, a closed login session): the task was interrupted,
not failed, whatever code it then exited with. ``bootId`` tells a runner after a reboot
that the task ended in an earlier boot. A SIGKILL of the wrapper leaves no EXIT and the
runner treats the exit status as unknown, as before.
"""

import json
import os
import signal
import subprocess
import sys
from datetime import UTC, datetime


def _boot_id() -> str:
    with open("/proc/sys/kernel/random/boot_id") as stream:
        return stream.read().strip()


def _identity(pid: int) -> dict:
    with open(f"/proc/{pid}/stat") as stream:
        fields = stream.read().rsplit(")", 1)[1].split()
    return {"pid": pid, "startTicks": int(fields[19]), "bootId": _boot_id()}


def _write(path: str, value: dict) -> None:
    temporary = f"{path}.{os.getpid()}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main(argv: list[str]) -> int:
    spawn_path, exit_path, report_fd = argv[0], argv[1], int(argv[2])
    report = os.fdopen(report_fd, "w")
    try:
        request = json.loads(sys.stdin.read())
        child = subprocess.Popen(
            request["argv"],
            env=request["env"],
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    except Exception as error:  # the runner reports why the task could not start
        message = f"{type(error).__name__}: {error}".replace("\n", " ")
        report.write(f"error {message}\n")
        report.close()
        return 127
    received = []
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, lambda number, _frame: received.append(int(number)))
    try:
        _write(spawn_path, {"process": _identity(child.pid), "wrapper": _identity(os.getpid())})
    except OSError:
        pass  # the runner records the start itself; this file only helps after a crash
    report.write(f"{child.pid}\n")
    report.close()
    code = child.wait()
    try:
        boot = _boot_id()
    except OSError:
        boot = None
    _write(
        exit_path,
        {
            "returncode": code,
            "signal": -code if code < 0 else None,
            "finishedAt": datetime.now(UTC).isoformat(),
            "bootId": boot,
            "hostSignal": received[0] if received else None,
        },
    )
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
