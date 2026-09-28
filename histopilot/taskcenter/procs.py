"""Spawn opaque task processes in private sessions and observe them by verified identity."""

import json
import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

from histopilot.storage.project_lock import (
    StorageError,
    _reject_symlink_components,
    ensure_managed_directory,
)
from histopilot.workers.training_process import (
    confirmed_process_alive,
    owned_processes,
    process_identity,
)

_CLOCK_TICKS = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
RESIDENT_CACHE_SECONDS = 10.0
_resident_cache: dict = {}
WRAPPER = Path(__file__).resolve().with_name("wrap.py")
SPAWN_REPORT_SECONDS = 30.0
MAX_JOURNAL_BYTES = 64 * 1024


def spawn(command: dict, *, lane: str, gpu: int | None, task: dict, journal: dict | None = None):
    """Start a task in its own session; output is appended unbuffered to the task log.

    With ``journal`` (``{"spawn": path, "exit": path}``) the task starts through the
    wrapper (``wrap.py``), which records the start and the exit status in those files and
    returns a ``Spawned`` handle; without it the task is a plain child ``Popen``.
    """
    log = Path(command["log"])
    _reject_symlink_components(log)
    ensure_managed_directory(log.parent)
    env = {
        **os.environ,
        **command.get("env", {}),
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": str(gpu) if gpu is not None else "",
        "HISTOPILOT_TASK_ID": task["id"],
        "HISTOPILOT_TASK_ATTEMPT": str(task["attempt"]),
        "HISTOPILOT_TASK_GPU": str(gpu) if gpu is not None else "",
    }
    descriptor = os.open(log, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        if journal is not None:
            return _spawn_wrapped(command, env, descriptor, journal)
        return subprocess.Popen(
            command["argv"],
            cwd=command["cwd"],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=descriptor,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        os.close(descriptor)


def _spawn_wrapped(command: dict, env: dict, log: int, journal: dict) -> "Spawned":
    for path in (journal["spawn"], journal["exit"]):
        _reject_symlink_components(Path(path))
        ensure_managed_directory(Path(path).parent)
        Path(path).unlink(missing_ok=True)  # never read an earlier attempt's record
    read_end, write_end = os.pipe()
    try:
        # The wrapper's own argv and environment name no task: process listings and
        # checks that look for a task's processes see only the task itself.
        wrapper = subprocess.Popen(
            [
                sys.executable,
                "-I",
                str(WRAPPER),
                str(journal["spawn"]),
                str(journal["exit"]),
                str(write_end),
            ],
            cwd=command["cwd"],
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
            stdin=subprocess.PIPE,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
            pass_fds=(write_end,),
        )
    except BaseException:
        os.close(read_end)
        raise
    finally:
        os.close(write_end)
    with os.fdopen(read_end) as report:
        try:
            wrapper.stdin.write(json.dumps({"argv": command["argv"], "env": env}).encode())
            wrapper.stdin.close()
            ready, _, _ = select.select([report], [], [], SPAWN_REPORT_SECONDS)
            line = report.readline().strip() if ready else ""
        except OSError as error:
            line = f"error {error}"
    if not line.isdecimal():
        wrapper.kill()
        wrapper.wait(timeout=5)
        message = line.removeprefix("error ").strip() or "the task wrapper did not start it"
        raise OSError(message)
    return Spawned(wrapper, int(line), journal["exit"])


class Spawned:
    """A task started through the wrapper: ``pid`` is the task, the wrapper reaps it.

    ``poll()`` and ``wait()`` answer like ``Popen`` from the wrapper's record of the task's
    exit. A wrapper that died without one leaves the handle ``orphaned``: the task is then
    followed by its identity, like a task adopted after a runner restart.
    """

    def __init__(self, wrapper: subprocess.Popen, pid: int, exit_path):
        self.wrapper = wrapper
        self.pid = pid
        self.exit_path = exit_path
        self.returncode = None
        self.record = None  # the wrapper's exit record, once read
        self.orphaned = False
        try:
            self.wrapper_identity = identity(wrapper.pid)
        except (OSError, ValueError, IndexError):
            self.wrapper_identity = None

    def poll(self):
        if self.returncode is not None or self.orphaned:
            return self.returncode
        if self.wrapper.poll() is None:
            return None
        recorded = read_exit(self.exit_path)
        if recorded is None:
            self.orphaned = True
            return None
        self.record = recorded
        self.returncode = recorded["returncode"]
        return self.returncode

    def wait(self, timeout=None):
        self.wrapper.wait(timeout=timeout)
        return self.poll()

    def kill(self) -> None:
        try:
            os.killpg(self.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        if self.wrapper.poll() is None:
            self.wrapper.kill()


def _read_journal(path) -> dict | None:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as stream:
            content = stream.read(MAX_JOURNAL_BYTES + 1)
        value = json.loads(content) if len(content) <= MAX_JOURNAL_BYTES else None
    except (OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def read_exit(path) -> dict | None:
    """The wrapper's exit record ``{"returncode", "signal", "finishedAt", "bootId",
    "hostSignal"}``, if written (records of older wrappers lack the last two)."""
    value = _read_journal(path) if path else None
    if value is None or type(value.get("returncode")) is not int:
        return None
    return value


def read_spawn(path) -> dict | None:
    """The wrapper's start record ``{"process", "wrapper"}``, if written and valid."""
    value = _read_journal(path) if path else None
    if value is None or not _valid(value.get("process")):
        return None
    return value


def identity(pid: int) -> dict:
    return process_identity(pid)


def _valid(value) -> bool:
    return (
        isinstance(value, dict)
        and type(value.get("pid")) is int
        and value["pid"] > 1
        and type(value.get("startTicks")) is int
        and isinstance(value.get("bootId"), str)
    )


def alive(identity: dict) -> bool:
    """Whether any member of the task's private session survives; unreadable reads as alive."""
    if not _valid(identity):
        return False
    try:
        return bool(owned_processes(identity, identity["pid"]))
    except (StorageError, OSError):
        return True


def leader_alive(identity: dict) -> bool:
    if not _valid(identity):
        return False
    try:
        return confirmed_process_alive(identity)
    except (StorageError, OSError):
        return True


def signal_leader(identity: dict, sig: int) -> bool:
    """Signal only the verified leader (pid + start ticks + boot id)."""
    try:
        if not _valid(identity) or not confirmed_process_alive(identity):
            return False
        os.kill(identity["pid"], sig)
        return True
    except (StorageError, ProcessLookupError, PermissionError):
        return False


def members(identity: dict) -> list[int]:
    if not _valid(identity):
        return []
    try:
        return [
            item["pid"] for item in owned_processes(identity, identity["pid"], descendants=True)
        ]
    except (StorageError, OSError):
        return []


def signal_group(identity: dict, sig: int) -> bool:
    """Signal the task's process group while verified members still reserve its id."""
    if not members(identity):
        return False
    try:
        os.killpg(identity["pid"], sig)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def kill_group(identity: dict) -> bool:
    return signal_group(identity, signal.SIGKILL)


def _smaps_private_gb(pid: int) -> float | None:
    """Anonymous memory of one process in GiB.

    File-backed clean pages (memory-mapped feature packs, shared libraries) are left out:
    they are reclaimable page cache that MemAvailable already reports as available, and a
    pack page mapped by one fold alone would otherwise count as that fold's private RAM.
    """
    try:
        text = Path(f"/proc/{pid}/smaps_rollup").read_text()
    except OSError:
        return None
    return _anonymous_gb(text)


def _anonymous_gb(text: str) -> float | None:
    fields = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].endswith(":") and parts[1].isdecimal():
            fields[parts[0][:-1]] = int(parts[1])
    if "Pss_Anon" in fields:
        # Proportional, so a page shared by several processes is counted once in a sum.
        kib = fields["Pss_Anon"] + fields.get("Pss_Shmem", 0)
    elif "Private_Dirty" in fields:  # older kernels have no Pss_Anon
        kib = fields["Private_Dirty"]
    else:
        return None
    return kib / 1024**2


def _group_memory_gb(identity: dict, group: int) -> float | None:
    try:
        pids = [item["pid"] for item in owned_processes(identity, group, descendants=True)]
    except (StorageError, OSError):
        return None
    values = [value for pid in pids if (value := _smaps_private_gb(pid)) is not None]
    return sum(values) if values else None


def private_memory_gb(identity: dict) -> float | None:
    """Anonymous memory of every live member of the task's session, in GiB."""
    return _group_memory_gb(identity, identity["pid"]) if _valid(identity) else None


def lease_resident_gb(lease: dict) -> float | None:
    """Anonymous memory of a lease's live process group in GiB; None when it is unknown.

    ``lease`` is a ``leases.read_leases()`` row, whose process identity is read back from
    its file. Readings are cached briefly because admission asks every tick.
    """
    identity, group = lease.get("process"), lease.get("processGroupId")
    if identity is None and lease.get("file"):
        from histopilot.taskcenter import leases

        try:
            value = leases._read(leases.registry() / lease["file"])
        except (OSError, ValueError, TypeError, AttributeError):
            return None
        identity, group = value["process"], value.get("processGroupId")
    if not _valid(identity):
        return None
    key = (lease.get("file"), identity["pid"], identity["startTicks"], identity["bootId"])
    clock = time.monotonic()
    cached = _resident_cache.get(key)
    if cached is not None and clock - cached[0] < RESIDENT_CACHE_SECONDS:
        return cached[1]
    value = _group_memory_gb(identity, identity["pid"] if group is None else group)
    for item, entry in list(_resident_cache.items()):
        if clock - entry[0] >= RESIDENT_CACHE_SECONDS:
            _resident_cache.pop(item, None)
    _resident_cache[key] = (clock, value)
    return value


def cpu_seconds(identity: dict) -> float | None:
    """CPU time of live members, including the reaped children they waited for."""
    total, seen = 0, False
    for pid in members(identity):
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            # utime, stime, cutime, cstime are fields 14-17 (indices 11-14 after the name).
            total += sum(int(value) for value in fields[11:15])
            seen = True
        except (OSError, ValueError, IndexError):
            continue
    return total / _CLOCK_TICKS if seen else None


# Initializes CUDA on the one visible device through the driver API and allocates 1 MiB;
# needs no torch. Exit 3: no CUDA driver library to probe with.
_CUDA_PROBE = r"""
import ctypes, sys
try:
    cuda = ctypes.CDLL("libcuda.so.1")
except OSError as error:
    print(error)
    sys.exit(3)
def check(name, *args):
    code = getattr(cuda, name)(*args)
    if code:
        text = ctypes.c_char_p()
        cuda.cuGetErrorString(code, ctypes.byref(text))
        print(f"{name}: {(text.value or b'').decode()} ({code})")
        sys.exit(1)
device, context, memory = ctypes.c_int(), ctypes.c_void_p(), ctypes.c_uint64()
check("cuInit", 0)
check("cuDeviceGet", ctypes.byref(device), 0)
check("cuCtxCreate_v2", ctypes.byref(context), 0, device)
check("cuMemAlloc_v2", ctypes.byref(memory), ctypes.c_size_t(1 << 20))
check("cuMemFree_v2", memory)
check("cuCtxDestroy_v2", context)
"""
CUDA_PROBE_SECONDS = 60.0


def cuda_probe(gpu: int, *, timeout: float = CUDA_PROBE_SECONDS) -> tuple[str, str | None]:
    """Whether a fresh process can initialize CUDA on ``gpu`` and allocate memory there.

    Returns ("ok", None), ("failed", reason) or ("unavailable", reason) when this machine
    has no CUDA driver library to probe with.
    """
    env = {
        **os.environ,
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": str(gpu),
    }
    try:
        done = subprocess.run(
            [sys.executable, "-I", "-c", _CUDA_PROBE],
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return "failed", f"CUDA initialization did not finish within {timeout:.0f} s"
    except OSError as error:
        return "unavailable", str(error)
    message = (done.stdout + done.stderr).strip()[-500:] or None
    if done.returncode == 0:
        return "ok", None
    if done.returncode == 3:
        return "unavailable", message
    return "failed", message or f"the CUDA probe exited with code {done.returncode}"
