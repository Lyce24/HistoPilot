"""Disposable Linux process trees exercise ownership without running any models."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from histopilot.storage.project_lock import StorageError
from histopilot.workers import train_batch
from histopilot.workers.packing_process import write_json
from histopilot.workers.training_process import (
    confirmed_process_alive,
    owned_processes,
    process_identity,
    stop_owned_processes,
)


@pytest.fixture
def isolated_worker_tree(tmp_path):
    """A session leader and one child; cleanup always owns this exact session."""
    children = []

    def create(*, ignore_term=False):
        path = tmp_path / f"process-{len(children)}.json"
        ready = path.with_suffix(".ready")
        program = """
import json, os, signal, sys, time
from pathlib import Path
path, ready, ignore_term = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] == "True"
child = os.fork()
if child == 0:
    if ignore_term:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    ready.write_text("ready")
    while True:
        time.sleep(1)
fields = Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()
path.write_text(json.dumps({"pid": os.getpid(), "startTicks": int(fields[19]),
    "bootId": Path("/proc/sys/kernel/random/boot_id").read_text().strip(), "childPid": child}))
while True:
    time.sleep(1)
"""
        process = subprocess.Popen(
            [sys.executable, "-c", program, str(path), str(ready), str(ignore_term)],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        children.append(process)
        deadline = time.monotonic() + 5
        while not ready.exists() or not path.exists():
            if process.poll() is not None or time.monotonic() > deadline:
                raise AssertionError("Disposable process fixture did not start.")
            time.sleep(0.01)
        metadata = json.loads(path.read_text())
        child = metadata.pop("childPid")
        return process, metadata, process_identity(child)

    yield create
    for process in children:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
        process.stderr.close()


def test_orphan_descendants_retain_resources_and_receive_escalated_cancellation(
    isolated_worker_tree, tmp_path, monkeypatch
):
    leader, identity, child = isolated_worker_tree(ignore_term=True)
    unrelated, unrelated_identity, _ = isolated_worker_tree()
    leader.kill()
    leader.wait(timeout=5)
    assert not confirmed_process_alive(identity)
    assert owned_processes(identity, leader.pid) == [child]
    monkeypatch.setattr(train_batch.tempfile, "gettempdir", lambda: str(tmp_path))
    with train_batch._leases() as (registry, active):
        assert active == []
        lease = registry / f"lease-{leader.pid}.json"
        write_json(
            lease,
            {
                "process": identity,
                "processGroupId": leader.pid,
                "cpus": 2,
                "ramGb": 2,
                "gpu": 0,
                "runsPerGpu": 1,
                "batchId": "batch",
                "runId": "fold",
            },
        )
    with train_batch._leases() as (_, active):
        assert len(active) == 1 and active[0]["process"] == identity
        assert not train_batch.available_device(
            {"cpuThreadsPerRun": 1, "dataLoaderWorkers": 0, "ramGbPerRun": 1, "gpuIds": []},
            active,
            (2, 100),
        )[0]
    stop_owned_processes(identity, grace_seconds=0.05)
    assert not confirmed_process_alive(child)
    assert confirmed_process_alive(unrelated_identity)
    assert unrelated.poll() is None
    with train_batch._leases() as (_, active):
        assert active == []
    assert not lease.exists()


@pytest.mark.parametrize("run_id", ["fold", "batch"])
def test_old_training_and_compute_leases_retain_orphan_session(
    isolated_worker_tree, tmp_path, monkeypatch, run_id
):
    leader, identity, child = isolated_worker_tree()
    leader.kill()
    leader.wait(timeout=5)
    monkeypatch.setattr(train_batch.tempfile, "gettempdir", lambda: str(tmp_path))
    with train_batch._leases() as (registry, _):
        write_json(
            registry / f"lease-{leader.pid}.json",
            {"process": identity, "batchId": "batch", "runId": run_id},
        )
    with train_batch._leases() as (_, active):
        assert len(active) == 1
    assert confirmed_process_alive(child)


def test_group_ownership_rejects_prior_boot_or_recycled_leader(isolated_worker_tree):
    leader, identity, _ = isolated_worker_tree()
    assert owned_processes({**identity, "bootId": "previous-boot"}, leader.pid) == []
    assert owned_processes({**identity, "startTicks": identity["startTicks"] - 1}, leader.pid) == []
    leader.kill()
    leader.wait(timeout=5)
    assert owned_processes({**identity, "bootId": "previous-boot"}, leader.pid) == []


@pytest.mark.parametrize("target", ["leader", "child", "boot"])
def test_unreadable_evidence_blocks_reservation_release(
    isolated_worker_tree, monkeypatch, tmp_path, target
):
    leader, identity, child = isolated_worker_tree()
    if target == "child":
        leader.kill()
        leader.wait(timeout=5)
    monkeypatch.setattr(train_batch.tempfile, "gettempdir", lambda: str(tmp_path))
    with train_batch._leases() as (registry, _):
        lease = registry / f"lease-{leader.pid}.json"
        write_json(lease, {"process": identity, "processGroupId": leader.pid})
    blocked = {
        "leader": Path(f"/proc/{leader.pid}/stat"),
        "child": Path(f"/proc/{child['pid']}/stat"),
        "boot": Path("/proc/sys/kernel/random/boot_id"),
    }[target]
    original = Path.read_text

    def read(path, *args, **kwargs):
        if path == blocked:
            raise PermissionError("Injected unreadable ownership evidence")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    with pytest.raises(StorageError, match="Cannot"):
        with train_batch._leases():
            pytest.fail("Unknown ownership must not become available capacity.")
    assert lease.exists()


def test_owned_group_never_includes_callers_nonisolated_peers():
    identity = process_identity()
    assert owned_processes(identity, identity["pid"], descendants=True) == [identity]
