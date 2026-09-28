"""Runner leases interoperate with the legacy registry without side effects on reads."""

import json
import os
import tempfile

import pytest

from histopilot.storage.project_lock import StorageError, writer_lock
from histopilot.taskcenter import leases
from histopilot.workers import train_batch
from histopilot.workers.training_process import process_identity


@pytest.fixture
def registry(tmp_path, monkeypatch):
    folder = tmp_path / "tmp"
    folder.mkdir(mode=0o700)
    monkeypatch.setattr(tempfile, "tempdir", str(folder))
    return folder / f"histopilot-training-{os.getuid()}"


def dead_identity(pid=999_999):
    return {"pid": pid, "startTicks": 1, "bootId": "an-earlier-boot"}


def task(identity="task-1"):
    return {"id": identity, "group": {"kind": "mil-batch", "id": "batch-1"}, "ownerKey": "owner-1"}


def test_task_lease_is_published_in_the_legacy_format_read_and_removed(registry):
    assert leases.read_leases() == []
    me = process_identity()
    name = leases.write_task_lease(task(), me, 0, cpus=3, ram_gb=6.0, runs_per_gpu=5, supervisor=me)
    assert name == f"lease-{me['pid']}.json"
    value = json.loads((registry / name).read_text())
    # Exactly the fields a legacy scheduler of another checkout reads.
    assert value == {
        "process": me,
        "processGroupId": me["pid"],
        "gpu": 0,
        "cpus": 3,
        "ramGb": 6.0,
        "runsPerGpu": 5,
        "batchId": "batch-1",
        "runId": "task-1",
        "kind": "task-center",
        "taskId": "task-1",
        "supervisor": me,
    }
    [read] = leases.read_leases()
    assert read["taskId"] == "task-1" and read["live"] is True and read["gpu"] == 0
    leases.remove_task_lease(name)
    leases.remove_task_lease(name)
    assert leases.read_leases() == []
    with pytest.raises(ValueError):
        leases.remove_task_lease("../state.json")


def test_lease_values_are_clamped_for_legacy_readers(registry):
    me = process_identity()
    name = leases.write_task_lease(
        task(), me, None, cpus=0, ram_gb=0.0, runs_per_gpu=40, supervisor=None
    )
    value = json.loads((registry / name).read_text())
    assert (value["cpus"], value["ramGb"], value["runsPerGpu"]) == (1, 0.1, 16)


def test_reading_is_tolerant_and_never_deletes(registry):
    registry.mkdir(mode=0o700)
    dead = {
        "process": dead_identity(),
        "processGroupId": 999_999,
        "gpu": 0,
        "cpus": 4,
        "ramGb": 8,
        "runsPerGpu": 2,
        "batchId": "legacy",
        "runId": "run-1",
    }
    (registry / "lease-999999.json").write_text(json.dumps(dead))
    me = process_identity()
    preparation = {
        "process": dead_identity(123_456),
        "processGroupId": 123_456,
        "supervisor": me,
        "kind": "extraction",
        "gpu": 0,
        "cpus": 0,
        "ramGb": 0,
        "runsPerGpu": 1,
    }
    (registry / "lease-123456-preparation-1.json").write_text(json.dumps(preparation))
    (registry / "lease-1.json").write_text("{not json")
    (registry / "lease-2.json").write_text(json.dumps({**dead, "process": {"pid": 2}}))
    (registry / "lease-abc.json").write_text("{}")
    found = {item["file"]: item for item in leases.read_leases()}
    assert found["lease-999999.json"]["live"] is False
    assert found["lease-999999.json"]["taskId"] is None
    assert found["lease-123456-preparation-1.json"]["live"] is True  # its supervisor is alive
    assert found["lease-123456-preparation-1.json"]["kind"] == "extraction"
    assert found["lease-1.json"] == {"invalid": True, "file": "lease-1.json"}
    assert found["lease-2.json"] == {"invalid": True, "file": "lease-2.json"}
    assert found["lease-abc.json"] == {"invalid": True, "file": "lease-abc.json"}
    assert len(list(registry.glob("lease-*.json"))) == 5
    foreign = leases.foreign(list(found.values()), {"task-1"})
    assert [item["file"] for item in foreign] == ["lease-123456-preparation-1.json"]
    own = {**found["lease-123456-preparation-1.json"], "taskId": "task-1"}
    assert leases.foreign([own], {"task-1"}) == []


def test_unsafe_registry_is_refused(registry):
    registry.mkdir(mode=0o700)
    registry.chmod(0o777)
    try:
        with pytest.raises(StorageError) as error:
            leases.read_leases()
        assert error.value.code == "TRAINING_REGISTRY_UNSAFE"
    finally:
        registry.chmod(0o700)


def test_the_registry_lock_is_the_legacy_writers_lock(registry):
    me = process_identity()
    with leases.registry_lock() as folder:
        assert folder == registry
        # Legacy schedulers take the same lock around read, spawn and write.
        with pytest.raises(StorageError) as error:
            with writer_lock(registry, timeout=0.05):
                pass
        assert error.value.code == "PROJECT_BUSY"
        with pytest.raises(StorageError):
            with leases.registry_lock(timeout=0.05):
                pass
        # Writes that assume the held lock do not wait for it.
        name = leases.write_task_lease(
            task(), me, 0, cpus=1, ram_gb=1.0, runs_per_gpu=4, supervisor=me, locked=True
        )
        [read] = leases.read_leases()
        assert read["process"] == me and read["processGroupId"] == me["pid"]
        leases.remove_task_lease(name, locked=True)
        assert leases.read_leases() == []
    with writer_lock(registry, timeout=0.05):
        pass


# -- This checkout's legacy batch scheduler, removed with the tmux path ---------------------


@pytest.mark.legacy_tmux
def test_task_lease_is_readable_by_legacy_schedulers_and_removed(registry):
    me = process_identity()
    name = leases.write_task_lease(task(), me, 0, cpus=3, ram_gb=6.0, runs_per_gpu=5, supervisor=me)
    with train_batch._leases() as (_, active):
        assert [item["taskId"] for item in active] == ["task-1"]
        resources = {
            "cpuThreadsPerRun": 1,
            "dataLoaderWorkers": 0,
            "ramGbPerRun": 1.0,
            "gpuIds": [0],
            "runsPerGpu": 8,
        }
        assert train_batch.available_device(resources, active, (64, 64.0)) == (True, 0)
    leases.remove_task_lease(name)
    with train_batch._leases() as (_, active):
        assert active == []
