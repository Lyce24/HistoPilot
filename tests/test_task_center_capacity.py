"""Capacity resolution and the admission rules of the Task Center runner."""

import json
import mmap
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta

import pytest
from test_task_center_runner import OutOfMemory, center, fake_host, registry

from histopilot.taskcenter import capacity, leases, procs
from histopilot.taskcenter.model import DEFAULT_SETTINGS, merge_settings, normalize_request

__all__ = ["center", "registry"]

SETTLED = "2026-01-01T00:00:00+00:00"  # started long ago: its VRAM shows in live free memory


def host(*, gpus=((24.0, 20.0),), cpus=16, available=64.0, total=128.0):
    return {
        "cpuCount": cpus,
        "totalRamGb": total,
        "availableRamGb": available,
        "gpus": [
            {
                "index": index,
                "uuid": f"GPU-{index}",
                "name": "Test GPU",
                "totalMemoryGb": size,
                "freeMemoryGb": free,
            }
            for index, (size, free) in enumerate(gpus)
        ],
    }


def task(*, gpu=None, resources=None, started=None, **request):
    return {
        "id": "t",
        "request": normalize_request({"lane": "gpu", "cpuThreads": 2, "ramGb": 4.0, **request}),
        "gpu": gpu,
        "resources": resources,
        "startedAt": started,
    }


def ago(seconds):
    return (datetime.now(UTC) - timedelta(seconds=seconds)).isoformat()


def settings(**patch):
    return merge_settings({**DEFAULT_SETTINGS, **patch})


def decide(candidate, running=(), foreign=(), machine=None, resident=None, **setting):
    machine = machine or host()
    effective = capacity.effective(settings(**setting), machine)
    usage = capacity.usage(list(running), list(foreign), machine, resident=resident)
    return capacity.admit(candidate, usage, machine, effective)


@pytest.fixture(autouse=True)
def fresh_resident_cache():
    procs._resident_cache.clear()
    yield
    procs._resident_cache.clear()


def test_host_and_effective_settings():
    def probe():
        return {"gpus": [], "gpuProbeError": "nvidia-smi is unavailable."}

    snapshot = capacity.host(gpu_probe=probe)
    assert snapshot["gpus"] == [] and snapshot["gpuProbeError"] == "nvidia-smi is unavailable."
    assert snapshot["cpuCount"] >= 1 and "physicalCpuCount" in snapshot
    effective = capacity.effective(
        settings(gpuSlots={"1": 6}), host(gpus=((24.0, 20.0), (48.0, 40.0)), cpus=36, total=188.7)
    )
    assert effective["gpuSlots"] == {0: 4, 1: 6}
    assert effective["cpuTaskSlots"] == 9
    assert effective["reserves"]["cpuThreads"] == 2
    assert effective["reserves"]["ramGb"] == pytest.approx(9.435)
    assert effective["vramReserves"] == {0: pytest.approx(1.2), 1: pytest.approx(2.4)}
    fixed = capacity.effective(
        settings(cpuTaskSlots=3, reserves={"cpuThreads": 4, "ramGb": 30.0, "vramGb": 0.5}), host()
    )
    assert fixed["cpuTaskSlots"] == 3
    assert fixed["reserves"] == {"cpuThreads": 4, "ramGb": 30.0, "vramGb": 0.5}
    assert fixed["vramReserves"] == {0: 0.5}


def test_gpu_slots_and_memory():
    running = [task(gpu=0, vramGb=2.0) for _ in range(4)]
    assert decide(task(vramGb=2.0), running) == (False, None, "Waiting for a GPU slot (4/4)")
    assert decide(task(vramGb=2.0), running, gpuSlots={"0": 5}) == (True, 0, None)
    heavy = [task(gpu=0, vramGb=10.0), task(gpu=0, vramGb=10.0)]
    assert decide(task(vramGb=4.0), heavy) == (False, None, "Waiting for GPU memory")
    tight = host(gpus=((24.0, 3.0),))
    neighbour = [task(gpu=0, vramGb=0.5, started=SETTLED)]
    assert decide(task(vramGb=2.5), neighbour, machine=tight) == (
        False,
        None,
        "Waiting for GPU memory",
    )
    assert decide(task(vramGb=1.5), neighbour, machine=tight) == (True, 0, None)
    assert decide(task(vramGb=1.0), machine=host(gpus=())) == (False, None, "No GPU available")


def test_gpu_choice_prefers_fewer_tasks_then_more_free_memory():
    machine = host(gpus=((24.0, 10.0), (24.0, 20.0), (24.0, 22.0)))
    assert decide(task(vramGb=1.0), machine=machine) == (True, 2, None)
    assert decide(task(vramGb=1.0), [task(gpu=2, vramGb=1.0)], machine=machine) == (True, 1, None)


def test_exclusive_gpus():
    legacy_refit = {"gpu": 0, "cpus": 3, "ramGb": 8.0, "runsPerGpu": 1, "live": True}
    assert decide(task(vramGb=1.0), foreign=[legacy_refit]) == (
        False,
        None,
        "GPU 0 is reserved exclusively by another job",
    )
    shared = {"gpu": 0, "cpus": 3, "ramGb": 8.0, "runsPerGpu": 4, "live": True}
    assert decide(task(vramGb=1.0), foreign=[shared]) == (True, 0, None)
    assert decide(task(vramGb=1.0, exclusiveGpu=True), foreign=[shared]) == (
        False,
        None,
        "Waiting for an idle GPU",
    )
    assert decide(task(vramGb=1.0, exclusiveGpu=True)) == (True, 0, None)
    exclusive = task(gpu=0, vramGb=1.0, exclusiveGpu=True)
    assert decide(task(vramGb=1.0), [exclusive]) == (False, None, "Waiting for an idle GPU")
    usage = capacity.usage([], [shared], host())
    assert usage["gpus"][0]["slots"] == 1 and usage["cpu"]["committedThreads"] == 3


def test_cpu_threads_cpu_slots_and_service_tasks():
    machine = host(cpus=8)
    running = [task(gpu=0, vramGb=1.0, cpuThreads=2, dataWorkers=2)]
    assert decide(task(vramGb=1.0, cpuThreads=2, dataWorkers=1), running, machine=machine) == (
        False,
        None,
        "Waiting for CPU threads (4/6)",
    )
    # A task larger than the whole budget still runs when nothing else holds threads.
    assert decide(task(vramGb=1.0, cpuThreads=16), machine=machine) == (True, 0, None)
    cpu_running = [task(lane="cpu", cpuThreads=1)]
    assert decide(task(lane="cpu", cpuThreads=1), cpu_running, machine=machine, cpuTaskSlots=1) == (
        False,
        None,
        "Waiting for a CPU task slot (1/1)",
    )
    assert decide(task(vramGb=1.0, cpuThreads=1), cpu_running, machine=machine, cpuTaskSlots=1) == (
        True,
        0,
        None,
    )
    service = task(lane="cpu", cpuThreads=1, ramGb=0.5, service=True)
    full = [task(gpu=0, vramGb=1.0, cpuThreads=6)]
    assert decide(service, full + cpu_running, machine=machine, cpuTaskSlots=1) == (
        True,
        None,
        None,
    )
    usage = capacity.usage([service], [], machine)
    assert usage["cpu"] == {"committedThreads": 0, "cpuTasks": 0}
    assert usage["ram"]["committedGb"] == 0.5


def test_ram_counts_requested_memory_not_yet_resident():
    machine = host(available=12.0, total=64.0)  # reserve min(16, max(2, 3.2)) = 3.2
    other = [task(gpu=0, vramGb=1.0, ramGb=0.5, resources={"privateRamGb": 0.5})]
    assert decide(task(vramGb=1.0, ramGb=8.0), other, machine=machine) == (True, 0, None)
    assert decide(task(vramGb=1.0, ramGb=9.0), other, machine=machine) == (
        False,
        None,
        "Waiting for RAM",
    )
    starting = [task(gpu=0, vramGb=1.0, ramGb=6.0)]
    assert decide(task(vramGb=1.0, ramGb=4.0), starting, machine=machine) == (
        False,
        None,
        "Waiting for RAM",
    )
    resident = [task(gpu=0, vramGb=1.0, ramGb=6.0, resources={"privateRamGb": 5.5})]
    assert decide(task(vramGb=1.0, ramGb=4.0), resident, machine=machine) == (True, 0, None)


def test_blocking_scopes():
    assert capacity.blocking_scope("Waiting for CPU threads (4/6)") == "all"
    assert capacity.blocking_scope("Waiting for RAM") == "all"
    assert capacity.blocking_scope("Waiting for a CPU task slot (1/1)") == "cpu"
    assert capacity.blocking_scope("Waiting for a GPU slot (4/4)") == "gpu"
    assert capacity.blocking_scope("No GPU available") == "gpu"


# --- A request that never fits may still run alone ---------------------------------------


def test_vram_above_the_gpu_budget_runs_alone_on_an_idle_gpu():
    small = host(gpus=((16.0, 15.5),))  # reserve max(1, 0.8) = 1.0, so 15.0 GiB budget
    big = task(vramGb=16.5)  # e.g. a formula estimate larger than the card
    assert decide(big, machine=small) == (True, 0, None)
    waiting = (
        "Waiting for an idle GPU: needs 16.5 GiB of GPU memory, more than any GPU offers "
        "(15.0 GiB after the reserve), so it will run alone"
    )
    busy = [task(gpu=0, vramGb=2.0, started=SETTLED)]
    assert decide(big, busy, machine=small) == (False, None, waiting)
    assert capacity.blocking_scope(waiting) == "gpu"
    lease = {"gpu": 0, "cpus": 1, "ramGb": 0.0, "runsPerGpu": 4, "live": True}
    assert decide(big, foreign=[lease], machine=small) == (False, None, waiting)
    assert decide(big, [task(gpu=0, vramGb=2.0, started=ago(10))], machine=small)[2] == waiting
    # Memory held by a program without a lease is waited out, alone or not: next to it the
    # task would only run out of memory, fold after fold.
    occupied = host(gpus=((24.0, 6.0),))
    outside = "Waiting for GPU memory: 18.0 GiB of GPU 0 is in use outside HistoPilot"
    assert decide(task(vramGb=10.0), machine=occupied) == (False, None, outside)
    assert capacity.blocking_scope(outside) == "gpu"
    assert decide(task(vramGb=10.0), busy, machine=occupied) == (
        False,
        None,
        "Waiting for GPU memory",
    )
    exclusive = {"gpu": 0, "cpus": 1, "ramGb": 0.0, "runsPerGpu": 1, "live": True}
    assert decide(task(vramGb=10.0), foreign=[exclusive], machine=occupied) == (
        False,
        None,
        "GPU 0 is reserved exclusively by another job",
    )


def test_an_oversized_task_waits_for_a_larger_gpu_rather_than_a_smaller_idle_one():
    machine = host(gpus=((8.0, 7.5), (24.0, 20.0)))  # budgets 7.0 and 22.8 GiB
    busy = [task(gpu=1, vramGb=2.0, started=SETTLED)]
    assert decide(task(vramGb=16.0), busy, machine=machine) == (True, 1, None)
    assert decide(task(vramGb=16.0), busy, machine=machine, gpuSlots={"1": 1}) == (
        False,
        None,
        "Waiting for GPU memory",
    )
    reason = decide(task(vramGb=30.0), busy, machine=machine)[2]
    assert reason.startswith("Waiting for an idle GPU: needs 30.0 GiB") and "22.8 GiB" in reason
    # Alone only once the larger GPU is really empty, not just free of HistoPilot tasks.
    assert decide(task(vramGb=30.0), machine=machine) == (
        False,
        None,
        "Waiting for GPU memory: 4.0 GiB of GPU 1 is in use outside HistoPilot",
    )
    empty = host(gpus=((8.0, 7.5), (24.0, 23.5)))
    assert decide(task(vramGb=30.0), machine=empty) == (True, 1, None)


def test_run_alone_needs_a_really_empty_gpu():
    """A 24 GiB GPU with 1 GiB free used to admit anything alone; every fold then OOMed."""
    nearly_full = host(gpus=((24.0, 1.0),))
    assert decide(task(vramGb=2.4), machine=nearly_full) == (
        False,
        None,
        "Waiting for GPU memory: 23.0 GiB of GPU 0 is in use outside HistoPilot",
    )
    # Within the reserve (1.2 GiB) plus a little slack the GPU counts as empty ...
    assert decide(task(vramGb=30.0), machine=host(gpus=((24.0, 22.4),))) == (True, 0, None)
    assert decide(task(vramGb=30.0), machine=host(gpus=((24.0, 22.0),)))[0] is False
    # ... and without a live reading nothing is known to be in use.
    unknown = host(gpus=((24.0, 1.0),))
    unknown["gpus"][0]["freeMemoryGb"] = None
    assert decide(task(vramGb=30.0), machine=unknown) == (True, 0, None)
    # A task that fits next to the foreign memory still runs.
    assert decide(task(vramGb=0.5), machine=host(gpus=((24.0, 2.0),))) == (True, 0, None)


def test_ram_above_what_is_free_runs_alone_when_no_other_task_holds_ram():
    machine = host(available=12.0, total=64.0)  # 12 - 3.2 = 8.8 GiB usable now
    hungry = task(vramGb=1.0, ramGb=20.0)
    assert decide(hungry, machine=machine) == (True, 0, None)
    # A service coordinator waits for the folds it submitted; it never makes them wait.
    coordinator = task(lane="cpu", cpuThreads=1, ramGb=0.5, service=True)
    assert decide(hungry, [coordinator], machine=machine) == (True, 0, None)
    fold = [task(gpu=0, vramGb=1.0, resources={"privateRamGb": 4.0}, started=SETTLED)]
    assert decide(hungry, fold, machine=machine) == (False, None, "Waiting for RAM")
    huge = task(vramGb=1.0, ramGb=70.0)
    reason = decide(huge, fold, machine=machine)[2]
    assert reason == (
        "Waiting for RAM: needs 70.0 GiB, more than this machine offers (60.8 GiB after the "
        "reserve), so it will run alone once no other task holds RAM"
    )
    assert capacity.blocking_scope(reason) == "all"
    assert decide(huge, machine=machine) == (True, 0, None)
    # A live foreign reservation means the task would not be alone, even once it is fully
    # resident: that memory is in use, and the foreign job ends on its own.
    lease = {"gpu": None, "cpus": 2, "ramGb": 8.0, "live": True}
    for gb in (1.0, 8.0, 12.0):
        beside = {"foreign": [lease], "machine": machine, "resident": lambda _, gb=gb: gb}
        waiting = decide(huge, **beside)
        assert waiting[0] is False and waiting[2].startswith("Waiting for RAM: needs 70.0 GiB")
        assert decide(hungry, **beside) == (False, None, "Waiting for RAM")
    assert decide(huge, foreign=[{**lease, "ramGb": 0.0}], machine=machine) == (True, 0, None)


def test_oom_backoff_that_fits_only_an_idle_gpu_runs_alone_instead_of_stalling(center):
    center.store.update_settings({"gpuSlots": {"0": 5}})
    center.enqueue(
        "E1",
        center.spec("big", group="g1", request={"vramGb": 16.0}),
        center.spec("e1-1", group="g1", order=1, request={"vramGb": 1.0}),
    )
    runner = center.runner(
        adapters=lambda name: OutOfMemory(), host_probe=fake_host(total=24.0, free=23.0)
    )
    assert set(runner.tick()["started"]) == {"big", "e1-1"}
    center.tick_until(runner, lambda: center.started("big"))
    (center.folder("big") / "started").unlink()
    center.release("big", 3)  # CUDA OOM: requeued once with a request that fits only alone
    center.tick_until(runner, lambda: center.store.get("big")["attempt"] == 2)
    (center.folder("big") / "release").unlink()
    later = [center.spec(f"e2-{i}", group="g2", order=i, request={"vramGb": 2.0}) for i in (1, 2)]
    center.enqueue("E2", *later)
    runner.tick()
    assert center.state("big") == "queued" and center.state("e2-1") == "queued"
    center.release("e1-1")
    center.tick_until(runner, lambda: center.started("big"))
    assert center.running() == {"big"}
    for _ in range(3):
        runner.tick()
    assert center.running() == {"big"}
    assert center.store.get("e2-1")["waitingReason"] == "Waiting for GPU memory"
    center.release("big")
    center.tick_until(runner, lambda: center.started("e2-1") and center.started("e2-2"))


def test_ram_request_above_idle_availability_runs_alone_then_the_queue_continues(center):
    # 64 GiB available, reserve min(16, max(2, 6.4)) = 6.4: 57.6 GiB usable.
    center.enqueue("A", center.spec("hungry", group="gA", request={"ramGb": 60.0}))
    center.enqueue(
        "B",
        center.spec("b1", group="gB", order=1),
        center.spec("bcpu", group="gB", order=2, request={"lane": "cpu"}),
    )
    runner = center.runner()
    result = runner.tick()
    assert result["started"] == ["hungry"]
    assert result["waiting"]["b1"] == result["waiting"]["bcpu"] == "Waiting for RAM"
    center.release("hungry")
    center.tick_until(runner, lambda: center.started("b1") and center.started("bcpu"))


# --- VRAM that tasks are still allocating -------------------------------------------------


def test_vram_of_tasks_still_allocating_is_pending_against_live_free_memory():
    machine = host(gpus=((24.0, 8.0),))  # a program without a lease holds 16 GiB
    effective = capacity.effective(settings(defaultGpuSlots=5), machine)
    usage = capacity.usage([], [], machine)
    fold = task(vramGb=2.4)
    admitted, reason = [], None
    while len(admitted) < 5:
        ok, gpu, reason = capacity.admit(fold, usage, machine, effective)
        if not ok:
            break
        capacity.claim(usage, fold, gpu)
        admitted.append(gpu)
    # 8 - 1.2 = 6.8 GiB free after the reserve holds two, not five.
    assert admitted == [0, 0] and reason == "Waiting for GPU memory"
    assert usage["gpus"][0]["pendingVramGb"] == pytest.approx(4.8)
    for started, pending in ((ago(30), 2.4), (None, 2.4), (ago(120), 0.0)):
        running = [task(gpu=0, vramGb=2.4, started=started)]
        assert capacity.usage(running, [], machine)["gpus"][0]["pendingVramGb"] == pending
    recent = [task(gpu=0, vramGb=2.4, started=ago(30)) for _ in range(2)]
    assert decide(fold, recent, machine=machine, defaultGpuSlots=5) == (
        False,
        None,
        "Waiting for GPU memory",
    )
    later = (datetime.now(UTC) + timedelta(seconds=120)).isoformat()
    assert capacity.usage(recent, [], machine, now=later)["gpus"][0]["pendingVramGb"] == 0.0
    settled = [task(gpu=0, vramGb=2.4, started=SETTLED) for _ in range(2)]
    assert decide(fold, settled, machine=machine, defaultGpuSlots=5) == (True, 0, None)


def test_runner_admits_only_what_live_free_vram_holds_in_one_tick(center):
    center.store.update_settings({"defaultGpuSlots": 5})
    runner = center.runner(host_probe=fake_host(total=24.0, free=8.0, cpus=36, available=150.0))
    request = {"vramGb": 2.4, "cpuThreads": 2, "dataWorkers": 2, "ramGb": 5.0}
    specs = [center.spec(f"t{i}", group="E1", order=i, request=request) for i in range(5)]
    center.enqueue("E1", *specs)
    result = runner.tick()
    assert result["started"] == ["t0", "t1"]
    assert result["waiting"]["t2"] == "Waiting for GPU memory"
    runner.tick()
    assert center.running() == {"t0", "t1"}


# --- Foreign reservations -----------------------------------------------------------------


def test_foreign_reservations_not_yet_resident_count_as_pending_ram():
    machine = {
        "cpuCount": 32,
        "totalRamGb": 64.0,
        "availableRamGb": 50.0,  # reserve 3.2 GiB
        "gpus": [{"index": 0, "totalMemoryGb": 48.0, "freeMemoryGb": 40.0}],
    }
    legacy = {"gpu": 0, "cpus": 4, "ramGb": 40.0, "runsPerGpu": 4, "live": True}
    effective = capacity.effective(settings(), machine)
    fold = task(vramGb=4.0, dataWorkers=2, ramGb=12.0)

    def admitted(resident):
        usage = capacity.usage([], [legacy], machine, resident=lambda _: resident)
        count = 0
        while count < 4 and capacity.admit(fold, usage, machine, effective)[0]:
            capacity.claim(usage, fold, 0)
            count += 1
        return count, usage["ram"]["pendingGb"] - 12.0 * count

    # A just-started legacy fold: nothing resident yet (or unknown) reserves all 40 GiB.
    assert admitted(None) == (0, 40.0) and admitted(0.0) == (0, 40.0)
    assert admitted(20.0) == (2, 20.0)
    assert admitted(40.0) == (3, 0.0)


def test_lease_residency_is_measured_from_its_process_group(registry):
    script = "import time; block = b'x' * (96 << 20); print('ready', flush=True); time.sleep(60)"
    child = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        assert child.stdout.readline().strip() == "ready"
        registry.mkdir(mode=0o700)
        lease = {
            "process": procs.identity(child.pid),
            "processGroupId": child.pid,
            "gpu": None,
            "cpus": 1,
            "ramGb": 1.0,
            "runsPerGpu": 1,
            "kind": "legacy",
        }
        (registry / f"lease-{child.pid}.json").write_text(json.dumps(lease))
        [row] = leases.read_leases()
        assert row["live"]
        resident = procs.lease_resident_gb(row)
        assert 0.09 < resident < 0.5
        procs._resident_cache.clear()
        # A row without the identity is read back from its lease file.
        bare = {
            key: value for key, value in row.items() if key not in {"process", "processGroupId"}
        }
        assert procs.lease_resident_gb(bare) == pytest.approx(resident, abs=0.02)
        usage = capacity.usage([], [row], host())
        assert usage["ram"]["pendingGb"] == pytest.approx(1.0 - resident)
        assert procs.lease_resident_gb({"file": "lease-1.json"}) is None
    finally:
        child.kill()
        child.wait()
    procs._resident_cache.clear()
    assert procs.lease_resident_gb(row) is None  # its group has ended


# --- Private RAM ---------------------------------------------------------------------------


def test_private_ram_counts_anonymous_pages_not_a_memory_mapped_pack(tmp_path):
    rollup = (
        "Rss: 900 kB\nPss_Anon: 100 kB\nPss_File: 800 kB\nPss_Shmem: 20 kB\n"
        "Private_Clean: 700 kB\nPrivate_Dirty: 110 kB\n"
    )
    assert procs._anonymous_gb(rollup) == pytest.approx(120 / 1024**2)
    older = "Rss: 900 kB\nPrivate_Clean: 700 kB\nPrivate_Dirty: 110 kB\n"
    assert procs._anonymous_gb(older) == pytest.approx(110 / 1024**2)
    assert procs._anonymous_gb("Rss: 900 kB\n") is None
    pack, size = tmp_path / "features.bin", 64 << 20
    with pack.open("wb") as stream:
        for _ in range(size >> 20):
            stream.write(os.urandom(1 << 20))
    before = procs._smaps_private_gb(os.getpid())
    with pack.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as pages:
        assert sum(pages[offset] for offset in range(0, size, mmap.PAGESIZE)) >= 0
        mapped = procs._smaps_private_gb(os.getpid())
    # Read by this process alone, the whole pack would have counted as private before.
    assert mapped - before < 0.25 * size / 1024**3
    block = b"x" * size
    assert procs._smaps_private_gb(os.getpid()) - mapped > 0.8 * size / 1024**3
    del block
