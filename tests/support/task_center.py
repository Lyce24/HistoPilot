"""The private Task Center each test gets, with helpers to read and drive its tasks.

``tests/conftest.py`` gives every test its own state directory and keeps the runner from
starting. Services built without an injected executor therefore queue their work in this
store exactly as in production. Tests read the queued tasks, move them through the states
the runner would record, or tick a real runner when the worker itself has to run.
"""

import copy
import subprocess
import time

from histopilot.taskcenter import procs
from histopilot.taskcenter.adapters.base import RunnerContext
from histopilot.taskcenter.client import default_client
from histopilot.taskcenter.model import ACTIVE, PENDING, utc_now_iso
from histopilot.taskcenter.runner import Runner


def fake_host(*, gpus=1, total=24.0, free=20.0, cpus=32, available=64.0):
    """A host probe with fixed CPU, RAM and GPU capacity."""
    snapshot = {
        "cpuCount": cpus,
        "totalRamGb": 128.0,
        "availableRamGb": available,
        "bootId": "test",
        "kernel": "test",
        "physicalCpuCount": cpus // 2,
        "gpus": [
            {
                "index": index,
                "uuid": f"GPU-{index}",
                "name": "Test GPU",
                "driverVersion": "1",
                "totalMemoryGb": total,
                "usedMemoryGb": total - free,
                "freeMemoryGb": free,
                "utilizationPercent": 10.0,
            }
            for index in range(gpus)
        ],
    }
    return lambda: copy.deepcopy(snapshot)


class Center:
    """The store ``default_client()`` resolves for this test, plus runners started here."""

    def __init__(self):
        self.client = default_client()
        self.store = self.client.store
        self.runners = []
        self.logs = []

    def tasks(self, *, kind=None, adapter=None, state=None, group=None):
        """Every task, oldest first, optionally filtered by kind, adapter, state or group id."""
        return [
            task
            for task in self.store.list(limit=None)
            if (kind is None or task["kind"] == kind)
            and (adapter is None or task["adapter"] == adapter)
            and (state is None or task["state"] == state)
            and (group is None or (task.get("group") or {}).get("id") == group)
        ]

    def task(self, task_id):
        return self.store.get(task_id)

    def state(self, task_id):
        return self.store.get(task_id)["state"]

    def start(self, task_id, **fields):
        """Record a queued task as running, as the runner does after it spawns the worker."""
        started = self.store.transition(
            task_id,
            from_states=tuple(PENDING | {"starting"}),
            to_state="running",
            started_at=utc_now_iso(),
            **fields,
        )
        assert started, f"{task_id} is {self.state(task_id)}, not waiting to start"

    def finish(self, task_id, state="succeeded", *, returncode=0, reason=None, error=None):
        """Record a terminal outcome the way the runner concludes an exited worker."""
        if self.state(task_id) not in ACTIVE:
            self.start(task_id)
        exit_record = {
            "reason": reason,
            "returncode": returncode,
            "lost": False,
            "signalled": False,
            "killed": False,
            "stopReason": None,
            "error": error,
        }
        outcome = self.store.conclude(
            task_id, to_state=state, stop_request=None, exit=exit_record, error=error
        )
        assert outcome == "concluded", f"{task_id} could not be concluded: {outcome}"

    def runner(self, **options):
        """A real runner over this store with a fixed host; ``tick_until`` drives it."""
        runner = Runner(
            self.store,
            **{
                "host_probe": fake_host(),
                "cuda_probe": lambda gpu: ("ok", None),
                "sample_interval": 0.0,
                "host_interval": 0.0,
                "log": self.logs.append,
                **options,
            },
        )
        self.runners.append(runner)
        runner.start()
        return runner

    def tick_until(self, runner, predicate, timeout=30.0):
        deadline = time.monotonic() + timeout
        while True:
            result = runner.tick()
            if predicate():
                return result
            if time.monotonic() > deadline:
                states = {task["id"]: task["state"] for task in self.store.list(limit=None)}
                raise AssertionError(f"Condition not reached; tasks: {states}; log: {self.logs}")
            time.sleep(0.02)

    def context(self, host=None):
        """The context an adapter hook receives from the runner."""
        return RunnerContext(
            store=self.store,
            now=utc_now_iso,
            settings=self.store.settings(),
            host=(host or fake_host())(),
            log=self.logs.append,
        )

    def cleanup(self):
        for runner in self.runners:
            runner.close()
        for task in self.store.list(limit=None):
            if task["process"]:
                procs.kill_group(task["process"])
        for runner in self.runners:
            for child in runner._procs.values():
                try:
                    child.kill()
                    child.wait(timeout=5)
                except (OSError, subprocess.SubprocessError):
                    pass
