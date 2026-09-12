"""No real tmux jobs: durable generic execution, cancellation and retry boundaries."""

import copy
import hashlib
import sys

import pytest

from histopilot.application.compute_jobs import ComputeJobService
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import ScientificStore
from histopilot.workers.packing_process import write_json
from histopilot.workers.training_process import read_json


class Executor:
    def __init__(self):
        self.sessions, self.calls = set(), []

    def running(self, session):
        return session in self.sessions

    def launch(self, session, python, plan, log, *, package_root):
        self.sessions.add(session)
        self.calls.append((session, python, plan, log, package_root))


@pytest.fixture
def job(tmp_path):
    store = ScientificStore(tmp_path, "compute-project")
    draft = store.create_draft("import", "Data", {})
    dataset = store.publish_dataset(
        draft["id"],
        expected_revision=1,
        manifest={"name": "Data"},
        artifacts={"slides.json": b'[{"slideId":"slide-1"}]'},
        operation_id="dataset",
    )
    record = store.publish_configuration(
        manifest={"kind": "model-evaluation", "name": "Test", "datasetId": dataset["id"]},
        operation_id="record",
    )
    executor = Executor()

    def runtime():
        return {
            "available": True,
            "python": sys.executable,
            "versions": {},
            "cudaAvailable": False,
            "gpuCount": 0,
        }

    service = ComputeJobService(store, executor=executor, runtime=runtime)
    plan = {
        "kind": "evaluation",
        "resources": {
            "gpuIds": [],
            "cpuThreadsPerRun": 1,
            "dataLoaderWorkers": 0,
            "ramGbPerRun": 0.01,
            "maxConcurrentRuns": 1,
            "runsPerGpu": 1,
        },
        "data": {"sourceStamps": {}},
    }
    return service, record["id"], plan, executor


def test_compute_launch_pins_code_and_retry_does_not_relaunch(job):
    service, identity, plan, executor = job
    original = service.launch(identity, plan, "launch")
    assert original["status"] == "queued"
    assert executor.calls[0][4].joinpath("histopilot/workers/compute_job.py").is_file()
    assert service.launch(identity, plan, "launch")["status"] == "queued"
    assert len(executor.calls) == 1
    changed = copy.deepcopy(plan)
    changed["resources"]["ramGbPerRun"] = 1
    with pytest.raises(StorageError, match="another compute request"):
        service.launch(identity, changed, "launch")


def test_interrupted_job_resumes_same_plan_and_preserves_audit(job):
    service, identity, plan, executor = job
    service.launch(identity, plan, "launch")
    executor.sessions.clear()
    assert service.status(identity)["status"] == "interrupted"
    with pytest.raises(StorageError, match="Resume"):
        service.launch(identity, plan, "new-launch")
    result = service.launch(identity, plan, "resume", resume=True)
    assert result["attempt"] == 2 and len(result["operations"]) == 2
    assert len(executor.calls) == 2


def test_changed_plan_and_archive_block_resume(job):
    service, identity, plan, executor = job
    service.launch(identity, plan, "launch")
    executor.sessions.clear()
    path = service.folder(identity) / "plan.json"
    value = read_json(path)
    value["kind"] = "refit"
    write_json(path, value)
    with pytest.raises(StorageError, match="plan changed"):
        service.launch(identity, plan, "resume", resume=True)


def test_cancel_keeps_pending_state_and_completion_blocks_relaunch(job):
    service, identity, plan, executor = job
    service.launch(identity, plan, "launch")
    state = service.cancel(identity, "cancel")
    assert state["status"] == "queued" and state["cancellationRequested"]
    executor.sessions.clear()
    folder = service.folder(identity)
    state = read_json(folder / "state.json")
    state.update(status="completed", result={"state": "succeeded", "runId": identity})
    write_json(folder / "result.json", state["result"])
    write_json(folder / "state.json", state)
    with pytest.raises(StorageError, match="completed"):
        service.launch(identity, plan, "again", resume=True)


def test_unknown_record_and_trashed_project_never_launch(job):
    service, identity, plan, executor = job
    with pytest.raises(StorageError):
        service.folder("../../escape")
    service.store.lifecycle.apply(
        {"project:compute-project": "trashed"},
        operation_id="trash",
        request_hash=hashlib.sha256(b"test").hexdigest(),
        expected_revision=0,
    )
    with pytest.raises(StorageError):
        service.launch(identity, plan, "launch")
    assert not executor.calls


def test_delayed_cancel_retry_does_not_cancel_resumed_attempt(job):
    service, identity, plan, executor = job
    service.launch(identity, plan, "launch")
    service.cancel(identity, "cancel")
    executor.sessions.clear()
    service.launch(identity, plan, "resume", resume=True)
    assert not service.status(identity)["cancellationRequested"]
    assert not service.cancel(identity, "cancel")["cancellationRequested"]
    assert service.cancel(identity, "cancel-new")["cancellationRequested"]


def test_lost_launch_acknowledgement_preserves_running_session(job, monkeypatch):
    service, identity, plan, executor = job
    launch = executor.launch

    def launch_then_timeout(*args, **kwargs):
        launch(*args, **kwargs)
        raise TimeoutError("Lost acknowledgement")

    monkeypatch.setattr(executor, "launch", launch_then_timeout)
    assert service.launch(identity, plan, "launch")["status"] == "queued"
    assert service.launch(identity, plan, "launch")["status"] == "queued"
    assert len(executor.calls) == 1
