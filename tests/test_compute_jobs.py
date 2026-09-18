"""No real tmux jobs: durable generic execution, cancellation and retry boundaries."""

import copy
import hashlib
import sys

import pytest
from test_worker_process_ownership import isolated_worker_tree as _worker_tree

from histopilot.application.compute_jobs import ComputeJobService
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import ScientificStore
from histopilot.workers.packing_process import write_json
from histopilot.workers.training_process import read_json

isolated_worker_tree = _worker_tree


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


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("owned_envelope", [False, True])
def test_accepted_launch_replay_is_read_only_and_preserves_legacy_request_hashes(job, monkeypatch, legacy, owned_envelope):
    service, identity, plan, executor = job
    if owned_envelope:
        plan.update(recordId=identity, recordContentHash=service.store.get_configuration(identity)["contentHash"])
    service.launch(identity, plan, "launch")
    folder = service.folder(identity)
    if legacy:
        state = read_json(folder / "state.json")
        state.pop("operationActions")
        write_json(folder / "state.json", state)
    before = (folder / "state.json").read_bytes(), (folder / "plan.json").read_bytes()
    monkeypatch.setattr(service, "runtime", lambda: pytest.fail("Accepted replay must not inspect runtime"))
    assert service.replay_launch(identity, "launch")["status"] == "queued"
    assert service.replay_launch(identity, "unknown") is None
    assert before == ((folder / "state.json").read_bytes(), (folder / "plan.json").read_bytes())
    assert len(executor.calls) == 1
    with pytest.raises(StorageError) as caught:
        service.replay_launch(identity, "launch", resume=True)
    assert caught.value.code == "OPERATION_CONFLICT"


def test_replay_keeps_numeric_default_compatibility_for_legacy_plans(job):
    service, identity, plan, _ = job
    from histopilot.schemas.development import ResourcePolicy

    plan["resources"] = ResourcePolicy(gpuIds=[]).model_dump()
    service.launch(identity, plan, "launch")
    path = service.folder(identity) / "state.json"
    state = read_json(path)
    state.pop("operationActions")
    write_json(path, state)
    assert service.replay_launch(identity, "launch")["status"] == "queued"


def test_replay_rejects_changed_resources_wrong_kind_and_altered_saved_plan(job):
    service, identity, plan, _ = job
    service.launch(identity, plan, "launch")
    with pytest.raises(StorageError) as caught:
        service.replay_launch(identity, "launch", resources={**plan["resources"], "ramGbPerRun": 1})
    assert caught.value.code == "OPERATION_CONFLICT"
    with pytest.raises(StorageError) as caught:
        service.replay_launch(identity, "launch", record_kind="model-interpretation")
    assert caught.value.code == "COMPUTE_NOT_FOUND"
    path = service.folder(identity) / "plan.json"
    changed = read_json(path)
    changed["data"]["changed"] = True
    write_json(path, changed)
    with pytest.raises(StorageError) as caught:
        service.replay_launch(identity, "launch")
    assert caught.value.code == "COMPUTE_PLAN_CHANGED"


def test_replay_still_checks_dependency_lifecycle(job):
    service, identity, plan, _ = job
    service.launch(identity, plan, "launch")
    record = service.store.get_configuration(identity)
    service.store.lifecycle.apply(
        {f"dataset:{record['manifest']['datasetId']}": "trashed"}, operation_id="trash-source",
        request_hash=hashlib.sha256(b"trash-source").hexdigest(), expected_revision=0)
    with pytest.raises(StorageError):
        service.replay_launch(identity, "launch")


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


def test_orphan_compute_loader_blocks_resume_until_cancelled(job, isolated_worker_tree):
    from histopilot.workers.training_process import confirmed_process_alive

    service, identity, plan, executor = job
    state = service.launch(identity, plan, "launch")
    leader, process, child = isolated_worker_tree()
    leader.kill()
    leader.wait(timeout=5)
    state.update(status="failed", process=process, processGroupId=process["pid"])
    write_json(service.folder(identity) / "state.json", state)
    executor.sessions.clear()
    assert service.status(identity)["status"] == "running"
    with pytest.raises(StorageError, match="active"):
        service.launch(identity, plan, "resume-orphan", resume=True)
    assert service.cancel(identity, "cancel")["status"] == "failed"
    assert not confirmed_process_alive(child)
    assert len(executor.calls) == 1


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


@pytest.mark.parametrize("corruption", ["json", "shape", "symlink", "nonfinite", "overflow"])
def test_optional_progress_cannot_block_status_or_cancellation(job, corruption):
    service, identity, plan, executor = job
    service.launch(identity, plan, "launch")
    folder = service.folder(identity)
    progress = folder / "progress.json"
    if corruption == "symlink":
        progress.symlink_to(folder / "state.json")
    else:
        progress.write_text(
            {
                "json": "{",
                "shape": "[]",
                "nonfinite": '{"loss": NaN}',
                "overflow": '{"loss": 1e999}',
            }[corruption]
        )
    state = service.status(identity)
    assert state["status"] == "queued"
    assert state["progress"] is None and state["progressWarning"]
    assert service.cancel(identity, "cancel")["cancellationRequested"]
    executor.sessions.clear()
    assert service.status(identity)["status"] == "cancelled"


def test_corrupt_authoritative_state_remains_a_structured_error(job):
    service, identity, plan, _executor = job
    service.launch(identity, plan, "launch")
    (service.folder(identity) / "state.json").write_text("{")
    with pytest.raises(StorageError) as error:
        service.status(identity)
    assert error.value.code == "TRAINING_STATE_INVALID"


def test_lost_acknowledgement_preserves_fast_worker_completion(job, monkeypatch):
    service, identity, plan, executor = job

    def finish_then_timeout(*_args, **_kwargs):
        folder = service.folder(identity)
        state = read_json(folder / "state.json")
        result = {"state": "succeeded", "runId": identity}
        state.update(
            status="completed",
            result=result,
            process={"pid": 2147483000, "startTicks": 1, "bootId": "old-boot"},
        )
        write_json(folder / "result.json", result)
        write_json(folder / "state.json", state)
        raise TimeoutError("Lost acknowledgement")

    monkeypatch.setattr(executor, "launch", finish_then_timeout)
    assert service.launch(identity, plan, "launch")["status"] == "completed"
    assert service.status(identity)["status"] == "completed"


def test_worker_completion_during_session_probe_is_not_overwritten(job, monkeypatch):
    service, identity, plan, executor = job
    pending_probe = False

    def launch_then_timeout(*_args, **_kwargs):
        nonlocal pending_probe
        pending_probe = True
        raise TimeoutError("Lost acknowledgement")

    def inspect(_session):
        nonlocal pending_probe
        if pending_probe:
            pending_probe = False
            folder = service.folder(identity)
            state = read_json(folder / "state.json")
            result = {"state": "succeeded", "runId": identity}
            state.update(
                status="completed",
                result=result,
                process={"pid": 2147483000, "startTicks": 1, "bootId": "old-boot"},
            )
            write_json(folder / "result.json", result)
            write_json(folder / "state.json", state)
        return False

    monkeypatch.setattr(executor, "launch", launch_then_timeout)
    monkeypatch.setattr(executor, "running", inspect)
    shown = service.launch(identity, plan, "launch")
    assert shown["status"] == "completed"
    assert service.launch(identity, plan, "launch")["status"] == "completed"
    assert read_json(service.folder(identity) / "state.json") == shown
