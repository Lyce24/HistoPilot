"""Durable generic execution, cancellation and retry boundaries of Task Center compute jobs."""

import copy
import hashlib

import pytest
from support.training import runtime

from histopilot.application.compute_jobs import ComputeJobService
from histopilot.storage.io import read_json_bounded, write_json_atomic
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import ScientificStore

PLAN = {
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


def evaluation_record(tmp_path):
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
    return store, record


@pytest.fixture
def job(tmp_path, task_center):
    store, record = evaluation_record(tmp_path)
    service = ComputeJobService(store, runtime=runtime, task_center=task_center.client)
    return service, record["id"], copy.deepcopy(PLAN)


def test_compute_launch_pins_code_and_retry_does_not_relaunch(job, task_center):
    service, identity, plan = job
    original = service.launch(identity, plan, "launch")
    assert original["status"] == "queued"
    folder = service.folder(identity)
    assert (folder / "compute" / "histopilot" / "workers" / "compute_job.py").is_file()
    assert service.launch(identity, plan, "launch")["status"] == "queued"
    assert [task["attempt"] for task in task_center.tasks()] == [1]
    changed = copy.deepcopy(plan)
    changed["resources"]["ramGbPerRun"] = 1
    with pytest.raises(StorageError, match="another compute request"):
        service.launch(identity, changed, "launch")


def test_pinned_launch_freezes_submitted_code_and_copies_its_archive(
    job, task_center, monkeypatch, tmp_path
):
    from histopilot.application import compute_jobs
    from histopilot.workers.compute_archive import prepare_compute_archive
    from histopilot.workers.training_process import compute_snapshot

    service, identity, plan = job
    submitted = compute_snapshot()
    source = prepare_compute_archive(tmp_path / "batch", submitted) / "histopilot"
    # The live checkout moved on after submission; the job must not notice.
    monkeypatch.setattr(compute_jobs, "compute_snapshot", lambda: {"sha256": "edited", "files": {}})
    state = service.launch(identity, plan, "launch", pinned=(submitted, source))
    assert read_json_bounded(service.folder(identity) / "plan.json")["code"] == submitted
    command = task_center.task(state["taskId"])["command"]
    assert command["cwd"] == str(service.folder(identity) / "compute")


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("owned_envelope", [False, True])
def test_accepted_launch_replay_is_read_only_and_preserves_legacy_request_hashes(
    job, task_center, monkeypatch, legacy, owned_envelope
):
    service, identity, plan = job
    if owned_envelope:
        plan.update(
            recordId=identity,
            recordContentHash=service.store.get_configuration(identity)["contentHash"],
        )
    service.launch(identity, plan, "launch")
    folder = service.folder(identity)
    if legacy:
        state = read_json_bounded(folder / "state.json")
        state.pop("operationActions")
        write_json_atomic(folder / "state.json", state)
    before = (folder / "state.json").read_bytes(), (folder / "plan.json").read_bytes()
    monkeypatch.setattr(
        service, "runtime", lambda: pytest.fail("Accepted replay must not inspect runtime")
    )
    assert service.replay_launch(identity, "launch")["status"] == "queued"
    assert service.replay_launch(identity, "unknown") is None
    assert before == ((folder / "state.json").read_bytes(), (folder / "plan.json").read_bytes())
    assert len(task_center.tasks()) == 1
    with pytest.raises(StorageError) as caught:
        service.replay_launch(identity, "launch", resume=True)
    assert caught.value.code == "OPERATION_CONFLICT"


def test_replay_keeps_numeric_default_compatibility_for_legacy_plans(job):
    service, identity, plan = job
    from histopilot.schemas.development import ResourcePolicy

    plan["resources"] = ResourcePolicy(gpuIds=[]).model_dump()
    service.launch(identity, plan, "launch")
    path = service.folder(identity) / "state.json"
    state = read_json_bounded(path)
    state.pop("operationActions")
    write_json_atomic(path, state)
    assert service.replay_launch(identity, "launch")["status"] == "queued"


def test_replay_rejects_changed_resources_wrong_kind_and_altered_saved_plan(job):
    service, identity, plan = job
    service.launch(identity, plan, "launch")
    with pytest.raises(StorageError) as caught:
        service.replay_launch(identity, "launch", resources={**plan["resources"], "ramGbPerRun": 1})
    assert caught.value.code == "OPERATION_CONFLICT"
    with pytest.raises(StorageError) as caught:
        service.replay_launch(identity, "launch", record_kind="model-interpretation")
    assert caught.value.code == "COMPUTE_NOT_FOUND"
    path = service.folder(identity) / "plan.json"
    changed = read_json_bounded(path)
    changed["data"]["changed"] = True
    write_json_atomic(path, changed)
    with pytest.raises(StorageError) as caught:
        service.replay_launch(identity, "launch")
    assert caught.value.code == "COMPUTE_PLAN_CHANGED"


def test_replay_still_checks_dependency_lifecycle(job):
    service, identity, plan = job
    service.launch(identity, plan, "launch")
    record = service.store.get_configuration(identity)
    service.store.lifecycle.apply(
        {f"dataset:{record['manifest']['datasetId']}": "trashed"},
        operation_id="trash-source",
        request_hash=hashlib.sha256(b"trash-source").hexdigest(),
        expected_revision=0,
    )
    with pytest.raises(StorageError):
        service.replay_launch(identity, "launch")


def test_interrupted_job_resumes_same_plan_and_preserves_audit(job, task_center):
    service, identity, plan = job
    task_id = service.launch(identity, plan, "launch")["taskId"]
    task_center.finish(task_id, "interrupted", returncode=None, reason="lost")
    assert service.status(identity)["status"] == "interrupted"
    with pytest.raises(StorageError, match="Resume"):
        service.launch(identity, plan, "new-launch")
    result = service.launch(identity, plan, "resume", resume=True)
    assert result["attempt"] == 2 and len(result["operations"]) == 2
    assert task_center.task(task_id)["attempt"] == 2


def test_changed_plan_and_archive_block_resume(job, task_center):
    service, identity, plan = job
    task_id = service.launch(identity, plan, "launch")["taskId"]
    task_center.finish(task_id, "interrupted", returncode=None, reason="lost")
    path = service.folder(identity) / "plan.json"
    value = read_json_bounded(path)
    value["kind"] = "refit"
    write_json_atomic(path, value)
    with pytest.raises(StorageError, match="plan changed"):
        service.launch(identity, plan, "resume", resume=True)


def test_cancel_requested_while_running_and_completion_blocks_relaunch(job, task_center):
    service, identity, plan = job
    task_id = service.launch(identity, plan, "launch")["taskId"]
    task_center.start(task_id)
    state = service.cancel(identity, "cancel")
    assert state["status"] != "cancelled" and state["cancellationRequested"]
    # The worker finished and recorded its receipt before it saw the request.
    folder = service.folder(identity)
    saved = read_json_bounded(folder / "state.json")
    saved.update(status="completed", result={"state": "succeeded", "runId": identity})
    write_json_atomic(folder / "result.json", saved["result"])
    write_json_atomic(folder / "state.json", saved)
    task_center.finish(task_id, "succeeded")
    assert service.status(identity)["status"] == "completed"
    with pytest.raises(StorageError, match="completed"):
        service.launch(identity, plan, "again", resume=True)


def test_unknown_record_and_trashed_project_never_launch(job, task_center):
    service, identity, plan = job
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
    assert task_center.tasks() == []


def test_delayed_cancel_retry_does_not_cancel_resumed_attempt(job):
    service, identity, plan = job
    service.launch(identity, plan, "launch")
    assert service.cancel(identity, "cancel")["status"] == "cancelled"
    service.launch(identity, plan, "resume", resume=True)
    assert not service.status(identity)["cancellationRequested"]
    assert not service.cancel(identity, "cancel")["cancellationRequested"]
    assert service.cancel(identity, "cancel-new")["cancellationRequested"]


@pytest.mark.parametrize("corruption", ["json", "shape", "symlink", "nonfinite", "overflow"])
def test_optional_progress_cannot_block_status_or_cancellation(job, corruption):
    service, identity, plan = job
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
    cancelled = service.cancel(identity, "cancel")
    assert cancelled["cancellationRequested"] and cancelled["status"] == "cancelled"


def test_corrupt_authoritative_state_remains_a_structured_error(job):
    service, identity, plan = job
    service.launch(identity, plan, "launch")
    (service.folder(identity) / "state.json").write_text("{")
    with pytest.raises(StorageError) as error:
        service.status(identity)
    assert error.value.code == "TRAINING_STATE_INVALID"


def test_task_titles_name_the_kind_once():
    def title(name, plan):
        record = {"id": "configuration-1", "manifest": {"name": name}}
        return ComputeJobService._task_title(record, plan)

    inference = {"kind": "evaluation", "purpose": "inference"}
    # Apply models names its runs after their purpose already.
    assert title("Inference · Baseline · Study · seed ensemble", inference) == (
        "Inference · Baseline · Study · seed ensemble"
    )
    assert title("Evaluation · Study", {"kind": "evaluation"}) == "Evaluation · Study"
    assert title("Study · fold 1", {"kind": "refit"}) == "Refit · Study · fold 1"
    assert title(None, {"kind": "interpretation"}) == "Attention · configuration-1"


# -- Records launched before the Task Center, on their tmux executor ---------------------------
