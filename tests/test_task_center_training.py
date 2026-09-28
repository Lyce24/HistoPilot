"""Managed MIL batches run as Task Center tasks and keep every legacy evidence contract."""

import os
import runpy
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_training_control_preview import preview_context  # noqa: F401

from histopilot.application.feature_bundles import _hash
from histopilot.application.model_experiments import (
    ModelExperimentService,
    experiment_stage,
    predictors_settled,
)
from histopilot.application.training import TrainingService, task_folder
from histopilot.schemas.development import DevelopmentBatchSpec
from histopilot.schemas.model_experiments import (
    CreateModelExperiment,
    SubmitModelExperiment,
    UpdateModelExperiment,
)
from histopilot.storage.project_lock import StorageError
from histopilot.taskcenter import capacity, ids
from histopilot.taskcenter.adapters import adapter as registered_adapter
from histopilot.taskcenter.adapters.base import AdapterError
from histopilot.taskcenter.adapters.mil import MilCollectAdapter, MilFoldAdapter
from histopilot.taskcenter.model import LIVE, TERMINAL, utc_now_iso
from histopilot.workers.managed_collect import final_status
from histopilot.workers.packing_process import write_json
from histopilot.workers.training_process import read_json, save_state

support = runpy.run_path(str(Path(__file__).with_name("test_development_batches.py")))
setup_support = runpy.run_path(str(Path(__file__).with_name("test_experiment_setup.py")))

# Serialized exactly as every pre-Task-Center spec stored its default resources.
LEGACY_RESOURCES = {
    "maxConcurrentRuns": 1,
    "gpuIds": [0],
    "runsPerGpu": 1,
    "cpuThreadsPerRun": 2,
    "dataLoaderWorkers": 2,
    "ramGbPerRun": 8.0,
}
# previewHash values computed with the code before resources became optional.
LEGACY_OWNED_PREVIEW_HASH = "de83da255be2afc6adc13efee87644dd2d0762643bef8df731d304c1f9326593"
LEGACY_PLAIN_PREVIEW_HASH = "fb0d006935bdb3704af7625c81254dbe11fceaf8bddebd3df065499931adf887"
LEGACY_EXPERIMENT = {
    "id": "draft-legacy-experiment",
    "revision": 3,
    "name": "Frozen before the Task Center",
    "payload": {"notes": "", "tags": []},
}


def runtime(**_options):
    return {
        "available": True,
        "python": sys.executable,
        "versions": {},
        "cudaAvailable": False,
        "gpuCount": 0,
        "findings": [],
    }


@pytest.fixture(autouse=True)
def quiet_gpu_probe(monkeypatch):
    """Requeue provenance never calls the real nvidia-smi in tests."""
    monkeypatch.setattr("histopilot.taskcenter.adapters.mil.gpu_snapshot", lambda: {"gpus": []})


def tiny_spec(spec, **changes):
    values = spec.model_dump()
    values.update(mode="single", trainingSeeds=[11], **changes)
    values["recipe"].update(maxEpochs=1, bagSize=2, batchSize=2)
    return DevelopmentBatchSpec.model_validate(values)


@pytest.fixture
def managed(tmp_path, monkeypatch, task_center):
    monkeypatch.setattr("histopilot.application.training.gpu_snapshot", lambda: {"gpus": []})
    monkeypatch.setattr("histopilot.workers.training_process.gpu_snapshot", lambda: {"gpus": []})
    development, spec, _source = support["batch"].__wrapped__(tmp_path)
    assert "resources" not in spec.model_dump()
    spec = tiny_spec(spec)
    preview = development.preview(spec)
    frozen = development.freeze(
        spec, preview["previewHash"], "managed-batch", {"tag": "Managed batch"}
    )
    task_center.store.update_settings(
        {"defaults": {"cpuThreadsPerRun": 1, "dataLoaderWorkers": 0}, "cancelGraceSeconds": 5}
    )
    service = TrainingService(
        development.store,
        development.filesystem,
        runtime=runtime,
        task_center=task_center.client,
    )
    folder = development.store.folder / "training" / frozen["id"]
    return SimpleNamespace(
        development=development,
        service=service,
        frozen=frozen,
        identity=frozen["id"],
        center=task_center,
        store=task_center.store,
        client=task_center.client,
        folder=folder,
        spec=spec,
        messages=task_center.logs,
    )


def make_runner(context, *, probe=runtime, clock=time.monotonic):
    """A real runner with the MIL adapters, on this machine's CPUs and no GPU."""
    adapters = {"mil-fold": MilFoldAdapter(runtime=probe), "mil-collect": MilCollectAdapter()}
    return context.center.runner(
        clock=clock,
        host_probe=lambda: capacity.host(gpu_probe=lambda: {"gpus": []}),
        adapters=lambda name: adapters.get(name) or registered_adapter(name),
        sample_interval=1.0,
        host_interval=1.0,
    )


def group_tasks(context):
    return context.client.group(
        "mil-batch", context.identity, str(context.development.store.folder)
    )["tasks"]


def drive(context, runner, until, timeout=900):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        runner.tick()
        if until():
            return
        time.sleep(0.2)
    tasks = [(task["kind"], task["state"], task["waitingReason"]) for task in group_tasks(context)]
    raise AssertionError(f"Timed out: {tasks}\n" + "\n".join(context.messages[-20:]))


def attempt_events(context):
    import json

    lines = (context.folder / "attempts.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def settled(context):
    tasks = group_tasks(context)
    return bool(tasks) and all(task["state"] in TERMINAL for task in tasks)


def fold_running(context):
    return any(
        task["kind"] == "mil-fold" and task["state"] == "running" for task in group_tasks(context)
    )


def assert_completed_evidence(context):
    state = read_json(context.folder / "state.json")
    assert state["status"] == "completed" and state["executor"] == "task-center"
    assert state["runCounts"]["completed"] == state["runCounts"]["total"] == 5
    for run in state["runs"]:
        result = read_json(context.folder / "runs" / run["id"] / "result.json")
        # Promotion and exports require the state copy to equal the fold's receipt.
        assert run["result"] == result
        assert run["metrics"] == result["metrics"]
        assert run["checkpointPath"] == result["bestCheckpointPath"]
        assert read_json(context.folder / "runs" / run["id"] / "plan.json")["device"] == "cpu"
    results = read_json(context.folder / "results.json")
    assert results["status"] == "completed"
    assert results["oof"] and all(Path(row["path"]).is_file() for row in results["oof"])
    assert all(candidate["complete"] for candidate in results["candidates"])
    assert read_json(context.folder / "collect-result.json")["final"] is True
    execution = context.service.execution(context.identity)
    assert execution["status"] == "completed"
    assert execution["taskCenter"]["running"] == execution["taskCenter"]["queued"] == 0
    returned = context.service.results(context.identity)
    assert returned["status"] == "completed" and returned["oof"] == results["oof"]
    return state


# -- serialization -------------------------------------------------------------------------


def test_legacy_frozen_spec_keeps_its_serialization_and_preview_hash(preview_context, monkeypatch):  # noqa: F811
    snapshot = {
        "dataset": {"id": "frozen-dataset"},
        "protocol": {"id": "protocol"},
        "inputs": {"protocolId": "protocol", "featureBundleId": "bundle"},
    }
    monkeypatch.setattr(
        "histopilot.application.development.input_snapshot", lambda *_a, **_k: dict(snapshot)
    )
    legacy = {
        **DevelopmentBatchSpec(
            experimentId=LEGACY_EXPERIMENT["id"],
            experimentRevision=LEGACY_EXPERIMENT["revision"],
            experimentName=LEGACY_EXPERIMENT["name"],
            batchName="Sweep v1",
            inputs={"protocolId": "protocol", "featureBundleId": "bundle"},
        ).model_dump(),
        "resources": LEGACY_RESOURCES,
    }
    parsed = DevelopmentBatchSpec.model_validate(legacy, context={"legacy": True})
    stored = parsed.model_dump()
    assert stored["resources"] == LEGACY_RESOURCES
    # A stored legacy spec re-validates to exactly the bytes it was frozen with.
    again = DevelopmentBatchSpec.model_validate(stored, context={"legacy": True}).model_dump()
    assert again == stored and _hash(again) == _hash(stored)
    service = preview_context.service
    owned = service._preview(parsed, experiment_record=LEGACY_EXPERIMENT)
    assert owned["previewHash"] == LEGACY_OWNED_PREVIEW_HASH
    assert owned["inputSnapshot"]["resources"] == LEGACY_RESOURCES
    plain_values = {
        key: value
        for key, value in legacy.items()
        if key not in {"experimentId", "experimentRevision"}
    }
    plain = DevelopmentBatchSpec.model_validate(plain_values, context={"legacy": True})
    assert service._preview(plain)["previewHash"] == LEGACY_PLAIN_PREVIEW_HASH
    sharing = DevelopmentBatchSpec.model_validate(
        {**plain_values, "resources": {**LEGACY_RESOURCES, "runsPerGpu": 2}},
        context={"legacy": True},
    )
    assert "GPU_SHARING" in {row["code"] for row in service._preview(sharing)["findings"]}

    new = DevelopmentBatchSpec.model_validate(
        {key: value for key, value in legacy.items() if key != "resources"}
    )
    assert new.resources is None and "resources" not in new.model_dump()
    reviewed = service._preview(new, experiment_record=LEGACY_EXPERIMENT)
    assert "resources" not in reviewed["spec"] and "resources" not in reviewed["inputSnapshot"]
    assert "GPU_SHARING" not in {row["code"] for row in reviewed["findings"]}
    assert reviewed["previewHash"] != LEGACY_OWNED_PREVIEW_HASH


def test_setup_frozen_before_the_change_submits_with_its_preview_hash(tmp_path):
    fixture = setup_support["setup"].__wrapped__(tmp_path, SimpleNamespace(param="legacy"))
    service, record, request, _target_split, _source, training = fixture
    record = service.setup_inputs(record["id"], request)
    plan = {
        **DevelopmentBatchSpec(
            experimentName=record["name"],
            batchName="Baseline",
            inputs=record["inputs"],
            trainingSeeds=[42],
            predictorPolicy={"method": "skip", "refitPercentile": None},
        ).model_dump(),
        # What every setup saved before resources became optional.
        "resources": LEGACY_RESOURCES,
    }
    record = service.update(
        record["id"],
        UpdateModelExperiment(
            name=record["name"],
            expectedRevision=record["revision"],
            batchPlans=[{"id": "baseline", "spec": plan}],
        ),
    )
    frozen = setup_support["freeze"](service, record)
    setup = frozen["frozenSetup"]["manifest"]
    assert setup["batchPlans"][0]["spec"]["resources"] == LEGACY_RESOURCES
    submitted = setup_support["submit"](service, frozen)
    assert submitted["submission"]["status"] == "submitted"
    batch = service.store.get_configuration(submitted["submission"]["batchIds"][0])
    assert batch["manifest"]["spec"] == setup["batchPlans"][0]["spec"]
    assert batch["manifest"]["inputSnapshot"]["resources"] == LEGACY_RESOURCES
    assert {"planId": "baseline", "previewHash": batch["manifest"]["previewHash"]} in setup[
        "batchPreviews"
    ]
    assert training.launches == [batch["id"]]


def test_task_center_launches_ignore_a_frozen_setups_legacy_resources(managed):
    """Design section 4.7: legacy resources are read but ignored under the Task Center."""
    context = managed
    frozen = context.frozen
    legacy = {
        **LEGACY_RESOURCES,
        "gpuIds": [1],  # a GPU this machine does not have
        "cpuThreadsPerRun": 6,
        "dataLoaderWorkers": 5,
    }
    batch = {
        **frozen,
        "manifest": {
            **frozen["manifest"],
            "spec": {**frozen["manifest"]["spec"], "resources": legacy},
        },
    }
    plan, _freshness = context.service._prepare(batch)
    # The device kind follows this machine (no CUDA here); threads and loader workers are
    # the Task Center defaults. Nothing blocks on the frozen GPU ids.
    assert plan["resources"]["gpuIds"] == []
    assert (plan["resources"]["cpuThreadsPerRun"], plan["resources"]["dataLoaderWorkers"]) == (
        1,
        0,
    )


# -- launch, queue and status --------------------------------------------------------------


def test_launch_enqueues_one_task_per_fold_and_queued_work_never_reads_interrupted(managed):
    context = managed
    state = context.service.launch(context.identity, "launch-once")
    assert state["status"] == "queued" and state["executor"] == "task-center"
    assert state["sessionName"].startswith("tc-") and "resourcePlan" not in state
    assert state["taskGroup"] == {
        "kind": "mil-batch",
        "id": context.identity,
        "projectFolder": str(context.development.store.folder),
    }
    plan = read_json(context.folder / "plan.json")
    assert plan["executionMode"] == "task-center" and _hash(plan) == state["planHash"]
    assert plan["resources"] == {
        "maxConcurrentRuns": 1,
        "gpuIds": [],
        "runsPerGpu": 1,
        "cpuThreadsPerRun": 1,
        "dataLoaderWorkers": 0,
        "ramGbPerRun": 8.0,
    }
    tasks = group_tasks(context)
    folds = [task for task in tasks if task["kind"] == "mil-fold"]
    [final] = [task for task in tasks if task["kind"] == "mil-collect"]
    key = task_folder(context.folder)
    assert [task["id"] for task in folds] == [
        ids.fold_task_id(key, run["id"]) for run in plan["runs"]
    ]
    assert {task["state"] for task in folds} == {"queued"} and final["state"] == "blocked"
    assert final["id"] == ids.collect_task_id(key, True) and final["priority"] == "interactive"
    assert {row["task"] for row in context.store.dependencies(final["id"])} == {
        task["id"] for task in folds
    }
    fold = folds[0]
    assert fold["request"]["lane"] == "cpu" and fold["request"]["cpuThreads"] == 1
    assert fold["request"]["workloadKey"] == fold["request"]["workload"]["key"]
    assert fold["command"]["argv"][1:] == [
        "-u",
        "-m",
        "histopilot.workers.managed_fold",
        str(context.folder / "plan.json"),
        plan["runs"][0]["id"],
    ]
    assert fold["command"]["cwd"] == str(context.folder / "compute")
    assert fold["labels"]["fold"] == 1 and fold["labels"]["trainingSeed"] == 11
    owner = context.store.owner(fold["ownerKey"])
    assert owner["kind"] == "mil-batch" and owner["id"] == context.identity
    # Replays return the accepted launch without enqueueing again.
    assert context.service.launch(context.identity, "launch-once")["status"] == "queued"
    assert len(group_tasks(context)) == 6
    with pytest.raises(StorageError) as active:
        context.service.launch(context.identity, "launch-twice")
    assert active.value.code == "TRAINING_ACTIVE"

    view = context.service.execution(context.identity)
    assert view["status"] == "queued" and view["taskCenter"]["queued"] == 6
    assert view["taskCenter"]["ownerKey"] == owner["key"] and not view["taskCenter"]["held"]
    context.store.hold_owner(owner["key"], True)
    held = context.service.execution(context.identity)
    assert held["status"] == "queued" and held["taskCenter"]["held"]
    context.store.hold_owner(owner["key"], False)

    def unavailable(*_args, **_kwargs):
        raise StorageError("store offline", "TASK_CENTER_UNAVAILABLE", 503)

    original = context.client.group
    context.client.group = unavailable
    try:
        unknown = context.service.execution(context.identity)
    finally:
        context.client.group = original
    assert unknown["status"] == "queued" and unknown["taskCenter"] is None
    assert "TASK_CENTER_UNAVAILABLE" in {item["code"] for item in unknown["findings"]}

    # Only once no task can move the batch does it read as interrupted.
    context.client.cancel_group(
        "mil-batch", context.identity, str(context.development.store.folder)
    )
    assert context.service.execution(context.identity)["status"] == "interrupted"


def test_a_busy_host_queues_the_batch_instead_of_refusing_it(managed, monkeypatch):
    from histopilot.application import training

    real = training.host_snapshot

    def busy():
        return {**real(), "availableRamGb": 1.0, "totalRamGb": 64.0}

    monkeypatch.setattr(training, "host_snapshot", busy)
    # Free memory is the runner's admission decision, not a launch error.
    assert managed.service.launch(managed.identity, "launch-busy")["status"] == "queued"
    monkeypatch.setattr(training, "host_snapshot", lambda: {**busy(), "totalRamGb": 4.0})
    with pytest.raises(StorageError) as error:
        managed.service._prepare(managed.development.store.get_configuration(managed.identity))
    assert error.value.code == "TRAINING_RESOURCES_UNAVAILABLE"


def test_failed_enqueue_is_recorded_and_the_same_operation_resumes(managed):
    context = managed
    original = context.client.enqueue
    calls = []

    def failing(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise StorageError("store offline", "TASK_CENTER_UNAVAILABLE", 503)
        return original(*args, **kwargs)

    context.client.enqueue = failing
    with pytest.raises(StorageError) as failure:
        context.service.launch(context.identity, "launch")
    assert failure.value.code == "TRAINING_LAUNCH_FAILED"
    state = read_json(context.folder / "state.json")
    assert state["status"] == "failed"
    assert state["findings"][0]["code"] == "TRAINING_LAUNCH_FAILED"
    assert not (context.folder / "operations.json").exists()
    retried = context.service.launch(context.identity, "launch")
    assert retried["status"] == "queued" and len(calls) == 2
    assert {run["attempt"] for run in retried["runs"]} == {2}
    assert len(group_tasks(context)) == 6


# -- runner end to end ---------------------------------------------------------------------


@pytest.mark.slow
def test_managed_batch_runs_every_fold_and_collects_pinned_results(managed):
    context = managed
    context.service.launch(context.identity, "launch")
    runner = make_runner(context)
    drive(context, runner, lambda: settled(context))
    tasks = group_tasks(context)
    assert all(task["state"] == "succeeded" for task in tasks), [
        (task["kind"], task["state"], task["error"]) for task in tasks
    ]
    assert_completed_evidence(context)
    fold_ids = {task["id"] for task in tasks if task["kind"] == "mil-fold"}
    measured = {
        row["taskId"]
        for row in context.store.measurements(kinds=("mil-fold",))
        if row["exitReason"] == "ok"
    }
    assert measured == fold_ids
    assert not (context.folder / "cancel.json").exists()
    with pytest.raises(StorageError) as done:
        context.service.launch(context.identity, "resume-done", resume=True)
    assert done.value.code == "TRAINING_NOT_RESUMABLE"


@pytest.mark.slow
def test_cancel_stops_running_folds_cancels_pending_ones_and_resume_completes(managed):
    context = managed
    context.store.update_settings({"cpuTaskSlots": 1})
    context.service.launch(context.identity, "launch")
    runner = make_runner(context)
    drive(context, runner, lambda: fold_running(context))
    cancelled = context.service.cancel(context.identity, "cancel")
    assert cancelled["cancelRequested"] and cancelled["status"] in {"queued", "running"}
    assert context.service.cancel(context.identity, "cancel")["cancelRequested"]
    states = {task["id"]: task["state"] for task in group_tasks(context)}
    raw = read_json(context.folder / "state.json")
    pending = [run for run in raw["runs"] if run["status"] == "cancelled"]
    assert len(pending) >= 3
    assert {run["error"] for run in pending} == {"Cancelled before start."}
    key = task_folder(context.folder)
    assert {states[ids.fold_task_id(key, run["id"])] for run in pending} == {"cancelled"}
    assert "stopping" in states.values() or "running" in states.values()

    drive(context, runner, lambda: settled(context))
    execution = context.service.execution(context.identity)
    assert execution["status"] == "cancelled"
    assert set(execution["runCounts"]) and execution["runCounts"]["queued"] == 0
    assert execution["runCounts"]["running"] == 0
    assert read_json(context.folder / "results.json")["status"] == "cancelled"
    final = context.store.get(ids.collect_task_id(key, True))
    assert final["state"] == "succeeded"

    context.store.update_settings({"cpuTaskSlots": 8})
    resumed = context.service.launch(context.identity, "resume", resume=True)
    assert resumed["status"] == "queued" and not (context.folder / "cancel.json").exists()
    assert any(task["state"] in LIVE for task in group_tasks(context))
    assert context.store.get(final["id"])["state"] == "blocked"
    drive(context, runner, lambda: settled(context))
    state = assert_completed_evidence(context)
    resumed_ids = {run["id"] for run in pending}
    assert {run["attempt"] for run in state["runs"] if run["id"] in resumed_ids} == {2}


@pytest.mark.slow
def test_single_run_cancels_finish_the_batch_cancelled_and_resume_reruns_them(managed):
    context = managed
    context.store.update_settings({"cpuTaskSlots": 1})
    context.service.launch(context.identity, "launch")
    runner = make_runner(context)
    drive(context, runner, lambda: fold_running(context))
    folds = [task for task in group_tasks(context) if task["kind"] == "mil-fold"]
    [running] = [task for task in folds if task["state"] == "running"]
    queued = [task for task in folds if task["state"] == "queued"]
    stopped, pending, store_only = (
        task["adapterData"]["runId"] for task in (running, queued[-1], queued[-2])
    )
    view = context.service.cancel_runs(context.identity, [pending, stopped, pending], "cancel-2")
    runs = {run["id"]: run for run in view["runs"]}
    assert (runs[pending]["status"], runs[pending]["error"]) == (
        "cancelled",
        "Cancelled before start.",
    )
    assert context.store.get(queued[-1]["id"])["state"] == "cancelled"
    stopping = context.store.get(running["id"])
    assert (stopping["state"], stopping["stopRequest"]) == ("stopping", "cancel")
    # Only these runs stop: no batch marker, and the rest of the batch continues.
    assert not view["cancelRequested"] and not (context.folder / "cancel.json").exists()
    assert view["status"] in {"queued", "running"}
    assert context.store.get(queued[0]["id"])["state"] == "queued"
    assert context.service.cancel_runs(context.identity, [stopped, pending], "cancel-2")["runs"]
    with pytest.raises(StorageError) as conflict:
        context.service.cancel_runs(context.identity, [store_only], "cancel-2")
    assert conflict.value.code == "OPERATION_CONFLICT"
    with pytest.raises(StorageError) as unknown:
        context.service.cancel_runs(context.identity, ["no-such-run"], "cancel-unknown")
    assert (unknown.value.code, unknown.value.status_code) == ("TRAINING_RUN_NOT_FOUND", 404)
    # A store-only task cancel has no batch hook; the final collection records it.
    context.client.cancel_task(queued[-2]["id"])

    context.store.update_settings({"cpuTaskSlots": 8})
    drive(context, runner, lambda: settled(context))
    state = read_json(context.folder / "state.json")
    runs = {run["id"]: run for run in state["runs"]}
    assert (runs[stopped]["status"], runs[stopped]["error"]) == (
        "cancelled",
        "Cancelled while running.",
    )
    assert runs[pending]["error"] == runs[store_only]["error"] == "Cancelled before start."
    assert state["runCounts"]["cancelled"] == 3 and state["runCounts"]["completed"] == 2
    assert state["status"] == "cancelled"
    assert read_json(context.folder / "results.json")["status"] == "cancelled"
    execution = context.service.execution(context.identity)
    assert execution["status"] == "cancelled" and not execution["cancelRequested"]

    resumed = context.service.launch(context.identity, "resume", resume=True)
    assert {run["id"] for run in resumed["runs"] if run["attempt"] == 2} == {
        stopped,
        pending,
        store_only,
    }
    drive(context, runner, lambda: settled(context))
    assert_completed_evidence(context)


def lose(task):
    """Lose a task with its wrapper, as a machine restart does: no exit status survives."""
    os.kill(task["process"]["wrapper"]["pid"], signal.SIGKILL)
    os.killpg(task["process"]["pid"], signal.SIGKILL)


@pytest.mark.slow
def test_lost_runner_interrupts_and_auto_resumes_the_running_fold(managed):
    context = managed
    context.store.update_settings({"cpuTaskSlots": 1})
    context.service.launch(context.identity, "launch")
    first = make_runner(context)
    drive(context, first, lambda: fold_running(context))
    [task] = [
        task
        for task in group_tasks(context)
        if task["kind"] == "mil-fold" and task["state"] == "running"
    ]
    # The host lost the fold and its runner (for example a WSL restart).
    lose(task)
    first._procs[task["id"]].wait(timeout=30)
    first.close()  # its runner lock went with it
    second = make_runner(context)
    resumed = context.store.get(task["id"])
    assert resumed["state"] == "queued" and resumed["attempt"] == 2
    assert any(
        (event["detail"] or {}).get("autoResumed") for event in context.store.events(task["id"])
    )
    run_id = task["adapterData"]["runId"]
    run = next(
        row for row in read_json(context.folder / "state.json")["runs"] if row["id"] == run_id
    )
    assert run["status"] == "queued" and run["attempt"] == 2 and "process" not in run
    [event] = [event for event in attempt_events(context) if event.get("runId") == run_id]
    assert (event["action"], event["attempt"]) == ("auto-resume", 2)
    assert event["provenance"]["host"]["bootId"]
    view = context.service.execution(context.identity)
    assert view["status"] in {"queued", "running"}
    context.store.update_settings({"cpuTaskSlots": 8})
    drive(context, second, lambda: settled(context))
    assert_completed_evidence(context)


@pytest.mark.slow
def test_a_fold_lost_in_a_restart_resumes_once_its_runtime_answers(managed, monkeypatch):
    from histopilot.taskcenter.adapters import mil

    context = managed
    context.store.update_settings({"cpuTaskSlots": 1})
    context.service.launch(context.identity, "launch")
    first = make_runner(context)
    drive(context, first, lambda: fold_running(context))
    [task] = [
        task
        for task in group_tasks(context)
        if task["kind"] == "mil-fold" and task["state"] == "running"
    ]
    lose(task)
    first._procs[task["id"]].wait(timeout=30)
    first.close()
    probes = []

    def cold_then_warm(**_options):
        probes.append(1)
        if len(probes) == 1:
            raise RuntimeError("probe timed out after 45 s")  # a cold import after a reboot
        return runtime()

    skew = [0.0]
    second = make_runner(context, probe=cold_then_warm, clock=lambda: time.monotonic() + skew[0])
    assert probes == [1] and context.store.get(task["id"])["state"] == "interrupted"
    # The undecided fold may still resume, so the batch is not finalized without it.
    final = context.store.get(ids.collect_task_id(task_folder(context.folder), True))
    assert final["state"] == "blocked"
    skew[0] = 3600.0  # the runner's and the adapter's retry pauses have passed
    monkeypatch.setattr(mil, "RUNTIME_RETRY_SECONDS", 0.0)
    context.store.update_settings({"cpuTaskSlots": 8})
    drive(context, second, lambda: settled(context))
    resumed = context.store.get(task["id"])
    assert (resumed["state"], resumed["attempt"]) == ("succeeded", 2)
    assert assert_completed_evidence(context)


@pytest.mark.slow
def test_stop_and_hold_requeues_running_folds_until_release(managed):
    context = managed
    context.service.launch(context.identity, "launch")
    runner = make_runner(context)
    drive(context, runner, lambda: fold_running(context))
    running = [
        task
        for task in group_tasks(context)
        if task["kind"] == "mil-fold" and task["state"] == "running"
    ]
    context.store.hold_owner(running[0]["ownerKey"], True)
    context.store.request_stop([task["id"] for task in running], "pause")
    drive(
        context,
        runner,
        lambda: (
            not any(
                task["kind"] == "mil-fold" and task["state"] not in {"queued", "succeeded"}
                for task in group_tasks(context)
            )
        ),
    )
    state = read_json(context.folder / "state.json")
    runs = {run["id"]: run for run in state["runs"]}
    paused = [context.store.get(task["id"]) for task in running]
    requeued = [task for task in paused if task["state"] == "queued"]
    assert requeued and all(task["attempt"] == 2 for task in requeued)
    events = {event.get("runId"): event for event in attempt_events(context)}
    for task in requeued:
        run = runs[task["adapterData"]["runId"]]
        assert run["status"] == "queued" and run["attempt"] == 2 and "process" not in run
        assert (events[run["id"]]["action"], events[run["id"]]["attempt"]) == ("pause", 2)
    # Pausing is not cancelling: no marker, and the batch waits instead of finishing.
    assert not (context.folder / "cancel.json").exists()
    view = context.service.execution(context.identity)
    assert view["status"] == "queued" and view["taskCenter"]["held"]
    context.store.hold_owner(running[0]["ownerKey"], False)
    drive(context, runner, lambda: settled(context))
    assert_completed_evidence(context)


# -- adapters ------------------------------------------------------------------------------


def fabricated_batch(tmp_path, *run_ids):
    folder = tmp_path / "project" / "training" / "batch"
    for run_id in run_ids:
        (folder / "runs" / run_id).mkdir(parents=True)
    plan = {
        "batchId": "batch",
        "runtime": {"versions": {}},
        "runs": [{"id": run_id} for run_id in run_ids],
    }
    write_json(folder / "plan.json", plan)
    save_state(
        folder,
        {
            "batchId": "batch",
            "status": "running",
            "planHash": _hash(plan),
            "runs": [{"id": run_id, "status": "running", "attempt": 1} for run_id in run_ids],
        },
    )
    return folder


def fabricated_task(folder, run_id=None, **fields):
    data = {"batchFolder": str(folder), "batchId": "batch"}
    data.update({"runId": run_id, "final": False} if run_id else {"final": True})
    return {
        "id": "task-" + (run_id or "collect"),
        "kind": "mil-fold" if run_id else "mil-collect",
        "attempt": 1,
        "ownerKey": "owner-unknown",
        "group": {"kind": "mil-batch", "id": "batch"},
        "projectFolder": str(folder.parents[1]),
        "queuedAt": "2026-01-01T00:00:01+00:00",
        "startedAt": "2026-01-01T00:00:02+00:00",
        "command": {"log": str(folder / "run.log")},
        "adapterData": data,
        **fields,
    }


def exit_record(returncode=0, *, stop=None, lost=False):
    return {
        "returncode": returncode,
        "lost": lost,
        "signalled": False,
        "killed": False,
        "stopReason": stop,
    }


def test_fold_outcomes_come_from_receipts_never_from_the_exit_code(tmp_path, task_center):
    folder = fabricated_batch(tmp_path, "done", "sigterm", "oom", "stale", "lost", "paused")
    ctx = task_center.context()
    adapter = MilFoldAdapter(runtime=runtime)
    result = {
        "runId": "done",
        "state": "succeeded",
        "metrics": {"validation": {}},
        "bestCheckpointPath": "best.ckpt",
        "cudaPeakReservedBytes": 2**30,
        "epochsCompleted": 3,
    }
    write_json(folder / "runs" / "done" / "result.json", result)
    decision = adapter.on_exit(fabricated_task(folder, "done"), exit_record(0), ctx)
    assert decision["state"] == "succeeded"
    assert decision["measurement"] == {"peakVramGb": 1.0, "epochs": 3}
    # SIGTERM during fitting exits 0 without a result: that is not success.
    decision = adapter.on_exit(fabricated_task(folder, "sigterm"), exit_record(0), ctx)
    assert decision["state"] == "failed" and decision["exitReason"] == "error"
    write_json(
        folder / "runs" / "oom" / "failure.json",
        {"error": "CUDA out of memory.", "category": "out_of_memory", "at": utc_now_iso()},
    )
    decision = adapter.on_exit(fabricated_task(folder, "oom"), exit_record(1), ctx)
    assert (decision["state"], decision["exitReason"]) == ("failed", "oom")
    write_json(
        folder / "runs" / "stale" / "failure.json",
        {"error": "old attempt", "category": "out_of_memory", "at": "2025-01-01T00:00:00+00:00"},
    )
    decision = adapter.on_exit(fabricated_task(folder, "stale"), exit_record(1), ctx)
    assert (decision["state"], decision["exitReason"]) == ("failed", "error")
    decision = adapter.on_exit(fabricated_task(folder, "lost"), exit_record(None, lost=True), ctx)
    assert decision["state"] == "interrupted"
    decision = adapter.on_exit(fabricated_task(folder, "paused"), exit_record(0, stop="pause"), ctx)
    assert decision["state"] == "requeue"
    state = read_json(folder / "state.json")
    runs = {run["id"]: run for run in state["runs"]}
    assert runs["done"]["status"] == "completed" and runs["done"]["result"] == result
    assert runs["sigterm"]["error"] == "Training process exited with code 0. See run.log."
    assert runs["oom"]["failureCategory"] == "out_of_memory"
    assert runs["stale"]["error"] == "Training process exited with code 1. See run.log."
    assert runs["lost"]["status"] == "interrupted" and runs["paused"]["status"] == "queued"
    assert len((folder / "events.jsonl").read_text().splitlines()) == 3

    assert adapter.can_requeue(fabricated_task(folder, "lost"), ctx)
    assert not adapter.can_requeue(fabricated_task(folder, "done"), ctx)
    adapter.on_requeue(fabricated_task(folder, "lost"), ctx)
    lost = next(run for run in read_json(folder / "state.json")["runs"] if run["id"] == "lost")
    assert lost["status"] == "queued" and lost["attempt"] == 2 and "error" not in lost
    assert adapter.prepare(fabricated_task(folder, "done"), ctx)["skip"]["exitReason"] == (
        "already-complete"
    )
    write_json(folder / "cancel.json", {"requestedAt": utc_now_iso(), "operationId": "cancel"})
    assert not adapter.can_requeue(fabricated_task(folder, "lost"), ctx)
    assert adapter.prepare(fabricated_task(folder, "lost"), ctx)["skip"]["state"] == "cancelled"
    lost = next(run for run in read_json(folder / "state.json")["runs"] if run["id"] == "lost")
    assert lost["status"] == "cancelled" and lost["error"] == "Cancelled before start."
    decision = adapter.on_exit(fabricated_task(folder, "sigterm"), exit_record(0), ctx)
    assert decision["state"] == "cancelled"


def test_final_collection_applies_only_its_own_receipt(tmp_path, task_center):
    folder = fabricated_batch(tmp_path, "run")
    ctx = task_center.context()
    adapter = MilCollectAdapter()
    final = fabricated_task(folder)
    assert adapter.on_exit(final, exit_record(75), ctx)["exitReason"] == "busy"
    write_json(
        folder / "collect-result.json",
        {"status": "running", "final": False, "at": utc_now_iso(), "error": None},
    )
    decision = adapter.on_exit(final, exit_record(1), ctx)
    assert decision["state"] == "failed"
    state = read_json(folder / "state.json")
    assert state["status"] == "failed"
    assert state["findings"][0]["code"] == "TRAINING_RESULTS_FAILED"
    assert state["runs"][0]["status"] == "interrupted"
    write_json(
        folder / "collect-result.json",
        {"status": "cancelled", "final": True, "at": utc_now_iso(), "error": None},
    )
    assert adapter.on_exit(final, exit_record(0), ctx)["state"] == "succeeded"
    assert read_json(folder / "state.json")["status"] == "cancelled"
    assert adapter.can_requeue(final, ctx)
    assert final_status({"runs": [{"status": "completed"}]}, cancel_requested=True) == "cancelled"
    assert (
        final_status(
            {"runs": [{"status": "completed"}, {"status": "failed"}]}, cancel_requested=False
        )
        == "failed"
    )
    assert final_status({"runs": [{"status": "completed"}]}, cancel_requested=False) == (
        "completed"
    )
    assert final_status({"runs": [{"status": "queued"}]}, cancel_requested=False) == "interrupted"


def test_final_collection_superseded_by_requeued_folds_collects_again(tmp_path, task_center):
    folder = fabricated_batch(tmp_path, "run")
    ctx = task_center.context()
    adapter = MilCollectAdapter()
    final = fabricated_task(folder, queuedAt=utc_now_iso())
    # A collection that finishes before the runner records startedAt is still this attempt's.
    write_json(
        folder / "collect-result.json",
        {"status": "failed", "final": True, "at": utc_now_iso(), "error": None},
    )
    final["startedAt"] = utc_now_iso()
    assert adapter.on_exit(final, exit_record(0), ctx)["state"] == "succeeded"
    assert read_json(folder / "state.json")["status"] == "failed"

    # A resume requeued a fold (under the batch lock) while this collection was running.
    state = read_json(folder / "state.json")
    # Resume writes a fresh state without the previous attempt's finishedAt.
    state.pop("finishedAt")
    state.update(status="queued", runs=[{"id": "run", "status": "queued", "attempt": 2}])
    save_state(folder, state)
    ctx.store.enqueue(
        {
            "kind": "mil-batch",
            "id": "batch",
            "projectId": "project",
            "projectFolder": final["projectFolder"],
            "title": "batch",
        },
        [
            {
                "id": "task-run",
                "kind": "mil-fold",
                "adapter": "mil-fold",
                "group": {"kind": "mil-batch", "id": "batch"},
                "request": {"lane": "cpu"},
                "command": {
                    "argv": [sys.executable, "-c", "pass"],
                    "cwd": str(folder),
                    "log": str(folder / "runs" / "run" / "run.log"),
                },
            }
        ],
    )
    for exit in (exit_record(0), exit_record(None, lost=True), exit_record(None)):
        decision = adapter.on_exit(final, exit, ctx)
        assert (decision["state"], decision["exitReason"]) == ("requeue", "busy")
    state = read_json(folder / "state.json")
    assert state["status"] == "queued" and state["runs"][0]["status"] == "queued"
    assert "finishedAt" not in state


def test_prepare_adopts_only_an_exact_result_left_by_a_lost_runner(tmp_path, task_center):
    folder = fabricated_batch(tmp_path, "done", "other")
    ctx = task_center.context()
    adapter = MilFoldAdapter(runtime=runtime)
    result = {
        "runId": "done",
        "state": "succeeded",
        "metrics": {"validation": {}},
        "bestCheckpointPath": "best.ckpt",
    }
    write_json(folder / "runs" / "done" / "result.json", result)
    # Another run's receipt in this folder is not this run's success.
    write_json(folder / "runs" / "other" / "result.json", result)
    skip = adapter.prepare(fabricated_task(folder, "done"), ctx)["skip"]
    assert (skip["state"], skip["exitReason"]) == ("succeeded", "already-complete")
    assert adapter.prepare(fabricated_task(folder, "other", request={"lane": "cpu"}), ctx) is None
    runs = {run["id"]: run for run in read_json(folder / "state.json")["runs"]}
    assert runs["done"]["status"] == "completed"
    assert runs["done"]["result"] == read_json(folder / "runs" / "done" / "result.json")
    assert runs["other"]["status"] == "running" and "result" not in runs["other"]


POST_FIT_ERROR = "The verified feature pack changed after the batch was launched."


def test_a_result_the_worker_later_failed_is_never_success(tmp_path, monkeypatch, task_center):
    from histopilot.workers import train_batch

    folder = fabricated_batch(tmp_path, "run")
    run_folder = folder / "runs" / "run"
    write_json(run_folder / "plan.json", {"code": None, "data": {"sourceStamps": {}}})
    checks = []

    def check_inputs(_data):
        checks.append(1)
        if len(checks) == 2:  # the post-fit check: a source changed while the fold trained
            raise ValueError(POST_FIT_ERROR)

    def train_fold(plan, output_dir, *, checkpoint_path=None):
        # The fit writes its receipt itself, before the worker's post-fit check.
        result = {
            "runId": "run",
            "state": "succeeded",
            "metrics": {"validation": {"auroc": 0.9}},
            "bestCheckpointPath": str(output_dir / "best.ckpt"),
        }
        write_json(output_dir / "result.json", result)
        return result

    monkeypatch.setattr(train_batch, "_check_inputs", check_inputs)
    monkeypatch.setitem(
        sys.modules, "histopilot.training.fold", SimpleNamespace(train_fold=train_fold)
    )
    with pytest.raises(ValueError):
        train_batch.run_fold_worker(run_folder / "plan.json")
    assert (run_folder / "result.json").exists()
    ctx = task_center.context()
    adapter = MilFoldAdapter(runtime=runtime)
    # Legacy failed this fold; neither its exit code nor an unobservable one makes it success.
    for code in (1, 0, None):
        decision = adapter.on_exit(fabricated_task(folder, "run"), exit_record(code), ctx)
        assert (decision["state"], decision["error"]) == ("failed", POST_FIT_ERROR)
    run = read_json(folder / "state.json")["runs"][0]
    assert run["status"] == "failed" and "result" not in run and "metrics" not in run
    # A later attempt that exits cleanly without a new result does not inherit the old one.
    later = fabricated_task(folder, "run", queuedAt=utc_now_iso())
    decision = adapter.on_exit(later, exit_record(0), ctx)
    assert decision["state"] == "failed" and "exited with code 0" in decision["error"]
    decision = adapter.on_exit(later, exit_record(None, lost=True), ctx)
    assert decision["state"] == "interrupted"
    # Retries and automatic resumes do not adopt it without training either.
    retry = fabricated_task(folder, "run", request={"lane": "cpu"})
    assert adapter.prepare(retry, ctx) is None
    assert read_json(folder / "state.json")["runs"][0]["status"] == "interrupted"
    # A receipt written after the failure (a later attempt succeeded) is adopted.
    failed_at = (run_folder / "failure.json").stat().st_mtime_ns
    os.utime(run_folder / "result.json", ns=(failed_at + 10**9, failed_at + 10**9))
    assert adapter.prepare(retry, ctx)["skip"]["exitReason"] == "already-complete"
    assert read_json(folder / "state.json")["runs"][0]["status"] == "completed"


def test_a_cancel_that_races_a_pause_is_never_requeued(tmp_path, task_center):
    from histopilot.storage.project_lock import writer_lock

    folder = fabricated_batch(tmp_path, "upgraded", "paused", "single")
    ctx = task_center.context()
    adapter = MilFoldAdapter(runtime=runtime)
    upgraded = fabricated_task(folder, "upgraded")
    ctx.store.enqueue(
        {
            "kind": "mil-batch",
            "id": "batch",
            "projectId": "project",
            "projectFolder": upgraded["projectFolder"],
            "title": "batch",
        },
        [
            {
                "id": upgraded["id"],
                "kind": "mil-fold",
                "adapter": "mil-fold",
                "group": upgraded["group"],
                "request": {"lane": "cpu"},
                "command": {
                    "argv": [sys.executable, "-c", "pass"],
                    "cwd": str(folder),
                    "log": str(folder / "runs" / "upgraded" / "run.log"),
                },
                "adapterData": upgraded["adapterData"],
            }
        ],
    )
    assert ctx.store.transition(upgraded["id"], from_states=("queued",), to_state="running")
    ctx.store.request_stop([upgraded["id"]], "pause")
    # A single-run cancel upgraded the pause while the runner's listing still said pause.
    ctx.store.request_stop([upgraded["id"]], "cancel")
    decision = adapter.on_exit(upgraded, exit_record(0, stop="pause"), ctx)
    assert decision["state"] == "cancelled"

    # A cancelled run is skipped without waiting for the batch lock, and never requeued.
    state = read_json(folder / "state.json")
    state["runs"][2].update(status="cancelled", error="Cancelled before start.")
    save_state(folder, state)
    with writer_lock(folder):
        skip = adapter.prepare(fabricated_task(folder, "single"), ctx)["skip"]
    assert (skip["state"], skip["exitReason"]) == ("cancelled", "cancelled")
    with pytest.raises(AdapterError) as refused:
        adapter.on_requeue(fabricated_task(folder, "single"), ctx)
    assert refused.value.fatal

    # The batch cancel marker outranks a pause both on exit and on requeue.
    write_json(folder / "cancel.json", {"requestedAt": utc_now_iso(), "operationId": "cancel"})
    decision = adapter.on_exit(fabricated_task(folder, "paused"), exit_record(0, stop="pause"), ctx)
    assert decision["state"] == "cancelled"
    state = read_json(folder / "state.json")
    state["runs"][1]["status"] = "queued"  # as a pause concluded just before the marker
    save_state(folder, state)
    with pytest.raises(AdapterError) as refused:
        adapter.on_requeue(fabricated_task(folder, "paused", exit={"reason": "paused"}), ctx)
    assert refused.value.fatal
    runs = {run["id"]: run for run in read_json(folder / "state.json")["runs"]}
    assert runs["paused"]["status"] == "cancelled" and runs["paused"]["attempt"] == 1
    assert runs["single"]["error"] == "Cancelled before start."
    assert not (folder / "attempts.jsonl").exists()


PAUSABLE = r"""
import pathlib, signal, sys, time
folder = pathlib.Path(sys.argv[1])
def stop(*_):
    time.sleep(0.5)  # save a checkpoint
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
(folder / "started").write_text("started")
while True:
    time.sleep(0.02)
"""


def test_cancel_while_the_runner_concludes_a_paused_fold_cancels_it(tmp_path, task_center):
    import threading

    from histopilot.storage.project_lock import writer_lock
    from histopilot.taskcenter import procs

    folder = fabricated_batch(tmp_path, "r1")
    state = read_json(folder / "state.json")
    state.update(status="queued", runs=[{"id": "r1", "status": "queued", "attempt": 1}])
    save_state(folder, state)
    project = str(folder.parents[1])
    client = task_center.client
    context = SimpleNamespace(
        center=task_center,
        store=task_center.store,
        client=client,
        identity="batch",
        development=SimpleNamespace(store=SimpleNamespace(folder=project)),
        messages=task_center.logs,
    )
    work = tmp_path / "work"
    work.mkdir()
    owner = {"kind": "experiment", "id": "E2", "projectId": "p", "projectFolder": project}
    enqueued = client.enqueue(
        {**owner, "title": "E2"},
        [
            {
                "id": "fold-r1",
                "kind": "mil-fold",
                "adapter": "mil-fold",
                "title": "fold",
                "group": {"kind": "mil-batch", "id": "batch"},
                "request": {"lane": "cpu", "cpuThreads": 1, "ramGb": 0.1},
                "command": {
                    "argv": [sys.executable, "-c", PAUSABLE, str(work)],
                    "cwd": str(tmp_path),
                    "log": str(work / "run.log"),
                },
                "adapterData": {
                    "batchFolder": str(folder),
                    "runId": "r1",
                    "batchId": "batch",
                    "final": False,
                },
            }
        ],
    )
    runner = make_runner(context)
    drive(context, runner, lambda: (work / "started").exists(), timeout=30)
    # "Stop & hold" on the experiment, then the fold saves its checkpoint and exits.
    context.store.hold_owner(enqueued["owner"]["key"], True)
    assert context.store.request_stop(["fold-r1"], "pause") == ["fold-r1"]
    runner.tick()
    deadline = time.monotonic() + 30
    while procs.leader_alive(context.store.get("fold-r1")["process"]):
        assert time.monotonic() < deadline
        time.sleep(0.02)
    locked = threading.Event()

    def cancel():
        # What TrainingService._cancel_managed does under the batch writer lock.
        with writer_lock(folder, timeout=5):
            locked.set()
            time.sleep(0.5)
            write_json(folder / "cancel.json", {"requestedAt": utc_now_iso()})
            for _ in range(2):
                client.cancel_group("mil-batch", "batch", project, exclude_kinds=("mil-collect",))

    thread = threading.Thread(target=cancel)
    thread.start()
    assert locked.wait(5)
    runner.tick()  # lists the fold as paused, then waits for the lock in on_exit
    thread.join()
    task = context.store.get("fold-r1")
    state = read_json(folder / "state.json")
    assert task["state"] == "cancelled" and task["attempt"] == 1
    assert state["runs"][0]["status"] == "cancelled"


@pytest.mark.parametrize("recorded", [True, False], ids=["single-run-cancel", "store-only"])
def test_a_cancel_during_a_pause_requeue_ends_the_task_cancelled(tmp_path, task_center, recorded):
    folder = fabricated_batch(tmp_path, "r1")
    store = task_center.store
    fold = fabricated_task(folder, "r1")
    store.enqueue(
        {
            "kind": "experiment",
            "id": "E2",
            "projectId": "project",
            "projectFolder": fold["projectFolder"],
            "title": "E2",
        },
        [
            {
                "id": fold["id"],
                "kind": "mil-fold",
                "adapter": "mil-fold",
                "group": fold["group"],
                "request": {"lane": "cpu"},
                "command": {
                    "argv": [sys.executable, "-c", "pass"],
                    "cwd": str(folder),
                    "log": str(folder / "runs" / "r1" / "run.log"),
                },
                "adapterData": fold["adapterData"],
            }
        ],
    )
    assert store.transition(fold["id"], from_states=("queued",), to_state="running")
    # "Stop & hold" concluded the fold: its run waits queued and the task awaits its requeue.
    state = read_json(folder / "state.json")
    state["runs"][0]["status"] = "queued"
    save_state(folder, state)
    concluded = store.conclude(
        fold["id"],
        to_state="interrupted",
        stop_request="pause",
        follow_up={"hook": "on_requeue", "reason": "paused", "since": utc_now_iso()},
        exit={"reason": "paused", "returncode": 0},
    )
    assert concluded == "concluded"

    class CancelledMeanwhile(MilFoldAdapter):
        def on_requeue(self, task, ctx):
            # A cancel lands while this hook waits for the batch lock: TrainingService.cancel_runs
            # also records the run, a store-only cancel does not.
            if store.get(task["id"])["stopRequest"] is None:
                store.request_stop([task["id"]], "cancel")
                if recorded:
                    current = read_json(folder / "state.json")
                    current["runs"][0].update(status="cancelled", error="Cancelled before start.")
                    save_state(folder, current)
            return super().on_requeue(task, ctx)

    adapters = {"mil-fold": CancelledMeanwhile(runtime=runtime), "mil-collect": MilCollectAdapter()}
    runner = task_center.runner(
        host_probe=lambda: capacity.host(gpu_probe=lambda: {"gpus": []}),
        adapters=lambda name: adapters.get(name) or registered_adapter(name),
    )
    runner.tick()
    task = store.get(fold["id"])
    run = read_json(folder / "state.json")["runs"][0]
    # The Task Center and the batch record agree: cancelled, never "interrupted" with an error,
    # and no attempt is recorded for a requeue that never happened.
    assert (task["state"], task["attempt"], task["error"]) == ("cancelled", 1, None)
    assert (run["status"], run["attempt"]) == ("cancelled", 1)
    assert run["error"] == "Cancelled while running."
    assert not (folder / "attempts.jsonl").exists()


def test_requeued_attempts_record_their_host_provenance(tmp_path, monkeypatch, task_center):
    import json

    from histopilot.taskcenter.adapters import mil
    from histopilot.workers.training_process import host_snapshot

    folder = fabricated_batch(tmp_path, "paused", "oom", "lost", "pending", "unknown")
    host = host_snapshot()
    gpus = [{"index": 0, "uuid": "GPU-a", "name": "RTX A5000", "driverVersion": "580.1"}]
    state = read_json(folder / "state.json")
    state.update(findings=[], provenance={"at": utc_now_iso(), "host": host, "gpus": gpus})
    save_state(folder, state)
    monkeypatch.setattr(mil, "gpu_snapshot", lambda: {"gpus": gpus})
    adapter = MilFoldAdapter(runtime=runtime)

    def requeue(run_id, **fields):
        # A fresh context per call: the provenance probe is cached for one runner tick.
        adapter.on_requeue(fabricated_task(folder, run_id, **fields), task_center.context())

    def host_changed():
        return [
            item
            for item in read_json(folder / "state.json")["findings"]
            if item["code"] == "TRAINING_HOST_CHANGED"
        ]

    requeue("paused", exit={"reason": "paused"})
    requeue("oom", exit={"reason": "oom"})
    assert host_changed() == []
    monkeypatch.setattr(mil, "gpu_snapshot", lambda: {"gpus": [], "gpuProbeError": "timed out"})
    requeue("unknown", exit={"reason": "paused"})
    assert host_changed() == []  # an unreadable driver is unknown, not changed
    # A restart with a driver update: the automatic resume records the new provenance.
    updated = [{**gpus[0], "driverVersion": "581.0"}]
    monkeypatch.setattr(mil, "gpu_snapshot", lambda: {"gpus": updated})
    monkeypatch.setattr(mil, "host_snapshot", lambda: {**host, "bootId": "after-reboot"})
    requeue("lost", exit={"reason": "lost"})
    requeue("pending", bookkeeping={"hook": "requeue_intent", "reason": "auto-resume"})
    events = [json.loads(line) for line in (folder / "attempts.jsonl").read_text().splitlines()]
    assert [(event["action"], event["runId"], event["attempt"]) for event in events] == [
        ("pause", "paused", 2),
        ("oom-backoff", "oom", 2),
        ("pause", "unknown", 2),
        ("auto-resume", "lost", 2),
        ("auto-resume", "pending", 2),
    ]
    assert events[0]["provenance"]["host"]["bootId"] == host["bootId"]
    assert events[0]["provenance"]["gpus"] == gpus
    assert events[3]["provenance"]["host"]["bootId"] == "after-reboot"
    assert events[3]["provenance"]["gpus"] == updated
    assert all(event["planHash"] == state["planHash"] and event["at"] for event in events)
    assert len(host_changed()) == 1
    # The launch provenance stays the reference for runs it recorded.
    assert read_json(folder / "state.json")["provenance"]["gpus"] == gpus


def test_a_retried_requeue_records_its_attempt_once(tmp_path, monkeypatch, task_center):
    import json

    folder = fabricated_batch(tmp_path, "run")
    adapter = MilFoldAdapter(runtime=runtime)
    rearmed = []
    monkeypatch.setattr(adapter, "_rearm_final", lambda task, ctx: rearmed.append(task["attempt"]))
    fold = fabricated_task(folder, "run", exit={"reason": "lost"})

    def events():
        lines = (folder / "attempts.jsonl").read_text().splitlines()
        return [(event["action"], event["attempt"]) for event in map(json.loads, lines)]

    # The store's requeue failed (or the runner died) after this hook, so the runner repeats it.
    for _ in range(2):
        adapter.on_requeue(fold, task_center.context())
    run = read_json(folder / "state.json")["runs"][0]
    assert (run["status"], run["attempt"], run["taskAttempt"]) == ("queued", 2, 2)
    assert events() == [("auto-resume", 2)]
    assert rearmed == [1, 1]  # the final collection is still re-armed by the repeated hook
    # That attempt ran and was lost again: its requeue is a new attempt.
    state = read_json(folder / "state.json")
    state["runs"][0]["status"] = "interrupted"
    save_state(folder, state)
    adapter.on_requeue({**fold, "attempt": 2}, task_center.context())
    run = read_json(folder / "state.json")["runs"][0]
    assert (run["status"], run["attempt"], run["taskAttempt"]) == ("queued", 3, 3)
    assert events() == [("auto-resume", 2), ("auto-resume", 3)]


def test_a_fold_requeued_after_finalization_rearms_the_final_collection(tmp_path, task_center):
    folder = fabricated_batch(tmp_path, "run")
    ctx = task_center.context()
    store, fold = ctx.store, fabricated_task(folder, "run")
    final_id = ids.collect_task_id(str(folder), True)
    command = {
        "argv": [sys.executable, "-c", "pass"],
        "cwd": str(folder),
        "log": str(folder / "batch.log"),
    }
    store.enqueue(
        {
            "kind": "mil-batch",
            "id": "batch",
            "projectId": "project",
            "projectFolder": fold["projectFolder"],
            "title": "batch",
        },
        [
            {
                "id": fold["id"],
                "kind": "mil-fold",
                "adapter": "mil-fold",
                "group": fold["group"],
                "request": {"lane": "cpu"},
                "command": command,
                "adapterData": fold["adapterData"],
            },
            {
                "id": final_id,
                "kind": "mil-collect",
                "adapter": "mil-collect",
                "group": fold["group"],
                "request": {"lane": "cpu"},
                "command": command,
                "adapterData": {"batchFolder": str(folder), "batchId": "batch", "final": True},
                "dependsOn": [{"task": fold["id"], "condition": "terminal"}],
            },
        ],
    )
    # The fold's requeue hook was deferred (a busy batch lock) while the final collection ran.
    assert store.transition(fold["id"], from_states=("queued",), to_state="running")
    assert store.transition(fold["id"], from_states=("running",), to_state="failed")
    assert store.promote_ready() == [final_id]
    assert store.transition(final_id, from_states=("queued",), to_state="running")
    assert store.transition(final_id, from_states=("running",), to_state="succeeded")
    state = read_json(folder / "state.json")
    state.update(status="failed", finishedAt=utc_now_iso())
    state["runs"][0]["status"] = "failed"
    save_state(folder, state)

    MilFoldAdapter(runtime=runtime).on_requeue(store.get(fold["id"]), ctx)
    assert store.get(final_id)["state"] == "blocked"
    state = read_json(folder / "state.json")
    assert state["status"] == "queued" and "finishedAt" not in state
    assert store.requeue([fold["id"]], reason="oom-backoff") == [fold["id"]]
    assert store.promote_ready() == [] and store.get(final_id)["state"] == "blocked"


def pinned_runtime(folder, python, versions):
    """Give a fabricated batch the interpreter and versions its folds were planned with."""
    plan = {**read_json(folder / "plan.json"), "runtime": {"python": python, "versions": versions}}
    write_json(folder / "plan.json", plan)
    state = read_json(folder / "state.json")
    state["planHash"] = _hash(plan)
    save_state(folder, state)


def test_an_unknown_runtime_defers_the_requeue_decision_instead_of_refusing(
    tmp_path, monkeypatch, task_center
):
    from histopilot.taskcenter.adapters import mil

    folder = fabricated_batch(tmp_path, "run", "done")
    python, versions = "/envs/train/bin/python", {"torch": "2.10.0"}
    pinned_runtime(folder, python, versions)
    ctx = task_center.context()
    answers = [
        RuntimeError("probe timed out"),  # e.g. a cold import right after a reboot
        {
            "available": False,
            "versions": {},
            "findings": [{"message": "Training runtime unavailable: No module named 'torch'."}],
        },
        {"available": True, "versions": versions},
    ]
    probes = []

    def probe(*, python, refresh):
        probes.append((python, refresh))
        answer = answers[min(len(probes), len(answers)) - 1]
        if isinstance(answer, Exception):
            raise answer
        return answer

    adapter, task = MilFoldAdapter(runtime=probe), fabricated_task(folder, "run")
    for _ in range(2):
        with pytest.raises(AdapterError) as unknown:
            adapter.can_requeue(task, ctx)
        assert unknown.value.transient
        assert str(unknown.value) == "Cannot check the training runtime yet: probe timed out"
    # The fold's own interpreter is probed, and a failed probe is not repeated every tick.
    assert probes == [(python, False)]
    monkeypatch.setattr(mil, "RUNTIME_RETRY_SECONDS", -1.0)
    with pytest.raises(AdapterError) as unavailable:
        adapter.can_requeue(task, ctx)
    assert unavailable.value.transient and "No module named 'torch'" in str(unavailable.value)
    assert adapter.can_requeue(task, ctx) is True and adapter.can_requeue(task, ctx) is True
    assert probes == [(python, False), (python, True), (python, True)]
    # Definitive answers need no probe: finished, cancelled or changed-plan folds never resume.
    assert adapter.can_requeue(fabricated_task(folder, "done"), ctx) is True
    state = read_json(folder / "state.json")
    state["runs"][1]["status"] = "cancelled"
    save_state(folder, state)
    assert adapter.can_requeue(fabricated_task(folder, "done"), task_center.context()) is False
    assert len(probes) == 3

    def other(*, python, refresh):
        return {"available": True, "versions": {"torch": "2.9.0"}}

    # Only known, different versions refuse; an interpreter that answered is known even when
    # a legacy requirement such as tmux marked the runtime unavailable.
    assert MilFoldAdapter(runtime=other).can_requeue(task, task_center.context()) is False

    def no_tmux(*, python, refresh):
        return {"available": False, "versions": versions, "findings": [{"message": "tmux"}]}

    assert MilFoldAdapter(runtime=no_tmux).can_requeue(task, task_center.context()) is True


def test_folds_waiting_on_one_slow_runtime_probe_wait_once_per_tick(tmp_path, task_center):
    """After a reboot every lost fold asks whether it may resume while one cold probe runs;
    the runner loop waits for that probe once per tick, not once per fold."""
    import dataclasses
    import threading

    runs = [f"run{index}" for index in range(6)]
    folder = fabricated_batch(tmp_path, *runs)
    python, versions = "/envs/train/bin/python", {"torch": "2.10.0"}
    pinned_runtime(folder, python, versions)
    ctx = task_center.context()
    release = threading.Event()

    def probe(*, python, refresh):
        release.wait(30)
        return {"available": True, "versions": versions}

    adapter = MilFoldAdapter(runtime=probe)
    try:
        start = time.monotonic()
        for run in runs:
            with pytest.raises(AdapterError) as pending:
                adapter.can_requeue(fabricated_task(folder, run), ctx)
            assert pending.value.retry_after
        assert time.monotonic() - start < 1.5  # one 0.5 s wait, not six
    finally:
        release.set()
    tick = dataclasses.replace(ctx, cache={})  # the next tick; the probe outlives ticks
    assert all(adapter.can_requeue(fabricated_task(folder, run), tick) for run in runs)


def test_training_runtime_probes_the_given_interpreter_once_per_executable(tmp_path, monkeypatch):
    from histopilot.adapters.native import runtime as native

    monkeypatch.setattr(native, "gpu_snapshot", lambda: {"gpus": []})
    calls = tmp_path / "calls"

    def interpreter(name, torch):
        path = tmp_path / name
        answer = f'{{"available": true, "python": "{path}", "versions": {{"torch": "{torch}"}}}}'
        path.write_text(f"#!/bin/sh\necho {name} >> {calls}\necho '{answer}'\n")
        path.chmod(0o755)
        return str(path)

    first, second = interpreter("first", "2.10.0"), interpreter("second", "2.9.0")
    assert native.training_runtime(python=first, refresh=True)["versions"] == {"torch": "2.10.0"}
    assert native.training_runtime(python=first)["python"] == first
    assert native.training_runtime(python=second)["versions"] == {"torch": "2.9.0"}
    assert calls.read_text().split() == ["first", "second"]
    missing = native.training_runtime(python=str(tmp_path / "missing"))
    assert not missing["available"] and missing["findings"][0]["code"] == (
        "TRAINING_RUNTIME_UNAVAILABLE"
    )


def test_contended_batch_lock_is_transient_for_every_state_write(
    tmp_path, monkeypatch, task_center
):
    from histopilot.storage.project_lock import writer_lock
    from histopilot.taskcenter.adapters import mil

    monkeypatch.setattr(mil, "writer_lock", lambda folder, timeout: writer_lock(folder))
    folder = fabricated_batch(tmp_path, "run")
    ctx = task_center.context()
    fold, final = MilFoldAdapter(runtime=runtime), MilCollectAdapter()
    write_json(
        folder / "collect-result.json",
        {"status": "completed", "final": True, "at": utc_now_iso(), "error": None},
    )
    before = (folder / "state.json").read_bytes()
    with writer_lock(folder):
        for hook in (
            lambda: fold.on_started(fabricated_task(folder, "run"), {"pid": 1}, None, ctx),
            lambda: fold.on_exit(fabricated_task(folder, "run"), exit_record(1), ctx),
            lambda: fold.on_requeue(fabricated_task(folder, "run"), ctx),
            lambda: final.on_exit(fabricated_task(folder), exit_record(0), ctx),
        ):
            with pytest.raises(AdapterError) as busy:
                hook()
            assert busy.value.transient
        write_json(folder / "cancel.json", {"requestedAt": utc_now_iso()})
        with pytest.raises(AdapterError) as busy:
            fold.prepare(fabricated_task(folder, "run"), ctx)
        assert busy.value.transient
    assert (folder / "state.json").read_bytes() == before


def test_managed_collect_defers_when_busy_and_never_regresses_final_results(tmp_path):
    from histopilot.workers.managed_collect import BUSY_EXIT, collect
    from histopilot.workers.packing_process import output_lock

    folder = fabricated_batch(tmp_path, "run")
    plan_path = folder / "plan.json"
    with output_lock(folder):
        assert collect(plan_path, final=True) == BUSY_EXIT
    assert not (folder / "collect-result.json").exists()
    state = read_json(folder / "state.json")
    state["status"] = "completed"
    save_state(folder, state)
    assert collect(plan_path, final=False) == 0
    receipt = read_json(folder / "collect-result.json")
    assert receipt["skipped"] and receipt["status"] == "completed" and not receipt["final"]
    assert not (folder / "results.json").exists()
    write_json(plan_path, {**read_json(plan_path), "batchId": "changed"})
    assert collect(plan_path, final=True) == 1
    assert "plan changed" in read_json(folder / "collect-result.json")["error"]


# -- experiments ---------------------------------------------------------------------------


def test_cancelled_task_center_predictors_keep_the_experiment_resumable():
    managed, legacy = {"executionMode": "task-center"}, {}
    assert predictors_settled(managed, {"status": "completed"})
    assert not predictors_settled(managed, {"status": "cancelled"})
    assert not predictors_settled(legacy, {"status": "cancelled", "executor": "task-center"})
    assert predictors_settled(legacy, {"status": "cancelled"})
    assert not predictors_settled(legacy, {"status": "attention"})


def test_experiment_stage_counts_only_completed_task_center_batches():
    rows = [{"id": "batch", "status": "cancelled"}]
    receipt = {"status": "submitted", "batchIds": ["batch"], "publications": []}
    assert experiment_stage(rows, receipt) == ("finished", True)
    managed = {**receipt, "executionMode": "task-center"}
    assert experiment_stage(rows, managed) == ("running", True)
    assert experiment_stage([{"id": "batch", "status": "completed"}], managed) == (
        "finished",
        True,
    )


def test_cancelled_managed_batch_keeps_its_experiment_running_and_resumable(managed):
    context = managed
    development = context.development
    experiments = ModelExperimentService(
        development.store, development.filesystem, training=context.service
    )
    spec = context.spec
    record = experiments.create(
        CreateModelExperiment(
            name="Managed study",
            operationId="create",
            inputs=spec.inputs,
            predictorPolicy={"method": "skip", "refitPercentile": None},
        )
    )
    plan = {**spec.model_dump(), "batchName": "Managed plan"}
    record = experiments.update(
        record["id"],
        UpdateModelExperiment(
            name="Managed study", expectedRevision=1, batchPlans=[{"id": "tiny", "spec": plan}]
        ),
    )
    submitted = experiments.submit(
        record["id"],
        SubmitModelExperiment(expectedRevision=record["revision"], operationId="submit"),
    )
    assert submitted["submission"]["status"] == "submitted"
    receipt = development.store.get_draft(record["id"])["payload"]["submission"]
    assert receipt["executionMode"] == "task-center"
    [batch_id] = submitted["submission"]["batchIds"]
    [owner] = [row for row in context.store.owners() if row["kind"] == "experiment"]
    assert owner["kind"] == "experiment" and owner["id"] == record["id"]
    assert owner["title"] == "Managed study"

    context.identity = batch_id
    context.folder = development.store.folder / "training" / batch_id
    context.service.cancel(batch_id, "cancel-before-start")
    # No fold can run any more, so the batch reads cancelled while its final collection waits.
    assert context.service.execution(batch_id)["status"] == "cancelled"
    assert any(
        task["kind"] == "mil-collect" and task["state"] in LIVE for task in group_tasks(context)
    )
    runner = make_runner(context)
    drive(context, runner, lambda: settled(context))
    assert context.service.execution(batch_id)["status"] == "cancelled"
    current = experiments.get(record["id"])
    assert current["stage"] == "running"
    assert current["batches"][0]["status"] == "cancelled"
    experiments.require_training_action(record["id"], batch_id, resume=True)
    resumed = context.service.launch(batch_id, "resume", resume=True)
    assert resumed["status"] == "queued"
    assert experiments.get(record["id"])["stage"] == "running"
