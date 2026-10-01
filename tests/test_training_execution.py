"""Durable execution preserves scientific identity across launch, resume and cancellation."""

import copy
import sys
from pathlib import Path

import h5py
import pytest
from support.task_center import task_ids
from support.training import (
    attempts,
    batch_tasks,
    lose_batch,
    rewrite_batch,
    runtime,
    synthetic_results,
)
from support.training import tc_execution as tc_execution
from test_worker_process_ownership import isolated_worker_tree as _worker_tree

from histopilot.application.training import membership_plan_id
from histopilot.storage.io import read_json_bounded, write_json_atomic
from histopilot.storage.project_lock import StorageError
from histopilot.workers.train_batch import collect_results, execute_plan
from histopilot.workers.training_process import process_identity, save_state

isolated_worker_tree = _worker_tree


def test_launch_freezes_exact_work_and_idempotent_receipt(tc_execution, task_center):
    service, frozen, _source = tc_execution
    scientific_before = service.store.get_configuration(frozen["id"])
    state = service.launch(frozen["id"], "launch-once")
    assert state["status"] == "queued"
    assert state["runCounts"] == {
        "total": 5,
        "queued": 5,
        "running": 0,
        "completed": 0,
        "failed": 0,
        "cancelled": 0,
        "interrupted": 0,
    }
    # One task per run plus the final collection, each queued once.
    queued = attempts(task_center, frozen["id"])
    assert sorted(task["kind"] for task in batch_tasks(task_center, frozen["id"])) == [
        "mil-collect",
        *["mil-fold"] * 5,
    ]
    assert {attempt for _task, attempt in queued} == {1}
    assert service.launch(frozen["id"], "launch-once")["runs"] == state["runs"]
    assert attempts(task_center, frozen["id"]) == queued
    plan = read_json_bounded(Path(state["outputPath"]) / "plan.json")
    folds = {
        task["adapterData"]["runId"]: task["command"]
        for task in batch_tasks(task_center, frozen["id"], kind="mil-fold")
    }
    assert set(folds) == {run["id"] for run in plan["runs"]}
    for run in plan["runs"]:
        # Every fold runs the batch's archived package with bytecode disabled.
        command = folds[run["id"]]
        assert command["cwd"] == state["computePath"]
        assert command["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
        assert command["argv"][2:] == [
            "-m",
            "histopilot.workers.managed_fold",
            str(Path(state["outputPath"]) / "plan.json"),
            run["id"],
        ]
    protocol = service.store.get_configuration(frozen["manifest"]["spec"]["inputs"]["protocolId"])[
        "manifest"
    ]
    for run in plan["runs"]:
        selected = execute_plan(plan, run, None)
        expected = [
            row for row in protocol["memberships"] if membership_plan_id(row) == run["splitPlanId"]
        ]
        assert selected["data"]["memberships"] == expected
        assert selected["trainingSeed"] == 11
        assert selected["splitPlan"]["seed"] == 42
        assert selected["recipe"] == frozen["manifest"]["configurations"][0]["recipe"]
        assert selected["target"] == protocol["spec"]["target"]
        assert selected["device"] == "cpu"
    assert (
        plan["data"]["loadingPolicy"]
        == frozen["manifest"]["resolvedInputs"]["resolvedLoadingPolicy"]
    )
    assert plan["data"]["packPath"]
    assert plan["data"]["packStamps"]
    assert service.store.get_configuration(frozen["id"]) == scientific_before


@pytest.mark.parametrize("visibility", ["active", "archived", "trashed"])
def test_promoted_checkpoint_batch_cannot_resume_even_if_predictor_hidden(
    tc_execution, task_center, visibility
):
    import hashlib

    service, batch, _ = tc_execution
    state = service.launch(batch["id"], "original-launch")
    queued = attempts(task_center, batch["id"])
    lose_batch(task_center, batch["id"])
    predictor = service.store.publish_configuration(
        manifest={
            "kind": "frozen-predictor",
            "datasetId": batch["manifest"]["datasetId"],
            "batchId": batch["id"],
        },
        operation_id="freeze-test-predictor",
    )
    if visibility != "active":
        lifecycle = service.store.lifecycle
        lifecycle.apply(
            {f"configuration:{predictor['id']}": visibility},
            operation_id="hide-predictor",
            request_hash=hashlib.sha256(visibility.encode()).hexdigest(),
            expected_revision=lifecycle.read()["revision"],
        )
    plan = (Path(state["outputPath"]) / "plan.json").read_bytes()
    with pytest.raises(StorageError) as error:
        service.launch(batch["id"], "unsafe-resume", resume=True)
    assert error.value.code == "BATCH_HAS_FROZEN_PREDICTOR"
    assert attempts(task_center, batch["id"]) == queued
    assert (Path(state["outputPath"]) / "plan.json").read_bytes() == plan
    # A replay of the acknowledged old launch must not queue its folds again.
    assert service.launch(batch["id"], "original-launch")["batchId"] == batch["id"]
    assert attempts(task_center, batch["id"]) == queued
    assert {task["state"] for task in batch_tasks(task_center, batch["id"])} == {"interrupted"}


def test_archived_experiment_blocks_new_training_but_preserves_launch_replay(
    tc_execution, task_center
):
    import hashlib

    from histopilot.application.model_experiments import ModelExperimentService
    from histopilot.schemas.model_experiments import SubmitModelExperiment

    service, original, _ = tc_execution
    owner = service.store.create_draft(
        "experiment",
        "Owner",
        {"type": "model-experiment", "inputs": original["manifest"]["spec"]["inputs"]},
    )
    batch = rewrite_batch(
        service,
        original,
        lambda manifest: manifest["spec"].update(experimentId=owner["id"], experimentRevision=1),
    )
    with pytest.raises(StorageError) as unsubmitted:
        service.launch(batch["id"], "owned-launch")
    assert unsubmitted.value.code == "EXPERIMENT_SUBMISSION_REQUIRED"
    assert "Start the experiment before training" in str(unsubmitted.value)
    submitted = ModelExperimentService(service.store, service.filesystem, training=service).submit(
        owner["id"], SubmitModelExperiment(expectedRevision=1, operationId="submit-owner")
    )
    assert submitted["submission"]["status"] == "submitted", submitted["submission"]
    state = service.execution(batch["id"])
    operation = next(iter(read_json_bounded(Path(state["outputPath"]) / "operations.json")))
    queued = attempts(task_center, batch["id"])
    assert len(queued) == 6
    assert {task["group"]["id"] for task in task_center.tasks(kind="mil-fold")} == {batch["id"]}
    lose_batch(task_center, batch["id"])
    lifecycle = service.store.lifecycle
    lifecycle.apply(
        {f"draft:{owner['id']}": "archived"},
        operation_id="archive-owner",
        request_hash=hashlib.sha256(b"archive-owner").hexdigest(),
        expected_revision=lifecycle.read()["revision"],
    )
    with pytest.raises(StorageError) as error:
        service.launch(batch["id"], "resume-archived", resume=True)
    assert error.value.code == "EXPERIMENT_ARCHIVED"
    assert service.launch(batch["id"], operation)["batchId"] == state["batchId"]
    assert attempts(task_center, batch["id"]) == queued


def test_other_operations_do_not_launch_a_second_scheduler(tc_execution, task_center):
    service, frozen, _source = tc_execution
    service.launch(frozen["id"], "original")
    queued = attempts(task_center, frozen["id"])
    for resume in (False, True):
        with pytest.raises(StorageError) as error:
            service.launch(frozen["id"], "second", resume=resume)
        assert error.value.code == "TRAINING_ACTIVE"
    with pytest.raises(StorageError) as error:
        service.cancel(frozen["id"], "original")
    assert error.value.code == "OPERATION_CONFLICT"
    assert attempts(task_center, frozen["id"]) == queued


def test_interrupted_resume_preserves_completed_runs_and_increments_unfinished_attempts(
    tc_execution, task_center
):
    service, frozen, _source = tc_execution
    state = service.launch(frozen["id"], "first")
    folder = Path(state["outputPath"])
    # The fold adapter recorded the first run's receipt; the second was training when lost.
    state["runs"][0].update(
        status="completed", result={"runId": state["runs"][0]["id"], "state": "succeeded"}
    )
    completed = copy.deepcopy(state["runs"][0])
    state["runs"][1]["status"] = "running"
    state["status"] = "running"
    save_state(folder, state)
    lose_batch(task_center, frozen["id"], completed={completed["id"]})
    assert service.execution(frozen["id"])["status"] == "interrupted"
    resumed = service.launch(frozen["id"], "resume-once", resume=True)
    assert resumed["runs"][0] == completed
    assert all(run["status"] == "queued" and run["attempt"] == 2 for run in resumed["runs"][1:])
    # Only unfinished folds and the final collection are queued again.
    requeued = attempts(task_center, frozen["id"])
    done = {
        task["id"]
        for task in batch_tasks(task_center, frozen["id"], kind="mil-fold")
        if task["adapterData"]["runId"] == completed["id"]
    }
    assert {attempt for task, attempt in requeued if task in done} == {1}
    assert {attempt for task, attempt in requeued if task not in done} == {2}
    assert service.launch(frozen["id"], "resume-once", resume=True)["runs"] == resumed["runs"]
    assert attempts(task_center, frozen["id"]) == requeued


def test_live_orphan_child_blocks_duplicate_resume(tc_execution, task_center):
    service, frozen, _source = tc_execution
    state = service.launch(frozen["id"], "first")
    state["status"] = "running"
    state["runs"][0].update(status="running", process=process_identity())
    save_state(Path(state["outputPath"]), state)
    queued = attempts(task_center, frozen["id"])
    # The fold's task ended although its training process lives on.
    lose_batch(task_center, frozen["id"])
    current = service.execution(frozen["id"])
    assert any(item["code"] == "ORPHAN_TRAINING_RUN" for item in current["findings"])
    with pytest.raises(StorageError) as error:
        service.launch(frozen["id"], "unsafe-resume", resume=True)
    assert error.value.code == "TRAINING_ACTIVE"
    assert attempts(task_center, frozen["id"]) == queued


def test_cancel_is_durable_and_idempotent(tc_execution, task_center):
    service, frozen, _source = tc_execution
    state = service.launch(frozen["id"], "first")
    first = service.cancel(frozen["id"], "cancel-once")
    assert first["cancelRequested"]
    # Task Center: folds still queued are cancelled at once, so the batch reads cancelled.
    assert first["status"] == "cancelled"
    assert {task["state"] for task in batch_tasks(task_center, frozen["id"], kind="mil-fold")} == {
        "cancelled"
    }
    request_path = Path(state["outputPath"]) / "cancel.json"
    original = request_path.read_bytes()
    queued = attempts(task_center, frozen["id"])
    assert service.cancel(frozen["id"], "cancel-once")["cancelRequested"]
    assert request_path.read_bytes() == original
    assert attempts(task_center, frozen["id"]) == queued
    assert {attempt for _task, attempt in queued} == {1}


def legacy_batch(service, frozen, status):
    """Model a batch launched before the Task Center: its state names no executor."""
    state = service.launch(frozen["id"], "first")
    folder = Path(state["outputPath"])
    saved = read_json_bounded(folder / "state.json")
    for key in ("executor", "taskGroup"):
        saved.pop(key)
    saved.update(status=status, sessionName="hp-train-0123456789abcdef")
    for run in saved["runs"]:
        run["status"] = "completed" if status == "completed" else "running"
    save_state(folder, saved)
    plan = read_json_bounded(folder / "plan.json")
    plan.pop("executionMode")
    write_json_atomic(folder / "plan.json", plan)
    return folder, read_json_bounded(folder / "state.json")


@pytest.mark.parametrize("status", ["running", "completed"])
def test_a_batch_from_before_the_task_center_is_read_only(tc_execution, status):
    service, frozen, _source = tc_execution
    folder, saved = legacy_batch(service, frozen, status)
    execution = service.execution(frozen["id"])
    if status == "completed":
        # A finished batch shows exactly its saved record.
        assert execution["status"] == "completed" and execution["findings"] == []
        assert execution["runs"] == saved["runs"]
    else:
        # Nothing runs it any more: an unfinished one has stopped for good.
        assert execution["status"] == "interrupted"
        assert {run["status"] for run in execution["runs"]} == {"interrupted"}
        assert execution["runCounts"]["interrupted"] == len(execution["runs"])
        assert [item["code"] for item in execution["findings"]] == ["CREATED_BEFORE_TASK_CENTER"]
    assert "taskCenter" not in execution
    actions = (
        lambda: service.launch(frozen["id"], "again"),
        lambda: service.launch(frozen["id"], "resume", resume=True),
        lambda: service.cancel(frozen["id"], "cancel"),
        lambda: service.cancel_runs(frozen["id"], [saved["runs"][0]["id"]], "cancel-run"),
    )
    for action in actions:
        with pytest.raises(StorageError) as refused:
            action()
        assert (refused.value.code, refused.value.status_code) == (
            "CREATED_BEFORE_TASK_CENTER",
            409,
        )
        assert "Created before the Task Center" in str(refused.value)
    assert not (folder / "cancel.json").exists()
    assert read_json_bounded(folder / "state.json") == saved


def test_code_pinned_before_the_task_center_cannot_run_as_fold_tasks(
    tc_execution, task_center, monkeypatch, tmp_path
):
    service, frozen, _source = tc_execution
    # An archive of pre-Task Center code has no managed_fold entry point.
    archive = tmp_path / "pinned-before-the-task-center"
    (archive / "histopilot" / "workers").mkdir(parents=True)
    monkeypatch.setattr(
        "histopilot.application.training.prepare_compute_archive", lambda *_a, **_k: archive
    )
    with pytest.raises(StorageError) as refused:
        service.launch(frozen["id"], "launch")
    assert refused.value.code == "CREATED_BEFORE_TASK_CENTER"
    assert service.execution(frozen["id"]) is None
    assert batch_tasks(task_center, frozen["id"]) == []


def test_orphan_cancellation_signals_only_verified_child_identity(
    tc_execution, task_center, monkeypatch
):
    service, frozen, _source = tc_execution
    state = service.launch(frozen["id"], "first")
    child = {"pid": 2147483000, "startTicks": 123, "bootId": "fixture-boot"}
    state["status"] = "running"
    state["runs"][0].update(status="running", process=child)
    state["runs"][1].update(status="running", process={**child, "pid": 2147483001})
    save_state(Path(state["outputPath"]), state)
    # No task tracks the workers any more, so the batch itself must stop them.
    lose_batch(task_center, frozen["id"])
    monkeypatch.setattr(
        "histopilot.application.training.process_alive", lambda value: value == child
    )
    signals = []
    monkeypatch.setattr(
        "histopilot.application.training.os.killpg",
        lambda pid, signum: signals.append((pid, signum)),
    )
    service.cancel(frozen["id"], "cancel-orphan")
    assert len(signals) == 1
    assert signals[0][0] == child["pid"]
    service.cancel(frozen["id"], "cancel-orphan")
    assert len(signals) == 1


@pytest.mark.parametrize(
    "saved_status,task_state",
    [("running", "interrupted"), ("failed", "failed"), ("completed", "succeeded")],
)
def test_orphan_loader_blocks_resume_and_accepts_cancellation(
    tc_execution,
    task_center,
    isolated_worker_tree,
    saved_status,
    task_state,
):
    from histopilot.workers.training_process import confirmed_process_alive

    service, frozen, _ = tc_execution
    state = service.launch(frozen["id"], "first")
    leader, identity, child = isolated_worker_tree()
    leader.kill()
    leader.wait(timeout=5)
    state.update(status=saved_status)
    state["runs"][0].update(status=saved_status, process=identity)
    save_state(Path(state["outputPath"]), state)
    queued = attempts(task_center, frozen["id"])
    # Every task ended, but a data loader of the first fold survives its worker.
    lose_batch(task_center, frozen["id"], state=task_state)
    assert service.execution(frozen["id"])["status"] == "running"
    with pytest.raises(StorageError) as error:
        service.launch(frozen["id"], "resume-orphan", resume=True)
    assert error.value.code == "TRAINING_ACTIVE"
    cancelled = service.cancel(frozen["id"], "cancel-orphan")
    assert cancelled["cancelRequested"]
    assert not confirmed_process_alive(child)
    assert attempts(task_center, frozen["id"]) == queued


def test_orphan_cleanup_failure_does_not_skip_later_workers(tc_execution, task_center, monkeypatch):
    import signal

    service, frozen, _ = tc_execution
    state = service.launch(frozen["id"], "first")
    children = [
        {"pid": 2147483000 + index, "startTicks": 123, "bootId": "fixture-boot"}
        for index in range(2)
    ]
    state.update(status="failed")
    for run, process in zip(state["runs"], children):
        run.update(status="failed", process=process)
    save_state(Path(state["outputPath"]), state)
    lose_batch(task_center, frozen["id"], state="failed")
    monkeypatch.setattr(
        "histopilot.application.training.process_alive", lambda value: value in children
    )
    events = []
    monkeypatch.setattr(
        "histopilot.application.training.os.killpg",
        lambda pid, signum: events.append(("signal", pid, signum)),
    )

    def stop(process):
        events.append(("drain", process["pid"]))
        if process == children[0]:
            raise StorageError("Injected unkillable process", "TRAINING_CLEANUP_FAILED")

    monkeypatch.setattr("histopilot.application.training.stop_owned_processes", stop)
    with pytest.raises(StorageError, match="every orphan worker"):
        service.cancel(frozen["id"], "cancel-orphans")
    assert events == [
        ("signal", children[0]["pid"], signal.SIGTERM),
        ("signal", children[1]["pid"], signal.SIGTERM),
        ("drain", children[0]["pid"]),
        ("drain", children[1]["pid"]),
    ]
    assert (Path(state["outputPath"]) / "cancel.json").exists()


@pytest.mark.parametrize(
    "condition,expected",
    [
        ("runtime", "TRAINING_RUNTIME_UNAVAILABLE"),
        ("model", "TRAINING_MODEL_UNSUPPORTED"),
        ("split", "TRAINING_SPLIT_UNSUPPORTED"),
        ("task", "TRAINING_TASK_UNSUPPORTED"),
        ("loading", "TRAINING_INPUTS_STALE"),
        ("plans", "TRAINING_SPLITS_CHANGED"),
    ],
)
def test_preflight_rejects_unsupported_or_changed_intent(
    tc_execution, task_center, condition, expected
):
    service, frozen, _source = tc_execution
    before = task_ids(task_center)
    if condition == "runtime":
        service.runtime = lambda: {
            **runtime(),
            "available": False,
            "findings": [{"message": "Torch missing"}],
        }
    elif condition == "model":
        frozen = rewrite_batch(
            service,
            frozen,
            lambda manifest: manifest["configurations"][0]["recipe"].update(model="unsupported"),
        )
    elif condition in {"split", "task"}:
        protocol = service.store.get_configuration(
            frozen["manifest"]["spec"]["inputs"]["protocolId"]
        )["manifest"]
        if condition == "split":
            protocol["spec"]["split"]["mode"] = "monte_carlo"
        else:
            protocol["spec"]["target"]["task"] = "survival"
        changed = service.store.publish_configuration(
            manifest=protocol, operation_id="changed-protocol"
        )
        frozen = rewrite_batch(
            service,
            frozen,
            lambda manifest: manifest["spec"]["inputs"].update(protocolId=changed["id"]),
        )
    elif condition == "loading":
        frozen = rewrite_batch(
            service,
            frozen,
            lambda manifest: manifest["resolvedInputs"].update(resolvedLoadingPolicy="native"),
        )
    else:
        frozen = rewrite_batch(
            service, frozen, lambda manifest: manifest["splitPlans"][0].update(slideCount=999)
        )
    with pytest.raises(StorageError) as error:
        service.launch(frozen["id"], "blocked")
    assert error.value.code == expected
    assert task_ids(task_center) == before


def test_changed_source_blocks_launch_and_existing_operation_still_replays(
    tc_execution, task_center
):
    service, frozen, source = tc_execution
    first = service.launch(frozen["id"], "first")
    queued = attempts(task_center, frozen["id"])
    with h5py.File(source / "s00.h5", "r+") as handle:
        handle["features"][0, 0] = 777
    assert service.launch(frozen["id"], "first")["runs"] == first["runs"]
    lose_batch(task_center, frozen["id"])
    with pytest.raises(StorageError) as error:
        service.launch(frozen["id"], "resume-stale", resume=True)
    assert error.value.code == "TRAINING_INPUTS_STALE"
    assert attempts(task_center, frozen["id"]) == queued


@pytest.mark.parametrize("changed", ["archived_code", "dependency_versions"])
def test_resume_cannot_mix_changed_training_implementations_or_dependencies(
    tc_execution, task_center, monkeypatch, changed
):
    service, frozen, _source = tc_execution
    state = service.launch(frozen["id"], "first")
    queued = attempts(task_center, frozen["id"])
    lose_batch(task_center, frozen["id"])
    path = Path(state["outputPath"]) / "plan.json"
    original = path.read_bytes()
    if changed == "archived_code":
        archived = Path(state["computePath"]) / "histopilot" / "training" / "fold.py"
        archived.write_text(archived.read_text() + "\n# changed code\n")
    else:
        service.runtime = lambda: {**runtime(), "versions": {"torch": "changed-version"}}
    with pytest.raises(StorageError) as error:
        service.launch(frozen["id"], "changed-resume", resume=True)
    assert error.value.code == "TRAINING_RUNTIME_CHANGED"
    assert attempts(task_center, frozen["id"]) == queued
    assert path.read_bytes() == original


def test_failed_launch_is_recorded_without_claiming_live_workers(
    tc_execution, task_center, monkeypatch
):
    service, frozen, _source = tc_execution
    before = task_ids(task_center)
    enqueue = task_center.client.enqueue

    def rejected(*_args, **_kwargs):
        raise OSError("Task Center store rejected the batch")

    monkeypatch.setattr(task_center.client, "enqueue", rejected)
    with pytest.raises(StorageError) as error:
        service.launch(frozen["id"], "launch-fails")
    assert error.value.code == "TRAINING_LAUNCH_FAILED"
    state = service.execution(frozen["id"])
    assert state["status"] == "failed"
    assert task_ids(task_center) == before
    original_plan = (Path(state["outputPath"]) / "plan.json").read_bytes()
    monkeypatch.setattr(task_center.client, "enqueue", enqueue)
    recovered = service.launch(frozen["id"], "launch-fails")
    queued = attempts(task_center, frozen["id"])
    assert recovered["status"] == "queued" and len(queued) == 6
    assert (Path(state["outputPath"]) / "plan.json").read_bytes() == original_plan
    assert (
        read_json_bounded(Path(state["outputPath"]) / "operations.json")["launch-fails"] == "launch"
    )
    assert service.launch(frozen["id"], "launch-fails")["status"] == "queued"
    assert attempts(task_center, frozen["id"]) == queued


@pytest.mark.parametrize("finished", [False, True])
def test_lost_enqueue_acknowledgement_preserves_training_worker_state(
    tc_execution, task_center, monkeypatch, finished
):
    service, frozen, _source = tc_execution
    enqueue = task_center.client.enqueue

    def enqueue_then_fail(*args, **kwargs):
        # The store committed the tasks and the runner started them; only the reply was lost.
        enqueue(*args, **kwargs)
        state_path = service._folder(frozen["id"]) / "state.json"
        state = read_json_bounded(state_path)
        state.update(
            status="failed" if finished else "running",
            findings=[{"severity": "error", "code": "WORKER_EVIDENCE", "message": "Retain me"}],
        )
        write_json_atomic(state_path, state)
        if finished:
            lose_batch(task_center, frozen["id"], state="failed")
        else:
            task_center.start(batch_tasks(task_center, frozen["id"], kind="mil-fold")[0]["id"])
        raise OSError("Lost acknowledgement")

    monkeypatch.setattr(task_center.client, "enqueue", enqueue_then_fail)
    shown = service.launch(frozen["id"], "launch")
    assert shown["status"] == ("failed" if finished else "running")
    assert shown["findings"][0]["code"] == "WORKER_EVIDENCE"
    queued = attempts(task_center, frozen["id"])
    assert len(queued) == 6
    assert service.launch(frozen["id"], "launch")["status"] == shown["status"]
    assert attempts(task_center, frozen["id"]) == queued


@pytest.mark.parametrize("content", ["{", "[]"])
def test_optional_progress_cannot_block_training_cancellation(tc_execution, content):
    service, frozen, _source = tc_execution
    state = service.launch(frozen["id"], "launch")
    run_folder = Path(state["runs"][0]["outputPath"])
    run_folder.mkdir(parents=True, exist_ok=True)
    (run_folder / "progress.json").write_text(content)
    shown = service.execution(frozen["id"])
    assert shown["status"] == "queued"
    assert shown["runs"][0]["progress"] is None
    assert shown["runs"][0]["progressWarning"]
    cancelled = service.cancel(frozen["id"], "cancel")
    # Task Center: the queued folds are cancelled at once.
    assert cancelled["cancelRequested"] and cancelled["status"] == "cancelled"


def test_real_oof_collection_checks_exact_identities_and_patient_metrics(tc_execution):
    pytest.importorskip("torch")
    pytest.importorskip("lightning")
    service, frozen, _source = tc_execution
    batch, state = synthetic_results(service, frozen)
    folder = service._folder(frozen["id"])
    collect_results(batch, state, folder)
    result = read_json_bounded(folder / "results.json")
    assert len(result["oof"]) == 1
    assert result["oof"][0]["slideCount"] == 30
    metrics = result["candidates"][0]["metrics"]
    assert metrics["count"] == 30
    assert metrics["accuracy"] == metrics["auroc"] == 1
    records = read_json_bounded(Path(result["oof"][0]["path"]))
    assert records["purpose"] == "development_assessment"
    assert len({row["slideId"] for row in records["records"]}) == 30


@pytest.mark.parametrize("corruption", ["duplicate", "patient", "label"])
def test_oof_collection_rejects_missing_duplicate_or_mismatched_identity(tc_execution, corruption):
    pytest.importorskip("torch")
    pytest.importorskip("lightning")
    service, frozen, _source = tc_execution
    batch, state = synthetic_results(service, frozen)
    path = Path(state["runs"][0]["result"]["predictions"]["assessment"])
    records = read_json_bounded(path)["records"]
    if corruption == "duplicate":
        records.append(copy.deepcopy(records[0]))
    else:
        records[0]["patientId" if corruption == "patient" else "label"] = "wrong"
    write_json_atomic(path, {"records": records, "classOrder": batch["target"]["classes"]})
    with pytest.raises(ValueError, match="exactly once|differs from frozen"):
        collect_results(batch, state, service._folder(frozen["id"]))


@pytest.mark.parametrize(
    "corruption", ["label_index", "class_order", "probability_sum", "negative_probability"]
)
def test_oof_scoring_rejects_invalid_class_indices_or_probability_contract(
    tc_execution, corruption
):
    pytest.importorskip("torch")
    pytest.importorskip("lightning")
    service, frozen, _source = tc_execution
    batch, state = synthetic_results(service, frozen)
    path = Path(state["runs"][0]["result"]["predictions"]["assessment"])
    document = read_json_bounded(path)
    if corruption == "label_index":
        document["records"][0]["labelIndex"] = 1 - document["records"][0]["labelIndex"]
    elif corruption == "class_order":
        document["classOrder"] = list(reversed(document["classOrder"]))
    elif corruption == "probability_sum":
        document["records"][0]["probabilities"] = [0.9, 0.9]
    else:
        document["records"][0]["probabilities"] = [-0.1, 1.1]
    write_json_atomic(path, document)
    with pytest.raises(ValueError):
        collect_results(batch, state, service._folder(frozen["id"]))


def test_incomplete_oof_results_are_not_reported_as_complete(tc_execution):
    service, frozen, _source = tc_execution
    batch, state = synthetic_results(service, frozen)
    state["runs"][-1]["status"] = "failed"
    collect_results(batch, state, service._folder(frozen["id"]))
    result = read_json_bounded(service._folder(frozen["id"]) / "results.json")
    assert result["oof"] == []
    assert result["candidates"][0]["complete"] is False
    assert result["candidates"][0]["metrics"] is None


def test_fold_tasks_recheck_features_after_launch_before_fitting(
    tc_execution, task_center, monkeypatch
):
    from types import SimpleNamespace

    from histopilot.taskcenter.adapters.mil import MilCollectAdapter, MilFoldAdapter
    from histopilot.workers import managed_collect, managed_fold

    service, frozen, source = tc_execution
    state = service.launch(frozen["id"], "first")
    folder = Path(state["outputPath"])
    with h5py.File(source / "s00.h5", "r+") as handle:
        handle["features"][0, 0] = 999

    def fit(*_args, **_kwargs):
        pytest.fail("Stale inputs must not start a fit.")

    monkeypatch.setitem(sys.modules, "histopilot.training.fold", SimpleNamespace(train_fold=fit))
    exited = {
        "returncode": 1,
        "lost": False,
        "signalled": False,
        "killed": False,
        "stopReason": None,
    }
    context = task_center.context()
    # Each fold task runs its worker, and the runner's fold adapter records the exit.
    for task in batch_tasks(task_center, frozen["id"], kind="mil-fold"):
        task_center.start(task["id"])
        with pytest.raises(ValueError, match="changed"):
            managed_fold.run(folder / "plan.json", task["adapterData"]["runId"])
        decision = MilFoldAdapter(runtime=runtime).on_exit(
            task_center.task(task["id"]), exited, context
        )
        assert decision["state"] == "failed"
        task_center.finish(task["id"], "failed", returncode=1, error=decision["error"])
    [final] = batch_tasks(task_center, frozen["id"], kind="mil-collect")
    task_center.start(final["id"])
    assert managed_collect.collect(folder / "plan.json", final=True) == 0
    decision = MilCollectAdapter().on_exit(
        task_center.task(final["id"]), {**exited, "returncode": 0}, context
    )
    task_center.finish(final["id"], decision["state"])
    state = service.execution(frozen["id"])
    assert state["status"] == "failed"
    assert state["runCounts"]["failed"] == 5
    assert all("Feature source changed" in run["error"] for run in state["runs"])


def test_resume_uses_verified_pinned_worker_after_application_update(
    tc_execution, task_center, monkeypatch
):
    service, frozen, _source = tc_execution
    first = service.launch(frozen["id"], "first")
    plan_path = Path(first["outputPath"]) / "plan.json"
    original = plan_path.read_bytes()
    lose_batch(task_center, frozen["id"])
    monkeypatch.setattr(
        "histopilot.application.training.compute_snapshot",
        lambda: {"sha256": "f" * 64, "files": {}},
    )
    resumed = service.launch(frozen["id"], "after-app-update", resume=True)
    assert plan_path.read_bytes() == original
    assert resumed["computePath"] == first["computePath"]
    assert any(row["code"] == "TRAINING_PINNED_CODE" for row in resumed["findings"])
    assert len((Path(first["outputPath"]) / "attempts.jsonl").read_text().splitlines()) == 2
    # The requeued folds still run the archived package of the first launch.
    folds = batch_tasks(task_center, frozen["id"], kind="mil-fold")
    assert {(task["attempt"], task["command"]["cwd"]) for task in folds} == {
        (2, first["computePath"])
    }


def test_resume_records_host_reboot_without_changing_the_plan(
    tc_execution, task_center, monkeypatch
):
    from histopilot.workers.training_process import host_snapshot

    service, frozen, _source = tc_execution
    first = service.launch(frozen["id"], "first")
    plan_path = Path(first["outputPath"]) / "plan.json"
    original = plan_path.read_bytes()
    lose_batch(task_center, frozen["id"])
    monkeypatch.setattr(
        "histopilot.application.training.host_snapshot",
        lambda: {**host_snapshot(), "bootId": "after-reboot"},
    )
    resumed = service.launch(frozen["id"], "after-reboot", resume=True)
    assert plan_path.read_bytes() == original
    assert any(row["code"] == "TRAINING_HOST_CHANGED" for row in resumed["findings"])
    assert resumed["provenance"]["host"]["bootId"] == "after-reboot"


def test_resume_rejects_changed_execution_plan_before_reusing_archive(tc_execution, task_center):
    service, frozen, _source = tc_execution
    state = service.launch(frozen["id"], "first")
    path = Path(state["outputPath"]) / "plan.json"
    changed = read_json_bounded(path)
    changed["configurations"][0]["recipe"]["learningRate"] = 0.5
    write_json_atomic(path, changed)
    queued = attempts(task_center, frozen["id"])
    lose_batch(task_center, frozen["id"])
    with pytest.raises(StorageError) as error:
        service.launch(frozen["id"], "resume-modified-plan", resume=True)
    assert error.value.code == "TRAINING_PLAN_CHANGED"
    assert attempts(task_center, frozen["id"]) == queued


def test_resume_rejects_relocated_execution_without_mutating_immutable_evidence(
    tc_execution, task_center, monkeypatch
):
    import shutil

    service, frozen, _source = tc_execution
    launched = service.launch(frozen["id"], "first")
    queued = attempts(task_center, frozen["id"])
    lose_batch(task_center, frozen["id"])
    original_folder = Path(launched["outputPath"])
    moved_folder = original_folder.with_name("relocated-execution")
    shutil.copytree(original_folder, moved_folder)
    before = {
        name: (moved_folder / name).read_bytes()
        for name in ("plan.json", "state.json", "attempts.jsonl")
    }
    monkeypatch.setattr(service, "_folder", lambda _identity: moved_folder)
    with pytest.raises(StorageError) as error:
        service.launch(frozen["id"], "resume-relocated", resume=True)
    assert error.value.code == "TRAINING_LOCATION_CHANGED"
    assert {name: (moved_folder / name).read_bytes() for name in before} == before
    assert attempts(task_center, frozen["id"]) == queued


def test_resume_rejects_incompatible_frozen_session_without_rehashing_plan(
    tc_execution, task_center
):
    from histopilot.storage.io import content_hash

    service, frozen, _source = tc_execution
    launched = service.launch(frozen["id"], "first")
    queued = attempts(task_center, frozen["id"])
    lose_batch(task_center, frozen["id"])
    folder = Path(launched["outputPath"])
    legacy_plan = read_json_bounded(folder / "plan.json")
    legacy_plan["sessionName"] = "legacy-session-policy"
    write_json_atomic(folder / "plan.json", legacy_plan)
    launched["planHash"] = content_hash(legacy_plan)
    save_state(folder, launched)
    before = {
        name: (folder / name).read_bytes() for name in ("plan.json", "state.json", "attempts.jsonl")
    }
    with pytest.raises(StorageError) as error:
        service.launch(frozen["id"], "resume-other-session", resume=True)
    assert error.value.code == "TRAINING_LOCATION_CHANGED"
    assert {name: (folder / name).read_bytes() for name in before} == before
    assert attempts(task_center, frozen["id"]) == queued


@pytest.mark.parametrize("drift", ["code", "versions", "pythonVersion"])
def test_partial_submission_pins_code_and_rejects_environment_drift_without_touching_live_batch(
    tc_execution, task_center, monkeypatch, drift
):
    from histopilot.application.model_experiments import ModelExperimentService
    from histopilot.schemas.model_experiments import SubmitModelExperiment

    training, original, _source = tc_execution

    def launched():
        """Batches with queued tasks, in the order they were launched."""
        return list(
            dict.fromkeys(task["group"]["id"] for task in task_center.tasks(kind="mil-fold"))
        )

    owner = training.store.create_draft(
        "experiment",
        "Owned comparison",
        {
            "type": "model-experiment",
            "inputs": original["manifest"]["spec"]["inputs"],
        },
    )
    owned = []
    for index in range(2):
        manifest = copy.deepcopy(original["manifest"])
        manifest["spec"].update(
            experimentId=owner["id"], experimentRevision=1, batchName=f"Batch {index + 1}"
        )
        owned.append(
            training.store.publish_configuration(manifest=manifest, operation_id=f"owned-{index}")
        )
    experiments = ModelExperimentService(training.store, training.filesystem, training=training)
    command = SubmitModelExperiment(expectedRevision=1, operationId="submit-comparison")
    original_launch = training.launch
    launch_count = 0

    def lose_launch(identity, operation):
        nonlocal launch_count
        launch_count += 1
        if launch_count == 2:
            raise StorageError(
                "Synthetic interrupted submission before dispatch", "TEST_INTERRUPTED"
            )
        return original_launch(identity, operation)

    monkeypatch.setattr(training, "launch", lose_launch)
    partial = experiments.submit(owner["id"], command)
    assert partial["submission"]["status"] == "attention"
    assert len(launched()) == 1
    live_batch = training.execution(launched()[0])
    queued = attempts(task_center, live_batch["batchId"])
    assert live_batch["status"] == "queued"
    original_state = copy.deepcopy(live_batch)
    contract = training.store.get_draft(owner["id"])["payload"]["submission"]["executionContract"]
    assert contract["code"]["sha256"] and contract["runtime"]["python"]
    monkeypatch.setattr(training, "launch", original_launch)
    original_runtime = training.runtime
    original_snapshot = __import__(
        "histopilot.application.training", fromlist=["compute_snapshot"]
    ).compute_snapshot
    if drift == "code":
        monkeypatch.setattr(
            "histopilot.application.training.compute_snapshot",
            lambda: {**original_snapshot(), "sha256": "0" * 64},
        )
    elif drift == "versions":
        training.runtime = lambda: {**original_runtime(), "versions": {"torch": "different"}}
    else:
        training.runtime = lambda: {**original_runtime(), "pythonVersion": "different"}
    rejected = experiments.submit(owner["id"], command)
    if drift == "code":
        # The retried batch copies the first batch's archived code instead of the edit.
        assert rejected["submission"]["status"] == "submitted"
        assert len(launched()) == 2
        assert (
            read_json_bounded(training._folder(launched()[1]) / "plan.json")["code"]
            == contract["code"]
        )
        return
    assert rejected["submission"]["error"]["code"] == "EXPERIMENT_RUNTIME_CHANGED"
    assert rejected["configurationLocked"] and rejected["stage"] == "running"
    assert len(launched()) == 1
    assert training.execution(live_batch["batchId"]) == original_state
    assert attempts(task_center, live_batch["batchId"]) == queued
    training.runtime = original_runtime
    monkeypatch.setattr("histopilot.application.training.compute_snapshot", original_snapshot)
    recovered = experiments.submit(owner["id"], command)
    assert recovered["submission"]["status"] == "submitted"
    assert len(launched()) == 2
    assert attempts(task_center, live_batch["batchId"]) == queued
    assert recovered["submission"]["batchIds"] == partial["submission"]["batchIds"]
