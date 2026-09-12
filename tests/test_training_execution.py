"""Durable execution preserves scientific identity across launch, resume and cancellation."""

import copy
import runpy
import sys
from pathlib import Path

import h5py
import pytest

from histopilot.application.training import TrainingService, membership_plan_id
from histopilot.schemas.development import DevelopmentBatchSpec
from histopilot.storage.project_lock import StorageError
from histopilot.workers.packing_process import write_json
from histopilot.workers.train_batch import _run_plan, available_device, collect_results, run_batch
from histopilot.workers.training_process import process_identity, read_json, save_state

support = runpy.run_path(str(Path(__file__).with_name("test_development_batches.py")))


class FakeExecutor:
    def __init__(self):
        self.sessions = set()
        self.launches = []
        self.failure = None

    def available(self):
        return True

    def running(self, session):
        return session in self.sessions

    def launch(self, session, python, plan, log, *, package_root):
        if self.failure:
            raise self.failure
        self.sessions.add(session)
        self.launches.append((session, python, plan, log))


def runtime():
    return {
        "available": True,
        "python": sys.executable,
        "versions": {},
        "cudaAvailable": False,
        "gpuCount": 0,
        "findings": [],
    }


@pytest.fixture
def execution(tmp_path, monkeypatch):
    monkeypatch.setattr("histopilot.application.training.gpu_snapshot", lambda: {"gpus": []})
    monkeypatch.setattr("histopilot.workers.training_process.gpu_snapshot", lambda: {"gpus": []})
    development, spec, source = support["batch"].__wrapped__(tmp_path)
    values = spec.model_dump()
    values.update(mode="single", trainingSeeds=[11])
    values["recipe"].update(maxEpochs=1, bagSize=2, batchSize=2)
    values["resources"].update(gpuIds=[], cpuThreadsPerRun=1, dataLoaderWorkers=0, ramGbPerRun=0.01)
    spec = DevelopmentBatchSpec.model_validate(values)
    preview = development.preview(spec)
    frozen = development.freeze(
        spec, preview["previewHash"], "execution-batch", {"tag": "Executable batch"}
    )
    executor = FakeExecutor()
    service = TrainingService(
        development.store, development.filesystem, executor=executor, runtime=runtime
    )
    return service, frozen, executor, source


def rewrite_batch(service, original, change):
    manifest = copy.deepcopy(original["manifest"])
    change(manifest)
    return service.store.publish_configuration(manifest=manifest, operation_id="modified-batch")


def test_launch_freezes_exact_work_and_idempotent_receipt(execution):
    service, frozen, executor, _source = execution
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
    assert len(executor.launches) == 1
    assert service.launch(frozen["id"], "launch-once")["runs"] == state["runs"]
    assert len(executor.launches) == 1
    plan = read_json(Path(state["outputPath"]) / "plan.json")
    protocol = service.store.get_configuration(frozen["manifest"]["spec"]["inputs"]["protocolId"])[
        "manifest"
    ]
    for run in plan["runs"]:
        selected = _run_plan(plan, run, None)
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
def test_promoted_checkpoint_batch_cannot_resume_even_if_predictor_hidden(execution, visibility):
    import hashlib

    service, batch, executor, _ = execution
    state = service.launch(batch["id"], "original-launch")
    executor.sessions.clear()
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
    assert len(executor.launches) == 1
    assert (Path(state["outputPath"]) / "plan.json").read_bytes() == plan
    # A replay of the acknowledged old launch must not start another process.
    assert service.launch(batch["id"], "original-launch")["batchId"] == batch["id"]
    assert len(executor.launches) == 1


def test_archived_experiment_blocks_new_training_but_preserves_launch_replay(execution):
    import hashlib

    from histopilot.application.model_experiments import ModelExperimentService
    from histopilot.schemas.model_experiments import SubmitModelExperiment

    service, original, executor, _ = execution
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
    submitted = ModelExperimentService(service.store, service.filesystem, training=service).submit(
        owner["id"], SubmitModelExperiment(expectedRevision=1, operationId="submit-owner")
    )
    assert submitted["submission"]["status"] == "submitted", submitted["submission"]
    state = service.execution(batch["id"])
    operation = next(iter(read_json(Path(state["outputPath"]) / "operations.json")))
    executor.sessions.clear()
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
    assert len(executor.launches) == 1


def test_other_operations_do_not_launch_a_second_scheduler(execution):
    service, frozen, executor, _source = execution
    service.launch(frozen["id"], "original")
    for resume in (False, True):
        with pytest.raises(StorageError) as error:
            service.launch(frozen["id"], "second", resume=resume)
        assert error.value.code == "TRAINING_ACTIVE"
    with pytest.raises(StorageError) as error:
        service.cancel(frozen["id"], "original")
    assert error.value.code == "OPERATION_CONFLICT"
    assert len(executor.launches) == 1


def test_interrupted_resume_preserves_completed_runs_and_increments_unfinished_attempts(execution):
    service, frozen, executor, _source = execution
    state = service.launch(frozen["id"], "first")
    folder = Path(state["outputPath"])
    state["runs"][0].update(
        status="completed", result={"runId": state["runs"][0]["id"], "state": "succeeded"}
    )
    completed = copy.deepcopy(state["runs"][0])
    state["runs"][1]["status"] = "running"
    state["status"] = "running"
    save_state(folder, state)
    executor.sessions.clear()
    assert service.execution(frozen["id"])["status"] == "interrupted"
    resumed = service.launch(frozen["id"], "resume-once", resume=True)
    assert resumed["runs"][0] == completed
    assert all(run["status"] == "queued" and run["attempt"] == 2 for run in resumed["runs"][1:])
    assert len(executor.launches) == 2
    assert service.launch(frozen["id"], "resume-once", resume=True)["runs"] == resumed["runs"]
    assert len(executor.launches) == 2


def test_live_orphan_child_blocks_duplicate_resume(execution):
    service, frozen, executor, _source = execution
    state = service.launch(frozen["id"], "first")
    state["status"] = "running"
    state["runs"][0].update(status="running", process=process_identity())
    save_state(Path(state["outputPath"]), state)
    executor.sessions.clear()
    current = service.execution(frozen["id"])
    assert any(item["code"] == "ORPHAN_TRAINING_RUN" for item in current["findings"])
    with pytest.raises(StorageError) as error:
        service.launch(frozen["id"], "unsafe-resume", resume=True)
    assert error.value.code == "TRAINING_ACTIVE"
    assert len(executor.launches) == 1


def test_cancel_is_durable_and_idempotent(execution):
    service, frozen, executor, _source = execution
    state = service.launch(frozen["id"], "first")
    first = service.cancel(frozen["id"], "cancel-once")
    assert first["cancelRequested"]
    request_path = Path(state["outputPath"]) / "cancel.json"
    original = request_path.read_bytes()
    assert service.cancel(frozen["id"], "cancel-once")["cancelRequested"]
    assert request_path.read_bytes() == original
    assert len(executor.launches) == 1


def test_orphan_cancellation_signals_only_verified_child_identity(execution, monkeypatch):
    service, frozen, executor, _source = execution
    state = service.launch(frozen["id"], "first")
    child = {"pid": 2147483000, "startTicks": 123, "bootId": "fixture-boot"}
    state["status"] = "running"
    state["runs"][0].update(status="running", process=child)
    state["runs"][1].update(status="running", process={**child, "pid": 2147483001})
    save_state(Path(state["outputPath"]), state)
    executor.sessions.clear()
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
    "condition,expected",
    [
        ("runtime", "TRAINING_RUNTIME_UNAVAILABLE"),
        ("gpu", "TRAINING_GPU_UNAVAILABLE"),
        ("model", "TRAINING_MODEL_UNSUPPORTED"),
        ("split", "TRAINING_SPLIT_UNSUPPORTED"),
        ("task", "TRAINING_TASK_UNSUPPORTED"),
        ("loading", "TRAINING_INPUTS_STALE"),
        ("plans", "TRAINING_SPLITS_CHANGED"),
    ],
)
def test_preflight_rejects_unsupported_or_changed_intent(execution, condition, expected):
    service, frozen, executor, _source = execution
    if condition == "runtime":
        service.runtime = lambda: {
            **runtime(),
            "available": False,
            "findings": [{"message": "Torch missing"}],
        }
    elif condition == "gpu":
        frozen = rewrite_batch(
            service, frozen, lambda manifest: manifest["spec"]["resources"].update(gpuIds=[0])
        )
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
    assert not executor.launches


def test_changed_source_blocks_launch_and_existing_operation_still_replays(execution):
    service, frozen, executor, source = execution
    first = service.launch(frozen["id"], "first")
    with h5py.File(source / "s00.h5", "r+") as handle:
        handle["features"][0, 0] = 777
    assert service.launch(frozen["id"], "first")["runs"] == first["runs"]
    executor.sessions.clear()
    with pytest.raises(StorageError) as error:
        service.launch(frozen["id"], "resume-stale", resume=True)
    assert error.value.code == "TRAINING_INPUTS_STALE"
    assert len(executor.launches) == 1


@pytest.mark.parametrize("changed", ["archived_code", "dependency_versions"])
def test_resume_cannot_mix_changed_training_implementations_or_dependencies(
    execution, monkeypatch, changed
):
    service, frozen, executor, _source = execution
    state = service.launch(frozen["id"], "first")
    executor.sessions.clear()
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
    assert len(executor.launches) == 1
    assert path.read_bytes() == original


def test_failed_launch_is_recorded_without_claiming_live_workers(execution):
    service, frozen, executor, _source = execution
    executor.failure = RuntimeError("tmux launch rejected")
    with pytest.raises(StorageError) as error:
        service.launch(frozen["id"], "launch-fails")
    assert error.value.code == "TRAINING_LAUNCH_FAILED"
    state = service.execution(frozen["id"])
    assert state["status"] == "failed"
    assert not executor.launches
    original_plan = (Path(state["outputPath"]) / "plan.json").read_bytes()
    executor.failure = None
    recovered = service.launch(frozen["id"], "launch-fails")
    assert recovered["status"] == "queued" and len(executor.launches) == 1
    assert (Path(state["outputPath"]) / "plan.json").read_bytes() == original_plan
    assert read_json(Path(state["outputPath"]) / "operations.json")["launch-fails"] == "launch"
    assert service.launch(frozen["id"], "launch-fails")["status"] == "queued"
    assert len(executor.launches) == 1


@pytest.mark.parametrize("finished", [False, True])
def test_lost_launch_acknowledgement_preserves_training_worker_state(
    execution, monkeypatch, finished
):
    service, frozen, executor, _source = execution
    original = executor.launch

    def launch_then_timeout(*args, **kwargs):
        original(*args, **kwargs)
        state_path = args[2].parent / "state.json"
        state = read_json(state_path)
        state.update(
            status="failed" if finished else "running",
            process={"pid": 2147483000, "startTicks": 1, "bootId": "old-boot"},
            findings=[{"severity": "error", "code": "WORKER_EVIDENCE", "message": "Retain me"}],
        )
        write_json(state_path, state)
        if finished:
            executor.sessions.clear()
        raise TimeoutError("Lost acknowledgement")

    monkeypatch.setattr(executor, "launch", launch_then_timeout)
    shown = service.launch(frozen["id"], "launch")
    assert shown["status"] == ("failed" if finished else "running")
    assert shown["findings"][0]["code"] == "WORKER_EVIDENCE"
    assert service.launch(frozen["id"], "launch")["status"] == shown["status"]
    assert len(executor.launches) == 1


@pytest.mark.parametrize("record_identity", [False, True])
def test_worker_outcome_during_session_probe_is_not_overwritten(
    execution, monkeypatch, record_identity
):
    service, frozen, executor, _source = execution
    pending_probe = False

    def launch_then_timeout(*_args, **_kwargs):
        nonlocal pending_probe
        pending_probe = True
        raise TimeoutError("Lost acknowledgement")

    def inspect(_session):
        nonlocal pending_probe
        if pending_probe:
            pending_probe = False
            path = service._folder(frozen["id"]) / "state.json"
            state = read_json(path)
            state.update(
                status="failed",
                process={"pid": 2147483000, "startTicks": 1, "bootId": "old-boot"}
                if record_identity
                else None,
                findings=[{"severity": "error", "code": "WORKER_EVIDENCE", "message": "Retain me"}],
            )
            write_json(path, state)
        return False

    monkeypatch.setattr(executor, "launch", launch_then_timeout)
    monkeypatch.setattr(executor, "running", inspect)
    shown = service.launch(frozen["id"], "launch")
    assert shown["status"] == "failed"
    assert shown["findings"][0]["code"] == "WORKER_EVIDENCE"
    assert service.launch(frozen["id"], "launch")["findings"] == shown["findings"]
    assert read_json(service._folder(frozen["id"]) / "state.json") == shown


@pytest.mark.parametrize("content", ["{", "[]"])
def test_optional_progress_cannot_block_training_cancellation(execution, content):
    service, frozen, _executor, _source = execution
    state = service.launch(frozen["id"], "launch")
    run_folder = Path(state["runs"][0]["outputPath"])
    run_folder.mkdir(parents=True, exist_ok=True)
    (run_folder / "progress.json").write_text(content)
    shown = service.execution(frozen["id"])
    assert shown["status"] == "queued"
    assert shown["runs"][0]["progress"] is None
    assert shown["runs"][0]["progressWarning"]
    assert service.cancel(frozen["id"], "cancel")["cancelRequested"]


@pytest.mark.parametrize("failure", ["spawn", "identity", "lease"])
def test_scheduler_cleans_up_every_child_if_startup_registration_fails(
    execution, monkeypatch, failure
):
    from contextlib import contextmanager

    from histopilot.workers import train_batch as worker

    service, frozen, _executor, _source = execution
    frozen = rewrite_batch(
        service,
        frozen,
        lambda manifest: manifest["spec"]["resources"].update(maxConcurrentRuns=2),
    )
    state = service.launch(frozen["id"], "launch")
    folder = Path(state["outputPath"])
    registry = folder / "test-leases"
    registry.mkdir()
    children, streams, signals = [], [], []

    @contextmanager
    def leases():
        yield registry, []

    class Child:
        def __init__(self, pid):
            self.pid, self.code = pid, None

        def poll(self):
            return self.code

        def wait(self, timeout):
            self.code = -15
            return self.code

    def spawn(*_args, **kwargs):
        streams.append(kwargs["stdout"])
        if failure == "spawn":
            raise OSError("Injected spawn failure")
        child = Child(2147483000 + len(children))
        children.append(child)
        return child

    def identity(pid=None):
        if failure == "identity" and pid is not None and len(children) == 2:
            raise ProcessLookupError("Injected PID inspection failure")
        return {"pid": pid or 2147482999, "startTicks": 1, "bootId": "test"}

    def write(path, document):
        if failure == "lease" and path.parent == registry and len(children) == 2:
            raise OSError("Injected lease publication failure")
        write_json(path, document)

    def stop(pid, signum):
        signals.append((pid, signum))
        if pid == 2147483000:
            # The first child exits between poll and signal. Its disappearing
            # process group must not prevent cleanup of the second child.
            raise ProcessLookupError("Exited during cleanup")

    monkeypatch.setattr(worker, "_leases", leases)
    monkeypatch.setattr(worker, "_capacity", lambda: (256, 1000))
    monkeypatch.setattr(worker.subprocess, "Popen", spawn)
    monkeypatch.setattr(worker, "process_identity", identity)
    monkeypatch.setattr(worker, "write_json", write)
    monkeypatch.setattr(worker.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(worker.os, "killpg", stop)
    monkeypatch.setattr(worker.ResourceTelemetry, "record", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(worker.time, "sleep", lambda *_args: None)
    run_batch(folder / "plan.json")
    saved = read_json(folder / "state.json")
    assert saved["status"] == "failed"
    assert saved["runCounts"]["failed"] == saved["runCounts"]["total"]
    assert all(stream.closed for stream in streams)
    assert all(child.poll() is not None for child in children)
    assert {pid for pid, _signum in signals} == {child.pid for child in children}
    assert not list(registry.glob("lease-*.json"))


@pytest.mark.parametrize(
    "resources,active,capacity,expected",
    [
        ({"gpuIds": []}, [], (8, 16), (True, None)),
        (
            {"gpuIds": [0, 1]},
            [{"gpu": 0, "cpus": 2, "ramGb": 2, "runsPerGpu": 1}],
            (8, 16),
            (True, 1),
        ),
        (
            {"gpuIds": [0], "runsPerGpu": 2},
            [{"gpu": 0, "cpus": 2, "ramGb": 2, "runsPerGpu": 1}],
            (8, 16),
            (False, None),
        ),
        (
            {"gpuIds": [0], "runsPerGpu": 2},
            [{"gpu": 0, "cpus": 2, "ramGb": 2, "runsPerGpu": 2}],
            (8, 16),
            (True, 0),
        ),
        (
            {"gpuIds": []},
            [{"gpu": None, "cpus": 7, "ramGb": 1, "runsPerGpu": 1}],
            (8, 16),
            (False, None),
        ),
        (
            {"gpuIds": []},
            [{"gpu": None, "cpus": 1, "ramGb": 15, "runsPerGpu": 1}],
            (8, 16),
            (False, None),
        ),
    ],
)
def test_resource_leases_enforce_cross_batch_cpu_ram_and_gpu_limits(
    resources, active, capacity, expected
):
    requested = {
        "cpuThreadsPerRun": 2,
        "dataLoaderWorkers": 0,
        "ramGbPerRun": 2,
        "runsPerGpu": 1,
        **resources,
    }
    assert available_device(requested, active, capacity) == expected


def test_ram_reservations_do_not_double_count_already_resident_training_memory():
    requested = {
        "gpuIds": [],
        "cpuThreadsPerRun": 2,
        "dataLoaderWorkers": 0,
        "ramGbPerRun": 8,
        "runsPerGpu": 1,
    }
    resident = [{"gpu": None, "cpus": 2, "ramGb": 8, "rssGb": 8, "runsPerGpu": 1}]
    assert available_device(requested, resident, (8, 8)) == (True, None)
    # A newly started worker has reserved RAM that has not yet appeared in RSS.
    pending = [{**resident[0], "rssGb": 1}]
    assert available_device(requested, pending, (8, 8)) == (False, None)


def synthetic_results(service, frozen):
    batch, _guard = service._prepare(frozen)
    rows = []
    for run in batch["runs"]:
        folder = service._folder(frozen["id"]) / "runs" / run["id"]
        folder.mkdir(parents=True)
        records = []
        for membership in batch["memberships"][run["splitPlanId"]]:
            if membership["partition"] == "test":
                index = batch["target"]["classes"].index(membership["label"])
                records.append(
                    {
                        "slideId": membership["slideId"],
                        "patientId": membership["patientId"],
                        "label": membership["label"],
                        "labelIndex": index,
                        "probabilities": [0.9, 0.1] if index == 0 else [0.1, 0.9],
                    }
                )
        predictions = folder / "assessment-predictions.json"
        write_json(predictions, {"records": records, "classOrder": batch["target"]["classes"]})
        result = {
            "runId": run["id"],
            "state": "succeeded",
            "predictions": {"assessment": str(predictions)},
        }
        write_json(folder / "result.json", result)
        rows.append({**run, "status": "completed", "result": result, "outputPath": str(folder)})
    return batch, {"status": "completed", "runs": rows}


def test_real_oof_collection_checks_exact_identities_and_patient_metrics(execution):
    pytest.importorskip("torch")
    pytest.importorskip("lightning")
    service, frozen, _executor, _source = execution
    batch, state = synthetic_results(service, frozen)
    folder = service._folder(frozen["id"])
    collect_results(batch, state, folder)
    result = read_json(folder / "results.json")
    assert len(result["oof"]) == 1
    assert result["oof"][0]["slideCount"] == 30
    metrics = result["candidates"][0]["metrics"]
    assert metrics["count"] == 30
    assert metrics["accuracy"] == metrics["auroc"] == 1
    records = read_json(Path(result["oof"][0]["path"]))
    assert records["purpose"] == "development_assessment"
    assert len({row["slideId"] for row in records["records"]}) == 30


@pytest.mark.parametrize("corruption", ["duplicate", "patient", "label"])
def test_oof_collection_rejects_missing_duplicate_or_mismatched_identity(execution, corruption):
    pytest.importorskip("torch")
    pytest.importorskip("lightning")
    service, frozen, _executor, _source = execution
    batch, state = synthetic_results(service, frozen)
    path = Path(state["runs"][0]["result"]["predictions"]["assessment"])
    records = read_json(path)["records"]
    if corruption == "duplicate":
        records.append(copy.deepcopy(records[0]))
    else:
        records[0]["patientId" if corruption == "patient" else "label"] = "wrong"
    write_json(path, {"records": records, "classOrder": batch["target"]["classes"]})
    with pytest.raises(ValueError, match="exactly once|differs from frozen"):
        collect_results(batch, state, service._folder(frozen["id"]))


@pytest.mark.parametrize(
    "corruption", ["label_index", "class_order", "probability_sum", "negative_probability"]
)
def test_oof_scoring_rejects_invalid_class_indices_or_probability_contract(execution, corruption):
    pytest.importorskip("torch")
    pytest.importorskip("lightning")
    service, frozen, _executor, _source = execution
    batch, state = synthetic_results(service, frozen)
    path = Path(state["runs"][0]["result"]["predictions"]["assessment"])
    document = read_json(path)
    if corruption == "label_index":
        document["records"][0]["labelIndex"] = 1 - document["records"][0]["labelIndex"]
    elif corruption == "class_order":
        document["classOrder"] = list(reversed(document["classOrder"]))
    elif corruption == "probability_sum":
        document["records"][0]["probabilities"] = [0.9, 0.9]
    else:
        document["records"][0]["probabilities"] = [-0.1, 1.1]
    write_json(path, document)
    with pytest.raises(ValueError):
        collect_results(batch, state, service._folder(frozen["id"]))


def test_incomplete_oof_results_are_not_reported_as_complete(execution):
    service, frozen, _executor, _source = execution
    batch, state = synthetic_results(service, frozen)
    state["runs"][-1]["status"] = "failed"
    collect_results(batch, state, service._folder(frozen["id"]))
    result = read_json(service._folder(frozen["id"]) / "results.json")
    assert result["oof"] == []
    assert result["candidates"][0]["complete"] is False
    assert result["candidates"][0]["metrics"] is None


def test_scheduler_adopts_exact_completed_child_results_after_lost_parent(execution, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("lightning")
    service, frozen, _executor, _source = execution
    launched = service.launch(frozen["id"], "first")
    folder = Path(launched["outputPath"])
    _batch, completed = synthetic_results(service, frozen)
    assert all(run["status"] == "queued" for run in read_json(folder / "state.json")["runs"])
    monkeypatch.setattr("histopilot.workers.train_batch.signal.signal", lambda *_args: None)
    run_batch(folder / "plan.json")
    state = read_json(folder / "state.json")
    assert state["status"] == "completed"
    assert state["runCounts"]["completed"] == 5
    assert len(read_json(folder / "results.json")["oof"]) == 1


def test_scheduler_cancels_queued_runs_without_starting_children(execution, monkeypatch):
    service, frozen, _executor, _source = execution
    state = service.launch(frozen["id"], "first")
    folder = Path(state["outputPath"])
    service.cancel(frozen["id"], "cancel-before-start")
    monkeypatch.setattr("histopilot.workers.train_batch.signal.signal", lambda *_args: None)
    monkeypatch.setattr(
        "histopilot.workers.train_batch.subprocess.Popen",
        lambda *_args, **_kwargs: pytest.fail("Cancelled batch must not start a child."),
    )
    run_batch(folder / "plan.json")
    state = read_json(folder / "state.json")
    assert state["status"] == "cancelled"
    assert state["runCounts"]["cancelled"] == 5


def test_scheduler_rechecks_features_after_launch_before_starting_any_fold(execution, monkeypatch):
    service, frozen, _executor, source = execution
    state = service.launch(frozen["id"], "first")
    folder = Path(state["outputPath"])
    with h5py.File(source / "s00.h5", "r+") as handle:
        handle["features"][0, 0] = 999
    monkeypatch.setattr("histopilot.workers.train_batch.signal.signal", lambda *_args: None)
    monkeypatch.setattr(
        "histopilot.workers.train_batch.subprocess.Popen",
        lambda *_args, **_kwargs: pytest.fail("Stale inputs must not start a training child."),
    )
    run_batch(folder / "plan.json")
    state = read_json(folder / "state.json")
    assert state["status"] == "failed"
    assert state["runCounts"]["failed"] == 5
    assert any(item["code"] == "TRAINING_WORKER_FAILED" for item in state["findings"])


def test_resume_uses_verified_pinned_worker_after_application_update(execution, monkeypatch):
    service, frozen, executor, _source = execution
    first = service.launch(frozen["id"], "first")
    plan_path = Path(first["outputPath"]) / "plan.json"
    original = plan_path.read_bytes()
    executor.sessions.clear()
    monkeypatch.setattr(
        "histopilot.application.training.compute_snapshot",
        lambda: {"sha256": "f" * 64, "files": {}},
    )
    resumed = service.launch(frozen["id"], "after-app-update", resume=True)
    assert plan_path.read_bytes() == original
    assert resumed["computePath"] == first["computePath"]
    assert any(row["code"] == "TRAINING_PINNED_CODE" for row in resumed["findings"])
    assert len((Path(first["outputPath"]) / "attempts.jsonl").read_text().splitlines()) == 2


def test_resume_records_host_reboot_without_changing_the_plan(execution, monkeypatch):
    from histopilot.workers.training_process import host_snapshot

    service, frozen, executor, _source = execution
    first = service.launch(frozen["id"], "first")
    plan_path = Path(first["outputPath"]) / "plan.json"
    original = plan_path.read_bytes()
    executor.sessions.clear()
    monkeypatch.setattr(
        "histopilot.application.training.host_snapshot",
        lambda: {**host_snapshot(), "bootId": "after-reboot"},
    )
    resumed = service.launch(frozen["id"], "after-reboot", resume=True)
    assert plan_path.read_bytes() == original
    assert any(row["code"] == "TRAINING_HOST_CHANGED" for row in resumed["findings"])
    assert resumed["provenance"]["host"]["bootId"] == "after-reboot"


@pytest.mark.parametrize(
    "message,expected_dispatches,expected_status",
    [
        ("CUDA unknown error", 1, "interrupted"),
        ("CUDA out of memory", 5, "failed"),
    ],
)
def test_scheduler_halts_dispatch_on_device_failure_and_preserves_checkpoint(
    execution, monkeypatch, message, expected_dispatches, expected_status
):
    from contextlib import contextmanager

    from histopilot.workers.training_process import now

    service, frozen, executor, _source = execution
    service.runtime = lambda: {**runtime(), "cudaAvailable": True, "gpuCount": 1}
    frozen = rewrite_batch(
        service,
        frozen,
        lambda manifest: manifest["spec"]["resources"].update(gpuIds=[0], maxConcurrentRuns=1),
    )
    launched = service.launch(frozen["id"], "first")
    folder = Path(launched["outputPath"])
    checkpoint = Path(launched["runs"][0]["outputPath"]) / "last.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"existing checkpoint must survive")
    registry = folder / "fake-leases"
    registry.mkdir()
    calls = []

    @contextmanager
    def leases():
        yield registry, []

    class FailedChild:
        pid = 2147483000

        def poll(self):
            return 1

    def spawn(command, **kwargs):
        calls.append(command)
        plan_path = Path(command[-1])
        write_json(plan_path.parent / "failure.json", {"error": message, "at": now()})
        return FailedChild()

    monkeypatch.setattr("histopilot.workers.train_batch._leases", leases)
    monkeypatch.setattr("histopilot.workers.train_batch.subprocess.Popen", spawn)
    monkeypatch.setattr("histopilot.workers.train_batch.signal.signal", lambda *_args: None)
    monkeypatch.setattr("histopilot.workers.train_batch.time.sleep", lambda *_args: None)
    monkeypatch.setattr(
        "histopilot.workers.train_batch.process_identity",
        lambda pid=None: {"pid": pid or 2147483001, "startTicks": 1, "bootId": "fake"},
    )
    run_batch(folder / "plan.json")
    state = read_json(folder / "state.json")
    assert len(calls) == expected_dispatches
    assert state["status"] == expected_status
    assert checkpoint.read_bytes() == b"existing checkpoint must survive"
    if expected_status == "interrupted":
        assert state["runCounts"]["interrupted"] == 4
        assert state["runCounts"]["failed"] == 1
        assert state["dispatchHalted"]["code"] == "TRAINING_GPU_DISPATCH_HALTED"
    else:
        assert "dispatchHalted" not in state


def test_resume_rejects_changed_execution_plan_before_reusing_archive(execution):
    service, frozen, executor, _source = execution
    state = service.launch(frozen["id"], "first")
    path = Path(state["outputPath"]) / "plan.json"
    changed = read_json(path)
    changed["configurations"][0]["recipe"]["learningRate"] = 0.5
    write_json(path, changed)
    executor.sessions.clear()
    with pytest.raises(StorageError) as error:
        service.launch(frozen["id"], "resume-modified-plan", resume=True)
    assert error.value.code == "TRAINING_PLAN_CHANGED"
    assert len(executor.launches) == 1


def test_resume_rejects_relocated_execution_without_mutating_immutable_evidence(
    execution, monkeypatch
):
    import shutil

    service, frozen, executor, _source = execution
    launched = service.launch(frozen["id"], "first")
    original_folder = Path(launched["outputPath"])
    moved_folder = original_folder.with_name("relocated-execution")
    shutil.copytree(original_folder, moved_folder)
    before = {
        name: (moved_folder / name).read_bytes()
        for name in ("plan.json", "state.json", "attempts.jsonl")
    }
    executor.sessions.clear()
    monkeypatch.setattr(service, "_folder", lambda _identity: moved_folder)
    with pytest.raises(StorageError) as error:
        service.launch(frozen["id"], "resume-relocated", resume=True)
    assert error.value.code == "TRAINING_LOCATION_CHANGED"
    assert {name: (moved_folder / name).read_bytes() for name in before} == before
    assert len(executor.launches) == 1


def test_resume_rejects_incompatible_frozen_session_without_rehashing_plan(execution):
    from histopilot.application.feature_bundles import _hash

    service, frozen, executor, _source = execution
    launched = service.launch(frozen["id"], "first")
    folder = Path(launched["outputPath"])
    legacy_plan = read_json(folder / "plan.json")
    legacy_plan["sessionName"] = "legacy-session-policy"
    write_json(folder / "plan.json", legacy_plan)
    launched["planHash"] = _hash(legacy_plan)
    save_state(folder, launched)
    before = {
        name: (folder / name).read_bytes() for name in ("plan.json", "state.json", "attempts.jsonl")
    }
    executor.sessions.clear()
    with pytest.raises(StorageError) as error:
        service.launch(frozen["id"], "resume-other-session", resume=True)
    assert error.value.code == "TRAINING_LOCATION_CHANGED"
    assert {name: (folder / name).read_bytes() for name in before} == before
    assert len(executor.launches) == 1


@pytest.mark.parametrize("drift", ["code", "versions", "pythonVersion"])
def test_partial_experiment_submission_rejects_environment_drift_without_touching_live_batch(
    execution, monkeypatch, drift
):
    from histopilot.application.model_experiments import ModelExperimentService
    from histopilot.schemas.model_experiments import SubmitModelExperiment

    training, original, executor, _source = execution
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
    assert len(executor.launches) == 1
    live_batch = training.execution(executor.launches[0][2].parent.name)
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
    assert rejected["submission"]["error"]["code"] == "EXPERIMENT_RUNTIME_CHANGED"
    assert rejected["configurationLocked"] and rejected["stage"] == "running"
    assert len(executor.launches) == 1
    assert training.execution(live_batch["batchId"]) == original_state
    training.runtime = original_runtime
    monkeypatch.setattr("histopilot.application.training.compute_snapshot", original_snapshot)
    recovered = experiments.submit(owner["id"], command)
    assert recovered["submission"]["status"] == "submitted"
    assert len(executor.launches) == 2
    assert recovered["submission"]["batchIds"] == partial["submission"]["batchIds"]
