"""GPU observations require compatible frozen execution evidence, without Torch."""

import copy
import json
import os
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from histopilot.application import runtime_evidence as evidence
from histopilot.application.development import _plan_metadata
from histopilot.application.feature_bundles import _hash
from histopilot.application.runtime_workload import workload_for_recipe
from histopilot.schemas.nnmil import resolve_nnmil_plan
from histopilot.storage.project_lock import StorageError
from histopilot.workers.training_process import process_identity


def save(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def observed(tmp_path, monkeypatch):
    recipe = {
        "model": "nnmil",
        "batchSize": 2,
        "maxEpochs": 10,
        "bagSizeMode": "training_median",
        "bagSize": None,
        "evalBatchSize": 1,
    }
    rows = [
        {
            "slideId": name,
            "patientId": name,
            "label": "mut",
            "partition": role,
            "seed": 42,
            "fold": 0,
            "phase": "development",
        }
        for name, role in (("a", "train"), ("b", "val"), ("c", "test"))
    ]
    files = {
        row["slideId"]: {"patchCount": count, "dimensions": 768}
        for row, count in zip(rows, (100, 150, 200), strict=True)
    }
    target = {
        "task": "binary_classification",
        "unit": "patient",
        "classes": ["wt", "mut"],
        "positiveClass": "mut",
    }
    split = {"id": _hash(_plan_metadata(rows[0])), **_plan_metadata(rows[0])}
    run = {
        "id": "run-one",
        "candidateId": "candidate-one",
        "splitPlanId": split["id"],
        "trainingSeed": 42,
        "status": "planned",
    }
    configs = {
        "protocol-one": {
            "contentHash": "protocol-hash",
            "manifest": {"spec": {"target": target}, "memberships": rows},
        },
        "bundle-one": {"contentHash": "bundle-hash", "manifest": {}},
        "batch-one": {
            "contentHash": "batch-hash",
            "manifest": {
                "kind": "mil-batch",
                "spec": {"inputs": {"protocolId": "protocol-one", "featureBundleId": "bundle-one"}},
                "runs": [run],
                "configurations": [{"id": "candidate-one", "recipe": recipe}],
                "splitPlans": [split],
            },
        },
    }
    manifest = configs["batch-one"]["manifest"]
    code = {"sha256": "current-code", "files": {}}
    resources = {
        "maxConcurrentRuns": 1,
        "runsPerGpu": 1,
        "gpuIds": [0],
        "cpuThreadsPerRun": 2,
        "dataLoaderWorkers": 2,
        "ramGbPerRun": 8,
    }
    plan = {
        "batchId": "batch-one",
        "batchContentHash": "batch-hash",
        "protocolId": "protocol-one",
        "protocolContentHash": "protocol-hash",
        "featureBundleId": "bundle-one",
        "featureBundleContentHash": "bundle-hash",
        "target": target,
        "code": code,
        "resources": resources,
        "runtime": {"python": "/training/python", "versions": {"torch": "2.10"}},
        "memberships": {split["id"]: rows},
        "data": {"featureFiles": files, "loadingPolicy": "mmap", "featureDim": 768},
        **{key: manifest[key] for key in ("runs", "configurations", "splitPlans")},
    }
    run_plan = resolve_nnmil_plan(
        {
            "runId": run["id"],
            "batchId": "batch-one",
            "batchContentHash": "batch-hash",
            "candidateId": run["candidateId"],
            "trainingSeed": 42,
            "splitPlan": split,
            "recipe": recipe,
            "target": target,
            "code": code,
            "resources": resources,
            "runtime": plan["runtime"],
            "device": "cuda",
            "data": {**plan["data"], "memberships": rows},
        }
    )
    current = {
        **run,
        "status": "running",
        "gpu": 0,
        "process": process_identity(),
        "processGroupId": os.getpid(),
        "startedAt": (datetime.now(UTC) - timedelta(seconds=1)).isoformat(),
    }
    state = {
        "batchId": "batch-one",
        "planHash": _hash(plan),
        "runs": [current],
        "provenance": {"gpus": [{"index": 0, "uuid": "GPU-0"}]},
    }
    progress = {
        "epoch": 3,
        "cudaPeakAllocatedBytes": 2 * 1024**3,
        "cudaPeakReservedBytes": 3 * 1024**3,
    }
    batch_folder = tmp_path / "training" / "batch-one"
    run_folder = batch_folder / "runs" / "run-one"
    run_folder.mkdir(parents=True)
    save(batch_folder / "plan.json", plan)
    save(batch_folder / "state.json", state)
    save(run_folder / "plan.json", run_plan)
    save(run_folder / "progress.json", progress)
    store = SimpleNamespace(folder=tmp_path, get_configuration=lambda identity: configs[identity])
    workload = workload_for_recipe(recipe, rows, files, "protocol-hash", "bundle-hash", "mmap", 2)
    workload.update(runtimeVersions=plan["runtime"]["versions"], runtimePython="/training/python")
    monkeypatch.setattr(evidence, "compute_snapshot", lambda: code)
    return SimpleNamespace(
        store=store,
        workload=workload,
        plan=plan,
        state=state,
        run_plan=run_plan,
        progress=progress,
        configs=configs,
        batch_folder=batch_folder,
        run_folder=run_folder,
    )


def collect(fixture):
    return evidence.observed_runtime_runs(fixture.store, [fixture.workload])


def test_live_complete_epoch_is_usable_and_read_only(observed):
    before = {str(p): p.read_bytes() for p in observed.store.folder.rglob("*.json")}
    assert collect(observed) == [
        {
            "key": observed.workload["key"],
            "gpuUuid": "GPU-0",
            "trainingPatches": 50,
            "evaluationPatches": 150,
            "peakReservedGpuGb": 3,
            "stage": "fit",
            "epoch": 3,
            "batchId": "batch-one",
            "runId": "run-one",
        }
    ]
    assert before == {str(p): p.read_bytes() for p in observed.store.folder.rglob("*.json")}


def test_successful_assessment_uses_final_peak(observed):
    result = {
        "runId": "run-one",
        "state": "succeeded",
        "epochsCompleted": 4,
        "cudaPeakAllocatedBytes": 3 * 1024**3,
        "cudaPeakReservedBytes": 4 * 1024**3,
        **{key: observed.run_plan[key] for key in ("effectiveRecipe", "nnmilPlanning")},
    }
    observed.state["runs"][0].update(status="completed", result=result)
    save(observed.batch_folder / "state.json", observed.state)
    save(observed.run_folder / "result.json", result)
    row = collect(observed)[0]
    assert (row["stage"], row["epoch"], row["peakReservedGpuGb"], row["evaluationPatches"]) == (
        "assessment",
        4,
        4,
        200,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("epoch", 0),
        ("epoch", True),
        ("epoch", 11),
        ("cudaPeakReservedBytes", float("nan")),
        ("cudaPeakReservedBytes", float("inf")),
        ("cudaPeakReservedBytes", 0),
        ("cudaPeakReservedBytes", True),
        ("cudaPeakAllocatedBytes", 4 * 1024**3),
    ],
)
def test_invalid_incomplete_or_impossible_counters_are_not_evidence(observed, field, value):
    save(observed.run_folder / "progress.json", {**observed.progress, field: value})
    assert collect(observed) == []


@pytest.mark.parametrize("status", ["failed", "cancelled", "interrupted", "queued"])
def test_unsuccessful_runs_do_not_lend_partial_capacity(observed, status):
    observed.state["runs"][0].update(status=status, failureCategory="out_of_memory")
    save(observed.batch_folder / "state.json", observed.state)
    assert collect(observed) == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("runId", "other-run"),
        ("candidateId", "other-candidate"),
        ("trainingSeed", 999),
        ("batchContentHash", "foreign-hash"),
        ("device", "cpu"),
        ("code", {"sha256": "old-code"}),
        ("runtime", {"python": "/wrong/python"}),
    ],
)
def test_mismatched_frozen_run_provenance_is_rejected(observed, field, value):
    save(observed.run_folder / "plan.json", {**observed.run_plan, field: value})
    assert collect(observed) == []


def test_fitting_membership_and_resolved_recipe_cannot_be_changed(observed):
    changed = copy.deepcopy(observed.run_plan)
    changed["data"]["memberships"][0]["patientId"] = "foreign-patient"
    save(observed.run_folder / "plan.json", changed)
    assert collect(observed) == []
    changed = copy.deepcopy(observed.run_plan)
    changed["effectiveRecipe"]["bagSize"] = 1
    save(observed.run_folder / "plan.json", changed)
    assert collect(observed) == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("protocolHash", "other"),
        ("featureBundleHash", "other"),
        ("key", "other"),
        ("runtimePython", "/different/python"),
        ("runtimeVersions", {"torch": "1.0"}),
    ],
)
def test_other_workloads_or_runtimes_are_not_reused(observed, field, value):
    observed.workload[field] = value
    assert collect(observed) == []


def test_changed_current_code_or_unknown_gpu_is_not_reused(observed, monkeypatch):
    monkeypatch.setattr(evidence, "compute_snapshot", lambda: {"sha256": "new-code"})
    assert collect(observed) == []
    monkeypatch.setattr(evidence, "compute_snapshot", lambda: observed.plan["code"])
    observed.state["provenance"]["gpus"] = []
    save(observed.batch_folder / "state.json", observed.state)
    assert collect(observed) == []


@pytest.mark.parametrize("gpu", [True, 1, -1])
def test_unassigned_gpu_does_not_establish_capacity(observed, gpu):
    observed.state["runs"][0]["gpu"] = gpu
    save(observed.batch_folder / "state.json", observed.state)
    assert collect(observed) == []


def test_reused_pid_does_not_validate_stale_progress(observed):
    observed.state["runs"][0]["process"]["startTicks"] += 1
    save(observed.batch_folder / "state.json", observed.state)
    assert collect(observed) == []


@pytest.mark.parametrize("kind", ["partial", "symlink", "hardlink", "fifo", "oversized"])
def test_unsafe_or_partial_progress_is_unavailable_without_blocking(observed, kind, tmp_path):
    path = observed.run_folder / "progress.json"
    path.unlink()
    if kind == "partial":
        path.write_text('{"epoch":3')
    elif kind in {"symlink", "hardlink"}:
        outside = tmp_path / "outside.json"
        save(outside, observed.progress)
        path.symlink_to(outside) if kind == "symlink" else os.link(outside, path)
    elif kind == "fifo":
        os.mkfifo(path)
    else:
        path.write_bytes(b" " * (1024**2 + 1))
    assert collect(observed) == []


def test_missing_training_directory_is_not_created(tmp_path):
    store = SimpleNamespace(folder=tmp_path)
    assert evidence.observed_runtime_runs(store, [{"key": "test"}]) == []
    assert list(tmp_path.iterdir()) == []


@pytest.fixture
def leases(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence.tempfile, "gettempdir", lambda: str(tmp_path))
    folder = tmp_path / f"histopilot-training-{os.getuid()}"
    folder.mkdir(mode=0o700)
    identity = process_identity()
    lease = {
        "process": identity,
        "processGroupId": identity["pid"],
        "gpu": 0,
        "cpus": 6,
        "ramGb": 8,
        "runsPerGpu": 1,
        "batchId": "batch-one",
        "runId": "run-one",
    }
    path = folder / f"lease-{identity['pid']}.json"
    save(path, lease)
    return folder, path, lease


def test_live_leases_preserve_scheduler_budgets_and_are_not_mutated(leases):
    _, path, lease = leases
    before = path.read_bytes()
    rows = evidence.read_active_leases()
    assert len(rows) == 1
    assert {key: rows[0][key] for key in ("cpus", "ramGb", "runsPerGpu", "gpu")} == {
        "cpus": 6,
        "ramGb": 8,
        "runsPerGpu": 1,
        "gpu": 0,
    }
    assert rows[0]["process"] == lease["process"]
    assert rows[0]["rssGb"] > 0
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("cpus", True),
        ("cpus", 0),
        ("ramGb", -1),
        ("ramGb", float("inf")),
        ("runsPerGpu", 0),
        ("runsPerGpu", 17),
        ("gpu", True),
        ("gpu", -1),
        ("process", {"pid": os.getpid(), "startTicks": "invalid", "bootId": "boot"}),
    ],
)
def test_malformed_lease_means_unknown_not_idle(leases, field, value):
    _, path, lease = leases
    save(path, {**lease, field: value})
    assert evidence.read_active_leases() is None


def test_stale_leases_are_ignored_without_cleanup(leases):
    _, path, lease = leases
    lease["process"]["bootId"] = "previous-boot"
    save(path, lease)
    assert evidence.read_active_leases() == []
    assert path.exists()


def test_unknown_process_ownership_means_unknown(leases, monkeypatch):
    def unknown(*args):
        raise StorageError("Cannot verify process", "TRAINING_PROCESS_UNKNOWN")

    monkeypatch.setattr(evidence, "owned_processes", unknown)
    assert evidence.read_active_leases() is None


def test_missing_lease_directory_is_not_created(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence.tempfile, "gettempdir", lambda: str(tmp_path))
    assert evidence.read_active_leases() == []
    assert list(tmp_path.iterdir()) == []


def test_symlink_lease_registry_is_unknown(leases, tmp_path):
    folder, _, _ = leases
    moved = tmp_path / "moved"
    folder.rename(moved)
    folder.symlink_to(moved, target_is_directory=True)
    assert evidence.read_active_leases() is None


@pytest.mark.parametrize("status", ["running", "completed"])
@pytest.mark.parametrize("epoch,usable", [(1, False), (5, False), (6, True)])
def test_curriculum_peak_requires_full_size_completed_epoch(observed, status, epoch, usable):
    recipe = {
        **observed.plan["configurations"][0]["recipe"],
        "bagSizeMode": "fixed",
        "bagSize": 100,
        "bagCurriculum": True,
        "bagCurriculumStart": 10,
        "bagCurriculumEnd": 100,
        "bagCurriculumWarmupEpochs": 5,
    }
    observed.plan["configurations"][0]["recipe"] = recipe
    base = {
        key: value
        for key, value in observed.run_plan.items()
        if key not in {"effectiveRecipe", "nnmilPlanning"}
    }
    run_plan = resolve_nnmil_plan({**base, "recipe": recipe})
    observed.workload = workload_for_recipe(
        recipe,
        run_plan["data"]["memberships"],
        run_plan["data"]["featureFiles"],
        "protocol-hash",
        "bundle-hash",
        "mmap",
        2,
    )
    current = observed.state["runs"][0]
    current["status"] = status
    if status == "completed":
        result = {
            "runId": "run-one",
            "state": "succeeded",
            "epochsCompleted": epoch,
            "cudaPeakAllocatedBytes": 2 * 1024**3,
            "cudaPeakReservedBytes": 3 * 1024**3,
            **{key: run_plan[key] for key in ("effectiveRecipe", "nnmilPlanning")},
        }
        current["result"] = result
        save(observed.run_folder / "result.json", result)
    else:
        save(observed.run_folder / "progress.json", {**observed.progress, "epoch": epoch})
    observed.state["planHash"] = _hash(observed.plan)
    save(observed.batch_folder / "plan.json", observed.plan)
    save(observed.batch_folder / "state.json", observed.state)
    save(observed.run_folder / "plan.json", run_plan)
    rows = collect(observed)
    assert bool(rows) == usable
    if usable:
        assert rows[0]["trainingPatches"] == 100


@pytest.mark.parametrize("started", [None, "missing", "invalid", "2026-01-01T00:00:00"])
def test_missing_or_invalid_attempt_start_disables_live_evidence(observed, started):
    current = observed.state["runs"][0]
    if started == "missing":
        del current["startedAt"]
    else:
        current["startedAt"] = started
    save(observed.batch_folder / "state.json", observed.state)
    assert collect(observed) == []


def test_old_progress_cannot_be_reattributed_to_retry_on_another_gpu(observed):
    current = observed.state["runs"][0]
    current["gpu"] = 1
    observed.state["provenance"]["gpus"].append({"index": 1, "uuid": "GPU-1"})
    observed.plan["resources"]["gpuIds"].append(1)
    observed.state["planHash"] = _hash(observed.plan)
    save(observed.batch_folder / "plan.json", observed.plan)
    save(observed.batch_folder / "state.json", observed.state)
    save(observed.run_folder / "plan.json", observed.run_plan)
    path = observed.run_folder / "progress.json"
    old = (datetime.now(UTC) - timedelta(hours=1)).timestamp()
    os.utime(path, (old, old))
    assert collect(observed) == []
    save(path, observed.progress)
    assert collect(observed)[0]["gpuUuid"] == "GPU-1"


def test_progress_replaced_during_read_is_not_evidence(observed, monkeypatch):
    read = evidence.ScientificStore._read_file

    def changed(path, maximum):
        raw = read(path, maximum)
        if path.name == "progress.json":
            stamp = path.stat()
            os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1_000_000))
        return raw

    monkeypatch.setattr(evidence.ScientificStore, "_read_file", changed)
    assert collect(observed) == []
