"""The Phase 0 estimator reads saved runs read-only and explains its parallelism suggestion."""

import hashlib
import json
import os
from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pytest

from histopilot.taskcenter import estimator

T0 = datetime(2026, 9, 26, 7, 0, tzinfo=UTC)
GIB = 1024**3
RECIPE = {
    "model": "nnmil",
    "bagSizeMode": "training_median",
    "batchSize": 32,
    "evalBatchSize": 1,
    "attentionDim": 256,
    "precision": "bf16-mixed",
    "maxEpochs": 100,
}


def at(minutes):
    return (T0 + timedelta(minutes=minutes)).isoformat()


def host(**changes):
    return {
        "cpuCount": 36,
        "totalRamGb": 188.7,
        "availableRamGb": 179.0,
        "gpus": [
            {"index": 0, "name": "GPU A", "totalMemoryGb": 24.0, "freeMemoryGb": 23.0},
        ],
        **changes,
    }


def fold(train, evaluation, *, dim=64):
    """One split: fitting bags of the given sizes and one evaluation slide."""
    rows = [{"slideId": f"t{i}", "patientId": f"t{i}", "partition": "train"} for i in range(3)]
    rows.append({"slideId": "v", "patientId": "v", "partition": "val"})
    files = {f"t{i}": {"patchCount": train, "dimensions": dim} for i in range(3)}
    files["v"] = {"patchCount": evaluation, "dimensions": dim}
    return rows, files


def run(run_id, split, start, end, *, peak=2.0, epochs=10, status="completed", gpu=0, **extra):
    result = {
        "state": "succeeded",
        "runId": run_id,
        "cudaPeakReservedBytes": int(peak * GIB),
        "epochsCompleted": epochs,
        "resumedFrom": extra.pop("resumedFrom", None),
        "assessmentOnlyResume": extra.pop("assessmentOnlyResume", False),
    }
    return {
        "id": run_id,
        "split": split,
        "status": status,
        "gpu": gpu,
        "attempt": extra.pop("attempt", 1),
        "startedAt": at(start),
        "finishedAt": at(end),
        "result": result,
        **extra,
    }


def write_batch(project, batch_id, splits, runs, *, recipe=RECIPE, telemetry=None, gpu="gpu-a"):
    """A legacy batch folder: batch plan.json, state.json and optional telemetry.jsonl."""
    folder = project / "training" / batch_id
    folder.mkdir(parents=True)
    files = {}
    memberships = {}
    for split_id, (rows, split_files) in splits.items():
        memberships[split_id] = [{**row, "slideId": f"{split_id}-{row['slideId']}"} for row in rows]
        files.update({f"{split_id}-{slide}": entry for slide, entry in split_files.items()})
    plan = {
        "batchId": batch_id,
        "configurations": [{"id": "candidate-1", "number": 1, "recipe": recipe}],
        "runs": [
            {
                "id": row["id"],
                "candidateId": "candidate-1",
                "splitPlanId": row["split"],
                "trainingSeed": 1,
            }
            for row in runs
        ],
        "memberships": memberships,
        "data": {"featureFiles": files, "loadingPolicy": "mmap", "featureDim": 64},
        "resources": {"gpuIds": [0], "runsPerGpu": 4, "maxConcurrentRuns": 4},
    }
    state = {
        "batchId": batch_id,
        "status": "completed",
        "provenance": {"gpus": [{"index": 0, "uuid": gpu, "name": "GPU A"}]},
        "runs": [{key: value for key, value in row.items() if key != "split"} for row in runs],
    }
    (folder / "plan.json").write_text(json.dumps(plan))
    (folder / "state.json").write_text(json.dumps(state))
    if telemetry is not None:
        lines = [
            json.dumps(
                {
                    "at": at(index),
                    "host": {"totalRamGb": 100.0, "availableRamGb": 100.0 - used},
                    "gpus": [{"index": 0, "utilizationPercent": utilization}],
                    "runs": [{"runId": f"r{i}"} for i in range(runs_live)],
                }
            )
            for index, (runs_live, used, utilization) in enumerate(telemetry)
        ]
        (folder / "telemetry.jsonl").write_text("\n".join(lines) + "\n{partial")
    return folder


def tree(folder):
    return {
        str(path.relative_to(folder)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in folder.rglob("*")
    }


@pytest.fixture(autouse=True)
def empty_cache():
    estimator._cache.clear()
    yield
    estimator._cache.clear()


# --- Workloads ---------------------------------------------------------------------------


def profile(recipe=None, *, entries=None, memberships=None):
    rows = memberships or [
        {"slideId": "a", "patientId": "a", "partition": "train"},
        {"slideId": "b", "patientId": "b", "partition": "train"},
        {"slideId": "c", "patientId": "c", "partition": "val"},
        {"slideId": "d", "patientId": "d", "partition": "test"},
    ]
    files = entries or {
        identity: {"patchCount": count, "dimensions": 1024}
        for identity, count in zip("abcd", [100, 300, 2000, 8000], strict=True)
    }
    return estimator.fold_workloads(recipe or {}, rows, files, "mmap")


def test_fold_training_cap_excludes_validation_and_assessment_sizes():
    result = profile(
        {"model": "nnmil", "bagSizeMode": "training_median", "batchSize": 32, "evalBatchSize": 1}
    )
    assert result["trainingPatches"] == 100
    assert result["evaluationPatches"] == result["sourcePatches"] == 8000
    assert result["batchSize"] == 32 and result["evalBatchSize"] == 1
    assert estimator.workload_key(result) == result["key"] and len(result["key"]) == 24
    assert not {"protocolHash", "featureBundleHash", "classCount"} & set(result)


def test_fold_full_bags_independent_evaluation_cap_and_curriculum():
    whole = profile({"bagSize": None, "batchSize": 3})
    assert whole["trainingPatches"] == 300 and whole["evaluationPatches"] == 8000
    assert whole["evalBatchSize"] == 3
    capped = profile({"bagSize": 50, "evalBagSize": 20})
    assert capped["trainingPatches"] == 50 and capped["evaluationPatches"] == 20
    assert capped["sourcePatches"] == 8000
    recipe = {
        "bagSize": 30,
        "bagCurriculum": True,
        "bagCurriculumStart": 20,
        "bagCurriculumEnd": 250,
    }
    before = deepcopy(recipe)
    assert profile(recipe)["trainingPatches"] == 250
    assert recipe == before


def test_relaxed_key_ignores_optimizer_loss_protocol_and_patch_counts():
    key = profile()["key"]
    for recipe in (
        {"learningRate": 0.003, "weightDecay": 0.0, "evalBatchSize": 1},
        {"optimizer": "sgd"},
        {"lossType": "focal"},
        {"classWeighting": "inverse_prevalence"},
        {"dropout": 0.5},
        {"maxEpochs": 7},
    ):
        assert profile(recipe)["key"] == key, recipe
    other_fold = profile(
        entries={
            identity: {"patchCount": count, "dimensions": 1024}
            for identity, count in zip("abcd", [900, 700, 20, 80], strict=True)
        }
    )
    assert (
        other_fold["key"] == key and other_fold["trainingPatches"] != profile()["trainingPatches"]
    )
    for recipe in (
        {"batchSize": 2},
        {"precision": "16-mixed"},
        {"model": "nnmil"},
        {"attentionDim": 256},
        {"evalBatchSize": 2},
        {"gradientCheckpointing": True},
    ):
        assert profile(recipe)["key"] != key, recipe
    workload = profile()
    assert estimator.workload_key({**workload, "loadingPolicy": "native"}) != key
    assert estimator.workload_key({**workload, "featureDimension": 512}) != key


@pytest.mark.parametrize("problem", ["duplicate", "missing", "dimension", "count", "no_fit"])
def test_incomplete_or_ambiguous_fold_is_not_guessed(problem):
    rows = [{"slideId": "a", "partition": "train"}, {"slideId": "b", "partition": "val"}]
    files = {identity: {"patchCount": 100, "dimensions": 1024} for identity in "ab"}
    if problem == "duplicate":
        rows.append(rows[0])
    elif problem == "missing":
        del files["a"]
    elif problem == "dimension":
        files["b"]["dimensions"] = 512
    elif problem == "count":
        files["b"]["patchCount"] = 0
    else:
        rows[0]["partition"] = "test"
    with pytest.raises(ValueError):
        profile(entries=files, memberships=rows)


# --- Evidence ----------------------------------------------------------------------------


def test_training_evidence_reads_completed_gpu_runs_without_writing(tmp_path, monkeypatch):
    project = tmp_path / "project"
    folder = write_batch(
        project,
        "batch-1",
        {"f0": fold(1000, 4000)},
        [
            run("run-ok", "f0", 0, 10, peak=1.5, epochs=20),
            run("run-cpu", "f0", 0, 10, gpu=None),
            run("run-failed", "f0", 0, 10, status="failed"),
            run("run-zero", "f0", 0, 10, peak=0),
            run("run-resumed", "f0", 10, 20, resumedFrom="last.ckpt"),
            run("run-retry", "f0", 10, 20, attempt=2),
        ],
    )
    # A fold plan is several MB on real batches and must never be read.
    (folder / "runs" / "run-ok").mkdir(parents=True)
    (folder / "runs" / "run-ok" / "plan.json").write_text("{not json")
    read = []
    original = estimator._read_bytes
    monkeypatch.setattr(
        estimator,
        "_read_bytes",
        lambda path, *args: read.append(path.relative_to(folder)) or original(path, *args),
    )
    before = tree(project)
    rows = estimator.observations_from_training(project)
    assert tree(project) == before
    assert [str(path) for path in read] == ["plan.json", "state.json", "telemetry.jsonl"]
    assert [row["runId"] for row in rows] == ["run-ok", "run-resumed", "run-retry"]
    first = rows[0]
    assert first["source"] == "training" and first["batchId"] == "batch-1"
    assert first["gpuIndex"] == 0 and first["gpuUuid"] == "gpu-a" and first["gpuName"] == "GPU A"
    assert first["peakVramGb"] == pytest.approx(1.5)
    assert first["epochs"] == 20 and first["wallSeconds"] == 600
    assert first["secondsPerEpoch"] == 30 and first["resumed"] is False
    assert first["workload"]["trainingPatches"] == 500  # Half the fitting median.
    assert first["workload"]["evaluationPatches"] == 4000
    assert first["workloadKey"] == first["workload"]["key"]
    assert (
        first["taskId"]
        == "task-"
        + hashlib.sha256("\0".join(["mil-fold", str(folder), "run-ok"]).encode()).hexdigest()[:32]
    )
    assert [row["resumed"] for row in rows[1:]] == [True, True]
    assert rows.diagnostics == {"batches": 1, "used": 1, "skipped": 0, "errors": []}


def test_fold_task_ids_match_the_task_center_formula(tmp_path):
    ids = pytest.importorskip("histopilot.taskcenter.ids")
    folder = tmp_path / "project" / "training" / "batch"
    assert estimator._fold_task_id(folder, "run-1") == ids.fold_task_id(str(folder), "run-1")


def test_unreadable_batches_are_skipped_and_counted(tmp_path):
    project = tmp_path / "project"
    write_batch(project, "good", {"f0": fold(1000, 4000)}, [run("run-1", "f0", 0, 10)])
    broken = write_batch(project, "broken", {"f0": fold(1000, 4000)}, [run("run-2", "f0", 0, 10)])
    (broken / "state.json").write_text("{")
    nonfinite = write_batch(
        project, "nonfinite", {"f0": fold(1000, 4000)}, [run("run-3", "f0", 0, 10)]
    )
    state = json.loads((nonfinite / "state.json").read_text())
    (nonfinite / "state.json").write_text(json.dumps(state).replace('"attempt": 1', '"x": NaN'))
    elsewhere = write_batch(
        tmp_path / "other", "linked", {"f0": fold(1000, 4000)}, [run("run-4", "f0", 0, 10)]
    )
    os.symlink(elsewhere, project / "training" / "linked")
    (project / "training" / "empty").mkdir()
    rows = estimator.observations_from_training(project)
    assert [row["runId"] for row in rows] == ["run-1"]
    assert rows.diagnostics["skipped"] == 2 and rows.diagnostics["used"] == 1
    assert estimator.observations_from_training(tmp_path / "missing") == []
    assert not (tmp_path / "missing").exists()


def test_cached_batches_are_reparsed_only_when_files_change(tmp_path, monkeypatch):
    project = tmp_path / "project"
    folder = write_batch(project, "batch", {"f0": fold(1000, 4000)}, [run("run-1", "f0", 0, 10)])
    first = estimator.observations_from_training(project)
    first[0]["meanConcurrency"] = 99
    monkeypatch.setattr(estimator, "_batch_observations", lambda *args: pytest.fail("reparsed"))
    again = estimator.observations_from_training(project)
    assert again[0]["meanConcurrency"] is None  # Callers receive copies.
    state = json.loads((folder / "state.json").read_text())
    state["runs"][0]["result"]["cudaPeakReservedBytes"] = 3 * GIB
    state["note"] = "changed"
    (folder / "state.json").write_text(json.dumps(state))
    monkeypatch.undo()
    assert estimator.observations_from_training(project)[0]["peakVramGb"] == pytest.approx(3.0)


def test_concurrency_is_time_weighted_across_batches_projects_and_gpus(tmp_path):
    write_batch(tmp_path / "a", "batch-a", {"f0": fold(1000, 4000)}, [run("a1", "f0", 0, 100)])
    write_batch(tmp_path / "b", "batch-b", {"f0": fold(1000, 4000)}, [run("b1", "f0", 50, 150)])
    write_batch(
        tmp_path / "c", "batch-c", {"f0": fold(1000, 4000)}, [run("c1", "f0", 0, 100)], gpu="gpu-b"
    )
    measured = [
        {
            "taskId": "task-x",
            "gpuUuid": "gpu-a",
            "startedAt": at(0),
            "finishedAt": at(100),
            "meanConcurrency": 7.0,
            "exitReason": "ok",
        },
        {
            "taskId": "task-cpu",
            "gpuIndex": None,
            "startedAt": at(0),
            "finishedAt": at(100),
            "exitReason": "ok",
        },
    ]
    rows = estimator.gather_observations(
        [tmp_path / "a", tmp_path / "b", tmp_path / "c"], measurements=measured
    )
    by = {row["runId"] or row["taskId"]: row["meanConcurrency"] for row in rows}
    # a1 overlaps b1 for half its time and the measured task for all of it.
    assert by == {"a1": 2.5, "b1": 2.0, "c1": 1.0, "task-x": 7.0, "task-cpu": None}


def test_managed_fold_measurement_merges_into_its_training_run(tmp_path):
    project = tmp_path / "project"
    folder = write_batch(project, "batch", {"f0": fold(1000, 4000)}, [run("run-1", "f0", 0, 10)])
    task = estimator._fold_task_id(folder, "run-1")
    rows = estimator.gather_observations(
        [project],
        measurements=[
            {
                "taskId": task,
                "attempt": 1,
                "exitReason": "ok",
                "gpuUuid": "gpu-a",
                "peakPrivateRamGb": 4.2,
                "meanCpuCores": 2.5,
                "startedAt": at(0),
                "finishedAt": at(10),
                "workload": json.dumps({"model": "nnmil"}),
            },
            {
                "taskId": task,
                "attempt": 2,
                "exitReason": "oom",
                "gpuUuid": "gpu-a",
                "peakVramGb": 20.0,
                "startedAt": at(20),
                "finishedAt": at(21),
            },
        ],
    )
    assert len(rows) == 2
    merged = next(row for row in rows if row["source"] == "training")
    assert merged["peakPrivateRamGb"] == 4.2 and merged["meanCpuCores"] == 2.5
    oom = next(row for row in rows if row["source"] == "measurement")
    assert oom["ok"] is False and oom["resumed"] is True
    result = estimator.estimate(merged["workload"], rows)
    assert result["ramGb"] == pytest.approx(4.7) and result["cpuCores"] == 2.5


# --- Estimates ---------------------------------------------------------------------------


def observation(workload, peak, **changes):
    return {
        "source": "training",
        "batchId": "batch",
        "runId": f"run-{peak}-{workload['evaluationPatches']}",
        "workloadKey": workload["key"],
        "workload": workload,
        "gpuIndex": 0,
        "gpuUuid": "gpu-a",
        "gpuName": "GPU A",
        "peakVramGb": peak,
        "ok": True,
        "telemetry": None,
        **changes,
    }


def sized(train, evaluation):
    rows, files = fold(train * 2, evaluation)
    return estimator.fold_workloads(RECIPE, rows, files, "mmap")


def test_each_fold_scales_from_its_nearest_measurement():
    small, large = sized(1000, 1000), sized(1000, 1500)
    observations = [observation(small, 2.0), observation(large, 2.2)]
    # The large fold was measured itself; scaling the small fold by 1.5x would overstate it.
    assert estimator.estimate(large, observations)["vramGb"] == round(2.2 * 1.15 + 0.3, 2)
    near = estimator.estimate(sized(1000, 1800), observations)
    assert near["basis"] == "measured" and near["sources"]["vram"] == "measured"
    assert near["vramGb"] == round(2.2 * 1.2 * 1.15 + 0.3, 2)
    # Smaller requests never reduce an observed peak.
    assert estimator.estimate(sized(500, 500), observations)["vramGb"] == round(2.2 * 1.15 + 0.3, 2)


def test_far_larger_bags_use_the_formula_calibrated_by_measurements():
    small, large = sized(1000, 1000), sized(1000, 1500)
    observations = [observation(small, 2.0), observation(large, 2.2)]
    huge = sized(1000, 6000)
    result = estimator.estimate(huge, observations)
    ratio = sorted([2.0 / estimator.gpu_estimate(small), 2.2 / estimator.gpu_estimate(large)])
    expected = estimator.gpu_estimate(huge) * (sum(ratio) / 2) * 1.15 + 0.3
    assert result["basis"] == "estimated" and result["sources"]["vram"] == "calibrated"
    assert result["vramGb"] == round(expected, 2) and result["evidence"] == 2


def test_unmatched_workloads_fall_back_to_formulas():
    workload = sized(1000, 1000)
    other = {
        **workload,
        "batchSize": 1,
        "key": estimator.workload_key({**workload, "batchSize": 1}),
    }
    result = estimator.estimate(workload, [observation(other, 1.0)], data_workers=2)
    assert result["basis"] == "estimated" and result["evidence"] == 0
    assert result["vramGb"] == round(estimator.gpu_estimate(workload), 2)
    assert result["ramGb"] == round(estimator.ram_formula(workload, 2) + 0.5, 2)
    assert result["cpuCores"] == 3.0
    assert result["sources"] == {"vram": "formula", "ram": "formula", "cpu": "default"}


def test_ram_comes_from_telemetry_deltas_and_measurements_override(tmp_path):
    project = tmp_path / "project"
    write_batch(
        project,
        "batch",
        {"f0": fold(1000, 4000)},
        [run("run-1", "f0", 0, 10)],
        # (live runs, host used GiB, GPU %): baseline is the last idle sample before runs.
        telemetry=[(0, 30.0, 0), (0, 10.0, 0), (2, 18.0, 40), (2, 19.0, 60), (1, 14.0, 20)],
    )
    rows = estimator.observations_from_training(project)
    telemetry = rows[0]["telemetry"]
    assert telemetry["ramPerRunGb"] == pytest.approx(4.0) and telemetry["ramSamples"] == 3
    assert telemetry["gpuUtilization"]["2"] == {"meanPercent": 50.0, "samples": 2}
    workload = rows[0]["workload"]
    result = estimator.estimate(workload, rows)
    assert result["ramGb"] == 4.5 and result["sources"]["ram"] == "telemetry"
    measured = [{**rows[0], "peakPrivateRamGb": 6.0}]
    assert estimator.estimate(workload, measured)["ramGb"] == 6.5


def test_measured_ram_is_a_high_percentile_so_one_inflated_peak_cannot_dominate():
    workload = sized(1000, 1000)
    peaks = [4.0, 4.2, 4.4, 4.1, 4.3] * 3
    rows = [
        observation(workload, 1.5, runId=f"run-{index}", peakPrivateRamGb=peak)
        for index, peak in enumerate([*peaks, 44.5])  # one fold that alone mapped its pack
    ]
    result = estimator.estimate(workload, rows)
    assert result["sources"]["ram"] == "measured"
    assert result["ramGb"] == pytest.approx(4.4 + 0.5)
    # Also among a few readings; one or two are used as they are, leaning to the higher.
    assert estimator.estimate(workload, [*rows[:2], rows[-1]])["ramGb"] == pytest.approx(4.68)
    assert estimator.estimate(workload, rows[:1])["ramGb"] == pytest.approx(4.5)
    two = estimator.estimate(workload, rows[-2:])["ramGb"]
    assert two == pytest.approx(4.3 + 0.9 * (44.5 - 4.3) + 0.5, abs=0.01)


def test_measured_ram_of_larger_folds_is_not_set_aside_for_a_fold_of_their_size():
    # One workload key spans cohorts: most readings come from small bags, a few from a
    # cohort whose bags are six times larger and legitimately need twice the RAM.
    small, large = sized(1000, 1000), sized(1000, 6000)
    rows = [
        observation(small, 1.5, runId=f"small-{index}", peakPrivateRamGb=4.0 + 0.05 * index)
        for index in range(10)
    ]
    rows += [
        observation(large, 3.0, runId=f"large-{index}", peakPrivateRamGb=9.0 + 0.1 * index)
        for index in range(3)
    ]
    assert estimator.estimate(large, rows)["ramGb"] == pytest.approx(9.18 + 0.5)
    # A small fold is not charged for the larger cohort, and a fold beyond every measured
    # size leans on the largest measured folds, never on the smallest.
    assert estimator.estimate(small, rows)["ramGb"] == pytest.approx(4.41 + 0.5, abs=0.01)
    assert estimator.estimate(sized(1000, 20000), rows)["ramGb"] == pytest.approx(9.68)


def test_telemetry_without_an_idle_sample_uses_the_minimum(tmp_path):
    project = tmp_path / "project"
    write_batch(
        project,
        "batch",
        {"f0": fold(1000, 4000)},
        [run("run-1", "f0", 0, 10)],
        telemetry=[(2, 20.0, 50), (1, 12.0, 50), (2, 22.0, 50)],
    )
    telemetry = estimator.observations_from_training(project)[0]["telemetry"]
    assert telemetry["ramPerRunGb"] == pytest.approx(4.0)  # median of 4, 0 and 5


# --- Suggestions -------------------------------------------------------------------------


def timed(workload, concurrency, seconds, **changes):
    return observation(
        workload,
        1.5,
        meanConcurrency=concurrency,
        secondsPerEpoch=seconds,
        resumed=changes.pop("resumed", False),
        runId=f"run-{concurrency}-{seconds}-{changes.pop('index', 0)}",
        **changes,
    )


def test_rising_throughput_suggests_one_more_than_measured_from_saved_runs(tmp_path):
    project = tmp_path / "project"
    splits = {f"f{i}": fold(1000 + 20 * i, 4000) for i in range(3)}
    # Four at once with near-constant epoch time, then a two-run tail.
    runs = [run(f"run-{i}", f"f{i % 3}", 0, 40, epochs=10) for i in range(4)]
    runs += [run("run-4", "f0", 40, 72, epochs=8), run("run-5", "f1", 40, 72, epochs=8)]
    write_batch(project, "batch", splits, runs)
    observations = estimator.gather_observations([project])
    assert sorted(row["meanConcurrency"] for row in observations) == [2, 2, 4, 4, 4, 4]
    result = estimator.suggest([], observations, host(), None)
    assert result["basis"] == "measured" and result["binding"] == "throughput"
    assert result["parallelGpuTasks"] == 5 and result["perGpu"] == {"0": 5}
    assert result["throughput"]["trend"] == "rising"
    assert result["throughput"]["observed"] == [
        {"concurrency": 4.0, "secondsPerEpoch": 240.0, "runs": 4},
        {"concurrency": 2.0, "secondsPerEpoch": 240.0, "runs": 2},
    ]
    assert result["limits"]["throughput"] == 5 and result["limits"]["runCount"] is None
    assert result["evidence"]["runs"] == 6 and result["evidence"]["batches"] == 1
    assert result["evidence"]["latest"] == at(72)
    assert any("throughput was still rising" in text for text in result["explanation"])
    assert any(text.startswith("GPU memory would allow") for text in result["explanation"])
    json.dumps(result, allow_nan=False)


def test_flat_or_falling_throughput_stops_at_the_knee():
    workload = sized(1000, 1000)
    flat = [timed(workload, 3.0, 40.0), timed(workload, 4.0, 50.0)]
    result = estimator.suggest([workload], flat, host(), None)
    assert result["throughput"]["trend"] == "flat" and result["limits"]["throughput"] == 4
    assert result["parallelGpuTasks"] == 4 and result["binding"] == "throughput"
    falling = [timed(workload, 2.0, 16.0), timed(workload, 4.0, 32.0)]
    result = estimator.suggest([workload], falling, host(), None)
    assert result["throughput"]["trend"] == "falling"
    assert result["parallelGpuTasks"] == 1 and result["binding"] == "throughput"


@pytest.mark.parametrize(
    ("utilization", "trend", "limit"),
    [(52.0, "rising", 5), (90.0, "flat", 4), (None, "unknown", 4)],
)
def test_single_concurrency_level_uses_gpu_utilisation(utilization, trend, limit):
    workload = sized(1000, 1000)
    telemetry = {"batchId": "batch", "ramPerRunGb": 4.0, "gpuUtilization": {}}
    if utilization is not None:
        telemetry["gpuUtilization"]["4"] = {"meanPercent": utilization, "samples": 10}
    rows = [timed(workload, 3.9, 16.0, index=i, telemetry=telemetry) for i in range(4)]
    result = estimator.suggest([workload], rows, host(), None)
    assert result["throughput"]["trend"] == trend and result["limits"]["throughput"] == limit
    assert result["parallelGpuTasks"] == limit


def test_resumed_runs_and_other_gpu_models_do_not_shape_throughput():
    workload = sized(1000, 1000)
    rows = [
        timed(workload, 4.0, 16.0),
        timed(workload, 2.0, 90.0, resumed=True),
        timed(workload, 1.0, 5.0, gpuName="Other GPU"),
    ]
    result = estimator.suggest([workload], rows, host(), None)
    assert result["throughput"]["observed"] == [
        {"concurrency": 4.0, "secondsPerEpoch": 16.0, "runs": 1}
    ]


def test_hardware_only_starts_at_four_or_the_smallest_resource():
    result = estimator.suggest([], [], host(), None)
    assert result["basis"] == "hardware_only" and result["binding"] == "default"
    assert result["parallelGpuTasks"] == 4
    assert result["perTask"] == {"vramGb": 3.0, "ramGb": 6.0, "cpuCores": 3.0, "cpuThreads": 4}
    assert result["limits"]["gpuMemory"] == 7 and result["limits"]["throughput"] is None
    small = host(gpus=[{"index": 0, "name": "GPU A", "totalMemoryGb": 8.0}])
    result = estimator.suggest([], [], small, None)
    assert result["parallelGpuTasks"] == 2 and result["binding"] == "gpu_memory"


def test_unmeasured_workloads_are_estimated_and_start_at_the_default():
    workload = sized(1000, 1000)
    result = estimator.suggest([workload], [], host(), None)
    assert result["basis"] == "estimated" and result["binding"] == "default"
    assert result["parallelGpuTasks"] == 4
    assert result["perTask"]["vramGb"] == round(estimator.gpu_estimate(workload), 2)
    assert "CAPACITY_UNMEASURED" in {row["code"] for row in result["findings"]}


def test_binding_is_the_smallest_limit_and_throughput_wins_ties():
    workload = sized(1000, 1000)
    rows = [timed(workload, 3.0, 40.0), timed(workload, 4.0, 50.0)]  # throughput 4
    tied = host(gpus=[{"index": 0, "name": "GPU A", "totalMemoryGb": 1 + 4 * 2.02}])
    result = estimator.suggest([workload], rows, tied, None)
    assert result["limits"]["gpuMemory"] == 4 and result["binding"] == "throughput"
    result = estimator.suggest([workload], rows, host(availableRamGb=20.0), None)
    assert result["binding"] == "ram" and result["parallelGpuTasks"] == result["limits"]["ram"]
    result = estimator.suggest([workload], rows, host(), None, run_count=2)
    assert result["binding"] == "run_count" and result["parallelGpuTasks"] == 2
    result = estimator.suggest([workload], rows, host(cpuCount=4), None)
    assert result["binding"] == "cpu" and result["limits"]["cpu"] == 0
    assert result["parallelGpuTasks"] == 1
    assert "CAPACITY_EXHAUSTED" in {row["code"] for row in result["findings"]}


def test_settings_reserves_and_running_private_ram_shape_the_limits():
    workload = sized(1000, 1000)
    rows = [timed(workload, 4.0, 16.0, peakPrivateRamGb=4.5, meanCpuCores=2.0)]
    settings = {
        "reserves": {"cpuThreads": 6, "ramGb": 10.0, "vramGb": 4.0},
        "defaults": {"cpuThreadsPerRun": 1, "dataLoaderWorkers": 0},
    }
    result = estimator.suggest([workload], rows, host(availableRamGb=40.0), settings)
    assert result["perTask"] == {"vramGb": 2.02, "ramGb": 5.0, "cpuCores": 2.0, "cpuThreads": 1}
    assert result["limits"]["gpuMemory"] == 9  # (24 - 4) / 2.02
    # (36 - 6) / (1 thread + 0 workers): the measured 2 cores do not shrink the reservation.
    assert result["limits"]["ram"] == 6 and result["limits"]["cpu"] == 30
    assert "Each task reserves 1 CPU thread: 1 for computing." in result["explanation"]
    running = [{"resources": {"privateRamGb": 10.0}}, {"resources": None}]
    result = estimator.suggest(
        [workload], rows, host(availableRamGb=40.0), settings, running=running
    )
    assert result["limits"]["ram"] == 8


@pytest.mark.parametrize(("cpus", "limit"), [(36, 8), (16, 3)])
def test_cpu_limit_uses_the_threads_admission_reserves(cpus, limit):
    from histopilot.taskcenter import capacity
    from histopilot.taskcenter.model import DEFAULT_SETTINGS, merge_settings, normalize_request

    workload = sized(1000, 1000)
    rows = [timed(workload, 4.0, 16.0, meanCpuCores=1.0)]
    machine = host(cpuCount=cpus, gpus=[{"index": 0, "name": "GPU A", "totalMemoryGb": 48.0}])
    settings = merge_settings({**DEFAULT_SETTINGS, "defaultGpuSlots": 16})
    result = estimator.suggest([workload], rows, machine, settings)
    assert result["perTask"]["cpuCores"] == 1.0 and result["perTask"]["cpuThreads"] == 4
    assert result["limits"]["cpu"] == limit
    reserved = "Each task reserves 4 CPU threads: 2 for computing and 2 for loading data."
    assert reserved in result["explanation"]
    # The runner admits exactly as many default-sized tasks before CPU threads run out.
    effective = capacity.effective(settings, machine)
    usage = capacity.usage([], [], machine)
    request = normalize_request({"lane": "gpu", "cpuThreads": 2, "dataWorkers": 2, "vramGb": 1.0})
    admitted = 0
    while admitted < 16:
        ok, gpu, reason = capacity.admit({"request": request}, usage, machine, effective)
        if not ok:
            assert reason.startswith("Waiting for CPU threads")
            break
        capacity.claim(usage, {"request": request}, gpu)
        admitted += 1
    assert admitted == limit


def test_this_workstation_still_gets_five_limited_by_throughput():
    """36 logical CPUs, 188 GiB RAM, one 24 GiB RTX A5000; 1.2-1.75 GiB VRAM, ~5 GiB RAM."""
    workload = sized(1000, 1000)
    telemetry = {
        "batchId": "batch",
        "ramPerRunGb": 4.64,
        "gpuUtilization": {"4": {"meanPercent": 52.0, "samples": 108}},
    }
    rows = [
        timed(workload, 4.0, 16.8, index=index, telemetry=telemetry, peakVramGb=peak)
        for index, peak in enumerate([1.21, 1.48, 1.69, 1.75] * 3)
    ]
    rows += [timed(workload, 2.7, 15.9, index=20, telemetry=telemetry, peakVramGb=1.22)]
    machine = host(
        totalRamGb=188.7,
        availableRamGb=179.0,
        gpus=[{"index": 0, "name": "NVIDIA RTX A5000", "totalMemoryGb": 24.0}],
    )
    result = estimator.suggest([workload], rows, machine, None)
    assert result["parallelGpuTasks"] == 5 and result["binding"] == "throughput"
    assert result["limits"]["cpu"] == 8 and result["limits"]["gpuMemory"] == 9
    assert result["perTask"]["ramGb"] == pytest.approx(5.14)


def test_multiple_gpus_share_host_limits_per_gpu():
    workload = sized(1000, 1000)
    gpus = [
        {"index": 0, "name": "GPU A", "totalMemoryGb": 24.0},
        {"index": 1, "name": "GPU A", "totalMemoryGb": 12.0},
        {"index": 2, "name": "GPU A", "totalMemoryGb": None},
    ]
    result = estimator.suggest([workload], [], host(gpus=gpus, cpuCount=18), None, run_count=5)
    assert result["limits"]["cpu"] == 2 and result["limits"]["runCount"] == 3
    assert set(result["perGpu"]) == {"0", "1"} and result["parallelGpuTasks"] == 2


def test_no_gpu_host_is_reported_without_a_gpu_limit():
    result = estimator.suggest([], [], host(gpus=[]), None)
    assert result["limits"]["gpuMemory"] is None and result["perGpu"] == {}
    assert "CAPACITY_NO_GPU" in {row["code"] for row in result["findings"]}
