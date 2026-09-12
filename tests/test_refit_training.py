"""Actual synthetic CPU refit: fixed epochs, no evaluation loaders, resumable state."""

import copy
import json
import runpy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

from histopilot.training.module import MILTrainModule  # noqa: E402
from histopilot.training.refit import RefitDataModule, train_refit  # noqa: E402

support = runpy.run_path(str(Path(__file__).with_name("test_mil_training.py")))


def plan(tmp_path):
    value = support["tiny_plan"](tmp_path)
    value["kind"] = "refit"
    value["epochBudget"] = {"epochs": 3, "percentile": 50}
    value["recipe"].update(maxEpochs=3, minEpochs=3, earlyStopping=False, bagSize=None, dropout=0.1)
    for row in value["data"]["memberships"]:
        row.update(partition="train", phase="refit", pool="development")
    return value


def test_fixed_epoch_refit_trains_whole_development_pool_without_validation(tmp_path, monkeypatch):
    value = plan(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("Refit must never run validation or test inference")

    monkeypatch.setattr(MILTrainModule, "validation_step", forbidden)
    result = train_refit(value, tmp_path / "model")
    assert result["epochsCompleted"] == 3
    assert not result["testDataUsed"] and not result["validationUsed"]
    assert result["trainingSlideCount"] == len(value["data"]["memberships"])
    history = json.loads(Path(result["historyPath"]).read_text())
    assert [row["epoch"] for row in history] == [0, 1, 2]
    assert all("validation" not in row for row in history)
    checkpoint = torch.load(result["lastCheckpointPath"], map_location="cpu", weights_only=True)
    assert len(checkpoint["metricsHistory"]) == 3
    assert checkpoint["optimizer_states"]
    loaded = MILTrainModule.load_from_checkpoint(
        result["bestCheckpointPath"], map_location="cpu", weights_only=True
    )
    assert loaded.recipe["maxEpochs"] == 3


def test_refit_rejects_external_data_or_unreviewed_epoch_budget(tmp_path):
    value = plan(tmp_path)
    invalid = copy.deepcopy(value)
    invalid["data"]["memberships"][0]["pool"] = "external_test"
    with pytest.raises(ValueError, match="development"):
        RefitDataModule(invalid)
    invalid = copy.deepcopy(value)
    invalid["recipe"]["earlyStopping"] = True
    with pytest.raises(ValueError, match="fixed epoch budget"):
        train_refit(invalid, tmp_path / "invalid")


@pytest.mark.parametrize("interrupted_step", [1, 7])
def test_refit_resume_replays_incomplete_epoch_with_exact_optimizer_and_rng(
    tmp_path, monkeypatch, interrupted_step
):
    from lightning.pytorch.utilities.exceptions import SIGTERMException

    value = plan(tmp_path)
    full = train_refit(copy.deepcopy(value), tmp_path / "full")
    original = MILTrainModule.training_step

    def interrupt(module, batch, batch_idx):
        if module.global_step >= interrupted_step:
            raise SIGTERMException()
        return original(module, batch, batch_idx)

    monkeypatch.setattr(MILTrainModule, "training_step", interrupt)
    output = tmp_path / "interrupted"
    with pytest.raises(SIGTERMException):
        train_refit(copy.deepcopy(value), output)
    assert not (output / "result.json").exists()
    monkeypatch.setattr(MILTrainModule, "training_step", original)
    resumed = train_refit(value, output, checkpoint_path=output / "last.ckpt")
    full_state = torch.load(full["lastCheckpointPath"], map_location="cpu", weights_only=True)
    resumed_state = torch.load(resumed["lastCheckpointPath"], map_location="cpu", weights_only=True)
    assert full_state["metricsHistory"] == resumed_state["metricsHistory"]
    for key, tensor in full_state["state_dict"].items():
        torch.testing.assert_close(tensor, resumed_state["state_dict"][key], rtol=0, atol=0)


def test_pinned_refit_worker_runs_created_plan_and_publishes_verified_predictor(tmp_path):
    import os
    import subprocess
    import sys

    from histopilot.application.compute_jobs import ComputeJobService
    from histopilot.schemas.development import ResourcePolicy
    from histopilot.schemas.predictors import LaunchRefit

    refit_support = runpy.run_path(str(Path(__file__).with_name("test_refit_predictors.py")))
    job_support = runpy.run_path(str(Path(__file__).with_name("test_compute_jobs.py")))
    service, _ = refit_support["registry"].__wrapped__(tmp_path)
    selection, _, _ = refit_support["refit_candidate"](service, epochs=(1, 1))
    executor = job_support["Executor"]()
    jobs = ComputeJobService(
        service.store,
        executor=executor,
        runtime=lambda: {
            "available": True,
            "python": sys.executable,
            "versions": {},
            "cudaAvailable": False,
            "gpuCount": 0,
        },
    )
    refits, record, _ = refit_support["create"](service, selection, jobs)
    refits.launch(
        record["id"],
        LaunchRefit(
            operationId="launch",
            resources=ResourcePolicy(
                gpuIds=[],
                dataLoaderWorkers=0,
                cpuThreadsPerRun=1,
                ramGbPerRun=0.01,
            ),
        ),
    )
    private_tmp = tmp_path / "worker-runtime"
    private_tmp.mkdir(mode=0o700)
    output = subprocess.run(
        [
            sys.executable,
            "-m",
            "histopilot.workers.compute_job",
            str(jobs.folder(record["id"]) / "plan.json"),
        ],
        cwd=jobs.folder(record["id"]) / "compute",
        env={**os.environ, "TMPDIR": str(private_tmp), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert output.returncode == 0, output.stderr
    state = jobs.status(record["id"])
    assert state["status"] == "completed", (state, output.stderr)
    published = refits.publish(record["id"], "publish-worker-model")
    assert published["manifest"]["method"] == "refit"
    assert published["manifest"]["checkpoints"][0]["bestEpoch"] == 1
    service.verify_checkpoints(published)
