"""Evaluation launch preserves reviewed cohorts, predictors and output provenance."""

import hashlib
import runpy
import sys
from pathlib import Path

import pytest

from histopilot.application.compute_jobs import ComputeJobService
from histopilot.application.evaluation_runs import EvaluationRunService
from histopilot.schemas.predictors import EvaluationRunSelection, SaveEvaluationRun
from histopilot.storage.project_lock import StorageError
from histopilot.workers.packing_process import write_json

support = runpy.run_path(str(Path(__file__).with_name("test_predictor_registry.py")))
jobs = runpy.run_path(str(Path(__file__).with_name("test_compute_jobs.py")))


@pytest.fixture
def evaluation(tmp_path, monkeypatch):
    predictors, cohort = support["registry"].__wrapped__(tmp_path)
    selection, *_ = support["candidate"](predictors)
    predictor, _ = support["freeze"](predictors, selection)
    service = EvaluationRunService(predictors.store, predictors.filesystem)

    def runtime():
        return {
            "available": True,
            "python": sys.executable,
            "versions": {},
            "cudaAvailable": False,
            "gpuCount": 0,
        }

    monkeypatch.setattr("histopilot.application.evaluation_runs.training_runtime", runtime)
    executor = jobs["Executor"]()
    service.jobs = ComputeJobService(service.store, executor=executor, runtime=runtime)
    selected = EvaluationRunSelection(
        predictorId=predictor["id"], cohortId=cohort["id"], name="External evaluation"
    )
    preview = service.preview(selected)
    assert preview["canSave"], preview
    document = service.save(
        SaveEvaluationRun(
            **selected.model_dump(), previewHash=preview["previewHash"], operationId="evaluation"
        )
    )
    return service, document, predictor, cohort, executor


def test_evaluation_launch_uses_exact_saved_memberships_and_predictor_checkpoints(evaluation):
    service, document, predictor, cohort, executor = evaluation
    identity = document["id"]
    assert service.get(identity)["execution"]["status"] == "not_started"
    plan = service._execution_plan(identity)
    assert plan["data"]["memberships"] == cohort["manifest"]["memberships"]
    assert plan["checkpoints"] == predictor["manifest"]["checkpoints"]
    assert plan["target"] == predictor["manifest"]["target"]
    assert plan["method"] == "ensemble"
    assert service.launch(identity, "launch")["status"] == "queued"
    assert service.launch(identity, "launch")["status"] == "queued"
    assert len(executor.calls) == 1


def test_changed_predictor_checkpoint_blocks_launch_before_worker_creation(evaluation):
    service, document, predictor, _, executor = evaluation
    Path(predictor["manifest"]["checkpoints"][0]["path"]).write_bytes(b"changed")
    with pytest.raises(StorageError, match="checkpoint changed"):
        service.launch(document["id"], "launch")
    assert not executor.calls


def test_auto_device_resume_keeps_initial_cpu_assignment_when_cuda_appears(evaluation, monkeypatch):
    service, document, _, _, executor = evaluation
    identity = document["id"]
    original = service.launch(identity, "launch")
    runtime = {**service.jobs.runtime(), "cudaAvailable": True, "gpuCount": 1}
    monkeypatch.setattr("histopilot.application.evaluation_runs.training_runtime", lambda: runtime)
    service.jobs.runtime = lambda: runtime
    assert service._execution_plan(identity)["resources"]["gpuIds"] == []
    assert service.launch(identity, "launch")["planHash"] == original["planHash"]
    assert len(executor.calls) == 1
    executor.sessions.clear()
    resumed = service.launch(identity, "resume", resume=True)
    assert resumed["planHash"] == original["planHash"]
    assert resumed["attempt"] == 2


def test_auto_device_resume_reports_missing_saved_gpu_and_recovers(evaluation, monkeypatch):
    service, document, _, _, executor = evaluation
    identity = document["id"]
    runtime = {**service.jobs.runtime(), "cudaAvailable": True, "gpuCount": 1}
    monkeypatch.setattr("histopilot.application.evaluation_runs.training_runtime", lambda: runtime)
    service.jobs.runtime = lambda: runtime
    original = service.launch(identity, "launch")
    executor.sessions.clear()
    runtime.update(cudaAvailable=False, gpuCount=0)
    assert service._execution_plan(identity)["resources"]["gpuIds"] == [0]
    with pytest.raises(StorageError) as error:
        service.launch(identity, "resume", resume=True)
    assert error.value.code == "TRAINING_GPU_UNAVAILABLE"
    assert len(executor.calls) == 1
    runtime.update(cudaAvailable=True, gpuCount=1)
    resumed = service.launch(identity, "resume", resume=True)
    assert resumed["planHash"] == original["planHash"]
    assert resumed["attempt"] == 2


def test_artifact_exports_require_completed_verified_output(evaluation):
    service, document, _, _, executor = evaluation
    identity = document["id"]
    state = service.launch(identity, "launch")
    with pytest.raises(StorageError, match="after the job finishes"):
        service.artifact(identity, "slide-predictions.csv")
    folder = service.jobs.folder(identity)
    path = folder / "slide-predictions.csv"
    content = b"slideId,prediction\nslide-1,class-0\n"
    path.write_bytes(content)
    state.update(
        status="completed",
        result={
            "state": "succeeded",
            "runId": identity,
            "artifacts": {
                path.name: {
                    "path": str(path),
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            },
        },
    )
    write_json(folder / "result.json", state["result"])
    write_json(folder / "state.json", state)
    executor.sessions.clear()
    assert service.artifact(identity, path.name) == content
    path.write_bytes(content + b"tampered")
    with pytest.raises(StorageError, match="output changed"):
        service.artifact(identity, path.name)
    with pytest.raises(StorageError, match="not found"):
        service.artifact(identity, "../../plan.json")
