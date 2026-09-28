"""Evaluation launch preserves reviewed cohorts, predictors and output provenance."""

import hashlib
import runpy
import sys
from pathlib import Path

import pytest
from support.evaluation import compute_tasks

from histopilot.application.compute_jobs import ComputeJobService
from histopilot.application.evaluation_runs import EvaluationRunService
from histopilot.schemas.predictors import EvaluationRunSelection, SaveEvaluationRun
from histopilot.storage.project_lock import StorageError
from histopilot.workers.packing_process import write_json
from histopilot.workers.training_process import read_json

support = runpy.run_path(str(Path(__file__).with_name("test_predictor_registry.py")))


@pytest.fixture
def evaluation(tmp_path, monkeypatch, task_center):
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
    service.jobs = ComputeJobService(
        service.store, runtime=runtime, task_center=task_center.client
    )
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
    return service, document, predictor, cohort


def test_evaluation_launch_uses_exact_saved_memberships_and_predictor_checkpoints(
    evaluation, task_center
):
    service, document, predictor, cohort = evaluation
    identity = document["id"]
    assert service.get(identity)["execution"]["status"] == "not_started"
    plan = service._execution_plan(identity)
    assert plan["data"]["memberships"] == cohort["manifest"]["memberships"]
    assert plan["checkpoints"] == predictor["manifest"]["checkpoints"]
    assert plan["target"] == predictor["manifest"]["target"]
    assert plan["method"] == "ensemble"
    state = service.launch(identity, "launch")
    assert state["status"] == "queued"
    assert service.launch(identity, "launch")["status"] == "queued"
    assert [(task["id"], task["attempt"]) for task in compute_tasks(task_center)] == [
        (state["taskId"], 1)
    ]


def test_accepted_evaluation_retry_never_rebuilds_expensive_plan(evaluation, task_center, monkeypatch):
    service, document, _, _ = evaluation
    first = service.launch(document["id"], "launch")
    monkeypatch.setattr(service, "_execution_plan", lambda *_: pytest.fail(
        "Accepted evaluation retry must not hold the project lock while rehashing source inputs"))
    assert service.launch(document["id"], "launch")["planHash"] == first["planHash"]
    assert [task["attempt"] for task in compute_tasks(task_center)] == [1]
    with pytest.raises(StorageError) as caught:
        service.launch(document["id"], "launch", resume=True)
    assert caught.value.code == "OPERATION_CONFLICT"


def test_retry_is_acknowledgement_but_new_resume_still_revalidates_sources(evaluation, task_center):
    service, document, predictor, _ = evaluation
    task_id = service.launch(document["id"], "launch")["taskId"]
    Path(predictor["manifest"]["checkpoints"][0]["path"]).write_bytes(b"changed")
    assert service.launch(document["id"], "launch")["status"] == "queued"
    task_center.finish(task_id, "interrupted", returncode=None, reason="lost")
    with pytest.raises(StorageError, match="checkpoint changed"):
        service.launch(document["id"], "resume", resume=True)
    assert task_center.task(task_id)["attempt"] == 1


def test_changed_predictor_checkpoint_blocks_launch_before_worker_creation(evaluation, task_center):
    service, document, predictor, _ = evaluation
    Path(predictor["manifest"]["checkpoints"][0]["path"]).write_bytes(b"changed")
    with pytest.raises(StorageError, match="checkpoint changed"):
        service.launch(document["id"], "launch")
    assert compute_tasks(task_center) == []


def test_auto_device_resume_keeps_initial_cpu_assignment_when_cuda_appears(
    evaluation, task_center, monkeypatch
):
    service, document, _, _ = evaluation
    identity = document["id"]
    original = service.launch(identity, "launch")
    runtime = {**service.jobs.runtime(), "cudaAvailable": True, "gpuCount": 1}
    monkeypatch.setattr("histopilot.application.evaluation_runs.training_runtime", lambda: runtime)
    service.jobs.runtime = lambda: runtime
    assert service._execution_plan(identity)["resources"]["gpuIds"] == []
    assert service.launch(identity, "launch")["planHash"] == original["planHash"]
    assert task_center.task(original["taskId"])["attempt"] == 1
    task_center.finish(original["taskId"], "interrupted", returncode=None, reason="lost")
    resumed = service.launch(identity, "resume", resume=True)
    assert resumed["planHash"] == original["planHash"]
    assert resumed["attempt"] == 2
    task = task_center.task(original["taskId"])
    assert (task["attempt"], task["request"]["lane"]) == (2, "cpu")


def test_auto_device_resume_reports_missing_saved_gpu_and_recovers(
    evaluation, task_center, monkeypatch
):
    service, document, _, _ = evaluation
    identity = document["id"]
    runtime = {**service.jobs.runtime(), "cudaAvailable": True, "gpuCount": 1}
    monkeypatch.setattr("histopilot.application.evaluation_runs.training_runtime", lambda: runtime)
    service.jobs.runtime = lambda: runtime
    original = service.launch(identity, "launch")
    task_center.finish(original["taskId"], "interrupted", returncode=None, reason="lost")
    runtime.update(cudaAvailable=False, gpuCount=0)
    assert service._execution_plan(identity)["resources"]["gpuIds"] == [0]
    with pytest.raises(StorageError) as error:
        service.launch(identity, "resume", resume=True)
    assert error.value.code == "TRAINING_GPU_UNAVAILABLE"
    assert task_center.task(original["taskId"])["attempt"] == 1
    runtime.update(cudaAvailable=True, gpuCount=1)
    resumed = service.launch(identity, "resume", resume=True)
    assert resumed["planHash"] == original["planHash"]
    assert resumed["attempt"] == 2
    task = task_center.task(original["taskId"])
    assert (task["attempt"], task["request"]["lane"]) == (2, "gpu")


def test_artifact_exports_require_completed_verified_output(evaluation, task_center):
    service, document, _, _ = evaluation
    identity = document["id"]
    task_id = service.launch(identity, "launch")["taskId"]
    with pytest.raises(StorageError, match="after the job finishes"):
        service.artifact(identity, "slide-predictions.csv")
    folder = service.jobs.folder(identity)
    path = folder / "slide-predictions.csv"
    content = b"slideId,prediction\nslide-1,class-0\n"
    path.write_bytes(content)
    # The worker records its verified outputs, then its task concludes.
    state = read_json(folder / "state.json")
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
    task_center.finish(task_id, "succeeded")
    assert service.artifact(identity, path.name) == content
    path.write_bytes(content + b"tampered")
    with pytest.raises(StorageError, match="output changed"):
        service.artifact(identity, path.name)
    with pytest.raises(StorageError, match="not found"):
        service.artifact(identity, "../../plan.json")


def test_review_namespace_override_is_a_structured_error(evaluation):
    service, _, predictor, cohort = evaluation
    selected = EvaluationRunSelection(predictorId=predictor["id"], cohortId=cohort["id"], name="Review", patientIdentifiers="independent")
    with pytest.raises(StorageError) as error:
        service._review_cohort(selected, {}, {"spec": {"purpose": "review"}})
    assert error.value.code == "INVALID_REVIEW_COHORT"
