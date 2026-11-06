"""A run's dataset shift analysis: previewed, queued as a CPU compute task, run from its
frozen plan without labels, and read back only through its receipts. And the model card that
reports it."""

import json

import pytest
from fastapi.testclient import TestClient
from support.compute import complete, compute_tasks
from support.predictors import candidate, freeze, registry
from support.training import runtime

from histopilot.application.compute_jobs import ComputeJobService
from histopilot.application.evaluation_runs import EvaluationRunService
from histopilot.application.model_cards import model_card, render_markdown
from histopilot.application.shift import KIND, ShiftService
from histopilot.schemas.predictors import EvaluationRunSelection, SaveEvaluationRun
from histopilot.shift import analyze_plan
from histopilot.storage.io import read_json_bounded, write_json_atomic
from histopilot.storage.project_lock import StorageError
from histopilot.workers.compute_job import verify_plan_inputs


def save_run(service, predictor, cohort, name):
    selected = EvaluationRunSelection(predictorId=predictor["id"], cohortId=cohort["id"], name=name)
    preview = service.preview(selected)
    assert preview["canSave"], preview
    return service.save(
        SaveEvaluationRun(
            **selected.model_dump(), previewHash=preview["previewHash"], operationId=name
        )
    )


@pytest.fixture
def study(tmp_path, monkeypatch, task_center):
    predictors, cohort = registry.__wrapped__(tmp_path)
    selection, *_ = candidate(predictors)
    predictor, _ = freeze(predictors, selection)
    runs = EvaluationRunService(predictors.store, predictors.filesystem)
    monkeypatch.setattr("histopilot.application.evaluation_runs.training_runtime", runtime)
    run = save_run(runs, predictor, cohort, "External evaluation")
    shift = ShiftService(
        predictors.store, predictors.filesystem, runtime=runtime, task_center=task_center.client
    )
    shift.jobs = ComputeJobService(
        predictors.store, runtime=runtime, task_center=task_center.client
    )
    return shift, runs, run, predictor, cohort


def run_job(service, identity, center):
    """What the Task Center's compute worker does with the frozen plan, then its receipts."""
    folder = service.jobs.folder(identity)
    plan = read_json_bounded(folder / "plan.json")
    verify_plan_inputs(plan)
    result = analyze_plan(plan, folder)
    verify_plan_inputs(plan)
    complete(service, identity, result, center)
    return plan


def test_a_run_previews_starts_and_reads_its_dataset_shift(study, task_center):
    service, runs, run, predictor, cohort = study
    status = service.status(run["id"])
    assert status["exists"] is False and status["canRun"] is True and status["report"] is None
    preview = service.preview(run["id"])
    assert preview["canRun"] and preview["status"] == "not_started"
    assert preview["development"]["slides"] >= 2 and preview["cohort"]["slides"] >= 2
    state = service.start(run["id"], preview["previewHash"], "shift-1")
    assert state["status"] == "queued"
    # Accepted again, the same operation queues nothing new.
    assert service.start(run["id"], preview["previewHash"], "shift-1")["status"] == "queued"
    [task] = [
        task for task in compute_tasks(task_center) if task["labels"]["computeKind"] == "shift"
    ]
    assert task["request"]["lane"] == "cpu" and task["title"].startswith("Dataset shift · ")
    identity = status["analysisId"]
    record = service.store.get_configuration(identity)
    assert record["manifest"]["kind"] == KIND
    plan = run_job(service, identity, task_center)
    # The job reads features, never a label.
    assert '"label"' not in json.dumps(plan)
    development = plan["data"]["development"]["featureFiles"]
    assert set(development).isdisjoint(plan["data"]["cohort"]["featureFiles"])
    done = service.status(run["id"])
    assert done["execution"]["status"] == "completed"
    summary = done["report"]["summary"]
    assert summary["developmentSlides"] == len(development)
    assert summary["cohortSlides"] == len(plan["data"]["cohort"]["featureFiles"])
    # A second run of the same predictor on the same cohort shares the analysis.
    again = save_run(runs, predictor, cohort, "Second evaluation")
    assert service.status(again["id"])["analysisId"] == identity
    assert service.status(again["id"])["report"] == done["report"]


def test_a_stale_preview_and_changed_inputs_or_outputs_are_refused(study, task_center):
    service, _runs, run, *_ = study
    preview = service.preview(run["id"])
    with pytest.raises(StorageError) as error:
        service.start(run["id"], "0" * 64, "stale")
    assert error.value.code == "PREVIEW_STALE"
    service.start(run["id"], preview["previewHash"], "shift")
    identity = preview["analysisId"]
    folder = service.jobs.folder(identity)
    plan = read_json_bounded(folder / "plan.json")
    changed = json.loads(json.dumps(plan))
    slide = next(iter(changed["data"]["cohort"]["featureFiles"]))
    del changed["data"]["cohort"]["featureFiles"][slide]
    with pytest.raises(ValueError, match="differ from its frozen record"):
        verify_plan_inputs(changed)
    run_job(service, identity, task_center)
    report = folder / "shift.json"
    write_json_atomic(report, {**read_json_bounded(report), "summary": {}})
    with pytest.raises(StorageError) as error:
        service.status(run["id"])
    assert error.value.code == "SHIFT_RESULT_CHANGED"


def test_performance_by_shift_needs_a_finished_analysis(study):
    service, _runs, run, *_ = study
    with pytest.raises(StorageError) as error:
        service.performance(run["id"])
    assert error.value.code == "SHIFT_NOT_COMPLETED"


def test_the_model_card_names_its_evidence_and_limits(study, task_center):
    service, _runs, run, predictor, _cohort = study
    preview = service.preview(run["id"])
    service.start(run["id"], preview["previewHash"], "shift")
    run_job(service, preview["analysisId"], task_center)
    card = model_card(service.store, service.filesystem, predictor["id"])
    assert card["predictor"]["id"] == predictor["id"]
    assert card["predictor"]["contentHash"] == predictor["contentHash"]
    assert card["task"]["classes"] == predictor["manifest"]["target"]["classes"]
    assert [row["runId"] for row in card["evaluations"]] == [run["id"]]
    assert card["evaluations"][0]["shift"]["cohortSlides"] >= 2
    assert card["intendedUse"].startswith("Research use only")
    assert any("training seed" in item for item in card["limitations"])
    text = render_markdown(card)
    for heading in (
        "# Model card",
        "## Model",
        "## Development",
        "## Evaluations",
        "## Limitations",
    ):
        assert heading in text
    assert "Dataset shift:" in text and predictor["id"] in text


def test_the_card_and_shift_routes_answer_through_their_router(study):
    from fastapi import FastAPI

    from histopilot.api.shift import shift_router

    service, _runs, run, predictor, _cohort = study

    class Projects:
        def scientific_store(self, identity):
            assert identity == "project"
            return service.store

    app = FastAPI()
    app.include_router(shift_router(Projects(), service.filesystem))
    with TestClient(app) as client:
        base = "/api/v1/projects/project"
        card = client.get(f"{base}/predictors/{predictor['id']}/card?format=markdown")
        assert card.status_code == 200, card.text
        assert card.headers["content-type"].startswith("text/markdown")
        assert "attachment" in card.headers["content-disposition"]
        assert "# Model card" in card.text
        as_json = client.get(f"{base}/predictors/{predictor['id']}/card")
        assert as_json.json()["predictor"]["id"] == predictor["id"]
        shift = client.get(f"{base}/evaluation-runs/{run['id']}/shift")
        assert shift.status_code == 200 and shift.json()["exists"] is False
        preview = client.post(f"{base}/evaluation-runs/{run['id']}/shift/preview")
        assert preview.status_code == 200 and preview.json()["canRun"] is True
