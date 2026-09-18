"""Coordinator recovery checks use temporary state and never create real jobs."""

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

spec = importlib.util.spec_from_file_location(
    "blca_project_runner", Path(__file__).resolve().parents[1] / "scripts/run_blca_project.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def study(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({
        "schemaVersion": 1, "workspace": str(tmp_path / "workspace"),
        "projectPath": str(tmp_path / "unopened-project"), "projectId": "fixture-project",
        "protocolId": "fixture-protocol", "bundleId": "fixture-bundle",
        "packId": "pack-" + "a" * 64, "experimentId": "fixture-experiment",
        "batchPlanSaved": True, "checks": [],
    }))
    value = runner.Study(path)
    helper = ModuleType("blca_final_analysis")
    helper.prepare = Mock(return_value={})
    helper.finalize = Mock(side_effect=AssertionError("Unexpected final-analysis call"))
    monkeypatch.setitem(sys.modules, "blca_final_analysis", helper)
    monkeypatch.setattr(runner.time, "sleep", Mock(side_effect=AssertionError("Unexpected polling")))
    experiment = {"id": "fixture-experiment", "revision": 2,
                  "submission": {"status": "submitted"}}
    predictor = SimpleNamespace(status=Mock(return_value={"status": "running"}))
    value.experiments = SimpleNamespace(
        get=Mock(side_effect=lambda _: copy.deepcopy(experiment)),
        submit=Mock(return_value=copy.deepcopy(experiment)),
        training=SimpleNamespace(execution=Mock(return_value={"status": "running", "runs": []})),
        _predictors=Mock(return_value=predictor),
    )
    value.store = SimpleNamespace(list_configurations=Mock(return_value=[{
        "id": "fixture-batch", "manifest": {"spec": {"experimentId": "fixture-experiment"}},
    }]))
    monkeypatch.setattr(runner, "PredictorService", Mock(side_effect=AssertionError("Unexpected publication access")))
    return value, helper, predictor


@pytest.mark.parametrize("submission", [{"status": "attention", "error": {"code": "LAUNCH_FAILED"}}, {}])
def test_returned_submission_rejection_is_not_recorded_as_success(study, submission):
    value, _, _ = study
    value.experiments.submit.return_value = {"id": "fixture-experiment", "submission": submission}
    with pytest.raises(RuntimeError, match="submission needs recovery"):
        value.train()
    saved = json.loads(value.path.read_text())
    assert not saved.get("submitted")
    assert value.experiments.submit.call_count == 1
    value.experiments.training.execution.assert_not_called()
    evidence = json.loads((value.path.parent / "evidence/experiment-submission.json").read_text())
    assert evidence["submission"] == submission


@pytest.mark.parametrize("missing", ["training", "predictor"])
def test_missing_worker_receipt_stops_and_keeps_submission_evidence(study, missing):
    value, _, predictor = study
    value.save(submitted=True)
    if missing == "training":
        value.experiments.training.execution.return_value = None
    else:
        predictor.status.return_value = None
    with pytest.raises(RuntimeError, match="no execution receipt"):
        value.train()
    value.experiments.submit.assert_not_called()
    runner.time.sleep.assert_not_called()
    saved = json.loads(value.path.read_text())
    assert saved["submitted"] is True
    assert "predictorIds" not in saved
    evidence = json.loads((value.path.parent / "evidence/experiment-missing-execution.json").read_text())
    assert evidence["id"] == "fixture-experiment"


def test_predictor_attention_state_stops_instead_of_polling_forever(study):
    value, _, predictor = study
    value.save(submitted=True)
    value.experiments.training.execution.return_value = {"status": "completed", "runs": []}
    failure = {"status": "attention", "error": {"code": "REFIT_FAILED", "message": "fixture failure"},
               "items": [{"method": "refit", "status": "failed"}]}
    predictor.status.return_value = failure
    with pytest.raises(RuntimeError, match="requires recovery"):
        value.train()
    runner.time.sleep.assert_not_called()
    assert json.loads((value.path.parent / "evidence/predictor-status.json").read_text()) == failure
    assert "predictorIds" not in json.loads(value.path.read_text())


def test_helper_checkpoint_ids_survive_main_exception_handler(study, monkeypatch):
    value, helper, _ = study
    acknowledged = {"clinicalAnalysisIds": {"ensemble": "saved-analysis"},
                    "attentionBatchId": "saved-attention-job",
                    "interpretationIds": {"selected-slide": "saved-attention-job"}}

    def failed_after_checkpoint(path):
        saved = json.loads(path.read_text())
        saved.update(acknowledged)
        runner.write_json(path, saved)
        raise RuntimeError("fixture failure after acknowledged publication")

    helper.finalize.side_effect = failed_after_checkpoint
    monkeypatch.setattr(runner.Study, "services", Mock())
    monkeypatch.setattr(sys, "argv", ["run_blca_project.py", "--state", str(value.path), "--phase", "finalize"])
    with pytest.raises(RuntimeError, match="after acknowledged publication"):
        runner.main()
    saved = json.loads(value.path.read_text())
    assert all(saved[key] == expected for key, expected in acknowledged.items())
    assert saved["lastError"]["type"] == "RuntimeError"
    assert "after acknowledged publication" in saved["lastError"]["message"]
    assert saved["stage"] != "complete"
    helper.finalize.assert_called_once()
    runner.time.sleep.assert_not_called()


def test_finalize_waits_for_verified_completion_before_completing_study(study, monkeypatch):
    value, helper, _ = study
    prior_error = {"type": "RuntimeError", "message": "Recoverable earlier attempt"}
    value.save(lastError=prior_error)
    statuses = iter(["attention_prepared", "attention_running", "complete"])

    def advance(path):
        saved = json.loads(path.read_text())
        saved.update(finalAnalysisStatus=next(statuses), attentionBatchId="one-saved-job")
        runner.write_json(path, saved)
        return saved

    helper.finalize.side_effect = advance
    sleep = Mock()
    monkeypatch.setattr(runner.time, "sleep", sleep)
    value.finalize()
    assert helper.finalize.call_count == 3
    assert [call.args for call in sleep.call_args_list] == [(15,), (15,)]
    saved = json.loads(value.path.read_text())
    assert saved["stage"] == saved["finalAnalysisStatus"] == "complete"
    assert saved["attentionBatchId"] == "one-saved-job"
    assert saved["lastError"] is None
    assert saved["errorHistory"] == [prior_error]
    evidence = json.loads((value.path.parent / "evidence/finalize-return.json").read_text())
    assert evidence["finalAnalysisStatus"] == "complete"


@pytest.mark.parametrize("status", ["attention_needs_review", "waiting_for_evaluations"])
def test_finalize_stops_on_recovery_state_and_preserves_current_job(study, status):
    value, helper, _ = study

    def needs_review(path):
        saved = json.loads(path.read_text())
        saved.update(finalAnalysisStatus=status, attentionBatchId="recoverable-job")
        runner.write_json(path, saved)
        return saved

    helper.finalize.side_effect = needs_review
    with pytest.raises(RuntimeError, match=f"requires recovery: {status}"):
        value.finalize()
    assert value.state["attentionBatchId"] == "recoverable-job"
    assert value.state["stage"] != "complete"
    helper.finalize.assert_called_once()
    runner.time.sleep.assert_not_called()
