"""Experimental MIL controls survive completed-fold publication and refit review."""

import json
import runpy
from pathlib import Path

import pytest

from histopilot.application.evaluation_runs import EvaluationRunService
from histopilot.application.feature_bundles import _hash
from histopilot.schemas.development import TrainingRecipe
from histopilot.schemas.evaluations import InferenceSettings
from histopilot.schemas.predictors import EvaluationRunSelection, SaveEvaluationRun
from histopilot.schemas.training_controls import resolve_stopping
from histopilot.workers.packing_process import write_json
from histopilot.workers.train_batch import _run_plan

support = runpy.run_path(str(Path(__file__).with_name("test_predictor_registry.py")))
registry = support["registry"]


def candidate(service, monkeypatch, **recipe):
    # Reuse the complete frozen-protocol/receipt fixture with a different recipe,
    # preserving its candidate ID and batch/configuration hashes from creation.
    with monkeypatch.context() as patch:
        patch.setitem(
            support["candidate"].__globals__,
            "TrainingRecipe",
            lambda **kwargs: TrainingRecipe(**{**kwargs, **recipe}),
        )
        return support["candidate"](service, refit_ready=True)


def save_results(folder, state):
    for run in state["runs"]:
        write_json(folder / "runs" / run["id"] / "result.json", run["result"])
    write_json(folder / "state.json", state)


def fallback_candidate(service, monkeypatch):
    selection, folder, state = candidate(
        service,
        monkeypatch,
        checkpointMetric="validation_auroc",
        maxEpochs=4,
        minValidationPositives=100,
        fixedEpochBudget=2,
        patientAggregation="mean_logits",
        ensembleAggregation="mean_logit",
    )
    plan = json.loads((folder / "plan.json").read_text())
    recipe = plan["configurations"][0]["recipe"]
    for run in state["runs"]:
        effective, decision = resolve_stopping(
            recipe, plan["target"], plan["memberships"][run["splitPlanId"]]
        )
        original = Path(run["result"]["bestCheckpointPath"])
        final = original.with_name("last.ckpt")
        final.write_bytes(b"Synthetic final epoch checkpoint distinct from validation optimum.")
        run["result"].update(
            checkpointMetric=effective["checkpointMetric"],
            stoppingDecision=decision,
            bestCheckpointPath=str(final),
            lastCheckpointPath=str(final),
            epochsCompleted=2,
            bestEpoch=2,
        )
    save_results(folder, state)
    return selection, folder, state


@pytest.mark.parametrize("model", ["mean_pool", "max_pool"])
def test_pooling_models_publish_completed_fold_checkpoints(registry, monkeypatch, model):
    service, _ = registry
    selection, _, _ = candidate(service, monkeypatch, model=model)
    predictor, _ = support["freeze"](service, selection)
    assert predictor["manifest"]["recipe"]["model"] == model
    assert len(predictor["manifest"]["checkpoints"]) == 2


@pytest.mark.parametrize("strategy", ["cohort_balanced", "cohort_label_balanced"])
def test_cohort_enriched_worker_memberships_publish_and_refit(registry, monkeypatch, strategy):
    service, _ = registry
    selection, folder, state = candidate(
        service, monkeypatch, samplingStrategy=strategy, cohortColumn="site"
    )
    plan = json.loads((folder / "plan.json").read_text())
    plan["data"]["cohortValues"] = {
        "site": {identity: "TCGA" for identity in plan["data"]["featureFiles"]}
    }
    for run in state["runs"]:
        write_json(folder / "runs" / run["id"] / "plan.json", _run_plan(plan, run, None))
        run["result"]["bestEpoch"] = 2
    state["planHash"] = _hash(plan)
    write_json(folder / "plan.json", plan)
    save_results(folder, state)
    predictor, _ = support["freeze"](service, selection)
    assert predictor["manifest"]["recipe"]["samplingStrategy"] == strategy
    refit = service.preview(selection.model_copy(update={"method": "refit"}))
    assert refit["canFreeze"], refit
    assert all(
        row["cohort"] == "TCGA" for row in refit["manifest"]["planTemplate"]["data"]["memberships"]
    )


def test_publication_pins_patient_and_ensemble_logit_aggregation(registry, monkeypatch):
    service, _ = registry
    selection, _, _ = candidate(
        service, monkeypatch, patientAggregation="mean_logits", ensembleAggregation="mean_logit"
    )
    predictor, _ = support["freeze"](service, selection)
    assert predictor["manifest"]["aggregation"] == "mean_logit"
    assert predictor["manifest"]["patientAggregation"] == "mean_logits"


def test_evaluation_inherits_logit_scoring_and_pins_the_resolved_plan(registry, monkeypatch):
    service, cohort = registry
    choice, _, _ = candidate(
        service, monkeypatch, patientAggregation="mean_logits", ensembleAggregation="mean_logit"
    )
    predictor, _ = support["freeze"](service, choice)
    evaluations = EvaluationRunService(service.store, service.filesystem)
    settings = InferenceSettings.model_validate(
        {
            **cohort["manifest"]["spec"]["inference"],
            "patientAggregation": "predictor",
            "device": "cpu",
        }
    )
    selection = EvaluationRunSelection(
        predictorId=predictor["id"],
        cohortId=cohort["id"],
        name="Inherited scoring",
        inference=settings,
    )
    preview = evaluations.preview(selection)
    assert preview["canSave"], preview
    assert preview["manifest"]["inference"]["patientAggregation"] == "mean_logits"
    saved = evaluations.save(
        SaveEvaluationRun(
            **selection.model_dump(),
            previewHash=preview["previewHash"],
            operationId="inherit-scoring",
        )
    )
    monkeypatch.setattr(
        "histopilot.application.evaluation_runs.training_runtime", lambda: {"cudaAvailable": False}
    )
    plan = evaluations._execution_plan(saved["id"])
    assert plan["aggregation"] == "mean_logit"
    assert plan["inference"]["patientAggregation"] == "mean_logits"


def test_fallback_publishes_final_checkpoint_and_uses_its_epoch_for_refit(registry, monkeypatch):
    service, _ = registry
    selection, _, state = fallback_candidate(service, monkeypatch)
    predictor, _ = support["freeze"](service, selection)
    assert {row["path"] for row in predictor["manifest"]["checkpoints"]} == {
        run["result"]["lastCheckpointPath"] for run in state["runs"]
    }
    assert predictor["manifest"]["recipe"]["checkpointMetric"] == "validation_auroc"
    preview = service.preview(selection.model_copy(update={"method": "refit"}))
    assert preview["canFreeze"], preview
    assert preview["manifest"]["epochBudget"]["epochs"] == 2
    assert "minValidationPositives" not in preview["manifest"]["recipe"]
    assert "fixedEpochBudget" not in preview["manifest"]["recipe"]


@pytest.mark.parametrize("tamper", ["decision", "selection", "epoch", "completed", "missing"])
def test_fallback_promotion_rejects_tampered_stopping_evidence(registry, monkeypatch, tamper):
    service, _ = registry
    selection, folder, state = fallback_candidate(service, monkeypatch)
    result = state["runs"][0]["result"]
    if tamper == "decision":
        result["stoppingDecision"]["positivePatients"] += 1
    elif tamper == "selection":
        result["bestCheckpointPath"] = str(
            Path(result["bestCheckpointPath"]).with_name("best.ckpt")
        )
    elif tamper == "epoch":
        result["bestEpoch"] = 1
    elif tamper == "completed":
        result["epochsCompleted"] = 3
    else:
        result.pop("stoppingDecision")
    # Mirror changed evidence into the scheduler state to ensure the stopping
    # contract itself catches the problem, rather than a stale duplicate receipt.
    save_results(folder, state)
    preview = service.preview(selection)
    assert not preview["canFreeze"]
    assert preview["findings"][0]["code"] == "PREDICTOR_PROVENANCE_CHANGED"


def test_plateau_refit_freezes_constant_schedule_adjustment(registry, monkeypatch):
    service, _ = registry
    selection, folder, state = candidate(service, monkeypatch, lrScheduler="plateau")
    for run in state["runs"]:
        run["result"]["bestEpoch"] = 2
    save_results(folder, state)
    preview = service.preview(selection.model_copy(update={"method": "refit"}))
    assert preview["canFreeze"], preview
    manifest = preview["manifest"]
    assert manifest["sourceRecipe"]["lrScheduler"] == "plateau"
    assert manifest["recipe"]["lrScheduler"] == "none"
    assert "constant learning rate" in manifest["recipeAdjustments"]["lrScheduler"]
    assert manifest["planTemplate"]["recipe"] == manifest["recipe"]


def test_external_evaluation_inherits_frozen_threshold_and_analysis_policy(registry, monkeypatch):
    service, cohort = registry
    choice, _, _ = candidate(service, monkeypatch, decisionThreshold=0.7, evalBagSize=100,
                             analysis={"bootstrapResamples": 500, "bootstrapSeed": 11, "oneSlideSeed": 19})
    predictor, _ = support["freeze"](service, choice)
    evaluations = EvaluationRunService(service.store, service.filesystem)
    selection = EvaluationRunSelection(
        predictorId=predictor["id"], cohortId=cohort["id"], name="Frozen threshold",
        inference=InferenceSettings(patientAggregation="predictor", decisionThreshold="predictor", device="cpu"),
    )
    preview = evaluations.preview(selection)
    assert preview["canSave"], preview
    assert preview["manifest"]["inference"]["decisionThreshold"] == 0.7
    assert preview["manifest"]["inference"]["patientAggregation"] == "mean"
    assert preview["manifest"]["analysis"] == predictor["manifest"]["recipe"]["analysis"]
    saved = evaluations.save(SaveEvaluationRun(**selection.model_dump(),
        previewHash=preview["previewHash"], operationId="freeze-threshold"))
    monkeypatch.setattr("histopilot.application.evaluation_runs.training_runtime", lambda: {"cudaAvailable": False})
    plan = evaluations._execution_plan(saved["id"])
    assert plan["inference"]["decisionThreshold"] == 0.7
    assert plan["analysis"]["bootstrapResamples"] == 500
    assert plan["bagPolicy"] == {"evalBagSize": 100, "trainingSeed": choice.trainingSeed}
    changed = selection.model_copy(update={"inference": InferenceSettings(decisionThreshold=0.5)})
    rejected = evaluations.preview(changed)
    assert not rejected["canSave"]
    assert rejected["findings"][0]["code"] == "EVALUATION_THRESHOLD_MISMATCH"
