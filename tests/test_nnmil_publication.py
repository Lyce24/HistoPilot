"""nnMIL fold evidence survives promotion, refit planning, and external policy freezing."""

import json
from pathlib import Path

import pytest
from test_mil_experimental_publication import candidate, registry, save_results, support

from histopilot.application.evaluation_runs import EvaluationRunService
from histopilot.schemas.predictors import EvaluationRunSelection
from histopilot.workers.train_batch import _run_plan

__all__ = ["registry"]


def completed_nnmil(service, monkeypatch, selection="best_validation"):
    choice, folder, state = candidate(
        service, monkeypatch, model="nnmil", attentionDim=2, bagSizeMode="training_median",
        nnmilCheckpointSelection=selection, nnmilWindowStrideDivisor=2,
        nnmilWindowSeed=17, nnmilWindowAggregation="mean_probabilities",
    )
    plan = json.loads((folder / "plan.json").read_text())
    for run in state["runs"]:
        run_plan = _run_plan(plan, run, None)
        result = run["result"]
        result.update(effectiveRecipe=run_plan["effectiveRecipe"],
                      nnmilPlanning=run_plan["nnmilPlanning"], bestEpoch=1,
                      selectedEpoch=2 if selection == "latest" else 1,
                      checkpointSelection=selection)
        if selection == "latest":
            latest = Path(result["bestCheckpointPath"]).with_name("last.ckpt")
            latest.write_bytes(b"A distinct completed final checkpoint.")
            result.update(bestCheckpointPath=str(latest), lastCheckpointPath=str(latest))
    save_results(folder, state)
    return choice, folder, state


@pytest.mark.parametrize("selection", ["best_validation", "latest"])
def test_nnmil_promotes_selected_weights_and_resolved_fold_settings(registry, monkeypatch, selection):
    service, _ = registry
    choice, _, _ = completed_nnmil(service, monkeypatch, selection)
    predictor, _ = support["freeze"](service, choice)
    manifest = predictor["manifest"]
    assert manifest["recipe"]["model"] == "nnmil"
    # This fixture has no submitted construction policy; the checkpoint policy
    # must not create metadata reserved for submitted candidate selection.
    assert "candidateNumber" not in manifest
    assert manifest["recipe"]["nnmilWindowAggregation"] == "mean_probabilities"
    for checkpoint in manifest["checkpoints"]:
        assert checkpoint["selectedEpoch"] == (2 if selection == "latest" else 1)
        assert checkpoint["bestEpoch"] == 1
        assert checkpoint["checkpointSelection"] == selection
        assert checkpoint["effectiveRecipe"]["bagSize"] == checkpoint["nnmilPlanning"]["bagSize"]


@pytest.mark.parametrize("field", ["effectiveRecipe", "nnmilPlanning", "selectedEpoch"])
def test_changed_resolution_or_checkpoint_epoch_blocks_promotion(registry, monkeypatch, field):
    service, _ = registry
    choice, folder, state = completed_nnmil(service, monkeypatch, "latest")
    receipt = state["runs"][0]["result"]
    if field == "selectedEpoch":
        receipt[field] = 1
    else:
        receipt[field]["bagSize"] += 1
    save_results(folder, state)
    preview = service.preview(choice)
    assert not preview["canFreeze"]
    assert preview["findings"][0]["code"] == "PREDICTOR_PROVENANCE_CHANGED"


@pytest.mark.parametrize("selection,budget", [("best_validation", 1), ("latest", 2)])
def test_refit_uses_selected_epoch_and_recomputes_training_population(registry, monkeypatch, selection, budget):
    service, _ = registry
    choice, _, _ = completed_nnmil(service, monkeypatch, selection)
    preview = service.preview(choice.model_copy(update={"method": "refit"}))
    assert preview["canFreeze"], preview
    manifest = preview["manifest"]
    assert manifest["epochBudget"]["epochs"] == budget
    assert manifest["nnmilPlanning"]["trainingSlideCount"] == manifest["trainingSlideCount"]
    assert manifest["effectiveRecipe"]["maxEpochs"] == budget
    assert manifest["planTemplate"]["effectiveRecipe"] == manifest["effectiveRecipe"]
    assert manifest["planTemplate"]["nnmilPlanning"] == manifest["nnmilPlanning"]
    assert {row["partition"] for row in manifest["planTemplate"]["data"]["memberships"]} == {"train"}


def test_external_evaluation_freezes_nnmil_window_policy_and_all_patches(registry, monkeypatch):
    service, cohort = registry
    choice, _, _ = completed_nnmil(service, monkeypatch)
    predictor, _ = support["freeze"](service, choice)
    evaluations = EvaluationRunService(service.store, service.filesystem)
    preview = evaluations.preview(EvaluationRunSelection(
        predictorId=predictor["id"], cohortId=cohort["id"], name="nnMIL external",
        inference={**cohort["manifest"]["spec"]["inference"],
                   "patientAggregation": "predictor", "device": "cpu"},
    ))
    assert preview["canSave"], preview
    manifest = preview["manifest"]
    assert manifest["bagPolicy"]["evalBagSize"] is None
    assert manifest["predictor"]["contentHash"] == predictor["contentHash"]
