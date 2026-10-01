"""One predictor from every seed group of a configuration, built from verified fold checkpoints."""

import copy
import json
import runpy
from pathlib import Path

import numpy as np
import pytest
from support.predictors import managed as managed
from support.predictors import registry as registry
from support.predictors import two_seeds

from histopilot.application.evaluation_runs import EvaluationRunService
from histopilot.schemas.predictors import (
    EvaluationRunSelection,
    FreezeSeedEnsemble,
    SeedEnsembleSelection,
)
from histopilot.storage.project_lock import StorageError


def _selection(source, name="Seed ensemble"):
    return SeedEnsembleSelection(
        experimentId=source.experimentId,
        batchId=source.batchId,
        candidateId=source.candidateId,
        name=name,
    )


def _freeze(predictors, selection, operation="seed-ensemble"):
    preview = predictors.preview_seed_ensemble(selection)
    assert preview["canFreeze"], preview["findings"]
    return predictors.freeze_seed_ensemble(
        FreezeSeedEnsemble(
            **selection.model_dump(), previewHash=preview["previewHash"], operationId=operation
        )
    ), preview


def test_seed_ensemble_pools_each_seed_groups_verified_checkpoints(registry):
    predictors, _cohort = registry
    sources, _folder = two_seeds(predictors)
    selection = _selection(sources[0])
    [choice] = [
        row
        for row in predictors.seed_ensemble_choices(sources[0].experimentId)["items"]
        if row["batchId"] == selection.batchId
    ]
    assert choice["eligible"], choice["reason"]
    assert (choice["trainingSeeds"], choice["seedGroups"], choice["members"]) == ([11, 22], 2, 4)
    record, preview = _freeze(predictors, selection)
    manifest = record["manifest"]
    assert manifest["method"] == "seed_ensemble"
    assert manifest["trainingSeeds"] == [11, 22] and "trainingSeed" not in manifest
    assert [group["trainingSeed"] for group in manifest["seedGroups"]] == [11, 22]
    # Every member is the checkpoint its own seed group's fold ensemble would use.
    for source in sources:
        group = predictors.preview(source.model_copy(update={"method": "ensemble"}))
        assert group["canFreeze"], group["findings"]
        pooled = [
            {key: value for key, value in row.items() if key not in {"trainingSeed", "splitSeed"}}
            for row in manifest["checkpoints"]
            if row["trainingSeed"] == source.trainingSeed
        ]
        assert pooled == group["manifest"]["checkpoints"]
        assert manifest["aggregation"] == group["manifest"]["aggregation"]
    predictors.verify_checkpoints(record)
    # A replay returns the same record; another build of the same configuration is refused.
    assert (
        predictors.freeze_seed_ensemble(
            FreezeSeedEnsemble(
                **selection.model_dump(),
                previewHash=preview["previewHash"],
                operationId="seed-ensemble",
            )
        )["id"]
        == record["id"]
    )
    again = predictors.preview_seed_ensemble(selection)
    assert [row["code"] for row in again["findings"]] == ["EXPERIMENT_ALREADY_FROZEN"]
    with pytest.raises(StorageError) as error:
        predictors.freeze_seed_ensemble(
            FreezeSeedEnsemble(
                **selection.model_dump(), previewHash=preview["previewHash"], operationId="again"
            )
        )
    assert error.value.code == "EXPERIMENT_ALREADY_FROZEN"
    [after] = [
        row
        for row in predictors.seed_ensemble_choices(sources[0].experimentId)["items"]
        if row["batchId"] == selection.batchId
    ]
    assert after["existingPredictorId"] == record["id"] and not after["eligible"]


def test_a_single_seed_group_has_no_seed_ensemble(registry):
    from support.predictors import candidate

    predictors, _cohort = registry
    source, *_ = candidate(predictors)
    preview = predictors.preview_seed_ensemble(_selection(source))
    assert not preview["canFreeze"]
    assert preview["findings"][0]["code"] == "SEED_ENSEMBLE_SINGLE_GROUP"


def test_seed_ensembles_build_after_predictor_work_closes(managed):
    service, identity, _jobs, selections = managed
    service.launch(identity, "start")
    service.advance(identity)
    service.cancel(identity, "finish-predictors")
    predictors = service.builds.predictors
    # Per-seed creation is closed with the experiment's predictor work ...
    closed = predictors.preview(selections[0].model_copy(update={"method": "ensemble"}))
    assert closed["findings"][0]["code"] == "EXPERIMENT_PREDICTORS_LOCKED"
    # ... but a seed ensemble trains nothing and reopens nothing, so it is still available.
    record, _preview = _freeze(predictors, _selection(selections[0]))
    assert record["manifest"]["method"] == "seed_ensemble"


def test_seed_ensembles_are_evaluated_like_any_predictor(registry, monkeypatch):
    from support.training import runtime

    predictors, bound = registry
    sources, _folder = two_seeds(predictors)
    record, _preview = _freeze(predictors, _selection(sources[0]))
    monkeypatch.setattr("histopilot.application.evaluation_runs.training_runtime", runtime)
    runs = EvaluationRunService(predictors.store, predictors.filesystem)
    # An independent cohort binds to whichever predictor evaluates it.
    spec = {key: bound["manifest"]["spec"][key] for key in ("datasetId", "target", "eligibility")}
    draft = predictors.store.create_draft(
        "experiment", "Independent", {"type": "evaluation-cohort", "spec": spec}
    )
    checked = runs.cohorts.preview(draft["id"], 1)
    assert checked["canFreeze"], checked["findings"]
    cohort = runs.cohorts.freeze(draft["id"], 1, checked["previewHash"], "independent-cohort")
    review = runs.preview(
        EvaluationRunSelection(predictorId=record["id"], cohortId=cohort["id"], name="Pooled")
    )
    assert review["canSave"], review["findings"]
    assert review["manifest"]["bagPolicy"]["trainingSeed"] == 11
    assert review["manifest"]["labelSource"] == "cohort"


def test_each_member_keeps_its_own_training_seed_for_evaluation_bags(tmp_path):
    """Real tiny checkpoints: a seed ensemble equals the mean of its per-seed models."""
    pytest.importorskip("torch")
    pytest.importorskip("lightning")
    from histopilot.training.inference import evaluate

    support = runpy.run_path(str(Path(__file__).with_name("test_inference_execution.py")))
    plan = support["_evaluation_plan"](tmp_path)
    [checkpoint] = plan["checkpoints"]
    # Subsampled evaluation bags make each member's seed visible in its predictions.
    single = {**plan, "method": "refit", "bagPolicy": {"evalBagSize": 2, "trainingSeed": 5}}
    first = evaluate(copy.deepcopy(single), tmp_path / "seed-5")
    second = evaluate(
        {**copy.deepcopy(single), "bagPolicy": {"evalBagSize": 2, "trainingSeed": 9}},
        tmp_path / "seed-9",
    )
    pooled = evaluate(
        {
            **copy.deepcopy(single),
            "method": "seed_ensemble",
            "checkpoints": [
                {**checkpoint, "trainingSeed": 5, "splitSeed": 1},
                {**checkpoint, "trainingSeed": 9, "splitSeed": 1},
            ],
        },
        tmp_path / "pooled",
    )
    assert pooled["checkpointCount"] == 2 and first["slideCount"] == second["slideCount"]

    def probabilities(folder):
        records = json.loads((tmp_path / folder / "predictions.json").read_text())["records"]
        return np.asarray([row["probabilities"] for row in records])

    five, nine = probabilities("seed-5"), probabilities("seed-9")
    assert not np.allclose(five, nine)
    assert probabilities("pooled") == pytest.approx((five + nine) / 2)
    with pytest.raises(ValueError, match="record its training seed"):
        evaluate(
            {**copy.deepcopy(single), "method": "seed_ensemble", "checkpoints": [checkpoint] * 2},
            tmp_path / "unseeded",
        )
