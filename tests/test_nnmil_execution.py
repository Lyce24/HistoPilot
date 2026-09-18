"""Real feature manifests resolve identically at preview, freeze and worker launch."""

from copy import deepcopy

import pytest
from test_training_execution import execution

from histopilot.application.development import DevelopmentService
from histopilot.schemas.development import DevelopmentBatchSpec
from histopilot.storage.project_lock import StorageError
from histopilot.workers.train_batch import _run_plan

__all__ = ["execution"]


def freeze_nnmil(service, batch):
    values = deepcopy(batch["manifest"]["spec"])
    values["recipe"].update(model="nnmil", attentionDim=2, bagSizeMode="training_median")
    values["batchName"] = "nnMIL fitting bags"
    spec = DevelopmentBatchSpec.model_validate(values)
    development = DevelopmentService(service.store, service.filesystem)
    preview = development.preview(spec)
    assert preview["canFreeze"], preview["findings"]
    result = development.freeze(spec, preview["previewHash"], "nnmil-batch", {"tag": "nnMIL"})
    return result, preview


def test_preview_freeze_and_launch_share_the_same_fold_fingerprint(execution):
    service, batch, executor, _ = execution
    frozen, preview = freeze_nnmil(service, batch)
    plan, _guard = service._prepare(frozen)
    assert plan["nnmilPlanning"] == preview["nnmilPlanning"]
    for run in plan["runs"]:
        worker = _run_plan(plan, run, None)
        summary = next(item for item in plan["nnmilPlanning"]
                       if item["candidateId"] == run["candidateId"]
                       and item["splitPlanId"] == run["splitPlanId"])
        assert worker["nnmilPlanning"] == {key: value for key, value in summary.items()
                                          if key not in {"candidateId", "splitPlanId"}}
        assert worker["effectiveRecipe"]["bagSize"] == summary["bagSize"]
        assert worker["recipe"] == frozen["manifest"]["configurations"][0]["recipe"]
    assert not executor.launches


def test_stale_frozen_bag_preview_blocks_launch(execution):
    service, batch, executor, _ = execution
    frozen, _ = freeze_nnmil(service, batch)
    changed = deepcopy(frozen)
    changed["manifest"]["nnmilPlanning"][0]["bagSize"] += 1
    with pytest.raises(StorageError, match="frozen MIL bag preview") as error:
        service._prepare(changed)
    assert error.value.code == "TRAINING_INPUTS_STALE"
    assert not executor.launches
