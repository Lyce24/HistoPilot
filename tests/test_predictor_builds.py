"""Many configuration/seed groups build independently and retry without duplicates."""

import copy
import runpy
from pathlib import Path
from uuid import uuid4

import pytest

from histopilot.application.feature_bundles import _hash
from histopilot.application.predictor_builds import PredictorBuildService
from histopilot.schemas.predictors import (
    ApplyPredictorBuilds,
    PredictorBuildSelection,
    PredictorSourceSelection,
)
from histopilot.storage.project_lock import StorageError
from histopilot.workers.packing_process import write_json
from histopilot.workers.train_batch import _run_plan
from histopilot.workers.training_process import read_json

support = runpy.run_path(str(Path(__file__).with_name("test_refit_predictors.py")))
registry = support["registry"]


def two_seeds(service):
    source, original_folder, original_state = support["refit_candidate"](service, epochs=(2, 4))
    original = service.store.get_configuration(source.batchId)
    manifest = copy.deepcopy(original["manifest"])
    originals = manifest["runs"]
    manifest["runs"] = [
        {**row, "id": "run-" + _hash([row["id"], seed]), "trainingSeed": seed}
        for seed in (11, 22)
        for row in originals
    ]
    batch = service.store.publish_configuration(manifest=manifest, operation_id=uuid4().hex)
    plan = {
        **read_json(original_folder / "plan.json"),
        "batchId": batch["id"],
        "batchContentHash": batch["contentHash"],
        "runs": manifest["runs"],
    }
    folder = service.store.folder / "training" / batch["id"]
    states = []
    for index, run in enumerate(manifest["runs"]):
        run_folder = folder / "runs" / run["id"]
        run_folder.mkdir(parents=True)
        checkpoint = run_folder / "best.ckpt"
        checkpoint.write_bytes(b"seed checkpoint")
        result = {
            **original_state["runs"][index % 2]["result"],
            "runId": run["id"],
            "bestCheckpointPath": str(checkpoint),
        }
        write_json(run_folder / "result.json", result)
        write_json(run_folder / "plan.json", _run_plan(plan, run, None))
        states.append({**run, "status": "completed", "result": result})
    write_json(folder / "plan.json", plan)
    write_json(
        folder / "state.json",
        {"batchId": batch["id"], "status": "completed", "planHash": _hash(plan), "runs": states},
    )
    return [
        source.model_copy(update={"batchId": batch["id"], "trainingSeed": seed})
        for seed in (11, 22)
    ], folder


def request_for(selections, **changes):
    fields = set(PredictorSourceSelection.model_fields)
    return PredictorBuildSelection(
        selections=[item.model_dump(include=fields) for item in selections],
        method="both",
        **changes,
    )


def apply_request(service, request, operation="build-both"):
    preview = service.preview(request)
    assert preview["canBuild"], preview
    return ApplyPredictorBuilds(
        **request.model_dump(), previewHash=preview["previewHash"], operationId=operation
    ), preview


def test_both_builds_each_training_seed_and_reuses_exact_groups(registry):
    predictors, _ = registry
    selections, _ = two_seeds(predictors)
    service = PredictorBuildService(predictors.store, predictors.filesystem)
    request = request_for(selections)
    apply, preview = apply_request(service, request)
    assert preview["counts"] == {"total": 4, "create": 4, "reuse": 0, "blocked": 0}
    result = service.apply(apply)
    assert result["status"] == "completed"
    assert {row["status"] for row in result["items"]} == {"created"}
    assert len({row["recordId"] for row in result["items"]}) == 4
    assert {row["selection"]["trainingSeed"] for row in result["items"]} == {11, 22}
    assert len(predictors.list()["items"]) == 2
    choices = [
        row for row in predictors.choices()["items"] if row["batchId"] == selections[0].batchId
    ]
    assert len({row["existingRefitId"] for row in choices}) == 2
    assert len({row["existingPredictorIds"]["ensemble"] for row in choices}) == 2
    from types import SimpleNamespace

    from histopilot.application.model_experiments import ModelExperimentService

    experiments = ModelExperimentService(
        predictors.store,
        predictors.filesystem,
        training=SimpleNamespace(execution=lambda *args, **kwargs: None),
    )
    experiment = next(
        row
        for row in experiments.list(summary=True)["items"]
        if row["id"] == selections[0].experimentId
    )
    assert {row["trainingSeed"] for row in experiment["predictors"]} == {11, 22}
    assert experiment["predictorIds"]["ensemble"] in {row["id"] for row in experiment["predictors"]}
    assert service.apply(apply) == result == service.get(apply.operationId)
    fresh, reused = apply_request(service, request, "reuse-both")
    assert reused["counts"] == {"total": 4, "create": 0, "reuse": 4, "blocked": 0}
    assert {row["status"] for row in service.apply(fresh)["items"]} == {"reused"}


@pytest.mark.parametrize(
    "field", ["experimentId", "batchId", "candidateId", "trainingSeed", "splitSeed", "method"]
)
def test_predictor_uniqueness_uses_every_source_coordinate(registry, field):
    service, _ = registry
    selected, _, _ = support["refit_candidate"](service)
    selected = selected.model_copy(update={"method": "ensemble"})
    frozen, _ = support["support"]["freeze"](service, selected)
    assert service._existing(selected)["id"] == frozen["id"]
    values = selected.model_dump()
    values[field] = values[field] + 1 if isinstance(values[field], int) else "another-identity"
    assert service._existing(values) is None


def test_bulk_failure_retries_only_unfinished_items(registry, monkeypatch):
    predictors, _ = registry
    selections, _ = two_seeds(predictors)
    service = PredictorBuildService(predictors.store, predictors.filesystem)
    apply, _ = apply_request(service, request_for(selections))
    original = predictors.store.publish_configuration
    calls = 0

    def fail_once(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise StorageError("Temporary publication failure.", "TEST_FAILURE", 503)
        return original(**kwargs)

    monkeypatch.setattr(predictors.store, "publish_configuration", fail_once)
    partial = service.apply(apply)
    assert partial["status"] == "partial"
    assert sum(row["status"] == "created" for row in partial["items"]) == 3
    completed = service.apply(apply)
    assert completed["status"] == "completed"
    assert calls == 5
    for before, after in zip(partial["items"], completed["items"], strict=True):
        if before["status"] == "created":
            assert before == after


def test_crash_after_child_publication_is_recovered_without_republication(registry, monkeypatch):
    predictors, _ = registry
    selections, _ = two_seeds(predictors)
    service = PredictorBuildService(predictors.store, predictors.filesystem)
    apply, _ = apply_request(service, request_for(selections[:1]))
    original = write_json
    calls = 0

    def crash(path, value):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("Simulated connection/process failure after child commit")
        original(path, value)

    monkeypatch.setattr("histopilot.application.predictor_builds.write_json", crash)
    with pytest.raises(OSError):
        service.apply(apply)
    assert len(predictors.list()["items"]) == 1
    monkeypatch.setattr("histopilot.application.predictor_builds.write_json", original)
    result = service.apply(apply)
    assert result["status"] == "completed"
    assert len(predictors.list()["items"]) == 1
    assert len(predictors.store.list_configurations("predictor-refit")) == 1


def test_inactive_identity_blocks_new_build_and_completed_retry_does_not_restore(registry):
    predictors, _ = registry
    selections, _ = two_seeds(predictors)
    service = PredictorBuildService(predictors.store, predictors.filesystem)
    request = request_for(selections[:1])
    apply, _ = apply_request(service, request)
    result = service.apply(apply)
    key = f"configuration:{result['items'][0]['recordId']}"
    lifecycle = predictors.store.lifecycle
    lifecycle.apply(
        {key: "trashed"},
        operation_id="trash",
        request_hash=_hash(key),
        expected_revision=lifecycle.read()["revision"],
    )
    preview = service.preview(request)
    assert not preview["canBuild"]
    assert preview["items"][0]["findings"][0]["code"] == "PREDICTOR_RECORD_INACTIVE"
    assert service.apply(apply) == result
    assert lifecycle.read()["records"][key]["state"] == "trashed"


def test_incomplete_group_blocks_whole_review_and_stale_review_creates_nothing(registry):
    predictors, _ = registry
    selections, folder = two_seeds(predictors)
    service = PredictorBuildService(predictors.store, predictors.filesystem)
    request = request_for(selections)
    apply, _ = apply_request(service, request)
    state = read_json(folder / "state.json")
    state["runs"][-1]["status"] = "failed"
    write_json(folder / "state.json", state)
    assert not service.preview(request)["canBuild"]
    with pytest.raises(StorageError) as error:
        service.apply(apply)
    assert error.value.code == "PREVIEW_STALE"
    assert predictors.list()["items"] == []
    assert predictors.store.list_configurations("predictor-refit") == []


def test_saved_refit_reuse_verifies_its_original_source_checkpoint_hash(registry):
    predictors, _ = registry
    selections, folder = two_seeds(predictors)
    service = PredictorBuildService(predictors.store, predictors.filesystem)
    request = request_for(selections[:1])
    apply, _ = apply_request(service, request)
    service.apply(apply)
    state = read_json(folder / "state.json")
    Path(state["runs"][0]["result"]["bestCheckpointPath"]).write_bytes(b"changed source checkpoint")
    preview = service.preview(request)
    assert not preview["canBuild"]
    assert preview["items"][1]["findings"][0]["code"] == "REFIT_EVIDENCE_CHANGED"


def test_modified_durable_receipt_is_rejected_before_retry_or_history_read(registry):
    predictors, _ = registry
    selections, _ = two_seeds(predictors)
    service = PredictorBuildService(predictors.store, predictors.filesystem)
    request, _ = apply_request(service, request_for(selections[:1]))
    service.apply(request)
    path = service._folder(request.operationId) / "receipt.json"
    receipt = read_json(path)
    receipt["review"][0]["selection"]["trainingSeed"] = 999
    write_json(path, receipt)
    with pytest.raises(StorageError) as problem:
        service.apply(request)
    assert problem.value.code == "PREDICTOR_BUILD_RECEIPT_INVALID"
    with pytest.raises(StorageError):
        service.get(request.operationId)


def test_each_seed_can_publish_its_own_refit_and_bulk_reuses_published_models(registry):
    from histopilot.application.refits import RefitService
    from histopilot.schemas.predictors import LaunchRefit

    predictors, _ = registry
    selections, _ = two_seeds(predictors)
    service = PredictorBuildService(predictors.store, predictors.filesystem)
    request = request_for(selections)
    apply, _ = apply_request(service, request)
    built = service.apply(apply)
    jobs = support["FakeJobs"](predictors.store)
    refits = RefitService(predictors.store, predictors.filesystem, jobs)
    models = []
    for row in built["items"]:
        if row["method"] == "refit":
            refits.launch(row["recordId"], LaunchRefit(operationId="launch-" + row["key"]))
            jobs.complete(row["recordId"])
            models.append(refits.publish(row["recordId"], "publish-" + row["key"]))
    assert {model["manifest"]["trainingSeed"] for model in models} == {11, 22}
    assert len(predictors.list()["items"]) == 4
    preview = service.preview(request)
    assert preview["canBuild"], preview
    assert preview["counts"]["reuse"] == 4
    assert all(row["recordKind"] == "frozen-predictor" for row in preview["items"])
