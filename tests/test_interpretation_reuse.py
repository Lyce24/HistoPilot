"""Reuse verified active/completed scientific evidence independently of job reservations."""

import hashlib
from pathlib import Path

import pytest
from test_interpretation import study as study
from test_interpretation_gallery import gallery as gallery
from test_interpretation_gallery import visualize_request

from histopilot.workers.packing_process import write_json
from histopilot.workers.training_process import read_json


def mark_state(service, identity, status):
    folder = service.jobs.folder(identity)
    state = read_json(folder / "state.json")
    if status != "completed":
        write_json(folder / "state.json", {**state, "status": status, "result": None})
        return
    row = service.get(identity)["manifest"]["slides"][0]
    artifact = folder / "slide-0.json"
    write_json(
        artifact,
        {
            "slideId": row["slideId"],
            "patchCount": row["patchCount"],
            "probabilities": [0.5, 0.5],
            "patches": [
                {"index": index, "x": xy[0], "y": xy[1], "weight": 1 / 3, "percentile": 0.5}
                for index, xy in enumerate([(0, 0), (100, 0), (100, 100)])
            ],
        },
    )
    content = artifact.read_bytes()
    result = {
        "state": "succeeded",
        "runId": identity,
        "slides": [{"slideId": row["slideId"], "patchCount": row["patchCount"]}],
        "artifacts": {
            artifact.name: {
                "path": str(artifact),
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        },
    }
    write_json(folder / "result.json", result)
    write_json(folder / "state.json", {**state, "status": "completed", "result": result})


@pytest.mark.parametrize("status", ["queued", "running", "completed"])
def test_changed_resources_reuse_verified_existing_attention_without_new_job(gallery, status):
    service, source, executor, _ = gallery
    paths = [Path(source["slideFolder"]) / "001.png"]
    first = service.visualize(visualize_request(source, paths, operation="original-profile"))
    original = first["interpretations"][0]
    mark_state(service, original["id"], status)
    # This reservation would be too small to compute the bag, but reuse allocates no worker.
    request = visualize_request(
        source,
        paths,
        operation="changed-profile",
        resources={
            "gpuIds": [],
            "dataLoaderWorkers": 0,
            "cpuThreadsPerRun": 6,
            "ramGbPerRun": 0.25,
        },
    )
    reused = service.visualize(request)
    item = reused["items"][0]
    assert item["status"] == status and item["reused"], reused
    assert item["interpretationId"] == original["id"]
    assert item["interpretation"]["manifest"]["resources"] == original["manifest"]["resources"]
    assert service.visualize(request)["items"][0]["interpretationId"] == original["id"]
    assert len(executor.calls) == 1
    assert len(service.store.list_configurations("model-interpretation")) == 1


def test_failed_attention_preserves_requested_resource_change(gallery):
    service, source, executor, _ = gallery
    paths = [Path(source["slideFolder"]) / "001.png"]
    first = service.visualize(visualize_request(source, paths, operation="failed-original"))
    original = first["items"][0]["interpretationId"]
    mark_state(service, original, "failed")
    executor.sessions.clear()
    changed = service.visualize(
        visualize_request(
            source,
            paths,
            operation="new-resources",
            resources={
                "gpuIds": [],
                "dataLoaderWorkers": 0,
                "cpuThreadsPerRun": 6,
                "ramGbPerRun": 4,
            },
        )
    )
    assert changed["items"][0]["status"] == "queued", changed
    assert changed["items"][0]["interpretationId"] != original
    assert changed["interpretations"][0]["manifest"]["resources"]["ramGbPerRun"] == 4
    assert len(executor.calls) == 2


def test_completed_equivalent_is_preferred_to_failed_exact_resource_configuration(gallery):
    service, source, executor, _ = gallery
    paths = [Path(source["slideFolder"]) / "001.png"]
    first = service.visualize(visualize_request(source, paths, operation="first-profile"))
    old_id = first["items"][0]["interpretationId"]
    mark_state(service, old_id, "failed")
    executor.sessions.clear()
    second = service.visualize(
        visualize_request(
            source,
            paths,
            operation="second-profile",
            resources={
                "gpuIds": [],
                "dataLoaderWorkers": 0,
                "cpuThreadsPerRun": 6,
                "ramGbPerRun": 4,
            },
        )
    )
    completed_id = second["items"][0]["interpretationId"]
    mark_state(service, completed_id, "completed")
    result = service.visualize(visualize_request(source, paths, operation="first-profile-again"))
    assert result["items"][0]["status"] == "completed", result
    assert result["items"][0]["interpretationId"] == completed_id
    assert len(executor.calls) == 2


def test_source_representation_and_predictor_identity_are_not_reused_across(gallery):
    service, source, executor, artifact = gallery
    paths = [Path(source["slideFolder"]) / "001.png"]
    first = service.visualize(visualize_request(source, paths, operation="original-source"))
    original_id = first["items"][0]["interpretationId"]
    packed = service.visualize(
        visualize_request(
            {**source, "packArtifactId": artifact["id"]}, paths, operation="packed-source"
        )
    )
    assert packed["items"][0]["interpretationId"] != original_id
    predictor = service.predictors.get(source["predictorId"])
    different = service.store.publish_configuration(
        manifest={**predictor["manifest"], "name": "Other frozen predictor"},
        operation_id="different-predictor",
    )
    result = service.visualize(
        visualize_request(
            {**source, "predictorId": different["id"]}, paths, operation="different-source-model"
        )
    )
    assert result["items"][0]["interpretationId"] != original_id
    assert len(executor.calls) == 3


def test_changed_resource_reuse_still_rejects_corrupted_result_receipt(gallery):
    service, source, executor, _ = gallery
    paths = [Path(source["slideFolder"]) / "001.png"]
    first = service.visualize(visualize_request(source, paths, operation="original-completed"))
    identity = first["items"][0]["interpretationId"]
    mark_state(service, identity, "completed")
    write_json(service.jobs.folder(identity) / "result.json", {})
    result = service.visualize(
        visualize_request(
            source,
            paths,
            operation="different-resources",
            resources={
                "gpuIds": [],
                "dataLoaderWorkers": 0,
                "cpuThreadsPerRun": 6,
                "ramGbPerRun": 4,
            },
        )
    )
    assert result["items"][0]["status"] == "error"
    assert result["items"][0]["error"]["code"] == "COMPUTE_RESULT_CHANGED"
    assert len(executor.calls) == 1
