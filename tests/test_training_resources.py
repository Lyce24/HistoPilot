"""Resource charts read the selected batch's recorded, bounded, optional evidence."""

import copy
import json
import os
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from histopilot.application import training_resources as resources
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import ScientificStore


@pytest.fixture
def resource_store(tmp_path):
    batch = {"manifest": {"kind": "mil-batch", "runs": [{"id": "run-one"}]}}
    store = SimpleNamespace(folder=tmp_path, get_configuration=lambda identity: batch)
    folder = tmp_path / "training" / "batch-one"
    folder.mkdir(parents=True)
    return store, folder / "telemetry.jsonl", batch


def row(index=0):
    return {
        "at": (datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=index * 15)).isoformat(),
        "host": {
            "cpuCount": 16,
            "totalRamGb": 128.0,
            "availableRamGb": 100.0,
            "cpuUtilizationPercent": 0.0,
            "bootId": "boot-one",
            "kernel": "test",
        },
        "gpus": [
            {
                "index": 0,
                "name": "GPU one",
                "uuid": "gpu-one",
                "driverVersion": "560",
                "totalMemoryGb": 24.0,
                "usedMemoryGb": 1.5,
                "freeMemoryGb": None,
                "utilizationPercent": 0.0,
            }
        ],
        "runs": [{"runId": "run-one", "pid": 123, "rssGb": 2.0}],
    }


def save(path, rows):
    path.write_bytes(b"".join(json.dumps(item).encode() + b"\n" for item in rows))


def read(store):
    return resources.training_resources(store, "batch-one")


def test_missing_resource_history_remains_pending_without_creating_files(resource_store):
    store, path, _batch = resource_store
    assert read(store) == {"batchId": "batch-one", "rows": [], "totalRows": 0, "truncated": False}
    assert not path.exists()


def test_saved_resources_preserve_zero_unknown_and_original_timestamps(resource_store):
    store, path, _batch = resource_store
    old = row()
    del old["host"]["cpuUtilizationPercent"]
    missing = row(2)
    missing["host"]["cpuUtilizationPercent"] = None
    missing.update(gpus=[], gpuProbeError="Driver unavailable")
    expected = [old, row(1), missing]
    save(path, expected)
    before = path.read_bytes()
    assert read(store) == {
        "batchId": "batch-one",
        "rows": expected,
        "totalRows": 3,
        "truncated": False,
    }
    assert path.read_bytes() == before


def test_complete_valid_neighbors_survive_corruption_and_partial_last_write(resource_store):
    store, path, _batch = resource_store
    save(path, [row()])
    with path.open("ab") as stream:
        stream.write(b"not-json\n")
        stream.write(json.dumps(row(1)).encode() + b"\n")
        stream.write(json.dumps(row(2)).encode())
    result = read(store)
    assert result["rows"] == [row(), row(1)]
    assert result["totalRows"] == 2
    assert "Skipped 1 invalid" in result["warning"]
    assert "incomplete" in result["warning"]
    with path.open("ab") as stream:
        stream.write(b"\n")
    assert read(store)["rows"] == [row(), row(1), row(2)]


@pytest.mark.parametrize(
    "change",
    [
        {"at": "invalid"},
        {"at": "2026-01-01T00:00:00"},
        {"host": None},
        {"host": {**row()["host"], "cpuUtilizationPercent": float("nan")}},
        {"host": {**row()["host"], "cpuUtilizationPercent": float("inf")}},
        {"host": {**row()["host"], "cpuUtilizationPercent": 101}},
        {"host": {**row()["host"], "cpuUtilizationPercent": True}},
        {"host": {**row()["host"], "cpuCount": True}},
        {"host": {**row()["host"], "totalRamGb": None}},
        {"host": {**row()["host"], "availableRamGb": 129}},
        {"host": {**row()["host"], "kernel": None}},
        {"gpus": [{**row()["gpus"][0], "usedMemoryGb": -1}]},
        {"runs": [{"runId": "other-batch-run", "pid": 123, "rssGb": 2}]},
        {"runs": [{"runId": "run-one", "pid": True, "rssGb": 2}]},
        {"runs": [{"runId": "run-one", "pid": 123, "rssGb": None}]},
        {"gpuProbeError": {}},
    ],
)
def test_invalid_counters_are_skipped_without_hiding_other_samples(resource_store, change):
    store, path, _batch = resource_store
    save(path, [{**row(), **change}, row(1)])
    result = read(store)
    assert result["rows"] == [row(1)]
    assert result["totalRows"] == 1
    assert "Skipped 1" in result["warning"]


def test_display_limit_retains_recent_contiguous_observations(resource_store):
    store, path, _batch = resource_store
    save(path, [row(index) for index in range(380)])
    result = read(store)
    assert result["truncated"]
    assert result["totalRows"] == 380
    assert len(result["rows"]) == 360
    assert result["rows"][0] == row(20)
    assert result["rows"][-1] == row(379)


@pytest.mark.parametrize("at_line_boundary", [False, True])
def test_file_read_limit_skips_only_incomplete_start_and_labels_count(
    resource_store, monkeypatch, at_line_boundary
):
    store, path, _batch = resource_store
    encoded = json.dumps(row()).encode() + b"\n"
    path.write_bytes(encoded * 20)
    maximum = len(encoded) * 3 + (0 if at_line_boundary else 10)
    monkeypatch.setattr(resources, "MAX_RESOURCE_BYTES", maximum)
    result = read(store)
    assert result["truncated"] and result["totalRowsIsLowerBound"]
    assert result["totalRows"] == 3
    assert result["rows"] == [row()] * 3
    assert "sample count excludes older file data" in result["warning"]


@pytest.mark.parametrize("alias", ["symlink", "hardlink", "parent", "fifo"])
def test_resource_history_refuses_aliases_and_nonregular_files(resource_store, tmp_path, alias):
    store, path, _batch = resource_store
    outside = tmp_path / "outside.jsonl"
    save(outside, [row()])
    before = outside.read_bytes()
    if alias == "symlink":
        path.symlink_to(outside)
    elif alias == "hardlink":
        os.link(outside, path)
    elif alias == "parent":
        path.parent.rmdir()
        path.parent.symlink_to(tmp_path)
        save(tmp_path / "telemetry.jsonl", [row()])
    else:
        os.mkfifo(path)
    result = read(store)
    assert result["rows"] == []
    assert "cannot be read safely" in result["warning"]
    assert outside.read_bytes() == before


def test_wrong_configuration_kind_is_rejected(resource_store):
    store, _path, batch = resource_store
    batch["manifest"]["kind"] = "protocol"
    with pytest.raises(StorageError) as error:
        read(store)
    assert error.value.code == "INVALID_BATCH"


def test_saved_file_does_not_authorize_a_missing_frozen_batch(resource_store):
    store, path, _batch = resource_store
    save(path, [row()])

    def missing(_identity):
        raise StorageError("Unknown batch", "CONFIGURATION_NOT_FOUND", 404)

    store.get_configuration = missing
    with pytest.raises(StorageError) as error:
        read(store)
    assert error.value.code == "CONFIGURATION_NOT_FOUND"


@pytest.mark.parametrize("scope", ["batch", "project"])
def test_archived_resource_history_remains_readable_while_trash_requires_restore(tmp_path, scope):
    folder = tmp_path / "project"
    folder.mkdir()
    store = ScientificStore(folder, "project-resources")
    draft = store.create_draft("import", "Source", {})
    dataset = store.publish_dataset(
        draft["id"],
        expected_revision=1,
        manifest={"kind": "dataset", "name": "Source"},
        artifacts={"source.csv": b"slide_id,label\ns1,positive\n"},
        operation_id="source-dataset",
    )
    batch = store.publish_configuration(
        manifest={"kind": "mil-batch", "datasetId": dataset["id"], "runs": [{"id": "run-one"}]},
        operation_id="frozen-batch",
    )
    path = folder / "training" / batch["id"] / "telemetry.jsonl"
    path.parent.mkdir(parents=True)
    save(path, [row()])
    before = path.read_bytes()
    key = f"configuration:{batch['id']}" if scope == "batch" else "project:project-resources"
    lifecycle = store.lifecycle
    lifecycle.apply({key: "archived"}, "archive", "a" * 64, 0)
    archived = lifecycle.read()
    assert resources.training_resources(store, batch["id"])["rows"] == [row()]
    assert lifecycle.read() == archived
    lifecycle.apply({key: "trashed"}, "trash", "b" * 64, 1)
    with pytest.raises(StorageError) as error:
        resources.training_resources(store, batch["id"])
    assert error.value.code == "RECORD_TRASHED"
    lifecycle.apply({key: "active"}, "restore", "c" * 64, 2)
    assert resources.training_resources(store, batch["id"])["rows"] == [row()]
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "identity", ["..", "../batch-one", "batch-one/other", "other\\batch", "x\0"]
)
def test_untrusted_batch_ids_never_reach_storage(resource_store, identity):
    store, _path, _batch = resource_store
    store.get_configuration = lambda _identity: pytest.fail("Invalid identity reached storage")
    with pytest.raises(StorageError) as error:
        resources.training_resources(store, identity)
    assert error.value.code == "TRAINING_BATCH_INVALID"


def test_resource_api_is_authenticated_scoped_and_never_probes_hardware(
    resource_store, monkeypatch, tmp_path
):
    from fastapi.testclient import TestClient

    from histopilot.api import create_app
    from histopilot.config import Settings

    store, path, batch = resource_store
    save(path, [row()])
    before = copy.deepcopy(batch)

    def scoped_store(self, identity):
        if identity != "project-one":
            raise StorageError("Unknown project", "PROJECT_NOT_FOUND", 404)
        return store

    monkeypatch.setattr(
        "histopilot.application.project_workspace.ProjectWorkspace.scientific_store", scoped_store
    )
    for name in ("host_snapshot", "gpu_snapshot", "cpu_times"):
        monkeypatch.setattr(
            f"histopilot.workers.training_process.{name}",
            lambda: pytest.fail("Resource history must not probe hardware"),
        )
    app = create_app(Settings(workspace=tmp_path / "workspace", data_roots=(tmp_path,)))
    endpoint = "/api/v1/projects/project-one/mil-experiments/batches/batch-one/resources/history"
    with TestClient(app, base_url="http://127.0.0.1:8787") as client:
        assert client.get(endpoint).status_code == 401
        token = client.get("/api/v1/session").json()["token"]
        client.headers["X-HistoPilot-Token"] = token
        response = client.get(endpoint)
        assert response.status_code == 200
        assert response.json()["rows"] == [row()]
        assert client.get(endpoint.replace("project-one", "other-project")).status_code == 404
        assert client.post(endpoint).status_code == 405
    assert batch == before
