"""Training charts read only bounded, safe, correctly attributed epoch evidence."""

import copy
import json
import os
from types import SimpleNamespace

import pytest

from histopilot.application.training_history import training_history
from histopilot.storage.project_lock import StorageError


@pytest.fixture
def history_store(tmp_path):
    batch = {
        "manifest": {
            "kind": "mil-batch",
            "runs": [{"id": "run-one", "candidateId": "candidate-one"}],
            "configurations": [{"id": "candidate-one", "recipe": {"maxEpochs": 10000}}],
        }
    }
    store = SimpleNamespace(folder=tmp_path, get_configuration=lambda identity: batch)
    folder = tmp_path / "training" / "batch-one" / "runs" / "run-one"
    folder.mkdir(parents=True)
    return store, folder / "history.json", batch


def row(epoch=0):
    return {
        "epoch": epoch,
        "step": epoch * 10,
        "trainingLoss": 0.0,
        "learningRate": 0.0001,
        "validation": {"loss": 0.4, "auroc": None, "accuracy": 0.0},
        "checkpointUnit": "patient",
    }


def read(store):
    return training_history(store, "batch-one", "run-one")


def test_history_uses_one_based_completed_epochs_and_retains_zero_and_unavailable(history_store):
    store, path, _batch = history_store
    content = json.dumps([row(), row(1)]).encode()
    path.write_bytes(content)
    result = read(store)
    assert [item["epoch"] for item in result["rows"]] == [1, 2]
    assert result["rows"][0]["trainingLoss"] == 0
    assert result["rows"][0]["validation"]["accuracy"] == 0
    assert result["rows"][0]["validation"]["auroc"] is None
    assert result["rows"][0]["checkpointUnit"] == "patient"
    assert result["totalRows"] == 2
    assert not result["truncated"]
    assert path.read_bytes() == content


def test_missing_history_is_pending_without_creating_files(history_store):
    store, path, _batch = history_store
    assert read(store) == {"runId": "run-one", "rows": [], "totalRows": 0, "truncated": False}
    assert not path.exists()


@pytest.mark.parametrize(
    "content",
    [
        "null",
        "{}",
        "[",
        json.dumps([row(1)]),
        json.dumps([row(), row()]),
        json.dumps([{**row(), "epoch": True}]),
        json.dumps([{**row(), "trainingLoss": True}]),
        json.dumps([{**row(), "trainingLoss": "0.5"}]),
        json.dumps([{**row(), "validation": None}]),
        json.dumps([{**row(), "validation": {"loss": float("nan")}}]),
        json.dumps([{**row(), "checkpointUnit": "unknown"}]),
        json.dumps([row(), {**row(1), "checkpointUnit": "slide"}]),
        json.dumps([{**row(), "trainingLoss": float("inf")}]),
        '[{"epoch":0,"trainingLoss":1e999}]',
        "[" * 1100 + "0" + "]" * 1100,
    ],
)
def test_invalid_optional_history_is_a_warning_not_an_execution_failure(history_store, content):
    store, path, _batch = history_store
    path.write_text(content)
    result = read(store)
    assert result["rows"] == []
    assert "invalid or cannot be read safely" in result["warning"]


def test_history_length_cannot_exceed_frozen_epoch_budget(history_store):
    store, path, batch = history_store
    batch["manifest"]["configurations"][0]["recipe"]["maxEpochs"] = 1
    path.write_text(json.dumps([row(), row(1)]))
    assert "warning" in read(store)


def test_large_histories_return_recent_contiguous_epochs_with_explicit_coverage(history_store):
    store, path, _batch = history_store
    path.write_text(json.dumps([row(index) for index in range(2100)]))
    result = read(store)
    assert result["totalRows"] == 2100
    assert result["truncated"]
    assert len(result["rows"]) == 2000
    assert result["rows"][0]["epoch"] == 101
    assert result["rows"][-1]["epoch"] == 2100


def test_history_file_size_is_bounded(history_store, monkeypatch):
    store, path, _batch = history_store
    monkeypatch.setattr("histopilot.application.training_history.MAX_HISTORY_BYTES", 20)
    path.write_text(json.dumps([row()]))
    assert "warning" in read(store)


@pytest.mark.parametrize("alias", ["symlink", "hardlink", "parent", "fifo"])
def test_history_refuses_nonregular_files_and_aliases(history_store, tmp_path, alias):
    store, path, _batch = history_store
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps([row()]))
    if alias == "symlink":
        path.symlink_to(outside)
    elif alias == "hardlink":
        os.link(outside, path)
    elif alias == "parent":
        path.parent.rmdir()
        path.parent.symlink_to(tmp_path)
        (tmp_path / "history.json").write_text(json.dumps([row()]))
    else:
        os.mkfifo(path)
    assert "warning" in read(store)
    assert json.loads(outside.read_text()) == [row()]


def test_history_requires_the_requested_run_to_belong_to_the_frozen_batch(history_store):
    store, _path, batch = history_store
    before = copy.deepcopy(batch)
    with pytest.raises(StorageError) as error:
        training_history(store, "batch-one", "../outside")
    assert error.value.code == "TRAINING_RUN_NOT_FOUND"
    assert batch == before


def test_corrupt_frozen_ids_do_not_turn_history_into_a_file_reader(history_store):
    store, _path, batch = history_store
    batch["manifest"]["runs"][0]["id"] = "../outside"
    with pytest.raises(StorageError) as error:
        training_history(store, "batch-one", "../outside")
    assert error.value.code == "TRAINING_RUN_INVALID"


def test_history_cannot_read_a_different_configuration_kind(history_store):
    store, _path, batch = history_store
    batch["manifest"]["kind"] = "protocol"
    with pytest.raises(StorageError) as error:
        read(store)
    assert error.value.code == "INVALID_BATCH"


def test_missing_frozen_candidate_is_a_structured_error(history_store):
    store, _path, batch = history_store
    batch["manifest"]["configurations"] = []
    with pytest.raises(StorageError) as error:
        read(store)
    assert error.value.code == "TRAINING_RUN_INVALID"


def test_history_api_is_authenticated_read_only_and_returns_saved_epochs(
    history_store, monkeypatch, tmp_path
):
    from fastapi.testclient import TestClient

    from histopilot.api import create_app
    from histopilot.config import Settings

    store, path, _batch = history_store
    path.write_text(json.dumps([row()]))
    monkeypatch.setattr(
        "histopilot.application.project_workspace.ProjectWorkspace.scientific_store",
        lambda self, identity: store,
    )
    app = create_app(Settings(workspace=tmp_path / "workspace", data_roots=(tmp_path,)))
    endpoint = "/api/v1/projects/project/mil-experiments/batches/batch-one/runs/run-one/history"
    with TestClient(app, base_url="http://127.0.0.1:8787") as client:
        assert client.get(endpoint).status_code == 401
        token = client.get("/api/v1/session").json()["token"]
        client.headers["X-HistoPilot-Token"] = token
        response = client.get(endpoint)
        assert response.status_code == 200
        assert response.json()["rows"][0]["epoch"] == 1
        assert client.post(endpoint).status_code == 405
