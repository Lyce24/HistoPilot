"""OOF downloads authorize frozen groups and reject inconsistent saved evidence."""

import csv
import hashlib
import io
from copy import deepcopy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_training_execution import execution, synthetic_results

from histopilot.api import create_app
from histopilot.application.feature_bundles import _hash
from histopilot.application.training_exports import training_oof_csv
from histopilot.config import Settings
from histopilot.storage.project_lock import StorageError
from histopilot.workers.packing_process import write_json
from histopilot.workers.train_batch import _run_plan
from histopilot.workers.training_process import read_json

__all__ = ["execution"]


@pytest.fixture
def exports(execution):
    service, frozen, _, _ = execution
    plan, state = synthetic_results(service, frozen)
    folder = service._folder(frozen["id"])
    write_json(folder / "plan.json", plan)
    records = []
    for run in state["runs"]:
        run_folder = folder / "runs" / run["id"]
        write_json(run_folder / "plan.json", _run_plan(plan, run, None))
        path = Path(run["result"]["predictions"]["assessment"])
        prediction = read_json(path)
        run["result"]["bestCheckpointPath"] = str(run_folder / "best.ckpt")
        prediction["checkpointPath"] = run["result"]["bestCheckpointPath"]
        write_json(path, prediction)
        write_json(run_folder / "result.json", run["result"])
        records.extend(prediction["records"])
    write_json(folder / "state.json", state)
    candidate = plan["configurations"][0]
    training_seed = plan["runs"][0]["trainingSeed"]
    split_seed = plan["splitPlans"][0]["seed"]
    key = hashlib.sha256(f"{candidate['id']}/{training_seed}/{split_seed}".encode()).hexdigest()[:24]
    document = {"batchId": frozen["id"], "candidateId": candidate["id"],
                "trainingSeed": training_seed, "splitSeed": split_seed,
                "protocolId": plan["protocolId"], "classOrder": plan["target"]["classes"],
                "records": records, "purpose": "development_assessment",
                "analysisInputHash": _hash({"records": records, "target": plan["target"],
                                            "recipe": candidate["recipe"], "code": plan["code"]})}
    path = folder / f"oof-{key}.json"
    write_json(path, document)
    arguments = (frozen["id"], candidate["id"], training_seed, split_seed)
    return service, arguments, folder, path


@pytest.mark.parametrize("unit", ["patient", "slide"])
def test_oof_csv_uses_frozen_labels_folds_and_class_order_without_torch(exports, unit):
    service, arguments, _, _ = exports
    result = training_oof_csv(service.store, *arguments, unit)
    rows = list(csv.DictReader(io.StringIO(result.decode())))
    assert len(rows) == 30
    assert {row["assessmentFold"] for row in rows} == {str(i) for i in range(5)}
    assert all(row["predictedLabel"] == row["label"] for row in rows)
    assert list(rows[0])[-2:] == ["probability:low", "probability:high"]
    assert ("slideIds" in rows[0]) == (unit == "patient")


@pytest.mark.parametrize("damage", ["missing_state", "unfinished", "duplicate_state",
                                    "receipt", "class_order", "probabilities", "label", "path",
                                    "run_recipe", "run_membership", "checkpoint", "hash", "source"])
def test_damaged_or_incomplete_oof_evidence_cannot_be_exported(exports, tmp_path, damage):
    service, arguments, folder, path = exports
    state = read_json(folder / "state.json")
    first = state["runs"][0]
    if damage == "missing_state":
        state["runs"].pop(0)
    elif damage == "unfinished":
        first["status"] = "failed"
    elif damage == "duplicate_state":
        state["runs"].append(deepcopy(first))
    elif damage == "receipt":
        write_json(folder / "runs" / first["id"] / "result.json", {"state": "failed"})
    elif damage == "path":
        outside = tmp_path / "outside.json"
        outside.write_bytes(Path(first["result"]["predictions"]["assessment"]).read_bytes())
        first["result"]["predictions"]["assessment"] = str(outside)
        write_json(folder / "runs" / first["id"] / "result.json", first["result"])
    elif damage in {"run_recipe", "run_membership"}:
        run_path = folder / "runs" / first["id"] / "plan.json"
        run_plan = read_json(run_path)
        if damage == "run_recipe":
            run_plan["recipe"]["patientAggregation"] = "mean_logits"
        else:
            run_plan["data"]["memberships"][0]["partition"] = "wrong"
        write_json(run_path, run_plan)
    elif damage == "checkpoint":
        prediction_path = Path(first["result"]["predictions"]["assessment"])
        prediction = read_json(prediction_path)
        prediction["checkpointPath"] = "another-model.ckpt"
        write_json(prediction_path, prediction)
    else:
        document = read_json(path)
        if damage == "class_order":
            document["classOrder"].reverse()
        elif damage == "probabilities":
            document["records"][0]["probabilities"] = [0.2, 0.8]
        elif damage == "label":
            document["records"][0]["labelIndex"] = 100
        elif damage == "hash":
            document["analysisInputHash"] = "0" * 64
        else:
            document["records"][0]["patientIdSource"] = "slide_fallback"
        write_json(path, document)
    write_json(folder / "state.json", state)
    with pytest.raises(StorageError) as error:
        training_oof_csv(service.store, *arguments, "patient")
    assert error.value.code in {"TRAINING_OOF_INVALID", "TRAINING_OOF_INCOMPLETE"}


def test_foreign_candidate_and_seed_are_not_downloadable(exports):
    service, arguments, _, _ = exports
    for candidate, train, split in (("foreign", arguments[2], arguments[3]),
                                    (arguments[1], 123, arguments[3]),
                                    (arguments[1], arguments[2], 123)):
        with pytest.raises(StorageError) as error:
            training_oof_csv(service.store, arguments[0], candidate, train, split, "slide")
        assert error.value.code == "TRAINING_OOF_NOT_FOUND"


def test_download_route_requires_session_and_valid_unit(exports, tmp_path, monkeypatch):
    service, arguments, _, _ = exports
    monkeypatch.setattr("histopilot.application.project_workspace.ProjectWorkspace.scientific_store",
                        lambda *_: service.store)
    app = create_app(Settings(workspace=tmp_path / "registry", data_roots=(tmp_path,)))
    batch, candidate, train, split = arguments
    route = f"/api/v1/projects/project/mil-experiments/batches/{batch}/oof/{candidate}/{train}/{split}"
    with TestClient(app, base_url="http://127.0.0.1:8787") as client:
        assert client.get(route + "/patient.csv").status_code == 401
        client.headers["X-HistoPilot-Token"] = client.get("/api/v1/session").json()["token"]
        response = client.get(route + "/patient.csv")
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("text/csv")
        assert "attachment;" in response.headers["content-disposition"]
        assert len(list(csv.DictReader(io.StringIO(response.text)))) == 30
        assert client.get(route + "/unknown.csv").status_code == 422
