"""The real browser/API story: import, map, freeze, attach, split and reopen."""

import csv
from dataclasses import replace

import h5py
import numpy as np
import pytest
from fastapi.testclient import TestClient

from histopilot.api import create_app
from histopilot.config import Settings


def connect(settings):
    return TestClient(create_app(settings), base_url="http://127.0.0.1:8787")


def auth(client):
    client.headers["X-HistoPilot-Token"] = client.get("/api/v1/session").json()["token"]


def post(client, path, payload, status=200):
    response = client.post(path, json=payload)
    assert response.status_code == status, response.text
    return response.json()


def fixture_sources(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    table = data / "bladder.csv"
    features = data / "features"
    features.mkdir()
    with table.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Slide_ID", "Patient_ID", "WHO 2022", "WHO 1973", "partition"])
        for patient in range(12):
            for slide in range(2):
                identity = f"{patient:03}.{slide}"
                writer.writerow(
                    [
                        identity,
                        f"P{patient:03}",
                        "low" if patient < 6 else "high",
                        "2" if patient % 6 >= 4 else "1",
                        "train" if slide == 0 else "test",
                    ]
                )
                with h5py.File(features / f"{identity}.h5", "w") as content:
                    content.create_dataset("features", data=np.ones((3, 4), dtype="float32"))
                    content.create_dataset("coords", data=np.ones((3, 2), dtype="int64"))
    return data, table, features


def test_import_feature_patient_split_freeze_and_new_registry_reopen(tmp_path):
    data, table, features = fixture_sources(tmp_path)
    settings = Settings(workspace=tmp_path / "registry-a", data_roots=(data,))
    with connect(settings) as client:
        auth(client)
        project = post(
            client,
            "/api/v1/projects",
            {"name": "Bladder", "storagePath": str(data / "experiment")},
            201,
        )
        base = f"/api/v1/projects/{project['id']}"
        inspected = post(client, base + "/imports/inspect", {"source": {"path": str(table)}})
        assert inspected["rowCount"] == 24
        import_spec = {
            "source": {"path": str(table)},
            "slideIdColumn": "Slide_ID",
            "patientIdColumn": "Patient_ID",
            "includeMissingSlides": True,
            "attributes": [
                {"key": key, "sourceColumn": key, "type": "categorical"}
                for key in ["WHO 2022", "WHO 1973", "partition"]
            ],
        }
        draft = post(
            client,
            base + "/drafts",
            {
                "kind": "import",
                "name": "Bladder import",
                "payload": {"type": "dataset-import", "spec": import_spec},
            },
            201,
        )
        import_url = base + f"/imports/{draft['id']}"
        preview = post(client, import_url + "/preview", {"expectedRevision": 1})
        assert preview["canFreeze"]
        freeze_intent = {
            "expectedRevision": 1,
            "previewHash": preview["previewHash"],
            "operationId": "freeze-dataset",
        }
        dataset = post(client, import_url + "/freeze", freeze_intent, 201)
        assert post(client, import_url + "/freeze", freeze_intent, 201) == dataset
        drafts_before = client.get(base + "/drafts").json()
        configurations_before = client.get(base + "/configurations").json()
        cohort = post(
            client,
            base + "/protocols/explore",
            {
                "datasetId": dataset["id"],
                "targetField": "WHO 2022",
                "rules": {"test": [{"field": "WHO 1973", "op": "eq", "value": "2"}]},
            },
        )
        assert cohort["valid"], cohort["findings"]
        assert cohort["cohort"]["totalSlides"] == 24
        assert cohort["cohort"]["patientCount"] == 12
        assert cohort["partitions"]["test"]["expanded"]["totalSlides"] == 8
        assert cohort["partitions"]["train"]["expanded"]["totalSlides"] == 16
        assert client.get(base + "/drafts").json() == drafts_before
        assert client.get(base + "/configurations").json() == configurations_before
        records = client.get(base + f"/datasets/{dataset['id']}/records?limit=5").json()
        assert records["total"] == 24 and len(records["records"]) == 5
        assert records["records"][0]["slideId"] == "000.0"
        chart = post(
            client, base + f"/datasets/{dataset['id']}/query", {"field": "WHO 2022", "limit": 2}
        )
        assert sum(row["count"] for row in chart["distribution"]["counts"]) == 24
        feature_spec = {"datasetId": dataset["id"], "path": str(features)}
        feature_preview = post(client, base + "/features/preview", feature_spec)
        feature = post(
            client,
            base + "/features/freeze",
            {
                **feature_spec,
                "previewHash": feature_preview["previewHash"],
                "operationId": "features",
            },
            201,
        )
        protocol_spec = {
            "datasetId": dataset["id"],
            "featureSetId": feature["id"],
            "target": {
                "field": "WHO 2022",
                "task": "binary_classification",
                "unit": "slide",
                "classes": ["low", "high"],
                "labels": {"low": "low", "high": "high"},
                "positiveClass": "high",
            },
            "split": {
                "mode": "kfold",
                "folds": 2,
                "seeds": [42, 19],
                "rules": {"test": [{"field": "WHO 1973", "op": "eq", "value": "2"}]},
            },
        }
        protocol_draft = post(
            client,
            base + "/drafts",
            {
                "kind": "experiment",
                "name": "Grade-2 holdout",
                "payload": {"type": "analysis-protocol", "spec": protocol_spec},
            },
            201,
        )
        protocol_url = base + f"/protocols/{protocol_draft['id']}"
        split = post(client, protocol_url + "/preview", {"expectedRevision": 1})
        assert split["canFreeze"], split["findings"]
        assert len(split["partitions"]) == 4
        assert all(
            item["test"]["patients"] == 4 and item["test"]["slides"] == 8
            for item in split["partitions"]
        )
        frozen_intent = {
            "expectedRevision": 1,
            "previewHash": split["previewHash"],
            "operationId": "protocol",
        }
        protocol = post(client, protocol_url + "/freeze", frozen_intent, 201)
        assert post(client, protocol_url + "/freeze", frozen_intent, 201) == protocol
        memberships = protocol["manifest"]["memberships"]
        assignments = {}
        for row in memberships:
            key = row["seed"], row["fold"], row["patientId"]
            assert assignments.setdefault(key, row["partition"]) == row["partition"]
        report = client.get(base + f"/protocols/{protocol['id']}/preflight").json()
        assert report["executionEnabled"] is False
        assert not report["executionReady"]
        with h5py.File(features / "000.0.h5", "a") as content:
            content.attrs["changed"] = "after freeze"
        report = client.get(base + f"/protocols/{protocol['id']}/preflight").json()
        assert not report["scientificReady"]
        assert any(item["code"] == "FEATURE_SOURCE_CHANGED" for item in report["findings"])
        blocked = post(client, base + "/jobs", {"protocolId": protocol["id"]}, 422)
        assert blocked["code"] == "PREFLIGHT_BLOCKED"
        assert client.get("/api/v1/jobs").json()["jobs"] == []
        # Imported slide-level splits that divide one patient must fail before publication.
        leaking = {
            **protocol_spec,
            "split": {
                "mode": "imported",
                "imported": {
                    "partitionField": "partition",
                    "partitionLabels": {"train": "train", "test": "test"},
                },
            },
        }
        invalid = post(
            client,
            base + "/drafts",
            {
                "kind": "experiment",
                "name": "Leaking legacy split",
                "payload": {"type": "analysis-protocol", "spec": leaking},
            },
            201,
        )
        invalid_url = base + f"/protocols/{invalid['id']}"
        rejected = post(client, invalid_url + "/preview", {"expectedRevision": 1})
        assert not rejected["canFreeze"]
        assert any(item["code"] == "IMPORTED_PATIENT_LEAKAGE" for item in rejected["findings"])
        response = client.post(
            invalid_url + "/freeze",
            json={
                "expectedRevision": 1,
                "previewHash": rejected["previewHash"],
                "operationId": "invalid",
            },
        )
        assert response.status_code in {409, 422}
    with connect(replace(settings, workspace=tmp_path / "registry-b")) as client:
        auth(client)
        reopened = post(client, "/api/v1/projects/open", {"path": project["storagePath"]})
        assert reopened["id"] == project["id"]
        assert client.get(base + f"/configurations/{protocol['id']}").json() == protocol
        assert client.get(base + f"/datasets/{dataset['id']}").json() == dataset
        workspace = client.get(base + "/workspace").json()
        assert workspace["dataset"]["slideCount"] == 24
        assert workspace["dataset"]["patientCount"] == 12


def cv_dataset(client, data):
    """A balanced generic metadata table, with enough groups for nested CV."""
    table = data / "metadata.csv"
    with table.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Slide_ID", "Patient_ID", "label", "site"])
        for patient in range(60):
            for slide in range(2):
                writer.writerow(
                    [
                        f"s{patient}-{slide}",
                        f"p{patient}",
                        str(patient % 2),
                        f"site-{patient // 20}",
                    ]
                )
    project = post(
        client,
        "/api/v1/projects",
        {"name": "Analysis", "storagePath": str(data / "experiment")},
        201,
    )
    base = f"/api/v1/projects/{project['id']}"
    draft = post(
        client,
        base + "/drafts",
        {
            "kind": "import",
            "name": "Source",
            "payload": {
                "type": "dataset-import",
                "spec": {
                    "source": {"path": str(table)},
                    "slideIdColumn": "Slide_ID",
                    "patientIdColumn": "Patient_ID",
                    "includeMissingSlides": True,
                    "attributes": [
                        {"key": key, "sourceColumn": key, "type": "text"}
                        for key in ("label", "site")
                    ],
                },
            },
        },
        201,
    )
    import_url = base + f"/imports/{draft['id']}"
    inspected = post(client, import_url + "/preview", {"expectedRevision": 1})
    assert inspected["canFreeze"]
    dataset = post(
        client,
        import_url + "/freeze",
        {"expectedRevision": 1, "previewHash": inspected["previewHash"], "operationId": "source"},
        201,
    )
    return project, base, dataset


@pytest.mark.parametrize("mode", ([], {}, 3, None))
def test_incomplete_strategy_metadata_returns_live_cohort_counts_without_server_error(
    tmp_path, mode
):
    data = tmp_path / "data"
    data.mkdir()
    settings = Settings(workspace=tmp_path / "registry", data_roots=(data,))
    with connect(settings) as client:
        auth(client)
        _project, base, dataset = cv_dataset(client, data)
        drafts_before = client.get(base + "/drafts").json()
        result = post(
            client,
            base + "/protocols/explore",
            {
                "datasetId": dataset["id"],
                "targetField": "label",
                "split": {"version": 2, "mode": mode},
            },
        )
        assert not result["valid"]
        assert result["cohort"]["totalSlides"] == 120
        assert result["cohort"]["patientCount"] == 60
        assert result["partitions"] is None
        assert any(finding["code"] == "INVALID_STRATEGY_CONFIG" for finding in result["findings"])
        assert client.get(base + "/drafts").json() == drafts_before


def test_nested_cv_api_freeze_and_reopen_preserves_inner_and_outer_memberships(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    settings = Settings(workspace=tmp_path / "registry-a", data_roots=(data,))
    with connect(settings) as client:
        auth(client)
        project, base, dataset = cv_dataset(client, data)
        spec = {
            "datasetId": dataset["id"],
            "target": {
                "field": "label",
                "task": "binary_classification",
                "unit": "patient",
                "classes": ["negative", "positive"],
                "labels": {"0": "negative", "1": "positive"},
                "positiveClass": "positive",
            },
            "split": {
                "version": 2,
                "mode": "nested_kfold",
                "outerFolds": 3,
                "innerFolds": 2,
                "seeds": [42],
                "validationFraction": 0.25,
            },
        }
        draft = post(
            client,
            base + "/drafts",
            {
                "kind": "experiment",
                "name": "Nested evaluation",
                "payload": {"type": "analysis-protocol", "spec": spec},
            },
            201,
        )
        protocol_url = base + f"/protocols/{draft['id']}"
        result = post(client, protocol_url + "/preview", {"expectedRevision": 1})
        assert result["canFreeze"], result["findings"]
        assert len(result["partitions"]) == 9
        assert {row["phase"] for row in result["memberships"]} == {"inner", "outer"}
        assert len({row["planId"] for row in result["memberships"]}) == 9
        intent = {
            "expectedRevision": 1,
            "previewHash": result["previewHash"],
            "operationId": "nested",
        }
        frozen = post(client, protocol_url + "/freeze", intent, 201)
        assert frozen["manifest"]["memberships"] == result["memberships"]
        assert frozen["manifest"]["spec"]["split"]["version"] == 2
        assert post(client, protocol_url + "/freeze", intent, 201) == frozen
    with connect(replace(settings, workspace=tmp_path / "registry-b")) as client:
        auth(client)
        reopened = post(client, "/api/v1/projects/open", {"path": project["storagePath"]})
        assert reopened["id"] == project["id"]
        assert client.get(base + f"/configurations/{frozen['id']}").json() == frozen
        assert post(client, protocol_url + "/freeze", intent, 201) == frozen
