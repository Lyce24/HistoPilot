"""The shipped walkthrough stays reproducible, internally consistent and isolated."""

import json
import math
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from histopilot.api import create_app
from histopilot.application.blca_demo import DEMO_ID, generate_demo, load_demo
from histopilot.config import Settings

API = "/api/v1"


def records(workspace):
    return {record["id"]: record for record in workspace["demoPipeline"]["records"]}


def step(record, identity):
    return next(item for item in record["steps"] if item["id"] == identity)


def prediction_rows(record):
    return step(record, "predictions")["table"]["rows"]


def test_packaged_demo_is_reproducible_and_has_all_stages():
    generated = generate_demo()
    assert generated == load_demo() == generate_demo()
    assert generated["project"]["id"] == DEMO_ID
    assert generated["demoPipeline"]["readOnly"] is True
    assert generated["demoPipeline"]["synthetic"] is True
    assert generated["executionEnabled"] is False
    assert Counter(item["module"] for item in generated["demoPipeline"]["records"]) == {
        "dataset": 1,
        "cohort": 1,
        "features": 1,
        "experiments": 2,
        "test-data": 1,
        "evaluation": 1,
        "clinical-utility": 1,
        "interpretation": 1,
    }
    assert len(records(generated)) == 9
    for record in records(generated).values():
        assert len({item["id"] for item in record["steps"]}) == len(record["steps"])
        assert len(record["steps"]) >= 2
    result = subprocess.run(
        [sys.executable, "-m", "histopilot.application.blca_demo", "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "reproducible" in result.stdout


def test_synthetic_slide_membership_and_folds_match_aggregate_design():
    workspace = load_demo()
    saved = records(workspace)
    rows = step(saved["blca-dataset"], "mapping")["table"]["rows"]
    assert len(rows) == 138
    assert len({row[0] for row in rows}) == 138
    assert len({row[1] for row in rows}) == 138
    assert Counter((row[2], row[3], row[4]) for row in rows) == {
        ("1", "low", "development"): 33,
        ("3", "high", "development"): 29,
        ("2", "low", "test"): 54,
        ("2", "high", "test"): 22,
    }
    development = {row[0] for row in rows if row[4] == "development"}
    test = {row[0] for row in rows if row[4] == "test"}
    assert not development & test
    membership = step(saved["blca-protocol"], "splits")["table"]["rows"]
    assert {row[0] for row in membership} == development
    assert len(membership) == len(development)
    assert {row[2] for row in membership} == {1, 2, 3, 4, 5}
    counts = Counter(row[2] for row in membership)
    assert max(counts.values()) - min(counts.values()) <= 2
    for identity in ("blca-baseline-v2", "blca-baseline-v3"):
        training = step(saved[identity], "runs")["runs"]
        for split_plan in training["batch"]["manifest"]["splitPlans"]:
            assert sum(split_plan["partitions"].values()) == 62
            assert split_plan["partitions"]["assessment"] == counts[split_plan["fold"] + 1]
    assert {row[0] for row in prediction_rows(saved["blca-evaluation"])} == test
    assert workspace["dataset"]["patientCount"] == 0
    assert workspace["dataset"]["fallbackSlideCount"] == 138
    assert workspace["patients"] == []
    assert all(slide["patientId"] == slide["specimenId"] == "" for slide in workspace["slides"])


def test_evaluation_roc_confusion_and_metrics_derive_from_published_synthetic_predictions():
    saved = records(load_demo())
    evaluation = saved["blca-evaluation"]
    rows = prediction_rows(evaluation)
    assert all(0 < row[2] < 1 for row in rows)
    assert all(row[3] == ("high" if row[2] >= 0.5 else "low") for row in rows)
    confusion = [
        [
            sum(row[1] == actual and row[3] == predicted for row in rows)
            for predicted in ("low", "high")
        ]
        for actual in ("low", "high")
    ]
    summary = step(evaluation, "metrics")
    assert [row[1:] for row in summary["table"]["rows"]] == confusion
    (tn, fp), (fn, tp) = confusion
    assert fp > 0 and fn > 0
    facts = {item["label"]: item["value"] for item in summary["facts"]}
    assert float(facts["Accuracy"]) == pytest.approx((tp + tn) / 76, abs=0.0005)
    roc = summary["chart"]["series"][0]["points"]
    assert roc[0] == {"x": 0, "y": 0}
    assert roc[-1] == {"x": 1, "y": 1}
    assert all(a["x"] <= b["x"] and a["y"] <= b["y"] for a, b in zip(roc, roc[1:]))
    auc = sum((b["x"] - a["x"]) * (a["y"] + b["y"]) / 2 for a, b in zip(roc, roc[1:]))
    assert float(facts["AUROC"]) == pytest.approx(auc, abs=0.0005)
    assert 0.85 < auc < 0.99
    # Clinical calibration and decision curves use these same exact rows.
    clinical = saved["blca-clinical-utility"]
    calibration = step(clinical, "calibration")
    assert sum(row[1] for row in calibration["table"]["rows"]) == len(rows)
    brier = sum((row[2] - int(row[1] == "high")) ** 2 for row in rows) / len(rows)
    assert float(calibration["facts"][0]["value"]) == pytest.approx(brier, abs=0.0005)
    thresholds = step(clinical, "thresholds")
    curve = thresholds["chart"]["series"][0]["points"]
    for point, row in zip(curve, thresholds["table"]["rows"]):
        threshold, table_tn, table_fp, table_fn, table_tp, benefit, _ = row
        expected_tp = sum(item[1] == "high" and item[2] >= threshold for item in rows)
        expected_fp = sum(item[1] == "low" and item[2] >= threshold for item in rows)
        assert (table_tp, table_fp) == (expected_tp, expected_fp)
        assert table_tn + table_fp + table_fn + table_tp == 76
        expected_benefit = expected_tp / 76 - expected_fp / 76 * threshold / (1 - threshold)
        assert benefit == pytest.approx(expected_benefit, abs=0.0000005)
        assert point == {"x": threshold, "y": expected_benefit}


def test_training_is_non_executable_and_resource_histories_are_coherent():
    saved = records(load_demo())
    for version in (2, 3):
        record = saved[f"blca-baseline-v{version}"]
        run_step = step(record, "runs")
        assert run_step["table"]["title"] == "Predictor creation"
        training = run_step["runs"]
        batch, execution = training["batch"], training["execution"]
        assert batch["manifest"]["executionImplemented"] is False
        assert batch["manifest"]["resolvedInputs"]["canPlan"] is False
        assert execution["batchId"] == batch["id"] == training["resources"]["batchId"]
        run_ids = {run["id"] for run in execution["runs"]}
        assert run_ids == set(training["histories"])
        assert execution["runCounts"]["completed"] == len(run_ids) == 5
        for run in execution["runs"]:
            history = training["histories"][run["id"]]
            assert history["synthetic"] is True and history["executable"] is False
            best = min(history["rows"], key=lambda row: row["validation"]["loss"])
            assert history["rows"][-1]["epoch"] - best["epoch"] == 8
            assert best["validation"] == run["metrics"]["validation"]["selected"]
            assert history["rows"][-1]["epoch"] == run["progress"]["epoch"] <= 40
            assert history["rows"][-1]["validation"] == run["progress"]["validation"]
            assert run["metrics"]["assessment"]["unit"] == "slide"
            assert run["metrics"]["assessment"]["patient"]["available"] is False
            assert all(row["checkpointUnit"] == "slide" for row in history["rows"])
            assert run["checkpointPath"].startswith("demo://blca/")
        samples = training["resources"]["rows"]
        timestamps = [datetime.fromisoformat(sample["at"]) for sample in samples]
        assert all((b - a).total_seconds() == 15 for a, b in zip(timestamps, timestamps[1:]))
        assert execution["telemetry"]["latest"] == samples[-1]
        for sample in samples:
            assert 0 <= sample["host"]["cpuUtilizationPercent"] <= 100
            assert 0 <= sample["host"]["availableRamGb"] <= sample["host"]["totalRamGb"]
            for gpu in sample["gpus"]:
                assert gpu["usedMemoryGb"] + gpu["freeMemoryGb"] == pytest.approx(
                    gpu["totalMemoryGb"]
                )
                assert 0 <= gpu["utilizationPercent"] <= 100
            assert all(item["runId"] in run_ids for item in sample["runs"])
        oof = step(record, "results")
        auc = float(next(item["value"] for item in oof["facts"] if item["label"] == "AUROC"))
        assert 0.94 < auc <= 1


def test_demo_has_no_filesystem_references_or_scientific_artifact_payloads():
    workspace = load_demo()
    assert workspace["project"]["storagePath"] == ""
    assert workspace["sources"] == workspace["project"]["sources"] == []
    serialized = json.dumps(workspace, allow_nan=False)
    assert not re.search(r"/(?:home|Users|mnt|tmp|var)/|[A-Za-z]:\\|TCGA-[A-Za-z0-9-]+", serialized)
    assert not re.search(r"data:image|base64|-----BEGIN.*PRIVATE KEY", serialized)
    assert "No patient identifiers" in serialized

    def inspect(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key.lower().endswith(("path", "uri")) and child:
                    assert isinstance(child, str) and child.startswith("demo://"), (key, child)
                inspect(child)
        elif isinstance(value, list):
            for child in value:
                inspect(child)
        elif isinstance(value, float):
            assert math.isfinite(value)

    inspect(workspace)


def test_blca_demo_api_is_authenticated_portable_and_cannot_create_real_state(tmp_path):
    root = tmp_path / "allowed"
    root.mkdir()
    settings = Settings(workspace=tmp_path / "workspace", data_roots=(root,))
    with TestClient(create_app(settings), base_url="http://127.0.0.1:8787") as client:
        prefix = f"{API}/projects/{DEMO_ID}"
        assert client.get(f"{prefix}/workspace").status_code == 401
        client.headers["X-HistoPilot-Token"] = client.get(f"{API}/session").json()["token"]
        response = client.get(f"{prefix}/workspace")
        assert response.status_code == 200
        assert response.json() == load_demo()
        assert str(tmp_path) not in response.text
        demos = [
            item
            for item in client.get(f"{API}/projects").json()["projects"]
            if item["mode"] == "synthetic-demo"
        ]
        assert [item["id"] for item in demos] == [DEMO_ID]
        assert client.get(f"{API}/projects/synthetic-v1/workspace").status_code == 200
        operations = [
            ("GET", "/storage", None),
            ("GET", "/datasets", None),
            ("POST", "/drafts", {"kind": "import", "name": "Unwanted state", "payload": {}}),
            ("PATCH", "", {"config": {"folds": 5}}),
            ("POST", "/sources", {"path": str(root), "role": "data"}),
            (
                "POST",
                "/mil-experiments/batches/blca-demo-batch-v2/launch",
                {"operationId": "demo-launch"},
            ),
        ]
        for method, path, body in operations:
            result = client.request(method, f"{prefix}{path}", json=body)
            assert result.status_code == 409, (method, path, result.text)
        assert client.get(f"{prefix}/workspace").json() == response.json()
        assert list(root.iterdir()) == []
        assert not list(settings.workspace.rglob("histopilot-project.json"))
        assert not list(settings.workspace.rglob("histopilot-state.sqlite"))
