"""Paired analysis uses immutable artifacts and exact patient memberships."""

import copy
import hashlib
import json
import runpy
from pathlib import Path

import pytest

from histopilot.schemas.predictors import CompareEvaluations
from histopilot.storage.project_lock import StorageError

support = runpy.run_path(str(Path(__file__).with_name("test_clinical.py")))
service = support["service"]


@pytest.fixture
def comparison(service, monkeypatch):
    clinical, selection, folder, execution = service
    current = clinical.evaluations
    left = current.get(selection.evaluationId)
    right = current.store.publish_configuration(
        manifest={**left["manifest"], "name": "Second fixed predictor"},
        operation_id="second-evaluation",
    )
    output = current.jobs.folder(right["id"])
    output.mkdir(parents=True)
    second_execution = copy.deepcopy(execution)
    for name, evidence in second_execution["result"]["artifacts"].items():
        path = output / name
        path.write_bytes((folder / name).read_bytes())
        evidence["path"] = str(path)
    states = {left["id"]: execution, right["id"]: second_execution}
    monkeypatch.setattr(current.jobs, "status", lambda identity, **kwargs: states[identity])
    request = CompareEvaluations(
        leftEvaluationId=left["id"],
        rightEvaluationId=right["id"],
        analysis={"bootstrapResamples": 200},
    )
    return current, request, output, second_execution


def test_completed_matched_evaluations_return_paired_intervals_with_artifact_hashes(comparison):
    current, request, *_ = comparison
    result = current.compare(request)
    assert result["statistics"]["available"]
    assert result["statistics"]["paired"]
    assert result["statistics"]["patientCount"] == 4
    assert result["predictionsSha256"]["left"] == result["predictionsSha256"]["right"]
    assert result["statistics"]["estimates"]["auroc"] == {
        "left": 0.75,
        "right": 0.75,
        "difference": 0,
    }
    assert result["statistics"]["intervals"]["auroc"] == {"lower": 0, "upper": 0}


@pytest.mark.parametrize("change", ["patient", "label", "missing", "aggregate", "checksum"])
def test_comparison_rejects_corrupt_or_mismatched_evidence_even_with_new_checksum(
    comparison, change
):
    current, request, folder, execution = comparison
    path = folder / "predictions.json"
    source = json.loads(path.read_text())
    if change == "patient":
        source["records"][0]["patientId"] = "someone-else"
    elif change == "label":
        source["records"][0].update(label="disease", labelIndex=0)
    elif change == "missing":
        source["records"].pop()
    elif change == "aggregate":
        source["patientRecords"][0]["probabilities"] = [0.9, 0.1]
    content = json.dumps(source, indent=2).encode()
    path.write_bytes(content)
    if change != "checksum":
        execution["result"]["artifacts"]["predictions.json"].update(
            bytes=len(content), sha256=hashlib.sha256(content).hexdigest()
        )
    with pytest.raises(StorageError):
        current.compare(request)


@pytest.mark.parametrize("service", ["fallback"], indirect=True)
def test_slide_fallback_id_does_not_establish_independent_patient(comparison):
    current, request, *_ = comparison
    with pytest.raises(StorageError, match="verified patient identities"):
        current.compare(request)
