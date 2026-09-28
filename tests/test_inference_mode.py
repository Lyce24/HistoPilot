"""Inference mode: unlabeled cohorts, prediction-only runs and label-free analysis."""

import copy
import csv
import hashlib
import io
import json
import math
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from histopilot.api.app import create_app
from histopilot.application.evaluation_runs import EvaluationRunService, run_purpose
from histopilot.application.inference_analysis import InferenceAnalysisService
from histopilot.application.predictors import reference
from histopilot.config import Settings
from histopilot.inference_summary import (
    agreement,
    cross_tab,
    describe,
    patient_member_probabilities,
    predicted_index,
    quantile,
    summarize,
)
from histopilot.schemas.case_review import CaseReviewQuery
from histopilot.schemas.evaluations import EvaluationSpec
from histopilot.schemas.inference import (
    InferenceAttentionRequest,
    InferenceExportQuery,
    InferenceSummaryQuery,
)
from histopilot.schemas.predictors import (
    CompareEvaluations,
    EvaluationRunSelection,
    SaveEvaluationRun,
)
from histopilot.storage.filesystem import LocalFilesystem
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import ScientificStore

support = runpy.run_path(str(Path(__file__).with_name("test_evaluations.py")))
registry = runpy.run_path(str(Path(__file__).with_name("test_predictor_registry.py")))

BINARY = {
    "task": "binary_classification",
    "unit": "slide",
    "classes": ["benign", "tumor"],
    "positiveClass": "tumor",
}
GRADES = {"task": "multiclass_classification", "unit": "slide", "classes": ["ND", "IND", "LG", "HG"]}


# Label-free summaries ---------------------------------------------------------


def test_decisions_follow_frozen_threshold_and_first_highest_class():
    assert predicted_index([0.7, 0.3], BINARY, 0.25) == 1
    assert predicted_index([0.7, 0.3], BINARY, 0.5) == 0
    assert predicted_index([0.2, 0.4, 0.4, 0.0], GRADES, 0.5) == 1
    row = describe({"probabilities": [0.1, 0.2, 0.6, 0.1]}, GRADES, 0.5)
    assert row == {
        "predictedIndex": 2,
        "predictedLabel": "LG",
        "confidence": 0.6,
        "margin": pytest.approx(0.4),
    }


def test_binary_margin_measures_distance_from_the_frozen_threshold():
    # With threshold 0.2, p(tumor)=0.19 is borderline even though benign has 0.81.
    near = describe({"probabilities": [0.81, 0.19]}, BINARY, 0.2)
    far = describe({"probabilities": [0.5, 0.5]}, BINARY, 0.2)
    assert near["predictedLabel"] == "benign" and near["margin"] == pytest.approx(0.05)
    assert far["predictedLabel"] == "tumor" and far["margin"] == pytest.approx(0.375)
    assert near["margin"] < far["margin"]
    # At a 0.5 threshold the margin equals top class minus runner-up.
    assert describe({"probabilities": [0.3, 0.7]}, BINARY, 0.5)["margin"] == pytest.approx(0.4)
    assert describe({"probabilities": [1.0, 0.0]}, BINARY, 0.2)["margin"] == pytest.approx(1.0)


def test_member_agreement_counts_votes_for_the_ensemble_decision():
    row = {
        "probabilities": [0.1, 0.1, 0.5, 0.3],
        "memberProbabilities": [[0.1, 0.1, 0.7, 0.1], [0.1, 0.1, 0.3, 0.5], [0.1, 0.1, 0.5, 0.3]],
    }
    members = describe(row, GRADES, 0.5)["memberAgreement"]
    assert members["agree"] == 2 and members["total"] == 3
    assert members["spread"] == pytest.approx(math.sqrt(0.08 / 3))
    assert "memberAgreement" not in describe({"probabilities": [0.4, 0.6]}, BINARY, 0.5)


def test_summary_describes_distribution_threshold_sweep_and_ensemble():
    rows = [
        {"probabilities": [1 - p, p], "memberProbabilities": [[1 - p, p], [1 - q, q]]}
        for p, q in [(0.9, 0.8), (0.58, 0.2), (0.1, 0.05), (0.3, 0.4)]
    ]
    result = summarize(rows, BINARY, 0.5)
    assert result["count"] == 4
    assert [(item["label"], item["count"]) for item in result["predicted"]] == [
        ("benign", 2), ("tumor", 2),
    ]
    assert result["predicted"][1]["meanConfidence"] == pytest.approx(0.74)
    assert result["confidence"]["counts"]["benign"][18] == 1
    assert result["confidence"]["counts"]["benign"][14] == 1
    assert sum(result["margin"]["counts"]["tumor"]) == 2
    binary = result["binary"]
    assert binary["positiveClass"] == "tumor" and binary["threshold"] == 0.5
    sweep = {item["threshold"]: item["positive"] for item in binary["sweep"]}
    assert sweep[0.5] == 2 and sweep[0.3] == 3 and sweep[0.95] == 0
    assert binary["nearThreshold"] == [{"band": 0.05, "count": 0}, {"band": 0.1, "count": 1}]
    assert sum(binary["positiveProbability"]["counts"]) == 4
    ensemble = result["ensemble"]
    assert ensemble["memberCount"] == 2 and ensemble["records"] == 4
    assert ensemble["unanimous"] == 3 and ensemble["disagreements"] == 1
    assert ensemble["agreement"] == [
        {"agree": 2, "count": 3}, {"agree": 1, "count": 1}, {"agree": 0, "count": 0},
    ]
    described = [describe(row, BINARY, 0.5) for row in rows]
    table = cross_tab(described, ["A", "B", "A", "Missing"], BINARY["classes"], limit=2)
    assert table["rows"][0] == {
        "value": "A",
        "count": 2,
        "counts": {"benign": 1, "tumor": 1},
        "meanConfidence": pytest.approx(0.9),
    }
    assert table["otherValues"] == 1
    assert table["other"] == {"count": 1, "counts": {"benign": 1, "tumor": 0}}


def test_run_agreement_kappa_and_patient_member_rules():
    assert agreement([0, 0, 1, 1], [0, 1, 1, 1], ["a", "b"]) == {
        "count": 4, "agreement": 0.75, "kappa": 0.5, "disagreements": 1,
        "matrix": [[1, 1], [0, 2]],
    }
    assert agreement([0, 0], [0, 0], ["a", "b"])["kappa"] is None
    slides = [
        {"memberProbabilities": [[0.2, 0.8], [0.6, 0.4]]},
        {"memberProbabilities": [[0.4, 0.6], [0.8, 0.2]]},
    ]
    assert patient_member_probabilities(slides, "mean") == [
        pytest.approx([0.3, 0.7]), pytest.approx([0.7, 0.3]),
    ]
    first, second = math.sqrt(0.2 * 0.4), math.sqrt(0.8 * 0.6)
    assert patient_member_probabilities(slides, "mean_logits")[0] == pytest.approx(
        [first / (first + second), second / (first + second)]
    )
    assert patient_member_probabilities([{"memberProbabilities": None}], "mean") is None
    assert quantile([1, 2, 3, 4], 0.5) == 2.5 and quantile([], 0.5) is None


# Cohorts and runs --------------------------------------------------------------


def test_inference_cohorts_are_unlabeled_and_keep_their_stored_purpose():
    base = {"datasetId": "dataset-" + "a" * 64, "eligibility": []}
    with pytest.raises(ValidationError, match="unlabeled"):
        EvaluationSpec.model_validate({**base, "purpose": "inference", "target": support["TARGET"]})
    spec = EvaluationSpec.model_validate(
        {**base, "purpose": "inference", "target": None, "patientIdentifiers": "independent"}
    )
    assert spec.model_dump(mode="json")["purpose"] == "inference"
    assert "purpose" not in EvaluationSpec.model_validate({**base, "target": None}).model_dump()
    # The earlier review name keeps its stricter rules and its stored value.
    assert EvaluationSpec.model_validate(
        {**base, "purpose": "review", "target": None}
    ).model_dump()["purpose"] == "review"
    with pytest.raises(ValidationError):
        EvaluationSpec.model_validate(
            {**base, "purpose": "review", "target": None, "patientIdentifiers": "independent"}
        )
    assert run_purpose({"purpose": "review"}) == run_purpose({"purpose": "inference"}) == "inference"
    assert run_purpose({}) == "evaluation"


def test_inference_cohort_freezes_membership_without_labels(evaluation):
    service, spec, _ = evaluation
    cohort = {
        "datasetId": spec["datasetId"],
        "purpose": "inference",
        "target": None,
        "eligibility": [{"field": "cohort", "op": "eq", "value": "test"}],
    }
    draft = support["draft"](service, cohort)
    result = service.preview(draft["id"], 1)
    assert result["canFreeze"], result["findings"]
    assert result["summary"]["labeledSlides"] == 0 and result["summary"]["classCounts"] == {}
    assert all(row["label"] is None for row in result["memberships"])
    note = next(item for item in result["findings"] if item["code"] == "INFERENCE_PREDICTIONS_ONLY")
    assert note["severity"] == "info"
    frozen = service.freeze(draft["id"], 1, result["previewHash"], "inference-freeze")
    assert frozen["manifest"]["spec"]["purpose"] == "inference"
    assert frozen["manifest"]["target"] is None


@pytest.fixture
def evaluation(tmp_path):
    folder = tmp_path / "project"
    folder.mkdir()
    return support["setup"](ScientificStore(folder, "project-inference"), tmp_path)


@pytest.mark.parametrize("development_unit", ["slide", "patient"])
def test_inference_discloses_patient_overlap_only_for_slide_predictors(
    evaluation, development_unit
):
    service, spec, _ = evaluation
    manifest = copy.deepcopy(service.store.get_configuration(spec["protocolId"])["manifest"])
    manifest["spec"]["target"]["unit"] = development_unit
    protocol = service.store.publish_configuration(manifest=manifest, operation_id="unit-protocol")
    data, _ = support["dataset"](service.store, "inference-patient", [
        {"slideId": "s2", "patientId": "p0", "attributes": {"label": "0", "cohort": "test"}},
    ])
    result = support["preview"](service, {
        **spec, "protocolId": protocol["id"], "datasetId": data["id"], "target": None,
        "purpose": "inference",
    })
    overlap = next(item for item in result["findings"] if item["code"] == "DEVELOPMENT_PATIENT_OVERLAP")
    assert overlap["severity"] == ("warning" if development_unit == "slide" else "error")
    assert ("flagged" in overlap["message"]) == (development_unit == "slide")
    assert result["canFreeze"] == (development_unit == "slide"), result["findings"]


def test_inference_blocks_development_slides_and_points_to_oof_predictions(evaluation):
    service, spec, _ = evaluation
    result = support["preview"](
        service, {**spec, "purpose": "inference", "target": None, "eligibility": []}
    )
    overlap = next(item for item in result["findings"] if item["code"] == "DEVELOPMENT_SLIDE_OVERLAP")
    assert overlap["severity"] == "error" and "out-of-fold" in overlap["message"]
    assert not result["canFreeze"]


def test_inference_run_is_unlabeled_metric_free_and_marked_in_its_plan(tmp_path, monkeypatch):
    predictors, bound = registry["registry"].__wrapped__(tmp_path)
    selection, *_ = registry["candidate"](predictors)
    predictor, _ = registry["freeze"](predictors, selection)
    runs = EvaluationRunService(predictors.store, predictors.filesystem)
    draft = support["draft"](runs.cohorts, {
        "datasetId": bound["manifest"]["datasetId"], "purpose": "inference", "target": None,
        "eligibility": [{"field": "cohort", "op": "eq", "value": "test"}],
    }, "Unlabeled test slides")
    preview = runs.cohorts.preview(draft["id"], 1)
    cohort = runs.cohorts.freeze(draft["id"], 1, preview["previewHash"], "inference-cohort")
    chosen = EvaluationRunSelection(
        predictorId=predictor["id"], cohortId=cohort["id"], name="Unlabeled predictions"
    )
    result = runs.preview(chosen)
    assert result["canSave"], result["findings"]
    manifest = result["manifest"]
    assert manifest["purpose"] == "inference" and "analysis" not in manifest
    assert result["executionNote"] == manifest["executionNote"]
    assert manifest["executionNote"].startswith("Predict the frozen inference cohort")
    document = runs.save(SaveEvaluationRun(
        **chosen.model_dump(), previewHash=result["previewHash"], operationId="inference-run"
    ))
    monkeypatch.setattr(
        "histopilot.application.evaluation_runs.training_runtime",
        lambda: {"available": True, "cudaAvailable": False, "gpuCount": 0,
                 "python": sys.executable, "versions": {}},
    )
    plan = runs._execution_plan(document["id"])
    assert plan["purpose"] == "inference" and "analysis" not in plan
    assert all(row["label"] is None for row in plan["data"]["memberships"])
    with pytest.raises(StorageError) as error:
        runs.compare(CompareEvaluations(
            leftEvaluationId=document["id"], rightEvaluationId=document["id"]
        ))
    assert error.value.code == "COMPARISON_REQUIRES_LABELS"


@pytest.mark.parametrize("unit", ["slide", "patient"])
def test_inference_run_gates_patient_overlap_by_prediction_unit(tmp_path, monkeypatch, unit):
    service, cohort = registry["registry"].__wrapped__(tmp_path)
    selection, *_ = registry["candidate"](service)
    predictor, _ = registry["freeze"](service, selection)
    runs = EvaluationRunService(service.store, service.filesystem)
    predictor = copy.deepcopy(predictor)
    altered = copy.deepcopy(runs.cohorts.get(cohort["id"]))
    predictor["manifest"]["target"]["unit"] = unit
    test = altered["manifest"]
    test["target"]["unit"] = unit
    test["spec"].update(purpose="inference", target=None, patientIdentifiers="independent")
    test["overlap"]["patientIds"] = ["development-patient"]
    for row in test["memberships"]:
        row["label"] = None
    monkeypatch.setattr(runs.predictors, "get", lambda _: predictor)
    monkeypatch.setattr(runs.cohorts, "get", lambda _: altered)
    result = runs.preview(
        EvaluationRunSelection(predictorId=predictor["id"], cohortId=cohort["id"], name="Inference")
    )
    assert result["canSave"] == (unit == "slide"), result
    if unit == "patient":
        assert result["findings"][0]["code"] == "EVALUATION_DEVELOPMENT_OVERLAP"
        assert "Inference would predict" in result["findings"][0]["message"]
    labeled = copy.deepcopy(altered)
    labeled["manifest"]["memberships"][0]["label"] = "low"
    monkeypatch.setattr(runs.cohorts, "get", lambda _: labeled)
    result = runs.preview(
        EvaluationRunSelection(predictorId=predictor["id"], cohortId=cohort["id"], name="Inference")
    )
    assert result["findings"][0]["code"] == "INVALID_INFERENCE_COHORT"


# Label-free analysis over saved predictions ------------------------------------

SLIDES = [
    ("GEJ1A", "p1", [0.7, 0.1, 0.1, 0.1], [[0.8, 0.1, 0.05, 0.05], [0.6, 0.1, 0.15, 0.15]], "A"),
    ("GEJ1B", "p1", [0.2, 0.2, 0.5, 0.1], [[0.1, 0.1, 0.7, 0.1], [0.3, 0.3, 0.3, 0.1]], "B"),
    ("GEJ2A", "p2", [0.1, 0.1, 0.2, 0.6], [[0.1, 0.1, 0.2, 0.6], [0.1, 0.1, 0.2, 0.6]], "A"),
    ("GEJ3A", "p3", [0.4, 0.35, 0.15, 0.1], [[0.5, 0.3, 0.1, 0.1], [0.3, 0.4, 0.2, 0.1]], None),
]
BUNDLE_ID = "configuration-" + "b" * 64


def _patients(records):
    groups = {}
    for row in records:
        groups.setdefault(row["patientId"], []).append(row)
    return [
        {
            "patientId": patient,
            "slideIds": [row["slideId"] for row in rows],
            "label": None,
            "labelIndex": None,
            "probabilities": [
                math.fsum(row["probabilities"][index] for row in rows) / len(rows)
                for index in range(4)
            ],
        }
        for patient, rows in groups.items()
    ]


@pytest.fixture
def inference_run(tmp_path, monkeypatch):
    store = ScientificStore(tmp_path, "inference-analysis")
    slides = tmp_path / "slides"
    metadata = [
        {"slideId": slide, "patientId": patient, "attributes": {"part": part},
         "slidePath": str(slides / f"{slide}.svs")}
        for slide, patient, _, _, part in SLIDES
    ]
    draft = store.create_draft("import", "BD", {})
    dataset = store.publish_dataset(
        draft["id"], expected_revision=1,
        manifest={"kind": "dataset", "name": "BD",
                  "dictionary": [{"key": "part", "sourceColumn": "Part"}]},
        artifacts={"records.json": json.dumps(metadata).encode()}, operation_id="dataset",
    )

    def publish(kind, op=None, **values):
        return store.publish_configuration(
            manifest={"kind": kind, "datasetId": dataset["id"], **values}, operation_id=op or kind
        )

    predictor = publish("frozen-predictor", target=GRADES)
    memberships = [
        {"slideId": slide, "patientId": patient, "patientIdSource": "source", "label": None}
        for slide, patient, *_ in SLIDES
    ]
    cohort = publish(
        "evaluation-cohort", target=None, memberships=memberships,
        spec={"purpose": "inference", "datasetId": dataset["id"], "target": None},
    )

    def run(name, records, op):
        return publish(
            "model-evaluation", op, purpose="inference", name=name, target=GRADES,
            inference={"decisionThreshold": 0.5, "patientAggregation": "mean",
                       "loadingPolicy": "per_slide", "packArtifactId": None},
            predictorId=predictor["id"], predictor=reference(predictor),
            cohortId=cohort["id"], cohort=reference(cohort),
            features={"bundle": {"id": BUNDLE_ID}},
            overlap={"slideIds": [], "patientIds": ["p1"], "patientsComparable": True},
        ), {"classOrder": GRADES["classes"], "records": records, "patientRecords": _patients(records)}

    records = [
        {"slideId": slide, "patientId": patient, "patientIdSource": "source", "label": None,
         "labelIndex": None, "probabilities": probabilities, "memberProbabilities": members}
        for slide, patient, probabilities, members, _ in SLIDES
    ]
    evaluation, predictions = run("Ensemble", records, "ensemble-run")
    service = InferenceAnalysisService(store, LocalFilesystem((tmp_path,)))
    executions = {}

    def install(document, content):
        folder = service.evaluations.jobs.folder(document["id"])
        folder.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(content).encode()
        path = folder / "predictions.json"
        path.write_bytes(payload)
        executions[document["id"]] = {"status": "completed", "result": {"artifacts": {
            "predictions.json": {"path": str(path), "bytes": len(payload),
                                 "sha256": hashlib.sha256(payload).hexdigest()},
        }}}
        return payload

    payload = install(evaluation, predictions)
    monkeypatch.setattr(
        service.evaluations.jobs, "status", lambda identity, **kwargs: executions[identity]
    )
    return SimpleNamespace(service=service, evaluation=evaluation, predictor=predictor,
                           cohort=cohort, dataset=dataset, payload=payload, run=run,
                           install=install, slides=slides, store=store)


def test_summary_counts_predictions_agreement_development_and_attribute(inference_run):
    service, identity = inference_run.service, inference_run.evaluation["id"]
    result = service.summary(identity, InferenceSummaryQuery())
    assert result["purpose"] == "inference" and result["unit"] == "slide"
    assert result["count"] == 4 and result["patients"] == 3
    assert {item["label"]: item["count"] for item in result["predicted"]} == {
        "ND": 2, "IND": 0, "LG": 1, "HG": 1,
    }
    assert "binary" not in result
    assert result["ensemble"]["unanimous"] == 2 and result["ensemble"]["disagreements"] == 2
    assert result["development"] == {
        "comparable": True, "patients": 1, "records": 2, "unknown": 0,
        "shared": {"ND": 1, "IND": 0, "LG": 1, "HG": 0},
        "new": {"ND": 1, "IND": 0, "LG": 0, "HG": 1},
    }
    assert result["attributes"] == [{"key": "part", "label": "Part"}]
    assert result["source"]["predictionsSha256"] == hashlib.sha256(inference_run.payload).hexdigest()
    breakdown = service.summary(identity, InferenceSummaryQuery(attribute="part"))["breakdown"]
    assert breakdown["label"] == "Part"
    assert [row["value"] for row in breakdown["rows"]] == ["A", "B", "Missing"]
    assert breakdown["rows"][0]["counts"] == {"ND": 1, "IND": 0, "LG": 0, "HG": 1}
    with pytest.raises(StorageError):
        service.summary(identity, InferenceSummaryQuery(attribute="grade"))


def test_patient_view_recomputes_member_agreement_under_frozen_rule(inference_run):
    result = inference_run.service.summary(
        inference_run.evaluation["id"], InferenceSummaryQuery(unit="patient")
    )
    assert result["unit"] == "patient" and result["count"] == 3
    assert {item["label"]: item["count"] for item in result["predicted"]} == {
        "ND": 2, "IND": 0, "LG": 0, "HG": 1,
    }
    # p1 averages both slides for each fold member; both still choose ND.
    assert result["ensemble"]["disagreements"] == 1


def test_label_free_comparison_reports_agreement_and_kappa(inference_run):
    records = [
        {"slideId": slide, "patientId": patient, "patientIdSource": "source", "label": None,
         "labelIndex": None, "probabilities": probabilities}
        for (slide, patient, *_), probabilities in zip(SLIDES, [
            [0.6, 0.2, 0.1, 0.1], [0.5, 0.2, 0.2, 0.1], [0.1, 0.1, 0.1, 0.7], [0.2, 0.6, 0.1, 0.1],
        ], strict=True)
    ]
    refit, predictions = inference_run.run("Refit", records, "refit-run")
    inference_run.install(refit, predictions)
    comparison = inference_run.service.summary(
        inference_run.evaluation["id"], InferenceSummaryQuery(comparisonId=refit["id"])
    )["comparison"]
    assert comparison["name"] == "Refit" and comparison["count"] == 4
    assert comparison["agreement"] == 0.5 and comparison["disagreements"] == 2
    assert comparison["kappa"] == pytest.approx(0.1875 / 0.6875)
    review = inference_run.service.cases.query(
        inference_run.evaluation["id"], CaseReviewQuery(comparisonId=refit["id"])
    )
    assert review["comparison"]["featureBundleId"] == BUNDLE_ID
    assert review["comparison"]["packArtifactId"] is None
    with pytest.raises(StorageError) as error:
        inference_run.service.summary(
            inference_run.evaluation["id"],
            InferenceSummaryQuery(comparisonId=inference_run.evaluation["id"]),
        )
    assert error.value.code == "INFERENCE_COMPARISON_INVALID"


def test_export_joins_predictions_flags_and_frozen_attributes(inference_run):
    payload = inference_run.service.export(
        inference_run.evaluation["id"], InferenceExportQuery(attributes=["part"])
    )
    rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))
    assert [row["Case"] for row in rows] == ["GEJ1A", "GEJ1B", "GEJ2A", "GEJ3A"]
    first = rows[0]
    assert first["Predicted"] == "ND" and float(first["Confidence"]) == 0.7
    assert float(first["P(HG)"]) == 0.1 and first["Members_agreeing"] == "2"
    assert first["Members"] == "2" and rows[1]["Members_agreeing"] == "1"
    assert [row["Development_patient"] for row in rows] == ["yes", "yes", "no", "no"]
    assert first["Part"] == "A" and rows[3]["Part"] == "Missing"
    assert first["Predictions_SHA256"] == hashlib.sha256(inference_run.payload).hexdigest()
    assert "Actual" not in first and "Outcome" not in first
    predictions_only = inference_run.service.export(
        inference_run.evaluation["id"], InferenceExportQuery(attributes=[])
    )
    assert "Part" not in predictions_only.decode("utf-8-sig").splitlines()[0]
    with pytest.raises(StorageError):
        inference_run.service.export(
            inference_run.evaluation["id"], InferenceExportQuery(attributes=["grade"])
        )


def test_review_queue_sorts_and_filters_without_outcomes(inference_run):
    cases, identity = inference_run.service.cases, inference_run.evaluation["id"]
    page = cases.query(identity, CaseReviewQuery(sort="confidence_asc"))
    assert page["purpose"] == "inference" and page["memberCount"] == 2
    assert page["developmentComparable"] is True
    assert [row["id"] for row in page["items"]] == ["GEJ3A", "GEJ1B", "GEJ2A", "GEJ1A"]
    assert page["summary"] == {"total": 4, "unlabeled": 4}
    margin = cases.query(identity, CaseReviewQuery(sort="margin_asc"))
    assert [row["id"] for row in margin["items"]] == ["GEJ3A", "GEJ1B", "GEJ2A", "GEJ1A"]
    shared = cases.query(identity, CaseReviewQuery(developmentPatients="shared"))
    assert {row["id"] for row in shared["items"]} == {"GEJ1A", "GEJ1B"}
    assert all(row["developmentPatient"] is True for row in shared["items"])
    new = cases.query(identity, CaseReviewQuery(developmentPatients="new"))
    assert {row["id"] for row in new["items"]} == {"GEJ2A", "GEJ3A"}
    disagree = cases.query(identity, CaseReviewQuery(memberDisagreement=True, sort="agreement_asc"))
    assert [row["id"] for row in disagree["items"]] == ["GEJ1B", "GEJ3A"]
    assert disagree["items"][0]["memberAgreement"]["agree"] == 1
    uncertain = cases.query(identity, CaseReviewQuery(maxConfidence=0.55))
    assert {row["id"] for row in uncertain["items"]} == {"GEJ1B", "GEJ3A"}


def test_slide_id_fallback_patients_are_unknown_not_new(inference_run):
    content = json.loads(inference_run.payload)
    content["records"][2]["patientIdSource"] = "slide_fallback"
    inference_run.install(inference_run.evaluation, content)
    identity = inference_run.evaluation["id"]
    development = inference_run.service.summary(identity, InferenceSummaryQuery())["development"]
    assert development["unknown"] == 1 and development["records"] == 2
    assert sum(development["new"].values()) == 1
    rows = list(csv.DictReader(io.StringIO(
        inference_run.service.export(identity, InferenceExportQuery(attributes=[])).decode("utf-8-sig")
    )))
    assert [row["Development_patient"] for row in rows] == ["yes", "yes", "", "no"]
    page = inference_run.service.cases.query(identity, CaseReviewQuery(unit="slide"))
    flags = {row["id"]: row["developmentPatient"] for row in page["items"]}
    assert flags["GEJ2A"] is None and flags["GEJ3A"] is False
    assert inference_run.service.cases.query(
        identity, CaseReviewQuery(unit="slide", developmentPatients="new")
    )["total"] == 1


def test_comparison_requires_the_case_review_pairing_context(inference_run):
    store, dataset_id = inference_run.store, inference_run.dataset["id"]
    other_target = {**GRADES, "field": "target_label"}
    predictor = store.publish_configuration(
        manifest={"kind": "frozen-predictor", "datasetId": dataset_id, "target": other_target},
        operation_id="other-target-predictor",
    )
    manifest = copy.deepcopy(inference_run.evaluation["manifest"])
    manifest.update(name="Other target", target=other_target, predictorId=predictor["id"],
                    predictor=reference(predictor))
    document = store.publish_configuration(manifest=manifest, operation_id="other-target-run")
    inference_run.install(document, json.loads(inference_run.payload))
    with pytest.raises(StorageError) as error:
        inference_run.service.summary(
            inference_run.evaluation["id"], InferenceSummaryQuery(comparisonId=document["id"])
        )
    assert error.value.code == "INFERENCE_COMPARISON_MISMATCH"


def test_malformed_member_probabilities_fail_closed(inference_run):
    evaluation = inference_run.evaluation
    content = json.loads(inference_run.payload)
    content["records"][0]["memberProbabilities"] = [[0.5, 0.5, 0.5, 0.5], [1, 0, 0, 0]]
    inference_run.install(evaluation, content)
    with pytest.raises(StorageError) as error:
        inference_run.service.summary(evaluation["id"], InferenceSummaryQuery())
    assert error.value.code == "CASE_REVIEW_EVIDENCE_INVALID"


def test_attention_uses_run_features_lineage_and_dataset_slide_folder(inference_run, monkeypatch):
    captured = []

    class Interpretation:
        def __init__(self, store, filesystem):
            self.gallery = SimpleNamespace(dataset_source=lambda dataset: {
                "slideFolder": str(inference_run.slides), "slideFolderFinding": None,
            })

        def visualize(self, request):
            captured.append(request)
            return {
                "items": [{"slidePath": path, "status": "queued"} for path in request.slidePaths],
                "interpretations": [],
            }

    monkeypatch.setattr("histopilot.application.interpretation.InterpretationService", Interpretation)
    identity = inference_run.evaluation["id"]
    result = inference_run.service.attention(identity, InferenceAttentionRequest(
        slideIds=["GEJ2A", "GEJ1A"], operationId="attention-1",
    ))
    request = captured[0]
    assert request.slidePaths == [
        str(inference_run.slides / "GEJ2A.svs"), str(inference_run.slides / "GEJ1A.svs"),
    ]
    assert request.evaluationId == identity and request.predictorId == inference_run.predictor["id"]
    assert request.featureBundleId == BUNDLE_ID and request.packArtifactId is None
    assert request.slideFolder == str(inference_run.slides)
    assert request.operationId.startswith("inference-attention-")
    assert [item["status"] for item in result["items"]] == ["queued", "queued"]
    with pytest.raises(StorageError) as error:
        inference_run.service.attention(identity, InferenceAttentionRequest(
            slideIds=["unknown"], operationId="attention-2",
        ))
    assert error.value.code == "INFERENCE_SLIDE_UNAVAILABLE"


def test_inference_routes_require_auth_and_return_summary_and_csv(inference_run, tmp_path, monkeypatch):
    app = create_app(Settings(workspace=tmp_path / "api-workspace", data_roots=(tmp_path,)))
    monkeypatch.setattr(app.state.projects, "scientific_store", lambda identity: inference_run.store)
    monkeypatch.setattr(
        "histopilot.api.inference.InferenceAnalysisService", lambda *args: inference_run.service
    )
    base = f"/api/v1/projects/bd/evaluation-runs/{inference_run.evaluation['id']}"
    with TestClient(app, base_url="http://127.0.0.1:8787") as client:
        assert client.post(base + "/inference/summary", json={}).status_code == 401
        client.headers["X-HistoPilot-Token"] = client.get("/api/v1/session").json()["token"]
        response = client.post(base + "/inference/summary", json={"attribute": "part"})
        assert response.status_code == 200, response.text
        assert response.json()["breakdown"]["rows"][0]["value"] == "A"
        response = client.post(base + "/inference/export", json={"unit": "slide"})
        assert response.status_code == 200 and "text/csv" in response.headers["content-type"]
        assert "GEJ3A" in response.text and "Development_patient" in response.text
        too_many = client.post(base + "/attention", json={
            "slideIds": [f"s{index}" for index in range(33)], "operationId": "attention",
        })
        assert too_many.status_code == 422


def test_patient_member_logit_votes_preserve_underflowed_probabilities():
    slides = [
        {"memberProbabilities": [[1.0, 0.0], [1.0, 0.0]],
         "memberLogProbabilities": [[0.0, -1000.0], [0.0, -1000.0]]},
        {"memberProbabilities": [[0.0, 1.0], [0.0, 1.0]],
         "memberLogProbabilities": [[-2000.0, 0.0], [-2000.0, 0.0]]},
    ]
    members = patient_member_probabilities(slides, "mean_logits")
    assert members == [pytest.approx([math.exp(-500), 1.0])] * 2
    assert patient_member_probabilities(
        [{"memberProbabilities": row["memberProbabilities"]} for row in slides], "mean_logits"
    ) is None  # Legacy zeros cannot establish the mean-logit member decision.


def test_margin_review_filters_exclusively_and_keeps_inference_export_label_free(inference_run):
    service, identity = inference_run.service.cases, inference_run.evaluation["id"]
    page = service.query(identity, CaseReviewQuery(maxMargin=0.2, sort="margin_asc"))
    assert [item["id"] for item in page["items"]] == ["GEJ3A"]
    assert page["featureBundleId"] == BUNDLE_ID and page["packArtifactId"] is None
    assert service.query(identity, CaseReviewQuery(maxMargin=0.0))["total"] == 0
    cutoff = page["items"][0]["margin"]
    assert service.query(identity, CaseReviewQuery(maxMargin=cutoff))["total"] == 0
    exported = list(csv.DictReader(io.StringIO(
        service.export(identity, CaseReviewQuery(maxMargin=0.2)).decode("utf-8-sig")
    )))
    assert len(exported) == 1 and exported[0]["Case"] == "GEJ3A"
    assert "Actual" not in exported[0] and "Outcome" not in exported[0]
    with pytest.raises(ValidationError):
        CaseReviewQuery(maxMargin=float("nan"))


@pytest.mark.parametrize("damage", ["count", "missing", "log_shape", "log_values"])
def test_inference_rejects_inconsistent_member_evidence(inference_run, damage):
    content = json.loads(inference_run.payload)
    row = content["records"][0]
    if damage == "count":
        row["memberProbabilities"].append(row["memberProbabilities"][0])
    elif damage == "missing":
        row.pop("memberProbabilities")
    elif damage == "log_shape":
        row["memberLogProbabilities"] = [[0.0, -1000.0]]
    else:
        row["memberLogProbabilities"] = [[0.0] * 4] * 2
    inference_run.install(inference_run.evaluation, content)
    with pytest.raises(StorageError) as error:
        inference_run.service.summary(inference_run.evaluation["id"], InferenceSummaryQuery())
    assert error.value.code == "CASE_REVIEW_EVIDENCE_INVALID"


def test_export_rejects_repeated_attributes_and_protects_dynamic_headers(inference_run, monkeypatch):
    with pytest.raises(ValidationError):
        InferenceExportQuery(attributes=["part", "part"])
    import histopilot.application.inference_analysis as analysis

    original = analysis.cohort_metadata
    def metadata(*args):
        lookup, _ = original(*args)
        return lookup, {"part": "Predicted", "formula": "=1+1", "escaped": "'=1+1"}
    monkeypatch.setattr(analysis, "cohort_metadata", metadata)
    rows = list(csv.DictReader(io.StringIO(
        inference_run.service.export(inference_run.evaluation["id"], InferenceExportQuery()).decode("utf-8-sig")
    )))
    assert rows[0]["Predicted"] == "ND"
    assert rows[0]["Attribute: Predicted [part]"] == "A"
    assert "'=1+1" in rows[0]
    assert "Attribute: '=1+1 [escaped]" in rows[0]
