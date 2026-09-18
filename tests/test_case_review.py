import copy
import csv
import hashlib
import io
import json

import pytest

from histopilot.application.case_review import CaseReviewService
from histopilot.application.predictors import reference
from histopilot.schemas.case_review import CaseReviewQuery
from histopilot.schemas.slide_reviews import SaveSlideReview
from histopilot.storage.filesystem import LocalFilesystem
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import ScientificStore


@pytest.fixture
def cases(tmp_path, monkeypatch):
    store = ScientificStore(tmp_path, "case-project")
    target = {"task": "binary_classification", "unit": "patient", "classes": ["disease", "clear"], "positiveClass": "disease"}
    inference = {"decisionThreshold": .25, "patientAggregation": "mean"}
    specs = [("s0", "p0", 1, .3), ("s0b", "p0", 1, .4), ("s1", "p1", 0, .1), ("s2", "p2", 0, .8), ("s3", "p3", None, .9)]
    metadata = [{"slideId": slide, "patientId": patient, "attributes": {"site": "A" if index < 2 else "B", "age": 30 if index < 2 else 40, "sampled": index < 2}, "slidePath": None} for index, (slide, patient, _, _) in enumerate(specs)]
    draft = store.create_draft("import", "Test", {})
    dataset = store.publish_dataset(draft["id"], expected_revision=1,
                                    manifest={"kind": "dataset", "name": "Test", "dictionary": [{"key": name, "sourceColumn": name.title()} for name in ("site", "age", "sampled")]},
                                    artifacts={"records.json": json.dumps(metadata).encode()}, operation_id="dataset")
    def publish(kind, op=None, **values):
        return store.publish_configuration(manifest={"kind": kind, "datasetId": dataset["id"], **values}, operation_id=op or kind)
    predictor = publish("frozen-predictor", target=target)
    source = {"classOrder": target["classes"], "records": [
        {"slideId": slide, "patientId": patient, "patientIdSource": "mapped", "labelIndex": label,
         "label": target["classes"][label] if label is not None else None, "probabilities": [probability, 1-probability]}
        for slide, patient, label, probability in specs
    ]}
    source["patientRecords"] = []
    for patient in ("p0", "p1", "p2", "p3"):
        rows = [row for row in source["records"] if row["patientId"] == patient]
        source["patientRecords"].append({**rows[0], "slideIds": [row["slideId"] for row in rows],
                                          "probabilities": [sum(row["probabilities"][index] for row in rows)/len(rows) for index in range(2)]})
    cohort = publish("evaluation-cohort", target=target, memberships=[{key: row[key] for key in ("slideId", "patientId", "patientIdSource", "label")} for row in source["records"]])
    evaluation = publish("model-evaluation", name="Baseline", target=target, inference=inference,
                         predictorId=predictor["id"], predictor=reference(predictor), cohortId=cohort["id"], cohort=reference(cohort))
    service = CaseReviewService(store, LocalFilesystem((tmp_path,)))
    executions = {}
    def install(document, predictions):
        folder = service.evaluations.jobs.folder(document["id"])
        folder.mkdir(parents=True, exist_ok=True)
        content = json.dumps(predictions).encode()
        path = folder / "predictions.json"
        path.write_bytes(content)
        executions[document["id"]] = {"status": "completed", "result": {"artifacts": {"predictions.json": {"path": str(path), "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}}}}
        return path
    path = install(evaluation, source)
    monkeypatch.setattr(service.evaluations.jobs, "status", lambda identity, **kwargs: executions[identity])
    return service, evaluation, dataset, source, path, install


def test_patient_cases_use_frozen_threshold_positive_class_and_aggregation(cases):
    service, evaluation, *_ = cases
    page = service.query(evaluation["id"], CaseReviewQuery())
    assert page["unit"] == "patient" and page["total"] == 4
    assert page["summary"] == {"total": 4, "false_positive": 1, "false_negative": 1, "correct": 1, "unlabeled": 1}
    fp = next(row for row in page["items"] if row["id"] == "p0")
    assert fp["predictedLabel"] == "disease" and fp["confidence"] == pytest.approx(.35)
    assert len(fp["slides"]) == 2
    assert fp["attributes"] == {"site": ["A"], "age": [30], "sampled": [True]}
    errors = service.query(evaluation["id"], CaseReviewQuery(outcome="error"))
    assert errors["total"] == 2
    cell = service.query(evaluation["id"], CaseReviewQuery(actualClass=1, predictedClass=0))
    assert [row["id"] for row in cell["items"]] == ["p0"]


def test_metadata_search_pagination_and_unlabeled_cases(cases):
    service, evaluation, *_ = cases
    page = service.query(evaluation["id"], CaseReviewQuery(unit="slide", attribute="site", attributeValue="A", limit=1))
    assert page["total"] == 2 and page["hasMore"]
    assert page["items"][0]["id"] == "s0b"
    page = service.query(evaluation["id"], CaseReviewQuery(outcome="unlabeled"))
    assert page["total"] == 1 and page["items"][0]["label"] is None
    page = service.query(evaluation["id"], CaseReviewQuery(search="s0b"))
    assert page["items"][0]["id"] == "p0"


def test_exports_include_review_revision_without_altering_predictions(cases):
    service, evaluation, dataset, _, path, _ = cases
    before = path.read_bytes()
    service.reviews.save(dataset["id"], "s1", SaveSlideReview(expectedRevision=0, status="review", notes="=Unsafe formula", evaluationId=evaluation["id"]))
    payload = service.export(evaluation["id"], CaseReviewQuery(outcome="false_negative"))
    rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))
    assert len(rows) == 1 and rows[0]["Review_revision"] == "1"
    assert rows[0]["Notes"] == "'=Unsafe formula"
    assert rows[0]["Predictions_SHA256"] == hashlib.sha256(before).hexdigest()
    assert path.read_bytes() == before


def test_oversized_review_exports_require_narrower_filters(cases, monkeypatch):
    service, evaluation, dataset, *_ = cases
    monkeypatch.setattr("histopilot.application.case_review.MAX_REVIEW_SLIDES", 2)
    with pytest.raises(StorageError, match="too many slides"):
        service.export(evaluation["id"], CaseReviewQuery())
    assert service.export(evaluation["id"], CaseReviewQuery(outcome="false_negative"))
    service.reviews.save(dataset["id"], "s1", SaveSlideReview(expectedRevision=0, notes="a" * 3000))
    monkeypatch.setattr("histopilot.application.case_review.MAX_REVIEW_RESPONSE_BYTES", 1000)
    with pytest.raises(StorageError, match="too much review text"):
        service.export(evaluation["id"], CaseReviewQuery(outcome="false_negative"))


@pytest.mark.parametrize(("attribute", "value"), [("age", "30"), ("sampled", "True")])
def test_metadata_filters_match_the_displayed_values_of_numeric_and_boolean_fields(cases, attribute, value):
    service, evaluation, *_ = cases
    all_cases = service.query(evaluation["id"], CaseReviewQuery(unit="slide"))
    assert value in next(row["values"] for row in all_cases["attributes"] if row["key"] == attribute)
    result = service.query(evaluation["id"], CaseReviewQuery(unit="slide", attribute=attribute, attributeValue=value))
    assert {row["id"] for row in result["items"]} == {"s0", "s0b"}


def test_corrupt_predictions_are_never_displayed(cases):
    service, evaluation, _, _, path, _ = cases
    path.write_text('{"classOrder": ["disease", "clear"]}')
    with pytest.raises(StorageError):
        service.query(evaluation["id"], CaseReviewQuery())


def test_same_cohort_comparison_aligns_case_identity_and_detects_disagreement(cases):
    service, evaluation, _, source, _, install = cases
    manifest = {**evaluation["manifest"], "name": "Other", "inference": {"decisionThreshold": .75, "patientAggregation": "mean"}}
    other = service.store.publish_configuration(manifest=manifest, operation_id="other")
    install(other, source)
    page = service.query(evaluation["id"], CaseReviewQuery(comparisonId=other["id"], outcome="disagreement"))
    assert page["total"] == 1 and page["items"][0]["id"] == "p0"
    assert page["items"][0]["comparison"]["predictedLabel"] == "clear"
    assert page["comparison"]["decisionThreshold"] == .75


def test_forged_patient_aggregation_is_rejected_even_with_updated_checksum(cases):
    service, evaluation, _, source, _, install = cases
    forged = copy.deepcopy(source)
    forged["patientRecords"][0]["probabilities"] = [.99, .01]
    install(evaluation, forged)
    with pytest.raises(StorageError, match="mean probabilities"):
        service.query(evaluation["id"], CaseReviewQuery())


def test_filters_cannot_request_unknown_classes_or_attributes(cases):
    service, evaluation, *_ = cases
    for query in (CaseReviewQuery(actualClass=4), CaseReviewQuery(attribute="unknown"), CaseReviewQuery(outcome="disagreement")):
        with pytest.raises(StorageError):
            service.query(evaluation["id"], query)
