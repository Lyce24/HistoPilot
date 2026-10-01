"""Reference standards score saved predictions against labels attached after the run.

A reference is one dataset column mapped to a run's classes for every slide of its cohort.
Scoring a run against a reference that repeats the cohort's own labels must give exactly the
cohort's scores, and every view that reads labels (metrics, subgroups, cases, clinical
utility, agreement) must follow the chosen reference.
"""

import csv
import hashlib
import io
import json

import pytest

from histopilot.application.case_review import CaseReviewService
from histopilot.application.clinical import ClinicalService
from histopilot.application.evaluation_runs import _SCORES, REFERENCE_SCORES
from histopilot.application.predictors import reference
from histopilot.application.references import ReferenceService
from histopilot.application.run_evidence import join_reference
from histopilot.application.run_performance import RunPerformanceService
from histopilot.inference_summary import agreement, weighted_kappa
from histopilot.schemas.case_review import CaseReviewQuery
from histopilot.schemas.clinical import ClinicalSelection
from histopilot.schemas.performance import PerformanceBreakdownQuery
from histopilot.schemas.references import ReferenceSelection, SaveReference
from histopilot.scoring import patient_predictions
from histopilot.storage.filesystem import LocalFilesystem
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import ScientificStore

BINARY = {
    "task": "binary_classification",
    "unit": "patient",
    "classes": ["disease", "clear"],
    "positiveClass": "disease",
}
INFERENCE = {"decisionThreshold": 0.25, "patientAggregation": "mean"}
ANALYSIS = {"bootstrapResamples": 200, "bootstrapSeed": 7, "oneSlideSeed": 3}
# slide, patient, cohort label, reader A's value, reader B's value, positive probability
SLIDES = [
    ("s0", "p0", "disease", "D", "D", 0.4),
    ("s0b", "p0", "disease", "D", "C", 0.3),
    ("s1", "p1", "clear", "C", "C", 0.2),
    ("s2", "p2", "clear", "C", "D", 0.7),
    ("s3", "p3", None, None, "D", 0.8),
    ("s4", "p4", "disease", "D", "D", 0.8),
    ("s5", "p5", "clear", "C", "?", 0.15),
]
MAPPING = {"D": "disease", "C": "clear"}


def _records():
    rows = []
    for slide, patient, _label, _a, _b, positive in SLIDES:
        members = [min(0.99, positive + 0.05), max(0.01, positive - 0.05)]
        rows.append(
            {
                "slideId": slide,
                "patientId": patient,
                "patientIdSource": "source",
                "label": None,
                "labelIndex": None,
                "probabilities": [positive, 1 - positive],
                "memberProbabilities": [[value, 1 - value] for value in members],
            }
        )
    return {
        "classOrder": BINARY["classes"],
        "records": rows,
        "patientRecords": patient_predictions(rows, "mean"),
    }


@pytest.fixture
def study(tmp_path, monkeypatch):
    _SCORES.clear()
    store = ScientificStore(tmp_path, "references")

    def dataset(name, rows):
        draft = store.create_draft("import", name, {})
        return store.publish_dataset(
            draft["id"],
            expected_revision=1,
            manifest={
                "kind": "dataset",
                "name": name,
                "dictionary": [{"key": key, "sourceColumn": key} for key in rows[0]["attributes"]],
            },
            artifacts={"records.json": json.dumps(rows).encode()},
            operation_id=f"dataset-{name}",
        )

    original = dataset(
        "Original",
        [
            {
                "slideId": slide,
                "patientId": patient,
                "patientIdSource": "source",
                "slidePath": None,
                "attributes": {"site": "A" if index < 4 else "B", "readerA": a, "readerB": b},
            }
            for index, (slide, patient, _label, a, b, _p) in enumerate(SLIDES)
        ],
    )
    # A later import adds a consensus column for most slides; s5 is not in it.
    revised = dataset(
        "Revised",
        [
            {
                "slideId": slide,
                "patientId": patient,
                "patientIdSource": "source",
                "slidePath": None,
                "attributes": {"consensus": label or "D"},
            }
            for slide, patient, label, _a, _b, _p in SLIDES[:-1]
        ],
    )

    def publish(kind, operation, **values):
        return store.publish_configuration(
            manifest={"kind": kind, "datasetId": original["id"], **values}, operation_id=operation
        )

    predictor = publish(
        "frozen-predictor",
        "predictor",
        target=BINARY,
        method="ensemble",
        checkpoints=[{"runId": "a"}, {"runId": "b"}],
        aggregation="mean_probability",
        # Runs on labeled cohorts record these bootstrap settings; reference scores reuse them.
        recipe={"analysis": ANALYSIS},
        # The development group whose out-of-fold predictions recalibration fits on.
        batchId="batch",
        candidateId="candidate",
        trainingSeed=7,
        splitSeed=42,
    )

    def cohort(name, labeled):
        memberships = [
            {
                "slideId": slide,
                "patientId": patient,
                "patientIdSource": "source",
                "label": label if labeled else None,
            }
            for slide, patient, label, _a, _b, _p in SLIDES
        ]
        return publish(
            "evaluation-cohort",
            name,
            target=BINARY if labeled else None,
            spec={"splitUnit": "patient", "purpose": None if labeled else "inference"},
            overlap={"deferred": True},
            memberships=memberships,
        )

    labeled_cohort, unlabeled_cohort = cohort("labeled", True), cohort("unlabeled", False)
    filesystem = LocalFilesystem((tmp_path,))
    cases = CaseReviewService(store, filesystem)
    executions = {}

    def run(name, cohort_document, *, blind, purpose=None, model=None):
        model = model or predictor
        document = publish(
            "model-evaluation",
            name,
            **({"purpose": purpose} if purpose else {}),
            name=name,
            experimentId="experiment",
            target=BINARY,
            inference=INFERENCE,
            **({} if purpose else {"analysis": ANALYSIS}),
            predictorId=model["id"],
            predictor=reference(model),
            cohortId=cohort_document["id"],
            cohort=reference(cohort_document),
            overlap={"slideIds": [], "patientIds": [], "patientsComparable": True},
            **({"labelSource": "cohort"} if blind else {}),
        )
        folder = cases.evaluations.jobs.folder(document["id"])
        folder.mkdir(parents=True, exist_ok=True)
        content = json.dumps(_records()).encode()
        path = folder / "predictions.json"
        path.write_bytes(content)
        executions[document["id"]] = {
            "status": "completed",
            "result": {
                "artifacts": {
                    "predictions.json": {
                        "path": str(path),
                        "bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                }
            },
        }
        return document

    monkeypatch.setattr(
        cases.evaluations.jobs, "status", lambda identity, **_: executions[identity]
    )
    references = ReferenceService(store, filesystem)

    def save(cohort_document, name, datasets, field, labels, classes=None):
        selection = {
            "cohortId": cohort_document["id"],
            "name": name,
            "datasetIds": [item["id"] for item in datasets],
            "field": field,
            "classes": classes or BINARY["classes"],
            "labels": labels,
        }
        preview = references.preview(ReferenceSelection.model_validate(selection))
        assert preview["canSave"], preview["findings"]
        return references.save(
            SaveReference.model_validate(
                {**selection, "previewHash": preview["previewHash"], "operationId": name}
            )
        )

    return {
        "store": store,
        "predictor": predictor,
        "cases": cases,
        "references": references,
        "run": run,
        "save": save,
        "original": original,
        "revised": revised,
        "labeled": labeled_cohort,
        "unlabeled": unlabeled_cohort,
    }


def test_a_preview_lists_the_column_values_over_the_cohort_and_counts_what_they_label(study):
    references = study["references"]
    selection = {
        "cohortId": study["unlabeled"]["id"],
        "name": "Reader B",
        "datasetIds": [study["original"]["id"]],
        "field": "readerB",
        "classes": BINARY["classes"],
        "labels": {},
    }
    empty = references.preview(ReferenceSelection.model_validate(selection))
    assert not empty["canSave"]
    assert {item["code"] for item in empty["findings"]} >= {"REFERENCE_NO_LABELS"}
    assert empty["manifest"]["summary"]["values"] == [
        {"value": "D", "count": 4, "label": None},
        {"value": "C", "count": 2, "label": None},
        {"value": "?", "count": 1, "label": None},
    ]
    mapped = references.preview(ReferenceSelection.model_validate({**selection, "labels": MAPPING}))
    summary = mapped["manifest"]["summary"]
    assert mapped["canSave"] and mapped["previewHash"]
    assert summary["labeledSlides"] == 6 and summary["unmappedSlides"] == 1
    assert summary["classCounts"] == {"disease": 4, "clear": 2}
    # p0's slides disagree for reader B; a patient-level score leaves p0 unlabeled.
    assert summary["conflictingPatients"] == 1
    assert {item["code"] for item in mapped["findings"]} == {
        "REFERENCE_UNMAPPED_VALUES",
        "REFERENCE_CONFLICTING_PATIENTS",
    }
    unknown = references.preview(
        ReferenceSelection.model_validate({**selection, "field": "missing", "labels": MAPPING})
    )
    assert not unknown["canSave"] and "REFERENCE_FIELD_UNKNOWN" in {
        item["code"] for item in unknown["findings"]
    }


def test_a_newer_dataset_labels_the_cohort_by_slide_id_and_leaves_the_rest_unlabeled(study):
    saved = study["save"](study["unlabeled"], "Consensus", [study["revised"]], "consensus", MAPPING)
    manifest = saved["manifest"]
    assert manifest["kind"] == "reference-standard" and manifest["matchedBy"] == "slideId"
    assert manifest["datasets"] == [reference(study["revised"])]
    assert manifest["summary"]["unmatchedSlides"] == 1
    assert {row["slideId"]: row["label"] for row in manifest["memberships"]}["s5"] is None
    # A reference is content-addressed: saving the same selection again returns it.
    again = study["save"](study["unlabeled"], "Consensus", [study["revised"]], "consensus", MAPPING)
    assert again["id"] == saved["id"]
    listed = study["references"].list(cohort_id=study["unlabeled"]["id"])["items"]
    assert [item["id"] for item in listed] == [saved["id"]]
    assert "memberships" not in listed[0]["manifest"]


def test_a_reference_repeating_the_cohort_labels_scores_exactly_like_the_cohort(study):
    labeled_run = study["run"]("Scored", study["labeled"], blind=True)
    predicted = study["run"]("Predicted", study["unlabeled"], blind=False, purpose="inference")
    # Reader A agrees with the cohort's labels on every slide.
    reader_a = study["save"](
        study["unlabeled"], "Reader A", [study["original"]], "readerA", MAPPING
    )
    evaluations = study["cases"].evaluations
    cohort_scores = evaluations.get(labeled_run["id"])["execution"]["result"]["metrics"]
    reference_scores = evaluations.reference_metrics(predicted["id"], reader_a["id"])
    for key in ("slide", "patient", "selected", "unit", "classOrder", "decisionThreshold"):
        assert reference_scores[key] == cohort_scores[key]
    assert reference_scores["reference"] == {"id": reader_a["id"], "name": "Reader A"}
    assert reference_scores["scoredBy"]["reference"] == reference(reader_a)
    # Kept on disk under its own key, beside the run's other scores.
    kept = list((evaluations.jobs.folder(predicted["id"]) / REFERENCE_SCORES).glob("*.json"))
    assert len(kept) == 1
    _SCORES.clear()
    assert evaluations.reference_metrics(predicted["id"], reader_a["id"]) == reference_scores


def test_a_reference_must_label_the_runs_cohort_with_its_classes(study):
    predicted = study["run"]("Predicted", study["unlabeled"], blind=False, purpose="inference")
    other = study["save"](study["labeled"], "Other cohort", [study["original"]], "readerA", MAPPING)
    evaluations = study["cases"].evaluations
    with pytest.raises(StorageError) as error:
        evaluations.reference_metrics(predicted["id"], other["id"])
    assert error.value.code == "REFERENCE_COHORT_MISMATCH"
    classes = study["save"](
        study["unlabeled"],
        "Three classes",
        [study["original"]],
        "readerA",
        {"D": "disease", "C": "clear"},
        classes=["disease", "clear", "unsure"],
    )
    with pytest.raises(StorageError) as error:
        evaluations.reference_metrics(predicted["id"], classes["id"])
    assert error.value.code == "REFERENCE_CLASSES_MISMATCH"


def test_downloads_against_a_reference_carry_its_labels_in_the_runs_own_table_form(study):
    predicted = study["run"]("Predicted", study["unlabeled"], blind=False, purpose="inference")
    reader_b = study["save"](
        study["unlabeled"], "Reader B", [study["original"]], "readerB", MAPPING
    )
    evaluations = study["cases"].evaluations
    slides = list(
        csv.DictReader(
            io.StringIO(
                evaluations.reference_download(
                    predicted["id"], reader_b["id"], "slide-predictions.csv"
                ).decode()
            )
        )
    )
    rows = {row["slideId"]: row for row in slides}
    assert rows["s2"]["label"] == "disease" and rows["s2"]["scored"] == "yes"
    # The unmapped value and p0's disagreeing slides are predicted but never scored.
    assert rows["s5"]["label"] == "" and rows["s5"]["scored"] == "no"
    assert rows["s0"]["label"] == "" and rows["s0"]["scored"] == "no"
    patients = list(
        csv.DictReader(
            io.StringIO(
                evaluations.reference_download(
                    predicted["id"], reader_b["id"], "patient-predictions.csv"
                ).decode()
            )
        )
    )
    assert sum(row["scored"] == "yes" for row in patients) == 4
    metrics = json.loads(
        evaluations.reference_download(predicted["id"], reader_b["id"], "metrics.json")
    )
    assert metrics == evaluations.reference_metrics(predicted["id"], reader_b["id"])
    with pytest.raises(StorageError) as error:
        evaluations.reference_download(predicted["id"], reader_b["id"], "predictions.json")
    assert error.value.code == "EVALUATION_ARTIFACT_NOT_FOUND"


def test_patients_whose_slides_disagree_stay_unlabeled_under_a_patient_target(study):
    reader_b = study["save"](
        study["unlabeled"], "Reader B", [study["original"]], "readerB", MAPPING
    )
    records = _records()["records"]
    manifest = {"target": BINARY}
    slides, conflicting = join_reference(manifest, reader_b["manifest"], records)
    assert conflicting == 1
    assert {row["slideId"] for row in slides if row["labelIndex"] is None} == {"s0", "s0b", "s5"}
    predicted = study["run"]("Predicted", study["unlabeled"], blind=False, purpose="inference")
    scores = study["cases"].evaluations.reference_metrics(predicted["id"], reader_b["id"])
    assert scores["conflictingPatients"] == 1
    assert scores["patient"]["count"] == 4


def test_cases_subgroups_and_clinical_utility_follow_the_chosen_reference(study):
    predicted = study["run"]("Predicted", study["unlabeled"], blind=False, purpose="inference")
    reader_a = study["save"](
        study["unlabeled"], "Reader A", [study["original"]], "readerA", MAPPING
    )
    cases = study["cases"]
    page = cases.query(
        predicted["id"], CaseReviewQuery(unit="slide", outcome="error", referenceId=reader_a["id"])
    )
    assert page["purpose"] == "evaluation" and page["reference"]["name"] == "Reader A"
    # Decisions at threshold 0.25: s2 is called disease against a clear label.
    assert [item["id"] for item in page["items"]] == ["s2"]
    unlabeled = cases.query(predicted["id"], CaseReviewQuery(unit="slide"))
    assert unlabeled["purpose"] == "inference" and unlabeled["reference"] is None
    performance = RunPerformanceService(cases.store, cases.evaluations.filesystem)
    performance.cases = cases
    with pytest.raises(StorageError) as error:
        performance.breakdown(predicted["id"], PerformanceBreakdownQuery(attribute="site"))
    assert error.value.code == "PERFORMANCE_REQUIRES_LABELS"
    breakdown = performance.breakdown(
        predicted["id"],
        PerformanceBreakdownQuery(unit="slide", attribute="site", referenceId=reader_a["id"]),
    )
    assert [row["value"] for row in breakdown["rows"]] == ["A", "B"]
    assert breakdown["overall"]["count"] == 6
    clinical = ClinicalService(cases.store, cases.evaluations.filesystem)
    clinical.evaluations = cases.evaluations
    preview = clinical.preview(
        ClinicalSelection(evaluationId=predicted["id"], referenceId=reader_a["id"])
    )
    assert preview["canSave"], preview["findings"]
    manifest = preview["manifest"]
    assert manifest["source"]["labelSource"] == "reference"
    assert manifest["source"]["reference"] == reference(reader_a)
    assert manifest["report"]["counts"]["labeled"] == 5
    without = clinical.preview(ClinicalSelection(evaluationId=predicted["id"]))
    assert not without["canSave"]


def test_agreement_pairs_the_run_with_every_label_source_of_its_cohort(study):
    labeled_run = study["run"]("Scored", study["labeled"], blind=True)
    reader_a = study["save"](study["labeled"], "Reader A", [study["original"]], "readerA", MAPPING)
    reader_b = study["save"](study["labeled"], "Reader B", [study["original"]], "readerB", MAPPING)
    result = study["cases"].evaluations.agreement(labeled_run["id"], "slide")
    assert [item["id"] for item in result["sources"]] == [
        "run",
        "cohort",
        reader_a["id"],
        reader_b["id"],
    ]
    assert result["sources"][1]["labeled"] == 6
    pairs = {(item["left"], item["right"]): item for item in result["pairs"]}
    assert len(pairs) == 6
    # Reader A repeats the cohort labels exactly.
    assert pairs[("cohort", reader_a["id"])]["agreement"] == 1.0
    assert pairs[("cohort", reader_a["id"])]["count"] == 6
    # The run against the cohort: decisions at 0.25 are wrong only on s2.
    assert pairs[("run", "cohort")]["disagreements"] == 1
    patient = study["cases"].evaluations.agreement(labeled_run["id"], "patient")
    assert patient["unit"] == "patient"
    # Reader B's p0 disagrees with itself, so it cannot label that patient.
    assert {item["id"]: item.get("labeled") for item in patient["sources"]}[reader_b["id"]] == 4


def test_weighted_kappa_reads_the_class_order_as_an_ordered_scale():
    matrix = [[5, 1, 0], [1, 3, 1], [0, 1, 4]]
    # Linear weights |i - j| / 2: observed 0.125 against expected 0.44921875.
    assert weighted_kappa(matrix) == pytest.approx(1 - 0.125 / 0.44921875)
    assert weighted_kappa([[3, 1], [0, 2]]) is None
    left = [0, 0, 1, 2]
    right = [0, 1, 1, 2]
    three = agreement(left, right, ["low", "mid", "high"])
    assert "weightedKappa" in three
    assert "weightedKappa" not in agreement([0, 1], [0, 1], ["a", "b"])


def development_oof(calls):
    """Overconfident out-of-fold predictions of 40 development slides, for any seed group."""

    def verified(_store, batch_id, candidate_id, training_seed, split_seed):
        calls.append((batch_id, candidate_id, training_seed, split_seed))
        slides = []
        for index in range(40):
            label = index % 2
            # Right 80% of the time, but always 95% sure.
            right = index % 5 != 0
            disease = 0.95 if (label == 0) == right else 0.05
            slides.append(
                {
                    "slideId": f"d{index}",
                    "patientId": f"dp{index}",
                    "patientIdSource": "source",
                    "label": BINARY["classes"][label],
                    "labelIndex": label,
                    "probabilities": [disease, 1 - disease],
                }
            )
        return {
            "slides": slides,
            "target": BINARY,
            "analysisInputHash": f"{training_seed}/{split_seed}",
        }

    return verified


def test_recalibration_fits_on_development_predictions_and_reports_the_cohort(study, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "histopilot.application.training_exports.verified_oof", development_oof(calls)
    )
    labeled_run = study["run"]("Scored", study["labeled"], blind=True)
    evaluations = study["cases"].evaluations
    report = evaluations.recalibration(labeled_run["id"], "slide")
    assert calls == [("batch", "candidate", 7, 42)]
    assert report["method"] == "platt" and report["unit"] == "slide"
    # Development predictions are twice as sure as they are right: the map softens them.
    assert 0 < report["parameters"]["slope"] < 1
    assert report["development"]["units"] == 40
    development = report["development"]
    assert development["recalibrated"]["ece"] < development["original"]["ece"]
    # The cohort's six labeled slides; s3 has no label.
    assert report["cohort"]["units"] == 6
    assert report["cohort"]["original"]["count"] == report["cohort"]["recalibrated"]["count"] == 6
    assert report["developmentSources"] == [
        {
            "batchId": "batch",
            "candidateId": "candidate",
            "trainingSeed": 7,
            "splitSeed": 42,
            "analysisInputHash": "7/42",
        }
    ]
    # Kept beside the run and served again without refitting.
    _SCORES.clear()
    assert evaluations.recalibration(labeled_run["id"], "slide") == report
    assert len(calls) == 1
    kept = list((evaluations.jobs.folder(labeled_run["id"]) / "recalibration").glob("*.json"))
    assert len(kept) == 1


def test_recalibration_takes_labels_from_a_reference_and_needs_labels(study, monkeypatch):
    monkeypatch.setattr("histopilot.application.training_exports.verified_oof", development_oof([]))
    predicted = study["run"]("Predicted", study["unlabeled"], blind=False, purpose="inference")
    evaluations = study["cases"].evaluations
    with pytest.raises(StorageError) as error:
        evaluations.recalibration(predicted["id"])
    assert error.value.code == "RECALIBRATION_REQUIRES_LABELS"
    reader_a = study["save"](
        study["unlabeled"], "Reader A", [study["original"]], "readerA", MAPPING
    )
    report = evaluations.recalibration(predicted["id"], "patient", reader_a["id"])
    assert report["reference"] == {"id": reader_a["id"], "name": "Reader A"}
    assert report["unit"] == "patient" and report["cohort"]["units"] == 5


def test_a_seed_ensemble_is_recalibrated_on_its_averaged_seed_groups(study, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "histopilot.application.training_exports.verified_oof", development_oof(calls)
    )
    ensemble = study["store"].publish_configuration(
        manifest={
            **study["predictor"]["manifest"],
            "method": "seed_ensemble",
            "seedGroups": [
                {"trainingSeed": 7, "splitSeed": 42, "runIds": ["a"]},
                {"trainingSeed": 8, "splitSeed": 42, "runIds": ["b"]},
            ],
        },
        operation_id="seed-ensemble",
    )
    run = study["run"]("Seed ensemble", study["labeled"], blind=True, model=ensemble)
    report = study["cases"].evaluations.recalibration(run["id"], "slide")
    assert [call[2] for call in calls] == [7, 8]
    assert report["development"]["seedGroups"] == 2
    assert [item["trainingSeed"] for item in report["developmentSources"]] == [7, 8]


def test_recalibration_needs_a_predictor_with_development_predictions(study):
    legacy = study["store"].publish_configuration(
        manifest={
            key: value
            for key, value in study["predictor"]["manifest"].items()
            if key not in {"batchId", "trainingSeed", "splitSeed"}
        },
        operation_id="legacy-predictor",
    )
    run = study["run"]("Legacy", study["labeled"], blind=True, model=legacy)
    with pytest.raises(StorageError) as error:
        study["cases"].evaluations.recalibration(run["id"])
    assert error.value.code == "RECALIBRATION_UNAVAILABLE"
