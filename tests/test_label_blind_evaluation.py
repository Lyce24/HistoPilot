"""Labeled evaluations are predicted label-blind and scored by the control service.

Each label-blind run is checked against a twin saved the earlier way, whose worker read
the cohort labels and stored them beside every prediction: both must answer every
question (metrics, case review, comparisons, downloads) identically.
"""

import copy
import csv
import hashlib
import io
import json
import runpy
from pathlib import Path

import pytest
from pydantic import ValidationError
from support.projects import TARGET, codes, dataset, preview, setup

from histopilot.application.case_review import CaseReviewService
from histopilot.application.evaluation_runs import _SCORES, SCORES_FILE
from histopilot.application.predictors import reference
from histopilot.application.run_evidence import join_labels, labels_withheld, strip_labels
from histopilot.application.run_metrics import run_metrics
from histopilot.application.run_performance import RunPerformanceService
from histopilot.cv_summary import evaluation_metrics
from histopilot.schemas.case_review import CaseReviewQuery
from histopilot.schemas.performance import PerformanceBreakdownQuery
from histopilot.schemas.predictors import CompareEvaluations
from histopilot.schemas.protocols import TargetSpec
from histopilot.schemas.target_splits import TargetSplitSpec
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
# slide, patient, cohort label, positive-class probability of the two fold members
SLIDES = [
    ("s0", "p0", "disease", (0.3, 0.5)),
    ("s0b", "p0", "disease", (0.4, 0.2)),
    ("s1", "p1", "clear", (0.1, 0.3)),
    ("s2", "p2", "clear", (0.8, 0.6)),
    ("s3", "p3", None, (0.9, 0.7)),
    ("s4", "p4", "disease", (0.7, 0.9)),
    ("s5", "p5", "clear", (0.2, 0.1)),
]


SITES = {"s0": "A", "s0b": "A", "s1": "A", "s2": "B", "s3": "B", "s4": "C", "s5": "C"}


def _records(labeled, shift=0.0):
    rows = []
    for slide, patient, label, members in SLIDES:
        members = [min(0.99, max(0.01, value + shift)) for value in members]
        mean = sum(members) / len(members)
        rows.append(
            {
                "slideId": slide,
                "patientId": patient,
                "patientIdSource": "source",
                "label": label if labeled else None,
                "labelIndex": BINARY["classes"].index(label) if labeled and label else None,
                "probabilities": [mean, 1 - mean],
                "memberProbabilities": [[value, 1 - value] for value in members],
            }
        )
    return rows


def _predictions(labeled, shift=0.0):
    records = _records(labeled, shift)
    return {
        "classOrder": BINARY["classes"],
        "records": records,
        "patientRecords": patient_predictions(records, "mean"),
    }


@pytest.fixture
def runs(tmp_path, monkeypatch):
    _SCORES.clear()
    store = ScientificStore(tmp_path, "label-blind")
    draft = store.create_draft("import", "Test", {})
    metadata = [
        {
            "slideId": slide,
            "patientId": patient,
            "attributes": {"site": SITES[slide]},
            "slidePath": None,
        }
        for slide, patient, _label, _members in SLIDES
    ]
    data = store.publish_dataset(
        draft["id"],
        expected_revision=1,
        manifest={"kind": "dataset", "name": "Test", "dictionary": [{"key": "site"}]},
        artifacts={"records.json": json.dumps(metadata).encode()},
        operation_id="dataset",
    )

    def publish(kind, operation, **values):
        return store.publish_configuration(
            manifest={"kind": kind, "datasetId": data["id"], **values}, operation_id=operation
        )

    predictor = publish(
        "frozen-predictor",
        "predictor",
        target=BINARY,
        method="ensemble",
        checkpoints=[{"runId": "a"}, {"runId": "b"}],
        aggregation="mean_probability",
    )
    memberships = [
        {"slideId": slide, "patientId": patient, "patientIdSource": "source", "label": label}
        for slide, patient, label, _members in SLIDES
    ]
    cohort = publish("evaluation-cohort", "cohort", target=BINARY, memberships=memberships)
    service = CaseReviewService(store, LocalFilesystem((tmp_path,)))
    executions = {}

    def run(name, *, blind, overlap=None, predictions=None, purpose=None):
        manifest = {
            **({"purpose": purpose} if purpose else {}),
            "name": name,
            "target": BINARY,
            "inference": INFERENCE,
            "analysis": ANALYSIS,
            "predictorId": predictor["id"],
            "predictor": reference(predictor),
            "cohortId": cohort["id"],
            "cohort": reference(cohort),
            "overlap": overlap or {"slideIds": [], "patientIds": [], "patientsComparable": True},
            **({"labelSource": "cohort"} if blind else {}),
        }
        document = publish("model-evaluation", name, **manifest)
        folder = service.evaluations.jobs.folder(document["id"])
        folder.mkdir(parents=True, exist_ok=True)
        content = json.dumps(predictions or _predictions(labeled=not blind)).encode()
        path = folder / "predictions.json"
        path.write_bytes(content)
        artifact = {
            "path": str(path),
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        executions[document["id"]] = {
            "status": "completed",
            "result": {"artifacts": {"predictions.json": artifact}},
        }
        return document

    monkeypatch.setattr(
        service.evaluations.jobs, "status", lambda identity, **kwargs: executions[identity]
    )
    return service, run, cohort


def test_label_blind_run_scores_exactly_like_its_labeled_twin(runs):
    service, run, _cohort = runs
    earlier = run("Earlier", blind=False)
    blind = run("Blind", blind=True)
    evaluations = service.evaluations
    metrics = evaluations.get(blind["id"])["execution"]["result"]["metrics"]
    # The earlier run's worker would have scored its labeled rows exactly this way.
    labeled = _records(labeled=True)
    patients = patient_predictions(labeled, "mean")
    assert metrics["slide"] == evaluation_metrics(labeled, BINARY, 0.25)
    assert metrics["patient"]["count"] == evaluation_metrics(patients, BINARY, 0.25)["count"]
    assert metrics["selected"] == metrics["patient"]
    assert metrics["slide"]["unlabeledCount"] == 1
    assert metrics["patientAnalysis"]["policy"]["bootstrapResamples"] == 200
    assert metrics["patient"]["confidenceIntervals"]
    assert metrics["scoredBy"]["cohort"] == reference(_cohort)
    assert "developmentExcluded" not in metrics
    # Scoring is read-only and deterministic: a second read gives the same document.
    _SCORES.clear()
    assert evaluations.get(blind["id"])["execution"]["result"]["metrics"] == metrics
    # An earlier run keeps the worker's result untouched: nothing is scored for it.
    assert "metrics" not in evaluations.get(earlier["id"])["execution"]["result"]


def test_scored_documents_are_kept_and_lists_never_wait_for_a_bootstrap(runs):
    service, run, _cohort = runs
    blind = run("Blind", blind=True)
    evaluations = service.evaluations
    kept = evaluations.jobs.folder(blind["id"]) / SCORES_FILE
    [listed] = [row for row in evaluations.list()["items"] if row["id"] == blind["id"]]
    quick = listed["execution"]["result"]["metrics"]
    assert quick["analysisPending"] is True and "patientAnalysis" not in quick
    assert not kept.exists()

    # A list that polls answers from memory instead of reading the predictions again.
    def unexpected(*_arguments, **_options):
        raise AssertionError("The run was scored again.")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(evaluations, "_score", unexpected)
        [again] = [row for row in evaluations.list()["items"] if row["id"] == blind["id"]]
    assert again["execution"]["result"]["metrics"] == quick
    full = evaluations.get(blind["id"])["execution"]["result"]["metrics"]
    assert quick["slide"] == full["slide"] and quick["patient"]["count"] == full["patient"]["count"]
    assert "analysisPending" not in full and kept.exists()
    # After a restart the kept document is read, bootstrap included.
    _SCORES.clear()
    [listed] = [row for row in evaluations.list()["items"] if row["id"] == blind["id"]]
    assert listed["execution"]["result"]["metrics"] == full
    # A damaged document is scored again rather than trusted.
    damaged = json.loads(kept.read_text())
    damaged["metrics"]["slide"]["auroc"] = 1.0
    kept.write_text(json.dumps(damaged))
    _SCORES.clear()
    assert evaluations.get(blind["id"])["execution"]["result"]["metrics"] == full


def test_both_generations_review_the_same_cases_and_outcomes(runs):
    service, run, _cohort = runs
    earlier, blind = run("Earlier", blind=False), run("Blind", blind=True)
    for unit in ("slide", "patient"):
        query = CaseReviewQuery(unit=unit, outcome="all", limit=50)
        left, right = service.query(earlier["id"], query), service.query(blind["id"], query)
        assert [
            (row["id"], row["label"], row["outcome"], row["predictedLabel"])
            for row in left["items"]
        ] == [
            (row["id"], row["label"], row["outcome"], row["predictedLabel"])
            for row in right["items"]
        ]
        assert left["summary"] == right["summary"]


def test_downloads_join_the_cohort_labels(runs):
    service, run, _cohort = runs
    blind = run("Blind", blind=True)
    evaluations = service.evaluations
    metrics = json.loads(evaluations.download(blind["id"], "metrics.json"))
    assert metrics == evaluations.get(blind["id"])["execution"]["result"]["metrics"]
    slides = list(
        csv.DictReader(
            io.StringIO(evaluations.download(blind["id"], "slide-predictions.csv").decode())
        )
    )
    assert [row["slideId"] for row in slides] == [slide for slide, *_ in SLIDES]
    assert {row["slideId"]: row["label"] for row in slides}["s3"] == ""
    assert {row["slideId"]: row["scored"] for row in slides} == {
        slide: "no" if label is None else "yes" for slide, _patient, label, _members in SLIDES
    }
    assert {row["memberCount"] for row in slides} == {"2"}
    patients = list(
        csv.DictReader(
            io.StringIO(evaluations.download(blind["id"], "patient-predictions.csv").decode())
        )
    )
    assert {row["patientId"]: row["label"] for row in patients} == {
        "p0": "disease",
        "p1": "clear",
        "p2": "clear",
        "p3": "",
        "p4": "disease",
        "p5": "clear",
    }
    # The worker's label-free file stays available unchanged.
    assert (
        json.loads(evaluations.download(blind["id"], "predictions.json"))["records"][0]["label"]
        is None
    )


def test_label_blind_runs_refuse_stored_labels(runs):
    service, run, _cohort = runs
    tampered = run("Tampered", blind=True, predictions=_predictions(labeled=True))
    result = service.evaluations.get(tampered["id"])["execution"]["result"]
    assert "metrics" not in result
    assert result["metricsError"]["code"] == "EVALUATION_EVIDENCE_INVALID"
    with pytest.raises(Exception, match="label-blind run stored labels"):
        service.query(tampered["id"], CaseReviewQuery(unit="slide", outcome="all"))


def test_development_patients_are_predicted_but_never_scored(runs):
    service, run, _cohort = runs
    overlap = {"slideIds": [], "patientIds": ["p2"], "patientsComparable": True}
    blind = run("Blind", blind=True, overlap=overlap)
    metrics = service.evaluations.get(blind["id"])["execution"]["result"]["metrics"]
    labeled = [row for row in _records(labeled=True) if row["patientId"] != "p2"]
    assert metrics["slide"] == evaluation_metrics(labeled, BINARY, 0.25)
    assert metrics["developmentExcluded"]["slides"] == 1
    assert metrics["developmentExcluded"]["patients"] == 1
    assert metrics["patientAnalysis"]["uncertainty"]["patientCount"] == 4
    # Their predictions stay visible and are flagged in every table.
    review = service.query(blind["id"], CaseReviewQuery(unit="slide", outcome="all", limit=50))
    assert {row["id"] for row in review["items"]} >= {"s2"}
    table = list(
        csv.DictReader(
            io.StringIO(service.evaluations.download(blind["id"], "slide-predictions.csv").decode())
        )
    )
    assert {row["slideId"]: (row["developmentPatient"], row["scored"]) for row in table}["s2"] == (
        "yes",
        "no",
    )


def test_paired_comparison_joins_labels_for_both_generations(runs):
    service, run, _cohort = runs
    earlier = run("Earlier", blind=False)
    blind = run("Blind", blind=True, predictions=_predictions(labeled=False, shift=0.05))
    compared = service.evaluations.compare(
        CompareEvaluations(leftEvaluationId=earlier["id"], rightEvaluationId=blind["id"])
    )
    assert compared["statistics"]["patientCount"] == 5


def test_performance_breaks_down_by_attribute_over_the_scored_records(runs):
    service, run, _cohort = runs
    overlap = {"slideIds": [], "patientIds": ["p5"], "patientsComparable": True}
    blind = run("Blind", blind=True, overlap=overlap)
    performance = RunPerformanceService(service.store, service.evaluations.filesystem)
    performance.cases = service  # the fixture's case review reads its faked job results
    result = performance.breakdown(
        blind["id"], PerformanceBreakdownQuery(unit="slide", attribute="site")
    )
    scored = [row for row in _records(labeled=True) if row["patientId"] != "p5"]
    by_site = {row["value"]: row for row in result["rows"]}
    for site in ("A", "B", "C"):
        rows = [row for row in scored if SITES[row["slideId"]] == site]
        assert {
            key: value for key, value in by_site[site].items() if key != "value"
        } == evaluation_metrics(rows, BINARY, 0.25)
    # Groups are ordered by labeled count; the development patient's slide is never scored.
    assert [row["value"] for row in result["rows"]] == ["A", "B", "C"]
    assert result["developmentExcluded"] == 1
    assert result["overall"] == evaluation_metrics(scored, BINARY, 0.25)
    with pytest.raises(StorageError) as error:
        performance.breakdown(blind["id"], PerformanceBreakdownQuery(attribute="unknown"))
    assert error.value.code == "PERFORMANCE_ATTRIBUTE_INVALID"
    unlabeled = run(
        "Unlabeled", blind=False, purpose="inference", predictions=_predictions(labeled=False)
    )
    with pytest.raises(StorageError) as error:
        performance.breakdown(unlabeled["id"], PerformanceBreakdownQuery(attribute="site"))
    assert error.value.code == "PERFORMANCE_REQUIRES_LABELS"


def test_label_join_contract():
    manifest = {"target": BINARY, "labelSource": "cohort"}
    cohort = {
        "memberships": [
            {"slideId": slide, "patientId": patient, "label": label}
            for slide, patient, label, _ in SLIDES
        ]
    }
    joined = join_labels(manifest, cohort, _records(labeled=False))
    assert [row["labelIndex"] for row in joined] == [0, 0, 1, 1, None, 0, 1]
    assert labels_withheld(manifest) and not labels_withheld({"target": BINARY})
    with pytest.raises(ValueError, match="stored labels"):
        join_labels(manifest, cohort, _records(labeled=True))
    with pytest.raises(ValueError, match="membership"):
        join_labels(manifest, cohort, _records(labeled=False)[:-1])
    # Earlier runs must have stored exactly the cohort's labels.
    changed = _records(labeled=True)
    changed[0]["label"] = "clear"
    with pytest.raises(ValueError, match="labels"):
        join_labels({"target": BINARY}, cohort, changed)
    assert strip_labels(cohort["memberships"])[0] == {"slideId": "s0", "patientId": "p0"}


def test_slide_split_metrics_score_slides_only():
    target = {**BINARY, "unit": "slide"}
    manifest = {"target": target, "inference": INFERENCE, "splitUnit": "slide"}
    labeled = [
        {**row, "labelIndex": None if row["label"] is None else row["labelIndex"]}
        for row in _records(labeled=True)
    ]
    metrics = run_metrics(manifest, labeled)
    assert metrics["selected"] == metrics["slide"] == evaluation_metrics(labeled, target, 0.25)
    assert metrics["patient"]["available"] is False
    assert metrics["patientAggregation"] is None
    assert "patientAnalysis" not in metrics


def test_only_testing_targets_keep_slides_unlabeled():
    unlabeled = {**TARGET, "missing": "unlabeled"}
    assert TargetSpec.model_validate(unlabeled).keeps_unlabeled
    spec = {"datasetId": "dataset-" + "0" * 64, "target": TARGET, "testTarget": unlabeled}
    assert TargetSplitSpec.model_validate(spec).testTarget.missing == "unlabeled"
    with pytest.raises(ValidationError, match="Training targets need a label"):
        TargetSplitSpec.model_validate({**spec, "target": unlabeled, "testTarget": None})


def test_testing_cohort_keeps_unlabeled_slides_as_unscored_members(tmp_path):
    folder = tmp_path / "project"
    folder.mkdir()
    service, spec, _ = setup(ScientificStore(folder, "project-unlabeled"), tmp_path)
    data, _rows = dataset(
        service.store,
        "partly-labeled",
        [
            {"slideId": "t0", "patientId": "q0", "attributes": {"label": "0", "cohort": "test"}},
            {"slideId": "t1", "patientId": "q1", "attributes": {"label": None, "cohort": "test"}},
            {
                "slideId": "t2",
                "patientId": "q2",
                "attributes": {"label": "unknown", "cohort": "test"},
            },
        ],
    )
    spec.update(
        datasetId=data["id"], featureBundleId=None, protocolId=None, developmentFeatureBundleId=None
    )
    blocked = preview(service, spec)
    assert {"MISSING_TARGET_LABEL", "UNMAPPED_TARGET_LABEL"} <= codes(blocked)
    kept = preview(
        service, {**spec, "target": {**TARGET, "missing": "unlabeled", "unmapped": "unlabeled"}}
    )
    assert kept["canFreeze"], kept["findings"]
    assert [row["label"] for row in kept["memberships"]] == ["low", None, None]
    assert kept["summary"]["includedSlides"] == 3 and kept["summary"]["labeledSlides"] == 1
    assert "UNLABELED_TEST_SLIDES" in {item["code"] for item in kept["findings"]}


def test_label_blind_worker_predictions_score_exactly_like_the_labeled_worker(tmp_path):
    """Real tiny checkpoints: the service scores label-blind predictions exactly as the
    worker scored the same predictions with labels, bootstrap analysis included."""
    pytest.importorskip("torch")
    pytest.importorskip("lightning")
    from histopilot.training.inference import evaluate

    support = runpy.run_path(str(Path(__file__).with_name("test_inference_execution.py")))
    plan = support["_evaluation_plan"](tmp_path)
    plan.update(method="ensemble", checkpoints=plan["checkpoints"] * 2, analysis=ANALYSIS)
    labeled = evaluate(copy.deepcopy(plan), tmp_path / "labeled")
    blind_plan = copy.deepcopy(plan)
    blind_plan["labelsWithheld"] = True
    blind_plan["data"]["memberships"] = strip_labels(plan["data"]["memberships"])
    blind = evaluate(blind_plan, tmp_path / "blind")
    assert blind["purpose"] == "evaluation" and blind["labelsWithheld"] is True
    assert "metrics" not in blind and blind["summary"]["purpose"] == "evaluation"
    predictions = json.loads((tmp_path / "blind/predictions.json").read_text())
    assert all(
        row["label"] is None and len(row["memberProbabilities"]) == 2
        for row in predictions["records"]
    )
    manifest = {
        "target": plan["target"],
        "inference": plan["inference"],
        "analysis": ANALYSIS,
        "labelSource": "cohort",
    }
    slides = join_labels(
        manifest, {"memberships": plan["data"]["memberships"]}, predictions["records"]
    )
    scored = run_metrics(manifest, slides)
    for key in (
        "slide",
        "patient",
        "selected",
        "patientAnalysis",
        "unit",
        "classOrder",
        "decisionThreshold",
    ):
        assert scored[key] == labeled["metrics"][key], key
    with pytest.raises(ValueError, match="Label-blind evaluations"):
        evaluate({**blind_plan, "data": plan["data"]}, tmp_path / "refused")
    with pytest.raises(ValueError, match="withhold"):
        evaluate({**blind_plan, "purpose": "inference"}, tmp_path / "inference")
    with pytest.raises(ValueError, match="withhold"):
        evaluate({**blind_plan, "labelsWithheld": 1}, tmp_path / "integer")
