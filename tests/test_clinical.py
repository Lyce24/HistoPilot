"""Clinical quantities, source authority, grouping, and immutable report exports."""

import copy
import hashlib
import json
import math
import subprocess
import sys

import pytest
from pydantic import ValidationError

from histopilot.application.clinical import ClinicalService, clinical_report
from histopilot.application.predictors import reference
from histopilot.schemas.clinical import ClinicalSelection, SaveClinicalAnalysis
from histopilot.storage.filesystem import LocalFilesystem
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import ScientificStore

EVALUATION_ID = "configuration-" + "a" * 64
TARGET = {
    "field": "diagnosis",
    "task": "binary_classification",
    "unit": "patient",
    "classes": ["disease", "clear"],
    "labels": {"yes": "disease", "no": "clear"},
    "positiveClass": "disease",
}
INFERENCE = {"decisionThreshold": 0.5, "patientAggregation": "mean"}


def predictions():
    rows = [
        {
            "slideId": f"s{index}",
            "patientId": f"p{index}",
            "labelIndex": label,
            "label": TARGET["classes"][label],
            "probabilities": [score, 1 - score],
        }
        for index, (label, score) in enumerate([(1, 0.1), (1, 0.4), (0, 0.35), (0, 0.8)])
    ]
    patients = [{**row, "slideIds": [row["slideId"]]} for row in rows]
    return {"classOrder": TARGET["classes"], "records": rows, "patientRecords": patients}


def report(value=None, *, target=None, inference=None, **kwargs):
    return clinical_report(
        value or predictions(),
        target or TARGET,
        inference or INFERENCE,
        ClinicalSelection(evaluationId=EVALUATION_ID, **kwargs),
    )


def test_known_clinical_statistics_use_frozen_positive_class_and_not_class_index_one():
    value = report(bins=2)
    assert value["unit"] == "patient"
    assert value["metrics"]["brierScore"] == pytest.approx(0.158125)
    assert value["metrics"]["brierReference"] == 0.25
    assert value["metrics"]["brierSkillScore"] == pytest.approx(0.3675)
    assert value["metrics"]["rocAuc"] == 0.75
    assert value["metrics"]["averagePrecision"] == pytest.approx(5 / 6)
    assert value["metrics"]["ece"] == pytest.approx(0.0875)
    assert value["metrics"]["mce"] == pytest.approx(0.2)
    assert value["metrics"]["logLoss"] == pytest.approx(-math.log(0.9 * 0.6 * 0.35 * 0.8) / 4)
    assert value["metrics"]["observedExpectedRatio"] == pytest.approx(2 / 1.65)
    operating = value["operatingPoint"]
    assert [operating[key] for key in ("tp", "fp", "tn", "fn")] == [1, 0, 2, 1]
    assert operating["sensitivity"] == 0.5
    assert operating["specificity"] == operating["ppv"] == 1
    assert operating["npv"] == operating["f1"] == pytest.approx(2 / 3)
    assert operating["positiveLikelihoodRatio"] is None
    assert operating["negativeLikelihoodRatio"] == 0.5
    assert operating["netBenefit"] == 0.25
    assert operating["treatAllNetBenefit"] == operating["treatNoneNetBenefit"] == 0
    assert operating["standardizedNetBenefit"] == 0.5
    assert operating["netInterventionsAvoidedPer100"] == 25
    assert operating["highRiskPer100"] == operating["truePositivePer100"] == 25
    assert value["uncertainty"]["method"] == "wilson_95"
    interval = value["uncertainty"]["operatingPoint"]["sensitivity"]
    assert interval == pytest.approx({"lower": 0.09453120573423074, "upper": 0.9054687942657693})
    assert value["thresholdSource"] == "frozen_evaluation"
    json.dumps(value, allow_nan=False)


def test_analysis_threshold_is_descriptive_and_does_not_mutate_frozen_inference():
    source = predictions()
    original = copy.deepcopy(source)
    value = report(source, threshold=0.35, thresholdMin=0.1, thresholdMax=0.9, thresholdSteps=5)
    assert value["operatingPoint"]["tp"] == 2
    assert value["operatingPoint"]["fp"] == 1
    assert value["frozenDecisionThreshold"] == 0.5
    assert value["thresholdSource"] == "descriptive_override"
    assert len(value["operatingCurve"]) == 5
    assert value["operatingCurve"][0]["threshold"] == 0.1
    assert value["operatingCurve"][-1]["threshold"] == 0.9
    assert source == original
    assert INFERENCE["decisionThreshold"] == 0.5


def test_tied_scores_receive_half_credit_for_roc_and_grouped_average_precision():
    source = predictions()
    for row in source["records"]:
        row["probabilities"] = [0.5, 0.5]
    value = report(source, unit="slide")
    assert value["metrics"]["rocAuc"] == value["metrics"]["averagePrecision"] == 0.5
    assert len(value["rocCurve"]) == 2
    assert value["operatingPoint"]["predictedPositive"] == 4


@pytest.mark.parametrize("label", [0, 1])
def test_single_class_reports_undefined_metrics_as_null(label):
    source = predictions()
    for row in source["records"]:
        row.update(labelIndex=label, label=TARGET["classes"][label])
    value = report(source, unit="slide")
    assert value["metrics"]["rocAuc"] is None
    assert value["metrics"]["brierSkillScore"] is None
    if label == 1:
        assert value["metrics"]["averagePrecision"] is None
        assert value["operatingPoint"]["sensitivity"] is None
    else:
        assert value["metrics"]["averagePrecision"] == 1
        assert value["operatingPoint"]["specificity"] is None
    json.dumps(value, allow_nan=False)


def test_unlabeled_rows_do_not_change_any_clinical_denominator():
    source = predictions()
    source["records"].append(
        {
            "slideId": "unlabeled",
            "patientId": None,
            "labelIndex": None,
            "label": None,
            "probabilities": [0.99, 0.01],
        }
    )
    value = report(source, unit="slide")
    assert value["counts"]["total"] == 5
    assert value["counts"]["labeled"] == 4
    assert value["counts"]["unlabeled"] == value["counts"]["missingPatientIds"] == 1
    assert value["metrics"]["brierScore"] == report()["metrics"]["brierScore"]
    assert value["operatingPoint"]["highRiskPer100"] == 25
    assert value["uncertainty"]["method"] == "unavailable"
    for row in source["records"]:
        row.update(labelIndex=None, label=None)
    with pytest.raises(StorageError) as error:
        report(source, unit="slide")
    assert error.value.code == "CLINICAL_NO_LABELED_OUTCOMES"


def test_patient_analysis_preserves_saved_group_mean_and_suppresses_correlated_slide_intervals():
    source = predictions()
    source["records"].append(
        {
            "slideId": "extra",
            "patientId": "p0",
            "labelIndex": None,
            "label": None,
            "probabilities": [0.7, 0.3],
        }
    )
    source["patientRecords"][0].update(slideIds=["s0", "extra"], probabilities=[0.4, 0.6])
    value = report(source)
    assert value["counts"]["labeled"] == 4
    assert value["counts"]["slides"] == 5
    assert value["metrics"]["brierScore"] == pytest.approx(0.195625)
    assert value["uncertainty"]["method"] == "wilson_95"
    assert report(source, unit="slide")["uncertainty"]["method"] == "unavailable"
    source["patientRecords"][0]["probabilities"] = [0.1, 0.9]
    with pytest.raises(StorageError, match="mean probabilities"):
        report(source)


def test_slide_id_fallbacks_cannot_establish_patient_clinical_independence():
    source = predictions()
    source["records"][0]["patientIdSource"] = "slide_fallback"
    source["records"][0]["patientId"] = source["records"][0]["slideId"]
    source["patientRecords"][0]["patientId"] = source["records"][0]["slideId"]
    with pytest.raises(StorageError) as error:
        report(source)
    assert error.value.code == "CLINICAL_PATIENT_IDENTITIES_UNVERIFIED"
    value = report(source, unit="slide")
    assert value["counts"]["fallbackPatientIds"] == 1
    assert value["counts"]["patients"] == 3
    assert value["uncertainty"]["method"] == "unavailable"
    assert all(value is None for value in value["uncertainty"]["operatingPoint"].values())
    assert any("not verified patients" in warning for warning in value["warnings"])


def test_patient_conflicting_labels_unavailable_and_missing_group_membership_are_rejected():
    source = predictions()
    source["records"][1]["patientId"] = "p2"
    source["patientRecords"] = []
    with pytest.raises(StorageError) as error:
        report(source)
    assert error.value.code == "CLINICAL_PATIENT_UNAVAILABLE"
    assert report(source, unit="slide")["counts"]["labeled"] == 4
    source = predictions()
    source["patientRecords"][0]["slideIds"] = ["other-slide"]
    with pytest.raises(StorageError, match="slide groups"):
        report(source)


def test_multiclass_requires_explicit_ovr_class_and_defines_full_brier_scaling():
    target = {
        **TARGET,
        "task": "multiclass_classification",
        "positiveClass": None,
        "classes": ["A", "B", "C"],
        "labels": {"A": "A", "B": "B", "C": "C"},
    }
    source = {
        "classOrder": target["classes"],
        "records": [
            {
                "slideId": "a",
                "patientId": "a",
                "labelIndex": 0,
                "label": "A",
                "probabilities": [0.7, 0.2, 0.1],
            },
            {
                "slideId": "b",
                "patientId": "b",
                "labelIndex": 2,
                "label": "C",
                "probabilities": [0.1, 0.3, 0.6],
            },
        ],
        "patientRecords": [],
    }
    with pytest.raises(StorageError, match="one-versus-rest"):
        report(source, target=target, unit="slide")
    value = report(source, target=target, unit="slide", positiveClass="A")
    assert value["metrics"]["multiclassBrierScore"] == pytest.approx(0.2)
    assert value["metrics"]["brierScore"] == pytest.approx(0.05)
    assert value["metrics"]["logLoss"] == pytest.approx(-(math.log(0.7) + math.log(0.9)) / 2)
    assert value["metrics"]["multiclassLogLoss"] == pytest.approx(
        -(math.log(0.7) + math.log(0.6)) / 2
    )
    assert value["thresholdSource"] == "descriptive_override"
    with pytest.raises(StorageError, match="frozen positive class"):
        report(positiveClass="clear")


def test_extreme_finite_log_losses_and_legacy_zero_probabilities_are_explicit():
    source = predictions()
    source["records"] = [
        {
            "slideId": "x",
            "patientId": "p",
            "labelIndex": 0,
            "label": "disease",
            "probabilities": [0, 1],
            "logProbabilities": [-1000, 0],
        }
    ]
    value = report(source, unit="slide")
    assert value["metrics"]["logLoss"] == 1000
    del source["records"][0]["logProbabilities"]
    legacy = report(source, unit="slide")
    assert legacy["metrics"]["logLoss"] == pytest.approx(-math.log(1e-300))
    assert any("1e-300" in warning for warning in legacy["warnings"])


def test_extreme_thresholds_and_empty_bins_always_serialize_finite_json():
    source = predictions()
    source["records"][0]["probabilities"] = [1, 0]
    source["records"][2]["probabilities"] = [0, 1]
    value = report(source, unit="slide", bins=50, threshold=5e-324, thresholdMin=5e-324)
    assert value["operatingPoint"]["netInterventionsAvoidedPer100"] is None
    assert value["calibration"][0]["count"] == value["calibration"][-1]["count"] == 1
    assert any(
        row["count"] == 0 and row["observedFraction"] is None for row in value["calibration"]
    )
    json.dumps(value, allow_nan=False)


def test_curve_display_is_bounded_without_approximating_exact_statistics():
    source = {"classOrder": TARGET["classes"], "records": [], "patientRecords": []}
    for index in range(3001):
        score = index / 3000
        label = 0 if index >= 1500 else 1
        source["records"].append(
            {
                "slideId": str(index),
                "patientId": str(index),
                "labelIndex": label,
                "label": TARGET["classes"][label],
                "probabilities": [score, 1 - score],
            }
        )
    value = report(source, unit="slide")
    assert len(value["rocCurve"]) == len(value["precisionRecallCurve"]) == 2001
    assert value["metrics"]["rocAuc"] == pytest.approx(1)
    assert value["metrics"]["averagePrecision"] == pytest.approx(1)


def test_patient_extreme_log_probabilities_must_preserve_slide_aggregation():
    source = predictions()
    for row in source["records"] + source["patientRecords"]:
        row.update(probabilities=[0, 1], logProbabilities=[-1000, 0])
    source["patientRecords"][0]["logProbabilities"] = [-900, 0]
    with pytest.raises(StorageError, match="log probabilities differ"):
        report(source)


@pytest.mark.parametrize(
    "update",
    [
        {"probabilities": [float("nan"), 0.5]},
        {"probabilities": [float("inf"), 0]},
        {"probabilities": [-0.1, 1.1]},
        {"probabilities": [0.2, 0.2]},
        {"probabilities": [True, False]},
        {"labelIndex": True},
        {"labelIndex": 20},
        {"label": "unknown"},
        {"logProbabilities": [0, 0]},
        {"patientId": []},
    ],
)
def test_invalid_prediction_records_are_never_silently_dropped(update):
    source = predictions()
    source["records"][0].update(update)
    with pytest.raises(StorageError):
        report(source, unit="slide")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threshold": 0},
        {"threshold": 1},
        {"threshold": float("nan")},
        {"threshold": True},
        {"bins": 1},
        {"bins": 51},
        {"bins": True},
        {"thresholdSteps": 502},
        {"thresholdMin": 0.7, "thresholdMax": 0.6},
        {"name": " "},
        {"predictions": []},
    ],
)
def test_request_contract_rejects_invalid_parameters_and_authoritative_browser_results(kwargs):
    with pytest.raises(ValidationError):
        ClinicalSelection(evaluationId=EVALUATION_ID, **kwargs)


@pytest.fixture
def service(tmp_path, monkeypatch, request):
    folder = tmp_path / "project"
    folder.mkdir()
    store = ScientificStore(folder, "clinical-project")
    draft = store.create_draft("import", "Test", {})
    dataset = store.publish_dataset(
        draft["id"],
        expected_revision=1,
        manifest={"name": "Test"},
        artifacts={},
        operation_id="dataset",
    )
    experiment = store.create_draft("experiment", "Study", {})

    def publish(kind, **values):
        return store.publish_configuration(
            manifest={"kind": kind, "datasetId": dataset["id"], **values}, operation_id=kind
        )

    predictor = publish("frozen-predictor", target=TARGET, experimentId=experiment["id"])
    source = predictions()
    fallback = getattr(request, "param", None) == "fallback"
    standalone = str(getattr(request, "param", "")).startswith("standalone-")
    independent = getattr(request, "param", None) in {"independent", "standalone-independent"}
    if fallback:
        source["records"][0]["patientId"] = source["records"][0]["slideId"]
        source["patientRecords"][0]["patientId"] = source["records"][0]["slideId"]
    cohort = publish(
        "evaluation-cohort",
        target={**TARGET, "field": "external_diagnosis", "labels": {"1": "disease", "0": "clear"}}
        if standalone
        else TARGET,
        memberships=[
            {
                **{key: row[key] for key in ("slideId", "patientId", "label")},
                **({"patientIdSource": "slide_fallback"} if fallback and index == 0 else {}),
            }
            for index, row in enumerate(source["records"])
        ],
        overlap={
            "slideIds": [],
            "patientIds": [],
            **({"patientsComparable": False} if independent or standalone else {}),
            **({"deferred": True} if standalone else {}),
        },
    )
    evaluation = publish(
        "model-evaluation",
        predictorId=predictor["id"],
        predictor=reference(predictor),
        cohortId=cohort["id"],
        cohort=reference(cohort),
        experimentId=experiment["id"],
        target=TARGET,
        inference=INFERENCE,
        **(
            {"overlap": {"slideIds": [], "patientIds": [], "patientsComparable": not independent}}
            if standalone
            else {}
        ),
    )
    current = ClinicalService(store, LocalFilesystem((tmp_path,)))
    output = current.evaluations.jobs.folder(evaluation["id"])
    output.mkdir(parents=True)
    metrics = {
        "classOrder": TARGET["classes"],
        "unit": TARGET["unit"],
        "positiveClass": "disease",
        "decisionThreshold": 0.5,
        "patientAggregation": "mean_probabilities",
    }
    artifacts = {}
    for name, value in (("predictions.json", source), ("metrics.json", metrics)):
        path = output / name
        content = json.dumps(value).encode()
        path.write_bytes(content)
        artifacts[name] = {
            "path": str(path),
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    execution = {
        "status": "completed",
        "planHash": "b" * 64,
        "result": {"artifacts": artifacts, "inputHash": "c" * 64},
    }
    monkeypatch.setattr(current.evaluations.jobs, "status", lambda *args, **kwargs: execution)
    return current, ClinicalSelection(evaluationId=evaluation["id"]), output, execution


def save(service):
    current, selection, *_ = service
    preview = current.preview(selection)
    assert preview["canSave"], preview
    request = SaveClinicalAnalysis(
        **selection.model_dump(), previewHash=preview["previewHash"], operationId="clinical"
    )
    return current.save(request), request


def test_service_saves_immutable_report_with_all_lineage_and_replays_without_external_artifacts(
    service,
):
    current, selection, folder, _ = service
    saved, request = save(service)
    manifest = saved["manifest"]
    assert manifest["evaluationId"] == selection.evaluationId
    assert manifest["predictorId"] == manifest["predictor"]["id"]
    assert manifest["experimentId"].startswith("draft-")
    assert (
        manifest["source"]["predictionsSha256"]
        == hashlib.sha256((folder / "predictions.json").read_bytes()).hexdigest()
    )
    assert current.list()["items"] == [saved]
    assert current.get(saved["id"]) == saved
    for name in ("predictions.json", "metrics.json"):
        (folder / name).unlink()
    assert current.save(request) == saved
    content, media = current.artifact(saved["id"], "report.json")
    assert media == "application/json"
    assert json.loads(content) == saved
    for name in ("operating-curves.csv", "calibration.csv", "roc.csv", "precision-recall.csv"):
        content, media = current.artifact(saved["id"], name)
        assert media == "text/csv" and len(content.splitlines()) > 2
    with pytest.raises(StorageError) as error:
        current.artifact(saved["id"], "../predictions.json")
    assert error.value.code == "CLINICAL_ARTIFACT_NOT_FOUND"


def test_service_rejects_tampered_or_unfinished_evaluation_evidence(service):
    current, selection, folder, execution = service
    execution["status"] = "running"
    assert current.preview(selection)["findings"][0]["code"] == "EVALUATION_NOT_COMPLETED"
    execution["status"] = "completed"
    path = folder / "predictions.json"
    path.write_bytes(path.read_bytes() + b" ")
    assert current.preview(selection)["findings"][0]["code"] == "EVALUATION_RESULT_CHANGED"


@pytest.mark.parametrize("service", ["fallback"], indirect=True)
def test_clinical_uses_cohort_fallback_provenance_for_legacy_predictions(service):
    current, selection, folder, execution = service
    original = (folder / "predictions.json").read_bytes()
    assert "patientIdSource" not in json.loads(original)["records"][0]
    preview = current.preview(selection)
    assert not preview["canSave"]
    assert preview["findings"][0]["code"] == "CLINICAL_PATIENT_IDENTITIES_UNVERIFIED"
    selection = selection.model_copy(update={"unit": "slide"})
    preview = current.preview(selection)
    assert preview["canSave"]
    assert preview["manifest"]["report"]["uncertainty"]["method"] == "unavailable"
    assert preview["manifest"]["report"]["counts"]["fallbackPatientIds"] == 1
    assert (folder / "predictions.json").read_bytes() == original

    # Even a matching artifact receipt cannot promote a fallback to a verified
    # identity when it contradicts the authoritative frozen cohort.
    value = json.loads(original)
    value["records"][0]["patientIdSource"] = "source"
    content = json.dumps(value).encode()
    (folder / "predictions.json").write_bytes(content)
    execution["result"]["artifacts"]["predictions.json"].update(
        bytes=len(content), sha256=hashlib.sha256(content).hexdigest()
    )
    preview = current.preview(selection)
    assert not preview["canSave"]
    assert "identity provenance" in preview["findings"][0]["message"]


@pytest.mark.parametrize("service", ["independent"], indirect=True)
def test_clinical_report_preserves_unverifiable_cross_dataset_patient_overlap(service):
    current, selection, *_ = service
    preview = current.preview(selection)
    assert preview["canSave"]
    assert any(
        "Cross-dataset patient overlap could not be verified" in warning
        for warning in preview["manifest"]["report"]["warnings"]
    )


@pytest.mark.parametrize("service", ["standalone-shared", "standalone-independent"], indirect=True)
def test_clinical_uses_independent_target_contract_and_reviewed_evaluation_overlap(service):
    current, selection, *_ = service
    evaluation = current.evaluations.get(selection.evaluationId)["manifest"]
    preview = current.preview(selection)
    assert preview["canSave"], preview
    warned = any(
        "Cross-dataset patient overlap could not be verified" in warning
        for warning in preview["manifest"]["report"]["warnings"]
    )
    assert warned is (not evaluation["overlap"]["patientsComparable"])


def test_changed_review_inputs_require_new_preview_and_operation_replay_matches_intent(service):
    current, selection, *_ = service
    preview = current.preview(selection)
    stale = SaveClinicalAnalysis(
        **{**selection.model_dump(), "threshold": 0.3},
        previewHash=preview["previewHash"],
        operationId="stale",
    )
    with pytest.raises(StorageError) as error:
        current.save(stale)
    assert error.value.code == "PREVIEW_STALE"
    _, request = save(service)
    with pytest.raises(StorageError) as error:
        current.save(request.model_copy(update={"bins": 5}))
    assert error.value.code == "OPERATION_CONFLICT"


@pytest.mark.parametrize(
    "artifact,mutate",
    [
        ("metrics.json", lambda value: value.update(decisionThreshold=0.9)),
        ("predictions.json", lambda value: value["records"][0].update(patientId="different")),
        ("predictions.json", lambda value: value.update(classOrder=["clear", "disease"])),
        ("predictions.json", lambda value: value["records"].append(value["records"][0])),
    ],
)
def test_valid_checksums_cannot_authorize_mismatched_scientific_evidence(service, artifact, mutate):
    current, selection, folder, execution = service
    path = folder / artifact
    value = json.loads(path.read_bytes())
    mutate(value)
    content = json.dumps(value).encode()
    path.write_bytes(content)
    execution["result"]["artifacts"][artifact].update(
        bytes=len(content), sha256=hashlib.sha256(content).hexdigest()
    )
    assert not current.preview(selection)["canSave"]


def test_clinical_control_service_does_not_import_training_libraries():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import histopilot.application.clinical; "
            "assert not any(name in sys.modules for name in ('torch', 'h5py', 'sklearn', 'scipy'))",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
