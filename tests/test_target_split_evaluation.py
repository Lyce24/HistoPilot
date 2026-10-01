"""Frozen testing partitions survive predictor binding without losing their purpose."""

import copy
import json
import sys

import pytest
from support import projects
from support.predictors import candidate, freeze, registry

from histopilot.application.evaluation_runs import EvaluationRunService, run_purpose
from histopilot.application.target_splits import TargetSplitService
from histopilot.schemas.predictors import (
    CompareEvaluations,
    EvaluationRunSelection,
    SaveEvaluationRun,
)
from histopilot.storage.project_lock import StorageError


@pytest.fixture(params=["evaluation", "separate-testing-labels", "inference"])
def testing_partition(tmp_path, request):
    predictors, original_cohort = registry.__wrapped__(tmp_path)
    selection, *_ = candidate(predictors)
    predictor, _ = freeze(predictors, selection)
    targets = TargetSplitService(predictors.store, predictors.filesystem)
    spec = {
        "datasetId": original_cohort["manifest"]["datasetId"],
        "target": copy.deepcopy(original_cohort["manifest"]["spec"]["target"]),
        "split": {
            "method": "rules",
            "testRules": [{"field": "cohort", "op": "eq", "value": "test"}],
        },
    }
    if request.param == "separate-testing-labels":
        # A second target column must keep the same class meanings while its raw
        # values can differ. Existing verified features cover the same slide IDs.
        source = predictors.store.get_dataset(spec["datasetId"])
        rows = json.loads(predictors.store.read_artifact(source["id"], "records.json"))
        for row in rows:
            row["attributes"]["testing_label"] = {"0": "negative", "1": "positive"}[
                row["attributes"]["label"]
            ]
        imported = predictors.store.create_draft("import", "Separate testing labels", {})
        dataset = predictors.store.publish_dataset(
            imported["id"],
            expected_revision=1,
            manifest={
                **source["manifest"],
                "dictionary": [
                    *source["manifest"]["dictionary"],
                    {
                        "key": "testing_label",
                        "sourceColumn": "testing_label",
                        "owner": "slide",
                        "type": "text",
                    },
                ],
            },
            artifacts={"records.json": json.dumps(rows).encode()},
            operation_id="separate-testing-labels",
        )
        spec["datasetId"] = dataset["id"]
        spec["testTarget"] = {
            **spec["target"],
            "field": "testing_label",
            "labels": {"negative": "low", "positive": "high"},
        }
    elif request.param == "inference":
        # Raw target values exist, but choosing inference must never retain them.
        spec["testTarget"] = None
    draft = predictors.store.create_draft(
        "experiment", "Training and reserved testing", {"type": "target-split", "spec": spec}
    )
    preview = targets.preview(draft["id"], 1)
    assert preview["canFreeze"], preview["findings"]
    frozen = targets.freeze(draft["id"], 1, preview["previewHash"], "fixed-membership")
    runs = EvaluationRunService(predictors.store, predictors.filesystem)
    cohort = runs.cohorts.get(frozen["evaluationCohortId"])
    return (
        runs,
        predictor,
        frozen,
        cohort,
        "inference" if request.param == "inference" else "evaluation",
    )


def test_derived_testing_cohort_binds_to_predictor_and_preserves_exact_rows(
    testing_partition, monkeypatch
):
    runs, predictor, source, cohort, purpose = testing_partition
    testing = [row for row in source["manifest"]["memberships"] if row["partition"] == "test"]
    assert {row["slideId"] for row in testing} == {"s2", "s3"}
    assert cohort["current"]
    assert cohort["manifest"]["spec"]["sourceTargetSplitId"] == source["id"]
    assert {row["slideId"] for row in cohort["manifest"]["memberships"]} == {"s2", "s3"}
    chosen = EvaluationRunSelection(
        predictorId=predictor["id"], cohortId=cohort["id"], name="Frozen test partition"
    )
    preview = runs.preview(chosen)
    assert preview["canSave"], preview["findings"]
    manifest = preview["manifest"]
    assert run_purpose(manifest) == purpose
    assert manifest["summary"]["includedSlides"] == 2
    assert manifest["summary"]["developmentSlideOverlap"] == 0
    assert manifest["summary"]["developmentPatientOverlap"] == 0
    assert manifest["coverage"]["selectedSlideIds"] == ["s2", "s3"]
    assert manifest["summary"]["labeledSlides"] == (0 if purpose == "inference" else 2)
    assert ("analysis" in manifest) == (purpose == "evaluation")
    if purpose == "inference":
        assert cohort["manifest"]["spec"]["purpose"] == "inference"
        assert cohort["manifest"]["target"] is None
        assert all(row["label"] is None for row in cohort["manifest"]["memberships"])
    record = runs.save(
        SaveEvaluationRun(
            **chosen.model_dump(), previewHash=preview["previewHash"], operationId="save-run"
        )
    )
    assert runs.get(record["id"])["execution"]["status"] == "not_started"
    monkeypatch.setattr(
        "histopilot.application.evaluation_runs.training_runtime",
        lambda: {
            "available": True,
            "cudaAvailable": False,
            "gpuCount": 0,
            "python": sys.executable,
            "versions": {},
        },
    )
    plan = runs._execution_plan(record["id"])
    # Labeled runs predict label-blind; inference cohorts carry no labels to withhold.
    assert plan.get("labelsWithheld", False) == (purpose == "evaluation")
    assert plan["data"]["memberships"] == (
        [
            {key: value for key, value in row.items() if key != "label"}
            for row in cohort["manifest"]["memberships"]
        ]
        if purpose == "evaluation"
        else cohort["manifest"]["memberships"]
    )
    assert run_purpose(plan) == purpose
    assert ("analysis" in plan) == (purpose == "evaluation")
    # Planning alone never creates a worker or result artifacts.
    assert not runs.jobs.folder(record["id"]).exists()
    if purpose == "inference":
        with pytest.raises(StorageError) as comparison:
            runs.compare(
                CompareEvaluations(leftEvaluationId=record["id"], rightEvaluationId=record["id"])
            )
        assert comparison.value.code == "COMPARISON_REQUIRES_LABELS"


@pytest.mark.parametrize("purpose", ["evaluation", "inference"])
def test_slide_unit_preserves_metadata_and_never_groups_training_or_testing(
    tmp_path, monkeypatch, purpose
):
    from histopilot.application.predictors import PredictorService
    from histopilot.storage.filesystem import LocalFilesystem
    from histopilot.storage.scientific import ScientificStore

    folder = tmp_path / "project"
    folder.mkdir()
    store = ScientificStore(folder, "slide-experiment")
    filesystem = LocalFilesystem((tmp_path,))
    # One shared identity has conflicting slide targets and occurs on both sides
    # of the frozen partition. Other slides have no patient identity at all.
    rows = [
        {
            "slideId": f"s{i}",
            "patientId": None if i % 7 == 0 else "shared",
            "patientIdSource": "metadata",
            "attributes": {"label": str(i % 2), "cohort": "development" if i < 32 else "test"},
        }
        for i in range(40)
    ]
    source, _ = projects.dataset(store, rows=rows)
    features, _, _ = projects.bundle(store, tmp_path, source, [row["slideId"] for row in rows])
    targets = TargetSplitService(store, filesystem)
    spec = {
        "datasetId": source["id"],
        "splitUnit": "slide",
        "target": {**projects.TARGET, "unit": "slide"},
        "split": {
            "method": "rules",
            "testRules": [{"field": "cohort", "op": "eq", "value": "test"}],
        },
        **({"testTarget": None} if purpose == "inference" else {}),
    }
    draft = store.create_draft(
        "experiment", "Slide experiment", {"type": "target-split", "spec": spec}
    )
    review = targets.preview(draft["id"], 1)
    assert review["canFreeze"], review["findings"]
    frozen = targets.freeze(draft["id"], 1, review["previewHash"], "slide-partition")
    protocol = targets.derive_protocol(
        frozen["id"],
        {
            "version": 4,
            "mode": "kfold",
            "folds": 2,
            "seeds": [42],
            "validationFraction": 0.2,
            "pools": {"trainSelection": "remaining"},
        },
    )
    assert protocol["manifest"]["spec"]["splitUnit"] == "slide"
    membership = protocol["manifest"]["memberships"]
    original = {row["slideId"]: row for row in rows}
    assert {row["slideId"] for row in membership} == {f"s{i}" for i in range(32)}
    assert all(row["patientId"] == original[row["slideId"]]["patientId"] for row in membership)
    assert {row["partition"] for row in membership if row["patientId"] == "shared"} == {
        "train",
        "val",
        "test",
    }
    assert protocol["manifest"]["summary"]["includedGroups"] == 32
    assert protocol["manifest"]["summary"]["includedPatients"] == 0

    predictors = PredictorService(store, filesystem)
    selection, *_ = candidate(predictors, feature_bundle_id=features["id"])
    predictor, _ = freeze(predictors, selection)
    runs = EvaluationRunService(store, filesystem)
    cohort = runs.cohorts.get(frozen["evaluationCohortId"])
    assert cohort["current"], cohort["findings"]
    assert cohort["manifest"]["spec"]["splitUnit"] == "slide"
    chosen = EvaluationRunSelection(
        predictorId=predictor["id"], cohortId=cohort["id"], name="Slide testing"
    )
    review = runs.preview(chosen)
    assert review["canSave"], review["findings"]
    manifest = review["manifest"]
    assert manifest["splitUnit"] == "slide"
    assert manifest["summary"]["includedSlides"] == 8
    assert manifest["summary"]["includedPatients"] == 0
    assert manifest["summary"]["developmentPatientOverlap"] == 0
    assert "analysis" not in manifest
    record = runs.save(
        SaveEvaluationRun(
            **chosen.model_dump(), previewHash=review["previewHash"], operationId="slide-run"
        )
    )
    monkeypatch.setattr(
        "histopilot.application.evaluation_runs.training_runtime",
        lambda: {
            "available": True,
            "cudaAvailable": False,
            "gpuCount": 0,
            "python": sys.executable,
            "versions": {},
        },
    )
    plan = runs._execution_plan(record["id"])
    assert plan["splitUnit"] == "slide"
    assert "analysis" not in plan
    assert plan.get("labelsWithheld", False) == (purpose == "evaluation")
    assert plan["data"]["memberships"] == (
        [
            {key: value for key, value in row.items() if key != "label"}
            for row in cohort["manifest"]["memberships"]
        ]
        if purpose == "evaluation"
        else cohort["manifest"]["memberships"]
    )
    assert all(
        row["patientId"] == original[row["slideId"]]["patientId"]
        for row in plan["data"]["memberships"]
    )
    assert {row["slideId"] for row in plan["data"]["memberships"]} == {
        f"s{i}" for i in range(32, 40)
    }
    assert not runs.jobs.folder(record["id"]).exists()
