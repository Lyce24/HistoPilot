"""Fixed train/test construction and leak-free downstream derivations."""

import copy
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from histopilot.api.scientific import scientific_router
from histopilot.application.evaluations import EvaluationService
from histopilot.application.protocols import ProtocolService
from histopilot.application.target_splits import TargetSplitService
from histopilot.schemas.evaluations import EvaluationSpec
from histopilot.schemas.target_splits import TargetSplitSpec
from histopilot.storage.filesystem import LocalFilesystem
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import ScientificStore

TARGET = {
    "field": "label",
    "task": "binary_classification",
    "unit": "patient",
    "classes": ["low", "high"],
    "labels": {"0": "low", "1": "high"},
    "positiveClass": "high",
}
TRAINING = {
    "version": 4,
    "mode": "kfold",
    "folds": 3,
    "seeds": [42],
    "validationFraction": 0.2,
    "pools": {"trainSelection": "remaining"},
}


@pytest.fixture
def construction(tmp_path):
    rows = [
        {
            "slideId": f"s{patient:02}-{slide}",
            "patientId": f"p{patient:02}",
            "attributes": {
                "label": str(patient % 2),
                "site": "A" if patient < 32 else "B",
                "partition": "train" if patient < 32 else "test",
                "number": str(patient),
            },
        }
        for patient in range(40)
        for slide in range(2)
    ]
    (tmp_path / "project").mkdir()
    store = ScientificStore(tmp_path / "project", "project-targets")
    source = store.create_draft("import", "Dataset", {})
    dataset = store.publish_dataset(
        source["id"],
        expected_revision=1,
        manifest={
            "kind": "dataset",
            "dictionary": [
                {"key": field, "sourceColumn": field, "owner": "slide", "type": "text"}
                for field in rows[0]["attributes"]
            ],
        },
        artifacts={"records.json": json.dumps(rows).encode()},
        operation_id="dataset",
    )
    filesystem = LocalFilesystem((tmp_path,))
    service = TargetSplitService(store, filesystem)
    spec = {"datasetId": dataset["id"], "target": copy.deepcopy(TARGET)}
    return store, service, spec, rows


def frozen(construction, settings=None):
    store, service, spec, _rows = construction
    if settings:
        spec = {**spec, "split": settings}
    draft = store.create_draft(
        "experiment", "Target and test", {"type": "target-split", "spec": spec}
    )
    preview = service.preview(draft["id"], 1)
    assert preview["canFreeze"], preview["findings"]
    result = service.freeze(
        draft["id"],
        1,
        preview["previewHash"],
        "freeze-" + draft["id"],
        version_label={"tag": draft["id"]},
    )
    return result, draft, preview


def errors(result):
    return {item["code"] for item in result["findings"] if item["severity"] == "error"}


def test_default_random_partition_is_fixed_grouped_and_feature_free(construction, monkeypatch):
    store, service, spec, _rows = construction
    monkeypatch.setattr(
        store,
        "get_configuration",
        lambda *_: pytest.fail("Construction read features/configurations"),
    )
    first = service.preview_spec(spec)
    second = service.preview_spec(spec)
    assert first == second
    assert first["canFreeze"], first["findings"]
    summary = first["summary"]
    assert (summary["trainingPatients"], summary["testingPatients"]) == (32, 8)
    assert (summary["trainingSlides"], summary["testingSlides"]) == (64, 16)
    assert summary["testingClassCounts"] == {"low": 8, "high": 8}
    roles = {}
    for row in first["memberships"]:
        assert roles.setdefault(row["patientId"], row["partition"]) == row["partition"]
        assert "fold" not in row and "seed" not in row
    assert set(first["spec"]) == {"datasetId", "target", "predictors", "eligibility", "split"}
    assert set(first["partitions"]) == {"train", "test"}
    assert first["spec"]["split"]["stratify"] is False


def test_random_split_reports_the_testing_share_it_reached(construction):
    _store, service, spec, _rows = construction
    exact = service.preview_spec(spec)
    reached = ("testingUnits", "splitUnits", "achievedTestFraction")
    assert [exact["summary"][key] for key in reached] == [8, 40, 0.2]
    assert "TEST_FRACTION_DIFFERS" not in {item["code"] for item in exact["findings"]}
    # Forty one-patient strata each round 10% down to no testing patient.
    split = {"method": "random", "testFraction": 0.1, "stratify": True, "stratifyField": "number"}
    rounded = service.preview_spec({**spec, "split": split})
    assert [rounded["summary"][key] for key in reached] == [0, 40, 0]
    [warning] = [item for item in rounded["findings"] if item["code"] == "TEST_FRACTION_DIFFERS"]
    assert warning["severity"] == "warning"
    assert warning["message"].startswith(
        "Testing holds 0 of 40 patient groups (0%), not the requested 10%. "
        "Each value of number is split on its own"
    )
    live = service.partition_preview({"datasetId": spec["datasetId"], "split": split})
    assert [live["summary"][key] for key in reached] == [0, 40, 0]
    assert "TEST_FRACTION_DIFFERS" in {item["code"] for item in live["findings"]}
    imported = {
        "method": "imported",
        "partitionField": "partition",
        "trainValues": ["train"],
        "testValues": ["test"],
    }
    assert (
        "achievedTestFraction" not in service.preview_spec({**spec, "split": imported})["summary"]
    )


def test_deprecated_predictors_stay_empty_and_keep_stored_hashes(construction):
    _store, service, spec, _rows = construction
    # Every stored draft and frozen spec carries "predictors": []; it must keep its hash.
    stored = service.preview_spec({**spec, "predictors": []})
    assert stored["spec"]["predictors"] == []
    assert stored["previewHash"] == service.preview_spec(spec)["previewHash"]
    with pytest.raises(ValidationError, match="does not declare tabular predictors"):
        TargetSplitSpec.model_validate({**spec, "predictors": ["number"]})


def test_random_seed_changes_membership_not_population(construction):
    _store, service, spec, _rows = construction
    first = service.preview_spec(spec)
    second = service.preview_spec({**spec, "split": {"seed": 7}})
    assert first["memberships"] != second["memberships"]
    assert {row["slideId"] for row in first["memberships"]} == {
        row["slideId"] for row in second["memberships"]
    }


@pytest.mark.parametrize("method", ["rules", "imported"])
def test_explicit_metadata_partition_preserves_patients(construction, method):
    _store, service, spec, _rows = construction
    settings = {
        "method": method,
        **(
            {"testRules": [{"field": "site", "op": "eq", "value": "B"}]}
            if method == "rules"
            else {"partitionField": "partition", "trainValues": ["train"], "testValues": ["test"]}
        ),
    }
    result = service.preview_spec({**spec, "split": settings})
    assert result["canFreeze"], result["findings"]
    assert result["summary"]["testingSlides"] == 16
    assert all(
        row["partition"] == ("train" if int(row["patientId"][1:]) < 32 else "test")
        for row in result["memberships"]
    )


def test_rules_conflict_after_patient_expansion_blocks(construction):
    _store, service, spec, _rows = construction
    result = service.preview_spec(
        {
            **spec,
            "split": {
                "method": "rules",
                "trainRules": [{"field": "slideId", "op": "eq", "value": "s00-0"}],
                "testRules": [{"field": "slideId", "op": "eq", "value": "s00-1"}],
            },
        }
    )
    assert "OVERLAPPING_PATIENT_RULES" in errors(result)
    assert not result["memberships"]


def test_explicit_unmatched_rows_are_excluded_before_label_checks(construction):
    _store, service, spec, _rows = construction
    spec = {
        **spec,
        "split": {"method": "rules", "trainRules": [{"field": "number", "op": "lt", "value": 20}]},
    }
    result = service.preview_spec(spec)
    assert result["canFreeze"], result["findings"]
    assert result["summary"]["includedSlides"] == 40
    assert result["summary"]["excludedSlides"] == 40


@pytest.mark.parametrize(
    "changes",
    [
        {"featureBundleId": "configuration-" + "b" * 64},
        {"split": {"method": "random", "folds": 5}},
        {"split": {"testFraction": 1}},
        {"split": {"testFraction": True}},
        {
            "split": {
                "method": "imported",
                "partitionField": "partition",
                "trainValues": ["a"],
                "testValues": ["a"],
            }
        },
    ],
)
def test_target_split_schema_has_no_features_or_training_design(construction, changes):
    _store, _service, spec, _rows = construction
    with pytest.raises(ValidationError):
        TargetSplitSpec.model_validate({**spec, **changes})


def test_target_and_partition_alias_cannot_be_the_same(construction):
    _store, service, spec, _rows = construction
    result = service.preview_spec(
        {
            **spec,
            "split": {
                "method": "imported",
                "partitionField": "label",
                "trainValues": ["0"],
                "testValues": ["1"],
            },
        }
    )
    assert "SPLIT_TARGET_LEAKAGE" in errors(result)


def test_freeze_creates_exact_testing_cohort_and_replays_after_restart(construction):
    store, service, _spec, _rows = construction
    target, draft, preview = frozen(construction)
    cohort_id = target["evaluationCohortId"]
    assert cohort_id
    cohort = EvaluationService(store, service.filesystem).get(cohort_id)
    assert cohort["current"], cohort["findings"]
    assert "Testing" in cohort["versionLabel"]["tag"]
    assert cohort["manifest"]["spec"]["sourceTargetSplitId"] == target["id"]
    test_ids = {row["slideId"] for row in preview["memberships"] if row["partition"] == "test"}
    assert {row["slideId"] for row in cohort["manifest"]["memberships"]} == test_ids
    assert service.create_test_cohort(target["id"])["id"] == cohort_id
    restarted = TargetSplitService(
        ScientificStore(store.folder, store.project_id), service.filesystem
    )
    assert (
        restarted.freeze(
            draft["id"],
            1,
            preview["previewHash"],
            "freeze-" + draft["id"],
            version_label={"tag": draft["id"]},
        )
        == target
    )
    assert len(store.list_configurations("evaluation-cohort")) == 1


def test_zero_testing_allows_later_external_cohort(construction):
    target, _draft, preview = frozen(construction, {"testFraction": 0})
    assert preview["summary"]["trainingPatients"] == 40
    assert preview["summary"]["testingPatients"] == 0
    assert target["evaluationCohortId"] is None


def test_derived_training_never_sees_testing_rows_and_is_idempotent(construction):
    store, service, _spec, _rows = construction
    target, _draft, preview = frozen(construction)
    protocol = service.derive_protocol(target["id"], TRAINING)
    assert service.derive_protocol(target["id"], TRAINING) == protocol
    assert protocol["manifest"]["sourceTargetSplit"]["id"] == target["id"]
    train_ids = {row["slideId"] for row in preview["memberships"] if row["partition"] == "train"}
    test_patients = {
        row["patientId"] for row in preview["memberships"] if row["partition"] == "test"
    }
    memberships = protocol["manifest"]["memberships"]
    assert {row["slideId"] for row in memberships} == train_ids
    assert not {row["patientId"] for row in memberships} & test_patients
    # Reopening/copying a derived protocol retains its exact source restriction.
    copied = store.create_draft(
        "experiment", "Copy", {"type": "analysis-protocol", "spec": protocol["manifest"]["spec"]}
    )
    reopened = ProtocolService(store).preview(copied["id"], 1)
    assert reopened["memberships"] == memberships
    assert protocol["manifest"]["spec"]["sourceTargetSplitId"] == target["id"]


def test_testing_derivation_cannot_change_target(construction):
    store, service, _spec, _rows = construction
    target, _draft, _preview = frozen(construction)
    altered = {**target["manifest"]["spec"]["target"], "positiveClass": "low"}
    spec = EvaluationSpec(
        datasetId=target["manifest"]["datasetId"], target=altered, sourceTargetSplitId=target["id"]
    )
    with pytest.raises(StorageError) as caught:
        EvaluationService(store, service.filesystem)._prepare(spec)
    assert caught.value.code == "TARGET_SPLIT_TARGET_MISMATCH"


def test_target_split_api_preview_freeze_list_and_test_cohort(construction):
    store, service, spec, _rows = construction
    draft = store.create_draft("experiment", "API target", {"type": "target-split", "spec": spec})
    app = FastAPI()
    app.include_router(
        scientific_router(
            SimpleNamespace(scientific_store=lambda _identity: store), service.filesystem
        )
    )
    base = "/api/v1/projects/project-targets"
    with TestClient(app) as client:
        preview = client.post(
            base + f"/target-splits/{draft['id']}/preview", json={"expectedRevision": 1}
        )
        assert preview.status_code == 200, preview.text
        response = client.post(
            base + f"/target-splits/{draft['id']}/freeze",
            json={
                "expectedRevision": 1,
                "previewHash": preview.json()["previewHash"],
                "operationId": "api-target",
                "versionLabel": {"tag": "API target"},
            },
        )
        assert response.status_code == 201, response.text
        frozen = response.json()
        assert client.get(base + f"/target-splits/{frozen['id']}").json() == frozen
        assert (
            client.get(base + "/configurations", params={"kind": "target-split"}).json()[
                "configurations"
            ][0]["id"]
            == frozen["id"]
        )
        assert (
            client.post(base + f"/target-splits/{frozen['id']}/test-cohort").json()[
                "evaluationCohortId"
            ]
            == frozen["evaluationCohortId"]
        )


def replace_rows(service, monkeypatch, update, dataset_id=None):
    dataset_id = dataset_id or service.store.list_datasets()[0]["id"]
    dataset, fields, rows = service.protocols._load_dataset(dataset_id)
    update(rows)
    monkeypatch.setattr(
        service.protocols, "_load_dataset", lambda _identity: (dataset, fields, rows)
    )


@pytest.mark.parametrize(
    "problem,code",
    [
        ("missing", "MISSING_LABEL"),
        ("unmapped", "UNMAPPED_LABEL"),
        ("identity", "MISSING_PATIENT_ID"),
        ("mixed", "MIXED_PATIENT_LABELS"),
        ("duplicate", "DUPLICATE_SLIDE_ID"),
    ],
)
def test_invalid_patient_target_metadata_blocks_partition(construction, monkeypatch, problem, code):
    _store, service, spec, _rows = construction

    def update(rows):
        if problem == "missing":
            rows[0]["attributes"]["label"] = None
        elif problem == "unmapped":
            rows[0]["attributes"]["label"] = "unknown"
        elif problem == "identity":
            rows[0]["patientId"] = None
        elif problem == "mixed":
            rows[0]["attributes"]["label"] = "1"
        else:
            rows[0]["slideId"] = rows[1]["slideId"]

    replace_rows(service, monkeypatch, update)
    result = service.preview_spec(spec)
    assert code in errors(result)
    assert not result["memberships"]


def test_imported_patient_leakage_blocks_before_label_exclusion(construction, monkeypatch):
    _store, service, spec, _rows = construction

    def update(rows):
        rows[0]["attributes"].update(partition="test", label=None)

    replace_rows(service, monkeypatch, update)
    spec["target"]["missing"] = "exclude"
    spec["split"] = {
        "method": "imported",
        "partitionField": "partition",
        "trainValues": ["train"],
        "testValues": ["test"],
    }
    result = service.preview_spec(spec)
    assert "IMPORTED_PATIENT_LEAKAGE" in errors(result)
    assert not result["canFreeze"]


def test_unselected_missing_labels_do_not_block_and_order_is_deterministic(
    construction, monkeypatch
):
    _store, service, spec, _rows = construction
    spec["split"] = {
        "method": "rules",
        "trainRules": [{"field": "number", "op": "lt", "value": 20}],
    }
    baseline = service.preview_spec(spec)

    def update(rows):
        for row in rows:
            if int(row["patientId"][1:]) >= 20:
                row["attributes"]["label"] = None
        rows.reverse()

    replace_rows(service, monkeypatch, update)
    actual = service.preview_spec(spec)
    assert actual == baseline


def test_slide_labels_may_differ_within_group_without_patient_leakage(construction, monkeypatch):
    _store, service, spec, _rows = construction
    spec["target"]["unit"] = "slide"

    def update(rows):
        for row in rows:
            row["attributes"]["label"] = row["slideId"][-1]

    replace_rows(service, monkeypatch, update)
    result = service.preview_spec(spec)
    assert result["canFreeze"], result["findings"]
    roles = {}
    for row in result["memberships"]:
        assert roles.setdefault(row["patientId"], row["partition"]) == row["partition"]


def test_same_physical_slide_cannot_cross_training_testing(construction, monkeypatch):
    _store, service, spec, _rows = construction
    spec["split"] = {
        "method": "imported",
        "partitionField": "partition",
        "trainValues": ["train"],
        "testValues": ["test"],
    }

    def update(rows):
        rows[0]["slidePath"] = "/slides/duplicated.svs"
        rows[-1]["slidePath"] = "/slides/duplicated.svs"

    replace_rows(service, monkeypatch, update)
    result = service.preview_spec(spec)
    assert "TRAIN_TEST_SOURCE_OVERLAP" in errors(result)


def test_partition_preview_needs_no_target_and_matches_final_selection(construction, monkeypatch):
    store, service, spec, _rows = construction
    monkeypatch.setattr(store, "get_configuration", lambda *_: pytest.fail("Read configuration"))
    request = {"datasetId": spec["datasetId"]}
    partition = service.partition_preview(request)
    final = service.preview_spec(spec)
    assert partition["valid"]
    assert partition["membershipStatus"] == "fixed"
    assert partition["partitions"]["train"]["slides"] == final["summary"]["selectedTrainingSlides"]
    assert (
        partition["partitions"]["test"]["patients"] == final["summary"]["selectedTestingPatients"]
    )
    assert all("target" not in part for part in partition["partitions"].values())
    assert partition["summary"]["selectedSlides"] == 80
    assert partition["summary"]["excludedSlides"] == 0


def test_partition_first_filter_counts_and_raw_distributions(construction):
    _store, service, spec, _rows = construction
    request = {
        "datasetId": spec["datasetId"],
        "eligibility": [{"field": "number", "op": "lt", "value": 20}],
        "split": {
            "method": "rules",
            "trainRules": [{"field": "number", "op": "lt", "value": 6}],
            "testRules": [{"field": "number", "op": "eq", "value": 10}],
        },
        "targetFields": {"train": "label", "test": "label"},
    }
    result = service.partition_preview(request)
    assert result["valid"], result["findings"]
    summary = result["summary"]
    assert (summary["totalSlides"], summary["eligibleSlides"], summary["selectedSlides"]) == (
        80,
        40,
        14,
    )
    assert (summary["eligibilityExcludedSlides"], summary["partitionExcludedSlides"]) == (40, 26)
    assert summary["trainingPatients"] == 6
    target = result["partitions"]["train"]["target"]
    assert target["values"] == [
        {"value": "0", "slides": 6, "patients": 3},
        {"value": "1", "slides": 6, "patients": 3},
    ]
    assert target["classCounts"] == {}
    assert result["partitions"]["test"]["target"]["distinctCount"] == 1


def test_raw_metadata_stratification_and_legacy_prompt(construction):
    _store, service, spec, _rows = construction
    legacy = service.partition_preview(
        {"datasetId": spec["datasetId"], "split": {"stratify": True}}
    )
    assert "STRATIFICATION_FIELD_REQUIRED" in errors(legacy)
    assert legacy["membershipStatus"] == "provisional"
    assert legacy["summary"]["selectedSlides"] == 80
    assert not service.preview_spec({**spec, "split": {"stratify": True}})["canFreeze"]
    valid = service.preview_spec({**spec, "split": {"stratify": True, "stratifyField": "label"}})
    assert valid["canFreeze"], valid["findings"]
    assert valid["summary"]["testingClassCounts"] == {"low": 8, "high": 8}


def test_stratification_handles_mixed_and_missing_patient_values(construction, monkeypatch):
    _store, service, spec, _rows = construction

    def update(rows):
        rows[0]["attributes"]["site"] = None
        rows[3]["attributes"]["site"] = "different"

    replace_rows(service, monkeypatch, update)
    request = {"datasetId": spec["datasetId"], "split": {"stratify": True, "stratifyField": "site"}}
    first = service.partition_preview(request)
    assert first["valid"], first["findings"]
    assert first == service.partition_preview(request)
    assert first["summary"]["selectedSlides"] == 80


def test_target_mapping_and_exclusions_never_reassign_selected_patients(construction, monkeypatch):
    _store, service, spec, _rows = construction
    original = service.preview_spec(spec)
    roles = {row["slideId"]: row["partition"] for row in original["memberships"]}
    changed_target = {**spec["target"], "labels": {"0": "high", "1": "low"}}
    remapped = service.preview_spec({**spec, "target": changed_target})
    assert {row["slideId"]: row["partition"] for row in remapped["memberships"]} == roles

    def update(rows):
        for row in rows:
            if row["patientId"] in {"p00", "p01"}:
                row["attributes"]["label"] = None

    replace_rows(service, monkeypatch, update)
    changed_target["missing"] = "exclude"
    excluded = service.preview_spec({**spec, "target": changed_target})
    assert excluded["canFreeze"], excluded["findings"]
    assert excluded["summary"]["selectedSlides"] == 80
    assert excluded["summary"]["includedSlides"] == 76
    for row in excluded["memberships"]:
        assert row["partition"] == roles[row["slideId"]]
    for role in ("train", "test"):
        assert original["partitions"][role]["slides"] == excluded["partitions"][role]["slides"]


def test_patient_distribution_has_exclusive_verified_buckets(construction, monkeypatch):
    _store, service, spec, _rows = construction

    def update(rows):
        rows[0]["attributes"]["label"] = "1"  # p00: mixed mapped classes.
        rows[2]["attributes"]["label"] = None  # p01: partly missing.
        rows[4]["attributes"]["label"] = "unknown"  # p02: partly unmapped.
        for row in rows[6:8]:
            row["patientId"], row["patientIdSource"] = row["slideId"], "slide_fallback"

    replace_rows(service, monkeypatch, update)
    target = {**TARGET, "unit": "slide"}
    request = {"datasetId": spec["datasetId"], "split": {"testFraction": 0}, "target": target}
    result = service.partition_preview(request)
    part = result["partitions"]["train"]
    distribution = part["target"]
    assert part["patients"] == 39
    assert part["groups"] == 41
    assert part["fallbackSlides"] == 2
    assert distribution["mixedLabelPatients"] == 1
    assert distribution["missingLabelPatients"] == 1
    assert distribution["unmappedLabelPatients"] == 1
    assert sum(distribution["patientClassCounts"].values()) == 36
    assert (
        distribution["labeledPatients"]
        + distribution["unlabeledPatients"]
        + distribution["mixedLabelPatients"]
        == 39
    )
    assert distribution["missingSlides"] == distribution["unmappedSlides"] == 1
    assert distribution["mixedValuePatients"] == 3
    assert sum(item["patients"] for item in distribution["values"]) == 36


def test_different_raw_values_can_map_to_one_verified_patient_class(construction, monkeypatch):
    _store, service, spec, _rows = construction
    replace_rows(service, monkeypatch, lambda rows: rows[0]["attributes"].update(label="zero"))
    target = {**TARGET, "labels": {"0": "low", "zero": "low", "1": "high"}}
    result = service.partition_preview(
        {"datasetId": spec["datasetId"], "split": {"testFraction": 0}, "target": target}
    )
    distribution = result["partitions"]["train"]["target"]
    assert distribution["mixedValuePatients"] == 1
    assert distribution["mixedLabelPatients"] == 0
    assert distribution["patientClassCounts"] == {"low": 20, "high": 20}


def test_unfinished_target_mapping_preserves_live_partition_counts(construction):
    _store, service, spec, _rows = construction
    request = {
        "datasetId": spec["datasetId"],
        "targetFields": {"train": "label"},
        "target": {"field": "label", "classes": []},
    }
    result = service.partition_preview(request)
    assert "INVALID_TARGET_MAPPING" in errors(result)
    assert result["membershipStatus"] == "fixed"
    assert result["summary"]["trainingSlides"] == 64
    assert result["partitions"]["train"]["target"]["values"]


def test_inference_selection_never_reads_testing_labels(construction, monkeypatch):
    from histopilot.application.protocols import FilterEvaluator

    _store, service, spec, _rows = construction
    original = FilterEvaluator.field

    def reject_testing_label(row, field):
        if row["attributes"]["partition"] == "test" and field == "label":
            pytest.fail("Inference read a testing label")
        return original(row, field)

    monkeypatch.setattr(FilterEvaluator, "field", staticmethod(reject_testing_label))
    split = {
        "method": "imported",
        "partitionField": "partition",
        "trainValues": ["train"],
        "testValues": ["test"],
    }
    request = {**spec, "split": split, "testTarget": None}
    final = service.preview_spec(request)
    assert final["canFreeze"], final["findings"]
    assert final["summary"]["testingSlides"] == 16
    assert final["summary"]["testingClassCounts"] == {}
    assert final["summary"]["testingPurpose"] == "inference"
    assert all(row["label"] is None for row in final["memberships"] if row["partition"] == "test")
    assert "target" not in final["partitions"]["test"]
    preview = service.partition_preview({**request, "targetFields": {"test": "label"}})
    assert "target" not in preview["partitions"]["test"]


def publish_revised_rows(construction, rows):
    store, service, spec, _original = construction
    draft = store.create_draft("import", "Revised dataset", {})
    dataset = store.publish_dataset(
        draft["id"],
        expected_revision=1,
        manifest={
            "kind": "dataset",
            "dictionary": [
                {"key": field, "sourceColumn": field, "owner": "slide", "type": "text"}
                for field in rows[0]["attributes"]
            ],
        },
        artifacts={"records.json": json.dumps(rows).encode()},
        operation_id="revised-dataset",
    )
    return store, service, {**spec, "datasetId": dataset["id"]}, rows


def test_unlabeled_test_rows_freeze_exact_inference_cohort_and_training_source(construction):
    _store, _service, spec, original = construction
    rows = copy.deepcopy(original)
    for row in rows:
        if row["attributes"]["partition"] == "test":
            row["attributes"]["label"] = None
    construction = publish_revised_rows(construction, rows)
    store, service, spec, _rows = construction
    spec["testTarget"] = None
    settings = {
        "method": "imported",
        "partitionField": "partition",
        "trainValues": ["train"],
        "testValues": ["test"],
    }
    artifact, _draft, preview = frozen(construction, settings)
    cohort = EvaluationService(store, service.filesystem).get(artifact["evaluationCohortId"])
    assert cohort["current"], cohort["findings"]
    assert cohort["manifest"]["spec"]["purpose"] == "inference"
    assert cohort["manifest"]["target"] is None
    assert cohort["manifest"]["summary"]["labeledSlides"] == 0
    assert cohort["manifest"]["summary"]["includedSlides"] == 16
    assert {row["slideId"] for row in cohort["manifest"]["memberships"]} == {
        row["slideId"] for row in rows if row["attributes"]["partition"] == "test"
    }
    protocol = service.derive_protocol(artifact["id"], TRAINING)
    assert not {row["patientId"] for row in protocol["manifest"]["memberships"]} & {
        row["patientId"] for row in cohort["manifest"]["memberships"]
    }
    assert preview["summary"]["testingLabelExclusions"] == {}


def test_target_serialization_distinguishes_inherited_and_unlabeled(construction):
    _store, _service, spec, _rows = construction
    assert "testTarget" not in TargetSplitSpec.model_validate(spec).model_dump(mode="json")
    assert (
        TargetSplitSpec.model_validate({**spec, "testTarget": None}).model_dump(mode="json")[
            "testTarget"
        ]
        is None
    )
    with pytest.raises(ValidationError):
        TargetSplitSpec.model_validate({**spec, "testTarget": {**TARGET, "positiveClass": "low"}})
    with pytest.raises(ValidationError):
        TargetSplitSpec.model_validate(
            {**spec, "testTarget": {**TARGET, "labels": {"0": "high", "1": "low"}}}
        )


def test_frozen_testing_source_rejects_additional_filters(construction):
    target, _draft, _preview = frozen(construction)
    with pytest.raises(ValidationError):
        EvaluationSpec(
            datasetId=target["manifest"]["datasetId"],
            target=target["manifest"]["spec"]["target"],
            sourceTargetSplitId=target["id"],
            eligibility=[{"field": "label", "op": "eq", "value": "0"}],
        )


def test_partition_preview_api_without_target_or_draft(construction):
    store, service, spec, _rows = construction
    app = FastAPI()
    app.include_router(
        scientific_router(
            SimpleNamespace(scientific_store=lambda _identity: store), service.filesystem
        )
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/projects/project-targets/target-splits/partition-preview",
            json={"datasetId": spec["datasetId"], "targetFields": {"train": "label"}},
        )
    assert response.status_code == 200, response.text
    assert response.json()["partitions"]["train"]["target"]["distinctCount"] == 2
    assert response.json()["summary"]["selectedSlides"] == 80


def test_legacy_frozen_partition_replays_without_repartitioning(construction, monkeypatch):
    store, service, spec, _rows = construction
    preview = service.preview_spec(spec)
    legacy_spec = copy.deepcopy(preview["spec"])
    legacy_spec["split"]["stratify"] = True
    legacy_spec["split"].pop("stratifyField")
    legacy = {
        **{key: value for key, value in preview.items() if key != "partitions"},
        "kind": "target-split",
        "schemaVersion": 1,
        "datasetId": spec["datasetId"],
        "spec": legacy_spec,
        "algorithm": "histopilot-target-training-testing-v1",
    }
    draft = store.create_draft(
        "experiment", "Legacy frozen", {"type": "target-split", "spec": legacy_spec}
    )
    original = store.publish_configuration(
        draft["id"],
        expected_revision=1,
        manifest=legacy,
        operation_id="legacy-targets",
        version_label={"tag": "Legacy frozen"},
    )
    monkeypatch.setattr(
        service, "preview_spec", lambda *_: pytest.fail("Repartitioned frozen data")
    )
    replay = service.freeze(
        draft["id"],
        1,
        preview["previewHash"],
        "legacy-targets",
        version_label={"tag": "Legacy frozen"},
    )
    assert replay["manifest"] == original["manifest"]
    assert replay["id"] == original["id"]
    cohort = EvaluationService(store, service.filesystem).get(replay["evaluationCohortId"])
    assert cohort["current"], cohort["findings"]
    assert "purpose" not in cohort["manifest"]["spec"]
    assert {row["slideId"] for row in cohort["manifest"]["memberships"]} == {
        row["slideId"] for row in legacy["memberships"] if row["partition"] == "test"
    }
    protocol = service.derive_protocol(original["id"], TRAINING)
    assert {row["slideId"] for row in protocol["manifest"]["memberships"]} == {
        row["slideId"] for row in legacy["memberships"] if row["partition"] == "train"
    }


@pytest.mark.parametrize("separate_table", [False, True])
def test_testing_alias_maps_preserve_source_meaning_but_allow_separate_tables(
    construction, monkeypatch, separate_table
):
    _store, service, spec, _rows = construction
    dataset, fields, rows = service.protocols._load_dataset(spec["datasetId"])
    # Both attributes may be patient-owned; source table identity still matters.
    fields["label"]["owner"] = "patient"
    fields["alias"] = {**fields["label"], "key": "alias", "sourceColumn": "label"}
    dataset["manifest"]["provenance"] = {
        "mapping": {
            "patientAttributes": [{"key": "alias", "sourceColumn": "label"}]
            if separate_table
            else [],
        }
    }
    for row in rows:
        row["attributes"]["alias"] = row["attributes"]["label"]
    monkeypatch.setattr(
        service.protocols, "_load_dataset", lambda _identity: (dataset, fields, rows)
    )
    testing = {**TARGET, "field": "alias", "labels": {"0": "high", "1": "low"}}
    candidate = {**spec, "testTarget": testing}
    result = service.preview_spec(candidate)
    assert result["canFreeze"] is separate_table, result["findings"]
    assert ("TARGET_LABEL_MAPPING_MISMATCH" in errors(result)) is not separate_table
    exploration = service.partition_preview(candidate)
    assert ("TARGET_LABEL_MAPPING_MISMATCH" in errors(exploration)) is not separate_table
    assert exploration["summary"]["selectedSlides"] == 80


def test_training_design_cannot_refilter_frozen_training_population(construction):
    store, service, _spec, _rows = construction
    artifact, _draft, _preview = frozen(
        construction,
        {
            "method": "rules",
            "testRules": [{"field": "site", "op": "eq", "value": "B"}],
        },
    )
    narrowed = {
        **TRAINING,
        "pools": {
            "trainSelection": "rules",
            "rules": {"train": [{"field": "number", "op": "lt", "value": 16}]},
        },
    }
    with pytest.raises(StorageError) as caught:
        service.derive_protocol(artifact["id"], narrowed)
    assert caught.value.code == "TRAINING_SPLIT_BLOCKED"
    assert "preserve every frozen training slide" in str(caught.value)
    assert store.list_configurations("protocol") == []
    protocol = service.derive_protocol(artifact["id"], TRAINING)
    changed = {
        **protocol["manifest"]["spec"],
        "eligibility": [{"field": "number", "op": "lt", "value": 16}],
    }
    draft = store.create_draft(
        "experiment", "Narrowed copy", {"type": "analysis-protocol", "spec": changed}
    )
    preview = ProtocolService(store, service.filesystem).preview(draft["id"], 1)
    assert "TARGET_SPLIT_TRAINING_MEMBERSHIP_CHANGED" in errors(preview)
    assert not preview["canFreeze"]
    with pytest.raises(StorageError):
        ProtocolService(store, service.filesystem).freeze(
            draft["id"],
            1,
            preview["previewHash"],
            "refilter-copy",
            version_label={"tag": "Narrowed copy"},
        )


@pytest.mark.parametrize("method", ["rules", "imported"])
def test_requested_split_roles_survive_filtering_preview_and_freeze(construction, method):
    """Unequal named pools must stay attached to their roles, not UI order."""
    _store, _service, _spec, original = construction
    rows = copy.deepcopy(original)
    for row in rows:
        row["attributes"]["Requested_Split"] = row["attributes"]["partition"]
    store, service, spec, rows = publish_revised_rows(construction, rows)
    split = (
        {
            "method": "rules",
            "trainRules": [{"field": "Requested_Split", "op": "eq", "value": "train"}],
            "testRules": [{"field": "Requested_Split", "op": "eq", "value": "test"}],
        }
        if method == "rules"
        else {
            "method": "imported",
            "partitionField": "Requested_Split",
            "trainValues": ["train"],
            "testValues": ["test"],
        }
    )
    request = {"datasetId": spec["datasetId"], "split": split}
    before = service.partition_preview(request)
    assert (before["summary"]["trainingSlides"], before["summary"]["testingSlides"]) == (64, 16)
    eligibility = [{"field": "number", "op": "lt", "value": 36}]
    after = service.partition_preview({**request, "eligibility": eligibility})
    assert after["valid"], after["findings"]
    assert after["dataset"]["totalSlides"] == 80
    assert after["cohort"]["totalSlides"] == 72
    assert after["cohort"]["patientCount"] == 36
    assert (after["summary"]["trainingSlides"], after["summary"]["testingSlides"]) == (64, 8)
    draft = store.create_draft(
        "experiment",
        "Unequal requested pools",
        {"type": "target-split", "spec": {**spec, "split": split, "eligibility": eligibility}},
    )
    final = service.preview(draft["id"], 1)
    assert final["canFreeze"], final["findings"]
    expected = {
        role: {
            row["slideId"]
            for row in rows
            if row["attributes"]["Requested_Split"] == role
            and int(row["attributes"]["number"]) < 36
        }
        for role in ("train", "test")
    }
    actual = {
        role: {row["slideId"] for row in final["memberships"] if row["partition"] == role}
        for role in ("train", "test")
    }
    assert actual == expected
    for role in ("train", "test"):
        selected = after["partitions"][role]
        assert selected["slides"] == len(expected[role])
        assert selected["selection"]["directMatches"]["totalSlides"] == len(expected[role])
        assert selected["selection"]["assigned"]["totalSlides"] == len(expected[role])
    frozen = service.freeze(
        draft["id"],
        1,
        final["previewHash"],
        "requested-split-roles",
        version_label={"tag": "Requested split roles"},
    )
    cohort = EvaluationService(store, service.filesystem).get(frozen["evaluationCohortId"])
    assert {row["slideId"] for row in cohort["manifest"]["memberships"]} == expected["test"]


def test_rule_selection_distinguishes_direct_matches_patient_expansion_and_remaining(construction):
    _store, service, spec, _rows = construction
    request = {
        "datasetId": spec["datasetId"],
        "split": {
            "method": "rules",
            "testRules": [
                {"field": "slideId", "op": "eq", "value": "s00-0"},
            ],
        },
    }
    result = service.partition_preview(request)
    assert result["valid"], result["findings"]
    test = result["partitions"]["test"]
    assert test["selection"]["mode"] == "rules"
    assert test["selection"]["directMatches"]["totalSlides"] == 1
    assert test["selection"]["directMatches"]["patientCount"] == 1
    assert test["selection"]["expanded"]["totalSlides"] == test["slides"] == 2
    assert test["selection"]["assigned"]["patientCount"] == test["patients"] == 1
    train = result["partitions"]["train"]
    assert train["selection"]["mode"] == "remaining"
    assert train["selection"]["assigned"]["totalSlides"] == train["slides"] == 78
    assert train["patients"] == 39
    request["eligibility"] = [{"field": "slideId", "op": "ne", "value": "s00-1"}]
    narrowed = service.partition_preview(request)
    assert narrowed["cohort"]["totalSlides"] == 79
    assert narrowed["partitions"]["test"]["selection"]["expanded"]["totalSlides"] == 1
    assert narrowed["partitions"]["train"]["slides"] == 78


def test_overlapping_selection_does_not_present_rule_matches_as_assigned(construction):
    _store, service, spec, _rows = construction
    result = service.partition_preview(
        {
            "datasetId": spec["datasetId"],
            "split": {
                "method": "rules",
                "trainRules": [{"field": "slideId", "op": "eq", "value": "s00-0"}],
                "testRules": [{"field": "slideId", "op": "eq", "value": "s00-1"}],
            },
        }
    )
    assert "OVERLAPPING_PATIENT_RULES" in errors(result)
    assert result["membershipStatus"] == "provisional"
    for role in ("train", "test"):
        part = result["partitions"][role]
        assert part["selection"]["directMatches"]["totalSlides"] == 1
        assert part["selection"]["expanded"]["totalSlides"] == 2
        assert part["selection"]["assigned"]["totalSlides"] == part["slides"] == 0


def test_imported_selection_reports_patient_expansion_and_conflicts(construction, monkeypatch):
    _store, service, spec, _rows = construction

    def update(rows):
        rows[1]["attributes"]["partition"] = None
        rows[3]["attributes"]["partition"] = "test"

    replace_rows(service, monkeypatch, update)
    result = service.partition_preview(
        {
            "datasetId": spec["datasetId"],
            "split": {
                "method": "imported",
                "partitionField": "partition",
                "trainValues": ["train"],
                "testValues": ["test"],
            },
        }
    )
    assert "IMPORTED_PATIENT_LEAKAGE" in errors(result)
    train = result["partitions"]["train"]
    test = result["partitions"]["test"]
    assert train["selection"]["mode"] == test["selection"]["mode"] == "imported"
    assert train["selection"]["directMatches"]["totalSlides"] == 62
    assert train["selection"]["expanded"]["totalSlides"] == 64
    assert train["selection"]["assigned"]["totalSlides"] == train["slides"] == 62
    assert test["selection"]["directMatches"]["totalSlides"] == 17
    assert test["selection"]["expanded"]["totalSlides"] == 18
    assert test["selection"]["assigned"]["totalSlides"] == test["slides"] == 16


def test_no_testing_rules_are_distinct_from_zero_matching_testing_rules(construction):
    _store, service, spec, _rows = construction
    result = service.partition_preview(
        {"datasetId": spec["datasetId"], "split": {"method": "rules"}}
    )
    assert result["partitions"]["test"]["selection"]["mode"] == "none"
    assert result["partitions"]["train"]["selection"]["mode"] == "remaining"
    assert result["summary"]["trainingSlides"] == 80
    assert result["summary"]["testingSlides"] == 0


def slide_construction(construction, count=24, training_count=16):
    """Real slide experiments can share patients and have unresolved identities."""
    rows = [
        {
            "slideId": f"slide-{index:04}",
            "patientId": None if index % 11 == 0 else f"shared-{index % 7}",
            "attributes": {
                "label": str(index % 2),
                "Requested_Split": "train" if index < training_count else "test",
            },
        }
        for index in range(count)
    ]
    store, service, spec, rows = publish_revised_rows(construction, rows)
    spec.update(splitUnit="slide", target={**TARGET, "unit": "slide"})
    return store, service, spec, rows


def requested_settings(method):
    return (
        {
            "method": "rules",
            "trainRules": [{"field": "Requested_Split", "op": "eq", "value": "train"}],
            "testRules": [{"field": "Requested_Split", "op": "eq", "value": "test"}],
        }
        if method == "rules"
        else {
            "method": "imported",
            "partitionField": "Requested_Split",
            "trainValues": ["train"],
            "testValues": ["test"],
        }
    )


@pytest.mark.parametrize("method", ["rules", "imported"])
def test_slide_split_exact_requested_counts_ignore_patient_grouping(
    construction, monkeypatch, method
):
    _store, service, spec, rows = slide_construction(construction, 1111, 728)
    spec["split"] = requested_settings(method)
    monkeypatch.setattr(
        service.protocols,
        "_identity_findings",
        lambda *_: pytest.fail("Slide splitting must not require patient grouping"),
    )
    live = service.partition_preview(spec)
    result = service.preview_spec(spec)
    assert live["valid"], live["findings"]
    assert result["canFreeze"], result["findings"]
    assert (live["summary"]["trainingSlides"], live["summary"]["testingSlides"]) == (728, 383)
    assert (result["summary"]["trainingSlides"], result["summary"]["testingSlides"]) == (728, 383)
    assert result["summary"]["grouping"] == result["spec"]["splitUnit"] == "slide"
    assert result["algorithm"] == "histopilot-target-training-testing-slide-v1"
    for role in ("train", "test"):
        expected = {row["slideId"] for row in rows if row["attributes"]["Requested_Split"] == role}
        actual = {row["slideId"] for row in result["memberships"] if row["partition"] == role}
        assert actual == expected
        part = live["partitions"][role]
        assert part["groups"] == len(expected)
        assert part["patients"] == 0
        assert part["target"]["patientClassCounts"] == {}
        assert part["selection"]["assigned"]["patientCount"] == 0
        assert part["selection"]["directMatches"]["totalSlides"] == len(expected)
        assert part["selection"]["expanded"]["totalSlides"] == len(expected)
        assert part["selection"]["assigned"]["totalSlides"] == len(expected)
    original_identities = {row["slideId"]: row["patientId"] for row in rows}
    assert {
        row["slideId"]: row["patientId"] for row in result["memberships"]
    } == original_identities
    patient_spec = {
        **spec,
        "splitUnit": "patient",
        "target": {**spec["target"], "unit": "patient"},
    }
    monkeypatch.undo()
    patient = service.preview_spec(patient_spec)
    assert not patient["canFreeze"]
    assert errors(patient) & {"OVERLAPPING_PATIENT_RULES", "IMPORTED_PATIENT_LEAKAGE"}


def test_slide_random_assignments_are_independent_of_patient_ids(construction, monkeypatch):
    _store, service, spec, _rows = slide_construction(construction)
    first = service.preview_spec(spec)
    assert first["canFreeze"], first["findings"]
    original = {row["slideId"]: row["partition"] for row in first["memberships"]}
    replace_rows(
        service,
        monkeypatch,
        lambda rows: [row.update(patientId=None) for row in rows],
        dataset_id=spec["datasetId"],
    )
    without_patients = service.preview_spec(spec)
    assert without_patients["canFreeze"], without_patients["findings"]
    assert {row["slideId"]: row["partition"] for row in without_patients["memberships"]} == original
    assert all(row["patientId"] is None for row in without_patients["memberships"])
    assert (
        without_patients["summary"]["trainingGroups"]
        == without_patients["summary"]["trainingSlides"]
    )
    assert (
        without_patients["summary"]["testingGroups"] == without_patients["summary"]["testingSlides"]
    )


def test_slide_split_rejects_same_slide_rule_overlap_without_patient_findings(construction):
    _store, service, spec, _rows = slide_construction(construction)
    spec["split"] = {
        "method": "rules",
        "trainRules": [
            {"field": "Slide_ID", "op": "eq", "value": "slide-0001"},
        ],
        "testRules": [
            {"field": "Slide_ID", "op": "in", "value": ["slide-0001", "slide-0002"]},
        ],
    }
    live = service.partition_preview(spec)
    assert "OVERLAPPING_SLIDE_RULES" in errors(live)
    assert "OVERLAPPING_PATIENT_RULES" not in errors(live)
    assert live["partitions"]["train"]["slides"] == 0
    assert live["partitions"]["test"]["slides"] == 1
    assert not service.preview_spec(spec)["canFreeze"]


@pytest.mark.parametrize("same_partition", [False, True])
def test_slide_split_still_blocks_duplicate_physical_slides(
    construction, monkeypatch, same_partition
):
    _store, service, spec, _rows = slide_construction(construction)
    spec["split"] = requested_settings("imported")

    def update(rows):
        rows[1]["slidePath"] = "/slides/same-physical-slide.svs"
        rows[2 if same_partition else -1]["slidePath"] = rows[1]["slidePath"]

    replace_rows(service, monkeypatch, update, dataset_id=spec["datasetId"])
    result = service.preview_spec(spec)
    assert not result["canFreeze"]
    assert (
        "DUPLICATE_TRAINING_SLIDE_SOURCE" if same_partition else "TRAIN_TEST_SOURCE_OVERLAP"
    ) in errors(result)


def test_explicit_split_unit_matches_prediction_unit_and_legacy_serialization_is_unchanged(
    construction,
):
    _store, service, spec, _rows = construction
    legacy = TargetSplitSpec.model_validate(spec)
    assert legacy.splitUnit == "patient"
    assert "splitUnit" not in legacy.model_dump(mode="json")
    legacy_preview = service.preview_spec(spec)
    assert "splitUnit" not in legacy_preview["spec"]
    assert "splitUnit" not in legacy_preview["summary"]
    with pytest.raises(ValidationError, match="prediction unit must match"):
        TargetSplitSpec.model_validate({**spec, "splitUnit": "slide"})
    live = service.partition_preview({**spec, "splitUnit": "slide"})
    assert "SPLIT_TARGET_UNIT_MISMATCH" in errors(live)
    assert live["membershipStatus"] == "fixed"
    assert live["summary"]["trainingSlides"] > 0
    explicit = TargetSplitSpec.model_validate({**spec, "splitUnit": "patient"})
    assert explicit.model_dump(mode="json")["splitUnit"] == "patient"


@pytest.mark.parametrize("inference", [False, True])
def test_slide_split_freezes_exact_testing_source_and_derives_slide_training(
    construction, inference
):
    store, service, spec, rows = slide_construction(construction)
    if inference:
        spec["testTarget"] = None
    artifact, _draft, preview = frozen((store, service, spec, rows), requested_settings("imported"))
    cohort = EvaluationService(store, service.filesystem).get(artifact["evaluationCohortId"])
    assert cohort["current"], cohort["findings"]
    assert cohort["manifest"]["spec"]["splitUnit"] == "slide"
    assert {row["slideId"] for row in cohort["manifest"]["memberships"]} == {
        row["slideId"] for row in preview["memberships"] if row["partition"] == "test"
    }
    if inference:
        assert all(row["label"] is None for row in cohort["manifest"]["memberships"])
    protocol = service.derive_protocol(artifact["id"], TRAINING)
    assert protocol["manifest"]["spec"]["splitUnit"] == "slide"
    assert {row["slideId"] for row in protocol["manifest"]["memberships"]} == {
        row["slideId"] for row in preview["memberships"] if row["partition"] == "train"
    }
    original_identities = {row["slideId"]: row["patientId"] for row in rows}
    assert all(
        row["patientId"] == original_identities[row["slideId"]]
        for row in protocol["manifest"]["memberships"]
    )
    from histopilot.application.target_split_source import restrict_target_split_rows

    with pytest.raises(StorageError) as caught:
        restrict_target_split_rows(
            store,
            artifact["id"],
            spec["datasetId"],
            TargetSplitSpec.model_validate(spec).target,
            rows,
            "train",
            split_unit="patient",
        )
    assert caught.value.code == "TARGET_SPLIT_UNIT_MISMATCH"


def test_slide_testing_rule_leaves_remaining_slides_for_training(construction):
    _store, service, spec, rows = slide_construction(construction, 1111, 728)
    spec["split"] = {
        "method": "rules",
        "testRules": [{"field": "Requested_Split", "op": "eq", "value": "test"}],
    }
    live = service.partition_preview(spec)
    final = service.preview_spec(spec)
    assert live["valid"] and final["canFreeze"], final["findings"]
    assert (live["summary"]["trainingSlides"], live["summary"]["testingSlides"]) == (728, 383)
    assert live["partitions"]["train"]["selection"]["mode"] == "remaining"
    assert {row["slideId"] for row in final["memberships"] if row["partition"] == "train"} == {
        row["slideId"] for row in rows if row["attributes"]["Requested_Split"] == "train"
    }
    assert live["partitions"]["test"]["selection"]["directMatches"]["totalSlides"] == 383
    assert live["partitions"]["test"]["selection"]["expanded"]["totalSlides"] == 383


def test_slide_testing_can_take_every_slide_outside_training(construction):
    store, service, spec, rows = slide_construction(construction, 1111, 728)
    # 698 training slides leave the other 413 eligible slides for testing.
    held_back = [f"slide-{index:04}" for index in range(30)]
    settings = {
        "method": "rules",
        "trainRules": [
            {"field": "Requested_Split", "op": "eq", "value": "train"},
            {"field": "slideId", "op": "not_in", "value": held_back},
        ],
        "testRemaining": True,
    }
    training = {
        row["slideId"]
        for row in rows
        if row["attributes"]["Requested_Split"] == "train" and row["slideId"] not in held_back
    }
    live = service.partition_preview({**spec, "split": settings})
    assert live["valid"], live["findings"]
    assert (live["summary"]["trainingSlides"], live["summary"]["testingSlides"]) == (698, 413)
    assert live["summary"]["selectedSlides"] == live["summary"]["eligibleSlides"] == 1111
    assert live["partitions"]["train"]["selection"]["mode"] == "rules"
    assert live["partitions"]["test"]["selection"]["mode"] == "remaining"
    artifact, _draft, preview = frozen((store, service, spec, rows), settings)
    members = {
        role: {row["slideId"] for row in preview["memberships"] if row["partition"] == role}
        for role in ("train", "test")
    }
    assert members["train"] == training
    assert members["test"] == {row["slideId"] for row in rows} - training
    cohort = EvaluationService(store, service.filesystem).get(artifact["evaluationCohortId"])
    assert {row["slideId"] for row in cohort["manifest"]["memberships"]} == members["test"]


def test_testing_remainder_keeps_each_patient_group_on_one_side(construction):
    _store, service, spec, _rows = construction
    result = service.partition_preview(
        {
            "datasetId": spec["datasetId"],
            "split": {
                "method": "rules",
                "trainRules": [{"field": "slideId", "op": "eq", "value": "s00-0"}],
                "testRemaining": True,
            },
        }
    )
    assert result["valid"], result["findings"]
    train, test = result["partitions"]["train"], result["partitions"]["test"]
    # The matched slide's patient group trains whole; every other group is tested.
    assert (train["slides"], train["patients"]) == (2, 1)
    assert (test["slides"], test["patients"]) == (78, 39)
    assert test["selection"]["mode"] == "remaining"


def test_testing_remainder_needs_training_conditions(construction):
    _store, service, spec, _rows = construction
    result = service.partition_preview(
        {"datasetId": spec["datasetId"], "split": {"method": "rules", "testRemaining": True}}
    )
    assert "TESTING_REMAINDER_NEEDS_TRAINING_RULES" in errors(result)
    assert result["summary"]["trainingSlides"] == result["summary"]["testingSlides"] == 0
    assert {part["selection"]["mode"] for part in result["partitions"].values()} == {"none"}


def test_testing_remainder_is_rules_only_exclusive_and_keeps_historical_serialization(
    construction,
):
    _store, _service, spec, _rows = construction
    rules = {"method": "rules", "trainRules": [{"field": "site", "op": "eq", "value": "A"}]}
    base = TargetSplitSpec.model_validate({**spec, "split": rules}).model_dump(mode="json")
    assert "testRemaining" not in base["split"]
    unchanged = {**spec, "split": {**rules, "testRemaining": False}}
    assert TargetSplitSpec.model_validate(unchanged).model_dump(mode="json") == base
    remaining = {**spec, "split": {**rules, "testRemaining": True}}
    assert TargetSplitSpec.model_validate(remaining).model_dump(mode="json")["split"] == {
        **base["split"],
        "testRemaining": True,
    }
    with pytest.raises(ValidationError, match="only to the rules method"):
        TargetSplitSpec.model_validate(
            {**spec, "split": {"method": "random", "testRemaining": True}}
        )
    with pytest.raises(ValidationError, match="not both"):
        TargetSplitSpec.model_validate(
            {
                **spec,
                "split": {
                    **rules,
                    "testRemaining": True,
                    "testRules": [{"field": "site", "op": "eq", "value": "B"}],
                },
            }
        )
