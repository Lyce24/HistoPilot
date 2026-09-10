"""Patient grouping, explicit labels, safe filtering, split reproducibility and freeze CAS."""

import copy
import json
import os
import subprocess
import sys

import pytest
from pydantic import ValidationError

from histopilot.application.protocols import FilterEvaluator, FilterFailure, ProtocolService
from histopilot.schemas.protocols import Condition, ProtocolExploreRequest, ProtocolSpec
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import ScientificStore

DATASET_ID = "dataset-" + "a" * 64
FIELDS = ("label", "grade", "stage", "alias", "fold", "partition", "number", "name")


def records():
    return [
        {
            "slideId": f"s{patient:02d}-{slide}",
            "patientId": f"p{patient:02d}",
            "slidePath": None,
            "attributes": {
                "label": str(patient % 2),
                "grade": "2" if patient >= 12 else "1",
                "stage": "Primary" if slide == 0 else "Metastatic",
                "alias": "low" if patient % 2 == 0 else "high",
                "fold": str((patient // 2) % 3) if patient < 12 else "-1",
                "partition": "trainval" if patient < 12 else "test",
                "number": str(patient),
                "name": "ordinary",
            },
        }
        for patient in range(16)
        for slide in range(2)
    ]


def specification():
    return {
        "datasetId": DATASET_ID,
        "target": {
            "field": "label",
            "task": "binary_classification",
            "unit": "patient",
            "classes": ["low", "high"],
            "labels": {"0": "low", "1": "high"},
            "positiveClass": "high",
        },
        "split": {
            "mode": "kfold",
            "folds": 3,
            "seeds": [42, 7],
            "rules": {"test": [{"field": "grade", "op": "eq", "value": "2"}]},
        },
    }


class MemoryStore:
    def __init__(self, rows=None, spec=None):
        self.rows = records() if rows is None else rows
        self.draft = {
            "id": "draft-test",
            "revision": 1,
            "kind": "experiment",
            "status": "editable",
            "payload": {"type": "analysis-protocol", "spec": spec or specification()},
        }
        self.dataset = {
            "id": DATASET_ID,
            "contentHash": "a" * 64,
            "manifest": {
                "kind": "dataset",
                "dictionary": [
                    {"key": key, "sourceColumn": key, "owner": "slide", "type": "text"}
                    for key in FIELDS
                ],
            },
        }
        self.feature = None

    def get_draft(self, _identity):
        return copy.deepcopy(self.draft)

    def get_dataset(self, _identity):
        return copy.deepcopy(self.dataset)

    def read_artifact(self, _identity, name):
        assert name == "records.json"
        return json.dumps(self.rows).encode()

    def get_configuration(self, _identity):
        return copy.deepcopy(self.feature)


def preview(store=None):
    return ProtocolService(store or MemoryStore()).preview("draft-test", 1)


def codes(result):
    return {finding["code"] for finding in result["findings"] if finding["severity"] == "error"}


def assert_patient_disjoint(result):
    states = {}
    for item in result["memberships"]:
        key = (item["seed"], item["fold"], item["patientId"])
        assert key not in states or states[key] == item["partition"]
        states[key] = item["partition"]


def test_generated_kfold_expands_grade2_test_and_preserves_every_patient():
    result = preview()
    assert result["canFreeze"], result["findings"]
    assert result["executionEnabled"] is False
    assert result["summary"]["includedPatients"] == 16
    assert result["summary"]["includedSlides"] == 32
    assert result["summary"]["fixedPatients"] == {"train": 0, "val": 0, "test": 4}
    assert len(result["partitions"]) == 6
    assert len(result["memberships"]) == 32 * 3 * 2
    assert_patient_disjoint(result)
    for part in result["partitions"]:
        assert part["test"]["patients"] == 4
        assert part["val"]["classes"] == {"low": 2, "high": 2}
        assert part["train"]["classes"] == {"low": 4, "high": 4}
    assert all(
        item["partition"] == "test"
        for item in result["memberships"]
        if int(item["patientId"][1:]) >= 12
    )


def test_row_order_and_python_hash_seed_do_not_change_memberships_or_preview():
    store = MemoryStore()
    expected = preview(store)
    store.rows.reverse()
    assert preview(store) == expected
    code = """
import hashlib, json, runpy
values=runpy.run_path('tests/test_protocols.py')
result=values['preview']()
print(hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest())
"""
    outputs = []
    for hash_seed in ("1", "991"):
        outputs.append(
            subprocess.check_output(
                [sys.executable, "-c", code],
                env={**os.environ, "PYTHONHASHSEED": hash_seed},
                text=True,
            ).strip()
        )
    assert outputs[0] == outputs[1]
    by_seed = {
        seed: {
            item["patientId"]
            for item in expected["memberships"]
            if item["seed"] == seed and item["fold"] == 0 and item["partition"] == "val"
        }
        for seed in (7, 42)
    }
    assert by_seed[7] != by_seed[42]


@pytest.mark.parametrize("raw,code", [(None, "MISSING_LABEL"), ("unknown", "UNMAPPED_LABEL")])
def test_invalid_labels_block_without_implicit_row_drops(raw, code):
    store = MemoryStore()
    store.rows[0]["attributes"]["label"] = raw
    result = preview(store)
    assert not result["canFreeze"]
    assert code in codes(result)
    assert result["summary"]["eligibleSlides"] == 32
    assert result["memberships"] == []


@pytest.mark.parametrize("raw,policy", [(None, "missing"), ("unknown", "unmapped")])
def test_explicit_label_exclusion_is_counted_and_can_be_reviewed(raw, policy):
    store = MemoryStore()
    store.rows[0]["attributes"]["label"] = raw
    store.draft["payload"]["spec"]["target"][policy] = "exclude"
    result = preview(store)
    assert result["canFreeze"], result["findings"]
    assert result["summary"]["excludedSlides"] == 1
    assert result["summary"]["includedPatients"] == 16


@pytest.mark.parametrize("patient", [None, "", "  ", " p00"])
def test_missing_patient_identity_never_falls_back_to_slide_or_case(patient):
    store = MemoryStore()
    store.rows[0]["patientId"] = patient
    result = preview(store)
    assert "MISSING_PATIENT_ID" in codes(result)
    assert not result["canFreeze"]
    assert result["memberships"] == []


@pytest.mark.parametrize(
    "unit,expected",
    [("patient", "MIXED_PATIENT_LABELS"), ("slide", "MIXED_PATIENT_STRATIFICATION_UNSUPPORTED")],
)
def test_mixed_patient_labels_never_select_majority(unit, expected):
    store = MemoryStore()
    store.draft["payload"]["spec"]["target"]["unit"] = unit
    store.rows[0]["attributes"]["label"] = "1"
    result = preview(store)
    assert expected in codes(result)
    assert not result["canFreeze"]


def test_consistent_slide_target_still_splits_whole_patients():
    store = MemoryStore()
    store.draft["payload"]["spec"]["target"]["unit"] = "slide"
    result = preview(store)
    assert result["canFreeze"]
    assert result["summary"]["targetUnit"] == "slide"
    assert_patient_disjoint(result)


@pytest.mark.parametrize(
    "field,expected",
    [
        ("Patient_ID", "FORBIDDEN_PREDICTOR"),
        ("fold", "FORBIDDEN_PREDICTOR"),
        ("label", "TARGET_PREDICTOR_LEAKAGE"),
        ("alias", "TARGET_EQUIVALENT_PREDICTOR"),
    ],
)
def test_identifier_split_label_and_target_equivalent_predictors_block(field, expected):
    store = MemoryStore()
    store.draft["payload"]["spec"]["predictors"] = [field]
    result = preview(store)
    assert expected in codes(result)
    assert not result["canFreeze"]


def test_eligibility_conditions_apply_to_the_same_slide():
    store = MemoryStore()
    store.draft["payload"]["spec"]["eligibility"] = [
        {"field": "stage", "op": "eq", "value": "Primary"},
        {"field": "Slide_ID", "op": "regex", "value": "-1$"},
    ]
    result = preview(store)
    assert result["summary"]["eligibleSlides"] == 0
    assert "EMPTY_COHORT" in codes(result)


def test_fixed_rules_expand_matching_slide_to_patient_and_detect_cross_slide_overlap():
    store = MemoryStore()
    rules = store.draft["payload"]["spec"]["split"]["rules"]
    rules["test"] = [{"field": "Slide_ID", "op": "regex", "value": "^s1[2-5]-0$"}]
    result = preview(store)
    assert result["canFreeze"]
    assert result["summary"]["fixedPatients"]["test"] == 4
    rules["train"] = [{"field": "Slide_ID", "op": "regex", "value": "^s12-1$"}]
    result = preview(store)
    assert "OVERLAPPING_PATIENT_RULES" in codes(result)


def test_holdout_uses_declared_remaining_pool_ratios_and_class_minimums():
    store = MemoryStore()
    split = store.draft["payload"]["spec"]["split"]
    split.update(mode="holdout", ratios={"train": 0.8, "val": 0.2, "test": 0})
    result = preview(store)
    assert result["canFreeze"], result["findings"]
    assert all(part["fold"] is None for part in result["partitions"])
    assert all(part["test"]["patients"] == 4 for part in result["partitions"])
    assert all(part["train"]["patients"] == 10 for part in result["partitions"])
    assert all(part["val"]["patients"] == 2 for part in result["partitions"])
    assert_patient_disjoint(result)


def imported_store():
    store = MemoryStore()
    store.draft["payload"]["spec"]["split"] = {
        "mode": "imported",
        "folds": 3,
        "seeds": [42],
        "imported": {
            "foldField": "fold",
            "foldLabels": {"0": 0, "1": 1, "2": 2},
            "testFoldLabels": ["-1"],
        },
    }
    return store


def test_imported_folds_require_explicit_test_mapping_and_patient_coherence():
    store = imported_store()
    result = preview(store)
    assert result["canFreeze"], result["findings"]
    assert_patient_disjoint(result)
    store.draft["payload"]["spec"]["split"]["imported"]["testFoldLabels"] = []
    assert "INVALID_IMPORTED_FOLD" in codes(preview(store))
    store = imported_store()
    store.rows[0]["attributes"]["fold"] = "1"
    assert "IMPORTED_PATIENT_LEAKAGE" in codes(preview(store))


def test_imported_fold_range_must_be_complete():
    store = imported_store()
    for row in store.rows:
        if row["attributes"]["fold"] == "2":
            row["attributes"]["fold"] = "1"
    result = preview(store)
    assert "INCOMPLETE_IMPORTED_FOLDS" in codes(result)
    assert not result["canFreeze"]


def test_imported_partition_leakage_is_blocked():
    store = MemoryStore()
    store.draft["payload"]["spec"]["split"] = {
        "mode": "imported",
        "seeds": [42],
        "imported": {
            "partitionField": "partition",
            "partitionLabels": {"trainval": "train", "test": "val"},
        },
    }
    assert preview(store)["canFreeze"]
    store.rows[0]["attributes"]["partition"] = "test"
    assert "IMPORTED_PATIENT_LEAKAGE" in codes(preview(store))


@pytest.mark.parametrize(
    "condition,expected",
    [
        ({"field": "number", "op": "gte", "value": 2}, True),
        ({"field": "number", "op": "lt", "value": 2}, False),
        ({"field": "number", "op": "eq", "value": 2}, True),
        ({"field": "number", "op": "ne", "value": 3}, True),
        ({"field": "stage", "op": "in", "value": ["Primary", "Other"]}, True),
        ({"field": "stage", "op": "not_in", "value": ["Metastatic"]}, True),
        ({"field": "slidePath", "op": "exists", "value": False}, True),
    ],
)
def test_typed_filter_operations(condition, expected):
    assert FilterEvaluator().matches(records()[4], Condition.model_validate(condition)) is expected


def test_invalid_regex_and_nonnumeric_filter_values_block_preview():
    store = MemoryStore()
    store.draft["payload"]["spec"]["eligibility"] = [{"field": "name", "op": "regex", "value": "["}]
    assert "INVALID_REGEX" in codes(preview(store))
    store.draft["payload"]["spec"]["eligibility"] = [{"field": "name", "op": "lt", "value": 10}]
    assert "INVALID_FILTER_VALUE" in codes(preview(store))


def test_adversarial_regex_times_out_instead_of_hanging():
    row = records()[0]
    row["attributes"]["name"] = "a" * 30000 + "!"
    condition = Condition(field="name", op="regex", value="(a+)+$")
    with pytest.raises(FilterFailure) as error:
        FilterEvaluator().matches(row, condition)
    assert error.value.code == "REGEX_TIMEOUT"


@pytest.mark.parametrize(
    "condition",
    [
        {"field": "grade", "op": "lt", "value": "2"},
        {"field": "grade", "op": "gt", "value": True},
        {"field": "grade", "op": "exists", "value": "yes"},
        {"field": "grade", "op": "in", "value": "1"},
        {"field": "grade", "op": "regex", "value": "x" * 513},
    ],
)
def test_invalid_filter_value_types_are_rejected(condition):
    with pytest.raises(ValidationError):
        Condition.model_validate(condition)


def test_large_finite_integer_threshold_does_not_overflow_validation():
    condition = Condition(field="number", op="lt", value=10**400)
    assert FilterEvaluator().matches(records()[0], condition)


@pytest.mark.parametrize(
    "change",
    [
        {"folds": 1},
        {"folds": 11},
        {"folds": True},
        {"seeds": [True]},
        {"seeds": [-1]},
        {"seeds": [2**32]},
        {"seeds": [1, 1]},
        {"ratios": {"train": 0.8, "val": 0.4, "test": 0}},
    ],
)
def test_invalid_split_parameters_are_rejected(change):
    spec = specification()
    spec["split"].update(change)
    with pytest.raises(ValidationError):
        ProtocolSpec.model_validate(spec)


def test_infeasible_class_partition_counts_block_freeze():
    store = MemoryStore()
    store.draft["payload"]["spec"]["constraints"] = {
        "minPatientsPerClass": 3,
        "minPatientsPerPartition": 1,
    }
    result = preview(store)
    assert "PARTITION_CLASS_TOO_SMALL" in codes(result)
    assert not result["canFreeze"]


def test_feature_binding_requires_same_dataset_and_complete_cohort_coverage():
    store = MemoryStore()
    store.draft["payload"]["spec"]["featureSetId"] = "configuration-" + "f" * 64
    store.feature = {
        "contentHash": "f" * 64,
        "manifest": {
            "kind": "feature",
            "datasetId": DATASET_ID,
            "files": [{"slideId": row["slideId"]} for row in store.rows],
        },
    }
    assert preview(store)["canFreeze"]
    store.feature["manifest"]["files"].pop()
    assert "MISSING_FEATURE_COVERAGE" in codes(preview(store))
    store.feature["manifest"]["datasetId"] = "dataset-" + "b" * 64
    assert "FEATURE_DATASET_MISMATCH" in codes(preview(store))


def test_multiclass_target_and_explicit_class_mapping():
    store = MemoryStore()
    for row in store.rows:
        patient = int(row["patientId"][1:])
        row["attributes"]["label"] = str(patient % 3)
    store.draft["payload"]["spec"]["target"] = {
        "field": "label",
        "task": "multiclass_classification",
        "classes": ["A", "B", "C"],
        "labels": {"0": "A", "1": "B", "2": "C"},
    }
    result = preview(store)
    assert result["canFreeze"], result["findings"]
    assert set(result["summary"]["classCounts"]) == {"A", "B", "C"}


def test_real_store_freeze_is_immutable_restartable_and_idempotent(tmp_path):
    store = ScientificStore(tmp_path, "project-test")
    store.initialize()
    imported = store.create_draft("import", "Source", {})
    dataset = store.publish_dataset(
        imported["id"],
        expected_revision=1,
        manifest=MemoryStore().dataset["manifest"],
        artifacts={"records.json": json.dumps(records()).encode()},
        operation_id="import-fixture",
    )
    spec = specification()
    spec["datasetId"] = dataset["id"]
    draft = store.create_draft(
        "experiment", "Grade prediction", {"type": "analysis-protocol", "spec": spec}
    )
    service = ProtocolService(store)
    result = service.preview(draft["id"], 1)
    assert result["canFreeze"]
    with pytest.raises(StorageError) as stale:
        service.freeze(draft["id"], 1, "0" * 64, "freeze")
    assert stale.value.code == "STALE_PREVIEW"
    frozen = service.freeze(draft["id"], 1, result["previewHash"], "freeze")
    assert frozen["manifest"]["memberships"] == result["memberships"]
    assert store.get_draft(draft["id"])["status"] == "frozen"
    restarted = ProtocolService(ScientificStore(tmp_path, "project-test"))
    assert restarted.freeze(draft["id"], 1, result["previewHash"], "freeze") == frozen
    with pytest.raises(StorageError):
        restarted.freeze(draft["id"], 1, result["previewHash"], "different-operation")
    assert store.get_configuration(frozen["id"]) == frozen


def test_changed_draft_revision_and_blocking_findings_prevent_publication():
    store = MemoryStore()
    store.draft["revision"] = 2
    with pytest.raises(StorageError) as error:
        ProtocolService(store).preview("draft-test", 1)
    assert error.value.code == "REVISION_CONFLICT"
    store.draft["revision"] = 1
    store.rows[0]["patientId"] = None
    service = ProtocolService(store)
    result = service.preview("draft-test", 1)
    with pytest.raises(StorageError) as error:
        service.freeze("draft-test", 1, result["previewHash"], "blocked")
    assert error.value.code == "PROTOCOL_PREFLIGHT_BLOCKED"


def test_protocol_identity_does_not_depend_on_draft_identity_revision_or_seed_order():
    store = MemoryStore()
    service = ProtocolService(store)
    first = service.preview("draft-one", 1)
    store.draft.update(id="draft-two", revision=4)
    store.draft["payload"]["spec"]["split"]["seeds"].reverse()
    second = service.preview("draft-two", 4)
    assert first == second


def test_arbitrarily_named_source_identifiers_and_oceanpath_k_fold_are_not_predictors():
    for key, source in (("patientAlias", "Registry number"), ("foldAlias", "k_fold")):
        store = MemoryStore()
        store.dataset["manifest"]["dictionary"].append(
            {"key": key, "sourceColumn": source, "owner": "slide", "type": "text"}
        )
        store.dataset["manifest"]["provenance"] = {
            "mapping": {"patientIdColumn": "Registry number", "slideIdColumn": "Slide accession"}
        }
        store.draft["payload"]["spec"]["predictors"] = [key]
        for row in store.rows:
            row["attributes"][key] = row["patientId"]
        result = preview(store)
        assert "FORBIDDEN_PREDICTOR" in codes(result)


def test_nondefault_holdout_ratios_cannot_silently_create_or_imply_a_kfold_test_set():
    store = MemoryStore()
    store.draft["payload"]["spec"]["split"]["ratios"] = {"train": 0.7, "val": 0.1, "test": 0.2}
    assert "UNUSED_HOLDOUT_RATIOS" in codes(preview(store))


def test_oversized_protocol_is_blocked_during_preview(monkeypatch):
    monkeypatch.setattr("histopilot.application.protocols.MAX_PROTOCOL_BYTES", 1000)
    result = preview()
    assert not result["canFreeze"]
    assert "PROTOCOL_DOCUMENT_LIMIT" in codes(result)
    assert result["memberships"] == []


def test_frozen_protocol_identity_is_shared_by_equivalent_new_drafts(tmp_path):
    store = ScientificStore(tmp_path, "project-test")
    store.initialize()
    imported = store.create_draft("import", "Source", {})
    dataset = store.publish_dataset(
        imported["id"],
        expected_revision=1,
        manifest=MemoryStore().dataset["manifest"],
        artifacts={"records.json": json.dumps(records()).encode()},
        operation_id="import-equivalent",
    )
    spec = specification()
    spec["datasetId"] = dataset["id"]
    service = ProtocolService(store)
    results = []
    for index in range(2):
        draft = store.create_draft(
            "experiment", f"Protocol {index}", {"type": "analysis-protocol", "spec": spec}
        )
        candidate = service.preview(draft["id"], 1)
        results.append(
            service.freeze(draft["id"], 1, candidate["previewHash"], f"equivalent-{index}")
        )
        assert store.get_draft(draft["id"])["status"] == "frozen"
    assert results[0] == results[1]


def explore(store=None, **request):
    return ProtocolService(store or MemoryStore()).explore(
        ProtocolExploreRequest(datasetId=DATASET_ID, **request)
    )


def test_live_cohort_needs_no_labels_or_feature_configuration_and_uses_every_row():
    store = MemoryStore()
    store.draft["payload"] = {}
    store.rows = [
        {**row, "slideId": f"{batch}-{row['slideId']}", "patientId": f"{batch}-{row['patientId']}"}
        for batch in range(10)
        for row in records()
    ]
    result = explore(store, targetField="label")
    assert result["valid"], result["findings"]
    assert result["dataset"]["totalSlides"] == result["cohort"]["totalSlides"] == 320
    assert result["cohort"]["patientCount"] == result["cohort"]["groupCount"] == 160
    assert result["cohort"]["fallbackSlideCount"] == result["cohort"]["unlinkedSlideCount"] == 0
    assert len(result["cohort"]["sample"]) == 5
    assert sum(value["slides"] for value in result["target"]["values"]) == 320
    assert result["partitions"]["train"]["selection"] == "remaining"
    assert result["partitions"]["train"]["expanded"]["totalSlides"] == 320
    assert result["partitions"]["val"]["expanded"]["totalSlides"] == 0
    assert result["unassigned"]["totalSlides"] == 0
    assert store.draft["payload"] == {}


def test_rules_test_expands_to_whole_patient_and_empty_train_uses_complement():
    store = MemoryStore()
    split = store.draft["payload"]["spec"]["split"]
    split.update(mode="rules")
    split["rules"] = {"test": [{"field": "Slide_ID", "op": "regex", "value": "^s1[2-5]-0$"}]}
    live = explore(store, rules=split["rules"])
    frozen = preview(store)
    assert live["valid"] and frozen["canFreeze"], frozen["findings"]
    assert live["partitions"]["test"]["directMatches"]["totalSlides"] == 4
    assert live["partitions"]["test"]["expanded"]["totalSlides"] == 8
    assert live["partitions"]["train"]["expanded"]["totalSlides"] == 24
    assert live["partitions"]["val"]["selection"] == "none"
    assert len(frozen["partitions"]) == 2
    assert {part["fold"] for part in frozen["partitions"]} == {0}
    for part in frozen["partitions"]:
        for role in ("train", "val", "test"):
            assert part[role]["slides"] == live["partitions"][role]["expanded"]["totalSlides"]
            assert part[role]["patients"] == live["partitions"][role]["expanded"]["patientCount"]
    assert_patient_disjoint(frozen)


def test_rules_validation_is_optional_but_explicit_validation_is_reserved():
    store = MemoryStore()
    split = store.draft["payload"]["spec"]["split"]
    split.update(mode="rules")
    split["rules"]["val"] = [{"field": "number", "op": "lt", "value": 2}]
    result = preview(store)
    assert result["canFreeze"], result["findings"]
    assert all(part["val"]["patients"] == 2 for part in result["partitions"])
    assert all(part["train"]["patients"] == 10 for part in result["partitions"])
    split["rules"] = {}
    result = preview(store)
    assert result["canFreeze"], result["findings"]
    assert all(part["train"]["patients"] == 16 for part in result["partitions"])
    assert all(part["val"]["patients"] == 0 for part in result["partitions"])


def test_explicit_train_rule_never_silently_drops_unassigned_groups():
    store = MemoryStore()
    split = store.draft["payload"]["spec"]["split"]
    split.update(mode="rules")
    split["rules"]["train"] = [{"field": "number", "op": "lt", "value": 2}]
    live = explore(store, rules=split["rules"])
    assert "UNASSIGNED_RULE_GROUPS" in codes(live)
    assert not live["valid"]
    assert live["unassigned"]["patientCount"] == 10
    result = preview(store)
    assert "UNASSIGNED_RULE_GROUPS" in codes(result)
    assert not result["canFreeze"]
    assert result["memberships"] == []


def test_live_eligibility_applies_before_patient_expansion():
    result = explore(
        eligibility=[{"field": "stage", "op": "eq", "value": "Primary"}],
        rules={"test": [{"field": "grade", "op": "eq", "value": "2"}]},
    )
    assert result["cohort"]["totalSlides"] == 16
    assert result["cohort"]["patientCount"] == 16
    assert result["partitions"]["test"]["expanded"]["totalSlides"] == 4
    assert result["partitions"]["train"]["expanded"]["totalSlides"] == 12


@pytest.mark.parametrize(
    "condition,code",
    [
        ({"field": "name", "op": "regex", "value": "["}, "INVALID_REGEX"),
        ({"field": "missing-field", "op": "eq", "value": "x"}, "UNKNOWN_FIELD"),
        ({"field": "name", "op": "gt", "value": 2}, "INVALID_FILTER_VALUE"),
    ],
)
def test_invalid_live_rules_clear_counts_instead_of_showing_unfiltered_success(condition, code):
    result = explore(eligibility=[condition])
    assert not result["valid"]
    assert code in codes(result)
    assert result["cohort"] is result["partitions"] is result["unassigned"] is None
    assert result["dataset"]["totalSlides"] == 32
    result = explore(rules={"test": [condition]})
    assert result["cohort"]["totalSlides"] == 32
    assert result["partitions"] is result["unassigned"] is None
    assert code in codes(result)


def test_cross_slide_rule_overlap_clears_live_partition_counts():
    result = explore(
        rules={
            "train": [{"field": "Slide_ID", "op": "eq", "value": "s00-0"}],
            "test": [{"field": "Slide_ID", "op": "eq", "value": "s00-1"}],
        }
    )
    assert "OVERLAPPING_PATIENT_RULES" in codes(result)
    assert not result["valid"]
    assert result["partitions"] is None


def test_live_and_preview_cannot_hide_bad_numeric_values_behind_another_matching_slide():
    store = MemoryStore()
    store.rows[1]["attributes"]["number"] = "unknown"
    rules = {"test": [{"field": "number", "op": "gte", "value": 0}]}
    store.draft["payload"]["spec"]["split"]["rules"] = rules
    assert "INVALID_FILTER_VALUE" in codes(explore(store, rules=rules))
    assert "INVALID_FILTER_VALUE" in codes(preview(store))


def test_live_regex_budget_is_shared_by_eligibility_and_partition_rules(monkeypatch):
    # Each successful regex costs two milliseconds. Eligibility consumes 64ms;
    # continuing into fixed rules must hit the same request's 65ms budget.
    ticks = iter(index / 500 for index in range(1000))
    monkeypatch.setattr("histopilot.application.protocols.time.monotonic", lambda: next(ticks))
    monkeypatch.setattr("histopilot.application.protocols.REGEX_TOTAL_SECONDS", 0.065)
    result = explore(
        eligibility=[{"field": "name", "op": "regex", "value": "ordinary"}],
        rules={"test": [{"field": "grade", "op": "regex", "value": "2"}]},
    )
    assert "REGEX_TIMEOUT" in codes(result)
    assert result["cohort"]["totalSlides"] == 32
    assert result["partitions"] is None


def test_missing_patients_show_cohort_but_never_implicitly_enable_slide_grouping():
    store = MemoryStore()
    store.rows[0]["patientId"] = None
    result = explore(store)
    assert result["cohort"]["unlinkedSlideCount"] == 1
    assert result["cohort"]["fallbackSlideCount"] == 0
    assert result["partitions"] is None
    assert "MISSING_PATIENT_ID" in codes(result)


def test_acknowledged_fallback_groups_are_distinct_from_verified_patient_counts():
    store = MemoryStore()
    for row in store.rows[:2]:
        row.update(patientId=row["slideId"], patientIdSource="slide_fallback")
    live = explore(store)
    assert live["valid"], live["findings"]
    assert live["cohort"]["patientCount"] == 15
    assert live["cohort"]["fallbackSlideCount"] == 2
    assert live["cohort"]["groupCount"] == 17
    assert live["cohort"]["sample"][0]["patientIdSource"] == "slide_fallback"
    result = preview(store)
    assert result["canFreeze"], result["findings"]
    assert result["summary"]["includedPatients"] == 15
    assert result["summary"]["includedGroups"] == 17
    assert result["summary"]["grouping"] == "patient_with_slide_fallback"
    assert any(item["code"] == "SLIDE_ID_FALLBACK_GROUPING" for item in result["findings"])
    assert any(row.get("patientIdSource") == "slide_fallback" for row in result["memberships"])


def test_fallback_identity_collision_blocks_live_and_final_assignments():
    store = MemoryStore()
    row = store.rows[0]
    row.update(patientId=row["slideId"], patientIdSource="slide_fallback")
    store.rows[1]["patientId"] = row["slideId"]
    assert "FALLBACK_PATIENT_ID_COLLISION" in codes(explore(store))
    assert "FALLBACK_PATIENT_ID_COLLISION" in codes(preview(store))


def test_generated_split_modes_leave_unfixed_live_cohort_for_generation():
    result = explore(
        splitMode="kfold", rules={"test": [{"field": "grade", "op": "eq", "value": "2"}]}
    )
    assert result["valid"]
    assert result["partitions"]["train"]["selection"] == "none"
    assert result["partitions"]["train"]["expanded"]["totalSlides"] == 0
    assert result["unassigned"]["totalSlides"] == 24
