"""Development cohort selection is independent of later inference cohorts.

A development (version 4) protocol selects a training pool and, optionally, a fixed
validation pool. Rows outside both pools stay outside the protocol; they are not a
reserved test set.
"""

import copy
import json
from collections import Counter, defaultdict

import pytest
from pydantic import ValidationError

from histopilot.application.protocols import ProtocolService
from histopilot.schemas.protocols import ProtocolExploreRequest, ProtocolSpec
from histopilot.storage.project_lock import StorageError

DATASET_ID = "dataset-" + "a" * 64
MODES = ("kfold", "monte_carlo", "leave_one_domain_out", "nested_kfold", "held_out")
# Patients whose partition value is "test": outside every development pool.
EXTERNAL_TEST = {f"patient-{patient:03}" for patient in range(100, 120)}
FIXED_VALIDATION = {f"patient-{patient:03}" for patient in range(80, 100)}


def source_rows():
    return [
        {
            "slideId": f"slide-{patient:03}-{slide}",
            "patientId": f"patient-{patient:03}",
            "patientIdSource": "source",
            "slidePath": None,
            "attributes": {
                "label": str(patient % 2),
                "site": f"site-{patient // 20}",
                "partition": "train" if patient < 100 else "test",
                "official": "train" if patient < 80 else "val" if patient < 100 else "test",
            },
        }
        for patient in range(120)
        for slide in range(2)
    ]


def specification(mode="kfold", *, fixed_validation=False, imported=False, **split):
    field = "official" if fixed_validation else "partition"
    roles = ["train", "val"] if fixed_validation else ["train"]
    pools = {
        "source": "imported" if imported else "rules",
        "trainSelection": "rules",
        "validationSource": "fixed" if fixed_validation else "training_fraction",
        "rules": {}
        if imported
        else {role: [{"field": field, "op": "eq", "value": role}] for role in roles},
        "imported": {"partitionField": field, "partitionLabels": {role: role for role in roles}}
        if imported
        else None,
    }
    config = {
        "version": 4,
        "mode": mode,
        "folds": 5,
        "outerFolds": 5,
        "innerFolds": 4,
        "repeats": 3,
        "seeds": [42],
        "validationFraction": 0.2,
        "testFraction": 0.2,
        "stratify": True,
        "pools": pools,
    }
    if mode == "leave_one_domain_out":
        config.update(domainField="site", domainPolicy="all")
    config.update(split)
    return {
        "datasetId": DATASET_ID,
        "target": {
            "field": "label",
            "task": "binary_classification",
            "unit": "patient",
            "classes": ["negative", "positive"],
            "labels": {"0": "negative", "1": "positive"},
            "positiveClass": "positive",
        },
        "split": config,
    }


class Store:
    def __init__(self, mode="kfold", **split):
        self.rows = source_rows()
        self.draft = {
            "id": "draft-pools",
            "revision": 1,
            "kind": "experiment",
            "status": "editable",
            "payload": {"type": "analysis-protocol", "spec": specification(mode, **split)},
        }
        self.dataset = {
            "id": DATASET_ID,
            "contentHash": "a" * 64,
            "manifest": {
                "kind": "dataset",
                "dictionary": [
                    {"key": key, "sourceColumn": key, "owner": "slide", "type": "text"}
                    for key in ("label", "site", "partition", "official")
                ],
            },
        }

    def get_draft(self, _identity):
        return copy.deepcopy(self.draft)

    def get_dataset(self, _identity):
        return copy.deepcopy(self.dataset)

    def read_artifact(self, _identity, name):
        assert name == "records.json"
        return json.dumps(self.rows).encode()


def development_store(mode="kfold", **kwargs):
    return Store(mode, **kwargs)


def preview(store):
    return ProtocolService(store).preview("draft-pools", 1)


def successful(store):
    result = preview(store)
    assert result["canFreeze"], result["findings"]
    return result


def codes(result):
    return {finding["code"] for finding in result["findings"] if finding["severity"] == "error"}


def plan_rows(result):
    grouped = defaultdict(list)
    for row in result["memberships"]:
        grouped[row["planId"]].append(row)
    return dict(grouped)


def patients(rows, role=None):
    return {row["patientId"] for row in rows if role is None or row["partition"] == role}


def assert_grouped(result):
    summaries = {plan["planId"]: plan for plan in result["partitions"]}
    assert len(summaries) == len(result["partitions"])
    assert set(summaries) == set(plan_rows(result))
    for plan_id, rows in plan_rows(result).items():
        by_patient = defaultdict(set)
        for row in rows:
            by_patient[row["patientId"]].add(row["partition"])
        assert all(len(roles) == 1 for roles in by_patient.values())
        assert set(Counter(row["patientId"] for row in rows).values()) == {2}
        assert len({row["slideId"] for row in rows}) == len(rows)
        for role in {row["partition"] for row in rows}:
            assert summaries[plan_id][role]["patients"] == len(patients(rows, role))
            assert summaries[plan_id][role]["slides"] == 2 * len(patients(rows, role))


def assert_blocked(store):
    try:
        result = preview(store)
    except StorageError as error:
        assert error.code == "INVALID_PROTOCOL_SPEC"
        return
    assert not result["canFreeze"]
    assert any(finding["severity"] == "error" for finding in result["findings"])


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("fixed_validation", [False, True])
def test_all_development_strategies_preserve_grouping_without_final_plans(mode, fixed_validation):
    store = development_store(mode, fixed_validation=fixed_validation)
    result = successful(store)
    assert result["summary"]["scope"] == "development"
    assert result["summary"]["splitVersion"] == 4
    assert result["summary"]["finalPlanCount"] == 0
    assert result["summary"]["poolCounts"]["test"]["groups"] == 0
    assert result["summary"]["includedPatients"] == 100
    assert all(row["pool"] == "development" for row in result["memberships"])
    assert all(plan["phase"] != "final" for plan in result["partitions"])
    assert not patients(result["memberships"]) & EXTERNAL_TEST
    assert_grouped(result)
    assert any(row["partition"] == "test" for row in result["memberships"])
    assert result["summary"]["roleDescriptions"]["test"].startswith("Development assessment")
    if mode in {"kfold", "nested_kfold", "leave_one_domain_out"}:
        assert result["summary"]["oofCoverage"]["complete"]


def test_missing_validation_percentage_defaults_to_fifteen_percent():
    store = development_store()
    store.draft["payload"]["spec"]["split"].pop("validationFraction")
    result = successful(store)
    assert result["spec"]["split"]["validationFraction"] == 0.15
    for plan in result["partitions"]:
        # Each fold has 80 development groups; 15% reserves 12 for stopping.
        assert plan["val"]["groups"] == 12
        assert plan["train"]["groups"] == 68


def test_explicit_twenty_percent_validation_setting_is_preserved():
    result = successful(development_store(validationFraction=0.2))
    assert result["spec"]["split"]["validationFraction"] == 0.2
    for plan in result["partitions"]:
        assert plan["val"]["groups"] == 16
        assert plan["train"]["groups"] == 64


@pytest.mark.parametrize("mode", MODES)
def test_fixed_validation_never_becomes_cv_test_tuning_or_training(mode):
    result = successful(development_store(mode, fixed_validation=True))
    for rows in plan_rows(result).values():
        assert patients(rows, "val") == FIXED_VALIDATION
        for role in ("train", "test", "tune"):
            assert patients(rows, role).isdisjoint(FIXED_VALIDATION)
    assert result["summary"]["poolCounts"]["train"]["patients"] == 80
    assert result["summary"]["poolCounts"]["val"]["patients"] == 20
    assert_grouped(result)


@pytest.mark.parametrize("mode", MODES)
def test_pool_assignments_are_reproducible_after_row_and_seed_reordering(mode):
    store = development_store(mode, seeds=[7, 42])
    result = successful(store)
    store.rows.reverse()
    store.draft["payload"]["spec"]["split"]["seeds"].reverse()
    assert successful(store) == result


def test_kfold_oof_coverage_applies_to_selected_training_pool_only():
    result = successful(development_store("kfold"))
    tests = Counter()
    for rows in plan_rows(result).values():
        tests.update(patients(rows, "test"))
    assert set(tests) == {f"patient-{patient:03}" for patient in range(100)}
    assert set(tests.values()) == {1}
    assert result["summary"]["oofCoverage"]["groups"] == 100
    assert result["summary"]["oofCoverage"]["complete"] is True
    for plan in result["partitions"]:
        for role in ("train", "val", "test"):
            counts = plan[role]["classes"]
            assert abs(counts["negative"] - counts["positive"]) <= 1


def test_nested_inner_plans_exclude_outer_assessment_groups():
    result = successful(development_store("nested_kfold"))
    plans = list(plan_rows(result).values())
    outer = [rows for rows in plans if rows[0]["phase"] == "outer"]
    inner = [rows for rows in plans if rows[0]["phase"] == "inner"]
    assert len(outer) == 5
    assert len(inner) == 20
    for outer_rows in outer:
        outer_fold = outer_rows[0]["outerFold"]
        outer_test = patients(outer_rows, "test")
        for inner_rows in inner:
            if inner_rows[0]["outerFold"] == outer_fold:
                assert patients(inner_rows).isdisjoint(outer_test | EXTERNAL_TEST)
                assert {row["partition"] for row in inner_rows} == {"train", "val", "tune"}
    assert_grouped(result)


def test_domain_cv_filters_fixed_validation_from_its_held_out_site():
    store = development_store("leave_one_domain_out", fixed_validation=True)
    excluded = {f"patient-{patient:03}" for patient in range(80, 90)}
    for row in store.rows:
        if row["patientId"] in excluded:
            row["attributes"]["site"] = "site-0"
    result = successful(store)
    summaries = {plan["planId"]: plan for plan in result["partitions"]}
    for plan_id, rows in plan_rows(result).items():
        if rows[0].get("domain") == "site-0":
            assert patients(rows, "val") == FIXED_VALIDATION - excluded
            omitted = summaries[plan_id]["excludedValidation"]
            assert set(omitted["groupIds"]) == excluded
            assert omitted["groups"] == 10
            assert omitted["slides"] == 20
        else:
            assert patients(rows, "val") == FIXED_VALIDATION


def test_domain_cv_rejects_fixed_validation_entirely_from_held_out_site():
    store = development_store("leave_one_domain_out", fixed_validation=True)
    for row in store.rows:
        if row["patientId"] in FIXED_VALIDATION:
            row["attributes"]["site"] = "site-0"
    assert_blocked(store)


@pytest.mark.parametrize("invalid_domain", (None, "conflicting"))
def test_domain_cv_requires_resolved_consistent_domains_for_fixed_validation(invalid_domain):
    store = development_store("leave_one_domain_out", fixed_validation=True)
    store.rows[160]["attributes"]["site"] = invalid_domain
    assert_blocked(store)


def test_domain_selection_uses_domains_in_the_development_pool_not_outside_rows():
    store = development_store(
        "leave_one_domain_out", domainPolicy="selected", heldOutDomains=["site-5"]
    )
    result = preview(store)
    assert not result["canFreeze"]
    assert "UNKNOWN_HELD_OUT_DOMAIN" in codes(result)


def test_missing_training_pool_rules_do_not_silently_assign_every_row():
    store = development_store()
    store.draft["payload"]["spec"]["split"]["pools"]["rules"]["train"] = []
    result = preview(store)
    assert not result["canFreeze"]
    assert "TRAIN_POOL_REQUIRED" in codes(result)


@pytest.mark.parametrize("invalid", ("overlap", "empty_fixed_val"))
def test_conflicting_or_incomplete_pool_selections_block_freezing(invalid):
    store = development_store(fixed_validation=True)
    pools = store.draft["payload"]["spec"]["split"]["pools"]
    if invalid == "overlap":
        pools["rules"]["train"] = [{"field": "Slide_ID", "op": "regex", "value": "-0$"}]
        pools["rules"]["val"] = [{"field": "Slide_ID", "op": "regex", "value": "-1$"}]
    else:
        pools["rules"]["val"] = [{"field": "partition", "op": "eq", "value": "missing"}]
    assert_blocked(store)


@pytest.mark.parametrize("mode", MODES)
def test_predefined_train_and_validation_values_are_respected_by_all_strategies(mode):
    result = successful(development_store(mode, fixed_validation=True, imported=True))
    for rows in plan_rows(result).values():
        assert patients(rows, "val") == FIXED_VALIDATION
        assert patients(rows).isdisjoint(EXTERNAL_TEST)
    assert_grouped(result)


def test_predefined_partition_cannot_divide_one_patients_slides():
    store = development_store(fixed_validation=True, imported=True)
    store.rows[0]["attributes"]["official"] = "val"
    result = preview(store)
    assert not result["canFreeze"]
    assert "IMPORTED_PATIENT_LEAKAGE" in codes(result)


def test_predefined_pool_column_alias_cannot_be_a_model_input():
    store = development_store(imported=True, fixed_validation=True)
    store.dataset["manifest"]["dictionary"].append(
        {"key": "allocation", "sourceColumn": "official", "owner": "slide", "type": "text"}
    )
    for row in store.rows:
        row["attributes"]["allocation"] = row["attributes"]["official"]
    store.draft["payload"]["spec"]["predictors"] = ["allocation"]
    assert_blocked(store)


def test_membership_budget_counts_fixed_validation(monkeypatch):
    # Training-only CV uses 800 rows; every fold also lists the 40 fixed validation slides.
    monkeypatch.setattr("histopilot.application.protocols.MAX_MEMBERSHIPS", 900)
    result = preview(development_store("kfold", fixed_validation=True))
    assert not result["canFreeze"]
    assert result["memberships"] == []
    assert result["partitions"] == []
    assert "PROTOCOL_MEMBERSHIP_LIMIT" in codes(result)


@pytest.mark.parametrize("excluded_by", ["eligibility", "missing_target"])
def test_excluding_fixed_validation_cannot_silently_generate_a_replacement(excluded_by):
    store = development_store(fixed_validation=True, imported=True)
    spec = store.draft["payload"]["spec"]
    if excluded_by == "eligibility":
        spec["eligibility"] = [{"field": "official", "op": "ne", "value": "val"}]
    else:
        for row in store.rows:
            if row["attributes"]["official"] == "val":
                row["attributes"]["label"] = None
        spec["target"]["missing"] = "exclude"
    result = preview(store)
    assert not result["canFreeze"]
    assert "VAL_POOL_EMPTY" in codes(result)
    assert result["memberships"] == []

    # Revising the pools makes the change of validation policy explicit.
    spec["split"]["pools"]["validationSource"] = "training_fraction"
    spec["split"]["pools"]["imported"]["partitionLabels"]["val"] = "train"
    revised = successful(store)
    assert_grouped(revised)
    assert all(plan["val"]["groups"] > 0 for plan in revised["partitions"])


@pytest.mark.parametrize("imported", [False, True])
def test_combined_file_selects_development_before_label_validation(imported):
    store = development_store(imported=imported)
    for row in store.rows:
        if row["patientId"] in EXTERNAL_TEST:
            row["attributes"]["label"] = None
    result = successful(store)
    assert result["summary"]["includedSlides"] == 200
    assert result["summary"]["excludedSlides"] == 40
    assert result["summary"]["labelExclusions"] == {}
    request = ProtocolExploreRequest(
        datasetId=result["spec"]["datasetId"],
        targetField="label",
        split=result["spec"]["split"],
    )
    live = ProtocolService(store).explore(request)
    assert live["valid"], live["findings"]
    assert live["partitions"]["train"]["expanded"]["totalSlides"] == 200
    assert live["partitions"]["test"]["expanded"]["totalSlides"] == 0
    assert live["unassigned"]["totalSlides"] == 40
    assert live["target"]["distinctCount"] == 2
    assert all(item["value"] is not None for item in live["target"]["values"])


def test_all_eligible_development_needs_no_reserved_population():
    store = development_store()
    pools = store.draft["payload"]["spec"]["split"]["pools"]
    pools.update(trainSelection="remaining", rules={})
    result = successful(store)
    assert result["summary"]["includedPatients"] == 120
    assert result["summary"]["evaluationPlanCount"] == 5


@pytest.mark.parametrize("imported", [False, True])
def test_unlinked_outside_records_do_not_block_but_selected_unlinked_records_do(imported):
    store = development_store(imported=imported)
    for row in store.rows:
        if row["patientId"] in EXTERNAL_TEST:
            row["patientId"] = None
    result = successful(store)
    assert result["summary"]["includedSlides"] == 200
    request = ProtocolExploreRequest(
        datasetId=result["spec"]["datasetId"], targetField="label", split=result["spec"]["split"]
    )
    live = ProtocolService(store).explore(request)
    assert live["valid"], live["findings"]
    assert live["unassigned"]["unlinkedSlideCount"] == 40
    assert live["partitions"]["train"]["expanded"]["unlinkedSlideCount"] == 0

    store.rows[0]["patientId"] = None
    blocked = preview(store)
    assert not blocked["canFreeze"]
    assert "MISSING_PATIENT_ID" in {item["code"] for item in blocked["findings"]}
    assert blocked["memberships"] == []
    live = ProtocolService(store).explore(request)
    assert not live["valid"]
    assert live["partitions"] is None
    assert "MISSING_PATIENT_ID" in {item["code"] for item in live["findings"]}


def test_unlinked_rows_are_never_silently_dropped_from_all_eligible_training():
    store = development_store()
    store.draft["payload"]["spec"]["split"]["pools"].update(trainSelection="remaining", rules={})
    store.rows[0]["patientId"] = None
    result = preview(store)
    assert not result["canFreeze"]
    assert "MISSING_PATIENT_ID" in {item["code"] for item in result["findings"]}


@pytest.mark.parametrize("imported", [False, True])
def test_test_pool_input_cannot_enter_a_development_protocol(imported):
    spec = specification(imported=imported)
    test_pool = [{"field": "partition", "op": "eq", "value": "test"}]
    if imported:
        spec["split"]["pools"]["imported"]["partitionLabels"]["test"] = "test"
    else:
        spec["split"]["pools"]["rules"]["test"] = test_pool
    with pytest.raises(ValidationError, match="cannot reserve test data"):
        ProtocolSpec.model_validate(spec)


def test_patient_overlap_between_training_and_fixed_validation_is_blocked():
    store = development_store(fixed_validation=True)
    store.draft["payload"]["spec"]["split"]["pools"]["rules"]["val"] = [
        {"field": "slideId", "op": "eq", "value": "slide-000-1"}
    ]
    result = preview(store)
    assert not result["canFreeze"]
    assert "OVERLAPPING_PATIENT_RULES" in {item["code"] for item in result["findings"]}
