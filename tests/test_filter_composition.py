"""Composed filters select the same slides in preview, split pools and test cohorts."""

import runpy
from pathlib import Path

import pytest
from pydantic import ValidationError

from histopilot.application.evaluations import EvaluationService
from histopilot.application.protocols import FilterEvaluator, FilterFailure, ProtocolService
from histopilot.schemas.evaluations import EvaluationSpec
from histopilot.schemas.protocols import Condition, ConditionGroup, ProtocolExploreRequest
from histopilot.storage.filesystem import LocalFilesystem

support = runpy.run_path(str(Path(__file__).with_name("test_protocols.py")))
development = runpy.run_path(str(Path(__file__).with_name("test_development_protocols.py")))


def rule(field, value, op="eq"):
    return {"field": field, "op": op, "value": value}


def request(conditions):
    return ProtocolExploreRequest(datasetId=support["DATASET_ID"], eligibility=conditions)


def test_membership_and_nested_or_select_exact_cohorts_before_splitting():
    store = support["MemoryStore"]()
    store.dataset["manifest"]["dictionary"].append({"key": "cohort", "sourceColumn": "cohort", "type": "text"})
    for index, row in enumerate(store.rows):
        row["attributes"]["cohort"] = ("TCGA", "SurGen", "RIH")[index % 3]
    conditions = [
        rule("cohort", ["TCGA", "SurGen"], "in"),
        {"op": "any", "conditions": [rule("grade", "2"), rule("stage", "Primary")]},
    ]
    expected = {
        row["slideId"]
        for row in store.rows
        if row["attributes"]["cohort"] in {"TCGA", "SurGen"}
        and (row["attributes"]["grade"] == "2" or row["attributes"]["stage"] == "Primary")
    }
    result = ProtocolService(store).explore(request(conditions))
    assert result["valid"], result["findings"]
    assert result["cohort"]["totalSlides"] == len(expected)
    evaluator = FilterEvaluator()
    assert {
        row["slideId"]
        for row in store.rows
        if evaluator.conjunction(row, request(conditions).eligibility)
    } == expected


def test_nested_all_does_not_combine_evidence_from_different_patient_slides():
    group = {"op": "any", "conditions": [
        {"op": "all", "conditions": [rule("stage", "Primary"), rule("stage", "Metastatic")]},
        rule("grade", "2"),
    ]}
    result = ProtocolService(support["MemoryStore"]()).explore(request([group]))
    assert result["valid"], result["findings"]
    assert result["cohort"]["totalSlides"] == 8


def test_any_pool_expands_matching_patients_and_keeps_live_counts_consistent():
    store = development["development_store"]()
    spec = store.draft["payload"]["spec"]
    spec["split"]["pools"]["rules"]["train"] = [{"op": "any", "conditions": [
        rule("site", "site-0"), rule("site", "site-1"),
    ]}]
    service = ProtocolService(store)
    result = service.preview("draft-test", 1)
    assert result["canFreeze"], result["findings"]
    assert result["summary"]["includedSlides"] == 80
    assert len({row["patientId"] for row in result["memberships"]}) == 40
    live = service.explore(ProtocolExploreRequest(datasetId=spec["datasetId"], split=spec["split"]))
    assert live["valid"], live["findings"]
    assert live["partitions"]["train"]["expanded"]["totalSlides"] == 80


def test_test_cohort_uses_same_nested_composition(tmp_path):
    store = support["MemoryStore"]()
    store.folder = tmp_path
    store.draft["payload"] = {"type": "evaluation-cohort", "spec": {
        "datasetId": support["DATASET_ID"],
        "eligibility": [{"op": "any", "conditions": [rule("grade", "2"), rule("stage", "Primary")]}],
    }}
    result = EvaluationService(store, LocalFilesystem((tmp_path,))).preview("draft-test", 1)
    assert result["canFreeze"], result["findings"]
    assert result["summary"]["includedSlides"] == 20


@pytest.mark.parametrize("consumer", ["explore", "preview", "test"])
def test_nested_unknown_fields_are_reported_before_membership_is_frozen(consumer, tmp_path):
    conditions = [{"op": "any", "conditions": [rule("grade", "1"), rule("missing_field", "x")]}]
    store = support["MemoryStore"]()
    store.folder = tmp_path
    if consumer == "explore":
        result = ProtocolService(store).explore(request(conditions))
    elif consumer == "preview":
        store.draft["payload"]["spec"]["eligibility"] = conditions
        result = ProtocolService(store).preview("draft-test", 1)
    else:
        store.draft["payload"] = {"type": "evaluation-cohort", "spec": {
            "datasetId": support["DATASET_ID"], "eligibility": conditions,
        }}
        result = EvaluationService(store, LocalFilesystem((tmp_path,))).preview("draft-test", 1)
    assert "UNKNOWN_FIELD" in {item["code"] for item in result["findings"]}
    assert not result.get("canFreeze", result.get("valid"))


def test_nested_regex_is_validated_even_without_rows_or_behind_matching_alternative():
    conditions = request([{"op": "any", "conditions": [rule("grade", "1"), rule("name", "(", "regex")]}]).eligibility
    with pytest.raises(FilterFailure, match="regular expression is invalid"):
        FilterEvaluator().prepare(conditions)
    group = ConditionGroup.model_validate({"op": "any", "conditions": [rule("grade", "1"), rule("name", 2, "lt")]})
    with pytest.raises(FilterFailure, match="nonnumeric"):
        FilterEvaluator().matches(support["records"]()[0], group)


def test_flat_legacy_filters_preserve_their_serialized_form_and_and_meaning():
    conditions = [rule("grade", "2"), rule("stage", "Primary")]
    parsed = request(conditions)
    assert parsed.model_dump(mode="json")["eligibility"] == conditions
    assert all(isinstance(item, Condition) for item in parsed.eligibility)
    assert ProtocolService(support["MemoryStore"]()).explore(parsed)["cohort"]["totalSlides"] == 4
    assert EvaluationSpec(datasetId=support["DATASET_ID"], eligibility=[]).eligibility == []


def test_filter_composition_has_bounded_size_and_rejects_empty_groups():
    with pytest.raises(ValidationError):
        request([{"op": "any", "conditions": []}])
    with pytest.raises(ValidationError, match="at most 30 conditions"):
        request([{"op": "any", "conditions": [rule("grade", "1")] * 20}] * 2)
    nested = rule("grade", "1")
    for _ in range(5):
        nested = {"op": "all", "conditions": [nested]}
    with pytest.raises(ValidationError, match="at most four nested levels"):
        request([nested])
