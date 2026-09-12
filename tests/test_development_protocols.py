"""Development cohort selection is independent of later inference cohorts."""

import copy
import json
import runpy
from pathlib import Path

import pytest
from pydantic import ValidationError

from histopilot.application.protocols import ProtocolService
from histopilot.schemas.protocols import ProtocolExploreRequest, ProtocolSpec
from histopilot.storage.scientific import ScientificStore

support = runpy.run_path(str(Path(__file__).with_name("test_explicit_split_pools.py")))


def development_store(mode="kfold", **kwargs):
    store = support["Store"](mode, **kwargs)
    split = store.draft["payload"]["spec"]["split"]
    split["version"] = 4
    split["pools"]["rules"].pop("test", None)
    if split["pools"]["imported"]:
        split["pools"]["imported"]["partitionLabels"].pop("test", None)
    return store


@pytest.mark.parametrize("mode", support["MODES"])
@pytest.mark.parametrize("fixed_validation", [False, True])
def test_all_development_strategies_preserve_grouping_without_final_plans(mode, fixed_validation):
    store = development_store(mode, fixed_validation=fixed_validation)
    result = support["successful"](store)
    assert result["summary"]["scope"] == "development"
    assert result["summary"]["splitVersion"] == 4
    assert result["summary"]["finalPlanCount"] == 0
    assert result["summary"]["poolCounts"]["test"]["groups"] == 0
    assert result["summary"]["includedPatients"] == 100
    assert all(row["pool"] == "development" for row in result["memberships"])
    assert all(plan["phase"] != "final" for plan in result["partitions"])
    assert not support["patients"](result["memberships"]) & support["EXTERNAL_TEST"]
    support["assert_grouped"](result)
    assert any(row["partition"] == "test" for row in result["memberships"])
    assert result["summary"]["roleDescriptions"]["test"].startswith("Development assessment")
    if mode in {"kfold", "nested_kfold", "leave_one_domain_out"}:
        assert result["summary"]["oofCoverage"]["complete"]


@pytest.mark.parametrize("imported", [False, True])
def test_combined_file_selects_development_before_label_validation(imported):
    store = development_store(imported=imported)
    for row in store.rows:
        if row["patientId"] in support["EXTERNAL_TEST"]:
            row["attributes"]["label"] = None
    result = support["successful"](store)
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
    result = support["successful"](store)
    assert result["summary"]["includedPatients"] == 120
    assert result["summary"]["evaluationPlanCount"] == 5


@pytest.mark.parametrize("imported", [False, True])
def test_unlinked_outside_records_do_not_block_but_selected_unlinked_records_do(imported):
    store = development_store(imported=imported)
    for row in store.rows:
        if row["patientId"] in support["EXTERNAL_TEST"]:
            row["patientId"] = None
    result = support["successful"](store)
    assert result["summary"]["includedSlides"] == 200
    request = ProtocolExploreRequest(
        datasetId=result["spec"]["datasetId"], targetField="label", split=result["spec"]["split"]
    )
    live = ProtocolService(store).explore(request)
    assert live["valid"], live["findings"]
    assert live["unassigned"]["unlinkedSlideCount"] == 40
    assert live["partitions"]["train"]["expanded"]["unlinkedSlideCount"] == 0

    store.rows[0]["patientId"] = None
    blocked = support["preview"](store)
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
    result = support["preview"](store)
    assert not result["canFreeze"]
    assert "MISSING_PATIENT_ID" in {item["code"] for item in result["findings"]}


@pytest.mark.parametrize("imported", [False, True])
def test_test_pool_input_cannot_enter_a_development_protocol(imported):
    spec = support["specification"](imported=imported)
    spec["split"]["version"] = 4
    with pytest.raises(ValidationError, match="cannot reserve test data"):
        ProtocolSpec.model_validate(spec)


def test_patient_overlap_between_training_and_fixed_validation_is_blocked():
    store = development_store(fixed_validation=True)
    store.draft["payload"]["spec"]["split"]["pools"]["rules"]["val"] = [
        {"field": "slideId", "op": "eq", "value": "slide-000-1"}
    ]
    result = support["preview"](store)
    assert not result["canFreeze"]
    assert "OVERLAPPING_PATIENT_RULES" in {item["code"] for item in result["findings"]}


def test_development_freeze_does_not_rewrite_existing_legacy_protocol(tmp_path):
    store = ScientificStore(tmp_path, "project-test")
    store.initialize()
    imported = store.create_draft("import", "Source", {})
    source = support["Store"]()
    dataset = store.publish_dataset(
        imported["id"],
        expected_revision=1,
        manifest=source.dataset["manifest"],
        artifacts={"records.json": json.dumps(source.rows).encode()},
        operation_id="source",
    )
    service = ProtocolService(store)
    legacy_spec = copy.deepcopy(source.draft["payload"]["spec"])
    legacy_spec["datasetId"] = dataset["id"]
    legacy_draft = store.create_draft(
        "experiment", "Legacy", {"type": "analysis-protocol", "spec": legacy_spec}
    )
    legacy_preview = service.preview(legacy_draft["id"], 1)
    legacy = service.freeze(legacy_draft["id"], 1, legacy_preview["previewHash"], "legacy")
    original = store.get_configuration(legacy["id"])
    dev_spec = development_store().draft["payload"]["spec"]
    dev_spec["datasetId"] = dataset["id"]
    dev_draft = store.create_draft(
        "experiment", "Development", {"type": "analysis-protocol", "spec": dev_spec}
    )
    reviewed = service.preview(dev_draft["id"], 1)
    development = service.freeze(dev_draft["id"], 1, reviewed["previewHash"], "development")
    assert development["id"] != legacy["id"]
    assert store.get_configuration(legacy["id"]) == original
    assert service.freeze(legacy_draft["id"], 1, legacy_preview["previewHash"], "legacy") == legacy
