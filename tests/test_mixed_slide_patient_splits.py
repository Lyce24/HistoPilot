"""Mixed slide grades preserve patient isolation and per-slide supervision."""

import runpy
from collections import defaultdict
from pathlib import Path

import pytest

from histopilot.application.development_splits import ALGORITHM_V4, ALGORITHM_V4_MIXED

support = runpy.run_path(str(Path(__file__).with_name("test_development_protocols.py")))
pools = support["support"]


def mixed_store(mode="kfold", *, stratify=True, fixed_validation=False):
    store = support["development_store"](mode, stratify=stratify, fixed_validation=fixed_validation)
    store.draft["payload"]["spec"]["target"]["unit"] = "slide"
    for row in store.rows:
        patient = int(row["patientId"].split("-")[1])
        if patient < 40 and row["slideId"].endswith("-1"):
            row["attributes"]["label"] = str(1 - patient % 2)
    return store


@pytest.mark.parametrize("mode", pools["MODES"])
@pytest.mark.parametrize("stratify", [True, False])
@pytest.mark.parametrize("fixed_validation", [True, False])
def test_mixed_slide_labels_stay_with_patient_in_every_plan(mode, stratify, fixed_validation):
    store = mixed_store(mode, stratify=stratify, fixed_validation=fixed_validation)
    result = pools["successful"](store)
    pools["assert_grouped"](result)
    assert result["summary"]["algorithm"] == ALGORITHM_V4_MIXED
    assert result["summary"]["patientClassCounts"] == {"negative": 70, "positive": 70}
    assert any(
        item["code"] == "MIXED_SLIDE_LABEL_PATIENT_GROUPS" and item["severity"] == "warning"
        for item in result["findings"]
    )
    labels = {
        row["slideId"]: {"0": "negative", "1": "positive"}[row["attributes"]["label"]]
        for row in store.rows
    }
    assert all(row["label"] == labels[row["slideId"]] for row in result["memberships"])
    assert not pools["patients"](result["memberships"]) & pools["EXTERNAL_TEST"]
    summaries = {plan["planId"]: plan for plan in result["partitions"]}
    for plan_id, rows in pools["plan_rows"](result).items():
        represented = defaultdict(set)
        for row in rows:
            represented[(row["partition"], row["label"])].add(row["patientId"])
        for role in {row["partition"] for row in rows}:
            assert summaries[plan_id][role]["classes"] == {
                label: len(represented[(role, label)]) for label in ("negative", "positive")
            }


def test_mixed_patient_grade_order_does_not_change_assignments_or_preview():
    store = mixed_store()
    first = pools["successful"](store)
    store.rows.reverse()
    second = pools["successful"](store)
    assert first["memberships"] == second["memberships"]
    assert first["previewHash"] == second["previewHash"]


def test_homogeneous_slide_target_retains_existing_algorithm_and_assignments():
    store = support["development_store"]()
    first = pools["successful"](store)
    store.draft["payload"]["spec"]["target"]["unit"] = "slide"
    second = pools["successful"](store)
    assert second["summary"]["algorithm"] == ALGORITHM_V4
    assert first["memberships"] == second["memberships"]


def test_mixed_patient_target_remains_blocked():
    store = mixed_store()
    store.draft["payload"]["spec"]["target"]["unit"] = "patient"
    result = pools["preview"](store)
    assert not result["canFreeze"]
    assert "MIXED_PATIENT_LABELS" in {item["code"] for item in result["findings"]}


def test_legacy_mixed_slide_target_remains_blocked():
    store = mixed_store()
    store.draft["payload"]["spec"]["split"] = pools["specification"]()["split"]
    result = pools["preview"](store)
    assert not result["canFreeze"]
    assert "MIXED_PATIENT_STRATIFICATION_UNSUPPORTED" in {
        item["code"] for item in result["findings"]
    }


def test_multiple_slides_of_one_patient_do_not_satisfy_class_group_minimum():
    store = mixed_store()
    for row in store.rows:
        row["attributes"]["label"] = "0"
    store.rows[1]["attributes"]["label"] = "1"
    store.draft["payload"]["spec"]["constraints"] = {"minPatientsPerClass": 2}
    result = pools["preview"](store)
    assert not result["canFreeze"]
    assert result["summary"]["patientClassCounts"] == {"negative": 100, "positive": 1}
    assert "INSUFFICIENT_CLASS_PATIENTS" in {item["code"] for item in result["findings"]}
