"""Slide targets may keep each patient's slides in one fold; labels and scoring stay per slide."""

from collections import defaultdict

import pytest
from pydantic import ValidationError
from test_target_splits import TARGET, TRAINING, frozen, publish_revised_rows, requested_settings
from test_target_splits import construction as construction  # noqa: F401 - pytest fixture

from histopilot.schemas.protocols import SplitSpec
from histopilot.storage.project_lock import StorageError

GROUPED = {**TRAINING, "groupByPatient": True}


def case_rows(*, missing_patient=False):
    """Multi-part biopsy cases: four parts per case whose grades differ by part."""
    rows = []
    for case in range(18):
        for part in range(4):
            rows.append(
                {
                    "slideId": f"case{case:02}-{'ABCD'[part]}",
                    "patientId": None
                    if missing_patient and (case, part) == (1, 1)
                    else f"case{case:02}",
                    "attributes": {
                        "label": str((case + part // 2) % 2),
                        "Requested_Split": "train" if case < 15 else "test",
                    },
                }
            )
    return rows


def slide_target_split(construction, rows):
    store, service, spec, _ = publish_revised_rows(construction, rows)
    spec.update(splitUnit="slide", target={**TARGET, "unit": "slide"})
    result, _draft, _preview = frozen((store, service, spec, rows), requested_settings("rules"))
    return service, result


def roles_per_case(memberships):
    roles = defaultdict(lambda: defaultdict(set))
    for row in memberships:
        roles[row["planId"]][row["patientId"]].add(row["partition"])
    return roles


def test_grouped_slide_folds_keep_each_case_in_one_fold_and_validation_side(construction):
    service, split = slide_target_split(construction, case_rows())
    protocol = service.derive_protocol(split["id"], GROUPED)
    manifest = protocol["manifest"]
    assert manifest["spec"]["splitUnit"] == "slide"
    assert manifest["spec"]["target"]["unit"] == "slide"
    assert manifest["spec"]["split"]["groupByPatient"] is True
    memberships = manifest["memberships"]
    # Every plan places each case entirely in train, validation or the assessment fold.
    assert all(
        len(parts) == 1 for plan in roles_per_case(memberships).values() for parts in plan.values()
    )
    # Slides keep their own grades even when a case mixes grades.
    slide_labels = {row["slideId"]: row["label"] for row in memberships}
    assert slide_labels["case00-A"] != slide_labels["case00-C"]
    assert manifest["summary"]["grouping"] == "slide_labels_patient_folds"
    assert manifest["summary"]["includedPatients"] == 0  # slide-level reporting is unchanged
    assert manifest["summary"]["includedGroups"] == 15
    codes = {finding["code"] for finding in manifest["findings"]}
    assert "MIXED_SLIDE_LABEL_PATIENT_GROUPS" in codes
    # Each case lands in exactly one assessment fold per split seed.
    assessed = defaultdict(set)
    for row in memberships:
        if row["partition"] == "test":
            assessed[(row["seed"], row["patientId"])].add(row["planId"])
    assert assessed and all(len(plans) == 1 for plans in assessed.values())
    assert service.derive_protocol(split["id"], GROUPED)["id"] == protocol["id"]


def test_default_slide_folds_still_assign_slides_independently(construction):
    service, split = slide_target_split(construction, case_rows())
    protocol = service.derive_protocol(split["id"], TRAINING)
    manifest = protocol["manifest"]
    assert "groupByPatient" not in manifest["spec"]["split"]
    assert manifest["summary"]["grouping"] == "slide"
    assert any(
        len(parts) > 1
        for plan in roles_per_case(manifest["memberships"]).values()
        for parts in plan.values()
    )
    grouped = service.derive_protocol(split["id"], GROUPED)
    assert grouped["id"] != protocol["id"]


def test_historical_designs_serialize_unchanged():
    base = SplitSpec.model_validate(TRAINING).model_dump(mode="json")
    assert "groupByPatient" not in base
    assert (
        SplitSpec.model_validate({**TRAINING, "groupByPatient": False}).model_dump(mode="json")
        == base
    )
    assert SplitSpec.model_validate(GROUPED).model_dump(mode="json") == {
        **base,
        "groupByPatient": True,
    }
    with pytest.raises(ValidationError, match="development training designs"):
        SplitSpec.model_validate({"groupByPatient": True})


def test_grouped_slide_folds_require_every_training_slide_to_name_its_case(construction):
    service, split = slide_target_split(construction, case_rows(missing_patient=True))
    with pytest.raises(StorageError) as caught:
        service.derive_protocol(split["id"], GROUPED)
    assert caught.value.code == "TRAINING_SPLIT_BLOCKED"
    assert "Patient_ID" in str(caught.value)
    # Independent slide folds need no case identity.
    assert (
        service.derive_protocol(split["id"], TRAINING)["manifest"]["summary"]["grouping"] == "slide"
    )


def test_oof_assembly_rejects_a_case_split_across_folds_when_grouping_was_requested(tmp_path):
    pytest.importorskip("torch")
    from histopilot.storage.io import write_json_atomic
    from histopilot.workers.train_batch import collect_results

    target = {
        "unit": "slide",
        "task": "binary_classification",
        "classes": ["a", "b"],
        "positiveClass": "b",
    }
    runs, memberships = [], {}
    for fold in range(2):
        split = f"fold-{fold}"
        rows = [
            {
                "slideId": f"s-{fold}-{i}",
                "patientId": "shared",
                "label": target["classes"][i],
                "labelIndex": i,
                "probabilities": [0.8, 0.2] if i == 0 else [0.2, 0.8],
            }
            for i in range(2)
        ]
        memberships[split] = [{**row, "partition": "test"} for row in rows]
        path = tmp_path / f"{split}.json"
        write_json_atomic(path, {"classOrder": target["classes"], "records": rows})
        runs.append(
            {
                "id": split,
                "candidateId": "candidate",
                "splitPlanId": split,
                "trainingSeed": 42,
                "status": "completed",
                "result": {"predictions": {"assessment": str(path)}},
            }
        )
    plan = {
        "batchId": "batch",
        "protocolId": "protocol",
        "target": target,
        "splitUnit": "slide",
        "configurations": [
            {"id": "candidate", "recipe": {"analysis": {"bootstrapResamples": 200}}}
        ],
        "memberships": memberships,
        "splitPlans": [{"id": f"fold-{i}", "seed": 42} for i in range(2)],
    }
    collect_results(
        plan, {"status": "completed", "runs": runs}, tmp_path
    )  # independent slide folds
    with pytest.raises(ValueError, match="assessment patient appears in multiple folds"):
        collect_results(
            {**plan, "groupByPatient": True}, {"status": "completed", "runs": runs}, tmp_path
        )
