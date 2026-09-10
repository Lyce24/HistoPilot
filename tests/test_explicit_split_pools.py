"""User-selected training/test pools stay separate throughout version 3 CV."""

import copy
import json
import runpy
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from histopilot.api import create_app
from histopilot.application.protocols import ProtocolService
from histopilot.config import Settings
from histopilot.schemas.protocols import ProtocolExploreRequest
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import ScientificStore

DATASET_ID = "dataset-" + "a" * 64
MODES = ("kfold", "monte_carlo", "leave_one_domain_out", "nested_kfold", "held_out")
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
    roles = ["train", "val", "test"] if fixed_validation else ["train", "test"]
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
        "version": 3,
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


def preview(store):
    return ProtocolService(store).preview("draft-pools", 1)


def successful(store):
    result = preview(store)
    assert result["canFreeze"], result["findings"]
    return result


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


def test_missing_validation_percentage_defaults_to_fifteen_percent_for_v3():
    store = Store()
    store.draft["payload"]["spec"]["split"].pop("validationFraction")
    result = successful(store)
    assert result["spec"]["split"]["validationFraction"] == 0.15
    for plan in result["partitions"]:
        if plan["phase"] != "final":
            # Each fold has 80 development groups; 15% reserves 12 for stopping.
            assert plan["val"]["groups"] == 12
            assert plan["train"]["groups"] == 68


def test_explicit_twenty_percent_validation_setting_is_preserved_for_v3():
    result = successful(Store(validationFraction=0.2))
    assert result["spec"]["split"]["validationFraction"] == 0.2
    for plan in result["partitions"]:
        if plan["phase"] != "final":
            assert plan["val"]["groups"] == 16
            assert plan["train"]["groups"] == 64


@pytest.mark.parametrize("mode", MODES)
def test_external_test_is_reserved_until_final_evaluation_for_every_strategy(mode):
    result = successful(Store(mode))
    final = [rows for rows in plan_rows(result).values() if rows[0]["phase"] == "final"]
    cv = [rows for rows in plan_rows(result).values() if rows[0]["phase"] != "final"]
    assert len(final) == result["summary"]["finalPlanCount"] == 1
    assert bool(cv) == (mode != "held_out")
    for rows in cv:
        assert rows[0]["pool"] == "training"
        assert patients(rows).isdisjoint(EXTERNAL_TEST)
    for rows in final:
        assert rows[0]["pool"] == "external_test"
        assert patients(rows, "test") == EXTERNAL_TEST
        assert len(patients(rows, "train")) == 80
        assert len(patients(rows, "val")) == 20
    assert result["summary"]["poolCounts"]["train"]["patients"] == 100
    assert result["summary"]["poolCounts"]["test"]["patients"] == 20
    assert result["executionEnabled"] is False
    assert_grouped(result)


@pytest.mark.parametrize("mode", MODES)
def test_fixed_validation_never_becomes_cv_test_tuning_or_training(mode):
    result = successful(Store(mode, fixed_validation=True))
    for rows in plan_rows(result).values():
        assert patients(rows, "val") == FIXED_VALIDATION
        for role in ("train", "test", "tune"):
            assert patients(rows, role).isdisjoint(FIXED_VALIDATION)
    final = next(rows for rows in plan_rows(result).values() if rows[0]["phase"] == "final")
    assert len(patients(final, "train")) == 80
    assert patients(final, "test") == EXTERNAL_TEST
    assert result["summary"]["poolCounts"]["val"]["patients"] == 20
    assert_grouped(result)


@pytest.mark.parametrize("mode", MODES)
def test_pool_assignments_are_reproducible_after_row_and_seed_reordering(mode):
    store = Store(mode, seeds=[7, 42])
    result = successful(store)
    store.rows.reverse()
    store.draft["payload"]["spec"]["split"]["seeds"].reverse()
    assert successful(store) == result
    assert result["summary"]["finalPlanCount"] == 2


def test_kfold_oof_coverage_applies_to_selected_training_pool_only():
    result = successful(Store("kfold"))
    tests = Counter()
    for rows in plan_rows(result).values():
        if rows[0]["phase"] != "final":
            tests.update(patients(rows, "test"))
    assert set(tests) == {f"patient-{patient:03}" for patient in range(100)}
    assert set(tests.values()) == {1}
    assert result["summary"]["oofCoverage"]["groups"] == 100
    assert result["summary"]["oofCoverage"]["complete"] is True
    for plan in result["partitions"]:
        for role in ("train", "val", "test"):
            counts = plan[role]["classes"]
            assert abs(counts["negative"] - counts["positive"]) <= 1


def test_nested_inner_plans_exclude_outer_cv_test_and_external_test():
    result = successful(Store("nested_kfold"))
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
    store = Store("leave_one_domain_out", fixed_validation=True)
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
    store = Store("leave_one_domain_out", fixed_validation=True)
    for row in store.rows:
        if row["patientId"] in FIXED_VALIDATION:
            row["attributes"]["site"] = "site-0"
    assert_blocked(store)


@pytest.mark.parametrize("invalid_domain", (None, "conflicting"))
def test_domain_cv_requires_resolved_consistent_domains_for_fixed_validation(invalid_domain):
    store = Store("leave_one_domain_out", fixed_validation=True)
    store.rows[160]["attributes"]["site"] = invalid_domain
    assert_blocked(store)


def test_domain_selection_uses_domains_in_training_pool_not_external_test():
    store = Store("leave_one_domain_out", domainPolicy="selected", heldOutDomains=["site-5"])
    assert_blocked(store)


@pytest.mark.parametrize("missing", ("train", "test"))
def test_missing_explicit_pool_rules_do_not_silently_assign_every_row(missing):
    store = Store()
    store.draft["payload"]["spec"]["split"]["pools"]["rules"][missing] = []
    assert_blocked(store)


def test_remaining_training_pool_requires_the_explicit_remaining_selection():
    store = Store()
    pools = store.draft["payload"]["spec"]["split"]["pools"]
    pools["trainSelection"] = "remaining"
    pools["rules"]["train"] = []
    result = successful(store)
    assert result["summary"]["poolCounts"]["train"]["patients"] == 100
    assert result["summary"]["poolCounts"]["test"]["patients"] == 20


@pytest.mark.parametrize("invalid", ("overlap", "unassigned", "empty_test", "empty_fixed_val"))
def test_conflicting_or_incomplete_pool_selections_block_freezing(invalid):
    store = Store(fixed_validation=invalid == "empty_fixed_val")
    pools = store.draft["payload"]["spec"]["split"]["pools"]
    if invalid == "overlap":
        pools["rules"]["train"] = [{"field": "Slide_ID", "op": "regex", "value": "-0$"}]
        pools["rules"]["test"] = [{"field": "Slide_ID", "op": "regex", "value": "-1$"}]
    elif invalid == "unassigned":
        pools["rules"]["train"] = [{"field": "official", "op": "eq", "value": "train"}]
    else:
        role = "val" if invalid == "empty_fixed_val" else "test"
        pools["rules"][role] = [{"field": "partition", "op": "eq", "value": "missing"}]
    assert_blocked(store)


@pytest.mark.parametrize("mode", MODES)
def test_predefined_official_train_validation_test_are_respected_by_all_strategies(mode):
    result = successful(Store(mode, fixed_validation=True, imported=True))
    for rows in plan_rows(result).values():
        assert patients(rows, "val") == FIXED_VALIDATION
        if rows[0]["phase"] == "final":
            assert patients(rows, "test") == EXTERNAL_TEST
        else:
            assert patients(rows).isdisjoint(EXTERNAL_TEST)
    assert_grouped(result)


def test_predefined_partition_cannot_divide_one_patients_slides():
    store = Store(imported=True)
    store.rows[0]["attributes"]["partition"] = "test"
    assert_blocked(store)


def test_predefined_pool_column_alias_cannot_be_a_model_input():
    store = Store(imported=True, fixed_validation=True)
    store.dataset["manifest"]["dictionary"].append(
        {"key": "allocation", "sourceColumn": "official", "owner": "slide", "type": "text"}
    )
    for row in store.rows:
        row["attributes"]["allocation"] = row["attributes"]["official"]
    store.draft["payload"]["spec"]["predictors"] = ["allocation"]
    assert_blocked(store)


def test_membership_budget_counts_fixed_validation_and_final_evaluation(monkeypatch):
    # Training-only CV uses 800 rows; fixed validation and the final plan raise this to 1,240.
    monkeypatch.setattr("histopilot.application.protocols.MAX_MEMBERSHIPS", 1_000)
    result = preview(Store("kfold", fixed_validation=True))
    assert not result["canFreeze"]
    assert result["memberships"] == []
    assert result["partitions"] == []
    assert "PROTOCOL_MEMBERSHIP_LIMIT" in {finding["code"] for finding in result["findings"]}


def test_pool_exploration_reports_counts_before_target_mapping_is_complete():
    store = Store()
    split = store.draft["payload"]["spec"]["split"]
    result = ProtocolService(store).explore(
        ProtocolExploreRequest(datasetId=DATASET_ID, targetField="label", split=split)
    )
    assert result["valid"], result["findings"]
    assert result["selectionBasis"] == "pools"
    assert result["cohort"]["totalSlides"] == 240
    assert result["partitions"]["train"]["expanded"]["patientCount"] == 100
    assert result["partitions"]["test"]["expanded"]["patientCount"] == 20
    assert result["partitions"]["val"]["expanded"]["patientCount"] == 0
    assert result["unassigned"]["patientCount"] == 0


def test_version_two_preview_hash_is_unchanged_by_new_pool_fields():
    legacy = runpy.run_path(str(Path(__file__).with_name("test_cv_strategies.py")))
    result = legacy["preview"](legacy["Store"]("kfold"))
    assert "pools" not in result["spec"]["split"]
    assert (
        result["previewHash"] == "2fc3c27112d578d888894914d68ea6bd12d9dc538bbdccc73c2861b6f80cac3e"
    )


def test_version_three_pool_protocol_freezes_and_reopens_through_api(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    settings = Settings(workspace=tmp_path / "registry-a", data_roots=(data,))

    def connect(config):
        return TestClient(create_app(config), base_url="http://127.0.0.1:8787")

    def auth(client):
        client.headers["X-HistoPilot-Token"] = client.get("/api/v1/session").json()["token"]

    def post(client, url, body, status=200):
        response = client.post(url, json=body)
        assert response.status_code == status, response.text
        return response.json()

    with connect(settings) as client:
        auth(client)
        project = post(
            client,
            "/api/v1/projects",
            {"name": "Analysis", "storagePath": str(data / "experiment")},
            201,
        )
        base = f"/api/v1/projects/{project['id']}"
        memory = Store("nested_kfold", fixed_validation=True)
        storage = ScientificStore(Path(project["storagePath"]), project["id"])
        storage.initialize()
        imported = storage.create_draft("import", "Source", {})
        dataset = storage.publish_dataset(
            imported["id"],
            expected_revision=1,
            manifest=memory.dataset["manifest"],
            artifacts={"records.json": json.dumps(memory.rows).encode()},
            operation_id="source",
        )
        spec = memory.draft["payload"]["spec"]
        spec["datasetId"] = dataset["id"]
        live = post(
            client,
            base + "/protocols/explore",
            {"datasetId": dataset["id"], "split": spec["split"]},
        )
        assert live["valid"]
        assert live["partitions"]["train"]["expanded"]["patientCount"] == 80
        assert live["partitions"]["val"]["expanded"]["patientCount"] == 20
        assert live["partitions"]["test"]["expanded"]["patientCount"] == 20
        draft = post(
            client,
            base + "/drafts",
            {
                "kind": "experiment",
                "name": "Nested analysis",
                "payload": {"type": "analysis-protocol", "spec": spec},
            },
            201,
        )
        protocol_url = base + f"/protocols/{draft['id']}"
        result = post(client, protocol_url + "/preview", {"expectedRevision": 1})
        assert result["canFreeze"], result["findings"]
        intent = {
            "expectedRevision": 1,
            "previewHash": result["previewHash"],
            "operationId": "freeze-pools",
        }
        frozen = post(client, protocol_url + "/freeze", intent, 201)
        assert frozen["manifest"]["memberships"] == result["memberships"]
        assert frozen["manifest"]["summary"]["poolCounts"] == result["summary"]["poolCounts"]
    with connect(replace(settings, workspace=tmp_path / "registry-b")) as client:
        auth(client)
        reopened = post(client, "/api/v1/projects/open", {"path": project["storagePath"]})
        assert reopened["id"] == project["id"]
        assert client.get(base + f"/configurations/{frozen['id']}").json() == frozen
        assert post(client, protocol_url + "/freeze", intent, 201) == frozen


def test_final_test_class_minimum_is_enforced_for_domain_strategy():
    store = Store("leave_one_domain_out")
    for row in store.rows:
        if row["patientId"] in EXTERNAL_TEST:
            row["attributes"]["label"] = "0"
    result = preview(store)
    assert not result["canFreeze"]
    assert any(
        finding["code"] == "PARTITION_CLASS_TOO_SMALL"
        and finding["severity"] == "error"
        and "/final" in finding["message"]
        for finding in result["findings"]
    )
