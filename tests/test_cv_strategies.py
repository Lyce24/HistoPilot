"""Scientific partition guarantees for version 2 cross-validation strategies."""

import copy
import json
from collections import Counter, defaultdict

import pytest
from pydantic import ValidationError

from histopilot.application.protocols import ProtocolService
from histopilot.schemas.protocols import ProtocolSpec
from histopilot.storage.scientific import ScientificStore

DATASET_ID = "dataset-" + "a" * 64
MODES = ("kfold", "monte_carlo", "leave_one_domain_out", "nested_kfold", "held_out")
ROLES = {"train", "val", "test", "tune"}


def rows():
    """Two slides per patient, balanced labels within three independent sites."""
    return [
        {
            "slideId": f"slide-{patient:02d}-{slide}",
            "patientId": f"patient-{patient:02d}",
            "patientIdSource": "source",
            "slidePath": None,
            "attributes": {
                "label": str(patient % 2),
                "site": f"site-{patient // 20}",
                "partition": "test" if patient >= 48 else "train",
                "explicit": "test" if patient >= 48 else "val" if patient >= 36 else "train",
            },
        }
        for patient in range(60)
        for slide in range(2)
    ]


def specification(mode="kfold", **split):
    config = {
        "version": 2,
        "mode": mode,
        "folds": 3,
        "seeds": [42],
        "validationFraction": 0.25,
        "testFraction": 0.2,
        "stratify": True,
    }
    if mode == "leave_one_domain_out":
        config.update(domainField="site", domainPolicy="all")
    if mode == "nested_kfold":
        config.update(outerFolds=3, innerFolds=2)
    if mode == "monte_carlo":
        config["repeats"] = 4
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
        self.rows = rows()
        self.draft = {
            "id": "draft-cv",
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
                    for key in ("label", "site", "partition", "explicit")
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
    return ProtocolService(store).preview("draft-cv", 1)


def successful(store):
    result = preview(store)
    assert result["canFreeze"], result["findings"]
    return result


def plans(result):
    """Read concrete assignments, independently of the engine's count summaries."""
    grouped = defaultdict(list)
    for row in result["memberships"]:
        grouped[row["planId"]].append(row)
    return dict(grouped)


def patients(assignments, partition=None):
    return {
        row["patientId"]
        for row in assignments
        if partition is None or row["partition"] == partition
    }


def assert_disjoint(result):
    summaries = {plan["planId"]: plan for plan in result["partitions"]}
    assert len(summaries) == len(result["partitions"])
    assert set(summaries) == set(plans(result))
    for identity, assignments in plans(result).items():
        assert len({row["slideId"] for row in assignments}) == len(assignments)
        by_patient = defaultdict(set)
        for row in assignments:
            assert row["partition"] in ROLES
            by_patient[row["patientId"]].add(row["partition"])
        assert all(len(roles) == 1 for roles in by_patient.values())
        for partition in {row["partition"] for row in assignments}:
            summary = summaries[identity][partition]
            assert summary["patients"] == len(patients(assignments, partition))
            assert summary["slides"] == sum(row["partition"] == partition for row in assignments)


@pytest.mark.parametrize("mode", MODES)
def test_every_plan_keeps_all_slides_of_a_patient_together(mode):
    result = successful(Store(mode))
    assert_disjoint(result)
    for assignments in plans(result).values():
        assert set(Counter(row["patientId"] for row in assignments).values()) == {2}
        assert patients(assignments, "train")
        assert patients(assignments, "val")
    assert result["executionEnabled"] is False


def test_stratification_and_early_stopping_are_enabled_by_default():
    spec = specification()
    spec["split"].pop("stratify")
    spec["split"].pop("validationFraction")
    validated = ProtocolSpec.model_validate(spec)
    assert validated.split.stratify is True
    assert validated.split.validationFraction == 0.2
    store = Store()
    store.draft["payload"]["spec"] = spec
    result = successful(store)
    for plan in result["partitions"]:
        assert plan["val"]["patients"] == 8
        for role in ("train", "val", "test"):
            assert plan[role]["classes"]["negative"] == plan[role]["classes"]["positive"]


@pytest.mark.parametrize("mode", MODES)
def test_same_configuration_is_invariant_to_row_order_and_seed_order(mode):
    store = Store(mode, seeds=[42, 7])
    first = successful(store)
    store.rows.reverse()
    store.draft["payload"]["spec"]["split"]["seeds"].reverse()
    second = successful(store)
    assert second == first


def test_kfold_rotates_reported_test_once_per_patient_per_seed():
    result = successful(Store("kfold", seeds=[42, 7]))
    assert len(result["partitions"]) == 6
    assert {row["phase"] for row in result["memberships"]} == {"evaluation"}
    for seed in (7, 42):
        tests = Counter(
            (row["patientId"], row["slideId"])
            for row in result["memberships"]
            if row["seed"] == seed and row["partition"] == "test"
        )
        assert len(tests) == 120
        assert set(tests.values()) == {1}
    for assignments in plans(result).values():
        assert len(patients(assignments)) == 60
        assert len(patients(assignments, "test")) == 20
        assert len(patients(assignments, "val")) == 10
        assert len(patients(assignments, "train")) == 30


def test_changing_seed_changes_generated_memberships():
    first = successful(Store("kfold", seeds=[7]))
    second = successful(Store("kfold", seeds=[42]))

    def first_test(result):
        return {
            row["patientId"]
            for row in result["memberships"]
            if row["fold"] == 0 and row["partition"] == "test"
        }

    assert first_test(first) != first_test(second)


def test_validation_percentage_uses_development_pool_after_rotating_test():
    small = successful(Store("kfold", validationFraction=0.1))
    larger = successful(Store("kfold", validationFraction=0.4))
    for result, validation_size in ((small, 4), (larger, 16)):
        for assignments in plans(result).values():
            assert len(patients(assignments, "val")) == validation_size
            assert len(patients(assignments, "test")) == 20
            assert len(patients(assignments, "train")) == 40 - validation_size


@pytest.mark.parametrize(
    "mode,split,adjusted_role",
    [
        ("kfold", {"validationFraction": 0.01}, "val"),
        ("monte_carlo", {"testFraction": 0.01, "repeats": 1}, "test"),
    ],
)
def test_constraints_that_change_requested_fraction_produce_a_warning(mode, split, adjusted_role):
    store = Store(mode, **split)
    store.draft["payload"]["spec"]["constraints"] = {"minPatientsPerClass": 5}
    result = successful(store)
    adjustments = [
        finding for finding in result["findings"] if finding["code"] == "SPLIT_FRACTION_ADJUSTED"
    ]
    assert adjustments
    assert all(finding["severity"] == "warning" for finding in adjustments)
    for assignments in plans(result).values():
        # Five members of each class are required, substantially more than the requested 1%.
        assert len(patients(assignments, adjusted_role)) == 10
    assert_disjoint(result)


def test_normal_integer_rounding_does_not_report_a_constraint_adjustment():
    result = successful(Store("kfold", validationFraction=0.26))
    assert "SPLIT_FRACTION_ADJUSTED" not in {finding["code"] for finding in result["findings"]}
    for assignments in plans(result).values():
        # 26% of each 20-member class rounds to five; this is ordinary group rounding.
        assert len(patients(assignments, "val")) == 10


def test_monte_carlo_repeats_are_independent_and_have_internal_validation():
    result = successful(Store("monte_carlo"))
    assert len(result["partitions"]) == 4
    assert len({row["repeat"] for row in result["memberships"]}) == 4
    test_sets = []
    for assignments in plans(result).values():
        assert len(patients(assignments)) == 60
        assert len(patients(assignments, "test")) == 12
        assert len(patients(assignments, "val")) == 12
        assert len(patients(assignments, "train")) == 36
        test_sets.append(frozenset(patients(assignments, "test")))
    assert len(set(test_sets)) == 4
    # Repeated holdouts permit repeated testing; they are not an OOF partition.
    assert any(test_sets[i] & test_sets[j] for i in range(4) for j in range(i))


def test_domain_cv_holds_out_entire_sites_and_validates_only_on_source_sites():
    store = Store("leave_one_domain_out")
    result = successful(store)
    assert len(result["partitions"]) == 3
    source_sites = {row["patientId"]: row["attributes"]["site"] for row in store.rows}
    assert {part["domain"] for part in result["partitions"]} == {"site-0", "site-1", "site-2"}
    for assignments in plans(result).values():
        held_out = assignments[0]["domain"]
        for row in assignments:
            assert (row["partition"] == "test") == (source_sites[row["patientId"]] == held_out)
        assert len(patients(assignments, "test")) == 20
        assert len(patients(assignments, "val")) == 10


def test_selected_domain_policy_only_evaluates_requested_sites():
    result = successful(
        Store("leave_one_domain_out", domainPolicy="selected", heldOutDomains=["site-1"])
    )
    assert len(result["partitions"]) == 1
    assert result["partitions"][0]["domain"] == "site-1"
    assignments = next(iter(plans(result).values()))
    assert len(patients(assignments)) == 60
    assert patients(assignments, "test") == {f"patient-{index:02d}" for index in range(20, 40)}


@pytest.mark.parametrize("failure", ("missing", "conflicting", "unknown"))
def test_undefined_or_ambiguous_domain_identity_blocks_evaluation(failure):
    store = Store("leave_one_domain_out")
    if failure == "missing":
        store.rows[0]["attributes"]["site"] = None
    elif failure == "conflicting":
        store.rows[0]["attributes"]["site"] = "site-1"
    else:
        store.draft["payload"]["spec"]["split"].update(
            domainPolicy="selected", heldOutDomains=["unobserved-site"]
        )
    result = preview(store)
    assert not result["canFreeze"]
    assert any(finding["severity"] == "error" for finding in result["findings"])


def test_domain_test_missing_a_class_warns_without_rejecting_the_site():
    store = Store("leave_one_domain_out", domainPolicy="selected", heldOutDomains=["site-0"])
    for row in store.rows:
        if row["attributes"]["site"] == "site-0":
            row["attributes"]["label"] = "0"
    result = successful(store)
    assert any(finding["severity"] == "warning" for finding in result["findings"])
    assignments = next(iter(plans(result).values()))
    assert len(patients(assignments, "test")) == 20


def test_identifier_alias_cannot_be_used_as_a_site_or_cohort():
    store = Store("leave_one_domain_out")
    store.dataset["manifest"]["dictionary"][1]["sourceColumn"] = "Registry number"
    store.dataset["manifest"]["provenance"] = {
        "mapping": {"patientIdColumn": "Registry number", "slideIdColumn": "Slide accession"}
    }
    result = preview(store)
    assert not result["canFreeze"]
    assert "INVALID_DOMAIN_FIELD" in {finding["code"] for finding in result["findings"]}


def test_renaming_domain_column_does_not_make_it_a_model_input():
    store = Store("leave_one_domain_out")
    store.dataset["manifest"]["dictionary"].append(
        {"key": "site_copy", "sourceColumn": "site", "owner": "slide", "type": "text"}
    )
    for row in store.rows:
        row["attributes"]["site_copy"] = row["attributes"]["site"]
    store.draft["payload"]["spec"]["predictors"] = ["site_copy"]
    result = preview(store)
    assert not result["canFreeze"]
    assert "FORBIDDEN_PREDICTOR" in {finding["code"] for finding in result["findings"]}


def test_nested_cv_keeps_outer_test_out_of_every_inner_plan_and_separates_tuning():
    result = successful(Store("nested_kfold"))
    grouped = plans(result)
    outer = [members for members in grouped.values() if members[0]["phase"] == "outer"]
    inner = [members for members in grouped.values() if members[0]["phase"] == "inner"]
    assert len(outer) == 3
    assert len(inner) == 6
    outer_tests = []
    for outer_members in outer:
        outer_fold = outer_members[0]["outerFold"]
        outer_test = patients(outer_members, "test")
        outer_tests.append(outer_test)
        matching_inner = [members for members in inner if members[0]["outerFold"] == outer_fold]
        assert len(matching_inner) == 2
        tune_counts = Counter()
        for members in matching_inner:
            assert patients(members).isdisjoint(outer_test)
            assert {row["partition"] for row in members} == {"train", "val", "tune"}
            assert len(patients(members)) == 40
            assert len(patients(members, "tune")) == 20
            # This validation pool comes from inner training, never from tuning or outer test.
            assert abs(len(patients(members, "val")) - 5) <= 1
            tune_counts.update(patients(members, "tune"))
        assert set(tune_counts.values()) == {1}
        assert set(tune_counts) == patients(outer_members) - outer_test
        assert len(patients(outer_members, "val")) == 10
    assert len(set.union(*outer_tests)) == 60
    assert sum(map(len, outer_tests)) == 60
    assert_disjoint(result)


def test_held_out_fraction_plan_has_final_test_and_validation_from_remaining_training():
    result = successful(Store("held_out", heldOutSource="fractions"))
    assert len(result["partitions"]) == 1
    assignments = next(iter(plans(result).values()))
    assert len(patients(assignments, "test")) == 12
    assert len(patients(assignments, "val")) == 12
    assert len(patients(assignments, "train")) == 36


def test_held_out_test_rule_reserves_whole_patients_and_carves_automatic_validation():
    result = successful(
        Store(
            "held_out",
            heldOutSource="rules",
            rules={"test": [{"field": "Slide_ID", "op": "regex", "value": "^slide-5[0-9]-0$"}]},
        )
    )
    assignments = next(iter(plans(result).values()))
    assert patients(assignments, "test") == {f"patient-{index:02d}" for index in range(50, 60)}
    assert len(patients(assignments)) == 60
    assert abs(len(patients(assignments, "val")) - 12.5) <= 1
    assert patients(assignments, "train") | patients(assignments, "val") == {
        f"patient-{index:02d}" for index in range(50)
    }


def test_held_out_explicit_validation_is_kept_without_carving_a_second_validation_pool():
    result = successful(
        Store(
            "held_out",
            heldOutSource="rules",
            rules={
                role: [{"field": "explicit", "op": "eq", "value": role}] for role in ("test", "val")
            },
        )
    )
    assignments = next(iter(plans(result).values()))
    assert len(patients(assignments, "test")) == 12
    assert len(patients(assignments, "val")) == 12
    assert len(patients(assignments, "train")) == 36
    assert patients(assignments, "val") == {f"patient-{index:02d}" for index in range(36, 48)}


def test_held_out_conflicting_patient_rules_block_the_plan():
    store = Store(
        "held_out",
        heldOutSource="rules",
        rules={
            "test": [{"field": "Slide_ID", "op": "eq", "value": "slide-00-0"}],
            "val": [{"field": "Slide_ID", "op": "eq", "value": "slide-00-1"}],
        },
    )
    result = preview(store)
    assert not result["canFreeze"]
    assert any(finding["severity"] == "error" for finding in result["findings"])


@pytest.mark.parametrize("field", ("partition", "explicit"))
def test_predefined_test_is_preserved_with_optional_validation(field):
    result = successful(
        Store(
            "held_out",
            heldOutSource="imported",
            imported={
                "partitionField": field,
                "partitionLabels": {"train": "train", "test": "test", "val": "val"},
            },
        )
    )
    assignments = next(iter(plans(result).values()))
    assert patients(assignments, "test") == {f"patient-{index:02d}" for index in range(48, 60)}
    assert len(patients(assignments, "val")) == 12
    assert len(patients(assignments, "train")) == 36
    assert_disjoint(result)


def test_predefined_split_cannot_put_one_patients_slides_in_different_roles():
    store = Store(
        "held_out",
        heldOutSource="imported",
        imported={
            "partitionField": "partition",
            "partitionLabels": {"train": "train", "test": "test"},
        },
    )
    store.rows[0]["attributes"]["partition"] = "test"
    result = preview(store)
    assert not result["canFreeze"]
    assert "IMPORTED_PATIENT_LEAKAGE" in {finding["code"] for finding in result["findings"]}


@pytest.mark.parametrize("source", ("rules", "imported"))
def test_held_out_cannot_silently_evaluate_on_early_stopping_validation(source):
    extra = {"heldOutSource": source}
    if source == "rules":
        extra["rules"] = {"val": [{"field": "explicit", "op": "eq", "value": "val"}]}
    else:
        extra["imported"] = {
            "partitionField": "partition",
            "partitionLabels": {"train": "train", "test": "val"},
        }
    try:
        result = preview(Store("held_out", **extra))
    except ValidationError:
        return
    assert not result["canFreeze"]


@pytest.mark.parametrize(
    "mode,invalid",
    [
        ("kfold", {"folds": 1}),
        ("kfold", {"validationFraction": 0}),
        ("kfold", {"validationFraction": 1}),
        ("kfold", {"validationFraction": True}),
        ("monte_carlo", {"repeats": 0}),
        ("monte_carlo", {"testFraction": 0}),
        ("monte_carlo", {"testFraction": 1}),
        ("nested_kfold", {"outerFolds": 1}),
        ("nested_kfold", {"innerFolds": 1}),
        ("leave_one_domain_out", {"domainField": None}),
        ("leave_one_domain_out", {"domainPolicy": "selected", "heldOutDomains": []}),
    ],
)
def test_invalid_strategy_configuration_is_rejected(mode, invalid):
    with pytest.raises(ValidationError):
        ProtocolSpec.model_validate(specification(mode, **invalid))


def test_too_few_patients_for_stratified_folds_is_a_blocking_finding():
    store = Store("kfold", folds=10)
    store.rows = store.rows[:12]
    result = preview(store)
    assert not result["canFreeze"]
    assert any(finding["severity"] == "error" for finding in result["findings"])


@pytest.mark.parametrize(
    "limit,value,code",
    [
        ("MAX_MEMBERSHIPS", 100, "PROTOCOL_MEMBERSHIP_LIMIT"),
        ("MAX_PROTOCOL_BYTES", 1024, "PROTOCOL_DOCUMENT_LIMIT"),
    ],
)
def test_oversized_nested_plans_are_rejected_before_assigning_folds(
    monkeypatch, limit, value, code
):
    monkeypatch.setattr(f"histopilot.application.protocols.{limit}", value)

    def no_allocation(*_args, **_kwargs):
        pytest.fail("Oversized split attempted to allocate fold assignments.")

    monkeypatch.setattr("histopilot.application.modern_splits._folds", no_allocation)
    result = preview(Store("nested_kfold"))
    assert not result["canFreeze"]
    assert result["memberships"] == []
    assert code in {finding["code"] for finding in result["findings"]}


@pytest.mark.parametrize("domain_suffix", ["x" * 1000, "\\" * 500, '"' * 500])
def test_domain_bytes_are_bounded_before_allocation_in_both_plan_id_and_metadata(
    monkeypatch, domain_suffix
):
    store = Store("leave_one_domain_out")
    for row in store.rows:
        row["attributes"]["site"] += "-" + domain_suffix
    monkeypatch.setattr("histopilot.application.protocols.MAX_PROTOCOL_BYTES", 600_000)

    def no_allocation(*_args, **_kwargs):
        pytest.fail("Oversized domain plans reached patient allocation before byte rejection.")

    monkeypatch.setattr("histopilot.application.modern_splits._subset", no_allocation)
    result = preview(store)
    assert not result["canFreeze"]
    assert result["memberships"] == []
    assert result["partitions"] == []
    assert "PROTOCOL_DOCUMENT_LIMIT" in {finding["code"] for finding in result["findings"]}


def test_acknowledged_slide_id_fallback_remains_visible_in_modern_cv():
    store = Store()
    for row in store.rows:
        row["patientId"] = row["slideId"]
        row["patientIdSource"] = "slide_fallback"
    result = successful(store)
    assert result["summary"]["includedPatients"] == 0
    assert result["summary"]["includedGroups"] == 120
    assert result["summary"]["fallbackSlideCount"] == 120
    assert all(row["patientIdSource"] == "slide_fallback" for row in result["memberships"])
    assert "SLIDE_ID_FALLBACK_GROUPING" in {finding["code"] for finding in result["findings"]}
    for summary in result["partitions"]:
        for role in ("train", "val", "test"):
            assert summary[role]["patients"] == 0
            assert summary[role]["groups"] == summary[role]["slides"]
            assert summary[role]["fallbackSlides"] == summary[role]["slides"]


def test_legacy_kfold_hash_and_validation_rotation_remain_unchanged():
    store = Store()
    store.draft["payload"]["spec"]["split"] = {"mode": "kfold", "folds": 3, "seeds": [42]}
    result = successful(store)
    assert set(result["spec"]["split"]) == {"mode", "folds", "seeds", "ratios", "rules", "imported"}
    # Recorded from the legacy engine and its original six-field split serialization.
    assert (
        result["previewHash"] == "79a7c195bab080473b9668072c05c425ab092fe8d9d4e9c1c13b856f2dde5e8f"
    )
    assert len(result["partitions"]) == 3
    for summary in result["partitions"]:
        assert summary["train"]["patients"] == 40
        assert summary["val"]["patients"] == 20
        assert summary["test"]["patients"] == 0
    assert all("planId" not in row for row in result["memberships"])


def test_nested_frozen_protocol_preserves_every_plan_and_survives_reopening(tmp_path):
    memory = Store("nested_kfold")
    store = ScientificStore(tmp_path, "project-cv")
    store.initialize()
    imported = store.create_draft("import", "Source", {})
    dataset = store.publish_dataset(
        imported["id"],
        expected_revision=1,
        manifest=memory.dataset["manifest"],
        artifacts={"records.json": json.dumps(memory.rows).encode()},
        operation_id="import-cv",
    )
    spec = memory.draft["payload"]["spec"]
    spec["datasetId"] = dataset["id"]
    draft = store.create_draft(
        "experiment", "Analysis", {"type": "analysis-protocol", "spec": spec}
    )
    service = ProtocolService(store)
    result = service.preview(draft["id"], 1)
    assert result["canFreeze"], result["findings"]
    frozen = service.freeze(draft["id"], 1, result["previewHash"], "freeze-cv")
    assert frozen["manifest"]["memberships"] == result["memberships"]
    assert frozen["manifest"]["partitions"] == result["partitions"]
    reopened = ScientificStore(tmp_path, "project-cv")
    assert reopened.get_configuration(frozen["id"]) == frozen
    assert (
        ProtocolService(reopened).freeze(draft["id"], 1, result["previewHash"], "freeze-cv")
        == frozen
    )


@pytest.mark.parametrize("mode", MODES)
def test_version_two_fractions_do_not_use_legacy_holdout_ratios(mode):
    baseline = ProtocolService(Store(mode)).preview("draft-cv", 1)
    result = ProtocolService(Store(mode, ratios={"train": 0.64, "val": 0.16, "test": 0.2})).preview(
        "draft-cv", 1
    )
    assert result["canFreeze"], result["findings"]
    assert result["memberships"] == baseline["memberships"]
    assert result["partitions"] == baseline["partitions"]
