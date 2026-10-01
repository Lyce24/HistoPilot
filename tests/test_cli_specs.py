"""Spec files: header, validation per field, explicit science fields and @tag references."""

import json

import pytest

from histopilot import templates
from histopilot.client import ClientError, specs


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def targets_body():
    body = templates.describe()["starters"]["targetSplit"]
    body["datasetId"] = "dataset-" + "a" * 64
    body["target"].update(
        field="grade",
        task="binary_classification",
        classes=["low", "high"],
        labels={"low": "low", "high": "high"},
        positiveClass="high",
    )
    return body


def test_a_template_round_trips_through_a_yaml_file(tmp_path):
    body = targets_body()
    path = write(tmp_path, "targets.yaml", specs.render("targets", body, note="Targets & splits"))
    assert path.read_text().startswith("# Targets & splits\nkind: targets\nspecVersion: 1\n")
    assert specs.check("targets", specs.read(path, "targets")) == body


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("kind: cohort\nspecVersion: 1\n", "SPEC_KIND_MISMATCH"),
        ("kind: targets\nspecVersion: 2\n", "SPEC_VERSION_UNSUPPORTED"),
        ("- just\n- a list\n", "SPEC_INVALID"),
        ("kind: targets\n  bad: [indent\n", "SPEC_INVALID"),
    ],
)
def test_malformed_files_are_invalid_input(tmp_path, text, code):
    with pytest.raises(ClientError) as raised:
        specs.read(write(tmp_path, "spec.yaml", text), "targets")
    assert (raised.value.code, raised.value.exit_code) == (code, 2)


def test_leaving_out_a_science_field_is_refused_with_the_reason():
    body = targets_body()
    del body["splitUnit"], body["target"]["unit"]
    with pytest.raises(ClientError) as raised:
        specs.check("targets", body)
    error = raised.value
    assert error.code == "SPEC_FIELD_REQUIRED" and error.exit_code == 2
    assert {finding["field"] for finding in error.findings} == {"splitUnit", "target.unit"}
    assert "split by patient" in error.findings[0]["message"]


def test_unknown_keys_and_wrong_types_name_the_field():
    body = targets_body()
    body["splitUnitt"] = "slide"
    body["split"]["testFraction"] = "a fifth"
    with pytest.raises(ClientError) as raised:
        specs.check("targets", body)
    assert {finding["field"] for finding in raised.value.findings} == {
        "splitUnitt",
        "split.testFraction",
    }


def test_an_unlabeled_cohort_needs_no_target_unit():
    body = templates.describe()["starters"]["cohort"]
    body.update(purpose="inference", target=None)
    with pytest.raises(ClientError) as raised:
        specs.check("cohort", {key: value for key, value in body.items() if key != "inference"})
    assert [finding["field"] for finding in raised.value.findings] == ["inference"]


def test_tags_are_resolved_in_known_id_fields_only():
    body = {
        "datasetId": "@v1",
        "datasetIds": ["@v1", "dataset-x"],
        "name": "@not-a-reference",
        "nested": {"cohortId": "@c", "protocolId": "@p", "featureBundleId": "@b"},
    }
    resolved, notes = specs.resolve_tags(body, lambda noun, name: f"{noun}/{name[1:]}")
    # Tags are unique within one kind only, so each field looks among its own kind.
    assert resolved == {
        "datasetId": "dataset/v1",
        "datasetIds": ["dataset/v1", "dataset-x"],
        "name": "@not-a-reference",
        "nested": {
            "cohortId": "cohort/c",
            "protocolId": "configuration:protocol/p",
            "featureBundleId": "bundle/b",
        },
    }
    # One note per tag, naming the ID it stands for.
    assert [note["message"] for note in notes] == [
        "datasetId: @v1 is dataset/v1.",
        "datasetIds: @v1 is dataset/v1.",
        "cohortId: @c is cohort/c.",
        "protocolId: @p is configuration:protocol/p.",
        "featureBundleId: @b is bundle/b.",
    ]


def test_json_specs_are_read_too(tmp_path):
    body = targets_body()
    path = write(
        tmp_path, "targets.json", json.dumps({"kind": "targets", "specVersion": 1, **body})
    )
    assert specs.read(path, "targets") == body
