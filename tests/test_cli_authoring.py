"""Records authored from spec files: Targets & splits, cohorts, references, Apply models."""

import pytest
import yaml
from support import projects as fixtures
from support.cli import Service

from histopilot.storage.scientific import ScientificStore


@pytest.fixture
def study(tmp_path, monkeypatch):
    with Service(tmp_path, monkeypatch) as service:
        project = service.create_project()
        store = ScientificStore(service.settings.workspace / "projects" / "Study", project)
        rows = [
            {
                "slideId": f"s{index}",
                "patientId": f"p{index}",
                "attributes": {
                    "label": str(index % 2),
                    "cohort": "development" if index < 20 else "test",
                },
            }
            for index in range(40)
        ]
        _, cohort_spec, _ = fixtures.setup(store, service.data)
        data, _ = fixtures.dataset(store, operation="forty", rows=rows)
        service.cli("use", project)
        yield service, project, data["id"], cohort_spec


def spec_file(tmp_path, name, kind, body):
    path = tmp_path / name
    path.write_text(yaml.safe_dump({"kind": kind, "specVersion": 1, **body}, sort_keys=False))
    return path


def test_targets_are_frozen_from_a_spec_file_after_confirmation(study, tmp_path):
    service, _, dataset, _ = study
    body = service.cli("targets", "template", "--json").envelope["data"]
    body["datasetId"] = dataset
    body["target"].update(
        field="label",
        task="binary_classification",
        classes=["low", "high"],
        labels={"0": "low", "1": "high"},
        positiveClass="high",
    )
    path = spec_file(tmp_path, "targets.yaml", "targets", body)

    pending = service.cli("targets", "create", "--from", str(path), "--tag", "grade v1", "--json")
    assert pending.code == 7, pending.stdout
    preview = pending.envelope["data"]["preview"]
    assert preview["canFreeze"] is True and preview["draftId"].startswith("draft-")

    frozen = service.cli(
        "targets", "create", "--from", str(path), "--tag", "grade v1", "--yes", "--json"
    )
    assert frozen.code == 0, frozen.stdout
    # The confirmed rerun reuses the draft the unconfirmed run saved, instead of another.
    assert frozen.envelope["data"]["preview"]["draftId"] == preview["draftId"]
    listed = service.cli("targets", "list", "--json").envelope["data"]
    assert [row["versionLabel"]["tag"] for row in listed] == ["grade v1"]
    assert service.cli("targets", "show", "@grade v1", "--json").code == 0


def test_a_spec_without_its_split_unit_is_refused_before_any_draft(study, tmp_path):
    service, project, dataset, _ = study
    body = service.cli("targets", "template", "--json").envelope["data"]
    body["datasetId"] = dataset
    del body["splitUnit"]
    path = spec_file(tmp_path, "targets.yaml", "targets", body)
    refused = service.cli("targets", "create", "--from", str(path), "--tag", "t", "--yes", "--json")
    assert refused.code == 2 and refused.envelope["error"]["code"] == "SPEC_FIELD_REQUIRED"
    drafts = service.http.get(f"/api/v1/projects/{project}/drafts").json()["drafts"]
    assert not [draft for draft in drafts if draft["name"] == "t"]


def test_a_cohort_and_its_reference_standard_are_created_from_specs(study, tmp_path):
    service, _, _, cohort_spec = study
    body = {**service.cli("cohort", "template", "--json").envelope["data"], **cohort_spec}
    body["target"] = {**body["target"], "unit": "patient"}
    path = spec_file(tmp_path, "cohort.yaml", "cohort", body)
    created = service.cli(
        "cohort", "create", "--from", str(path), "--tag", "test cohort", "--yes", "--json"
    )
    assert created.code == 0, created.stdout
    cohort = created.envelope["data"]["result"]["id"]

    reference = {
        "cohortId": "@test cohort",
        "name": "Second reading",
        "datasetIds": [cohort_spec["datasetId"]],
        "field": "label",
        "classes": ["low", "high"],
        "labels": {"0": "low", "1": "high"},
    }
    path = spec_file(tmp_path, "reference.yaml", "reference", reference)
    saved = service.cli("reference", "create", "--from", str(path), "--yes", "--json")
    assert saved.code == 0, saved.stdout
    references = service.cli("reference", "list", "--json").envelope["data"]
    assert [row["manifest"]["name"] for row in references] == ["Second reading"]
    assert references[0]["manifest"]["cohortId"] == cohort
    assert "Second reading" in service.cli("reference", "list").stdout


def test_apply_models_writes_out_its_scope_and_refuses_unknown_predictors(study, tmp_path):
    service, _, _, _ = study
    body = service.cli("apply", "template", "--json").envelope["data"]
    assert body["scope"] == "selected" and body["inference"]["patientAggregation"] == "predictor"
    del body["scope"]
    path = spec_file(tmp_path, "apply.yaml", "apply", body)
    assert service.cli("apply", "preview", "--from", str(path), "--json").code == 2
