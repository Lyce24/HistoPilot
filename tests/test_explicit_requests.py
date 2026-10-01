"""Requests a script can retry, and fields whose service default differs from the browser's.

Project and draft creation honour an optional operation ID; a new experiment states its
setup version; an Apply models selection states its scope and every inference setting.
"""

import pytest
from pydantic import ValidationError
from support.cli import Service

from histopilot import templates
from histopilot.schemas.bulk_evaluations import BulkEvaluationSelection

API = "/api/v1"
COHORT = "configuration-" + "c" * 64
PREDICTOR = "configuration-" + "a" * 64


@pytest.fixture
def service(tmp_path, monkeypatch):
    with Service(tmp_path, monkeypatch) as current:
        # A project folder's parent must exist; the service never creates a tree.
        (current.settings.workspace / "projects").mkdir(parents=True, exist_ok=True)
        yield current


def code(response):
    return response.json().get("code")


def test_creating_a_project_again_with_its_operation_id_returns_it(service):
    folder = str(service.settings.workspace / "projects" / "Retried")
    request = {"name": "Retried", "storagePath": folder, "operationId": "project:1"}
    first = service.http.post(f"{API}/projects", json=request)
    again = service.http.post(f"{API}/projects", json=request)
    assert first.status_code == again.status_code == 201
    assert again.json()["id"] == first.json()["id"]
    assert [row["id"] for row in service.http.get(f"{API}/projects").json()["projects"]].count(
        first.json()["id"]
    ) == 1
    other = service.http.post(f"{API}/projects", json={**request, "name": "Another"})
    assert other.status_code == 409 and code(other) == "OPERATION_CONFLICT"
    # Without an ID, a second create in the same folder is refused, as before.
    plain = service.http.post(f"{API}/projects", json={"name": "Retried", "storagePath": folder})
    assert plain.status_code == 409 and code(plain) == "PROJECT_FOLDER_NOT_EMPTY"


def test_creating_a_draft_again_with_its_operation_id_returns_it(service):
    project = service.create_project()
    path = f"{API}/projects/{project}/drafts"
    request = {
        "kind": "experiment",
        "name": "Targets",
        "payload": {"type": "target-split", "spec": {}},
        "operationId": "targets-draft:1",
    }
    first = service.http.post(path, json=request)
    again = service.http.post(path, json=request)
    assert first.status_code == again.status_code == 201, first.text
    assert again.json()["id"] == first.json()["id"]
    assert len(service.http.get(path).json()["drafts"]) == 1
    changed = service.http.post(path, json={**request, "name": "Other"})
    assert changed.status_code == 409 and code(changed) == "OPERATION_CONFLICT"
    # The same operation ID in another project is another operation.
    elsewhere = service.create_project("Elsewhere")
    assert (
        service.http.post(f"{API}/projects/{elsewhere}/drafts", json=request).json()["id"]
        != first.json()["id"]
    )


def test_a_new_experiment_states_its_setup_version(service):
    project = service.create_project()
    path = f"{API}/projects/{project}/model-experiments"
    omitted = service.http.post(path, json={"name": "E", "operationId": "e:1"})
    assert omitted.status_code == 422 and code(omitted) == "SETUP_VERSION_REQUIRED"
    designed = service.http.post(path, json={"name": "E", "operationId": "e:2", "setupVersion": 1})
    legacy = service.http.post(path, json={"name": "L", "operationId": "e:3", "setupVersion": None})
    assert designed.status_code == legacy.status_code == 201
    assert designed.json()["setupVersion"] == 1 and not legacy.json().get("setupVersion")


def test_an_apply_selection_states_its_scope_and_every_inference_setting(service):
    project = service.create_project()
    path = f"{API}/projects/{project}/evaluation-runs/bulk/preview"
    no_scope = service.http.post(path, json={"cohortId": COHORT, "predictorIds": [PREDICTOR]})
    assert no_scope.status_code == 422 and "State scope: selected" in no_scope.text
    selection = {"cohortId": COHORT, "scope": "selected", "predictorIds": [PREDICTOR]}
    partial = service.http.post(path, json={**selection, "inference": {"batchSize": 4}})
    assert partial.status_code == 422 and code(partial) == "INFERENCE_SETTINGS_INCOMPLETE"
    assert "patientAggregation" in partial.json()["detail"]
    full = service.http.post(path, json={**selection, "inference": templates.APPLY["inference"]})
    assert code(full) != "INFERENCE_SETTINGS_INCOMPLETE"
    # Built in Python, as services and stored records do, a selection keeps its defaults.
    built = BulkEvaluationSelection.model_validate({**selection, "inference": {"batchSize": 4}})
    assert built.inference.decisionThreshold == 0.5
    with pytest.raises(ValidationError, match="State scope: selected"):
        BulkEvaluationSelection.model_validate({"cohortId": COHORT, "predictorIds": [PREDICTOR]})


def test_the_cli_creates_a_project_after_confirmation(service, tmp_path):
    folder = str(service.settings.workspace / "projects" / "From the CLI")
    unconfirmed = service.cli("project", "create", "--name", "CLI", "--folder", folder, "--json")
    assert unconfirmed.code == 7
    created = service.cli(
        "project", "create", "--name", "CLI", "--folder", folder, "--yes", "--json"
    )
    assert created.code == 0, created.stdout
    project = created.envelope["data"]["result"]
    assert project["name"] == "CLI" and project["storagePath"].endswith("From the CLI")
    listed = service.cli("project", "list", "--json").envelope["data"]
    assert project["id"] in [row["id"] for row in listed]
