"""Project-scoped scientific state survives service and registry replacement."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from histopilot.api import create_app
from histopilot.application.project_workspace import DESCRIPTOR
from histopilot.config import Settings
from histopilot.storage.scientific import SCHEMA_VERSION

API = "/api/v1"
ARTIFACT_NAME = "tables/slides.json"
ARTIFACT_BYTES = b'[{"Slide_ID":"fixture-slide","Patient_ID":null}]\n'


@pytest.fixture
def settings(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    return Settings(workspace=tmp_path / "registry", data_roots=(data,))


def connection(settings):
    return TestClient(create_app(settings), base_url="http://127.0.0.1:8787")


def authenticate(client):
    client.headers["X-HistoPilot-Token"] = client.get(f"{API}/session").json()["token"]


@pytest.fixture
def client(settings):
    with connection(settings) as current:
        authenticate(current)
        yield current


def create_project(client, settings, name="Bladder", folder=None):
    response = client.post(
        f"{API}/projects",
        json={
            "name": name,
            "storagePath": str(folder or settings.data_roots[0] / name),
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def create_draft(client, project_id, *, kind="import", name="Bladder import", payload=None):
    response = client.post(
        f"{API}/projects/{project_id}/drafts",
        json={
            "kind": kind,
            "name": name,
            "payload": payload if payload is not None else {"mapping": {"Slide_ID": "De ID"}},
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def publish_fixture(client, project_id):
    """Exercise publication only through the internal, server-owned store API."""
    draft = create_draft(client, project_id)
    store = client.app.state.projects.scientific_store(project_id)
    dataset = store.publish_dataset(
        draft["id"],
        expected_revision=1,
        manifest={"name": "fixture", "slideCount": 1, "patientCount": 0},
        artifacts={ARTIFACT_NAME: ARTIFACT_BYTES},
        operation_id="freeze-fixture",
    )
    return draft, dataset


def assert_error(response, status):
    assert response.status_code == status, response.text
    body = response.json()
    assert isinstance(body["detail"], str) and body["detail"]
    assert isinstance(body["code"], str) and body["code"]


def test_create_initializes_project_database_and_empty_scientific_state(client, settings):
    project = create_project(client, settings)
    folder = Path(project["storagePath"])
    database = folder / "histopilot-state.sqlite"
    assert database.is_file()
    with database.open("rb") as handle:
        assert handle.read(16) == b"SQLite format 3\x00"

    prefix = f"{API}/projects/{project['id']}"
    response = client.get(f"{prefix}/storage")
    assert response.status_code == 200, response.text
    status = response.json()
    assert status == client.app.state.projects.scientific_store(project["id"]).status()
    assert status["projectId"] == project["id"]
    assert status["schemaVersion"] == SCHEMA_VERSION
    assert status["database"] == "histopilot-state.sqlite"
    assert status["journalMode"] == "wal"
    assert status["draftCount"] == status["datasetCount"] == 0
    assert status["operations"] == []
    assert client.get(f"{prefix}/drafts").json() == {"drafts": []}
    assert client.get(f"{prefix}/datasets").json() == {"datasets": []}


@pytest.mark.parametrize("kind", ["import", "experiment"])
def test_draft_roundtrip_and_stale_revision_preserve_winning_edit(client, settings, kind):
    project = create_project(client, settings)
    prefix = f"{API}/projects/{project['id']}"
    draft = create_draft(client, project["id"], kind=kind)
    assert draft["projectId"] == project["id"]
    assert draft["kind"] == kind
    assert draft["name"] == "Bladder import"
    assert draft["payload"] == {"mapping": {"Slide_ID": "De ID"}}
    assert draft["revision"] == 1
    assert draft["status"] == "editable"
    assert client.get(f"{prefix}/drafts/{draft['id']}").json() == draft
    assert client.get(f"{prefix}/drafts").json() == {"drafts": [draft]}

    edit = {
        "expectedRevision": 1,
        "name": "Reviewed mapping",
        "payload": {"mapping": {"Slide_ID": "De ID", "Patient_ID": None}},
    }
    response = client.patch(f"{prefix}/drafts/{draft['id']}", json=edit)
    assert response.status_code == 200, response.text
    updated = response.json()
    assert updated["id"] == draft["id"]
    assert updated["projectId"] == project["id"]
    assert updated["kind"] == kind
    assert updated["revision"] == 2
    assert updated["name"] == edit["name"]
    assert updated["payload"] == edit["payload"]
    assert updated["createdAt"] == draft["createdAt"]

    stale = client.patch(
        f"{prefix}/drafts/{draft['id']}",
        json={"expectedRevision": 1, "name": "Stale tab", "payload": {"lost": True}},
    )
    assert_error(stale, 409)
    assert client.get(f"{prefix}/drafts/{draft['id']}").json() == updated
    assert client.get(f"{prefix}/drafts").json() == {"drafts": [updated]}


def test_published_dataset_metadata_and_artifact_survive_fresh_registry_and_move(
    settings, tmp_path
):
    folder = settings.data_roots[0] / "portable-bladder"
    with connection(settings) as first:
        authenticate(first)
        project = create_project(first, settings, folder=folder)
        prefix = f"{API}/projects/{project['id']}"
        editable = create_draft(first, project["id"], kind="experiment", name="Next experiment")
        frozen, dataset = publish_fixture(first, project["id"])
        response = first.get(f"{prefix}/datasets/{dataset['id']}")
        assert response.status_code == 200, response.text
        assert response.json() == dataset
        assert dataset["projectId"] == project["id"]
        assert dataset["manifest"]["name"] == "fixture"
        assert dataset["artifacts"][ARTIFACT_NAME]["sizeBytes"] == len(ARTIFACT_BYTES)
        frozen = first.get(f"{prefix}/drafts/{frozen['id']}").json()
        assert frozen["status"] == "frozen"
        assert frozen["revision"] == 2
        drafts = first.get(f"{prefix}/drafts").json()
        assert {draft["id"] for draft in drafts["drafts"]} == {editable["id"], frozen["id"]}
        assert first.get(f"{prefix}/datasets").json() == {"datasets": [dataset]}

    fresh = replace(settings, workspace=tmp_path / "fresh-registry")
    with connection(fresh) as second:
        authenticate(second)
        assert all(
            item["id"] != project["id"] for item in second.get(f"{API}/projects").json()["projects"]
        )
        opened = second.post(f"{API}/projects/open", json={"path": str(folder)})
        assert opened.status_code == 200, opened.text
        assert opened.json()["id"] == project["id"]
        assert second.get(f"{prefix}/drafts").json() == drafts
        assert second.get(f"{prefix}/datasets").json() == {"datasets": [dataset]}
        assert (
            second.app.state.projects.scientific_store(project["id"]).read_artifact(
                dataset["id"], ARTIFACT_NAME
            )
            == ARTIFACT_BYTES
        )

    moved = folder.with_name("moved-bladder")
    folder.rename(moved)
    with connection(fresh) as third:
        authenticate(third)
        unavailable = next(
            item
            for item in third.get(f"{API}/projects").json()["projects"]
            if item["id"] == project["id"]
        )
        assert unavailable["available"] is False
        opened = third.post(f"{API}/projects/open", json={"path": str(moved)})
        assert opened.status_code == 200, opened.text
        assert opened.json()["id"] == project["id"]
        assert opened.json()["storagePath"] == str(moved)
        assert third.get(f"{prefix}/drafts").json() == drafts
        assert third.get(f"{prefix}/datasets/{dataset['id']}").json() == dataset
        store = third.app.state.projects.scientific_store(project["id"])
        assert store.read_artifact(dataset["id"], ARTIFACT_NAME) == ARTIFACT_BYTES
        status = third.get(f"{prefix}/storage").json()
        assert status["draftCount"] == 2
        assert status["datasetCount"] == 1
        assert any(
            operation["id"] == "freeze-fixture"
            and operation["status"] == "published"
            and operation["datasetId"] == dataset["id"]
            for operation in status["operations"]
        )


def test_open_legacy_v1_descriptor_adds_state_without_rewriting_descriptor(client, settings):
    folder = settings.data_roots[0] / "legacy-bladder"
    folder.mkdir()
    document = {
        "format": "histopilot-project",
        "schemaVersion": 1,
        "id": "project-" + "a" * 32,
        "name": "Legacy bladder",
        "description": "Existing setup remains intact",
        "createdAt": "2026-09-09T12:00:00Z",
        "updatedAt": "2026-09-09T12:00:00Z",
        "config": {"targetColumn": "Binary WHO 2022", "seed": 42},
        "sources": [],
    }
    descriptor = folder / DESCRIPTOR
    original = (json.dumps(document, indent=4) + "\n").encode()
    descriptor.write_bytes(original)
    response = client.post(f"{API}/projects/open", json={"path": str(folder)})
    assert response.status_code == 200, response.text
    assert response.json()["id"] == document["id"]
    assert response.json()["config"] == document["config"]
    assert descriptor.read_bytes() == original
    assert (folder / "histopilot-state.sqlite").is_file()
    draft = create_draft(client, document["id"])
    assert draft["revision"] == 1
    assert descriptor.read_bytes() == original


def test_independent_registries_serialize_competing_draft_revisions(settings, tmp_path):
    fresh = replace(settings, workspace=tmp_path / "second-registry")
    with connection(settings) as first, connection(fresh) as second:
        authenticate(first)
        authenticate(second)
        project = create_project(first, settings)
        draft = create_draft(first, project["id"])
        opened = second.post(f"{API}/projects/open", json={"path": project["storagePath"]})
        assert opened.status_code == 200, opened.text
        path = f"{API}/projects/{project['id']}/drafts/{draft['id']}"

        def update(client, author):
            return client.patch(
                path,
                json={"expectedRevision": 1, "name": author, "payload": {"author": author}},
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            pending = [
                executor.submit(update, first, "First registry"),
                executor.submit(update, second, "Second registry"),
            ]
            responses = [result.result() for result in pending]
        assert sorted(response.status_code for response in responses) == [200, 409]
        winner = next(response.json() for response in responses if response.status_code == 200)
        assert winner["revision"] == 2
        assert winner["payload"] == {"author": winner["name"]}
        assert first.get(path).json() == second.get(path).json() == winner


def test_project_scoped_ids_do_not_expose_other_projects_drafts_or_datasets(client, settings):
    first = create_project(client, settings)
    second = create_project(client, settings, name="Another bladder experiment")
    draft, dataset = publish_fixture(client, first["id"])
    editable = create_draft(client, first["id"], kind="experiment")
    first_prefix = f"{API}/projects/{first['id']}"
    second_prefix = f"{API}/projects/{second['id']}"
    for draft_id in (draft["id"], editable["id"]):
        assert_error(client.get(f"{second_prefix}/drafts/{draft_id}"), 404)
        assert_error(
            client.patch(
                f"{second_prefix}/drafts/{draft_id}",
                json={"expectedRevision": 1, "name": "Foreign write", "payload": {}},
            ),
            404,
        )
    assert_error(client.get(f"{second_prefix}/datasets/{dataset['id']}"), 404)
    assert client.get(f"{second_prefix}/drafts").json() == {"drafts": []}
    assert client.get(f"{second_prefix}/datasets").json() == {"datasets": []}
    assert client.get(f"{first_prefix}/drafts/{editable['id']}").json() == editable
    assert client.get(f"{first_prefix}/datasets/{dataset['id']}").json() == dataset


@pytest.mark.parametrize("token", [None, "invalid-local-session-token"])
def test_scientific_read_and_write_endpoints_require_session_token(client, settings, token):
    project = create_project(client, settings)
    draft, dataset = publish_fixture(client, project["id"])
    prefix = f"{API}/projects/{project['id']}"
    original_token = client.headers.pop("X-HistoPilot-Token")
    if token is not None:
        client.headers["X-HistoPilot-Token"] = token
    requests = [
        ("GET", "/storage", None),
        ("GET", "/drafts", None),
        ("GET", f"/drafts/{draft['id']}", None),
        ("POST", "/drafts", {"kind": "import", "name": "Unauthorized", "payload": {}}),
        (
            "PATCH",
            f"/drafts/{draft['id']}",
            {"expectedRevision": 2, "name": "Unauthorized", "payload": {}},
        ),
        ("GET", "/datasets", None),
        ("GET", f"/datasets/{dataset['id']}", None),
    ]
    for method, path, payload in requests:
        response = client.request(method, f"{prefix}{path}", json=payload)
        assert response.status_code == 401, response.text
    client.headers["X-HistoPilot-Token"] = original_token
    status = client.get(f"{prefix}/storage").json()
    assert status["draftCount"] == status["datasetCount"] == 1


@pytest.mark.parametrize("path", ["storage", "drafts", "datasets"])
def test_synthetic_demo_has_no_real_project_scientific_store(client, settings, path):
    assert_error(client.get(f"{API}/projects/synthetic-v1/{path}"), 409)
    assert not (settings.workspace / "histopilot-state.sqlite").exists()


def test_synthetic_demo_rejects_scientific_draft_creation(client):
    response = client.post(
        f"{API}/projects/synthetic-v1/drafts",
        json={"kind": "import", "name": "Demo promotion", "payload": {}},
    )
    assert_error(response, 409)


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "dataset", "name": "Invalid", "payload": {}},
        {"kind": "import", "name": "Invalid", "payload": []},
        {"kind": "import", "name": "Invalid", "payload": {}, "projectId": "foreign"},
    ],
)
def test_invalid_draft_creation_leaves_store_empty(client, settings, payload):
    project = create_project(client, settings)
    prefix = f"{API}/projects/{project['id']}"
    response = client.post(f"{prefix}/drafts", json=payload)
    assert response.status_code in {400, 422}, response.text
    assert client.get(f"{prefix}/drafts").json() == {"drafts": []}


@pytest.mark.parametrize("revision", [None, 0, -1, True, "1", 1.5])
def test_invalid_expected_revision_cannot_update_draft(client, settings, revision):
    project = create_project(client, settings)
    draft = create_draft(client, project["id"])
    path = f"{API}/projects/{project['id']}/drafts/{draft['id']}"
    payload = {"name": "Invalid update", "payload": {"modified": True}}
    if revision is not None:
        payload["expectedRevision"] = revision
    response = client.patch(path, json=payload)
    assert response.status_code in {400, 422}, response.text
    assert client.get(path).json() == draft


def test_frozen_draft_rejects_further_api_edits(client, settings):
    project = create_project(client, settings)
    draft, dataset = publish_fixture(client, project["id"])
    prefix = f"{API}/projects/{project['id']}"
    frozen = client.get(f"{prefix}/drafts/{draft['id']}").json()
    response = client.patch(
        f"{prefix}/drafts/{draft['id']}",
        json={"expectedRevision": frozen["revision"], "name": "Rewrite frozen", "payload": {}},
    )
    assert_error(response, 409)
    assert client.get(f"{prefix}/drafts/{draft['id']}").json() == frozen
    assert client.get(f"{prefix}/datasets/{dataset['id']}").json() == dataset


def test_dataset_publication_is_not_exposed_as_client_authored_metadata(client, settings):
    project = create_project(client, settings)
    prefix = f"{API}/projects/{project['id']}"
    response = client.post(
        f"{prefix}/datasets",
        json={"manifest": {"name": "Unvalidated dataset", "patientCount": 138}},
    )
    assert response.status_code in {404, 405}, response.text
    assert client.get(f"{prefix}/datasets").json() == {"datasets": []}
