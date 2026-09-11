"""Presentation tags must survive reload without changing scientific references."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from histopilot.api import create_app
from histopilot.config import Settings


@pytest.fixture
def labelled_project(tmp_path):
    settings = Settings(workspace=tmp_path / "registry", data_roots=(tmp_path,))
    with TestClient(create_app(settings), base_url="http://127.0.0.1:8787") as client:
        client.headers["X-HistoPilot-Token"] = client.get("/api/v1/session").json()["token"]
        project = client.post(
            "/api/v1/projects",
            json={"name": "Version labels", "storagePath": str(tmp_path / "project")},
        ).json()
        store = client.app.state.projects.scientific_store(project["id"])
        draft = store.create_draft("import", "Dataset draft", {})
        dataset = store.publish_dataset(
            draft["id"],
            expected_revision=1,
            manifest={"kind": "dataset", "name": "Imported slides", "summary": {"slideCount": 1}},
            artifacts={"records.json": json.dumps([{"slideId": "slide_001"}]).encode()},
            operation_id="dataset",
        )
        feature = store.publish_configuration(
            manifest={
                "kind": "feature",
                "datasetId": dataset["id"],
                "spec": {"datasetId": dataset["id"], "encoderId": "example_encoder"},
                "files": [],
            },
            operation_id="feature",
        )
        protocol = store.publish_configuration(
            manifest={
                "kind": "protocol",
                "datasetId": dataset["id"],
                "spec": {"datasetId": dataset["id"], "featureSetId": feature["id"]},
                "memberships": [{"slideId": "slide_001", "role": "train"}],
            },
            operation_id="protocol",
        )
        yield client, settings, project, store, dataset, feature, protocol


def route(project, resource):
    collection = "datasets" if resource["id"].startswith("dataset-") else "configurations"
    return f"/api/v1/projects/{project['id']}/{collection}/{resource['id']}"


def test_labels_appear_on_reads_lists_and_workspace_without_rewriting_versions(labelled_project):
    client, settings, project, store, dataset, feature, protocol = labelled_project
    disk_path = store.folder / "datasets" / dataset["id"] / "manifest.json"
    disk_before = disk_path.read_bytes()
    for resource, tag in (
        (dataset, "Curated v1"),
        (feature, "Encoder v2"),
        (protocol, "Cohort v3"),
    ):
        response = client.put(
            route(project, resource) + "/label",
            json={
                "tag": tag,
                "note": "Reviewed slides.\nReady for comparison.",
                "expectedRevision": 0,
            },
        )
        assert response.status_code == 200, response.text
        label = response.json()
        assert label["tag"] == tag
        assert label["revision"] == 1
        updated = client.get(route(project, resource)).json()
        assert updated.pop("versionLabel") == label
        assert updated == resource
    prefix = f"/api/v1/projects/{project['id']}"
    assert (
        client.get(prefix + "/datasets").json()["datasets"][0]["versionLabel"]["tag"]
        == "Curated v1"
    )
    configs = client.get(prefix + "/configurations").json()["configurations"]
    assert {item["versionLabel"]["tag"] for item in configs} == {"Encoder v2", "Cohort v3"}
    workspace = client.get(prefix + "/workspace").json()
    assert workspace["dataset"]["id"] == dataset["id"]
    assert workspace["dataset"]["versionLabel"]["tag"] == "Curated v1"
    assert workspace["executionEnabled"] is False
    assert disk_path.read_bytes() == disk_before
    client.put(
        route(project, feature) + "/label",
        json={"tag": "Reviewed encoder", "note": "Renamed only", "expectedRevision": 1},
    ).raise_for_status()
    unchanged = client.get(route(project, protocol)).json()
    assert unchanged["manifest"]["spec"]["featureSetId"] == feature["id"]
    assert unchanged["contentHash"] == protocol["contentHash"]


def test_labels_follow_the_project_folder_into_a_fresh_registry(labelled_project, tmp_path):
    client, settings, project, store, dataset, feature, protocol = labelled_project
    client.put(
        route(project, dataset) + "/label",
        json={"tag": "Portable label", "note": "Saved with this project", "expectedRevision": 0},
    ).raise_for_status()
    fresh = replace(settings, workspace=tmp_path / "fresh-registry")
    with TestClient(create_app(fresh), base_url="http://127.0.0.1:8787") as reopened:
        reopened.headers["X-HistoPilot-Token"] = reopened.get("/api/v1/session").json()["token"]
        opened = reopened.post("/api/v1/projects/open", json={"path": project["storagePath"]})
        assert opened.status_code == 200, opened.text
        assert (
            reopened.get(route(project, dataset)).json()["versionLabel"]["tag"] == "Portable label"
        )


def test_labels_require_authentication_and_cannot_cross_project_boundaries(labelled_project):
    client, settings, project, store, dataset, feature, protocol = labelled_project
    payload = {"tag": "My version", "note": "", "expectedRevision": 0}
    token = client.headers.pop("X-HistoPilot-Token")
    for resource in (dataset, feature):
        assert client.put(route(project, resource) + "/label", json=payload).status_code == 401
    client.headers["X-HistoPilot-Token"] = token
    second = client.post(
        "/api/v1/projects",
        json={
            "name": "Another project",
            "storagePath": str(Path(project["storagePath"]).with_name("another")),
        },
    ).json()
    for resource in (dataset, feature, protocol):
        response = client.put(route(second, resource) + "/label", json=payload)
        assert response.status_code == 404, response.text
    assert "versionLabel" not in client.get(route(project, dataset)).json()


def test_stale_edits_do_not_replace_the_winning_tag(labelled_project):
    client, settings, project, store, dataset, feature, protocol = labelled_project
    endpoint = route(project, dataset) + "/label"
    first = client.put(
        endpoint, json={"tag": "Winner", "note": "First edit", "expectedRevision": 0}
    )
    assert first.status_code == 200, first.text
    stale = client.put(
        endpoint, json={"tag": "Other tab", "note": "Later edit", "expectedRevision": 0}
    )
    assert stale.status_code == 409, stale.text
    assert client.get(route(project, dataset)).json()["versionLabel"] == first.json()


@pytest.mark.parametrize(
    "update",
    [
        {"tag": "x" * 81},
        {"note": "x" * 2001},
        {"tag": "two\nlines"},
        {"tag": "bad\x00tag"},
        {"expectedRevision": -1},
        {"expectedRevision": True},
        {"manifest": {"name": "not a label"}},
    ],
)
def test_invalid_label_requests_leave_the_version_unchanged(labelled_project, update):
    client, settings, project, store, dataset, feature, protocol = labelled_project
    response = client.put(
        route(project, dataset) + "/label",
        json={"tag": "Tag", "note": "", "expectedRevision": 0, **update},
    )
    assert response.status_code == 422, response.text
    assert client.get(route(project, dataset)).json() == dataset


@pytest.mark.parametrize("kind", ["imports", "protocols", "features"])
@pytest.mark.parametrize("label", [None, {}, {"tag": ""}, {"tag": "   "}, {"tag": "x" * 81}])
def test_freeze_requires_a_personal_tag_before_any_publication(labelled_project, kind, label):
    client, settings, project, store, dataset, feature, protocol = labelled_project
    before = store.status()
    prefix = f"/api/v1/projects/{project['id']}"
    payload = {"previewHash": "a" * 64, "operationId": "missing-tag"}
    if kind == "features":
        endpoint = prefix + "/features/freeze"
        payload.update(datasetId=dataset["id"], path=str(store.folder))
    else:
        endpoint = prefix + f"/{kind}/draft-example/freeze"
        payload["expectedRevision"] = 1
    if label is not None:
        payload["versionLabel"] = label
    response = client.post(endpoint, json=payload)
    assert response.status_code == 422, response.text
    assert any(error["loc"][:2] == ["body", "versionLabel"] for error in response.json()["detail"])
    assert store.status() == before


def test_freeze_tag_conflict_keeps_draft_editable_and_retry_preserves_the_saved_label(
    labelled_project, tmp_path
):
    client, settings, project, store, original, feature, protocol = labelled_project
    prefix = f"/api/v1/projects/{project['id']}"
    client.put(
        route(project, original) + "/label",
        json={"tag": "Reviewed data", "expectedRevision": 0},
    ).raise_for_status()
    source = tmp_path / "slides.csv"
    source.write_text(
        "Slide_ID,Patient_ID,Diagnosis\nslide_A,patient_A,low\nslide_B,patient_B,high\n"
    )
    response = client.post(
        prefix + "/drafts",
        json={
            "kind": "import",
            "name": "Next reviewed import",
            "payload": {
                "type": "dataset-import",
                "spec": {
                    "source": {"path": str(source)},
                    "slideIdColumn": "Slide_ID",
                    "patientIdColumn": "Patient_ID",
                    "includeMissingSlides": True,
                },
            },
        },
    )
    assert response.status_code == 201, response.text
    draft = response.json()
    endpoint = prefix + f"/imports/{draft['id']}"
    preview = client.post(endpoint + "/preview", json={"expectedRevision": 1}).json()
    assert preview["canFreeze"]
    intent = {
        "expectedRevision": 1,
        "previewHash": preview["previewHash"],
        "operationId": "name-before-freeze",
        "versionLabel": {"tag": "REVIEWED DATA", "note": "Updated review."},
    }
    before = store.status()
    duplicate = client.post(endpoint + "/freeze", json=intent)
    assert duplicate.status_code == 409, duplicate.text
    assert store.status() == before
    assert store.get_draft(draft["id"])["status"] == "editable"
    assert len(store.list_datasets()) == 1

    intent["versionLabel"]["tag"] = "Reviewed data v2"
    response = client.post(endpoint + "/freeze", json=intent)
    assert response.status_code == 201, response.text
    frozen = response.json()
    assert frozen["versionLabel"]["tag"] == "Reviewed data v2"
    assert frozen["versionLabel"]["note"] == "Updated review."
    assert frozen["versionLabel"]["revision"] == 1
    assert "versionLabel" not in frozen["manifest"]
    assert (
        "Reviewed data v2"
        not in (store.folder / "datasets" / frozen["id"] / "manifest.json").read_text()
    )
    assert store.get_draft(draft["id"])["status"] == "frozen"

    changed = client.put(
        route(project, frozen) + "/label",
        json={
            "tag": "Reviewed data final",
            "note": "Renamed after freezing",
            "expectedRevision": 1,
        },
    )
    assert changed.status_code == 200, changed.text
    source.unlink()
    replay = client.post(endpoint + "/freeze", json=intent)
    assert replay.status_code == 201, replay.text
    assert replay.json()["id"] == frozen["id"]
    assert replay.json()["versionLabel"] == changed.json()
    modified_intent = {**intent, "versionLabel": {"tag": "A different original intent"}}
    rejected = client.post(endpoint + "/freeze", json=modified_intent)
    assert rejected.status_code == 409, rejected.text
    assert rejected.json()["code"] == "OPERATION_CONFLICT"
