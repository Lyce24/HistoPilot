"""Experiment entry flow, folder durability, isolation, and path validation."""

import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from histopilot.api import create_app
from histopilot.application.project_workspace import DESCRIPTOR
from histopilot.config import Settings

API = "/api/v1"


@pytest.fixture
def settings(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    return Settings(workspace=tmp_path / "workspace", data_roots=(data,))


def connection(settings):
    return TestClient(create_app(settings), base_url="http://127.0.0.1:8787")


def authenticate(client):
    client.headers["X-HistoPilot-Token"] = client.get(f"{API}/session").json()["token"]


@pytest.fixture
def client(settings):
    with connection(settings) as current:
        authenticate(current)
        yield current


def create(client, settings, name="Bladder", **overrides):
    response = client.post(
        f"{API}/projects",
        json={
            "name": name,
            "storagePath": str(settings.workspace / name),
            **overrides,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_create_optional_setup_is_folder_backed_and_survives_restart(settings):
    source = settings.data_roots[0]
    with connection(settings) as first:
        authenticate(first)
        created = create(
            first,
            settings,
            description="Grade prediction",
            dataPath=str(source),
            slidePath=str(source),
            featurePath=str(source),
            config={
                "task": "binary_classification",
                "targetColumn": "grade",
                "positiveLabel": "high",
                "seed": 7,
                "folds": 3,
                "encoderId": "uni2",
                "milId": "abmil",
            },
        )
        state = first.get(f"{API}/projects/{created['id']}/workspace").json()
        assert state["project"] == created
        assert state["mode"] == "local"
        assert state["executionEnabled"] is False
        assert state["split"] == {"id": "", "seed": 7, "groupBy": "patient_id"}
        assert state["dataset"] == {
            "id": "",
            "patientCount": 0,
            "fallbackSlideCount": 0,
            "groupCount": 0,
            "unlinkedSlideCount": 0,
            "specimenCount": 0,
            "slideCount": 0,
        }
        for key in (
            "patients",
            "slides",
            "featureSets",
            "results",
            "cohortSnapshots",
            "drafts",
            "exampleManifests",
        ):
            assert state[key] == []
        assert state["encoders"] and state["milModels"]
        assert {item["role"] for item in state["sources"]} == {"data", "slides", "features"}
        assert all(
            item["readOnly"] and item["importStatus"] == "not-imported" for item in state["sources"]
        )
    descriptor = settings.workspace / "Bladder" / DESCRIPTOR
    stored = json.loads(descriptor.read_text())
    assert stored["id"] == created["id"]
    assert stored["config"] == created["config"]
    assert stored["sources"] == created["sources"]
    with connection(settings) as second:
        authenticate(second)
        assert second.get(f"{API}/projects/{created['id']}/workspace").json() == state
        assert (
            second.post(f"{API}/projects/open", json={"path": created["storagePath"]}).json()
            == created
        )
        projects = second.get(f"{API}/projects").json()["projects"]
        assert [item for item in projects if item["mode"] == "local"] == [created]


def test_create_with_only_prerequisites_and_update_settings_later(client, settings):
    project = create(client, settings)
    assert project["config"] == {}
    assert project["sources"] == []
    payload = {"config": {"task": "multiclass_classification", "targetColumn": "grade", "folds": 4}}
    response = client.patch(f"{API}/projects/{project['id']}", json=payload)
    assert response.status_code == 200
    assert response.json()["config"] == payload["config"]
    assert response.json()["storagePath"] == project["storagePath"]
    assert response.json()["id"] == project["id"]
    state = client.get(f"{API}/projects/{project['id']}/workspace").json()
    assert state["project"] == response.json()
    cleared = client.patch(f"{API}/projects/{project['id']}", json={"config": {}})
    assert cleared.status_code == 200
    assert cleared.json()["config"] == {}
    assert (
        client.patch(f"{API}/projects/{project['id']}", json={"storagePath": "/tmp"}).status_code
        == 422
    )


def test_open_descriptor_in_fresh_registry_and_allow_moved_folder(settings, tmp_path):
    folder = settings.data_roots[0] / "Bladder"
    with connection(settings) as first:
        authenticate(first)
        created = create(first, settings, storagePath=str(folder))
    fresh = replace(settings, workspace=tmp_path / "fresh-registry")
    with connection(fresh) as second:
        authenticate(second)
        assert len(second.get(f"{API}/projects").json()["projects"]) == 1
        loaded = second.post(f"{API}/projects/open", json={"path": str(folder)})
        assert loaded.status_code == 200
        assert loaded.json() == created
        moved = folder.with_name("renamed")
        folder.rename(moved)
        unavailable = second.get(f"{API}/projects").json()["projects"][0]
        assert unavailable["available"] is False
        loaded = second.post(f"{API}/projects/open", json={"path": str(moved)})
        assert loaded.status_code == 200
        assert loaded.json()["id"] == created["id"]
        assert loaded.json()["storagePath"] == str(moved)
        assert second.get(f"{API}/projects/{created['id']}/workspace").status_code == 200


def test_project_paths_and_legacy_demo_remain_isolated(client, settings):
    first = create(client, settings)
    second = create(client, settings, name="CRC KRAS")
    source = str(settings.data_roots[0])
    with ThreadPoolExecutor(max_workers=4) as executor:
        responses = list(
            executor.map(
                lambda _: client.post(
                    f"{API}/projects/{first['id']}/sources", json={"path": source, "role": "slides"}
                ),
                range(6),
            )
        )
    assert all(response.status_code == 201 for response in responses)
    assert all(response.json() == responses[0].json() for response in responses)
    first_state = client.get(f"{API}/projects/{first['id']}/workspace").json()
    assert first_state["sources"] == [responses[0].json()]
    assert client.get(f"{API}/projects/{second['id']}/workspace").json()["sources"] == []
    demo = client.get(f"{API}/projects/synthetic-v1/workspace").json()
    assert demo["mode"] == "synthetic-demo"
    assert demo["project"]["id"] == "synthetic-v1"
    assert demo["patients"]
    assert demo["sources"] == []
    assert client.get(f"{API}/workspace").json()["patients"] == demo["patients"]
    legacy = client.post(f"{API}/sources", json={"path": source}).json()
    assert client.get(f"{API}/workspace").json()["sources"] == [legacy]
    assert (
        client.get(f"{API}/projects/{first['id']}/workspace").json()["sources"]
        == first_state["sources"]
    )


@pytest.mark.parametrize(
    "override",
    [
        {"name": "  "},
        {"storagePath": ""},
        {"config": {"seed": True}},
        {"config": {"seed": -1}},
        {"config": {"seed": 2**32}},
        {"config": {"folds": 1}},
        {"config": {"folds": 11}},
        {"config": {"folds": True}},
        {"config": {"task": "anything"}},
        {"config": {"encoderId": "unknown"}},
        {"config": {"milId": "unknown"}},
        {"config": {"targetColumn": ""}},
        {"config": {"command": "malicious"}},
        {"manifest": {"command": "malicious"}},
    ],
)
def test_invalid_setup_leaves_no_project_or_directory(client, settings, override):
    folder = settings.workspace / "invalid"
    response = client.post(
        f"{API}/projects",
        json={
            "name": "Invalid",
            "storagePath": str(folder),
            **override,
        },
    )
    assert response.status_code == 422
    assert not folder.exists()
    assert len(client.get(f"{API}/projects").json()["projects"]) == 1


def test_storage_picker_expands_storage_only_and_blocks_escape(client, settings, tmp_path):
    default = client.get(f"{API}/projects").json()["defaultStoragePath"]
    assert default == str(settings.workspace)
    roots = client.get(f"{API}/filesystem/roots", params={"purpose": "storage"}).json()["roots"]
    assert {root["path"] for root in roots} == {
        str(settings.workspace),
        str(settings.data_roots[0]),
    }
    assert (
        client.get(
            f"{API}/filesystem/list", params={"path": default, "purpose": "storage"}
        ).status_code
        == 200
    )
    assert client.get(f"{API}/filesystem/list", params={"path": default}).status_code == 403
    assert client.post(f"{API}/sources", json={"path": default}).status_code == 403
    outside = tmp_path / "private"
    outside.mkdir()
    (settings.workspace / "escape").symlink_to(outside, target_is_directory=True)
    for path in (
        outside / "new",
        settings.workspace / "escape" / "new",
        settings.workspace / ".." / "private" / "new",
    ):
        response = client.post(f"{API}/projects", json={"name": "Escape", "storagePath": str(path)})
        assert response.status_code == 403
    assert not (outside / "new").exists()
    assert client.post(f"{API}/projects/open", json={"path": str(outside)}).status_code == 403
    assert (
        client.post(
            f"{API}/projects", json={"name": "Relative", "storagePath": "relative"}
        ).status_code
        == 400
    )
    response = client.post(
        f"{API}/projects",
        json={
            "name": "Invalid source",
            "storagePath": str(settings.workspace / "bad-source"),
            "slidePath": str(outside),
        },
    )
    assert response.status_code == 403
    assert not (settings.workspace / "bad-source").exists()


def test_create_never_overwrites_existing_contents_or_existing_experiment(client, settings):
    folder = settings.workspace / "occupied"
    folder.mkdir()
    sentinel = folder / "original.svs"
    sentinel.write_bytes(b"original slide bytes")
    assert (
        client.post(
            f"{API}/projects", json={"name": "Occupied", "storagePath": str(folder)}
        ).status_code
        == 409
    )
    assert sentinel.read_bytes() == b"original slide bytes"
    assert not (folder / DESCRIPTOR).exists()
    empty = settings.workspace / "empty"
    empty.mkdir()
    created = create(client, settings, storagePath=str(empty))
    assert (
        client.post(
            f"{API}/projects", json={"name": "Duplicate", "storagePath": str(empty)}
        ).status_code
        == 409
    )
    assert client.get(f"{API}/projects/{created['id']}/workspace").json()["project"] == created
    copied = settings.workspace / "copied"
    shutil.copytree(empty, copied)
    assert client.post(f"{API}/projects/open", json={"path": str(copied)}).status_code == 409


@pytest.mark.parametrize("content", ["not json", "{}", '{"format":"other","schemaVersion":1}'])
def test_invalid_descriptor_and_missing_project_have_clear_errors(client, settings, content):
    folder = settings.workspace / "invalid-descriptor"
    folder.mkdir()
    assert client.post(f"{API}/projects/open", json={"path": str(folder)}).status_code == 404
    (folder / DESCRIPTOR).write_text(content)
    assert client.post(f"{API}/projects/open", json={"path": str(folder)}).status_code == 422
    assert client.get(f"{API}/projects/missing/workspace").status_code == 404
    assert client.patch(f"{API}/projects/missing", json={"config": {}}).status_code == 404


def test_descriptor_symlink_and_identity_replacement_are_rejected(client, settings, tmp_path):
    created = create(client, settings)
    folder = settings.workspace / "Bladder"
    descriptor = folder / DESCRIPTOR
    original = descriptor.read_text()
    descriptor.unlink()
    outside = tmp_path / "private-descriptor.json"
    outside.write_text(original)
    descriptor.symlink_to(outside)
    assert client.post(f"{API}/projects/open", json={"path": str(folder)}).status_code == 403
    assert client.get(f"{API}/projects/{created['id']}/workspace").status_code == 403
    descriptor.unlink()
    modified = json.loads(original)
    modified["id"] = "project-" + "0" * 32
    descriptor.write_text(json.dumps(modified))
    assert client.get(f"{API}/projects/{created['id']}/workspace").status_code == 409
    projects = client.get(f"{API}/projects").json()["projects"]
    assert projects[0]["available"] is False


def test_descriptor_must_be_bounded_regular_file(client, settings):
    folder = settings.workspace / "unsafe-descriptor"
    folder.mkdir()
    descriptor = folder / DESCRIPTOR
    descriptor.write_bytes(b" " * (1024 * 1024 + 1))
    assert client.post(f"{API}/projects/open", json={"path": str(folder)}).status_code == 422
    descriptor.unlink()
    descriptor.mkdir()
    assert client.post(f"{API}/projects/open", json={"path": str(folder)}).status_code == 422
    descriptor.rmdir()
    if hasattr(os, "mkfifo"):
        os.mkfifo(descriptor)
        assert client.post(f"{API}/projects/open", json={"path": str(folder)}).status_code == 422


def test_project_routes_require_token_and_patch_supports_dev_cors(settings):
    with connection(replace(settings, dev=True)) as client:
        assert client.get(f"{API}/projects").status_code == 401
        assert client.post(f"{API}/projects", json={}).status_code == 401
        assert client.post(f"{API}/projects/open", json={}).status_code == 401
        assert client.patch(f"{API}/projects/anything", json={"config": {}}).status_code == 401
        response = client.options(
            f"{API}/projects/anything",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "PATCH",
                "Access-Control-Request-Headers": "content-type,x-histopilot-token",
            },
        )
        assert response.status_code == 200
