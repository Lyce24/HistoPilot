"""Renaming a project changes its name only: never its ID, folder or records."""

import json

import pytest
from support.cli import Service

API = "/api/v1"


@pytest.fixture
def service(tmp_path, monkeypatch):
    with Service(tmp_path, monkeypatch) as current:
        project = current.create_project("Colon")
        current.cli("use", project)
        yield current, project


def descriptor(service, project):
    folder = service.http.get(f"{API}/projects/{project}/workspace").json()["project"][
        "storagePath"
    ]
    with open(f"{folder}/histopilot-project.json") as handle:
        return folder, json.load(handle)


def test_rename_previews_then_changes_the_name_only(service):
    cli, project = service
    folder, before = descriptor(cli, project)
    pending = cli.cli("project", "rename", "--name", "  CRC KRAS ", "--json")
    assert pending.code == 7
    preview = pending.envelope["data"]["preview"]
    assert (preview["projectId"], preview["from"], preview["to"]) == (project, "Colon", "CRC KRAS")
    assert descriptor(cli, project)[1]["name"] == "Colon"

    done = cli.cli("project", "rename", "--name", "CRC KRAS", "--yes", "--json")
    assert done.code == 0
    result = done.envelope["data"]["result"]
    assert (result["name"], result["previous"]) == ("CRC KRAS", "Colon")
    after_folder, after = descriptor(cli, project)
    assert after_folder == folder and after["id"] == before["id"] == project
    assert after["name"] == "CRC KRAS" and after["updatedAt"] >= before["updatedAt"]
    assert {key: value for key, value in after.items() if key not in {"name", "updatedAt"}} == {
        key: value for key, value in before.items() if key not in {"name", "updatedAt"}
    }
    listed = cli.http.get(f"{API}/projects").json()["projects"]
    assert [item["name"] for item in listed if item["id"] == project] == ["CRC KRAS"]


def test_a_stale_name_conflicts_and_a_retry_is_a_no_op(service):
    cli, project = service
    path = f"{API}/projects/{project}/name"
    renamed = cli.http.put(path, json={"name": "GEJ BE", "expectedName": "Colon"})
    assert renamed.status_code == 200 and renamed.json()["name"] == "GEJ BE"
    stale = cli.http.put(path, json={"name": "Other", "expectedName": "Colon"})
    assert stale.status_code == 409 and stale.json()["code"] == "PROJECT_NAME_CONFLICT"
    again = cli.http.put(path, json={"name": "GEJ BE", "expectedName": "Colon"})
    assert again.status_code == 200 and again.json()["previous"] == "GEJ BE"
    assert descriptor(cli, project)[1]["name"] == "GEJ BE"


def test_blank_names_and_the_demo_are_refused(service):
    cli, project = service
    blank = cli.http.put(
        f"{API}/projects/{project}/name", json={"name": "   ", "expectedName": "Colon"}
    )
    assert blank.status_code == 422
    assert cli.cli("project", "rename", "--name", "  ", "--json").code == 2
    assert cli.cli("project", "rename", "--name", "x" * 121, "--json").code == 2
    demo = cli.http.put(
        f"{API}/projects/blca-demo-v1/name", json={"name": "Mine", "expectedName": "BLCA demo"}
    )
    assert demo.status_code == 409 and demo.json()["code"] == "DEMO_PROJECT_READ_ONLY"
    assert descriptor(cli, project)[1]["name"] == "Colon"
