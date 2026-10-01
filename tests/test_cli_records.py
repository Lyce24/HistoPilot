"""Record nouns over an in-process project: lists, show, @tag names and verified downloads."""

import hashlib
import json

import pytest
from support import projects as fixtures
from support.cli import Service

from histopilot.client import Client, ClientError
from histopilot.client.records import download, recorded_sha256
from histopilot.client.transport import Response
from histopilot.storage.scientific import ScientificStore


@pytest.fixture
def study(tmp_path, monkeypatch):
    with Service(tmp_path, monkeypatch) as service:
        project = service.create_project()
        store = ScientificStore(service.settings.workspace / "projects" / "Study", project)
        fixtures.setup(store, service.data)
        service.cli("use", project)
        yield service, project


def test_every_kind_lists_its_records(study):
    service, _ = study
    datasets = service.cli("dataset", "list", "--json").envelope["data"]
    assert len(datasets) == 1 and datasets[0]["id"].startswith("dataset-")
    kinds = {
        row["manifest"]["kind"]
        for row in service.cli("configuration", "list", "--json").envelope["data"]
    }
    assert kinds == {"feature", "feature-bundle", "protocol"}
    assert len(service.cli("bundle", "list", "--json").envelope["data"]) == 1
    packs = service.cli("pack", "list", "--json").envelope["data"]
    assert [(job["state"], job["runState"]) for job in packs] == [("succeeded", "succeeded")]
    for noun in ("targets", "predictor", "cohort", "run", "apply", "reference", "analysis"):
        outcome = service.cli(noun, "list", "--json")
        assert outcome.code == 0 and outcome.envelope["data"] == [], noun
    text = service.cli("configuration", "list")
    assert text.code == 0 and text.stdout.startswith("ID") and "feature-bundle" in text.stdout


def test_records_are_shown_by_id_or_by_version_tag(study):
    service, project = study
    dataset = service.cli("dataset", "list", "--json").envelope["data"][0]["id"]
    labelled = service.http.put(
        f"/api/v1/projects/{project}/datasets/{dataset}/label",
        json={"tag": "baseline v1", "note": "", "expectedRevision": 0},
    )
    assert labelled.status_code == 200, labelled.text
    by_tag = service.cli("dataset", "show", "@baseline v1", "--json").envelope["data"]
    assert by_tag["id"] == dataset
    assert (
        service.cli("dataset", "show", "@Baseline V1", "--json").envelope["data"]["id"] == dataset
    )
    missing = service.cli("dataset", "show", "@nope", "--json")
    assert missing.code == 5 and missing.envelope["error"]["code"] == "TAG_NOT_FOUND"
    untagged = service.cli("predictor", "show", "@x", "--json")
    assert untagged.code == 2 and untagged.envelope["error"]["code"] == "USAGE_ERROR"
    shown = service.cli("dataset", "show", dataset)
    assert shown.code == 0 and "Add --json for every field." in shown.stdout


def test_dataset_records_are_paged(study):
    service, _ = study
    dataset = service.cli("dataset", "list", "--json").envelope["data"][0]["id"]
    first = service.cli("dataset", "records", dataset, "--limit", "3", "--json").envelope
    assert [row["slideId"] for row in first["data"]] == ["s0", "s1", "s2"]
    assert first["page"]["hasMore"] is True
    rest = service.cli("dataset", "records", dataset, "--offset", "3", "--json").envelope
    assert [row["slideId"] for row in rest["data"]] == ["s3"] and rest["page"]["hasMore"] is False


class Bytes:
    def __init__(self, content):
        self.content = content

    def send(self, method, path, *, headers, body, timeout):
        if path.endswith("/session"):
            return Response(200, json.dumps({"token": "t"}).encode(), {})
        return Response(200, self.content, {"content-type": "text/csv"})


def test_downloads_are_checked_against_the_recorded_hash():
    content = b"slide,probability\ns0,0.9\n"
    good = hashlib.sha256(content).hexdigest()
    client = Client(transport=Bytes(content))
    data, check = download(client, "/projects/p/evaluation-runs/r/artifacts/x.csv", expected=good)
    assert data == content and check == {"bytes": len(content), "sha256": good, "verified": True}
    with pytest.raises(ClientError) as raised:
        download(client, "/projects/p/evaluation-runs/r/artifacts/x.csv", expected="0" * 64)
    assert raised.value.code == "DOWNLOAD_MISMATCH"
    run = {"execution": {"result": {"artifacts": {"x.csv": {"sha256": good, "bytes": 26}}}}}
    assert recorded_sha256(run, "x.csv") == good and recorded_sha256(run, "y.csv") is None


def test_cleanup_previews_then_trashes_and_restores_with_confirmation(study):
    service, _ = study
    listed = service.cli("cleanup", "list", "--json").envelope["data"]
    draft = next(
        row for row in listed if row["kind"] == "evaluation-cohort" or row["type"] == "draft"
    )
    preview = service.cli("cleanup", "preview", draft["key"], "--action", "trash", "--json")
    assert preview.code == 0 and preview.envelope["data"]["canApply"] is True

    pending = service.cli("cleanup", "apply", draft["key"], "--action", "trash", "--json")
    assert pending.code == 7
    assert (
        pending.envelope["data"]["preview"]["previewHash"]
        == preview.envelope["data"]["previewHash"]
    )
    done = service.cli("cleanup", "apply", draft["id"], "--action", "trash", "--yes", "--json")
    assert done.code == 0, done.stdout
    states = {
        row["key"]: row["state"]
        for row in service.cli("cleanup", "list", "--json").envelope["data"]
    }
    assert states[draft["key"]] == "trashed"
    restored = service.cli(
        "cleanup", "apply", draft["key"], "--action", "restore", "--yes", "--json"
    )
    assert restored.code == 0, restored.stdout
    assert service.cli("cleanup", "apply", "nonsense", "--action", "trash", "--json").code == 2


def test_blocked_cleanup_refuses_before_changing_anything(study):
    service, _ = study
    dataset = service.cli("dataset", "list", "--json").envelope["data"][0]["id"]
    # The feature source and protocol still use the dataset.
    outcome = service.cli(
        "cleanup", "apply", f"dataset:{dataset}", "--action", "trash", "--yes", "--json"
    )
    assert outcome.code == 3
    error = outcome.envelope["error"]
    assert error["code"] == "PREVIEW_BLOCKED" and error["findings"]


def test_sources_are_listed_and_changes_need_confirmation(study):
    service, _ = study
    listed = service.cli("sources", "list", "--json").envelope["data"]
    assert listed["sources"] == []
    folder = service.data / "slides"
    folder.mkdir()
    pending = service.cli("sources", "add", str(folder), "--role", "slides", "--json")
    assert pending.code == 7 and pending.envelope["data"]["preview"]["routeClass"] == "admin"
    added = service.cli("sources", "add", str(folder), "--role", "slides", "--yes", "--json")
    assert added.code == 0, added.stdout


def test_backups_queue_an_archive_task_after_confirmation(study):
    service, _ = study
    archive = service.data / "study-backup.zip"
    pending = service.cli("backup", "export", str(archive), "--json")
    assert pending.code == 7
    assert pending.envelope["data"]["preview"]["archivePath"] == str(archive)
    queued = service.cli("backup", "export", str(archive), "--yes", "--json")
    assert queued.code == 0, queued.stdout
    jobs = service.cli("backup", "list", "--json").envelope["data"]
    assert len(jobs) == 1 and jobs[0]["runState"] in ("queued", "running")
