"""Page reads never wait for, or fail behind, a project writer (hardening 1.1 and 1.3)."""

import copy
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from histopilot.api.scientific import scientific_router
from histopilot.application.lifecycle import CleanupService
from histopilot.application.model_experiments import ModelExperimentService
from histopilot.application.operations import _missing_references
from histopilot.application.target_splits import TargetSplitService
from histopilot.schemas.model_experiments import CreateModelExperiment
from histopilot.storage import scientific
from histopilot.storage.filesystem import LocalFilesystem
from histopilot.storage.lifecycle import lifecycle_guard
from histopilot.storage.project_lock import StorageError, writer_lock
from histopilot.storage.scientific import ScientificStore

TARGET = {
    "field": "label",
    "task": "binary_classification",
    "unit": "patient",
    "classes": ["low", "high"],
    "labels": {"0": "low", "1": "high"},
    "positiveClass": "high",
}


@contextmanager
def writer_holds_project(folder):
    """Another thread keeps the lifecycle guard and writer lock until the block ends."""
    held, release = threading.Event(), threading.Event()

    def hold():
        with lifecycle_guard(folder), writer_lock(folder):
            held.set()
            release.wait(30)

    thread = threading.Thread(target=hold, daemon=True)
    thread.start()
    assert held.wait(10)
    try:
        yield
    finally:
        release.set()
        thread.join(10)


def quickly(call, limit=3.0):
    start = time.monotonic()
    result = call()
    assert time.monotonic() - start < limit
    return result


@pytest.fixture
def construction(tmp_path):
    rows = [
        {
            "slideId": f"s{patient:02}-{slide}",
            "patientId": f"p{patient:02}",
            "attributes": {"label": str(patient % 2)},
        }
        for patient in range(40)
        for slide in range(2)
    ]
    (tmp_path / "project").mkdir()
    store = ScientificStore(tmp_path / "project", "project-reads")
    source = store.create_draft("import", "Dataset", {})
    dataset = store.publish_dataset(
        source["id"],
        expected_revision=1,
        manifest={
            "kind": "dataset",
            "dictionary": [
                {"key": "label", "sourceColumn": "label", "owner": "slide", "type": "text"}
            ],
        },
        artifacts={"records.json": json.dumps(rows).encode()},
        operation_id="dataset",
    )
    service = TargetSplitService(store, LocalFilesystem((tmp_path,)))
    spec = {"datasetId": dataset["id"], "target": copy.deepcopy(TARGET)}
    return store, service, spec


def freeze(construction, operation="freeze", *, replay=False):
    store, service, spec = construction
    draft = store.create_draft("experiment", "Targets", {"type": "target-split", "spec": spec})
    preview = service.preview(draft["id"], 1)
    assert preview["canFreeze"], preview["findings"]

    def submit():
        return service.freeze(
            draft["id"], 1, preview["previewHash"], operation, version_label={"tag": operation}
        )

    return (submit(), submit) if replay else submit()


def test_store_reads_do_not_wait_for_a_writer_even_before_first_validation(construction):
    store = construction[0]
    with writer_holds_project(store.folder):
        for cold in (True, False):
            if cold:
                # A new service process has validated nothing yet.
                scientific._READY.clear()
            reader = ScientificStore(store.folder, store.project_id)
            assert quickly(lambda reader=reader: reader.list_datasets())
            assert quickly(lambda reader=reader: reader.list_drafts())
            quickly(lambda reader=reader: reader.list_configurations("target-split"))
            assert quickly(lambda reader=reader: reader.status())["schemaVersion"] == 4


def test_parallel_page_reads_do_not_report_busy_while_a_writer_holds_the_project(construction):
    store, service, _spec = construction
    target = freeze(construction)
    app = FastAPI()
    app.include_router(
        scientific_router(SimpleNamespace(scientific_store=lambda _id: store), service.filesystem)
    )
    base = "/api/v1/projects/project-reads"
    paths = [
        *(
            f"{base}/configurations?kind={kind}"
            for kind in ("protocol", "target-split", "feature", "evaluation-cohort")
        ),
        f"{base}/configurations",
        f"{base}/target-splits/{target['id']}",
        f"{base}/configurations/{target['id']}",
    ] * 2
    with TestClient(app) as client, writer_holds_project(store.folder):
        start = time.monotonic()
        with ThreadPoolExecutor(len(paths)) as pool:
            responses = list(pool.map(client.get, paths))
        # Each read would otherwise wait 5 s for the lock and then answer 409.
        assert time.monotonic() - start < 4
    assert [response.status_code for response in responses] == [200] * len(paths)
    assert responses[5].json() == target


def test_experiment_list_and_get_take_no_exclusive_lock(tmp_path):
    folder = tmp_path / "project"
    folder.mkdir()
    service = ModelExperimentService(
        ScientificStore(folder, "project-experiments"), LocalFilesystem((tmp_path,))
    )
    record = service.create(CreateModelExperiment(name="Parallel", operationId="create"))
    with writer_holds_project(folder):
        assert quickly(service.list)["items"][0]["id"] == record["id"]
        assert quickly(lambda: service.list(summary=True))["items"][0]["summary"] is True
        assert quickly(lambda: service.get(record["id"]))["name"] == "Parallel"
        assert quickly(lambda: CleanupService(service.store, service.filesystem).catalog())


def test_verified_configuration_cache_still_detects_a_changed_row(construction):
    store = construction[0]
    target = freeze(construction)
    assert [item["id"] for item in store.list_configurations("target-split")] == [target["id"]]
    connection = scientific.sqlite3.connect(store.path)
    try:
        document = json.loads(
            connection.execute(
                "SELECT document FROM configurations WHERE id=?", (target["id"],)
            ).fetchone()[0]
        )
        document["manifest"]["summary"]["trainingSlides"] += 1
        with connection:
            connection.execute(
                "UPDATE configurations SET document=? WHERE id=?",
                (json.dumps(document), target["id"]),
            )
    finally:
        connection.close()
    for read in (
        lambda: store.list_configurations("target-split"),
        lambda: store.get_configuration(target["id"]),
    ):
        with pytest.raises(StorageError) as error:
            read()
        assert error.value.code == "STORAGE_CORRUPT"


def test_kind_filter_matches_the_unfiltered_list(construction):
    store = construction[0]
    freeze(construction)
    everything = store.list_configurations()
    for kind in {item["manifest"]["kind"] for item in everything} | {"protocol"}:
        expected = [item for item in everything if item["manifest"]["kind"] == kind]
        assert store.list_configurations(kind) == expected
        assert store.list_configurations(kind) == expected  # cached kinds


def test_missing_references_match_the_exact_per_path_check(tmp_path):
    root, outside = tmp_path / "root", tmp_path / "outside"
    (root / "slides").mkdir(parents=True)
    (root / "empty").mkdir()
    outside.mkdir()
    (root / "slides" / "present.svs").write_bytes(b"x")
    (outside / "elsewhere.svs").write_bytes(b"x")
    os.symlink(outside / "elsewhere.svs", root / "slides" / "escape.svs")
    os.symlink(root / "slides" / "gone.svs", root / "slides" / "dangling.svs")
    project = tmp_path / "project"
    project.mkdir()
    paths = {
        str(root / "slides" / "present.svs"),
        str(root / "slides" / "absent.svs"),
        str(root / "slides" / "escape.svs"),
        str(root / "slides" / "dangling.svs"),
        str(root / "empty" / "absent.svs"),
        str(root / "missing-folder" / "a.svs"),
        str(root / "slides" / "present.svs" / "child"),
        str(outside / "elsewhere.svs"),
        str(project / "internal.json"),
    }
    filesystem = LocalFilesystem((root,))

    def exact():
        missing = []
        for value in sorted(paths):
            path = Path(value)
            if path.is_relative_to(project):
                continue
            allowed = filesystem._contains(path.resolve())
            if not allowed or not path.exists():
                missing.append({"path": value, "reason": "missing" if allowed else "outside-roots"})
        return missing

    assert sorted(_missing_references(filesystem, paths, project), key=str) == sorted(
        exact(), key=str
    )


def test_target_split_get_never_writes_and_freeze_reports_cohort_failure_as_warning(
    construction, monkeypatch
):
    store, service, _spec = construction

    def blocked(_self, _source, _identity, _operation):
        raise StorageError("The testing cohort cannot be prepared.", "TARGET_TESTING_BLOCKED", 422)

    with monkeypatch.context() as patch:
        patch.setattr(TargetSplitService, "_derive_test_cohort", blocked)
        target, retry = freeze(construction, replay=True)
        # A retried freeze replays the publication and repeats the same warning, not an error.
        assert retry() == target
    error = {"code": "TARGET_TESTING_BLOCKED", "message": "The testing cohort cannot be prepared."}
    failed = {"required": True, "id": None, "state": "failed", "error": error}
    assert target["testCohortError"] == error
    assert target["evaluationCohortId"] is None
    assert target["testCohort"] == failed
    # The failure is kept beside the frozen version, never in its content-addressed manifest.
    assert "testCohort" not in target["manifest"]
    record = store.folder / "target-splits" / target["id"] / "test-cohort-failure.json"
    assert json.loads(record.read_text())["error"] == error
    before = store.list_configurations(include_inactive=True)
    with writer_holds_project(store.folder):
        detail = quickly(lambda: service.get(target["id"]))
    # A reload still knows that freezing could not make the test cohort, so it offers the retry.
    assert "testCohortError" not in detail
    assert detail["testCohort"] == failed
    assert store.list_configurations(include_inactive=True) == before
    cohort = service.create_test_cohort(target["id"])
    detail = service.get(target["id"])
    assert detail["evaluationCohortId"] == cohort["id"]
    assert detail["testCohort"] == {"required": True, "id": cohort["id"], "state": "active"}
    assert not record.exists()


def test_target_split_get_reports_a_trashed_testing_cohort_instead_of_failing(construction):
    store, service, _spec = construction
    target = freeze(construction)
    cohort_id = target["evaluationCohortId"]
    assert cohort_id and target["testCohort"]["state"] == "active"
    store.lifecycle.apply(
        {f"configuration:{cohort_id}": "trashed"},
        "trash-cohort",
        hashlib.sha256(b"trash-cohort").hexdigest(),
        0,
    )
    detail = service.get(target["id"])
    assert detail["evaluationCohortId"] is None
    assert detail["testCohort"] == {"required": True, "id": cohort_id, "state": "trashed"}


def test_parallel_cold_reads_verify_each_configuration_once(construction, monkeypatch):
    store = construction[0]
    freeze(construction)
    scientific._VERIFIED_CONFIGURATIONS.clear()
    calls = []
    original = ScientificStore._verify_configuration

    def counted(self, identity, *args):
        calls.append(identity)
        time.sleep(0.05)
        return original(self, identity, *args)

    monkeypatch.setattr(ScientificStore, "_verify_configuration", counted)
    readers = [ScientificStore(store.folder, store.project_id) for _ in range(8)]
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda reader: reader.list_configurations("target-split"), readers))
    assert all(len(result) == 1 for result in results)
    assert sorted(calls) == sorted({item["id"] for item in store.list_configurations()})


def test_a_cache_cleared_by_another_thread_mid_listing_is_not_an_error(construction, monkeypatch):
    """Another reader may clear the verification cache (its size limit) between a row's
    check and its kind lookup; the row then counts as unverified instead of a 500."""
    store = construction[0]
    freeze(construction)
    expected = [item["id"] for item in store.list_configurations("target-split")]
    original = ScientificStore._verified

    def racing(self, *args):
        verified = original(self, *args)
        if verified:  # right after a positive check, another reader hits the size limit
            scientific._VERIFIED_CONFIGURATIONS.clear()
        return verified

    monkeypatch.setattr(ScientificStore, "_verified", racing)
    listed = store.list_configurations("target-split")
    assert [item["id"] for item in listed] == expected
