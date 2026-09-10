"""Folder portability, revision authority, and crash-safe scientific publication."""

import json
import os
import sqlite3
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

from histopilot.storage import scientific
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import DATABASE_FILE, ScientificStore


class Interrupted(BaseException):
    pass


@pytest.fixture
def store(tmp_path):
    folder = tmp_path / "experiment"
    folder.mkdir()
    (folder / "histopilot-project.json").write_bytes(b'{"legacy":"descriptor sentinel"}\n')
    current = ScientificStore(folder, "project-test")
    current.initialize()
    return current


@contextmanager
def database(store):
    connection = sqlite3.connect(store.path)
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def draft(store, name="Bladder import"):
    return store.create_draft("import", name, {"mapping": {"De ID": "Slide_ID"}})


def publish(store, source=None, *, operation="freeze", **overrides):
    source = source or draft(store)
    return store.publish_dataset(
        source["id"],
        **{
            "expected_revision": source["revision"],
            "manifest": {"name": "Bladder", "mapping": {"De ID": "Slide_ID"}},
            "artifacts": {
                "tables/slides.json": b'[{"Slide_ID":"fixture-001"}]',
                "sources/table.csv": b"De ID\nfixture-001\n",
            },
            "operation_id": operation,
            **overrides,
        },
    )


def interrupt_at(store, phase):
    def checkpoint(current):
        if current == phase:
            raise Interrupted(current)

    store._checkpoint = checkpoint


def reopened(store):
    return ScientificStore(store.folder, store.project_id)


def test_initialization_preserves_descriptor_and_uses_local_durable_metadata(store):
    original = (store.folder / "histopilot-project.json").read_bytes()
    saved = store.create_draft(
        "experiment",
        "Initial choices",
        {
            "status": "frozen",
            "projectId": "forged",
            "revision": 99,
            "id": "forged",
        },
    )
    assert saved["projectId"] == "project-test"
    assert saved["status"] == "editable"
    assert saved["revision"] == 1
    assert saved["id"] != "forged"
    assert saved["payload"]["status"] == "frozen"
    assert reopened(store).get_draft(saved["id"]) == saved
    assert reopened(store).list_drafts() == [saved]
    assert (store.folder / "histopilot-project.json").read_bytes() == original
    status = store.status()
    assert status["schemaVersion"] == 2
    assert status["journalMode"] == "wal"
    assert status["database"] == DATABASE_FILE
    assert status["draftCount"] == 1
    with store._connection() as connection:
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2


def test_draft_revision_cas_and_frozen_immutability(store):
    saved = draft(store)
    second = reopened(store)
    updated = store.update_draft(
        saved["id"], expected_revision=1, name="Reviewed", payload={"revision": 20}
    )
    assert updated["revision"] == 2
    with pytest.raises(StorageError) as error:
        second.update_draft(saved["id"], expected_revision=1, name="Stale", payload={})
    assert error.value.code == "REVISION_CONFLICT"
    assert second.get_draft(saved["id"]) == updated
    frozen = publish(store, updated)
    assert store.get_draft(saved["id"])["revision"] == 3
    assert store.get_draft(saved["id"])["status"] == "frozen"
    with pytest.raises(StorageError) as error:
        store.update_draft(saved["id"], expected_revision=3, name="Overwrite", payload={})
    assert error.value.code == "DRAFT_FROZEN"
    assert store.get_dataset(frozen["id"]) == frozen


@pytest.mark.parametrize("revision", [0, -1, True, 1.5, "1", 2**63, 2**100])
def test_invalid_internal_revisions_do_not_write(store, revision):
    saved = draft(store)
    with pytest.raises(StorageError) as error:
        store.update_draft(saved["id"], expected_revision=revision, name="Invalid", payload={})
    assert error.value.status_code == 422
    assert store.get_draft(saved["id"]) == saved


@pytest.mark.parametrize(
    "payload", [[], {"x": float("nan")}, {"x": float("inf")}, {1: "value"}, {"x": object()}]
)
def test_json_documents_are_bounded_and_finite(store, payload):
    with pytest.raises(StorageError) as error:
        store.create_draft("import", "Invalid", payload)
    assert error.value.status_code == 422
    assert store.list_drafts() == []
    with pytest.raises(StorageError) as error:
        store.create_draft("import", "Oversized", {"text": "x" * (1024 * 1024)})
    assert error.value.status_code == 413


def test_publication_idempotency_content_identity_and_portable_folder(store, tmp_path):
    source = draft(store)
    first = publish(store, source)
    on_disk = (store.folder / "datasets" / first["id"] / "manifest.json").read_bytes()
    assert publish(store, source) == first
    with pytest.raises(StorageError) as error:
        publish(store, source, manifest={"name": "Different"})
    assert error.value.code == "OPERATION_CONFLICT"
    second_source = draft(store, "Same inputs, different draft")
    second = publish(
        store,
        second_source,
        operation="repeat-import",
        artifacts={
            "sources/table.csv": b"De ID\nfixture-001\n",
            "tables/slides.json": b'[{"Slide_ID":"fixture-001"}]',
        },
    )
    assert second == first
    assert store.get_draft(second_source["id"])["status"] == "frozen"
    assert store.list_datasets() == [first]
    assert (store.folder / "datasets" / first["id"] / "manifest.json").read_bytes() == on_disk
    destination = tmp_path / "relocated"
    store.folder.rename(destination)
    moved = ScientificStore(destination, store.project_id)
    assert moved.get_dataset(first["id"]) == first
    assert moved.read_artifact(first["id"], "sources/table.csv") == b"De ID\nfixture-001\n"
    assert moved.get_draft(source["id"])["status"] == "frozen"


def test_parent_dataset_is_local_and_changes_create_a_child(store):
    first = publish(store)
    child = publish(
        store, operation="child", manifest={"name": "New version", "parentId": first["id"]}
    )
    assert child["parentId"] == first["id"]
    assert child["id"] != first["id"]
    assert store.get_dataset(first["id"]) == first
    with pytest.raises(StorageError) as error:
        publish(store, operation="missing-parent", manifest={"parentId": "dataset-" + "0" * 64})
    assert error.value.code == "DATASET_NOT_FOUND"


@pytest.mark.parametrize(
    "phase,published",
    [
        ("journal_committed", False),
        ("artifact_written", False),
        ("staging_complete", True),
        ("directory_published", True),
        ("database_committed", True),
    ],
)
def test_restart_recovers_only_complete_publications(store, phase, published):
    previous = publish(store, operation="previous")
    source = draft(store, "Next import")
    interrupt_at(store, phase)
    with pytest.raises(Interrupted):
        publish(store, source, operation="interrupted", manifest={"name": "Next"})
    restored = reopened(store)
    state = restored.status()
    assert restored.get_dataset(previous["id"]) == previous
    operation = next(item for item in state["operations"] if item["id"] == "interrupted")
    assert operation["status"] == ("published" if published else "interrupted")
    assert state["datasetCount"] == (2 if published else 1)
    assert restored.get_draft(source["id"])["status"] == ("frozen" if published else "editable")
    retried = publish(restored, source, operation="interrupted", manifest={"name": "Next"})
    assert restored.get_dataset(retried["id"]) == retried
    assert restored.status()["datasetCount"] == 2
    assert len(list((store.folder / "datasets").iterdir())) == 2


def test_real_process_exit_after_rename_recovers_publication(store):
    source = draft(store)
    script = """
import os, sys
from pathlib import Path
from histopilot.storage.scientific import ScientificStore
store = ScientificStore(Path(sys.argv[1]), sys.argv[2])
def checkpoint(phase):
    if phase == 'directory_published':
        os._exit(73)
store._checkpoint = checkpoint
store.publish_dataset(sys.argv[3], expected_revision=1, manifest={'name':'process crash'},
                      artifacts={'tables/slides.csv':b'Slide_ID\\nfixture-001\\n'},
                      operation_id='process-exit')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(store.folder), store.project_id, source["id"]],
        cwd=Path(__file__).resolve().parents[1],
        timeout=20,
        check=False,
    )
    assert result.returncode == 73
    restored = reopened(store)
    assert restored.status()["operations"][0]["status"] == "published"
    dataset = restored.list_datasets()[0]
    assert restored.read_artifact(dataset["id"], "tables/slides.csv") == b"Slide_ID\nfixture-001\n"
    assert restored.get_draft(source["id"])["status"] == "frozen"


def test_recovery_sync_failure_keeps_complete_output_unpublished_and_retryable(store, monkeypatch):
    source = draft(store)
    interrupt_at(store, "directory_published")
    with pytest.raises(Interrupted):
        publish(store, source)
    original = scientific.fsync_directory
    syncs = []

    def fail_sync(path):
        if path == store.folder / "datasets":
            raise StorageError("Injected sync failure", "STORAGE_SYNC_FAILED")
        original(path)

    with monkeypatch.context() as patch:
        patch.setattr(scientific, "fsync_directory", fail_sync)
        with pytest.raises(StorageError) as error:
            reopened(store).initialize()
        assert error.value.code == "STORAGE_SYNC_FAILED"
    with database(store) as connection:
        assert connection.execute("SELECT COUNT(*) FROM datasets").fetchone()[0] == 0
        assert connection.execute("SELECT status FROM publications").fetchone()[0] == "preparing"

    def record_sync(path):
        syncs.append(path)
        original(path)

    monkeypatch.setattr(scientific, "fsync_directory", record_sync)
    restored = reopened(store)
    restored.initialize()
    assert store.folder / "datasets" in syncs
    assert store.folder / ".staging" in syncs
    assert restored.get_draft(source["id"])["status"] == "frozen"


def test_corrupt_complete_stage_is_quarantined_without_freezing_draft(store):
    source = draft(store)
    interrupt_at(store, "staging_complete")
    with pytest.raises(Interrupted):
        publish(store, source)
    stage = next((store.folder / ".staging").iterdir())
    (stage / "sources" / "table.csv").write_bytes(b"changed bytes")
    restored = reopened(store)
    assert restored.status()["operations"][0]["status"] == "interrupted"
    assert restored.list_datasets() == []
    assert restored.get_draft(source["id"]) == source
    assert any(
        path.name.startswith("interrupted-") for path in (store.folder / ".staging").iterdir()
    )


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        ".",
        "/absolute",
        "a/../b",
        "a//b",
        "./a",
        "a\\b",
        "C:drive",
        "manifest.json",
        "manifest.json/nested",
        ".secret",
        "nested/.secret",
    ],
)
def test_artifact_paths_rejected_before_journaling(store, name):
    with pytest.raises(StorageError) as error:
        publish(store, artifacts={name: b"bad"})
    assert error.value.status_code == 422
    assert store.status()["operations"] == []
    assert not list((store.folder / "datasets").iterdir())
    assert not list((store.folder / ".staging").iterdir())


def test_artifact_prefix_collisions_and_limits_rejected_before_journaling(store, monkeypatch):
    with pytest.raises(StorageError):
        publish(store, artifacts={"tables": b"file", "tables/slides.csv": b"nested"})
    monkeypatch.setattr(scientific, "MAX_ARTIFACT_BYTES", 3)
    with pytest.raises(StorageError) as error:
        publish(store, artifacts={"file.bin": b"four"})
    assert error.value.status_code == 413
    assert store.status()["operations"] == []


@pytest.mark.parametrize(
    "change", ["same-length", "extra", "missing", "manifest", "symlink", "hardlink"]
)
def test_tampered_or_undeclared_artifacts_never_exposed(store, tmp_path, change):
    dataset = publish(store)
    folder = store.folder / "datasets" / dataset["id"]
    target = folder / "tables/slides.json"
    if change == "same-length":
        target.write_bytes(b"X" * target.stat().st_size)
    elif change == "extra":
        (folder / "undeclared.json").write_bytes(b"{}")
    elif change == "missing":
        target.unlink()
    elif change == "manifest":
        (folder / "manifest.json").write_text("{}")
    else:
        original = tmp_path / "external-original"
        original.write_bytes(target.read_bytes())
        target.unlink()
        if change == "symlink":
            target.symlink_to(original)
        else:
            os.link(original, target)
    for action in (
        lambda: store.get_dataset(dataset["id"]),
        store.list_datasets,
        lambda: store.read_artifact(dataset["id"], "tables/slides.json"),
    ):
        with pytest.raises(StorageError):
            action()


def test_unknown_final_is_not_overwritten_or_registered(store, tmp_path):
    template = publish(store)
    folder = tmp_path / "other"
    folder.mkdir()
    other = ScientificStore(folder, store.project_id)
    other.initialize()
    unknown = folder / "datasets" / template["id"]
    unknown.mkdir()
    (unknown / "sentinel").write_bytes(b"unrelated")
    with pytest.raises(StorageError) as error:
        publish(other)
    assert error.value.code == "PUBLICATION_CONFLICT"
    assert (unknown / "sentinel").read_bytes() == b"unrelated"
    assert other.list_datasets() == []
    assert other.status()["operations"] == []


@pytest.mark.parametrize("suffix", ["", "-wal", "-shm", "-journal"])
@pytest.mark.parametrize("alias", ["symlink", "hardlink"])
def test_sqlite_database_and_sidecars_reject_aliases(tmp_path, suffix, alias):
    folder = tmp_path / "experiment"
    folder.mkdir()
    outside = tmp_path / "sentinel"
    outside.write_bytes(b"do not modify")
    target = folder / (DATABASE_FILE + suffix)
    if alias == "symlink":
        target.symlink_to(outside)
    else:
        os.link(outside, target)
    with pytest.raises(StorageError) as error:
        ScientificStore(folder, "project-test").initialize()
    assert error.value.code == "STORAGE_UNSAFE_PATH"
    assert outside.read_bytes() == b"do not modify"


@pytest.mark.parametrize("replacement", [None, b"", b"corrupt"])
def test_missing_or_truncated_database_never_resets_scientific_state(store, replacement):
    frozen = publish(store)
    manifest = store.folder / "datasets" / frozen["id"] / "manifest.json"
    original = manifest.read_bytes()
    store.path.unlink()
    if replacement is not None:
        store.path.write_bytes(replacement)
    with pytest.raises(StorageError):
        reopened(store).initialize()
    assert manifest.read_bytes() == original
    if replacement is None:
        assert not store.path.exists()
    else:
        assert store.path.read_bytes() == replacement


def test_only_empty_version_zero_databases_can_initialize(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / DATABASE_FILE).touch()
    assert ScientificStore(empty, "project-empty").status()["schemaVersion"] == 2
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    connection = sqlite3.connect(foreign / DATABASE_FILE)
    connection.execute("CREATE TABLE original (value TEXT)")
    connection.execute("INSERT INTO original VALUES ('sentinel')")
    connection.commit()
    connection.close()
    with pytest.raises(StorageError):
        ScientificStore(foreign, "project-foreign").initialize()
    connection = sqlite3.connect(foreign / DATABASE_FILE)
    try:
        assert connection.execute("SELECT value FROM original").fetchone()[0] == "sentinel"
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
    finally:
        connection.close()


def test_initial_schema_failure_rolls_back_and_can_retry(tmp_path, monkeypatch):
    folder = tmp_path / "initialize"
    folder.mkdir()
    legacy = folder / "histopilot-project.json"
    legacy.write_bytes(b"legacy setup sentinel")
    store = ScientificStore(folder, "project-test")
    with monkeypatch.context() as patch:
        patch.setattr(scientific, "_SCHEMA", (*scientific._SCHEMA[:2], "INVALID SQL"))
        with pytest.raises(StorageError):
            store.initialize()
    with database(store) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
        assert connection.execute("PRAGMA application_id").fetchone()[0] == 0
        assert (
            connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == []
        )
    assert store.status()["schemaVersion"] == 2
    assert legacy.read_bytes() == b"legacy setup sentinel"


def test_future_schema_wrong_owner_and_missing_constraints_fail_closed(store):
    with pytest.raises(StorageError) as error:
        ScientificStore(store.folder, "another-project").initialize()
    assert error.value.code == "PROJECT_ID_MISMATCH"
    with database(store) as connection:
        connection.execute("PRAGMA user_version=99")
    with pytest.raises(StorageError) as error:
        reopened(store).initialize()
    assert error.value.code == "STORAGE_SCHEMA_UNSUPPORTED"
    with database(store) as connection:
        connection.execute("PRAGMA user_version=2")
        connection.execute("DROP TABLE drafts")
        connection.execute(
            "CREATE TABLE drafts (id TEXT, kind TEXT, name TEXT, payload TEXT, revision TEXT, status TEXT, created_at TEXT, updated_at TEXT)"
        )
    with pytest.raises(StorageError) as error:
        reopened(store).initialize()
    assert error.value.code == "STORAGE_SCHEMA_UNSUPPORTED"


def test_corrupt_dataset_document_cannot_escape_as_unhandled_error(store):
    dataset = publish(store)
    with database(store) as connection:
        malformed = {**dataset, "manifest": []}
        connection.execute(
            "UPDATE datasets SET document=? WHERE id=?", (json.dumps(malformed), dataset["id"])
        )
    with pytest.raises(StorageError) as error:
        store.get_dataset(dataset["id"])
    assert error.value.code == "STORAGE_CORRUPT"
