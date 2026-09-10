"""Configuration snapshots commit atomically alongside their scientific drafts."""

import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from histopilot.storage import scientific
from histopilot.storage.project_lock import StorageError, writer_lock
from histopilot.storage.scientific import ScientificStore


@pytest.fixture
def store(tmp_path):
    current = ScientificStore(tmp_path, "project-configurations")
    current.initialize()
    return current


def dataset(store):
    draft = store.create_draft("import", "Bladder", {})
    return store.publish_dataset(
        draft["id"],
        expected_revision=1,
        manifest={"kind": "dataset"},
        artifacts={"records.json": b"[]"},
        operation_id="dataset",
    )


def test_v1_migration_preserves_frozen_files_and_draft_bytes(store):
    frozen = dataset(store)
    editable = store.create_draft("experiment", "draft", {"seed": 42})
    original = (store.folder / "datasets" / frozen["id"] / "manifest.json").read_bytes()
    with sqlite3.connect(store.path) as connection:
        connection.execute("DROP TABLE configuration_publications")
        connection.execute("DROP TABLE configurations")
        connection.execute("UPDATE metadata SET value='1' WHERE key='schema_version'")
        connection.execute("PRAGMA user_version=1")
    reopened = ScientificStore(store.folder, store.project_id)
    assert reopened.status()["schemaVersion"] == 2
    assert reopened.get_draft(editable["id"]) == editable
    assert reopened.get_dataset(frozen["id"]) == frozen
    assert (store.folder / "datasets" / frozen["id"] / "manifest.json").read_bytes() == original


def test_atomic_migration_rolls_back_on_failure(store):
    with sqlite3.connect(store.path) as connection:
        connection.execute("DROP TABLE configuration_publications")
        connection.execute("DROP TABLE configurations")
        connection.execute("UPDATE metadata SET value='1' WHERE key='schema_version'")
        connection.execute("PRAGMA user_version=1")
    with patch.object(scientific, "_SCHEMA", (*scientific._SCHEMA[:-1], "INVALID SQL")):
        with pytest.raises(StorageError):
            store.initialize()
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE name='configurations'"
            ).fetchone()
            is None
        )
    store.initialize()


def test_invalid_v1_foreign_keys_are_rejected_before_any_schema_migration(store):
    with sqlite3.connect(store.path) as connection:
        connection.execute("DROP TABLE configuration_publications")
        connection.execute("DROP TABLE configurations")
        connection.execute("UPDATE metadata SET value='1' WHERE key='schema_version'")
        connection.execute("PRAGMA user_version=1")
        connection.execute(
            "INSERT INTO publications VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                "invalid",
                "hash",
                "missing-draft",
                1,
                "dataset-missing",
                "operation-invalid",
                "{}",
                "preparing",
                None,
                "now",
                "now",
            ),
        )
    with pytest.raises(StorageError) as error:
        store.initialize()
    assert error.value.code == "STORAGE_CORRUPT"
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        assert (
            connection.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()[
                0
            ]
            == "1"
        )
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE name='configurations'"
            ).fetchone()
            is None
        )


def test_snapshot_retry_revision_and_configuration_checksum(store):
    frozen = dataset(store)
    draft = store.create_draft("experiment", "Protocol", {"seed": 42})
    manifest = {
        "kind": "protocol",
        "datasetId": frozen["id"],
        "memberships": [{"patientId": "001"}],
    }
    saved = store.publish_configuration(
        draft["id"], expected_revision=1, manifest=manifest, operation_id="freeze"
    )
    assert store.get_draft(draft["id"])["status"] == "frozen"
    assert store.get_draft(draft["id"])["revision"] == 2
    assert (
        store.publish_configuration(
            draft["id"], expected_revision=1, manifest=manifest, operation_id="freeze"
        )
        == saved
    )
    assert store.list_configurations("protocol") == [saved]
    assert store.list_configurations("feature") == []
    with pytest.raises(StorageError, match="different request"):
        store.publish_configuration(
            draft["id"],
            expected_revision=1,
            manifest={**manifest, "changed": True},
            operation_id="freeze",
        )
    with pytest.raises(StorageError) as error:
        store.publish_configuration(
            draft["id"], expected_revision=1, manifest=manifest, operation_id="another"
        )
    assert error.value.code == "REVISION_CONFLICT"
    with sqlite3.connect(store.path) as connection:
        corrupt = {**saved, "manifest": {**manifest, "changed": True}}
        connection.execute(
            "UPDATE configurations SET document=? WHERE id=?", (json.dumps(corrupt), saved["id"])
        )
    with pytest.raises(StorageError, match="checksum"):
        store.get_configuration(saved["id"])


@pytest.mark.parametrize(
    "phase,committed", [("configuration_before_commit", False), ("configuration_committed", True)]
)
def test_interrupted_configuration_is_all_or_nothing(store, phase, committed):
    frozen = dataset(store)
    draft = store.create_draft("experiment", "Protocol", {})

    def interrupt(current):
        if current == phase:
            raise KeyboardInterrupt

    store._checkpoint = interrupt
    with pytest.raises(KeyboardInterrupt):
        store.publish_configuration(
            draft["id"],
            expected_revision=1,
            manifest={"kind": "protocol", "datasetId": frozen["id"]},
            operation_id="interrupted",
        )
    reopened = ScientificStore(store.folder, store.project_id)
    assert len(reopened.list_configurations()) == int(committed)
    assert reopened.get_draft(draft["id"])["status"] == ("frozen" if committed else "editable")


def test_concurrent_reads_wait_for_short_initialization_instead_of_spurious_busy_errors(store):
    draft = store.create_draft("experiment", "Read concurrently", {})
    with ThreadPoolExecutor(max_workers=8) as executor:
        with writer_lock(store.folder):
            futures = [executor.submit(store.get_draft, draft["id"]) for _ in range(8)]
            time.sleep(0.05)
        assert [future.result(timeout=5) for future in futures] == [draft] * 8
