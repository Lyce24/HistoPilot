"""Cleanup is reversible metadata, with concurrency and scientific write barriers."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from histopilot.storage import lifecycle
from histopilot.storage.lifecycle import LifecycleStore, lifecycle_guard
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import SCHEMA_VERSION, ScientificStore


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


@pytest.fixture
def stores(tmp_path):
    folder = tmp_path / "experiment"
    folder.mkdir()
    scientific = ScientificStore(folder, "project-cleanup")
    scientific.initialize()
    return scientific, LifecycleStore(folder, scientific.project_id)


def dataset(store, *, name="Original", operation="dataset"):
    draft = store.create_draft("import", name, {})
    return store.publish_dataset(
        draft["id"],
        expected_revision=1,
        manifest={"name": name},
        artifacts={"slides.json": b'[{"slideId":"slide-1"}]'},
        operation_id=operation,
    )


def change(store, refs, state, *, operation=None, action=None):
    operation = operation or f"operation-{store.read()['revision']}"
    return store.apply(
        {ref: state for ref in refs},
        operation,
        digest(operation),
        store.read()["revision"],
        action=action,
    )


def test_missing_sidecar_is_active_without_scientific_migration(stores):
    scientific, metadata = stores
    assert metadata.read()["revision"] == 0
    assert metadata.read()["records"] == {}
    assert not metadata.path.exists()
    metadata.assert_usable(["dataset:unrecorded"])
    assert scientific.status()["schemaVersion"] == SCHEMA_VERSION == 4


def test_bulk_archive_restore_and_operation_replay_preserve_newer_decisions(stores):
    _, metadata = stores
    refs = ["dataset:one", "configuration:two"]
    first = change(metadata, refs, "archived", operation="archive", action="archive")
    assert first["revision"] == 1
    assert first["audit"][0]["changes"][refs[0]] == {"from": "active", "to": "archived"}
    restored = change(metadata, refs, "active", operation="restore", action="restore")
    replay = metadata.apply(
        dict.fromkeys(refs, "archived"), "archive", digest("archive"), 0, action="archive"
    )
    assert replay == restored
    assert replay["revision"] == 2
    assert LifecycleStore(metadata.folder, metadata.project_id).read() == restored
    assert replay["operations"]["archive"]["revision"] == 1
    assert replay["operations"]["archive"]["action"] == "archive"


def test_cancellation_audit_has_no_visibility_changes(stores):
    _, metadata = stores
    result = metadata.apply({}, "cancel", digest("cancel"), 0, action="cancel")
    assert result["records"] == {}
    assert result["audit"][0]["action"] == "cancel"
    assert metadata.read() == result
    assert metadata.apply({}, "cancel", digest("cancel"), 0, action="cancel") == result


def test_operation_conflict_and_stale_review_do_not_write(stores):
    _, metadata = stores
    saved = change(metadata, ["dataset:one"], "archived", operation="first")
    before = metadata.path.read_bytes()
    for changes, operation, expected, code in [
        ({"dataset:one": "trashed"}, "first", 0, "OPERATION_CONFLICT"),
        ({"dataset:one": "active"}, "other", 0, "LIFECYCLE_REVISION_CONFLICT"),
    ]:
        with pytest.raises(StorageError) as raised:
            metadata.apply(changes, operation, digest(operation), expected)
        assert raised.value.code == code
        assert metadata.path.read_bytes() == before
    assert metadata.read() == saved


@pytest.mark.parametrize(
    "key",
    [
        "unknown:one",
        "dataset:../one",
        "dataset:one/two",
        "dataset:one:two",
        "dataset:",
        "dataset:a\\b",
        "dataset:\x00",
        "project: ",
    ],
)
def test_invalid_record_keys_do_not_create_sidecar(stores, key):
    _, metadata = stores
    with pytest.raises(StorageError) as raised:
        metadata.apply({key: "trashed"}, "bad", digest("bad"), 0)
    assert raised.value.status_code == 422
    assert not metadata.path.exists()


def test_atomic_replace_failure_keeps_previous_state_and_cleans_only_temporary_file(
    stores, monkeypatch
):
    _, metadata = stores
    change(metadata, ["dataset:one"], "archived")
    before = metadata.path.read_bytes()

    def fail_replace(*args):
        raise OSError("injected replacement failure")

    monkeypatch.setattr(lifecycle.os, "replace", fail_replace)
    with pytest.raises(StorageError) as raised:
        change(metadata, ["dataset:one"], "trashed")
    assert raised.value.code == "STORAGE_WRITE_FAILED"
    assert metadata.path.read_bytes() == before
    assert not list(metadata.folder.glob(".histopilot-lifecycle-*.tmp"))


def test_failure_after_rename_is_recoverable_by_exact_operation_retry(stores, monkeypatch):
    _, metadata = stores
    original = lifecycle.fsync_directory
    count = 0

    def fail_final_sync(folder):
        nonlocal count
        count += 1
        if count == 2:
            raise StorageError("injected final sync failure", "STORAGE_SYNC_FAILED")
        return original(folder)

    monkeypatch.setattr(lifecycle, "fsync_directory", fail_final_sync)
    with pytest.raises(StorageError):
        metadata.apply({"dataset:one": "archived"}, "first", digest("first"), 0)
    monkeypatch.setattr(lifecycle, "fsync_directory", original)
    assert metadata.read()["revision"] == 1
    result = metadata.apply({"dataset:one": "archived"}, "first", digest("first"), 0)
    assert result["revision"] == 1
    assert len(result["audit"]) == 1


@pytest.mark.parametrize("alias", ["symlink", "hardlink", "directory"])
def test_metadata_aliases_are_rejected_without_touching_target(stores, tmp_path, alias):
    _, metadata = stores
    target = tmp_path / "unmanaged.json"
    target.write_text("external sentinel")
    if alias == "symlink":
        metadata.path.symlink_to(target)
    elif alias == "hardlink":
        os.link(target, metadata.path)
    else:
        metadata.path.mkdir()
    with pytest.raises(StorageError) as raised:
        metadata.read()
    assert raised.value.code == "STORAGE_UNSAFE_PATH"
    with pytest.raises(StorageError):
        change(metadata, ["dataset:one"], "trashed")
    assert target.read_text() == "external sentinel"


def test_corrupt_state_audit_disagreement_blocks_reads_and_writes(stores):
    _, metadata = stores
    result = change(metadata, ["dataset:one"], "archived")
    result["records"]["dataset:one"]["state"] = "active"
    metadata.path.write_text(json.dumps(result))
    with pytest.raises(StorageError) as raised:
        metadata.read()
    assert raised.value.code == "LIFECYCLE_CORRUPT"
    with pytest.raises(StorageError):
        metadata.apply({"dataset:two": "trashed"}, "other", digest("other"), 1)


def test_guard_reentrancy_and_cross_process_mutation_serialization(stores):
    scientific, metadata = stores
    program = """
import sys
from pathlib import Path
from histopilot.storage.lifecycle import lifecycle_guard
from histopilot.storage.project_lock import StorageError
try:
    with lifecycle_guard(Path(sys.argv[1]), timeout=0):
        raise AssertionError('Concurrent lifecycle writer acquired the guard')
except StorageError as error:
    assert error.code == 'PROJECT_BUSY', error.code
print('blocked')
"""
    with lifecycle_guard(metadata.folder):
        with lifecycle_guard(metadata.folder, timeout=0):
            change(metadata, ["dataset:one"], "archived")
            scientific.create_draft("import", "Nested mutation", {})
        result = subprocess.run(
            [sys.executable, "-c", program, str(metadata.folder)],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parents[1],
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "blocked" in result.stdout
    with lifecycle_guard(metadata.folder, timeout=0):
        pass


def test_archive_hides_pickers_but_retains_provenance_and_scientific_bytes(stores):
    scientific, metadata = stores
    saved = dataset(scientific)
    configuration = scientific.publish_configuration(
        manifest={"kind": "protocol", "datasetId": saved["id"]},
        operation_id="configuration",
    )
    draft = scientific.create_draft("experiment", "Draft", {"datasetId": saved["id"]})
    original = (scientific.folder / "datasets" / saved["id"] / "manifest.json").read_bytes()
    refs = [
        f"dataset:{saved['id']}",
        f"configuration:{configuration['id']}",
        f"draft:{draft['id']}",
    ]
    change(metadata, refs, "archived")
    assert scientific.list_datasets() == []
    assert scientific.list_configurations() == []
    assert draft not in scientific.list_drafts()
    assert scientific.get_dataset(saved["id"]) == saved
    assert scientific.get_configuration(configuration["id"]) == configuration
    assert scientific.get_draft(draft["id"]) == draft
    assert scientific.list_datasets(include_inactive=True) == [saved]
    assert scientific.list_configurations(include_inactive=True) == [configuration]
    assert scientific.read_artifact(saved["id"], "slides.json") == b'[{"slideId":"slide-1"}]'
    assert (scientific.folder / "datasets" / saved["id"] / "manifest.json").read_bytes() == original
    scientific.update_draft(
        draft["id"], expected_revision=1, name="Archived but usable", payload=draft["payload"]
    )
    change(metadata, refs, "active")
    assert scientific.list_datasets() == [saved]


def test_trash_blocks_getters_and_mutations_but_cleanup_can_read_everything(stores):
    scientific, metadata = stores
    saved = dataset(scientific)
    configuration = scientific.publish_configuration(
        manifest={"kind": "protocol", "datasetId": saved["id"]},
        operation_id="configuration",
    )
    draft = scientific.create_draft("experiment", "Draft", {"datasetId": saved["id"]})
    change(
        metadata,
        [f"dataset:{saved['id']}", f"configuration:{configuration['id']}", f"draft:{draft['id']}"],
        "trashed",
    )
    for getter, identity, document in [
        (scientific.get_dataset, saved["id"], saved),
        (scientific.get_configuration, configuration["id"], configuration),
        (scientific.get_draft, draft["id"], draft),
    ]:
        with pytest.raises(StorageError) as raised:
            getter(identity)
        assert raised.value.code == "RECORD_TRASHED"
        assert getter(identity, include_inactive=True) == document
    with pytest.raises(StorageError, match="Trash"):
        scientific.read_artifact(saved["id"], "slides.json")
    assert scientific.read_artifact(saved["id"], "slides.json", include_inactive=True)
    with pytest.raises(StorageError, match="Trash"):
        scientific.update_draft(draft["id"], expected_revision=1, name="Changed", payload={})
    with pytest.raises(StorageError, match="Trash"):
        scientific.set_version_label("dataset", saved["id"], tag="Changed", expected_revision=0)
    assert scientific.get_draft(draft["id"], include_inactive=True)["revision"] == 1


def test_recursive_trashed_references_block_new_drafts_and_configuration_publication(stores):
    scientific, metadata = stores
    saved = dataset(scientific)
    protocol = scientific.publish_configuration(
        manifest={"kind": "protocol", "datasetId": saved["id"]},
        operation_id="protocol",
    )
    draft = scientific.create_draft("experiment", "Editable", {})
    change(metadata, [f"configuration:{protocol['id']}"], "trashed")
    nested = {"inputs": [{"reference": protocol["id"]}]}
    for mutate in [
        lambda: scientific.create_draft("experiment", "Bad reference", nested),
        lambda: scientific.update_draft(
            draft["id"], expected_revision=1, name="Bad reference", payload=nested
        ),
        lambda: scientific.publish_configuration(
            manifest={"kind": "feature", "datasetId": saved["id"], **nested},
            operation_id="bad-feature",
        ),
    ]:
        with pytest.raises(StorageError) as raised:
            mutate()
        assert raised.value.code == "RECORD_TRASHED"
    assert scientific.get_draft(draft["id"])["revision"] == 1
    assert scientific.list_configurations(include_inactive=True) == [protocol]
    metadata.assert_document_usable({"sourcePath": f"/mnt/{protocol['id']}.h5"})
    with pytest.raises(StorageError, match="Trash"):
        metadata.assert_document_usable({"byConfiguration": {protocol["id"]: True}})


def test_identical_publication_does_not_silently_revive_trashed_content(stores):
    scientific, metadata = stores
    saved = dataset(scientific)
    change(metadata, [f"dataset:{saved['id']}"], "trashed")
    draft = scientific.create_draft("import", "Reimport", {})
    with pytest.raises(StorageError) as raised:
        scientific.publish_dataset(
            draft["id"],
            expected_revision=1,
            manifest=saved["manifest"],
            artifacts={"slides.json": b'[{"slideId":"slide-1"}]'},
            operation_id="duplicate",
        )
    assert raised.value.code == "RECORD_TRASHED"
    assert scientific.get_draft(draft["id"])["status"] == "editable"
    assert scientific.list_datasets() == []


def test_project_trash_hides_lists_and_blocks_writes_without_hiding_cleanup_catalog(stores):
    scientific, metadata = stores
    saved = dataset(scientific)
    drafts = scientific.list_drafts()
    change(metadata, [f"project:{scientific.project_id}"], "trashed")
    assert scientific.list_datasets() == []
    assert scientific.list_drafts() == []
    assert scientific.list_configurations() == []
    assert scientific.list_datasets(include_inactive=True) == [saved]
    assert scientific.list_drafts(include_inactive=True) == drafts
    assert scientific.get_dataset(saved["id"], include_inactive=True) == saved
    with pytest.raises(StorageError, match="Trash"):
        scientific.get_dataset(saved["id"])
    with pytest.raises(StorageError, match="Trash"):
        scientific.create_draft("import", "Blocked", {})
    assert scientific.status()["schemaVersion"] == 4


def test_cancellation_targets_are_recorded_and_part_of_operation_identity(stores):
    _, metadata = stores
    targets = ["packing:one"]
    saved = metadata.apply({}, "cancel", digest("cancel"), 0, action="cancel", targets=targets)
    assert saved["audit"][0]["targets"] == targets
    assert saved["operations"]["cancel"]["targets"] == targets
    assert metadata.read() == saved
    with pytest.raises(StorageError) as raised:
        metadata.apply({}, "cancel", digest("cancel"), 0, action="cancel", targets=["packing:two"])
    assert raised.value.code == "OPERATION_CONFLICT"


def receipt(metadata, job_id, pack_id):
    folder = metadata.folder / "packing" / job_id
    folder.mkdir(parents=True)
    (folder / "result.json").write_text(
        json.dumps(
            {
                "jobId": job_id,
                "state": "succeeded",
                "artifact": {"id": pack_id, "jobId": job_id},
            }
        )
    )


def test_raw_pack_artifact_reference_cannot_bypass_trashed_receipt(stores):
    scientific, metadata = stores
    saved = dataset(scientific)
    job_id, pack_id = "packing-" + "a" * 32, "pack-" + "b" * 64
    receipt(metadata, job_id, pack_id)
    change(metadata, [f"packing:{job_id}"], "trashed")
    with pytest.raises(StorageError) as raised:
        scientific.publish_configuration(
            manifest={
                "kind": "feature-bundle",
                "datasetId": saved["id"],
                "packArtifactId": pack_id,
            },
            operation_id="bundle",
        )
    assert raised.value.code == "RECORD_TRASHED"
    assert scientific.list_configurations(include_inactive=True) == []
    metadata.assert_document_usable({"materializationId": "pack-" + "c" * 64})
    metadata.assert_document_usable({"sourcePath": f"/mnt/{pack_id}"})
    change(metadata, [f"packing:{job_id}"], "archived")
    metadata.assert_document_usable({"packArtifactId": pack_id})


def test_an_independent_live_receipt_can_authorize_same_pack_alias(stores):
    _, metadata = stores
    first, second, pack_id = "packing-" + "a" * 32, "packing-" + "b" * 32, "pack-" + "c" * 64
    receipt(metadata, first, pack_id)
    receipt(metadata, second, pack_id)
    change(metadata, [f"packing:{first}"], "trashed")
    metadata.assert_document_usable({"packArtifactId": pack_id})
    change(metadata, [f"packing:{second}"], "archived")
    metadata.assert_document_usable({"packArtifactId": pack_id})
    with pytest.raises(StorageError, match="Trash"):
        metadata.assert_document_usable({"jobId": first, "packArtifactId": pack_id})
    change(metadata, [f"packing:{second}"], "trashed")
    with pytest.raises(StorageError, match="Trash"):
        metadata.assert_document_usable({"packArtifactId": pack_id})


def test_two_concurrent_reviews_cannot_overwrite_each_other(stores):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    _, metadata = stores
    barrier = Barrier(2, timeout=3)

    def commit(index):
        current = LifecycleStore(metadata.folder, metadata.project_id)
        barrier.wait()
        try:
            current.apply({f"dataset:{index}": "trashed"}, str(index), digest(str(index)), 0)
            return "saved"
        except StorageError as error:
            return error.code

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(commit, [1, 2]))
    assert sorted(results) == ["LIFECYCLE_REVISION_CONFLICT", "saved"]
    assert metadata.read()["revision"] == 1
    assert len(metadata.read()["records"]) == 1
