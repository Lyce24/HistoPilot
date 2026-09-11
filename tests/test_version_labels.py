"""Personal version labels preserve scientific identity and reject lost or ambiguous edits."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from histopilot.application.protocols import ProtocolService
from histopilot.schemas.version_labels import SetVersionLabelRequest
from histopilot.storage import scientific
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import ScientificStore


@pytest.fixture
def versions(tmp_path):
    folder = tmp_path / "project"
    folder.mkdir()
    store = ScientificStore(folder, "project-version-labels")
    datasets = []
    drafts = []
    for name in ("Source", "Reviewed"):
        draft = store.create_draft("import", name, {})
        drafts.append(draft)
        datasets.append(
            store.publish_dataset(
                draft["id"],
                expected_revision=1,
                manifest={"kind": "dataset", "name": name},
                artifacts={"records.json": b'[{"slideId":"A"}]'},
                operation_id=name,
            )
        )
    configs = {
        kind: store.publish_configuration(
            manifest={"kind": kind, "datasetId": datasets[0]["id"]},
            operation_id=kind,
        )
        for kind in ("feature", "protocol")
    }
    return store, datasets, configs, drafts


def label(store, resource_type, resource, tag, *, note="", revision=0):
    return store.set_version_label(
        resource_type, resource["id"], tag=tag, note=note, expected_revision=revision
    )


def test_unlabelled_legacy_versions_keep_their_exact_public_shape(versions):
    store, datasets, configs, drafts = versions
    assert store.get_version_label("dataset", datasets[0]["id"]) is None
    assert store.get_version_label("configuration", configs["feature"]["id"]) is None
    assert store.get_dataset(datasets[0]["id"]) == datasets[0]
    assert store.get_configuration(configs["feature"]["id"]) == configs["feature"]


def test_labels_survive_reopen_and_relocation_without_changing_scientific_bytes(versions, tmp_path):
    store, datasets, configs, drafts = versions
    original_files = {
        path.relative_to(store.folder): path.read_bytes()
        for path in (store.folder / "datasets").rglob("*")
        if path.is_file()
    }
    with sqlite3.connect(store.path) as connection:
        original_documents = {
            table: connection.execute(f"SELECT document FROM {table} ORDER BY id").fetchall()
            for table in ("datasets", "configurations", "publications")
        }
    dataset_label = label(
        store,
        "dataset",
        datasets[0],
        "  BLCA v1  ",
        note="Reviewed cases.\nRetained original slide IDs.",
    )
    feature_label = label(
        store, "configuration", configs["feature"], "UNI v1 • 20×", note="Baseline embeddings"
    )
    assert dataset_label["tag"] == "BLCA v1"
    assert dataset_label["revision"] == 1
    assert store.get_dataset(datasets[0]["id"]) == {**datasets[0], "versionLabel": dataset_label}
    assert store.get_configuration(configs["feature"]["id"]) == {
        **configs["feature"],
        "versionLabel": feature_label,
    }
    assert {item["id"]: item for item in store.list_datasets()}[datasets[0]["id"]][
        "versionLabel"
    ] == dataset_label
    assert store.list_configurations("feature")[0]["versionLabel"] == feature_label
    with sqlite3.connect(store.path) as connection:
        for table, original in original_documents.items():
            assert (
                connection.execute(f"SELECT document FROM {table} ORDER BY id").fetchall()
                == original
            )
    for relative, original in original_files.items():
        assert (store.folder / relative).read_bytes() == original
    moved = tmp_path / "moved-project"
    store.folder.rename(moved)
    reopened = ScientificStore(moved, store.project_id)
    assert reopened.get_version_label("dataset", datasets[0]["id"]) == dataset_label
    assert reopened.get_configuration(configs["feature"]["id"])["versionLabel"] == feature_label
    assert reopened.read_artifact(datasets[0]["id"], "records.json") == b'[{"slideId":"A"}]'


def test_publication_retries_return_current_labels_without_changing_publications(versions):
    store, datasets, configs, drafts = versions
    saved = label(store, "dataset", datasets[0], "Review 1")
    current = label(store, "dataset", datasets[0], "Reviewed", revision=saved["revision"])
    repeated = store.publish_dataset(
        drafts[0]["id"],
        expected_revision=1,
        manifest=datasets[0]["manifest"],
        artifacts={"records.json": b'[{"slideId":"A"}]'},
        operation_id="Source",
    )
    assert repeated == {**datasets[0], "versionLabel": current}
    second_draft = store.create_draft("import", "Identical source", {})
    duplicate = store.publish_dataset(
        second_draft["id"],
        expected_revision=1,
        manifest=datasets[0]["manifest"],
        artifacts={"records.json": b'[{"slideId":"A"}]'},
        operation_id="Identical source",
    )
    assert duplicate == repeated
    feature_label = label(store, "configuration", configs["feature"], "Embeddings v1")
    for operation in ("feature", "same-features"):
        assert store.publish_configuration(
            manifest=configs["feature"]["manifest"], operation_id=operation
        ) == {
            **configs["feature"],
            "versionLabel": feature_label,
        }


def test_protocol_preview_freeze_and_replay_ignore_changes_to_personal_labels(tmp_path):
    folder = tmp_path / "protocol-project"
    folder.mkdir()
    store = ScientificStore(folder, "project-labelled-protocol")
    rows = [
        {
            "slideId": f"s{patient}",
            "patientId": f"p{patient}",
            "slidePath": None,
            "attributes": {"label": str(patient % 2)},
        }
        for patient in range(12)
    ]
    source = store.create_draft("import", "Source", {})
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
        operation_id="source",
    )
    features = store.publish_configuration(
        manifest={
            "kind": "feature",
            "datasetId": dataset["id"],
            "files": [{"slideId": row["slideId"]} for row in rows],
        },
        operation_id="features",
    )
    draft = store.create_draft(
        "experiment",
        "Protocol",
        {
            "type": "analysis-protocol",
            "spec": {
                "datasetId": dataset["id"],
                "featureSetId": features["id"],
                "target": {
                    "field": "label",
                    "task": "binary_classification",
                    "unit": "patient",
                    "classes": ["low", "high"],
                    "labels": {"0": "low", "1": "high"},
                    "positiveClass": "high",
                },
                "split": {"mode": "kfold", "folds": 3, "seeds": [42]},
            },
        },
    )
    service = ProtocolService(store)
    preview = service.preview(draft["id"], 1)
    assert preview["canFreeze"], preview["findings"]
    label(store, "dataset", dataset, "Reviewed source", note="Patient labels reviewed")
    label(store, "configuration", features, "Baseline embeddings")
    assert service.preview(draft["id"], 1) == preview

    frozen = service.freeze(draft["id"], 1, preview["previewHash"], "freeze-protocol")
    assert frozen["manifest"]["previewHash"] == preview["previewHash"]
    assert frozen["manifest"]["memberships"] == preview["memberships"]
    protocol_label = label(store, "configuration", frozen, "Cohort v1", note="Three folds")
    label(store, "dataset", dataset, "Renamed source", revision=1)
    label(store, "configuration", features, "Renamed embeddings", revision=1)
    assert service.freeze(draft["id"], 1, preview["previewHash"], "freeze-protocol") == {
        **frozen,
        "versionLabel": protocol_label,
    }


@pytest.mark.parametrize(
    "first,second", [("Baseline", " baseline "), ("Café", "Cafe\u0301"), ("Straße", "STRASSE")]
)
def test_tags_are_unicode_normalized_and_case_insensitively_unique_per_kind(
    versions, first, second
):
    store, datasets, configs, drafts = versions
    label(store, "dataset", datasets[0], first)
    with pytest.raises(StorageError) as error:
        label(store, "dataset", datasets[1], second)
    assert error.value.code == "VERSION_TAG_CONFLICT"
    assert store.get_version_label("dataset", datasets[1]["id"]) is None
    assert label(store, "configuration", configs["feature"], first)["tag"] == first
    assert label(store, "configuration", configs["protocol"], first)["tag"] == first


def test_rename_and_clear_release_tag_but_keep_revision_tombstone(versions):
    store, datasets, configs, drafts = versions
    first = label(store, "dataset", datasets[0], "v1", note="Original note")
    renamed = label(store, "dataset", datasets[0], "v2", note="Reviewed note", revision=1)
    assert renamed["revision"] == 2
    assert renamed["createdAt"] == first["createdAt"]
    assert renamed["updatedAt"] >= first["updatedAt"]
    label(store, "dataset", datasets[1], "v1")
    cleared = label(store, "dataset", datasets[0], "", revision=2)
    assert cleared["tag"] == cleared["note"] == ""
    assert cleared["revision"] == 3
    assert store.get_dataset(datasets[0]["id"])["versionLabel"] == cleared
    label(store, "dataset", datasets[1], "v2", revision=1)
    with pytest.raises(StorageError) as error:
        label(store, "dataset", datasets[0], "stale", revision=0)
    assert error.value.code == "REVISION_CONFLICT"
    assert store.get_version_label("dataset", datasets[0]["id"]) == cleared


def test_concurrent_edits_cannot_silently_overwrite_one_another(versions):
    store, datasets, configs, drafts = versions
    original = label(store, "dataset", datasets[0], "Original")

    def edit(tag):
        other = ScientificStore(store.folder, store.project_id)
        try:
            return label(other, "dataset", datasets[0], tag, revision=original["revision"])
        except StorageError as error:
            return error.code

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(edit, ["Alice", "Bob"]))
    successful = [outcome for outcome in outcomes if isinstance(outcome, dict)]
    assert len(successful) == 1
    assert next(outcome for outcome in outcomes if isinstance(outcome, str)) in {
        "PROJECT_BUSY",
        "REVISION_CONFLICT",
    }
    assert store.get_version_label("dataset", datasets[0]["id"]) == successful[0]
    with pytest.raises(StorageError) as error:
        label(store, "dataset", datasets[0], "retry stale edit", revision=1)
    assert error.value.code == "REVISION_CONFLICT"


@pytest.mark.parametrize("revision", [-1, True, 1.5, "0", 2**63 - 1])
def test_invalid_revision_never_changes_labels(versions, revision):
    store, datasets, configs, drafts = versions
    with pytest.raises(StorageError) as error:
        label(store, "dataset", datasets[0], "invalid", revision=revision)
    assert error.value.code == "INVALID_VERSION_LABEL"
    assert store.get_version_label("dataset", datasets[0]["id"]) is None


@pytest.mark.parametrize(
    "tag,note", [("x" * 81, ""), ("valid", "x" * 2001), ("bad\nname", ""), ("valid", "bad\x00note")]
)
def test_invalid_text_is_rejected_before_any_write(versions, tag, note):
    store, datasets, configs, drafts = versions
    with pytest.raises(StorageError) as error:
        label(store, "dataset", datasets[0], tag, note=note)
    assert error.value.code == "INVALID_VERSION_LABEL"
    assert store.get_version_label("dataset", datasets[0]["id"]) is None


def test_extra_client_authority_fields_are_rejected():
    with pytest.raises(ValidationError):
        SetVersionLabelRequest(tag="v1", note="", expectedRevision=0, kind="feature")


def test_unknown_or_wrong_resource_type_cannot_create_or_read_labels(versions):
    store, datasets, configs, drafts = versions
    for resource_type, resource_id, code in (
        ("dataset", configs["feature"]["id"], "DATASET_NOT_FOUND"),
        ("configuration", datasets[0]["id"], "CONFIGURATION_NOT_FOUND"),
        ("dataset", "dataset-missing", "DATASET_NOT_FOUND"),
        ("cohort", configs["protocol"]["id"], "INVALID_RESOURCE_TYPE"),
    ):
        for action in (
            lambda: store.get_version_label(resource_type, resource_id),
            lambda: store.set_version_label(
                resource_type, resource_id, tag="v1", note="", expected_revision=0
            ),
        ):
            with pytest.raises(StorageError) as error:
                action()
            assert error.value.code == code
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT count(*) FROM version_labels").fetchone()[0] == 0


@pytest.mark.parametrize(
    "phase,committed", [("version_label_before_commit", False), ("version_label_committed", True)]
)
def test_interrupted_label_edit_is_atomic(versions, phase, committed):
    store, datasets, configs, drafts = versions

    def interrupt(current):
        if current == phase:
            raise KeyboardInterrupt

    store._checkpoint = interrupt
    with pytest.raises(KeyboardInterrupt):
        label(store, "dataset", datasets[0], "v1")
    reopened = ScientificStore(store.folder, store.project_id)
    current = reopened.get_version_label("dataset", datasets[0]["id"])
    assert (current is not None) == committed
    assert reopened.get_dataset(datasets[0]["id"])["contentHash"] == datasets[0]["contentHash"]


def downgrade_v2(store):
    with sqlite3.connect(store.path) as connection:
        connection.execute("DROP TABLE publication_labels")
        connection.execute("DROP TABLE version_labels")
        connection.execute("UPDATE metadata SET value='2' WHERE key='schema_version'")
        connection.execute("PRAGMA user_version=2")


def test_v2_migration_preserves_datasets_configurations_and_existing_drafts(versions):
    store, datasets, configs, drafts = versions
    saved_drafts = store.list_drafts()
    manifests = {
        item["id"]: (store.folder / "datasets" / item["id"] / "manifest.json").read_bytes()
        for item in datasets
    }
    downgrade_v2(store)
    reopened = ScientificStore(store.folder, store.project_id)
    assert reopened.status()["schemaVersion"] == 4
    assert reopened.list_drafts() == saved_drafts
    for item in datasets:
        assert reopened.get_dataset(item["id"]) == item
        assert (store.folder / "datasets" / item["id"] / "manifest.json").read_bytes() == manifests[
            item["id"]
        ]
    for item in configs.values():
        assert reopened.get_configuration(item["id"]) == item
    assert label(reopened, "dataset", datasets[0], "v1")["revision"] == 1


def test_v2_migration_failure_rolls_back_and_retries_without_losing_data(versions):
    store, datasets, configs, drafts = versions
    downgrade_v2(store)
    with patch.object(scientific, "_SCHEMA", (*scientific._SCHEMA[:-1], "INVALID SQL")):
        with pytest.raises(StorageError):
            store.initialize()
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
        assert (
            connection.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()[
                0
            ]
            == "2"
        )
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE name='version_labels'"
            ).fetchone()
            is None
        )
    assert store.get_dataset(datasets[0]["id"]) == datasets[0]
    assert store.get_configuration(configs["feature"]["id"]) == configs["feature"]


def test_corrupt_label_metadata_is_detected_without_changing_scientific_artifacts(versions):
    store, datasets, configs, drafts = versions
    label(store, "dataset", datasets[0], "v1")
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE version_labels SET normalized_tag='wrong' WHERE resource_id=?",
            (datasets[0]["id"],),
        )
    with pytest.raises(StorageError) as error:
        store.get_version_label("dataset", datasets[0]["id"])
    assert error.value.code == "STORAGE_CORRUPT"
    assert store.read_artifact(datasets[0]["id"], "records.json") == b'[{"slideId":"A"}]'


@pytest.fixture(params=["dataset", "feature", "protocol"])
def named_publication(versions, request):
    store, datasets, configs, drafts = versions
    kind = request.param
    draft = store.create_draft("import" if kind == "dataset" else "experiment", "Candidate", {})
    values = {
        "expected_revision": 1,
        "manifest": {
            "kind": kind,
            "name": "Candidate",
            "datasetId": datasets[0]["id"],
            "previewHash": "reviewed-preview",
        },
        "operation_id": "named-publication",
    }
    if kind == "dataset":
        values["artifacts"] = {"records.json": b'[{"slideId":"B"}]'}
    return store, kind, draft, values


def publish_named(case, *, version_label=None, draft=None, **overrides):
    store, kind, original_draft, values = case
    publisher = store.publish_dataset if kind == "dataset" else store.publish_configuration
    return publisher(
        (draft or original_draft)["id"],
        **{**values, **overrides},
        version_label=version_label,
    )


def test_freeze_creates_name_atomically_without_putting_it_in_scientific_content(named_publication):
    store, kind, draft, values = named_publication
    result = publish_named(named_publication, version_label={"tag": "  v1 ", "note": " Reviewed "})
    assert result["versionLabel"]["tag"] == "v1"
    assert result["versionLabel"]["note"] == "Reviewed"
    assert result["versionLabel"]["revision"] == 1
    assert store.get_draft(draft["id"])["status"] == "frozen"
    table = "datasets" if kind == "dataset" else "configurations"
    with sqlite3.connect(store.path) as connection:
        saved = json.loads(
            connection.execute(
                f"SELECT document FROM {table} WHERE id=?", (result["id"],)
            ).fetchone()[0]
        )
    assert saved == {key: value for key, value in result.items() if key != "versionLabel"}
    if kind == "dataset":
        disk = (store.folder / "datasets" / result["id"] / "manifest.json").read_text()
        assert "versionLabel" not in disk
        assert "Reviewed" not in disk


@pytest.mark.parametrize("bad_label", [{}, {"tag": "  "}, {"tag": "v1", "note": "x" * 2001}])
def test_invalid_freeze_label_does_not_publish_or_freeze_draft(named_publication, bad_label):
    store, kind, draft, values = named_publication
    before = store.status()
    with pytest.raises(StorageError) as error:
        publish_named(named_publication, version_label=bad_label)
    assert error.value.code == "INVALID_VERSION_LABEL"
    assert store.status() == before
    assert store.get_draft(draft["id"])["status"] == "editable"


def test_taken_freeze_tag_fails_before_artifacts_or_receipts_are_created(
    named_publication, versions
):
    store, kind, draft, values = named_publication
    _, datasets, configs, _ = versions
    resource_type = "dataset" if kind == "dataset" else "configuration"
    existing = datasets[0] if kind == "dataset" else configs[kind]
    label(store, resource_type, existing, "Baseline")
    before = store.status()
    paths = set(store.folder.rglob("*"))
    with pytest.raises(StorageError) as error:
        publish_named(named_publication, version_label={"tag": " BASELINE "})
    assert error.value.code == "VERSION_TAG_CONFLICT"
    assert store.status() == before
    assert set(store.folder.rglob("*")) == paths
    assert store.get_draft(draft["id"])["status"] == "editable"


def test_named_freeze_retry_returns_current_label_and_rejects_different_original_intent(
    named_publication,
):
    store, kind, draft, values = named_publication
    original_label = {"tag": "Original", "note": "Reviewed"}
    result = publish_named(named_publication, version_label=original_label)
    resource_type = "dataset" if kind == "dataset" else "configuration"
    current = label(store, resource_type, result, "Renamed", note="New note", revision=1)
    assert publish_named(named_publication, version_label=original_label) == {
        **result,
        "versionLabel": current,
    }
    for changed in (
        {"tag": "Different", "note": "Reviewed"},
        {"tag": "Original", "note": "Other"},
        None,
    ):
        with pytest.raises(StorageError) as error:
            publish_named(named_publication, version_label=changed)
        assert error.value.code == "OPERATION_CONFLICT"
    if kind == "dataset":
        replay_args = {
            "draft_id": draft["id"],
            "expected_revision": 1,
            "preview_hash": values["manifest"]["previewHash"],
        }
        assert (
            store.replay_dataset_publication(
                values["operation_id"], **replay_args, version_label=original_label
            )["versionLabel"]
            == current
        )
        with pytest.raises(StorageError) as error:
            store.replay_dataset_publication(
                values["operation_id"], **replay_args, version_label={"tag": "Renamed"}
            )
        assert error.value.code == "OPERATION_CONFLICT"


def test_identical_contents_reuse_exact_label_without_renaming_or_revision_churn(named_publication):
    store, kind, draft, values = named_publication
    requested = {"tag": "v1", "note": "Reviewed"}
    result = publish_named(named_publication, version_label=requested)
    second = store.create_draft(draft["kind"], "Same contents", {})
    duplicate = publish_named(
        named_publication, draft=second, operation_id="same-contents", version_label=requested
    )
    assert duplicate == result
    third = store.create_draft(draft["kind"], "Conflicting name", {})
    for changed in ({"tag": "v2", "note": "Reviewed"}, {"tag": "v1", "note": "Different note"}):
        with pytest.raises(StorageError) as error:
            publish_named(
                named_publication, draft=third, operation_id="different-name", version_label=changed
            )
        assert error.value.code == "VERSION_LABEL_MISMATCH"
        assert "v1" in str(error.value)
        assert result["id"] in str(error.value)
        assert store.get_draft(third["id"])["status"] == "editable"


def test_identical_legacy_or_cleared_version_can_receive_name_during_freeze(named_publication):
    store, kind, draft, values = named_publication
    result = publish_named(named_publication)
    second = store.create_draft(draft["kind"], "Name legacy version", {})
    named = publish_named(
        named_publication, draft=second, operation_id="name-legacy", version_label={"tag": "v1"}
    )
    assert named["id"] == result["id"]
    assert named["versionLabel"]["revision"] == 1
    resource_type = "dataset" if kind == "dataset" else "configuration"
    label(store, resource_type, named, "", revision=1)
    third = store.create_draft(draft["kind"], "Name cleared version", {})
    renamed = publish_named(
        named_publication, draft=third, operation_id="name-cleared", version_label={"tag": "v2"}
    )
    assert renamed["id"] == result["id"]
    assert renamed["versionLabel"]["revision"] == 3


def test_failure_while_writing_freeze_label_never_exposes_unnamed_new_version(named_publication):
    store, kind, draft, values = named_publication
    before = store.status()

    def interrupt(phase):
        if phase == "publication_label_written":
            raise KeyboardInterrupt

    store._checkpoint = interrupt
    with pytest.raises(KeyboardInterrupt):
        publish_named(named_publication, version_label={"tag": "Atomic v1"})
    table = "datasets" if kind == "dataset" else "configurations"
    with sqlite3.connect(store.path) as connection:
        assert (
            connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            == before["datasetCount" if kind == "dataset" else "configurationCount"]
        )
        assert connection.execute("SELECT count(*) FROM version_labels").fetchone()[0] == 0
        assert (
            connection.execute("SELECT status FROM drafts WHERE id=?", (draft["id"],)).fetchone()[0]
            == "editable"
        )
    reopened = ScientificStore(store.folder, store.project_id)
    if kind == "dataset":
        recovered = next(
            item for item in reopened.list_datasets() if item["manifest"].get("name") == "Candidate"
        )
        assert recovered["versionLabel"]["tag"] == "Atomic v1"
        assert reopened.get_draft(draft["id"])["status"] == "frozen"
    else:
        assert reopened.status() == before
        result = publish_named((reopened, kind, draft, values), version_label={"tag": "Atomic v1"})
        assert result["versionLabel"]["tag"] == "Atomic v1"


@pytest.mark.parametrize(
    "phase,recoverable",
    [
        ("journal_committed", False),
        ("artifact_written", False),
        ("staging_complete", True),
        ("directory_published", True),
        ("database_committed", True),
    ],
)
def test_dataset_label_intent_survives_each_recovery_boundary(versions, phase, recoverable):
    store, datasets, configs, drafts = versions
    draft = store.create_draft("import", "Recover named dataset", {})
    kwargs = {
        "expected_revision": 1,
        "manifest": {"kind": "dataset", "name": "Recover named dataset"},
        "artifacts": {"records.json": b'[{"slideId":"Recovery"}]'},
        "operation_id": "recover-label",
        "version_label": {"tag": "Recoverable v1", "note": "Original intent"},
    }

    def interrupt(current):
        if current == phase:
            raise KeyboardInterrupt

    store._checkpoint = interrupt
    with pytest.raises(KeyboardInterrupt):
        store.publish_dataset(draft["id"], **kwargs)
    reopened = ScientificStore(store.folder, store.project_id)
    assert reopened.status()["datasetCount"] == len(datasets) + int(recoverable)
    assert reopened.get_draft(draft["id"])["status"] == ("frozen" if recoverable else "editable")
    result = reopened.publish_dataset(draft["id"], **kwargs)
    assert result["versionLabel"]["tag"] == "Recoverable v1"
    assert result["versionLabel"]["note"] == "Original intent"
    assert result["versionLabel"]["revision"] == 1


def test_preparing_publication_reserves_tag_against_edits_and_other_publications(versions):
    store, datasets, configs, drafts = versions
    candidate = store.create_draft("import", "Pending dataset", {})

    def interrupt(phase):
        if phase == "journal_committed":
            raise KeyboardInterrupt

    store._checkpoint = interrupt
    with pytest.raises(KeyboardInterrupt):
        store.publish_dataset(
            candidate["id"],
            expected_revision=1,
            manifest={"kind": "dataset", "name": "Pending"},
            artifacts={"records.json": b"[]"},
            operation_id="reserved-label",
            version_label={"tag": "Reserved"},
        )
    reopened = ScientificStore(store.folder, store.project_id)
    # Leave recovery deferred to exercise the durable reservation independently of
    # the normal eager recovery path and the process-local writer lock.
    with patch.object(reopened, "_recover_locked"):
        with pytest.raises(StorageError) as error:
            label(reopened, "dataset", datasets[0], " reserved ")
        assert error.value.code == "VERSION_TAG_CONFLICT"
        second = reopened.create_draft("import", "Other dataset", {})
        with pytest.raises(StorageError) as error:
            reopened.publish_dataset(
                second["id"],
                expected_revision=1,
                manifest={"kind": "dataset", "name": "Other"},
                artifacts={"records.json": b"[]"},
                operation_id="steal-reserved-label",
                version_label={"tag": "RESERVED"},
            )
        assert error.value.code == "VERSION_TAG_CONFLICT"
    # Missing staged artifacts make this attempt interrupted, releasing its claim.
    assert reopened.status()["operations"][-1]["status"] == "interrupted"
    assert label(reopened, "dataset", datasets[0], "Reserved")["revision"] == 1


def test_v3_migration_preserves_existing_mutable_labels_and_immutable_documents(versions):
    store, datasets, configs, drafts = versions
    saved = label(store, "dataset", datasets[0], "Existing v3 tag", note="Keep this note")
    with sqlite3.connect(store.path) as connection:
        connection.execute("DROP TABLE publication_labels")
        connection.execute("UPDATE metadata SET value='3' WHERE key='schema_version'")
        connection.execute("PRAGMA user_version=3")
    reopened = ScientificStore(store.folder, store.project_id)
    assert reopened.status()["schemaVersion"] == 4
    assert reopened.get_dataset(datasets[0]["id"]) == {**datasets[0], "versionLabel": saved}
    assert reopened.get_configuration(configs["protocol"]["id"]) == configs["protocol"]


def test_v3_migration_failure_rolls_back_without_losing_labels(versions):
    store, datasets, configs, drafts = versions
    saved = label(store, "dataset", datasets[0], "Keep v3 metadata")
    with sqlite3.connect(store.path) as connection:
        connection.execute("DROP TABLE publication_labels")
        connection.execute("UPDATE metadata SET value='3' WHERE key='schema_version'")
        connection.execute("PRAGMA user_version=3")
    with patch.object(scientific, "_SCHEMA", (*scientific._SCHEMA[:-1], "INVALID SQL")):
        with pytest.raises(StorageError):
            store.initialize()
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
        assert connection.execute("SELECT tag FROM version_labels").fetchone()[0] == saved["tag"]
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE name='publication_labels'"
            ).fetchone()
            is None
        )
    assert store.get_version_label("dataset", datasets[0]["id"]) == saved


@pytest.mark.parametrize("journal_damage", ["missing", "invalid-normalization"])
def test_recovery_never_publishes_unnamed_data_when_label_intent_is_corrupt(
    versions, journal_damage
):
    store, datasets, configs, drafts = versions
    draft = store.create_draft("import", "Named pending dataset", {})

    def interrupt(phase):
        if phase == "directory_published":
            raise KeyboardInterrupt

    store._checkpoint = interrupt
    with pytest.raises(KeyboardInterrupt):
        store.publish_dataset(
            draft["id"],
            expected_revision=1,
            manifest={"kind": "dataset", "name": "Corrupt intent"},
            artifacts={"records.json": b"[]"},
            operation_id="damaged-label",
            version_label={"tag": "Required name"},
        )
    with sqlite3.connect(store.path) as connection:
        if journal_damage == "missing":
            connection.execute("DELETE FROM publication_labels WHERE operation_id='damaged-label'")
        else:
            connection.execute(
                "UPDATE publication_labels SET normalized_tag='wrong' WHERE operation_id='damaged-label'"
            )
    reopened = ScientificStore(store.folder, store.project_id)
    status = reopened.status()
    assert status["datasetCount"] == len(datasets)
    operation = next(item for item in status["operations"] if item["id"] == "damaged-label")
    assert operation["status"] == "interrupted"
    assert operation["error"] == "STORAGE_CORRUPT"
    assert reopened.get_draft(draft["id"])["status"] == "editable"
    with pytest.raises(StorageError) as error:
        reopened.get_dataset(operation["datasetId"])
    assert error.value.code == "DATASET_NOT_FOUND"


def test_v3_unnamed_pending_publication_remains_recoverable_after_upgrade(versions):
    store, datasets, configs, drafts = versions
    draft = store.create_draft("import", "Legacy pending dataset", {})

    def interrupt(phase):
        if phase == "staging_complete":
            raise KeyboardInterrupt

    store._checkpoint = interrupt
    with pytest.raises(KeyboardInterrupt):
        store.publish_dataset(
            draft["id"],
            expected_revision=1,
            manifest={"kind": "dataset", "name": "Legacy pending"},
            artifacts={"records.json": b"[]"},
            operation_id="legacy-pending",
        )
    with sqlite3.connect(store.path) as connection:
        connection.execute("DROP TABLE publication_labels")
        connection.execute("UPDATE metadata SET value='3' WHERE key='schema_version'")
        connection.execute("PRAGMA user_version=3")
    reopened = ScientificStore(store.folder, store.project_id)
    status = reopened.status()
    assert status["schemaVersion"] == 4
    assert status["datasetCount"] == len(datasets) + 1
    operation = next(item for item in status["operations"] if item["id"] == "legacy-pending")
    assert operation["status"] == "published"
    assert "versionLabel" not in reopened.get_dataset(operation["datasetId"])
