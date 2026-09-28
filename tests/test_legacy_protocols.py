"""Frozen split version 1-3 protocols stay readable after their generators were removed.

`fixtures/legacy_protocols.json` holds protocols frozen by commit 818ccea, the last
that could still generate these split versions: v1 fixed-rule k-fold, v2 k-fold and
v3 explicit pools with an external test pool.
"""

import copy
import json
from pathlib import Path

import pytest

from histopilot.application.protocols import ProtocolService
from histopilot.schemas.protocols import ProtocolExploreRequest, ProtocolSpec
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import ScientificStore

FIXTURE = json.loads(
    Path(__file__).with_name("fixtures").joinpath("legacy_protocols.json").read_text()
)
VERSIONS = {"v1": 1, "v2": 2, "v3": 3}


def legacy_store(folder):
    """Publish the fixture dataset; its content-addressed identity matches the records."""
    store = ScientificStore(folder, "project-legacy")
    store.initialize()
    source = store.create_draft("import", "Source", {})
    dataset = store.publish_dataset(
        source["id"],
        expected_revision=1,
        manifest=FIXTURE["dataset"]["manifest"],
        artifacts={
            "records.json": json.dumps(FIXTURE["dataset"]["records"], sort_keys=True).encode()
        },
        operation_id="legacy-source",
    )
    assert dataset["id"] == FIXTURE["dataset"]["id"]
    return store


def publish_legacy(store, name, operation_id=None):
    """Store a frozen legacy protocol exactly as the old freeze published it."""
    protocol = FIXTURE["protocols"][name]
    draft = store.create_draft(
        "experiment",
        f"Legacy {name}",
        {"type": "analysis-protocol", "spec": copy.deepcopy(protocol["draftSpec"])},
    )
    frozen = store.publish_configuration(
        draft["id"],
        expected_revision=1,
        manifest=copy.deepcopy(protocol["manifest"]),
        operation_id=operation_id or f"legacy-{name}",
    )
    return draft, frozen


@pytest.mark.parametrize("name", VERSIONS)
def test_frozen_legacy_specs_still_validate_for_reading(name):
    protocol = FIXTURE["protocols"][name]
    for spec in (protocol["manifest"]["spec"], protocol["draftSpec"]):
        parsed = ProtocolSpec.model_validate(spec)
        assert parsed.split.version == VERSIONS[name]
    assert protocol["manifest"]["memberships"]
    assert protocol["manifest"]["summary"]["algorithm"] == protocol["manifest"]["algorithm"]


@pytest.mark.parametrize("name", VERSIONS)
def test_frozen_legacy_protocol_is_read_and_replayed_unchanged(tmp_path, name):
    store = legacy_store(tmp_path)
    draft, frozen = publish_legacy(store, name)
    assert frozen["manifest"] == FIXTURE["protocols"][name]["manifest"]
    reopened = ScientificStore(tmp_path, "project-legacy")
    assert reopened.get_configuration(frozen["id"]) == frozen
    # Retrying the original freeze returns the stored record; nothing is regenerated.
    preview_hash = frozen["manifest"]["previewHash"]
    replay = ProtocolService(reopened).freeze(draft["id"], 1, preview_hash, f"legacy-{name}")
    assert replay == frozen
    with pytest.raises(StorageError) as conflict:
        ProtocolService(reopened).freeze(draft["id"], 1, "0" * 64, f"legacy-{name}")
    assert conflict.value.code == "OPERATION_CONFLICT"


@pytest.mark.parametrize("name", VERSIONS)
def test_legacy_drafts_can_no_longer_be_previewed_or_frozen(tmp_path, name):
    store = legacy_store(tmp_path)
    draft = store.create_draft(
        "experiment",
        "Old editable draft",
        {"type": "analysis-protocol", "spec": FIXTURE["protocols"][name]["draftSpec"]},
    )
    service = ProtocolService(store)
    with pytest.raises(StorageError) as preview:
        service.preview(draft["id"], 1)
    assert preview.value.code == "LEGACY_PROTOCOL_SPLIT"
    assert preview.value.status_code == 422
    assert "frozen versions stay readable" in str(preview.value)
    preview_hash = FIXTURE["protocols"][name]["manifest"]["previewHash"]
    with pytest.raises(StorageError) as freeze:
        service.freeze(draft["id"], 1, preview_hash, "new-legacy-freeze")
    assert freeze.value.code == "LEGACY_PROTOCOL_SPLIT"
    assert store.get_draft(draft["id"])["status"] == "editable"


@pytest.mark.parametrize("name", VERSIONS)
def test_legacy_splits_get_cohort_counts_but_no_live_partition_counts(tmp_path, name):
    # Split versions 1-3 are frozen history; live partition counts cover version 4 only.
    store = legacy_store(tmp_path)
    spec = FIXTURE["protocols"][name]["manifest"]["spec"]
    result = ProtocolService(store).explore(
        ProtocolExploreRequest(
            datasetId=spec["datasetId"], targetField="label", split=spec["split"]
        )
    )
    assert not result["valid"]
    assert "INVALID_STRATEGY_CONFIG" in {finding["code"] for finding in result["findings"]}
    assert result["cohort"]["totalSlides"] == len(FIXTURE["dataset"]["records"])
    assert result["partitions"] is result["unassigned"] is None


def test_development_freeze_does_not_rewrite_existing_legacy_protocol(tmp_path):
    store = legacy_store(tmp_path)
    legacy_draft, legacy = publish_legacy(store, "v3", "legacy")
    original = store.get_configuration(legacy["id"])
    spec = {
        "datasetId": FIXTURE["dataset"]["id"],
        "target": FIXTURE["protocols"]["v3"]["draftSpec"]["target"],
        "split": {
            "version": 4,
            "mode": "kfold",
            "folds": 2,
            "seeds": [42],
            "pools": {"rules": {"train": [{"field": "partition", "op": "eq", "value": "train"}]}},
        },
    }
    dev_draft = store.create_draft(
        "experiment", "Development", {"type": "analysis-protocol", "spec": spec}
    )
    service = ProtocolService(store)
    reviewed = service.preview(dev_draft["id"], 1)
    development = service.freeze(dev_draft["id"], 1, reviewed["previewHash"], "development")
    assert development["id"] != legacy["id"]
    assert store.get_configuration(legacy["id"]) == original
    preview_hash = legacy["manifest"]["previewHash"]
    assert service.freeze(legacy_draft["id"], 1, preview_hash, "legacy") == legacy
