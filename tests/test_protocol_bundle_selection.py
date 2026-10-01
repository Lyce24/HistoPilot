"""Dataset selection defines protocols; bundles are bound later in Experiments."""

import copy
import runpy
from pathlib import Path
from unittest.mock import Mock

import pytest
from support import projects

from histopilot.application.protocols import ProtocolService
from histopilot.schemas.protocols import ProtocolExploreRequest, ProtocolSpec
from histopilot.storage.project_lock import StorageError

support = runpy.run_path(str(Path(__file__).with_name("test_protocols.py")))
BUNDLE_ID = "configuration-bundle"
FEATURE_ID = "configuration-features"


@pytest.fixture
def selection(monkeypatch):
    store = support["MemoryStore"]()
    spec = store.draft["payload"]["spec"]
    spec["featureBundleId"] = BUNDLE_ID
    spec["split"] = {
        "version": 4,
        "mode": "kfold",
        "folds": 2,
        "validationFraction": 0.25,
        "pools": {"trainSelection": "remaining"},
    }
    store.feature = {
        "id": FEATURE_ID,
        "contentHash": "f" * 64,
        "manifest": {
            "kind": "feature",
            "datasetId": "another-dataset-used-for-extraction",
            "files": [{"slideId": row["slideId"]} for row in store.rows[:24]]
            + [{"slideId": "bundle-only-slide"}],
        },
    }
    bundle = {
        "id": BUNDLE_ID,
        "contentHash": "b" * 64,
        "current": True,
        "findings": [],
        "manifest": {
            "kind": "feature-bundle",
            "datasetId": "another-dataset-used-for-extraction",
            "spec": {"featureSetId": FEATURE_ID},
            "feature": {"sourceContentHash": "verified-values"},
        },
    }
    service = Mock()
    service.get.return_value = bundle
    monkeypatch.setattr(
        "histopilot.application.feature_bundles.FeatureBundleService", lambda *_: service
    )
    return store, bundle, service


def test_bundle_settings_do_not_restrict_dataset_construction_or_live_counts(selection):
    store, _bundle, bundle_service = selection
    service = ProtocolService(store)
    preview = service.preview("draft-test", 1)
    assert preview["canFreeze"], preview["findings"]
    assert preview["summary"]["totalSlides"] == 32
    assert preview["summary"]["includedSlides"] == 32
    assert "featureExclusions" not in preview["summary"]
    assert preview["summary"]["labelExclusions"] == {}
    assert "featureBundle" not in preview
    assert {row["slideId"] for row in preview["memberships"]} == {
        row["slideId"] for row in store.rows
    }
    live = service.explore(
        ProtocolExploreRequest(
            datasetId=specification(store)["datasetId"],
            featureBundleId=BUNDLE_ID,
            targetField="label",
            split=specification(store)["split"],
        )
    )
    assert live["valid"], live["findings"]
    assert live["cohort"]["totalSlides"] == 32
    assert "featureBundle" not in live
    assert live["target"]["distinctCount"] == 2
    assert not any(item["value"] is None for item in live["target"]["values"])
    bundle_service.get.assert_not_called()


def specification(store):
    return store.draft["payload"]["spec"]


@pytest.mark.parametrize("failure", ["empty", "stale", "missing", "wrong-kind"])
def test_unusable_bundle_does_not_block_dataset_construction(selection, failure):
    store, bundle, service = selection
    if failure == "empty":
        store.feature["manifest"]["files"] = [{"slideId": "elsewhere"}]
    elif failure == "stale":
        bundle["current"] = False
    elif failure == "missing":
        service.get.side_effect = StorageError("Bundle unavailable.", "FEATURE_BUNDLE_NOT_FOUND")
    else:
        store.feature["manifest"]["kind"] = "protocol"
    result = ProtocolService(store).preview("draft-test", 1)
    assert result["canFreeze"], result["findings"]
    assert result["summary"]["includedSlides"] == 32
    live = ProtocolService(store).explore(
        ProtocolExploreRequest(
            datasetId=specification(store)["datasetId"],
            featureBundleId=BUNDLE_ID,
        )
    )
    assert live["valid"], live["findings"]
    assert live["cohort"]["totalSlides"] == 32
    service.get.assert_not_called()


def test_freeze_omits_bundle_and_is_independent_of_bundle_drift(selection):
    store, bundle, _service = selection
    store.publish_configuration = Mock(side_effect=lambda *args, **kwargs: kwargs["manifest"])
    service = ProtocolService(store)
    preview = service.preview("draft-test", 1)
    frozen = service.freeze("draft-test", 1, preview["previewHash"], "save")
    assert "featureBundle" not in frozen
    assert "featureBundleId" not in frozen["spec"]
    bundle["contentHash"] = "c" * 64
    assert service.freeze("draft-test", 1, preview["previewHash"], "save-again") == frozen


def test_obsolete_feature_choices_are_removed_when_loading_protocol_specs(selection):
    store, _bundle, _service = selection
    for extra in ({"featureSetId": FEATURE_ID}, {"featurePackId": "pack-" + "a" * 64}):
        parsed = ProtocolSpec.model_validate({**copy.deepcopy(specification(store)), **extra})
        assert not any(key.startswith("feature") for key in parsed.model_dump())


def test_dataset_only_protocol_binds_reusable_bundle_later_in_experiments(tmp_path, task_center):
    from histopilot.application.development import DevelopmentService
    from histopilot.application.mil_inputs import MILInputService
    from histopilot.application.model_experiments import ModelExperimentService
    from histopilot.schemas.development import DevelopmentBatchSpec
    from histopilot.schemas.mil import MILInputSpec
    from histopilot.schemas.model_experiments import CreateModelExperiment
    from histopilot.storage.filesystem import LocalFilesystem
    from histopilot.storage.scientific import ScientificStore

    folder = tmp_path / "project"
    folder.mkdir()
    store = ScientificStore(folder, "project-bundle-intersection")
    filesystem = LocalFilesystem((tmp_path,))
    encoded_rows = [
        {
            "slideId": f"s{i:02}",
            "patientId": f"p{i:02}",
            "attributes": {"label": str(i % 2), "cohort": "Site A" if i < 12 else "Site B"},
        }
        for i in range(26)
    ]
    original, _ = projects.dataset(store, rows=encoded_rows)
    bundle, _pack, _source = projects.bundle(
        store,
        tmp_path,
        original,
        [row["slideId"] for row in encoded_rows],
    )
    study_rows = encoded_rows[:24] + [
        {
            "slideId": "unencoded",
            "patientId": None,
            "attributes": {"label": None, "cohort": "Site C"},
        }
    ]
    study, _ = projects.dataset(store, operation="new-study", rows=study_rows)
    spec = {
        "datasetId": study["id"],
        "featureBundleId": bundle["id"],
        "target": projects.TARGET,
        "split": {
            "version": 4,
            "mode": "kfold",
            "folds": 2,
            "pools": {
                "rules": {"train": [{"field": "cohort", "op": "in", "value": ["Site A", "Site B"]}]}
            },
        },
    }
    protocols = ProtocolService(store, filesystem)
    draft = store.create_draft(
        "experiment", "Combined study", {"type": "analysis-protocol", "spec": spec}
    )
    preview = protocols.preview(draft["id"], 1)
    assert preview["canFreeze"], preview["findings"]
    assert preview["summary"]["includedSlides"] == 24
    assert "featureBundle" not in preview
    assert "featureBundleId" not in preview["spec"]
    frozen = protocols.freeze(draft["id"], 1, preview["previewHash"], "study-protocol")
    inputs = MILInputSpec(protocolId=frozen["id"], featureBundleId=bundle["id"])
    resolved = MILInputService(store, filesystem).preview(inputs)
    assert resolved["canPlan"], resolved["findings"]
    assert resolved["resolvedLoadingPolicy"] == "native"
    experiment = ModelExperimentService(store, filesystem).create(
        CreateModelExperiment(
            name="Combined study",
            inputs=inputs,
            operationId="study-experiment",
        )
    )
    plan = DevelopmentService(store, filesystem).preview(
        DevelopmentBatchSpec(
            experimentId=experiment["id"],
            experimentRevision=experiment["revision"],
            experimentName="Combined study",
            batchName="Baseline",
            inputs=inputs,
            mode="explicit",
            configurations=[{}],
            trainingSeeds=[42],
        )
    )
    assert plan["canFreeze"], plan["findings"]
    assert plan["inputSnapshot"]["dataset"]["id"] == study["id"]
    assert plan["inputSnapshot"]["featureBundle"]["id"] == bundle["id"]
    assert store.get_configuration(bundle["id"])["manifest"]["datasetId"] == original["id"]


def test_obsolete_required_coverage_does_not_block_construction(selection):
    store, _bundle, _service = selection
    spec = specification(store)
    spec["featureCoverage"] = "require"
    for index, row in enumerate(store.rows):
        row["attributes"]["label"] = str((index // 2) % 2)
    service = ProtocolService(store)
    result = service.preview("draft-test", 1)
    assert result["canFreeze"], result["findings"]
    assert result["summary"]["includedSlides"] == 32
    assert "featureCoverage" not in result["spec"]
    assert not any("FEATURE" in item["code"] for item in result["findings"])
    assert not any(item["code"] == "RESTRICTED_TO_FEATURE_COVERAGE" for item in result["findings"])
    live = service.explore(
        ProtocolExploreRequest(
            datasetId=spec["datasetId"],
            featureBundleId=BUNDLE_ID,
            featureCoverage="require",
            split=spec["split"],
        )
    )
    assert live["valid"], live["findings"]
    assert live["cohort"]["totalSlides"] == 32
    assert live["partitions"]["train"]["expanded"]["totalSlides"] == 32
    assert not any("FEATURE" in item["code"] for item in live["findings"])


def test_obsolete_bundle_restriction_cannot_hide_missing_target_labels(selection):
    store, _bundle, _service = selection
    specification(store)["featureCoverage"] = "restrict"
    for row in store.rows[24:]:
        row["attributes"]["label"] = None
    result = ProtocolService(store).preview("draft-test", 1)
    assert not result["canFreeze"]
    assert result["summary"]["eligibleSlides"] == 32
    assert "MISSING_LABEL" in {item["code"] for item in result["findings"]}
    assert result["memberships"] == []
    assert "featureCoverage" not in result["spec"]


def test_explicit_dataset_selection_persists_without_legacy_bundle_settings(selection):
    store, _bundle, _service = selection
    spec = specification(store)
    spec["featureCoverage"] = "require"
    selected = [row["slideId"] for row in store.rows[:24]]
    spec["split"]["pools"] = {
        "trainSelection": "rules",
        "rules": {"train": [{"field": "Slide_ID", "op": "in", "value": selected}]},
    }
    service = ProtocolService(store)
    result = service.preview("draft-test", 1)
    assert result["canFreeze"], result["findings"]
    assert result["summary"]["includedSlides"] == 24
    assert "featureCoverage" not in result["spec"]
    store.draft["payload"]["spec"] = result["spec"]
    reloaded = service.preview("draft-test", 1)
    assert result["previewHash"] == reloaded["previewHash"]
    live = service.explore(
        ProtocolExploreRequest(
            datasetId=spec["datasetId"],
            featureBundleId=BUNDLE_ID,
            featureCoverage="require",
            split=spec["split"],
        )
    )
    assert live["valid"], live["findings"]
    assert live["partitions"]["train"]["expanded"]["totalSlides"] == 24
    assert live["unassigned"]["totalSlides"] == 8
