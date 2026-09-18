"""Dataset/bundle intersection defines protocols before labels or patient splits."""

import copy
import runpy
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

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
    # These four patients are absent from the bundle and cannot affect target validation.
    for row in store.rows[24:]:
        row["attributes"]["label"] = None
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
    monkeypatch.setattr("histopilot.application.protocols.FeatureBundleService", lambda *_: service)
    return store, bundle, service


def test_bundle_intersection_precedes_target_checks_and_agrees_with_live_counts(selection):
    store, bundle, _service = selection
    service = ProtocolService(store)
    preview = service.preview("draft-test", 1)
    assert preview["canFreeze"], preview["findings"]
    assert preview["summary"]["totalSlides"] == 32
    assert preview["summary"]["includedSlides"] == 24
    assert preview["summary"]["bundleSlides"] == 25
    assert preview["summary"]["matchedSlides"] == 24
    assert preview["summary"]["featureExclusions"] == 8
    assert preview["summary"]["populationSource"] == "dataset_and_bundle"
    assert preview["summary"]["labelExclusions"] == {}
    assert preview["featureBundle"]["contentHash"] == bundle["contentHash"]
    assert {row["slideId"] for row in preview["memberships"]} == {
        row["slideId"] for row in store.rows[:24]
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
    assert live["cohort"]["totalSlides"] == 24
    assert live["matchedSlides"] == 24
    assert live["featureBundle"] == preview["featureBundle"]
    assert live["target"]["distinctCount"] == 2
    assert not any(item["value"] is None for item in live["target"]["values"])


def specification(store):
    return store.draft["payload"]["spec"]


@pytest.mark.parametrize("failure", ["empty", "stale", "missing", "wrong-kind"])
def test_unusable_bundle_never_falls_back_to_the_whole_dataset(selection, failure):
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
    assert not result["canFreeze"]
    live = ProtocolService(store).explore(
        ProtocolExploreRequest(
            datasetId=specification(store)["datasetId"],
            featureBundleId=BUNDLE_ID,
        )
    )
    assert not live["valid"]


def test_freeze_pins_bundle_and_rejects_changed_review(selection):
    store, bundle, _service = selection
    store.publish_configuration = Mock(side_effect=lambda *args, **kwargs: kwargs["manifest"])
    service = ProtocolService(store)
    preview = service.preview("draft-test", 1)
    frozen = service.freeze("draft-test", 1, preview["previewHash"], "save")
    assert frozen["featureBundle"] == preview["featureBundle"]
    assert frozen["spec"]["featureBundleId"] == BUNDLE_ID
    bundle["contentHash"] = "c" * 64
    with pytest.raises(StorageError, match="preview changed"):
        service.freeze("draft-test", 1, preview["previewHash"], "save-again")


def test_bundle_and_legacy_feature_choices_cannot_be_combined(selection):
    store, _bundle, _service = selection
    for extra in ({"featureSetId": FEATURE_ID}, {"featurePackId": "pack-" + "a" * 64}):
        with pytest.raises(ValidationError):
            ProtocolSpec.model_validate({**copy.deepcopy(specification(store)), **extra})


def test_verified_bundle_reused_by_new_dataset_through_development_planning(tmp_path):
    from histopilot.application.development import DevelopmentService
    from histopilot.application.mil_inputs import MILInputService
    from histopilot.application.model_experiments import ModelExperimentService
    from histopilot.schemas.development import DevelopmentBatchSpec
    from histopilot.schemas.mil import MILInputSpec
    from histopilot.schemas.model_experiments import CreateModelExperiment
    from histopilot.storage.filesystem import LocalFilesystem
    from histopilot.storage.scientific import ScientificStore

    fixtures = runpy.run_path(str(Path(__file__).with_name("test_evaluations.py")))
    folder = tmp_path / "project"
    folder.mkdir()
    store = ScientificStore(folder, "project-bundle-intersection")
    filesystem = LocalFilesystem((tmp_path,))
    encoded_rows = [
        {
            "slideId": f"s{i:02}",
            "patientId": f"p{i:02}",
            "attributes": {"label": str(i % 2), "cohort": "TCGA" if i < 12 else "SurGen"},
        }
        for i in range(26)
    ]
    original, _ = fixtures["dataset"](store, rows=encoded_rows)
    bundle, _pack, _source = fixtures["bundle"](
        store,
        tmp_path,
        original,
        [row["slideId"] for row in encoded_rows],
    )
    study_rows = encoded_rows[:24] + [
        {
            "slideId": "unencoded",
            "patientId": None,
            "attributes": {"label": None, "cohort": "RIH"},
        }
    ]
    study, _ = fixtures["dataset"](store, operation="new-study", rows=study_rows)
    spec = {
        "datasetId": study["id"],
        "featureBundleId": bundle["id"],
        "target": fixtures["TARGET"],
        "split": {
            "version": 4,
            "mode": "kfold",
            "folds": 2,
            "pools": {"trainSelection": "remaining"},
        },
    }
    protocols = ProtocolService(store, filesystem)
    draft = store.create_draft(
        "experiment", "Combined study", {"type": "analysis-protocol", "spec": spec}
    )
    preview = protocols.preview(draft["id"], 1)
    assert preview["canFreeze"], preview["findings"]
    assert preview["summary"]["includedSlides"] == 24
    assert preview["summary"]["bundleSlides"] == 26
    frozen = protocols.freeze(draft["id"], 1, preview["previewHash"], "study-protocol")
    inputs = MILInputSpec(protocolId=frozen["id"], featureBundleId=bundle["id"])
    resolved = MILInputService(store, filesystem).preview(inputs)
    assert resolved["canPlan"], resolved["findings"]
    assert resolved["resolvedLoadingPolicy"] == "native"
    experiment = ModelExperimentService(store, filesystem).create(CreateModelExperiment(
        name="Combined study", inputs=inputs, operationId="study-experiment",
    ))
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
