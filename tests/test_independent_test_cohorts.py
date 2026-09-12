"""Test cohorts freeze before feature extraction; model evaluation checks compatibility."""

import copy
import runpy
from pathlib import Path
from uuid import uuid4

import h5py
import pytest

from histopilot.application.bulk_evaluations import BulkEvaluationService
from histopilot.application.evaluation_runs import EvaluationRunService
from histopilot.application.evaluations import EvaluationService
from histopilot.schemas.bulk_evaluations import BulkEvaluationSelection
from histopilot.schemas.predictors import EvaluationRunSelection, SaveEvaluationRun
from histopilot.storage.filesystem import LocalFilesystem
from histopilot.storage.scientific import ScientificStore

support = runpy.run_path(str(Path(__file__).with_name("test_predictor_registry.py")))
cohorts = support["support"]


def independent(service, spec):
    spec = copy.deepcopy(spec)
    for key in ("protocolId", "developmentFeatureBundleId", "featureBundleId"):
        spec.pop(key, None)
    draft = cohorts["draft"](service, spec)
    preview = service.preview(draft["id"], 1)
    assert preview["canFreeze"], preview["findings"]
    return service.freeze(draft["id"], 1, preview["previewHash"], uuid4().hex)


@pytest.fixture
def evaluation(tmp_path):
    predictors, legacy = support["registry"].__wrapped__(tmp_path)
    selected, *_ = support["candidate"](predictors)
    predictor, _ = support["freeze"](predictors, selected)
    service = EvaluationRunService(predictors.store, predictors.filesystem)
    cohort = independent(service.cohorts, legacy["manifest"]["spec"])
    choice = EvaluationRunSelection(
        predictorId=predictor["id"], cohortId=cohort["id"], name="Independent evaluation"
    )
    return service, choice, cohort, predictor


def test_can_freeze_multiple_datasets_without_any_features_or_development(tmp_path, monkeypatch):
    folder = tmp_path / "project"
    folder.mkdir()
    store = ScientificStore(folder, "project-independent")
    first, _ = cohorts["dataset"](store)
    second, _ = cohorts["dataset"](
        store,
        "external",
        [
            {"slideId": "t0", "patientId": "q0", "attributes": {"label": "0", "cohort": "test"}},
            {"slideId": "t1", "patientId": "q1", "attributes": {"label": "1", "cohort": "test"}},
        ],
    )
    service = EvaluationService(store, LocalFilesystem((tmp_path,)))

    def no_feature_reads(*_args, **_kwargs):
        pytest.fail("Cohort creation must not require feature verification")

    monkeypatch.setattr(service.bundles, "get", no_feature_reads)
    frozen = independent(
        service,
        {
            "datasetId": first["id"],
            "datasetIds": [first["id"], second["id"]],
            "target": cohorts["TARGET"],
            "eligibility": [{"field": "cohort", "op": "eq", "value": "test"}],
        },
    )
    manifest = frozen["manifest"]
    assert manifest["schemaVersion"] == 2
    assert manifest["coverage"]["selectedSlideIds"] == ["s2", "s3", "t0", "t1"]
    assert manifest["coverage"]["deferred"]
    assert manifest["summary"]["classCounts"] == {"low": 2, "high": 2}
    assert len(manifest["bindings"]["datasets"]) == 2
    assert "protocol" not in manifest["bindings"]
    assert service.get(frozen["id"])["current"]
    assert store.list_configurations("feature-bundle") == []
    assert store.list_configurations("protocol") == []


def test_multidataset_duplicate_slide_ids_block_freeze(tmp_path):
    folder = tmp_path / "project"
    folder.mkdir()
    store = ScientificStore(folder, "project-independent")
    first, rows = cohorts["dataset"](store)
    second, _ = cohorts["dataset"](store, "duplicate-import", rows[:1])
    service = EvaluationService(store, LocalFilesystem((tmp_path,)))
    preview = cohorts["preview"](
        service,
        {
            "datasetId": first["id"],
            "datasetIds": [first["id"], second["id"]],
            "target": cohorts["TARGET"],
        },
    )
    assert not preview["canFreeze"]
    assert "DUPLICATE_SLIDE_ID" in cohorts["codes"](preview)


def test_initial_data_preview_supports_no_target_or_legacy_bindings(tmp_path):
    folder = tmp_path / "project"
    folder.mkdir()
    store = ScientificStore(folder, "project-independent")
    data, _ = cohorts["dataset"](store)
    service = EvaluationService(store, LocalFilesystem((tmp_path,)))
    preview = cohorts["preview"](
        service,
        {
            "datasetId": data["id"],
            "protocolId": "",
            "developmentFeatureBundleId": "",
            "featureBundleId": "",
            "target": None,
        },
    )
    assert preview["canFreeze"]
    assert preview["target"] is None
    assert preview["summary"]["includedSlides"] == 4
    assert preview["summary"]["classCounts"] == {}


def test_missing_or_changed_features_do_not_make_cohort_stale(evaluation):
    service, choice, cohort, predictor = evaluation
    bundle = service.store.get_configuration(
        predictor["manifest"]["inputs"]["features"]["bundle"]["id"]
    )
    feature = service.store.get_configuration(bundle["manifest"]["feature"]["id"])
    path = next(row["path"] for row in feature["manifest"]["files"] if row["slideId"] == "s2")
    with h5py.File(path, "r+") as handle:
        handle["features"][0, 0] = 99
    assert service.cohorts.get(cohort["id"])["current"]
    assert not service.preview(choice)["canSave"]


@pytest.mark.parametrize(
    "change",
    [
        {"unit": "slide"},
        {"classes": ["high", "low"]},
        {"positiveClass": "low"},
    ],
)
def test_target_can_freeze_independently_but_mismatch_blocks_evaluation(evaluation, change):
    service, choice, cohort, _predictor = evaluation
    spec = cohort["manifest"]["spec"]
    frozen = independent(service.cohorts, {**spec, "target": {**spec["target"], **change}})
    preview = service.preview(choice.model_copy(update={"cohortId": frozen["id"]}))
    assert not preview["canSave"]
    assert "TARGET_CONTRACT_MISMATCH" in cohorts["codes"](preview)


def test_development_overlap_is_checked_at_evaluation(evaluation):
    service, choice, cohort, _predictor = evaluation
    overlapping = independent(service.cohorts, {**cohort["manifest"]["spec"], "eligibility": []})
    assert service.cohorts.get(overlapping["id"])["current"]
    preview = service.preview(choice.model_copy(update={"cohortId": overlapping["id"]}))
    assert not preview["canSave"]
    assert {"DEVELOPMENT_SLIDE_OVERLAP", "DEVELOPMENT_PATIENT_OVERLAP"} <= cohorts["codes"](preview)


def test_evaluation_checks_exact_extracted_membership(evaluation, tmp_path):
    service, choice, cohort, _predictor = evaluation
    data = service.store.get_dataset(cohort["manifest"]["datasetId"])
    missing, _, _ = cohorts["bundle"](service.store, tmp_path, data, ["s2"], name="partial-test")
    preview = service.preview(choice.model_copy(update={"featureBundleId": missing["id"]}))
    assert not preview["canSave"]
    assert "MISSING_TEST_FEATURES" in cohorts["codes"](preview)
    assert service.cohorts.get(cohort["id"])["current"]


def test_evaluation_selects_packed_loading_and_checks_exact_pack_membership(
    evaluation, tmp_path, monkeypatch
):
    from histopilot.application import evaluations as module
    from histopilot.schemas.evaluations import InferenceSettings

    service, choice, cohort, _predictor = evaluation
    data = service.store.get_dataset(cohort["manifest"]["datasetId"])
    bundle, pack, _ = cohorts["bundle"](
        service.store, tmp_path, data, ["s2", "s3"], name="packed-test", pack=True
    )
    choice = choice.model_copy(
        update={
            "featureBundleId": bundle["id"],
            "inference": InferenceSettings(loadingPolicy="packed", packArtifactId=pack),
        }
    )
    preview = service.preview(choice)
    assert preview["canSave"], preview
    assert preview["manifest"]["coverage"]["packChecked"]
    original = module._layout

    def missing_slide(path):
        layout = original(path)
        return {**layout, "slides": [row for row in layout["slides"] if row["slideId"] != "s3"]}

    monkeypatch.setattr(module, "_layout", missing_slide)
    preview = service.preview(choice)
    assert not preview["canSave"]
    assert "MISSING_TEST_PACK_SLIDES" in cohorts["codes"](preview)


def test_automatic_feature_choice_is_pinned_for_saved_evaluation(evaluation, tmp_path, monkeypatch):
    service, choice, cohort, predictor = evaluation
    preview = service.preview(choice)
    assert preview["canSave"], preview
    request = SaveEvaluationRun(
        **choice.model_dump(), previewHash=preview["previewHash"], operationId="save-independent"
    )
    saved = service.save(request)
    assert service.save(request)["id"] == saved["id"]
    assert saved["manifest"]["features"] == predictor["manifest"]["inputs"]["features"]
    data = service.store.get_dataset(cohort["manifest"]["datasetId"])
    cohorts["bundle"](service.store, tmp_path, data, ["s2", "s3"], name="later-test")
    assert "EVALUATION_FEATURE_BUNDLE_AMBIGUOUS" in cohorts["codes"](service.preview(choice))
    monkeypatch.setattr("histopilot.application.evaluation_runs.training_runtime", lambda: {})
    plan = service._execution_plan(saved["id"])
    assert set(plan["data"]["featureFiles"]) == {"s2", "s3"}
    assert plan["data"]["memberships"] == cohort["manifest"]["memberships"]


def test_bulk_review_applies_selected_features_and_inference(evaluation, tmp_path):
    from histopilot.schemas.evaluations import InferenceSettings

    service, _choice, cohort, predictor = evaluation
    data = service.store.get_dataset(cohort["manifest"]["datasetId"])
    bundle, pack, _ = cohorts["bundle"](
        service.store, tmp_path, data, ["s2", "s3"], name="bulk-test", pack=True
    )
    bulk = BulkEvaluationService(service.store, service.filesystem, evaluations=service)
    choice = BulkEvaluationSelection(
        cohortId=cohort["id"],
        scope="selected",
        predictorIds=[predictor["id"]],
        featureBundleId=bundle["id"],
        inference=InferenceSettings(loadingPolicy="packed", packArtifactId=pack, batchSize=4),
    )
    preview = bulk.preview(choice)
    assert preview["canRun"], preview
    manifest = preview["items"][0]["evaluationManifest"]
    assert manifest["features"]["bundle"]["id"] == bundle["id"]
    assert manifest["inference"]["packArtifactId"] == pack
    assert manifest["inference"]["batchSize"] == 4


def test_multidataset_evaluation_uses_combined_exact_slide_membership(evaluation):
    service, choice, cohort, _predictor = evaluation
    sources = []
    for index in (2, 3):
        source, _ = cohorts["dataset"](
            service.store,
            f"source-{index}",
            [
                {
                    "slideId": f"s{index}",
                    "patientId": f"p{index}",
                    "attributes": {"label": str(index % 2), "cohort": "test"},
                }
            ],
        )
        sources.append(source["id"])
    merged = independent(
        service.cohorts,
        {
            **cohort["manifest"]["spec"],
            "datasetId": sources[0],
            "datasetIds": sources,
        },
    )
    preview = service.preview(choice.model_copy(update={"cohortId": merged["id"]}))
    assert preview["canSave"], preview
    assert preview["manifest"]["coverage"]["selectedSlideIds"] == ["s2", "s3"]
    assert preview["manifest"]["summary"]["includedSlides"] == 2
    assert preview["manifest"]["datasetIds"] == sources


def test_patient_namespace_declaration_is_selected_at_evaluation(evaluation):
    service, choice, cohort, predictor = evaluation
    source, _ = cohorts["dataset"](
        service.store,
        "separate-identifiers",
        [
            {
                "slideId": "s2",
                "patientId": "p0",
                "attributes": {"label": "0", "cohort": "test"},
            }
        ],
    )
    external = independent(
        service.cohorts,
        {
            **cohort["manifest"]["spec"],
            "datasetId": source["id"],
        },
    )
    selected = choice.model_copy(update={"cohortId": external["id"]})
    preview = service.preview(selected)
    assert "DEVELOPMENT_PATIENT_OVERLAP" in cohorts["codes"](preview)
    preview = service.preview(selected.model_copy(update={"patientIdentifiers": "independent"}))
    assert preview["canSave"], preview
    assert any(row["code"] == "PATIENT_OVERLAP_UNVERIFIABLE" for row in preview["findings"])
    bulk = BulkEvaluationService(service.store, service.filesystem, evaluations=service)
    review = bulk.preview(
        BulkEvaluationSelection(
            cohortId=external["id"],
            scope="selected",
            predictorIds=[predictor["id"]],
            patientIdentifiers="independent",
        )
    )
    assert review["canRun"], review
