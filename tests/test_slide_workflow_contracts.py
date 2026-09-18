"""Representation boundaries, legacy identities, and original-image slide review."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_case_review import cases
from test_evaluations import bundle, codes, evaluation, preview
from test_morphology import study
from test_predictor_registry import candidate, freeze, registry
from test_training_control_preview import preview_context

from histopilot.application.development import development_plans
from histopilot.application.evaluation_runs import EvaluationRunService
from histopilot.application.predictors import feature_contract, reference
from histopilot.application.slide_reviews import SlideReviewService
from histopilot.domain.features import representation_kind
from histopilot.schemas.case_review import CaseReviewQuery
from histopilot.schemas.development import TrainingRecipe
from histopilot.schemas.evaluations import InferenceSettings
from histopilot.schemas.predictors import EvaluationRunSelection
from histopilot.schemas.slide_reviews import SaveSlideReview
from histopilot.storage.project_lock import StorageError

__all__ = ["cases", "evaluation", "study", "registry", "preview_context"]


def test_patch_contract_identity_stays_unchanged_and_slide_kind_is_explicit():
    feature = {
        "id": "feature", "contentHash": "feature-hash",
        "manifest": {"files": [{"dimensions": 8, "dtype": "float32"}],
                     "spec": {"encoderId": "encoder"}},
    }
    frozen_bundle = {
        "id": "bundle", "contentHash": "bundle-hash",
        "manifest": {"feature": {"sourceContentHash": "source-hash"}},
    }
    legacy = {
        "feature": reference(feature), "bundle": reference(frozen_bundle),
        "dimensions": 8, "dtype": "float32", "encoderId": "encoder",
        "sourceContentHash": "source-hash", "extraction": None, "layout": {},
    }
    assert feature_contract(feature, frozen_bundle) == legacy
    feature["manifest"]["spec"]["featureKind"] = "patch"
    assert feature_contract(feature, frozen_bundle) == legacy
    feature["manifest"]["spec"]["featureKind"] = "slide"
    assert feature_contract(feature, frozen_bundle) == {**legacy, "featureKind": "slide"}
    assert representation_kind(legacy) == "patch"


@pytest.mark.parametrize("slide_side", ["developmentFeatureBundleId", "featureBundleId"])
def test_bound_evaluation_rejects_wrong_kind_even_with_matching_encoder_and_dimension(
    evaluation, tmp_path, slide_side,
):
    service, spec, _ = evaluation
    data = service.store.get_dataset(spec["datasetId"])
    slides, _, _ = bundle(service.store, tmp_path, data, [f"s{i}" for i in range(4)],
                          name="slide-features", feature_kind="slide")
    result = preview(service, {**spec, slide_side: slides["id"]})
    assert not result["canFreeze"]
    assert "FEATURE_KIND_MISMATCH" in codes(result)


@pytest.mark.parametrize("kind", ["patch", "slide"])
def test_automatic_test_bundle_resolution_uses_representation_kind(registry, tmp_path, kind):
    predictors, cohort = registry
    selection, *_ = candidate(predictors)
    predictor, _ = freeze(predictors, selection)
    service = EvaluationRunService(predictors.store, predictors.filesystem)
    data = service.store.get_dataset(cohort["manifest"]["datasetId"])
    slides, _, _ = bundle(service.store, tmp_path, data, [f"s{i}" for i in range(4)],
                          name="slide-features", feature_kind="slide")
    model = deepcopy(predictor["manifest"])
    if kind == "slide":
        model["inputs"]["features"]["featureKind"] = "slide"
    test = deepcopy(cohort["manifest"])
    test["spec"].pop("featureBundleId", None)
    chosen = service._test_bundle(SimpleNamespace(featureBundleId=None), model, test,
                                 InferenceSettings())
    expected = slides["id"] if kind == "slide" else model["inputs"]["features"]["bundle"]["id"]
    assert chosen == expected


def test_saved_predictor_contract_rejects_wrong_kind_at_evaluation_review(registry, monkeypatch):
    predictors, cohort = registry
    selection, *_ = candidate(predictors)
    predictor, _ = freeze(predictors, selection)
    service = EvaluationRunService(predictors.store, predictors.filesystem)
    altered = deepcopy(predictor)
    altered["manifest"]["inputs"]["features"]["featureKind"] = "slide"
    monkeypatch.setattr(service.predictors, "get", lambda _: altered)
    result = service.preview(EvaluationRunSelection(
        predictorId=predictor["id"], cohortId=cohort["id"], name="Wrong kind",
    ))
    assert not result["canSave"]
    assert result["findings"][0]["code"] == "EVALUATION_FEATURE_CONTRACT_MISMATCH"


@pytest.mark.parametrize("kind,analysis,allowed", [
    ("slide", None, False), ("patch", None, True), ("patch", {}, False),
])
def test_slide_encoder_provenance_is_checked_without_changing_legacy_patch_policy(
    registry, monkeypatch, kind, analysis, allowed,
):
    predictors, cohort = registry
    selection, *_ = candidate(predictors)
    predictor, _ = freeze(predictors, selection)
    service = EvaluationRunService(predictors.store, predictors.filesystem)
    model = deepcopy(predictor)
    model["manifest"]["recipe"]["analysis"] = analysis
    development = model["manifest"]["inputs"]["features"]
    if kind == "slide":
        development["featureKind"] = "slide"
    development["extraction"] = {"spec": {"options": {"slide_encoder": "titan"}}}
    external = deepcopy(development)
    external["feature"] = {"id": "external-feature", "contentHash": "external-hash"}
    external["extraction"]["spec"]["options"]["slide_encoder"] = "prism"
    monkeypatch.setattr(service.predictors, "get", lambda _: model)
    monkeypatch.setattr("histopilot.application.evaluation_runs.feature_contract",
                        lambda *_: deepcopy(external))
    result = service.preview(EvaluationRunSelection(
        predictorId=predictor["id"], cohortId=cohort["id"], name="Encoder provenance",
    ))
    assert result["canSave"] is allowed
    if not allowed:
        assert result["findings"][0]["code"] == "EVALUATION_EXTRACTION_MISMATCH"


@pytest.mark.parametrize("mode,model,compatible", [
    ("clinical", "abmil", True), ("image", "slide_linear", True),
    ("multimodal", "slide_mlp", True), ("image", "abmil", False),
])
def test_slide_study_preflight_accepts_clinical_arm_and_rejects_patch_models(
    preview_context, mode, model, compatible,
):
    recipe = TrainingRecipe(
        model=model, inputMode=mode, checkpointMetric="validation_loss",
        clinicalFields=[] if mode == "image" else [{"field": "age", "kind": "numeric"}],
    ).model_dump()
    context = preview_context
    findings = context.service._training_control_findings(
        [recipe], context.protocol, development_plans(context.protocol), feature_kind="slide",
    )
    assert (not any(row["severity"] == "error" for row in findings)) is compatible
    if not compatible:
        assert findings[0]["code"] == "TRAINING_FEATURE_KIND_MISMATCH"
    # Even legacy ABMIL recipes without explicit analysis must be blocked.
    recipe["analysis"] = None
    if not compatible:
        findings = context.service._training_control_findings(
            [recipe], context.protocol, development_plans(context.protocol), feature_kind="slide",
        )
        assert findings[0]["code"] == "TRAINING_FEATURE_KIND_MISMATCH"


@pytest.mark.parametrize("study", ["slide"], indirect=True)
def test_original_image_geometry_and_review_do_not_need_patch_coordinates(study):
    service, request, _, _ = study
    quality = service.quality(request.datasetId, "a", request.featureBundleId)
    assert quality["featureKind"] == "slide"
    assert quality["width"] == quality["height"] == 64
    assert quality["patchCount"] is None and quality["patches"] == []
    assert quality["coordinateBounds"] is None
    assert service.image(request.datasetId, "a").startswith(b"\x89PNG")
    assert service.image(request.datasetId, "a", region=(4, 8, 16, 20)).startswith(b"\x89PNG")
    reviews = SlideReviewService(service.store)
    saved = reviews.save(request.datasetId, "a", SaveSlideReview(
        expectedRevision=0, status="review", notes="Review region",
        regions=[{"id": "region", "x": 4, "y": 8, "width": 16, "height": 20}],
    ))
    assert saved["revision"] == 1 and saved["regions"][0]["x"] == 4
    assert reviews._read(request.datasetId, "a")["regions"] == saved["regions"]
    for action in (
        lambda: service.build(request),
        lambda: service.patch_region(request.datasetId, "a", request.featureBundleId, 0),
    ):
        with pytest.raises(StorageError) as error:
            action()
        assert error.value.code == "MORPHOLOGY_PATCH_FEATURES_REQUIRED"


@pytest.mark.parametrize("model,mode,supported", [
    ("slide_linear", "image", False), ("slide_mlp", "multimodal", False),
    ("abmil", "clinical", False), ("abmil", "image", True), ("nnmil", "image", True),
])
def test_case_review_exposes_actual_attention_capability(cases, model, mode, supported):
    service, evaluation, dataset, source, _, install = cases
    predictor = service.store.publish_configuration(
        manifest={"kind": "frozen-predictor", "datasetId": dataset["id"],
                  "target": evaluation["manifest"]["target"],
                  "recipe": {"model": model, "inputMode": mode}}, operation_id="new-model",
    )
    other = service.store.publish_configuration(
        manifest={**evaluation["manifest"], "predictorId": predictor["id"],
                  "predictor": reference(predictor)}, operation_id="new-evaluation",
    )
    install(other, source)
    page = service.query(other["id"], CaseReviewQuery(comparisonId=evaluation["id"]))
    assert page["supportsAttention"] is supported
    assert page["comparison"]["supportsAttention"] is True
