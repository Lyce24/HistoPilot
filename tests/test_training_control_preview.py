"""Data-dependent experimental controls are reviewable before batch freezing."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from histopilot.application.development import DevelopmentService
from histopilot.schemas.development import DevelopmentBatchSpec
from histopilot.storage.project_lock import StorageError


@pytest.fixture
def preview_context(tmp_path, monkeypatch):
    target = {
        "task": "binary_classification",
        "unit": "patient",
        "classes": ["negative", "positive"],
        "positiveClass": "positive",
    }
    examples = [
        ("a0", "negative", "train", "A"),
        ("a1", "positive", "train", "A"),
        ("b0", "negative", "train", "B"),
        ("b1", "positive", "train", "B"),
        ("v0", "negative", "val", "A"),
        ("v1", "positive", "val", "B"),
        ("t0", "negative", "test", "A"),
        ("t1", "positive", "test", "B"),
    ]
    rows = [
        {
            "slideId": slide,
            "patientId": slide,
            "label": label,
            "partition": role,
            "fold": fold,
            "seed": 42,
            "phase": "evaluation",
            "pool": "development",
        }
        for fold in range(2)
        for slide, label, role, _ in examples
    ]
    records = [
        {
            "slideId": slide,
            "patientId": slide,
            "attributes": {
                "cohort": cohort,
                "hospital": f"site-{cohort}",
            },
        }
        for slide, _, _, cohort in examples
    ]
    protocol = {
        "datasetId": "frozen-dataset",
        "memberships": rows,
        "spec": {"target": target, "split": {"version": 4, "mode": "kfold"}},
    }
    store = SimpleNamespace(
        folder=tmp_path,
        get_configuration=lambda _identity: {"manifest": deepcopy(protocol)},
        configuration_publication=lambda _operation: None,
    )
    dataset_reads = []

    def load_dataset(_service, dataset_id):
        dataset_reads.append(dataset_id)
        return {}, {}, deepcopy(records)

    monkeypatch.setattr(
        "histopilot.application.development.MILInputService.preview",
        lambda *_: {
            "canPlan": True,
            "findings": [],
        },
    )
    monkeypatch.setattr(
        "histopilot.application.development.ProtocolService._load_dataset", load_dataset
    )
    spec = DevelopmentBatchSpec(
        experimentName="Experimental controls",
        batchName="Review",
        inputs={"protocolId": "protocol", "featureBundleId": "bundle"},
    )
    return SimpleNamespace(
        service=DevelopmentService(store, None),
        spec=spec,
        protocol=protocol,
        records=records,
        dataset_reads=dataset_reads,
    )


def with_recipe(context, **recipe):
    return DevelopmentBatchSpec.model_validate(
        {
            **context.spec.model_dump(),
            "recipe": {**context.spec.recipe.model_dump(), **recipe},
        }
    )


def test_legacy_preview_hash_and_findings_remain_identical(preview_context, monkeypatch):
    context = preview_context
    # Legacy AUROC recipes intentionally retain their original preview behavior.
    spec = DevelopmentBatchSpec.model_validate({
        **with_recipe(context, checkpointMetric="validation_auroc").model_dump(),
        "recipe": {"checkpointMetric": "validation_auroc"},
        "selectionMetric": None, "candidateSelection": None,
    }, context={"legacy": True})
    for row in context.protocol["memberships"]:
        if row["partition"] == "val":
            row["label"] = "negative"
    current = context.service.preview(spec)
    monkeypatch.setattr(context.service, "_training_control_findings", lambda *_, **__: [])
    previous = context.service.preview(spec)
    assert current == previous
    assert current["canFreeze"]
    assert not context.dataset_reads


def test_incompatible_weight_count_is_deduplicated_and_blocks_freeze(preview_context):
    context = preview_context
    spec = with_recipe(context, classWeights=[1, 2, 3])
    spec = DevelopmentBatchSpec.model_validate(
        {
            **spec.model_dump(),
            "mode": "grid",
            "grid": {"learningRates": [0.0001, 0.0002], "weightDecays": [0.005], "maxEpochs": [40]},
        }
    )
    preview = context.service.preview(spec)
    assert not preview["canFreeze"] and not preview["runs"]
    assert len(preview["findings"]) == 1
    finding = preview["findings"][0]
    assert finding["code"] == "TRAINING_RECIPE_UNAVAILABLE"
    assert "one class weight" in finding["message"]
    assert "Affected configurations: 2; folds: 2" in finding["message"]
    with pytest.raises(StorageError) as error:
        context.service.freeze(spec, preview["previewHash"], "invalid", {"tag": "Invalid"})
    assert error.value.code == "BATCH_PREFLIGHT_BLOCKED"


def test_cohort_columns_are_loaded_once_and_checked_per_recipe(preview_context):
    context = preview_context
    spec = DevelopmentBatchSpec.model_validate(
        {
            **context.spec.model_dump(),
            "mode": "explicit",
            "configurations": [
                {"samplingStrategy": "cohort_balanced"},
                {"samplingStrategy": "cohort_label_balanced", "cohortColumn": "hospital"},
            ],
        }
    )
    preview = context.service.preview(spec)
    assert preview["canFreeze"], preview["findings"]
    assert len(preview["runs"]) == 4
    assert context.dataset_reads == ["frozen-dataset"]
    assert all("cohort" not in row for row in context.protocol["memberships"])
    spec = with_recipe(context, samplingStrategy="cohort_balanced", cohortColumn="missing")
    invalid = context.service.preview(spec)
    assert not invalid["canFreeze"]
    assert "nonempty cohort" in invalid["findings"][0]["message"]


def test_cohort_label_cells_require_both_classes_before_freeze(preview_context):
    context = preview_context
    for row in context.protocol["memberships"]:
        if row["slideId"] == "b1":
            row["label"] = "negative"
    preview = context.service.preview(
        with_recipe(context, samplingStrategy="cohort_label_balanced")
    )
    assert not preview["canFreeze"]
    assert "both binary classes" in preview["findings"][0]["message"]


@pytest.mark.parametrize(
    "recipe", [{"classWeighting": "inverse_prevalence"}, {"classWeightedSampling": True}]
)
def test_missing_training_class_is_reported_before_freeze(preview_context, recipe):
    context = preview_context
    for row in context.protocol["memberships"]:
        if row["partition"] == "train":
            row["label"] = "negative"
    preview = context.service.preview(with_recipe(context, **recipe))
    assert not preview["canFreeze"]
    assert "every target class in training" in preview["findings"][0]["message"]


def test_class_checks_do_not_borrow_labels_from_another_fold(preview_context):
    context = preview_context
    for row in context.protocol["memberships"]:
        if row["fold"] == 0 and row["partition"] == "train" and row["label"] == "positive":
            row["partition"] = "val"
    preview = context.service.preview(with_recipe(context, classWeighting="inverse_prevalence"))
    assert not preview["canFreeze"]
    assert len(preview["findings"]) == 1
    assert "Affected configurations: 1; folds: 1" in preview["findings"][0]["message"]


def test_bce_multiclass_and_unsupported_models_are_reported(preview_context):
    context = preview_context
    context.protocol["spec"]["target"]["task"] = "multiclass_classification"
    context.protocol["spec"]["target"]["classes"].append("other")
    invalid = context.service.preview(with_recipe(context, lossType="bce"))
    assert not invalid["canFreeze"]
    assert "BCE requires a binary" in invalid["findings"][0]["message"]
    invalid = context.service.preview(with_recipe(context, model="transmil"))
    assert not invalid["canFreeze"]
    assert invalid["findings"][0]["code"] == "TRAINING_MODEL_UNSUPPORTED"


def test_positive_fallback_is_a_reviewable_warning_and_explicit_budget_works(preview_context):
    context = preview_context
    preview = context.service.preview(
        with_recipe(
            context,
            checkpointMetric="validation_auroc",
            minValidationPositives=3,
            fixedEpochBudget=12,
        )
    )
    assert preview["canFreeze"]
    warnings = preview["findings"]
    assert len(warnings) == 1 and warnings[0]["severity"] == "warning"
    assert "1 positive patient (minimum 3)" in warnings[0]["message"]
    assert (
        "12 epochs" in warnings[0]["message"] and "final epoch checkpoint" in warnings[0]["message"]
    )
    assert "folds: 2" in warnings[0]["message"]
    explicit = context.service.preview(with_recipe(context, fixedEpochBudget=7))
    assert explicit["canFreeze"]
    assert "7 epochs" in explicit["findings"][0]["message"]
