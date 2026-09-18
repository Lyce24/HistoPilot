"""Train-only patient transforms preserve missingness and never learn external categories."""

import copy

import pytest

from histopilot.clinical_features import (
    clinical_rows,
    fit_clinical_preprocessor,
    transform_clinical,
)
from histopilot.schemas.development import TrainingRecipe

FIELDS = [{"field": "age", "kind": "numeric"}, {"field": "site", "kind": "categorical"}]


def evidence():
    rows = [
        {"slideId": slide, "patientId": patient, "partition": "train"}
        for slide, patient in [("a1", "a"), ("a2", "a"), ("b", "b"), ("c", "c")]
    ]
    values = {
        "a1": {"age": 20, "site": "A"},
        "a2": {"age": "20", "site": "A"},
        "b": {"age": 40, "site": "B"},
        "c": {"age": None, "site": None},
        "external": {"age": 10000, "site": "Z"},
    }
    return rows, values


def test_preprocessing_uses_unique_training_patients_only_and_freezes_unseen_categories():
    rows, values = evidence()
    fitted = fit_clinical_preprocessor(rows, values, FIELDS)
    assert fitted["trainingPatientCount"] == 3
    assert fitted["columns"][0]["median"] == fitted["columns"][0]["mean"] == 30
    assert fitted["columns"][1]["categories"] == ['"A"', '"B"']
    baseline = copy.deepcopy(fitted)
    missing, unknown = transform_clinical([values["c"], values["external"]], fitted)
    assert missing[:2] == [0, 1]
    assert missing[2:] == [0, 0, 1, 0]
    assert unknown[2:] == [0, 0, 0, 1]
    assert fitted == baseline
    values["external"] = {"age": -9999, "site": "EXTERNAL_CHANGED"}
    assert fit_clinical_preprocessor(rows, values, FIELDS) == fitted


def test_preprocessing_rejects_validation_rows_patient_conflicts_and_absent_columns():
    rows, values = evidence()
    with pytest.raises(ValueError, match="training partition"):
        fit_clinical_preprocessor([{**rows[0], "partition": "val"}], values, FIELDS)
    values["a2"]["age"] = 99
    with pytest.raises(ValueError, match="conflicting clinical"):
        clinical_rows(rows, values, FIELDS)
    del values["a2"]["age"]
    with pytest.raises(ValueError, match="lacks a declared"):
        clinical_rows(rows, values, FIELDS)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "broken", True, 1e40])
def test_numeric_inputs_fail_with_field_and_slide_context(value):
    rows, values = evidence()
    values["a1"]["age"] = value
    with pytest.raises(ValueError, match="a1, age"):
        fit_clinical_preprocessor(rows, values, FIELDS)


def test_historical_image_recipe_shape_and_explicit_clinical_schema():
    old = TrainingRecipe().model_dump()
    assert "inputMode" not in old and "clinicalFields" not in old
    recipe = TrainingRecipe(inputMode="multimodal", clinicalFields=FIELDS)
    assert TrainingRecipe.model_validate_json(recipe.model_dump_json()) == recipe
    for args in (
        {"inputMode": "clinical"},
        {"clinicalFields": FIELDS},
        {"inputMode": "clinical", "clinicalFields": FIELDS * 2},
    ):
        with pytest.raises(ValueError):
            TrainingRecipe(**args)
