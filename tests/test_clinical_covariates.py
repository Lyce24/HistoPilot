"""Clinical covariates come from the frozen dataset, per design unit, without label leakage."""

import pytest

from histopilot.application import clinical_inputs
from histopilot.application.clinical_inputs import clinical_field_choices, excluded_clinical_fields
from histopilot.clinical_features import (
    clinical_rows,
    fit_clinical_preprocessor,
    label_separation,
)
from histopilot.schemas.training_controls import validate_split_unit

FIELDS = [{"field": "site", "kind": "categorical"}, {"field": "age", "kind": "numeric"}]


def rows(patients=3, slides_per_patient=2):
    return [
        {
            "slideId": f"s{patient}{slide}",
            "patientId": f"p{patient}",
            "patientIdSource": "mapped",
            "partition": "train",
            "label": "high" if patient % 2 else "low",
        }
        for patient in range(patients)
        for slide in range(slides_per_patient)
    ]


def test_slide_designs_allow_per_slide_covariates_and_count_slides():
    memberships = rows()
    # Slides of one patient differ, and one has no verified patient.
    values = {row["slideId"]: {"site": row["slideId"][-1], "age": 50} for row in memberships}
    memberships[0] = {**memberships[0], "patientId": None}
    with pytest.raises(ValueError, match="verified patient"):
        clinical_rows(memberships, values, FIELDS)
    assert len(clinical_rows(memberships, values, FIELDS, unit="slide")) == 6
    fitted = fit_clinical_preprocessor(memberships, values, FIELDS, unit="slide")
    assert fitted["method"] == "training_slide_median_standardize_onehot_v1"
    assert fitted["trainingSlideCount"] == 6
    assert "trainingPatientCount" not in fitted
    assert fitted["columns"][0]["observedSlides"] == 6


def test_patient_designs_keep_their_receipt_keys():
    memberships = rows()
    values = {
        row["slideId"]: {"site": "A", "age": 40 + int(row["patientId"][1])} for row in memberships
    }
    fitted = fit_clinical_preprocessor(memberships, values, FIELDS)
    assert fitted["method"] == "training_patient_median_standardize_onehot_v1"
    assert fitted["trainingPatientCount"] == 3
    assert {"observedPatients"} <= set(fitted["columns"][0])


def test_slide_designs_accept_clinical_inputs():
    target = {"unit": "slide"}
    validate_split_unit({"inputMode": "multimodal", "clinicalFields": FIELDS}, target, "slide")


class Store:
    def __init__(self, source):
        self.source = source

    def get_configuration(self, _identity):
        return {"manifest": {"spec": self.source}}


def protocol():
    return {
        "datasetId": "dataset",
        "spec": {
            "target": {"field": "grade"},
            "sourceTargetSplitId": "split",
            "splitUnit": "slide",
        },
        "memberships": [],
    }


def test_fields_that_define_the_label_or_split_are_refused():
    source = {
        "testTarget": {"field": "test_grade"},
        "split": {
            "partitionField": "requested_split",
            "trainRules": [{"conditions": [{"field": "cohort", "op": "eq", "value": "A"}]}],
            "testRules": [{"field": "site", "op": "eq", "value": "B"}],
        },
    }
    excluded = excluded_clinical_fields(Store(source), protocol())
    assert set(excluded) == {"grade", "test_grade", "requested_split", "cohort", "site"}
    assert "prediction target" in excluded["grade"]


def test_choices_offer_typed_dataset_columns_and_explain_refusals(monkeypatch):
    dictionary = {
        "grade": {"key": "grade", "type": "categorical", "owner": "slide"},
        "age": {"key": "age", "type": "integer", "owner": "patient"},
        "biopsy_date": {"key": "biopsy_date", "type": "date", "owner": "slide"},
    }
    monkeypatch.setattr(
        clinical_inputs.ProtocolService, "_load_dataset", lambda _self, _id: ({}, dictionary, [])
    )
    choices = {row["field"]: row for row in clinical_field_choices(Store({}), None, protocol())}
    assert set(choices) == {"grade", "age"}  # dates are not offered
    assert (choices["age"]["kind"], choices["age"]["owner"], choices["age"]["excluded"]) == (
        "numeric",
        "patient",
        None,
    )
    assert choices["grade"]["excluded"]


def test_label_separation_flags_a_restated_label_but_not_noise():
    memberships = rows(patients=20, slides_per_patient=1)
    restated = {row["slideId"]: {"reader": row["label"]} for row in memberships}
    noise = {
        row["slideId"]: {"reader": "x" if index % 3 else "y"}
        for index, row in enumerate(memberships)
    }
    item = {"field": "reader", "kind": "categorical"}
    assert label_separation(memberships, restated, item) == 1.0
    assert label_separation(memberships, noise, item) < 0.8
    numeric = {
        row["slideId"]: {"score": 1.0 if row["label"] == "high" else 0.0} for row in memberships
    }
    assert (
        label_separation(memberships, numeric, {"field": "score", "kind": "numeric"}, "high") == 1.0
    )
