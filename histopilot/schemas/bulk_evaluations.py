"""Explicit test-cohort batch scope and a reviewed, fixed predictor snapshot."""

from typing import Literal

from pydantic import Field, field_validator, model_validator

from histopilot.resolvers import MAX_APPLY_PREDICTORS
from histopilot.schemas.evaluations import ConfigurationId, InferenceSettings
from histopilot.schemas.workspace import RequestModel


class BulkEvaluationSelection(RequestModel):
    cohortId: ConfigurationId
    scope: Literal["all", "selected"] = "all"
    predictorIds: list[ConfigurationId] | None = Field(
        default=None, max_length=MAX_APPLY_PREDICTORS
    )
    namePrefix: str = Field(default="Evaluation", min_length=1, max_length=60)
    featureBundleId: ConfigurationId | None = None
    inference: InferenceSettings | None = None
    patientIdentifiers: Literal["shared", "independent"] | None = None

    @field_validator("namePrefix")
    @classmethod
    def readable_name(cls, value):
        if not value.strip():
            raise ValueError("Enter an evaluation batch name.")
        return value.strip()

    @model_validator(mode="after")
    def explicit_scope(self):
        if self.predictorIds is not None and "scope" not in self.model_fields_set:
            raise ValueError(
                "State scope: selected to apply only these predictors; scope: all takes no IDs."
            )
        if self.scope == "selected" and not self.predictorIds:
            raise ValueError("Select at least one predictor.")
        if self.scope == "all" and self.predictorIds is not None:
            raise ValueError("All predictors are captured at review; do not supply selected IDs.")
        if self.predictorIds and len(set(self.predictorIds)) != len(self.predictorIds):
            raise ValueError("Predictor IDs must be distinct.")
        return self


class RunBulkEvaluation(BulkEvaluationSelection):
    reviewedPredictorIds: list[ConfigurationId] = Field(
        min_length=1, max_length=MAX_APPLY_PREDICTORS
    )
    previewHash: str = Field(pattern=r"^[a-f0-9]{64}$")
    operationId: str = Field(min_length=1, max_length=128)

    @field_validator("reviewedPredictorIds")
    @classmethod
    def exact_snapshot(cls, value):
        if len(set(value)) != len(value):
            raise ValueError("Reviewed predictor IDs must be distinct.")
        return sorted(value)


def inference_unstated(selection: BulkEvaluationSelection) -> list[str]:
    """Inference settings a request left out. Their defaults differ from Apply models, which
    applies each predictor's own patient aggregation and threshold ("predictor")."""
    if selection.inference is None:
        return []
    return [
        name
        for name in InferenceSettings.model_fields
        if name not in selection.inference.model_fields_set
    ]
