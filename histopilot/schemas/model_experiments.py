"""Typed commands for persistent model-development records, before inputs exist."""

from typing import Annotated, Literal

from pydantic import Field, StrictInt, field_validator, model_validator

from histopilot.schemas.development import DevelopmentBatchSpec
from histopilot.schemas.mil import MILInputSpec
from histopilot.schemas.workspace import RequestModel


class ExperimentPredictorPolicy(RequestModel):
    method: Literal["skip", "ensemble", "refit", "both"] = "ensemble"
    refitPercentile: float | None = Field(
        default=None, ge=1, le=100, allow_inf_nan=False, strict=True
    )

    @model_validator(mode="after")
    def chosen_epoch_policy(self):
        if self.method in {"refit", "both"} and self.refitPercentile is None:
            raise ValueError("Choose the best-checkpoint epoch percentile for refit training.")
        if self.method in {"skip", "ensemble"} and self.refitPercentile is not None:
            raise ValueError("This predictor choice does not use a refit epoch percentile.")
        return self


class ExperimentValues(RequestModel):
    name: str = Field(min_length=1, max_length=120)
    notes: str = Field(default="", max_length=10000)
    tags: list[str] = Field(default_factory=list, max_length=32)
    inputs: MILInputSpec | None = None
    predictorPolicy: ExperimentPredictorPolicy = Field(default_factory=ExperimentPredictorPolicy)

    @field_validator("name")
    @classmethod
    def readable_name(cls, value):
        if not value.strip():
            raise ValueError("Enter an experiment name.")
        return value.strip()

    @field_validator("tags")
    @classmethod
    def distinct_tags(cls, values):
        values = [value.strip() for value in values]
        if any(not value or len(value) > 80 for value in values):
            raise ValueError("Tags must contain between 1 and 80 characters.")
        if len(set(values)) != len(values):
            raise ValueError("Experiment tags must be distinct.")
        return values


class ExperimentBatchPlan(RequestModel):
    id: str = Field(min_length=1, max_length=128, pattern=r"^[\w.-]+$")
    spec: DevelopmentBatchSpec


class CreateModelExperiment(ExperimentValues):
    operationId: str = Field(min_length=1, max_length=200)
    sourceExperimentId: str | None = Field(default=None, min_length=1, max_length=128)


class UpdateModelExperiment(ExperimentValues):
    expectedRevision: Annotated[StrictInt, Field(ge=1, le=2**63 - 1)]
    batchPlans: list[ExperimentBatchPlan] = Field(default_factory=list, max_length=100)

    @field_validator("batchPlans")
    @classmethod
    def unique_plans(cls, values):
        if len({value.id for value in values}) != len(values):
            raise ValueError("Batch plan IDs must be distinct.")
        return values


class SubmitModelExperiment(RequestModel):
    expectedRevision: Annotated[StrictInt, Field(ge=1, le=2**63 - 1)]
    operationId: str = Field(min_length=1, max_length=200)
