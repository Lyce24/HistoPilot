"""Typed commands for persistent model-development records, before inputs exist."""

from typing import Annotated, Literal

from pydantic import Field, StrictInt, field_validator, model_validator

from histopilot.schemas.development import DevelopmentBatchSpec
from histopilot.schemas.mil import MILInputSpec
from histopilot.schemas.predictor_policy import ExperimentPredictorPolicy
from histopilot.schemas.protocols import SplitSpec
from histopilot.schemas.workspace import RequestModel


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
    setupVersion: Literal[1] | None = None
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


class ConfigureModelExperimentSetup(RequestModel):
    expectedRevision: Annotated[StrictInt, Field(ge=1, le=2**63 - 1)]
    datasetId: str = Field(min_length=1, max_length=128)
    targetSplitId: str = Field(min_length=1, max_length=128)
    featureBundleId: str = Field(min_length=1, max_length=128)
    loadingPolicy: Literal["auto", "native", "mmap"] = "auto"
    packArtifactId: str | None = Field(default=None, pattern=r"^pack-[a-f0-9]{64}$")
    trainingSplit: SplitSpec

    @model_validator(mode="after")
    def coherent_inputs(self):
        if self.trainingSplit.version != 4:
            raise ValueError(
                "Experimental setup requires a development-only training split (version 4)."
            )
        MILInputSpec(
            protocolId="derived",
            featureBundleId=self.featureBundleId,
            loadingPolicy=self.loadingPolicy,
            packArtifactId=self.packArtifactId,
        )
        return self


class FreezeModelExperimentSetup(SubmitModelExperiment):
    pass
