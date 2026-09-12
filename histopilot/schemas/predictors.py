"""Reviewed promotion of complete development candidates and evaluation plans."""

from typing import Annotated, Literal

from pydantic import Field, StrictInt, field_validator

from histopilot.schemas.development import ResourcePolicy
from histopilot.schemas.evaluations import ConfigurationId, InferenceSettings
from histopilot.schemas.workspace import RequestModel


class PredictorSelection(RequestModel):
    experimentId: str = Field(min_length=1, max_length=160)
    batchId: ConfigurationId
    candidateId: str = Field(pattern=r"^candidate-[a-f0-9]{64}$")
    trainingSeed: Annotated[StrictInt, Field(ge=0, le=2**32 - 1)]
    splitSeed: Annotated[StrictInt, Field(ge=0, le=2**32 - 1)]
    name: str = Field(min_length=1, max_length=120)
    method: Literal["ensemble", "refit"] = "ensemble"
    refitPercentile: float = Field(default=50.0, ge=1, le=100, allow_inf_nan=False, strict=True)

    @field_validator("name")
    @classmethod
    def nonempty_name(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Enter a predictor name.")
        return value


class FreezePredictor(PredictorSelection):
    previewHash: str = Field(pattern=r"^[a-f0-9]{64}$")
    operationId: str = Field(min_length=1, max_length=128)


class PredictorAction(RequestModel):
    operationId: str = Field(min_length=1, max_length=128)


class LaunchRefit(PredictorAction):
    resources: ResourcePolicy | None = None


class PredictorSourceSelection(RequestModel):
    experimentId: str = Field(min_length=1, max_length=160)
    batchId: ConfigurationId
    candidateId: str = Field(pattern=r"^candidate-[a-f0-9]{64}$")
    trainingSeed: Annotated[StrictInt, Field(ge=0, le=2**32 - 1)]
    splitSeed: Annotated[StrictInt, Field(ge=0, le=2**32 - 1)]


class PredictorBuildSelection(RequestModel):
    selections: list[PredictorSourceSelection] = Field(min_length=1, max_length=128)
    method: Literal["ensemble", "refit", "both"] = "both"
    refitPercentile: float = Field(default=50.0, ge=1, le=100, allow_inf_nan=False, strict=True)
    namePrefix: str | None = Field(default=None, min_length=1, max_length=100)

    @field_validator("selections")
    @classmethod
    def distinct_sources(cls, values):
        keys = [tuple(row.model_dump().values()) for row in values]
        if len(set(keys)) != len(keys):
            raise ValueError("Choose each configuration and seed group once.")
        return values

    @field_validator("namePrefix")
    @classmethod
    def readable_prefix(cls, value):
        if value is not None and not value.strip():
            raise ValueError("Enter a predictor name prefix.")
        return value.strip() if value is not None else None


class ApplyPredictorBuilds(PredictorBuildSelection):
    previewHash: str = Field(pattern=r"^[a-f0-9]{64}$")
    operationId: str = Field(min_length=1, max_length=128)


class EvaluationRunSelection(RequestModel):
    predictorId: ConfigurationId
    cohortId: ConfigurationId
    name: str = Field(min_length=1, max_length=120)
    featureBundleId: ConfigurationId | None = None
    inference: InferenceSettings | None = None
    patientIdentifiers: Literal["shared", "independent"] | None = None

    @field_validator("name")
    @classmethod
    def nonempty_name(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Enter an evaluation name.")
        return value


class SaveEvaluationRun(EvaluationRunSelection):
    previewHash: str = Field(pattern=r"^[a-f0-9]{64}$")
    operationId: str = Field(min_length=1, max_length=128)
