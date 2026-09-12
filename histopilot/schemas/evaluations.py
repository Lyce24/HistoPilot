"""Later test cohorts inherit development targets and never assign data splits."""

from typing import Annotated, Literal

from pydantic import Field, StrictInt, model_validator

from histopilot.schemas.protocols import (
    Conditions,
    ProtocolFreezeRequest,
    ProtocolPreviewRequest,
    TargetSpec,
)
from histopilot.schemas.workspace import RequestModel

ConfigurationId = Annotated[str, Field(pattern=r"^configuration-[a-f0-9]{64}$")]


class InferenceSettings(RequestModel):
    loadingPolicy: Literal["per_slide", "packed"] = "per_slide"
    packArtifactId: Annotated[str, Field(pattern=r"^pack-[a-f0-9]{64}$")] | None = None
    batchSize: Annotated[StrictInt, Field(ge=1, le=1024)] = 1
    numWorkers: Annotated[StrictInt, Field(ge=0, le=64)] = 0
    device: Literal["auto", "cpu", "cuda"] = "auto"
    precision: Literal["float32", "float16", "bfloat16"] = "float32"
    patientAggregation: Literal["mean", "max"] = "mean"
    decisionThreshold: float = Field(default=0.5, gt=0, lt=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def loading_contract(self):
        if (self.loadingPolicy == "packed") != (self.packArtifactId is not None):
            raise ValueError(
                "Packed loading requires a selected pack; per-slide loading does not use a pack."
            )
        if self.device == "cpu" and self.precision == "float16":
            raise ValueError("Use float32 or bfloat16 for CPU inference.")
        return self


class EvaluationSpec(RequestModel):
    protocolId: ConfigurationId
    developmentFeatureBundleId: ConfigurationId
    datasetId: str = Field(pattern=r"^dataset-[a-f0-9]{64}$")
    featureBundleId: ConfigurationId
    target: TargetSpec | None = None
    eligibility: Conditions = Field(default_factory=list)
    patientIdentifiers: Literal["shared", "independent"] = "shared"
    inference: InferenceSettings = Field(default_factory=InferenceSettings)


class EvaluationPreviewRequest(ProtocolPreviewRequest):
    pass


class EvaluationFreezeRequest(ProtocolFreezeRequest):
    pass
