"""Independent test membership and targets, with inference selected at evaluation."""

from typing import Annotated, Literal

from pydantic import Field, StrictInt, field_validator, model_serializer, model_validator

from histopilot.resolvers import INFERENCE_PURPOSES
from histopilot.schemas.protocols import (
    Conditions,
    ProtocolFreezeRequest,
    ProtocolPreviewRequest,
    TargetSpec,
)
from histopilot.schemas.workspace import RequestModel

ConfigurationId = Annotated[str, Field(pattern=r"^configuration-[a-f0-9]{64}$")]


def is_inference_purpose(value):
    return value in INFERENCE_PURPOSES


# Validation context for specs and selections read back from saved records.
STORED = {"stored": True}


class InferenceSettings(RequestModel):
    loadingPolicy: Literal["per_slide", "packed"] = "per_slide"
    packArtifactId: Annotated[str, Field(pattern=r"^pack-[a-f0-9]{64}$")] | None = None
    batchSize: Annotated[StrictInt, Field(ge=1, le=1024)] = 1
    numWorkers: Annotated[StrictInt, Field(ge=0, le=64)] = 0
    device: Literal["auto", "cpu", "cuda"] = "auto"
    precision: Literal["float32", "float16", "bfloat16"] = "float32"
    # "max" is retired: frozen predictors score patients by mean probabilities or mean
    # logits. Test cohorts saved with it stay readable (their spec feeds their content
    # hash), so only records read back with the STORED context may carry it.
    patientAggregation: Literal["mean", "mean_logits", "predictor", "max"] = "mean"
    decisionThreshold: (
        Annotated[float, Field(gt=0, lt=1, allow_inf_nan=False)] | Literal["predictor"]
    ) = 0.5

    @field_validator("patientAggregation")
    @classmethod
    def maximum_is_retired(cls, value, info):
        if value == "max" and not (info.context or {}).get("stored"):
            raise ValueError(
                "Maximum-probability patient aggregation is not supported. "
                "Choose mean probabilities or mean logits, the rules frozen predictors use."
            )
        return value

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
    splitUnit: Literal["slide", "patient"] = "patient"
    purpose: Literal["independent", "inference", "review"] = "independent"

    @model_serializer(mode="wrap")
    def preserve_legacy_purpose(self, handler):
        serialized = handler(self)
        if self.purpose == "independent":
            serialized.pop("purpose", None)
        if "splitUnit" not in self.model_fields_set:
            serialized.pop("splitUnit", None)
        if self.sourceTargetSplitId is None:
            serialized.pop("sourceTargetSplitId", None)
        return serialized

    @model_validator(mode="after")
    def inference_is_unlabeled(self):
        if (
            "splitUnit" in self.model_fields_set
            and self.target is not None
            and self.target.unit != self.splitUnit
        ):
            raise ValueError("The evaluation target must use the selected split unit.")
        if self.purpose == "inference" and self.target is not None:
            raise ValueError("Inference cohorts are unlabeled. Remove the prediction target.")
        if self.purpose == "review" and (
            self.target is not None or self.patientIdentifiers != "shared"
        ):
            raise ValueError(
                "Review predictions require unlabeled slides and shared patient identifiers."
            )
        return self

    # Keep old bindings readable; new cohorts have no development or feature dependencies.
    sourceTargetSplitId: ConfigurationId | None = None
    protocolId: ConfigurationId | None = None
    developmentFeatureBundleId: ConfigurationId | None = None
    datasetId: str = Field(pattern=r"^dataset-[a-f0-9]{64}$")
    datasetIds: list[Annotated[str, Field(pattern=r"^dataset-[a-f0-9]{64}$")]] | None = Field(
        default=None, min_length=1, max_length=64
    )
    featureBundleId: ConfigurationId | None = None
    target: TargetSpec | None = None
    eligibility: Conditions = Field(default_factory=list)
    patientIdentifiers: Literal["shared", "independent"] = "shared"
    inference: InferenceSettings = Field(default_factory=InferenceSettings)

    @field_validator("protocolId", "developmentFeatureBundleId", "featureBundleId", mode="before")
    @classmethod
    def empty_binding(cls, value):
        return None if value == "" else value

    @model_validator(mode="after")
    def dataset_selection(self):
        if self.sourceTargetSplitId and (
            self.purpose not in {"independent", "inference"}
            or self.eligibility
            or (self.purpose == "independent" and self.target is None)
            or (self.datasetIds is not None and self.datasetIds != [self.datasetId])
        ):
            raise ValueError(
                "A frozen testing partition uses its exact dataset rows without additional filters, with evaluation labels or unlabeled inference."
            )
        if self.datasetIds is not None:
            if len(set(self.datasetIds)) != len(self.datasetIds):
                raise ValueError("Select each test dataset once.")
            if self.datasetIds[0] != self.datasetId:
                raise ValueError("The primary test dataset must be the first selected dataset.")
        if self.protocolId and not (self.developmentFeatureBundleId and self.featureBundleId):
            raise ValueError("Legacy cohort bindings require both feature bundles.")
        return self


class EvaluationPreviewRequest(ProtocolPreviewRequest):
    pass


class EvaluationFreezeRequest(ProtocolFreezeRequest):
    pass
