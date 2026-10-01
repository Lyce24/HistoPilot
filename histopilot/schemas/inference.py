"""Label-free inference analysis, exports and attention requests."""

from typing import Literal

from pydantic import Field, field_validator, model_validator

from histopilot.schemas.development import ResourcePolicy
from histopilot.schemas.evaluations import ConfigurationId
from histopilot.schemas.interpretation import Dimension
from histopilot.schemas.workspace import RequestModel

Unit = Literal["selected", "slide", "patient"]


class InferenceSummaryQuery(RequestModel):
    unit: Unit = "selected"
    attribute: str | None = Field(default=None, min_length=1, max_length=200)
    comparisonId: ConfigurationId | None = None


class InferenceExportQuery(RequestModel):
    unit: Unit = "selected"
    # None exports every frozen attribute; an empty list exports predictions only.
    attributes: list[str] | None = Field(default=None, max_length=200)

    @field_validator("attributes")
    @classmethod
    def distinct_attributes(cls, values):
        if values is not None and (
            len(set(values)) != len(values) or any(not value for value in values)
        ):
            raise ValueError("Select each nonempty attribute once.")
        return values


class InferenceAttentionRequest(RequestModel):
    slideIds: list[str] = Field(min_length=1, max_length=32)
    patchWidthLevel0: Dimension | None = None
    patchHeightLevel0: Dimension | None = None
    resources: ResourcePolicy = Field(
        default_factory=lambda: ResourcePolicy(gpuIds=[], dataLoaderWorkers=0, ramGbPerRun=8.0)
    )
    operationId: str = Field(min_length=1, max_length=128)

    @field_validator("slideIds")
    @classmethod
    def distinct_slides(cls, values):
        if len(set(values)) != len(values) or any(not value for value in values):
            raise ValueError("Select each slide once.")
        return values

    @model_validator(mode="after")
    def paired_geometry(self):
        if (self.patchWidthLevel0 is None) != (self.patchHeightLevel0 is None):
            raise ValueError("Provide both patch width and height in level-0 pixels.")
        return self
