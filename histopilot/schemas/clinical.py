"""Clinical reports accept analysis choices, never client-supplied predictions."""

from typing import Annotated, Literal

from pydantic import Field, StrictInt, field_validator, model_validator

from histopilot.schemas.evaluations import ConfigurationId
from histopilot.schemas.workspace import RequestModel


class ClinicalSelection(RequestModel):
    evaluationId: ConfigurationId
    name: str = Field(default="Clinical utility analysis", min_length=1, max_length=120)
    unit: Literal["selected", "slide", "patient"] = "selected"
    positiveClass: str | None = Field(default=None, min_length=1, max_length=128)
    threshold: float | None = Field(default=None, gt=0, lt=1, allow_inf_nan=False)
    bins: Annotated[StrictInt, Field(ge=2, le=50)] = 10
    thresholdMin: float = Field(default=0.01, gt=0, lt=1, allow_inf_nan=False)
    thresholdMax: float = Field(default=0.99, gt=0, lt=1, allow_inf_nan=False)
    thresholdSteps: Annotated[StrictInt, Field(ge=2, le=501)] = 99

    @field_validator("name")
    @classmethod
    def readable_name(cls, value):
        if not value.strip():
            raise ValueError("Enter an analysis name.")
        return value.strip()

    @field_validator("threshold", "thresholdMin", "thresholdMax", mode="before")
    @classmethod
    def numeric_thresholds(cls, value):
        if value is not None and type(value) not in (int, float):
            raise ValueError("Thresholds must be numbers.")
        return value

    @model_validator(mode="after")
    def ordered_thresholds(self):
        if self.thresholdMin >= self.thresholdMax:
            raise ValueError("The minimum threshold must be below the maximum threshold.")
        return self


class SaveClinicalAnalysis(ClinicalSelection):
    previewHash: str = Field(pattern=r"^[a-f0-9]{64}$")
    operationId: str = Field(min_length=1, max_length=128)
