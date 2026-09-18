"""Bounded queries over frozen evaluation evidence."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt


class CaseReviewQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    unit: Literal["selected", "slide", "patient"] = "selected"
    outcome: Literal["all", "error", "false_positive", "false_negative", "correct", "unlabeled", "disagreement"] = "all"
    comparisonId: str | None = Field(default=None, pattern=r"^configuration-[a-f0-9]{64}$")
    actualClass: Annotated[StrictInt, Field(ge=0)] | None = None
    predictedClass: Annotated[StrictInt, Field(ge=0)] | None = None
    search: str = Field(default="", max_length=200)
    attribute: str | None = Field(default=None, max_length=200)
    attributeValue: str | None = Field(default=None, max_length=2000)
    minConfidence: float = Field(default=0, ge=0, le=1)
    offset: Annotated[StrictInt, Field(ge=0)] = 0
    limit: Annotated[StrictInt, Field(ge=1, le=100)] = 30
