"""Bounded exploratory feature views; never a training or cohort mutation."""

from typing import Annotated, Literal

from pydantic import Field, StrictInt, field_validator

from histopilot.schemas.evaluations import ConfigurationId
from histopilot.schemas.workspace import RequestModel

DatasetId = Annotated[str, Field(pattern=r"^dataset-[a-f0-9]{64}$")]


class MorphologyIndexRequest(RequestModel):
    datasetId: DatasetId
    featureBundleId: ConfigurationId
    maxSlides: Annotated[StrictInt, Field(ge=1, le=256)] = 128
    patchesPerSlide: Annotated[StrictInt, Field(ge=1, le=128)] = 32
    slideIds: list[str] = Field(default_factory=list, max_length=256)

    @field_validator("slideIds")
    @classmethod
    def unique_slides(cls, values):
        if any(not value or len(value) > 512 for value in values):
            raise ValueError("Slide IDs must contain 1–512 characters.")
        return sorted(set(values))


class MorphologyNeighborsRequest(RequestModel):
    indexId: str = Field(pattern=r"^[a-f0-9]{64}$")
    slideId: str = Field(min_length=1, max_length=512)
    mode: Literal["slide", "patch"] = "slide"
    patchIndex: Annotated[StrictInt, Field(ge=0)] | None = None
    limit: Annotated[StrictInt, Field(ge=1, le=30)] = 8
    otherSlidesOnly: bool = True
