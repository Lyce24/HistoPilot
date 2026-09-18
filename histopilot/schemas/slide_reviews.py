"""Human review metadata remains separate from immutable scientific labels."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

ReviewStatus = Literal["unreviewed", "accept", "exclude", "review"]


class ReviewRegion(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    id: str = Field(min_length=1, max_length=120)
    label: str = Field(default="", max_length=300)
    x: float = Field(ge=0, le=1_000_000_000)
    y: float = Field(ge=0, le=1_000_000_000)
    width: float = Field(gt=0, le=1_000_000_000)
    height: float = Field(gt=0, le=1_000_000_000)


class SlideReviewValues(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    status: ReviewStatus = "unreviewed"
    notes: str = Field(default="", max_length=8000)
    reviewer: str = Field(default="", max_length=120)
    reasons: list[Annotated[str, Field(min_length=1, max_length=120)]] = Field(
        default_factory=list, max_length=16
    )
    regions: list[ReviewRegion] = Field(default_factory=list, max_length=128)
    evaluationId: str | None = Field(default=None, pattern=r"^configuration-[a-f0-9]{64}$")

    @field_validator("notes", "reviewer")
    @classmethod
    def clean_text(cls, value):
        if "\x00" in value:
            raise ValueError("Review text must not contain null characters.")
        return value.strip()

    @model_validator(mode="after")
    def unique_entries(self):
        if len(self.reasons) != len(set(self.reasons)):
            raise ValueError("Review reasons must be unique.")
        if len(self.regions) != len({region.id for region in self.regions}):
            raise ValueError("Review region IDs must be unique.")
        return self


class SaveSlideReview(SlideReviewValues):
    expectedRevision: Annotated[StrictInt, Field(ge=0, lt=2**53)]


class ReviewRevision(SlideReviewValues):
    revision: Annotated[StrictInt, Field(ge=1, lt=2**53)]
    updatedAt: str = Field(min_length=1)


class SlideReviewDocument(SlideReviewValues):
    schemaVersion: Literal[1] = 1
    datasetId: str
    slideId: str
    revision: Annotated[StrictInt, Field(ge=0, lt=2**53)] = 0
    updatedAt: str | None = None
    history: list[ReviewRevision] = Field(default_factory=list)

    @model_validator(mode="after")
    def audit_chain(self):
        if len(self.history) != self.revision:
            raise ValueError("The review history must contain every revision.")
        for number, item in enumerate(self.history, start=1):
            if item.revision != number:
                raise ValueError("Review history revisions are not consecutive.")
        if self.history:
            latest = self.history[-1].model_dump()
            if self.model_dump(include=set(latest)) != latest:
                raise ValueError("The latest review must match its recorded history.")
        return self
