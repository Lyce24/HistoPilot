"""Small intent requests; clients cannot submit authoritative scientific records."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator


class RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CohortRequest(RequestModel):
    datasetId: str = Field(min_length=1, max_length=128)
    specimenType: Literal["Any", "all", "Primary", "Metastatic"] = "Any"
    msi: Literal["Any", "all", "MSS", "MSI-H"] = "Any"
    braf: Literal["Any", "all", "WT", "Mutant"] = "Any"

    @field_validator("specimenType", "msi", "braf", mode="before")
    @classmethod
    def normalize_all(cls, value: object) -> object:
        return "Any" if value == "all" else value


Seed = Annotated[StrictInt, Field(ge=0, le=2**32 - 1)]


class ExperimentRequest(RequestModel):
    cohortId: str = Field(min_length=1, max_length=128)
    pairs: list[str] = Field(min_length=1, max_length=32)
    seeds: list[Seed] = Field(min_length=1, max_length=20)
    folds: Annotated[StrictInt, Field(ge=2, le=10)]
    aggregation: Literal["mean", "max"] = "mean"


class SourceRequest(RequestModel):
    path: str = Field(min_length=1, max_length=4096)


class CreateDirectoryRequest(RequestModel):
    parentPath: str = Field(min_length=1, max_length=4096)
    name: str = Field(min_length=1, max_length=255)
    purpose: Literal["source", "storage"] = "source"


class ProjectConfig(RequestModel):
    """Optional planning choices, never an executable experiment specification."""

    task: Literal["binary_classification", "multiclass_classification"] | None = None
    targetColumn: str | None = Field(default=None, min_length=1, max_length=128)
    positiveLabel: str | None = Field(default=None, min_length=1, max_length=128)
    seed: Seed | None = None
    folds: Annotated[StrictInt, Field(ge=2, le=10)] | None = None
    encoderId: str | None = Field(default=None, min_length=1, max_length=128)
    milId: str | None = Field(default=None, min_length=1, max_length=128)


class ProjectRequest(RequestModel):
    name: str = Field(min_length=1, max_length=120)
    storagePath: str = Field(min_length=1, max_length=4096)
    description: str = Field(default="", max_length=2000)
    dataPath: str | None = Field(default=None, min_length=1, max_length=4096)
    slidePath: str | None = Field(default=None, min_length=1, max_length=4096)
    featurePath: str | None = Field(default=None, min_length=1, max_length=4096)
    config: ProjectConfig = Field(default_factory=ProjectConfig)

    @field_validator("name")
    @classmethod
    def nonempty_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Enter an experiment name.")
        return value


class OpenProjectRequest(SourceRequest):
    pass


class ProjectSourceRequest(SourceRequest):
    role: Literal["data", "slides", "features"] = "data"


class ProjectUpdateRequest(RequestModel):
    config: ProjectConfig
