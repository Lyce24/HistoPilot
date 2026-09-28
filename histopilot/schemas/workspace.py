"""Small intent requests; clients cannot submit authoritative scientific records."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator


class RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


Seed = Annotated[StrictInt, Field(ge=0, le=2**32 - 1)]


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
    # Older API clients may omit this; the UI always sends its editor baseline.
    expectedConfig: ProjectConfig | None = None
