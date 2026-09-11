"""Optional durable validation and packing of a saved feature version."""

from typing import Literal

from pydantic import Field

from histopilot.schemas.workspace import RequestModel


class FeaturePackSpec(RequestModel):
    featureSetId: str = Field(min_length=1, max_length=128)
    action: Literal["pack", "validate", "attach"] = "pack"
    outputPath: str | None = Field(default=None, min_length=1, max_length=4096)
    existingPath: str | None = Field(default=None, min_length=1, max_length=4096)
    dtype: Literal["preserve", "float16"] = "preserve"


class SubmitFeaturePackRequest(FeaturePackSpec):
    previewHash: str = Field(pattern=r"^[a-f0-9]{64}$")
    operationId: str = Field(min_length=1, max_length=200)


class SelectFeaturePackRequest(RequestModel):
    artifactId: str | None = Field(default=None, min_length=1, max_length=128)
