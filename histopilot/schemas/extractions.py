"""User intent for an isolated TRIDENT run against a frozen dataset."""

from typing import Any

from pydantic import Field

from histopilot.schemas.workspace import RequestModel


class ExtractionSpec(RequestModel):
    datasetId: str = Field(min_length=1, max_length=128)
    outputPath: str = Field(min_length=1, max_length=4096)
    options: dict[str, Any] = Field(default_factory=dict, max_length=100)


class SubmitExtractionRequest(ExtractionSpec):
    previewHash: str = Field(pattern=r"^[a-f0-9]{64}$")
    operationId: str = Field(min_length=1, max_length=200)
