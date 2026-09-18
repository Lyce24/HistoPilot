"""Explicit, bounded requests for project portability and source registration."""

from typing import Literal

from pydantic import Field

from histopilot.schemas.workspace import RequestModel


class PortabilityRequest(RequestModel):
    action: Literal["export", "verify", "restore"]
    archivePath: str = Field(min_length=1, max_length=4096)
    destinationPath: str | None = Field(default=None, min_length=1, max_length=4096)
    operationId: str = Field(min_length=1, max_length=200)


class RelinkSource(RequestModel):
    sourceId: str = Field(pattern=r"^source-[a-f0-9]{20}$")
    expectedPath: str = Field(min_length=1, max_length=4096)
    replacementPath: str = Field(min_length=1, max_length=4096)
