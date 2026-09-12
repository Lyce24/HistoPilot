"""Reviewed, bounded workspace organization requests; never filesystem paths."""

from typing import Annotated, Literal

from pydantic import Field, field_validator

from histopilot.schemas.workspace import RequestModel

RecordKey = Annotated[
    str,
    Field(
        min_length=3,
        max_length=220,
        pattern=r"^(project|dataset|configuration|draft|packing|extraction):[a-z0-9-]+$",
    ),
]


class CleanupSelection(RequestModel):
    action: Literal["archive", "trash", "restore"]
    keys: list[RecordKey] = Field(min_length=1, max_length=1000)

    @field_validator("keys")
    @classmethod
    def distinct_keys(cls, value):
        if len(value) != len(set(value)):
            raise ValueError("Select each record only once.")
        return sorted(value)


class ApplyCleanup(CleanupSelection):
    previewHash: str = Field(pattern=r"^[a-f0-9]{64}$")
    operationId: str = Field(min_length=1, max_length=200)


class CancelCleanupJob(RequestModel):
    key: RecordKey
    operationId: str = Field(min_length=1, max_length=200)
