"""Storage commands for unvalidated local drafts, never scientific publication."""

from typing import Annotated, Literal

from pydantic import Field, JsonValue, StrictInt, field_validator

from histopilot.schemas.workspace import RequestModel


class DraftValues(RequestModel):
    name: str = Field(min_length=1, max_length=120)
    payload: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def nonempty_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Enter a draft name.")
        return value


class CreateDraftRequest(DraftValues):
    kind: Literal["import", "experiment"]
    # Optional: the same ID again returns the draft the first request created.
    operationId: str | None = Field(default=None, min_length=1, max_length=200)


class UpdateDraftRequest(DraftValues):
    expectedRevision: Annotated[StrictInt, Field(ge=1, le=2**63 - 1)]
