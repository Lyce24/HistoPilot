"""Mutable presentation labels for immutable scientific versions."""

import unicodedata
from typing import Annotated

from pydantic import Field, StrictInt, field_validator

from histopilot.schemas.workspace import RequestModel


class VersionLabelValues(RequestModel):
    tag: str = Field(default="", max_length=80)
    note: str = Field(default="", max_length=2000)

    @field_validator("tag")
    @classmethod
    def readable_tag(cls, value: str) -> str:
        value = unicodedata.normalize("NFC", value.strip())
        if any(unicodedata.category(character).startswith("C") for character in value):
            raise ValueError("Version tags cannot contain control characters.")
        return value

    @field_validator("note")
    @classmethod
    def readable_note(cls, value: str) -> str:
        if any(
            unicodedata.category(character).startswith("C") and character not in "\n\r\t"
            for character in value
        ):
            raise ValueError(
                "Commit notes cannot contain control characters except line breaks or tabs."
            )
        return value.strip()


class SetVersionLabelRequest(VersionLabelValues):
    expectedRevision: Annotated[StrictInt, Field(ge=0, le=2**63 - 2)]


class FreezeVersionLabel(VersionLabelValues):
    """A personal name is part of the intent to publish a reviewed version."""

    tag: str = Field(min_length=1, max_length=80)

    @field_validator("tag")
    @classmethod
    def required_tag(cls, value: str) -> str:
        if not value:
            raise ValueError("Give this version a personal tag before freezing.")
        return value
