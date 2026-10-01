"""Requests for agent access: scoped tokens and a project's AI-exposure level."""

from typing import Annotated, Literal

from pydantic import Field, StrictInt

from histopilot.api.route_classes import (
    DEFAULT_TOKEN_SCOPES,
    EXPOSURE_LEVELS,
    MAX_TOKEN_DAYS,
    TOKEN_DAYS,
    TOKEN_SCOPES,
)
from histopilot.schemas.workspace import RequestModel

Scope = Literal[TOKEN_SCOPES]
Exposure = Literal[EXPOSURE_LEVELS]


class CreateAccessToken(RequestModel):
    projectId: str = Field(pattern=r"^project-[a-f0-9]{32}$")
    scopes: list[Scope] = Field(default_factory=lambda: list(DEFAULT_TOKEN_SCOPES), min_length=1)
    name: str = Field(default="", max_length=80)
    days: Annotated[StrictInt, Field(ge=1, le=MAX_TOKEN_DAYS)] = TOKEN_DAYS


class SetExposure(RequestModel):
    level: Exposure
    # The level the person reviewed, so two people's changes never cross unseen.
    expectedLevel: Exposure


class ResolveAgentRequest(RequestModel):
    # declined: a person said no. approved, failed or unknown: how a claimed replay ended.
    outcome: Literal["approved", "declined", "failed", "unknown"]
    # The replayed request's HTTP status, recorded for the audit trail.
    status: Annotated[StrictInt, Field(ge=100, le=599)] | None = None
