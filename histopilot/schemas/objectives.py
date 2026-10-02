"""Objectives: the scientific questions a project's experiments answer, and each
experiment's place under one: its role there and the series it belongs to."""

from typing import Annotated, Literal

from pydantic import Field, StrictInt, field_validator, model_validator

from histopilot.schemas.workspace import RequestModel

# How much an objective matters to the study, in a protocol's words.
OBJECTIVE_RANKS = ("primary", "secondary", "exploratory")
# What an experiment does for its objective, in the order the library lists them.
EXPERIMENT_ROLES = ("primary", "comparison", "variant", "generalization", "control", "tuning")

ObjectiveRank = Literal["primary", "secondary", "exploratory"]
ExperimentRole = Literal["primary", "comparison", "variant", "generalization", "control", "tuning"]
OBJECTIVE_ID_PATTERN = r"^objective-[0-9a-f]{32}$"
ObjectiveId = Annotated[str, Field(pattern=OBJECTIVE_ID_PATTERN)]
Revision = Annotated[StrictInt, Field(ge=1, le=2**63 - 1)]
OperationId = Annotated[str, Field(min_length=1, max_length=200)]


def _name(value: str, label: str) -> str:
    value = " ".join(value.split())
    if not value:
        raise ValueError(f"Enter {label}.")
    return value


class CreateObjective(RequestModel):
    name: str = Field(min_length=1, max_length=120)
    rank: ObjectiveRank = "secondary"
    description: str = Field(default="", max_length=2000)
    operationId: OperationId

    @field_validator("name")
    @classmethod
    def readable_name(cls, value: str) -> str:
        return _name(value, "an objective name")

    @field_validator("description")
    @classmethod
    def trimmed_description(cls, value: str) -> str:
        return value.strip()


class UpdateObjective(RequestModel):
    """Omitted fields keep their saved values."""

    expectedRevision: Revision
    name: str | None = Field(default=None, min_length=1, max_length=120)
    rank: ObjectiveRank | None = None
    description: str | None = Field(default=None, max_length=2000)

    @field_validator("name")
    @classmethod
    def readable_name(cls, value: str | None) -> str | None:
        return None if value is None else _name(value, "an objective name")

    @field_validator("description")
    @classmethod
    def trimmed_description(cls, value: str | None) -> str | None:
        return None if value is None else value.strip()


class RemoveObjective(RequestModel):
    expectedRevision: Revision
    operationId: OperationId


class ExperimentFiling(RequestModel):
    """Where an experiment sits: its objective, its role there, and its series and member.

    A blank series or member reads as none; a member needs its series."""

    objectiveId: ObjectiveId | None = None
    role: ExperimentRole | None = None
    series: str | None = Field(default=None, max_length=80)
    seriesMember: str | None = Field(default=None, max_length=80)

    @field_validator("series", "seriesMember")
    @classmethod
    def blank_is_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = " ".join(value.split())
        return value or None

    @model_validator(mode="after")
    def member_needs_series(self):
        if self.seriesMember and not self.series:
            raise ValueError("Name the series before its member.")
        return self


class FileExperiment(ExperimentFiling):
    """A whole filing, replacing the saved one if it is still the one the caller read."""

    expected: ExperimentFiling
