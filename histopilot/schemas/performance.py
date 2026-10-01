"""Requests over a labeled run's scored predictions."""

from pydantic import Field

from histopilot.schemas.evaluations import ConfigurationId
from histopilot.schemas.inference import Unit
from histopilot.schemas.workspace import RequestModel


class PerformanceBreakdownQuery(RequestModel):
    unit: Unit = "selected"
    attribute: str = Field(min_length=1, max_length=200)
    # Score against a reference standard of the cohort instead of the cohort's own labels.
    referenceId: ConfigurationId | None = None
