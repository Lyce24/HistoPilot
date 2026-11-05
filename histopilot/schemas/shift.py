"""Requests for a run's dataset shift analysis (application/shift.py)."""

from pydantic import Field

from histopilot.schemas.evaluations import ConfigurationId
from histopilot.schemas.inference import Unit
from histopilot.schemas.workspace import RequestModel


class StartShiftAnalysis(RequestModel):
    previewHash: str = Field(pattern=r"^[a-f0-9]{64}$")
    operationId: str = Field(min_length=1, max_length=128)


class ShiftPerformanceQuery(RequestModel):
    unit: Unit = "selected"
    # Score against a reference standard of the cohort instead of the cohort's own labels.
    referenceId: ConfigurationId | None = None
