"""Analysis choices frozen before fitting or external evaluation."""

from typing import Annotated, Literal

from pydantic import Field, StrictInt

from histopilot.schemas.workspace import RequestModel, Seed


class PatientAnalysisSettings(RequestModel):
    version: Literal[1] = 1
    confidenceLevel: Literal[0.95] = 0.95
    bootstrapResamples: Annotated[StrictInt, Field(ge=200, le=10000)] = 2000
    bootstrapSeed: Seed = 42
    oneSlideSeed: Seed = 42
