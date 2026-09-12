"""Predictor intent shared by editable batch plans and historical experiments."""

from typing import Literal

from pydantic import Field, model_validator

from histopilot.schemas.workspace import RequestModel


class ExperimentPredictorPolicy(RequestModel):
    method: Literal["skip", "ensemble", "refit", "both"] = "ensemble"
    refitPercentile: float | None = Field(
        default=None, ge=1, le=100, allow_inf_nan=False, strict=True
    )

    @model_validator(mode="after")
    def chosen_epoch_policy(self):
        if self.method in {"refit", "both"} and self.refitPercentile is None:
            raise ValueError("Choose the best-checkpoint epoch percentile for refit training.")
        if self.method in {"skip", "ensemble"} and self.refitPercentile is not None:
            raise ValueError("This predictor choice does not use a refit epoch percentile.")
        return self
