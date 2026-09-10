"""The v1 draft experiment manifest: one model pair, explicit frozen inputs."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

Identifier = Annotated[str, Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9_.:-]+$")]
Seed = Annotated[StrictInt, Field(ge=0, le=4294967295)]


class ExperimentSpec(BaseModel):
    """Shape validation only; the application must resolve every referenced artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    dataset_id: Identifier
    cohort_id: Identifier
    split_id: Identifier
    feature_set_id: Identifier
    encoder_id: Identifier
    mil_model: Identifier
    task: Literal["binary_classification"] = "binary_classification"
    positive_label: str = Field(default="Mutant", min_length=1)
    seeds: tuple[Seed, ...] = Field(min_length=1, max_length=100)
    folds: Annotated[StrictInt, Field(ge=2, le=10)] = 5
    aggregation: Literal["mean", "max"] = "mean"

    @field_validator("seeds")
    @classmethod
    def canonical_seeds(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(sorted(set(value)))
