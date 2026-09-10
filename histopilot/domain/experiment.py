"""Experiment intent, individual executions, and traceable result artifacts."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class Experiment:
    id: str
    project_id: str
    name: str
    cohort_id: str
    split_id: str
    feature_set_id: str
    mil_backend: str
    mil_model: str
    config_uri: str
    task: Literal["classification", "regression", "survival"] = "classification"


@dataclass(frozen=True, slots=True)
class Run:
    id: str
    experiment_id: str
    seed: int
    fold: int
    code_revision: str
    environment_uri: str
    status: Literal["planned", "queued", "running", "completed", "failed", "cancelled"] = "planned"
    job_id: str | None = None
    mil_checkpoint_uri: str | None = None
    mil_checkpoint_hash: str | None = None
    log_uri: str | None = None


@dataclass(frozen=True, slots=True)
class Result:
    id: str
    run_id: str
    partition: Literal["train", "validation", "test", "external"]
    metrics: tuple[tuple[str, float], ...]
    predictions_uri: str
    ground_truth_uri: str
    attention_uri: str | None = None
