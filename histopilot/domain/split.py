"""Explicit split assignments; group leakage checks belong to application audits."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    member_id: str
    group_id: str
    partition: Literal["train", "validation", "test"]
    fold: int = 0


@dataclass(frozen=True, slots=True)
class Split:
    id: str
    cohort_id: str
    strategy: str
    seed: int
    assignments: tuple[SplitAssignment, ...]
    group_by: Literal["patient", "specimen", "slide"] = "patient"
