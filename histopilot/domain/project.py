"""Project boundary for datasets, cohorts, artifacts, and experiments."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Project:
    id: str
    name: str
    description: str = ""
