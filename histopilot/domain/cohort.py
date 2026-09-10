"""A cohort's membership, target, and ground-truth source."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class Cohort:
    id: str
    dataset_version_id: str
    name: str
    target: str
    label_source_uri: str
    member_ids: tuple[str, ...]
    unit: Literal["patient", "specimen", "slide"] = "patient"
    filters: tuple[tuple[str, str], ...] = ()
