"""Planned audits must resolve source records before reporting pass/fail."""

from dataclasses import dataclass
from typing import Literal

from histopilot.domain import Cohort, FeatureSet, Split


@dataclass(frozen=True, slots=True)
class AuditFinding:
    code: str
    severity: Literal["info", "warning", "error"]
    message: str
    entity_ids: tuple[str, ...] = ()


class AuditService:
    def audit(
        self, *, cohort: Cohort, split: Split, features: FeatureSet | None = None
    ) -> tuple[AuditFinding, ...]:
        raise NotImplementedError(
            "Leakage, label, duplicate-slide, and feature compatibility audits are not implemented."
        )
