"""Execution boundary for future local or managed job runners."""

from dataclasses import dataclass
from typing import Protocol

from histopilot.domain.jobs import JobState


@dataclass(frozen=True, slots=True)
class JobStatus:
    id: str
    state: JobState
    log_uri: str
    message: str = ""


class JobPort(Protocol):
    def submit(self, *, run_id: str, manifest_uri: str, log_uri: str) -> str:
        """Submit a run manifest and return the runner's job ID."""
        ...

    def status(self, job_id: str) -> JobStatus:
        """Read the actual runner state; never infer success from submission."""
        ...
