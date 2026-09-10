"""Job state is separate from experiment intent and scientific run outcomes."""

from dataclasses import dataclass
from enum import StrEnum


class JobState(StrEnum):
    DRAFT = "draft"
    QUEUED = "queued"
    VALIDATING = "validating"
    STARTING = "starting"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"
    ORPHANED = "orphaned"


@dataclass(frozen=True, slots=True)
class Job:
    id: str
    run_id: str
    manifest_uri: str
    log_uri: str
    state: JobState = JobState.DRAFT
    gpu_ids: tuple[int, ...] = ()
    pid: int | None = None


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Produced by a trusted adapter; pass argv without a shell when implemented."""

    job_id: str
    manifest_uri: str
    executable: str
    arguments: tuple[str, ...]
    working_directory: str
    log_uri: str
    gpu_ids: tuple[int, ...] = ()
