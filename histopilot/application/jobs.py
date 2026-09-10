"""Job orchestration will bind a job port after persistence and audits exist."""

from histopilot.ports.jobs import JobStatus


class JobService:
    def submit(self, run_id: str) -> str:
        raise NotImplementedError("Run submission is not implemented; no job has been launched.")

    def status(self, job_id: str) -> JobStatus:
        raise NotImplementedError("Job status retrieval is not implemented.")
