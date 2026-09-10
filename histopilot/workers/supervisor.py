"""Future process supervision: scheduling, cancellation, and restart reconciliation."""

from histopilot.domain.jobs import ExecutionPlan, Job


class LocalProcessExecutor:
    """Reserved subprocess adapter, independent of HTTP requests/browser lifetime.

    The implementation must use a configured interpreter and an argv list with
    shell=False, set GPU assignment in the child environment, keep persistent
    logs, and reconcile actual process identity after service restarts. No CUDA
    tensors or model instances cross this boundary. No processes launch yet.
    """

    def submit(self, plan: ExecutionPlan) -> Job:
        raise NotImplementedError("Connect a worker and durable job registry before submission")

    def status(self, job_id: str) -> Job:
        raise NotImplementedError("Job process inspection is not connected")

    def cancel(self, job_id: str) -> None:
        raise NotImplementedError("Process cancellation is not connected")

    def logs(self, job_id: str) -> str:
        raise NotImplementedError("Worker log retrieval is not connected")

    def reconcile(self) -> tuple[Job, ...]:
        raise NotImplementedError("Interrupted/orphaned job reconciliation is not connected")
