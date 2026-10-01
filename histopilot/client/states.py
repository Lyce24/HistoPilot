"""One run state for anything that runs, over the service's several status vocabularies.

`runState` is `queued`, `running`, `succeeded`, `failed`, `needs-attention` (held,
interrupted without automatic resume, or the runner stopped) or `cancelled`, and `None`
before anything has started. The service's own fields are always kept alongside.
"""

RUN_STATES = ("queued", "running", "succeeded", "failed", "needs-attention", "cancelled")
SETTLED = frozenset({"succeeded", "failed", "needs-attention", "cancelled"})

_RUN_STATE = {
    # Task Center tasks.
    "blocked": "queued",
    "queued": "queued",
    "starting": "running",
    "running": "running",
    "stopping": "running",
    "succeeded": "succeeded",
    "failed": "failed",
    "cancelled": "cancelled",
    "interrupted": "needs-attention",
    # Task Center rollups and stage chips.
    "held": "needs-attention",
    "attention": "needs-attention",
    "runner-stopped": "needs-attention",
    "completed": "succeeded",
    # Experiments, batches, runs, jobs and archives.
    "waiting": "queued",
    "scheduled": "queued",
    "launching": "queued",
    "submitted": "queued",
    "cancelling": "running",
    "needs-attention": "needs-attention",
    "unknown": "needs-attention",
    "partial": "needs-attention",
}
# Nothing has run yet.
_NOT_STARTED = frozenset(
    {"not-started", "not_started", "created", "planned", "ready", "draft", "saved", "frozen"}
)


def run_state(status) -> str | None:
    if not isinstance(status, str) or status in _NOT_STARTED:
        return None
    return _RUN_STATE.get(status, "needs-attention")


def task_run_state(task: dict) -> str | None:
    """A Task Center task; an interrupted task awaiting automatic requeue is still queued,
    and a queued task of a held owner needs attention: it starts only once released."""
    if task.get("state") == "interrupted" and task.get("awaitingRequeue"):
        return "queued"
    state = run_state(task.get("state"))
    owner = task.get("owner")
    if state == "queued" and isinstance(owner, dict) and owner.get("held"):
        return "needs-attention"
    return state
