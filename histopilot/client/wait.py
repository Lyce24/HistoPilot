"""Waiting for work to settle, over any record that carries a `runState`.

A stopped runner never finishes anything, so a wait on queued work ends as soon as the
runner is found stopped instead of hanging. A deadline ends the wait with a timeout while
the work goes on.
"""

from collections.abc import Callable

from .api import Client
from .errors import ClientError
from .states import SETTLED

POLL_SECONDS = 2.0

_OUTCOMES = {
    "failed": ("WORK_FAILED", "The work failed."),
    "cancelled": ("WORK_CANCELLED", "The work was cancelled."),
    "needs-attention": ("WORK_NEEDS_ATTENTION", "The work stopped and needs attention."),
}


def runner_alive(client: Client) -> bool:
    runner = (client.get("/task-center/summary").get("runner")) or {}
    return bool(runner.get("alive"))


def wait(client: Client, read: Callable[[], dict]) -> dict:
    """Poll ``read`` until its ``runState`` settles; raise for anything but success."""
    while True:
        current = read()
        state = current.get("runState")
        if state in SETTLED:
            return settled(current)
        if state in (None, "queued") and not runner_alive(client):
            raise ClientError(
                "The Task Center runner is stopped, so queued work cannot start. "
                "Start it with `histopilot runner start`, then wait again.",
                code="WORK_NEEDS_ATTENTION",
                kind="work-failed",
                data=current,
            )
        client.pause(POLL_SECONDS)


def settled(record: dict) -> dict:
    state = record.get("runState")
    if state == "succeeded":
        return record
    code, message = _OUTCOMES.get(state, _OUTCOMES["needs-attention"])
    reason = record.get("statusReason") or (record.get("failure") or {}).get("title")
    raise ClientError(
        f"{message} {reason}" if reason else message,
        code=code,
        kind="work-failed",
        data=record,
    )
