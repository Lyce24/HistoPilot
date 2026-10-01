"""The service, its projects, experiments and Task Center, for the CLI and the agent tools.

Reads return the service's own fields, adding `runState` where something runs; the Task
Center's actions (a task's cancel or retry, an owner's, capacity) are sent as operations.
"""

from .api import Client, segment
from .errors import ClientError
from .paging import DEFAULT_LIMIT, served
from .states import run_state, task_run_state


def project_path(project: str) -> str:
    return f"/projects/{segment(project)}"


def listed_items(listed, key: str) -> list[dict]:
    """The rows of a list answer: under ``key``, or the answer itself when it is a list."""
    if isinstance(listed, dict):
        listed = listed.get(key)
    return list(listed or [])


# Service ------------------------------------------------------------------------------


def status(client: Client) -> dict:
    health = client.get("/health")
    # A scoped agent token never sees the machine's workspace or runtimes; it reads what it
    # reaches instead.
    system = {} if client.scoped else client.get("/system")
    access = client.get("/access") if client.scoped else None
    summary = client.get("/task-center/summary")
    try:
        code = client.get("/version")
    except ClientError as error:
        # Services from before the version route answer 404; they still work.
        if error.kind != "not-found":
            raise
        code = {}
    workers = system.get("workers") or {}
    runner = summary.get("runner") or {}
    return {
        "url": client.url,
        "version": health.get("version"),
        "serviceRevision": code.get("gitRevision"),
        "features": code.get("features"),
        "workspace": system.get("workspace"),
        "extraction": {
            "ready": bool(workers.get("executionEnabled")),
            "trident": bool((workers.get("trident") or {}).get("available")),
            "tmux": bool(workers.get("tmuxAvailable")),
        },
        "runner": {
            key: runner.get(key)
            for key in ("alive", "state", "codeCurrent", "autostart", "heartbeatAt")
        },
        "queuePaused": summary.get("paused"),
        "tasks": summary.get("counts") or {},
        "recentFailures": summary.get("recentFailures"),
        "access": access,
    }


# Projects -----------------------------------------------------------------------------


def projects(client: Client) -> list[dict]:
    return listed_items(client.get("/projects"), "projects")


def project(client: Client, identity: str) -> dict:
    return client.get(f"{project_path(identity)}/workspace")


# Experiments --------------------------------------------------------------------------


def experiment_path(project: str, identity: str = "") -> str:
    base = f"{project_path(project)}/model-experiments"
    return f"{base}/{segment(identity)}" if identity else base


def _experiment(item: dict) -> dict:
    return {**item, "runState": run_state(item.get("status"))}


def experiments(client: Client, project: str, *, state: str = "active") -> list[dict]:
    listed = client.get(experiment_path(project), query={"state": state, "summary": True})
    return [_experiment(item) for item in listed_items(listed, "items")]


def experiment(client: Client, project: str, identity: str) -> dict:
    return _experiment(client.get(experiment_path(project, identity)))


def experiment_id(client: Client, project: str, name: str, *, existing: bool = False) -> str:
    """The experiment ``name`` stands for: an ID as it is, else the one experiment with that
    name, compared without regard to case, preferring active experiments. Names need not be
    unique, so a name two experiments share is refused with their IDs. Anything else goes to
    the service as an ID, which answers with its own code when no experiment has it.

    ``existing`` refuses an experiment that is not there (EXPERIMENT_NOT_FOUND) at once, for
    a filter that would otherwise answer with an empty list."""
    if name.startswith("draft-") and not existing:
        return name
    try:
        listed = experiments(client, project, state="all")
    except ClientError:
        if existing:
            raise
        return name
    if any(item.get("id") == name for item in listed):
        return name
    named = [item for item in listed if (item.get("name") or "").casefold() == name.casefold()]
    matches = [item for item in named if item.get("state", "active") == "active"] or named
    if len(matches) > 1:
        raise ClientError(
            f"{len(matches)} experiments are named {name!r}: "
            f"{', '.join(item['id'] for item in matches)}. Name one by its ID.",
            code="EXPERIMENT_AMBIGUOUS",
            kind="refused",
        )
    if matches:
        return matches[0]["id"]
    if existing:
        raise ClientError(
            f"No experiment in this project has the name or ID {name!r}.",
            code="EXPERIMENT_NOT_FOUND",
            kind="not-found",
        )
    return name


def experiment_results(client: Client, project: str, identity: str) -> dict:
    return client.get(f"{experiment_path(project, identity)}/results")


# Task Center --------------------------------------------------------------------------


def _task(item: dict) -> dict:
    return {**item, "runState": task_run_state(item)}


def tasks(
    client: Client,
    *,
    state: str | None = None,
    kind: str | None = None,
    owner: str | None = None,
    project: str | None = None,
    limit: int = DEFAULT_LIMIT,
    offset: int = 0,
) -> tuple[list[dict], dict]:
    listed = client.get(
        "/task-center/tasks",
        query={
            "state": state,
            "kind": kind,
            "owner": owner,
            "project": project,
            "limit": limit,
            "offset": offset,
        },
    )
    items = [_task(item) for item in listed_items(listed, "tasks")]
    return items, served(items, offset=offset, limit=limit, has_more=bool(listed.get("hasMore")))


def task(client: Client, identity: str) -> dict:
    return _task(client.get(f"/task-center/tasks/{segment(identity)}"))


def task_log(client: Client, identity: str) -> str:
    return client.get(f"/task-center/tasks/{segment(identity)}/log", accept="text")


def task_action(client: Client, identity: str, action: str, *, operation_id=None) -> dict:
    return client.operation(
        "POST",
        f"/task-center/tasks/{segment(identity)}/{segment(action)}",
        {},
        prefix=f"task-{action}",
        operation_id=operation_id,
    )


def owners(client: Client, *, scope: str = "live") -> list[dict]:
    return listed_items(client.get("/task-center/owners", query={"scope": scope}), "owners")


def owner(client: Client, key: str) -> dict:
    return client.get(f"/task-center/owners/{segment(key)}")


def owner_action(
    client: Client, key: str, action: str, *, position=None, operation_id=None
) -> dict:
    body = {"position": position} if position else {}
    return client.operation(
        "POST",
        f"/task-center/owners/{segment(key)}/{segment(action)}",
        body,
        prefix=f"owner-{action}",
        operation_id=operation_id,
    )


def capacity(client: Client) -> dict:
    return client.get("/task-center/capacity")


def update_capacity(client: Client, changes: dict, *, operation_id=None) -> dict:
    return client.operation(
        "PUT", "/task-center/capacity", changes, prefix="capacity", operation_id=operation_id
    )


def task_log_from(client: Client, identity: str, offset: int) -> tuple[bytes, int, int]:
    """Log bytes from ``offset``: the bytes, the next offset and the log's current size."""
    response = client.get(
        f"/task-center/tasks/{segment(identity)}/log", query={"offset": offset}, accept="response"
    )
    data, headers = response.body, response.headers
    following = int(headers.get("x-histopilot-log-next-offset", offset + len(data)))
    return data, following, int(headers.get("x-histopilot-log-size", following))
