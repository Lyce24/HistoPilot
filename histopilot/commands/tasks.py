"""`histopilot tasks`: the machine-wide Task Center queue."""

import codecs
import sys

import typer

from histopilot.client import resources
from histopilot.client.errors import usage_error
from histopilot.client.paging import page
from histopilot.client.states import SETTLED
from histopilot.client.wait import wait

from . import output
from .common import Result, command


def _progress(task: dict) -> str | None:
    progress = task.get("progress") or {}
    if not isinstance(progress, dict):
        return None
    done, total = progress.get("completed", progress.get("done")), progress.get("total")
    if isinstance(done, int | float) and isinstance(total, int | float) and total:
        return f"{done:g}/{total:g}"
    fraction = progress.get("fraction")
    return f"{fraction:.0%}" if isinstance(fraction, int | float) else None


def register(app: typer.Typer) -> None:
    group = typer.Typer(no_args_is_help=True, help="The machine-wide Task Center queue.")
    app.add_typer(group, name="tasks", rich_help_panel="Task Center")

    @command(group, "list", lists=True)
    def list_(
        ctx,
        state: str | None = typer.Option(
            None,
            "--state",
            help="Comma-separated states or live, history, pending, active (default: all).",
        ),
        kind: str | None = typer.Option(None, "--kind", help="Comma-separated task kinds."),
        owner: str | None = typer.Option(None, "--owner", help="Owner key."),
        all_projects: bool = typer.Option(
            False, "--all-projects", help="Every project on this machine, not only --project."
        ),
    ):
        """List tasks: queue order for live states, most recent first for finished ones."""
        project = None if all_projects else ctx.optional_project()
        rows, bounds = resources.tasks(
            ctx.client,
            state=state,
            kind=kind,
            owner=owner,
            project=project,
            limit=ctx.limit,
            offset=ctx.offset,
        )
        columns = [
            ("ID", "id"),
            ("KIND", "kind"),
            ("STATE", "state"),
            ("RUN STATE", "runState"),
            ("PROGRESS", _progress),
            ("TITLE", "title"),
            ("UPDATED", "updatedAt"),
        ]
        return Result(rows, page=bounds, text=lambda: output.table(rows, columns))

    @command(group, "show")
    def show(ctx, task: str = typer.Argument(..., help="Task ID.")):
        """Show one task: owner, state, waiting reason, failure and the log's tail."""
        data = resources.task(ctx.client, task)
        owner = data.get("owner") or {}
        failure = data.get("failure") or {}

        def text():
            lines = output.fields(
                [
                    ("Task", f"{data.get('title')} ({data.get('id')})"),
                    ("Kind", data.get("kind")),
                    ("State", f"{data.get('state')} · run state {data.get('runState')}"),
                    ("Owner", f"{owner.get('title')} ({owner.get('kind')} {owner.get('id')})"),
                    ("Project", owner.get("projectName") or owner.get("projectId")),
                    ("Attempt", data.get("attempt")),
                    ("Waiting", data.get("waitingReason")),
                    ("Progress", _progress(data)),
                    (
                        "Failure",
                        " ".join(
                            str(failure[key]) for key in ("title", "advice") if failure.get(key)
                        )
                        or None,
                    ),
                    ("Updated", data.get("updatedAt")),
                ]
            )
            tail = data.get("logTail")
            return f"{lines}\n\n{tail.rstrip()}" if tail else lines

        return Result(data, text=text)

    @command(group, "log")
    def log(
        ctx,
        task: str = typer.Argument(..., help="Task ID."),
        offset: int | None = typer.Option(
            None, "--from", min=0, metavar="BYTE", help="Read from this byte offset."
        ),
        follow: bool = typer.Option(
            False, "--follow", "-f", help="Keep printing until the task settles."
        ),
    ):
        """Print a task's log, from a byte offset, or following it as it grows."""
        if follow:
            return _follow(ctx, task, offset or 0)
        if offset is None:
            text = resources.task_log(ctx.client, task)
            return Result(
                {"taskId": task, "text": text}, text=text.rstrip("\n") or "The log is empty."
            )
        data, following, size = resources.task_log_from(ctx.client, task, offset)
        text = data.decode("utf-8", "replace")
        result = {"taskId": task, "text": text, "offset": offset, "nextOffset": following}
        return Result({**result, "size": size}, text=text.rstrip("\n"))

    _register_actions(group)


FOLLOW_POLL_SECONDS = 1.0


def _follow(ctx, task: str, offset: int) -> Result:
    """Print new log text as it arrives; with --json, one JSON Lines event per chunk."""
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    while True:
        data, offset, size = resources.task_log_from(ctx.client, task, offset)
        text = decoder.decode(data)
        if text:
            if ctx.json_mode:
                output.emit(output.dump({"event": "log", "text": text, "nextOffset": offset}))
            else:
                sys.stdout.write(text)
                sys.stdout.flush()
        if offset < size:
            continue
        current = resources.task(ctx.client, task)
        if current["runState"] in SETTLED:
            # Anything written between the last read and the settled state.
            data, offset, _ = resources.task_log_from(ctx.client, task, offset)
            tail = decoder.decode(data, final=True)
            if tail and ctx.json_mode:
                output.emit(output.dump({"event": "log", "text": tail, "nextOffset": offset}))
            elif tail:
                sys.stdout.write(tail)
            result = {"taskId": task, "state": current["state"], "runState": current["runState"]}
            return Result({**result, "nextOffset": offset}, text=f"-- {current['state']}")
        ctx.client.pause(FOLLOW_POLL_SECONDS)


MOVES = ("top", "up", "down", "bottom")
# The owner's `actions` flag that allows each action; moves report up and down only.
_ALLOWED = {"top": "moveUp", "up": "moveUp", "down": "moveDown", "bottom": "moveDown"}


def _unavailable(action: str, subject: str) -> list[dict]:
    return [
        {
            "code": "TASK_ACTION_UNAVAILABLE",
            "message": f"{action.capitalize()} is not available for this {subject} now.",
            "severity": "error",
        }
    ]


def _task_change(ctx, task: str, action: str) -> Result:
    current = resources.task(ctx.client, task)
    owner = current.get("owner") or {}
    preview = {
        "action": action,
        "task": {key: current.get(key) for key in ("id", "kind", "title", "state", "runState")},
        "owner": {key: owner.get(key) for key in ("key", "title", "projectName")},
    }
    allowed = (current.get("actions") or {}).get(action)
    return ctx.commit(
        preview,
        lambda: resources.task_action(ctx.client, task, action, operation_id=ctx.operation_id),
        findings=[] if allowed else _unavailable(action, "task"),
        summary=f"{action.capitalize()} task {current.get('title')} ({current.get('state')}).",
    )


def _owner_change(ctx, key: str, action: str, position: str | None = None) -> Result:
    current = resources.owner(ctx.client, key)
    preview = {
        "action": action,
        "owner": {
            name: current.get(name)
            for name in ("key", "kind", "id", "title", "projectName", "held", "position")
        },
    }
    if position:
        preview["to"] = position
    allowed = (current.get("actions") or {}).get(_ALLOWED.get(position, action))
    counts = current.get("counts") or {}
    shown = ", ".join(f"{count} {state}" for state, count in counts.items() if count) or "no tasks"
    return ctx.commit(
        preview,
        lambda: resources.owner_action(
            ctx.client, key, action, position=position, operation_id=ctx.operation_id
        ),
        findings=[] if allowed else _unavailable(action, "owner"),
        summary=f"{action.capitalize()} {current.get('title')}: {shown}.",
    )


def _register_actions(group: typer.Typer) -> None:
    @command(group, "owners", lists=True)
    def owners(
        ctx,
        everything: bool = typer.Option(False, "--all", help="Include owners with no live work."),
    ):
        """List task owners (experiments, batches, records) in queue order."""
        items = resources.owners(ctx.client, scope="all" if everything else "live")
        rows, bounds = page(items, offset=ctx.offset, limit=ctx.limit)
        columns = [
            ("KEY", "key"),
            ("TITLE", "title"),
            ("PROJECT", "projectName"),
            ("POSITION", "position"),
            ("HELD", "held"),
            ("WAITING", "waitingReason"),
        ]
        return Result(rows, page=bounds, text=lambda: output.table(rows, columns))

    @command(group, "owner")
    def owner(ctx, key: str = typer.Argument(..., help="Owner key, from `tasks owners`.")):
        """Show one owner: counts, queue position, waiting reason and allowed actions."""
        data = resources.owner(ctx.client, key)
        return Result(data, text=lambda: output.record(data))

    for action, summary in (
        ("cancel", "Cancel a task, or every unfinished task of an owner."),
        ("retry", "Retry a failed, cancelled or interrupted task, or an owner's."),
    ):
        _task_or_owner_action(group, action, summary)

    @command(group, "hold", commit=True)
    def hold(ctx, key: str = typer.Argument(..., help="Owner key.")):
        """Hold an owner: it keeps its place in the queue but starts nothing."""
        return _owner_change(ctx, key, "hold")

    @command(group, "release", commit=True)
    def release(ctx, key: str = typer.Argument(..., help="Owner key.")):
        """Release a held owner."""
        return _owner_change(ctx, key, "release")

    @command(group, "stop", commit=True)
    def stop(ctx, key: str = typer.Argument(..., help="Owner key.")):
        """Stop an owner's running tasks and hold it; its work can resume later."""
        return _owner_change(ctx, key, "stop")

    @command(group, "move", commit=True)
    def move(
        ctx,
        key: str = typer.Argument(..., help="Owner key."),
        to: str = typer.Option(..., "--to", help="top, up, down or bottom."),
    ):
        """Move an owner in the queue."""
        if to not in MOVES:
            raise usage_error(f"Move --to one of: {', '.join(MOVES)}.")
        return _owner_change(ctx, key, "move", to)

    @command(group, "wait")
    def wait_(ctx, task: str = typer.Argument(..., help="Task ID.")):
        """Wait until a task settles: exit 0 when it succeeds, 9 when it does not."""
        final = wait(ctx.client, lambda: resources.task(ctx.client, task))
        return Result(final, text=f"{final.get('title')}: {final.get('state')}")

    @command(group, "capacity", commit=True)
    def capacity(
        ctx,
        parallel_gpu_tasks: int | None = typer.Option(
            None, "--parallel-gpu-tasks", min=1, max=16, help="Task slots on each GPU."
        ),
        cpu_task_slots: int | None = typer.Option(
            None, "--cpu-task-slots", min=1, max=128, help="CPU-lane tasks at once."
        ),
        paused: bool | None = typer.Option(
            None, "--pause/--unpause", help="Stop or restart new task starts."
        ),
        auto_resume: bool | None = typer.Option(
            None, "--auto-resume/--no-auto-resume", help="Requeue work lost to a restart."
        ),
    ):
        """Show the queue's capacity, or change it (an admin change: it affects every project)."""
        changes = {
            key: value
            for key, value in (
                ("parallelGpuTasks", parallel_gpu_tasks),
                ("cpuTaskSlots", cpu_task_slots),
                ("paused", paused),
                ("autoResume", auto_resume),
            )
            if value is not None
        }
        current = resources.capacity(ctx.client)
        if not changes:
            return Result(current, text=lambda: output.record(current))
        settings = current.get("settings") or {}
        preview = {
            "change": changes,
            "current": {key: settings.get(key) for key in changes},
            "routeClass": "admin",
        }
        return ctx.commit(
            preview,
            lambda: resources.update_capacity(ctx.client, changes, operation_id=ctx.operation_id),
            summary=f"Change Task Center capacity for every project: {changes}.",
        )


def _task_or_owner_action(group: typer.Typer, action: str, summary: str) -> None:
    @command(group, action, commit=True, help=summary)
    def change(
        ctx,
        task: str | None = typer.Argument(None, help="Task ID."),
        owner: str | None = typer.Option(
            None, "--owner", help=f"{action.capitalize()} an owner's work instead."
        ),
    ):
        if (task is None) == (owner is None):
            raise usage_error("Name a task, or an owner with --owner.")
        return _owner_change(ctx, owner, action) if owner else _task_change(ctx, task, action)
