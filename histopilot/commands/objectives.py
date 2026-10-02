"""`histopilot objective`: the scientific questions a project's experiments are filed under."""

import typer

from histopilot.client import objectives as goals
from histopilot.client import resources
from histopilot.client.errors import usage_error
from histopilot.client.paging import page
from histopilot.schemas.objectives import EXPERIMENT_ROLES, OBJECTIVE_RANKS

from . import output
from .common import Result, command

RANKS = ", ".join(OBJECTIVE_RANKS)


def _rank(value: str | None) -> str | None:
    if value is not None and value not in OBJECTIVE_RANKS:
        raise usage_error(f"Choose --rank from: {RANKS}.")
    return value


def _filed(client, project: str) -> dict[str, list[dict]]:
    """Every experiment filed under each objective, archived and trashed ones included."""
    filed: dict[str, list[dict]] = {}
    for item in resources.experiments(client, project, state="all"):
        if item.get("objectiveId"):
            filed.setdefault(item["objectiveId"], []).append(item)
    return filed


def _role_order(item: dict) -> tuple:
    role = item.get("role")
    return (
        EXPERIMENT_ROLES.index(role) if role in EXPERIMENT_ROLES else len(EXPERIMENT_ROLES),
        item.get("series") or "",
        item.get("seriesMember") or "",
        item.get("name") or "",
    )


def register(app: typer.Typer) -> typer.Typer:
    group = typer.Typer(
        no_args_is_help=True,
        help="Objectives: the scientific questions experiments are filed under.",
    )
    app.add_typer(group, name="objective", rich_help_panel="Records")

    @command(group, "list", lists=True)
    def list_(ctx):
        """List objectives: primary first, then secondary, then exploratory."""
        project = ctx.project()
        filed = _filed(ctx.client, project)
        items = [
            {**item, "experiments": len(filed.get(item["id"], []))}
            for item in goals.objectives(ctx.client, project)
        ]
        rows, bounds = page(items, offset=ctx.offset, limit=ctx.limit)
        columns = [
            ("ID", "id"),
            ("NAME", "name"),
            ("RANK", "rank"),
            ("EXPERIMENTS", "experiments"),
        ]
        return Result(
            rows,
            page=bounds,
            text=lambda: output.table(rows, columns, empty="No objectives yet."),
        )

    @command(group, "show")
    def show(ctx, objective: str = typer.Argument(..., help="Objective ID or name.")):
        """Show an objective and the experiments filed under it, by role."""
        project = ctx.project()
        data = goals.objective(
            ctx.client, project, goals.objective_id(ctx.client, project, objective)
        )
        experiments = sorted(_filed(ctx.client, project).get(data["id"], []), key=_role_order)
        data = {**data, "experiments": experiments}

        def text():
            lines = output.fields(
                [
                    ("Objective", f"{data['name']} ({data['id']})"),
                    ("Rank", data["rank"]),
                    ("Description", data.get("description") or None),
                    ("Revision", data["revision"]),
                    ("Updated", data["updatedAt"]),
                ]
            )
            table = output.table(
                experiments,
                [
                    ("ROLE", lambda row: row.get("role") or "-"),
                    ("SERIES", lambda row: row.get("series") or "-"),
                    ("MEMBER", lambda row: row.get("seriesMember") or "-"),
                    ("NAME", "name"),
                    ("ID", "id"),
                    ("RUN STATE", "runState"),
                ],
                empty="No experiments filed here yet.",
            )
            return f"{lines}\n\n{table}"

        return Result(data, text=text)

    @command(group, "create", commit=True)
    def create(
        ctx,
        name: str = typer.Option(..., "--name", help="The objective's name (1-120 characters)."),
        rank: str = typer.Option("secondary", "--rank", help=f"{RANKS}."),
        description: str = typer.Option("", "--description", help="What it asks; optional."),
    ):
        """Add an objective that experiments can be filed under."""
        project = ctx.project()
        body = {"name": name, "rank": _rank(rank), "description": description}
        preview = {"projectId": project, **body}
        return ctx.commit(
            preview,
            lambda: ctx.client.operation(
                "POST",
                goals.objectives_path(project),
                body,
                prefix="objective-create",
                operation_id=ctx.operation_id,
            ),
            summary=f"Add the {rank} objective {name.strip()!r}.",
        )

    @command(group, "update", commit=True)
    def update(
        ctx,
        objective: str = typer.Argument(..., help="Objective ID or name."),
        name: str | None = typer.Option(None, "--name", help="A new name."),
        rank: str | None = typer.Option(None, "--rank", help=f"{RANKS}."),
        description: str | None = typer.Option(None, "--description", help="A new description."),
    ):
        """Rename, re-rank or describe an objective; experiments stay filed under it."""
        project = ctx.project()
        changes = {
            key: value
            for key, value in {
                "name": name,
                "rank": _rank(rank),
                "description": description,
            }.items()
            if value is not None
        }
        if not changes:
            raise usage_error("Give --name, --rank or --description.")
        current = goals.objective(
            ctx.client, project, goals.objective_id(ctx.client, project, objective)
        )
        preview = {
            "objectiveId": current["id"],
            "from": {key: current.get(key) for key in changes},
            "to": changes,
        }
        return ctx.commit(
            preview,
            lambda: ctx.client.request(
                "PATCH",
                goals.objectives_path(project, current["id"]),
                body={"expectedRevision": current["revision"], **changes},
            ),
            summary=f"Update objective {current['name']!r}: "
            + ", ".join(f"{key} → {value!r}" for key, value in changes.items())
            + ".",
        )

    @command(group, "remove", commit=True)
    def remove(ctx, objective: str = typer.Argument(..., help="Objective ID or name.")):
        """Remove an objective. Its experiments stay as they are, unfiled from it."""
        project = ctx.project()
        current = goals.objective(
            ctx.client, project, goals.objective_id(ctx.client, project, objective)
        )
        filed = len(_filed(ctx.client, project).get(current["id"], []))
        preview = {"objectiveId": current["id"], "name": current["name"], "experiments": filed}
        return ctx.commit(
            preview,
            lambda: ctx.client.operation(
                "POST",
                f"{goals.objectives_path(project, current['id'])}/remove",
                {"expectedRevision": current["revision"]},
                prefix="objective-remove",
                operation_id=ctx.operation_id,
            ),
            summary=f"Remove objective {current['name']!r}. "
            + (
                f"Its {filed} filed experiment{' stays' if filed == 1 else 's stay'}, unfiled; "
                "experiments and results do not change."
                if filed
                else "No experiment is filed under it."
            ),
        )

    return group
