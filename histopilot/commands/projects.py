"""`histopilot project`: projects known to this service."""

import typer

from histopilot.api.route_classes import EXPOSURE_LEVELS
from histopilot.client import resources
from histopilot.client.api import segment
from histopilot.client.errors import usage_error
from histopilot.client.paging import page

from . import output
from .common import Result, command


def register(app: typer.Typer) -> None:
    group = typer.Typer(no_args_is_help=True, help="Projects known to this service.")
    app.add_typer(group, name="project", rich_help_panel="Records")

    @command(group, "list", lists=True)
    def list_(ctx):
        """List projects, most recently changed first."""
        rows, bounds = page(resources.projects(ctx.client), offset=ctx.offset, limit=ctx.limit)
        columns = [
            ("ID", "id"),
            ("NAME", "name"),
            ("AVAILABLE", "available"),
            ("UPDATED", "updatedAt"),
        ]
        return Result(rows, page=bounds, text=lambda: output.table(rows, columns))

    @command(group, "show")
    def show(
        ctx,
        identity: str | None = typer.Argument(
            None, metavar="PROJECT", help="Project ID (default: --project)."
        ),
    ):
        """Show a project: its folder, latest dataset and record counts."""
        data = resources.project(ctx.client, identity or ctx.project())
        summary, dataset = data.get("project") or {}, data.get("dataset") or {}
        counts = data.get("scientificSummary") or {}

        def text():
            return output.fields(
                [
                    ("Project", f"{summary.get('name')} ({summary.get('id')})"),
                    ("Folder", summary.get("storagePath")),
                    ("State", summary.get("lifecycleState")),
                    ("Dataset", dataset.get("name") and f"{dataset['name']} ({dataset.get('id')})"),
                    ("Slides", dataset.get("slideCount")),
                    ("Patients", dataset.get("patientCount")),
                    (
                        "Records",
                        ", ".join(f"{key} {value}" for key, value in counts.items()) or None,
                    ),
                ]
            )

        return Result(data, text=text)

    @command(group, "roadmap")
    def roadmap(
        ctx,
        identity: str | None = typer.Argument(
            None, metavar="PROJECT", help="Project ID (default: --project)."
        ),
    ):
        """Each stage's status as the roadmap page shows it, and the suggested next stage."""
        from histopilot.client import resolve

        data = resolve.roadmap(ctx.client, identity or ctx.project())

        def text():
            rows = [
                {
                    "step": stage["step"],
                    "stage": stage["shortTitle"] + (" (optional)" if stage.get("optional") else ""),
                    "status": stage["status"],
                    "detail": stage["evidence"]
                    + (
                        f" · needs {', '.join(stage['blockers'])}"
                        if stage["blockers"] and stage["status"] != "complete"
                        else ""
                    ),
                }
                for stage in data["stages"]
            ]
            table = output.table(
                rows,
                [("STEP", "step"), ("STAGE", "stage"), ("STATUS", "status"), ("DETAIL", "detail")],
            )
            chosen = next((row for row in data["stages"] if row["id"] == data["next"]), None)
            closing = (
                f"Next: {chosen['step']} {chosen['shortTitle']}"
                if chosen
                else "Every required stage is complete."
            )
            return f"{table}\n\n{closing}"

        return Result(data, text=text)

    @command(group, "create", commit=True)
    def create(
        ctx,
        name: str = typer.Option(..., "--name", help="Project name."),
        folder: str = typer.Option(
            ..., "--folder", help="A new or empty folder on the service's machine."
        ),
        description: str = typer.Option("", "--description", help="A short description."),
        data: str | None = typer.Option(None, "--data", help="Folder of tables to import."),
        slides: str | None = typer.Option(None, "--slides", help="Folder of whole-slide images."),
        features: str | None = typer.Option(
            None, "--features", help="Folder of extracted features."
        ),
    ):
        """Create a project in a new or empty folder, with its source folders.

        Rerunning the same command after a lost answer returns the project the first run
        created."""
        body = {
            "name": name,
            "storagePath": folder,
            "description": description,
            **{
                key: value
                for key, value in (
                    ("dataPath", data),
                    ("slidePath", slides),
                    ("featurePath", features),
                )
                if value is not None
            },
        }
        return ctx.commit(
            {"create": body},
            lambda: ctx.client.operation(
                "POST", "/projects", body, prefix="project", operation_id=ctx.operation_id
            ),
            summary=f"Create the project {name!r} in {folder}.",
        )

    @command(group, "open", commit=True)
    def open_(
        ctx,
        folder: str = typer.Argument(..., help="An existing project's folder."),
    ):
        """Load an existing project's folder into this service."""
        return ctx.commit(
            {"open": folder},
            lambda: ctx.client.request("POST", "/projects/open", body={"path": folder}),
            summary=f"Load the project in {folder}.",
        )

    register_exposure(group)


MEANING = {
    "none": "AI agents see nothing of this project.",
    "metadata": "Agents see names, designs, statuses and aggregate results; patient and slide "
    "IDs are pseudonymized, and paths, free text, per-case attributes, images and exports "
    "are withheld. This lowers risk; it is not de-identification.",
    "full": "Agents see everything, including slide images and case-level exports. Use it only "
    "for data you are free to send to an AI provider.",
}


def register_exposure(group: typer.Typer) -> None:
    @command(group, "exposure", commit=True)
    def exposure(
        ctx,
        identity: str | None = typer.Argument(
            None, metavar="PROJECT", help="Project ID (default: --project)."
        ),
        level: str | None = typer.Option(None, "--set", help="none, metadata or full."),
    ):
        """Show or change what AI agents may see of a project (an admin change)."""
        project = identity or ctx.project()
        path = f"/projects/{segment(project)}/ai-exposure"
        current = ctx.client.get(path)["level"]
        if level is None:
            data = {"projectId": project, "level": current, "meaning": MEANING[current]}
            return Result(data, text=f"{current}: {MEANING[current]}")
        if level not in EXPOSURE_LEVELS:
            raise usage_error(f"Choose --set from: {', '.join(EXPOSURE_LEVELS)}.")
        preview = {"projectId": project, "from": current, "to": level, "meaning": MEANING[level]}
        return ctx.commit(
            preview,
            lambda: ctx.client.request(
                "PUT", path, body={"level": level, "expectedLevel": current}
            ),
            summary=f"Change AI exposure of {project} from {current} to {level}. {MEANING[level]}",
        )
