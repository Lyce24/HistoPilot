"""`histopilot cleanup`, `backup` and `sources`: organizing records and moving studies."""

from pathlib import Path

import typer

from histopilot.client.api import segment
from histopilot.client.errors import usage_error
from histopilot.client.paging import page
from histopilot.client.resources import project_path
from histopilot.client.states import run_state
from histopilot.client.wait import wait

from . import output
from .common import Result, command

ACTIONS = ("archive", "trash", "restore")
_KEY_TYPES = ("project", "dataset", "configuration", "draft", "packing", "extraction")


def cleanup_key(value: str) -> str:
    """A cleanup key from ``type:id``, or from an ID whose prefix names its type."""
    if ":" in value:
        return value
    for kind in ("project", "dataset", "configuration", "draft"):
        if value.startswith(f"{kind}-"):
            return f"{kind}:{value}"
    raise usage_error(
        f"Name {value!r} as TYPE:ID with TYPE one of {', '.join(_KEY_TYPES)}; "
        "`histopilot cleanup list` shows every key."
    )


def _blockers(preview: dict) -> list[dict]:
    """The preview's blockers as findings: {code, message, severity, keys}."""
    return [
        {
            "code": item.get("code") or "CLEANUP_BLOCKED",
            "message": item.get("message") or "This selection cannot be applied.",
            "severity": "error",
            "keys": item.get("keys") or [],
        }
        for item in preview.get("blockers") or []
        if isinstance(item, dict)
    ]


def _cleanup_summary(action: str, preview: dict) -> str:
    count = preview.get("recordCount") or len(preview.get("keys") or [])
    records = "record" if count == 1 else "records"
    required = preview.get("requiredKeys") or []
    also = f" It also includes {len(required)} dependent records." if required else ""
    return f"{action.capitalize()} {count} {records}: {', '.join(preview['keys'])}.{also}"


def _archive_job(item: dict) -> dict:
    return {**item, "runState": run_state(item.get("state") or item.get("status"))}


def register(app: typer.Typer) -> None:
    cleanup = typer.Typer(no_args_is_help=True, help="Archive, trash and restore records.")
    app.add_typer(cleanup, name="cleanup", rich_help_panel="Cleanup and backups")
    backup = typer.Typer(no_args_is_help=True, help="Study archives: export, verify, restore.")
    app.add_typer(backup, name="backup", rich_help_panel="Cleanup and backups")
    sources = typer.Typer(no_args_is_help=True, help="The project's registered source folders.")
    app.add_typer(sources, name="sources", rich_help_panel="Cleanup and backups")

    # Cleanup --------------------------------------------------------------------------

    @command(cleanup, "list", lists=True)
    def cleanup_list(ctx):
        """List the project's records with their cleanup keys and states."""
        inventory = ctx.client.get(f"{project_path(ctx.project())}/cleanup")
        rows, bounds = page(inventory.get("items") or [], offset=ctx.offset, limit=ctx.limit)
        columns = [("KEY", "key"), ("KIND", "kind"), ("NAME", "name"), ("STATE", "state")]
        return Result(rows, page=bounds, text=lambda: output.table(rows, columns))

    def review(ctx, action: str, keys: list[str]) -> dict:
        if action not in ACTIONS:
            raise usage_error(f"Choose --action from: {', '.join(ACTIONS)}.")
        selection = {"action": action, "keys": sorted({cleanup_key(key) for key in keys})}
        return ctx.client.request(
            "POST", f"{project_path(ctx.project())}/cleanup/preview", body=selection
        )

    @command(cleanup, "preview")
    def cleanup_preview(
        ctx,
        keys: list[str] = typer.Argument(..., metavar="KEY...", help="Cleanup keys or IDs."),
        action: str = typer.Option(..., "--action", help="archive, trash or restore."),
    ):
        """Review what an archive, trash or restore would include, and what blocks it."""
        preview = review(ctx, action, keys)
        return Result(preview, text=lambda: output.record(preview))

    @command(cleanup, "apply", commit=True)
    def cleanup_apply(
        ctx,
        keys: list[str] = typer.Argument(..., metavar="KEY...", help="Cleanup keys or IDs."),
        action: str = typer.Option(..., "--action", help="archive, trash or restore."),
    ):
        """Archive, trash or restore records after reviewing the preview."""
        preview = review(ctx, action, keys)
        project = ctx.project()
        body = {"action": preview["action"], "keys": preview["keys"]}
        body["previewHash"] = preview["previewHash"]
        return ctx.commit(
            preview,
            lambda: ctx.client.operation(
                "POST",
                f"{project_path(project)}/cleanup/apply",
                body,
                prefix=f"cleanup-{action}",
                operation_id=ctx.operation_id,
            ),
            findings=_blockers(preview),
            summary=_cleanup_summary(action, preview),
        )

    # Backups --------------------------------------------------------------------------

    @command(backup, "list", lists=True)
    def backup_list(ctx):
        """List this project's archive jobs."""
        jobs = (
            ctx.client.get(f"{project_path(ctx.project())}/operations/archives").get("jobs") or []
        )
        rows, bounds = page([_archive_job(job) for job in jobs], offset=ctx.offset, limit=ctx.limit)
        columns = [
            ("ID", "id"),
            ("ACTION", lambda row: (row.get("request") or row).get("action")),
            ("STATE", "state"),
            ("RUN STATE", "runState"),
            ("UPDATED", "updatedAt"),
        ]
        return Result(rows, page=bounds, text=lambda: output.table(rows, columns))

    @command(backup, "show")
    def backup_show(ctx, job: str = typer.Argument(..., help="Archive job ID.")):
        """Show one archive job."""
        path = f"{project_path(ctx.project())}/operations/archives/{segment(job)}"
        data = _archive_job(ctx.client.get(path))
        return Result(data, text=lambda: output.record(data))

    def archive(ctx, action: str, archive_path: Path, destination: Path | None) -> Result:
        project = ctx.project()
        body = {"action": action, "archivePath": str(archive_path)}
        if destination is not None:
            body["destinationPath"] = str(destination)
        preview = {"routeClass": "admin", "project": project, **body}

        def send():
            job = ctx.client.operation(
                "POST",
                f"{project_path(project)}/operations/archives",
                body,
                prefix=f"archive-{action}",
                operation_id=ctx.operation_id,
            )
            if not ctx.wait:
                return job
            identity = job.get("id") or job.get("jobId")
            path = f"{project_path(project)}/operations/archives/{segment(identity)}"
            return wait(ctx.client, lambda: _archive_job(ctx.client.get(path)))

        return ctx.commit(preview, send, summary=f"{action.capitalize()} {archive_path}.")

    @command(backup, "export", commit=True, waits=True)
    def backup_export(
        ctx,
        archive_path: Path = typer.Argument(
            ..., metavar="ARCHIVE", help="New archive file on the service's machine."
        ),
    ):
        """Export the project to a study archive (runs as a Task Center task)."""
        return archive(ctx, "export", archive_path, None)

    @command(backup, "verify", commit=True, waits=True)
    def backup_verify(
        ctx,
        archive_path: Path = typer.Argument(
            ..., metavar="ARCHIVE", help="Archive file on the service's machine."
        ),
    ):
        """Verify every checksum in a study archive."""
        return archive(ctx, "verify", archive_path, None)

    @command(backup, "restore", commit=True, waits=True)
    def backup_restore(
        ctx,
        archive_path: Path = typer.Argument(
            ..., metavar="ARCHIVE", help="Archive file on the service's machine."
        ),
        destination: Path = typer.Option(..., "--to", help="New study folder to restore into."),
    ):
        """Restore a study archive into a new folder."""
        return archive(ctx, "restore", archive_path, destination)

    for action, summary in (
        ("cancel", "Cancel an archive job."),
        ("retry", "Requeue a failed or interrupted archive job."),
    ):
        _archive_action(backup, action, summary)

    # Sources --------------------------------------------------------------------------

    @command(sources, "list")
    def sources_list(ctx):
        """List registered source folders and whether they are still where they were."""
        data = ctx.client.get(f"{project_path(ctx.project())}/operations/sources")
        rows = data.get("sources") or []
        columns = [("ID", "id"), ("ROLE", "role"), ("PATH", "path"), ("AVAILABLE", "available")]
        return Result(
            data, text=lambda: output.table(rows, columns, empty="No registered sources.")
        )

    @command(sources, "add", commit=True)
    def sources_add(
        ctx,
        folder: Path = typer.Argument(..., help="Folder under a data root."),
        role: str = typer.Option(..., "--role", help="data, slides or features."),
    ):
        """Register a read-only source folder; nothing is scanned or copied."""
        project = ctx.project()
        body = {"path": str(folder), "role": role}
        return ctx.commit(
            {"routeClass": "admin", "project": project, **body},
            lambda: ctx.client.request("POST", f"{project_path(project)}/sources", body=body),
            summary=f"Register {folder} as a {role} source of {project}.",
        )

    @command(sources, "relink", commit=True)
    def sources_relink(
        ctx,
        source: str = typer.Argument(..., help="Source ID."),
        expected: Path = typer.Option(..., "--from", help="The path it is registered at now."),
        replacement: Path = typer.Option(..., "--to", help="Where the folder moved."),
    ):
        """Point a moved source folder at its new place."""
        project = ctx.project()
        body = {
            "sourceId": source,
            "expectedPath": str(expected),
            "replacementPath": str(replacement),
        }
        return ctx.commit(
            {"routeClass": "admin", "project": project, **body},
            lambda: ctx.client.request(
                "POST", f"{project_path(project)}/operations/sources/relink", body=body
            ),
            summary=f"Relink {source}: {expected} → {replacement}.",
        )


def _archive_action(group: typer.Typer, action: str, summary: str) -> None:
    @command(group, action, commit=True, help=summary)
    def change(ctx, job: str = typer.Argument(..., help="Archive job ID.")):
        path = f"{project_path(ctx.project())}/operations/archives/{segment(job)}"
        current = _archive_job(ctx.client.get(path))
        preview = {"action": action, "job": {key: current.get(key) for key in ("id", "state")}}
        return ctx.commit(
            preview,
            lambda: ctx.client.request("POST", f"{path}/{action}"),
            summary=f"{action.capitalize()} archive job {job} ({current.get('state')}).",
        )
