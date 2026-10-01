"""Recovery commands that work without an open project or a running web service."""

import json
import zipfile
from pathlib import Path

import typer

from histopilot.config import load_settings


def launch_archive_recovery(settings, archive, destination=None, *, task_center=None):
    """Queue the recovery as a Task Center task; this starts the runner if it is stopped."""
    from histopilot.application.portability_jobs import submit_recovery

    return submit_recovery(settings, archive, destination, task_center=task_center)


def register_archive_commands(app):
    def execute(archive, destination, workspace, data_root, config):
        try:
            settings = load_settings(
                config, workspace=workspace, data_roots=tuple(data_root) if data_root else None
            )
            job = launch_archive_recovery(settings, archive, destination)
        except (ValueError, OSError, RuntimeError, zipfile.BadZipFile) as error:
            typer.echo(f"Cannot start archive recovery: {error}", err=True)
            raise typer.Exit(1) from error
        typer.echo(json.dumps(job, indent=2))
        if job["status"] == "failed":
            typer.echo(f"Cannot start archive recovery: {job['error']}", err=True)
            raise typer.Exit(1)
        typer.echo(
            "Queued in the Task Center. Follow it with `histopilot runner status` or on the "
            "Task Center page; `histopilot runner start` starts the runner if it is stopped."
        )
        typer.echo(f"Cancel: touch {job['cancelPath']}")
        typer.echo(
            "Read the state JSON for the verified completion result before opening the restored folder."
        )

    @app.command("restore-study", rich_help_panel="Local service")
    def restore_study(
        archive: Path = typer.Argument(..., exists=True, dir_okay=False),
        destination: Path = typer.Argument(..., help="New destination folder; never overwritten."),
        workspace: Path | None = typer.Option(None),
        data_root: list[Path] | None = typer.Option(
            None, help="Permitted archive/destination root; repeatable."
        ),
        config: Path | None = typer.Option(None),
    ):
        """Verify and restore an archive as a Task Center task, without the original project."""
        execute(archive, destination, workspace, data_root, config)

    @app.command("verify-study", rich_help_panel="Local service")
    def verify_study(
        archive: Path = typer.Argument(..., exists=True, dir_okay=False),
        workspace: Path | None = typer.Option(None),
        data_root: list[Path] | None = typer.Option(None),
        config: Path | None = typer.Option(None),
    ):
        """Verify every archive checksum as a Task Center task without the web service."""
        execute(archive, None, workspace, data_root, config)
