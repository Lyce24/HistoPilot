"""Recovery commands that work without an open project or a running web service."""

import json
import zipfile
from pathlib import Path
from uuid import uuid4

import typer

from histopilot.config import load_settings


def launch_archive_recovery(settings, archive, destination=None, *, executor=None):
    from histopilot.application.operations import _archive_manifest, _now, permitted_path
    from histopilot.application.portability_jobs import PortabilityExecutor
    from histopilot.storage.filesystem import LocalFilesystem
    from histopilot.storage.project_lock import StorageError, ensure_managed_directory, writer_lock
    from histopilot.storage.scientific import ScientificStore
    from histopilot.workers.packing_process import write_json

    roots = LocalFilesystem((settings.workspace, *settings.data_roots))
    archive = permitted_path(roots, str(archive), existing=True)
    ScientificStore._regular(archive)
    with zipfile.ZipFile(archive) as zipped:
        manifest = _archive_manifest(zipped)
    if destination is not None:
        destination = permitted_path(roots, str(destination))
        if destination.exists():
            raise StorageError("Restore into a new folder.", "PORTABILITY_INVALID", 422)
    executor = executor or PortabilityExecutor()
    if not executor.available():
        raise StorageError(
            "tmux is required for durable archive recovery.", "TMUX_UNAVAILABLE", 422
        )
    import hashlib

    operation = f"recovery:{uuid4()}"
    identity_hash = hashlib.sha256(f"{manifest['projectId']}:{operation}".encode()).hexdigest()
    registry = settings.workspace / "portability-jobs"
    ensure_managed_directory(registry)
    folder = registry / f"portability-{identity_hash}"
    with writer_lock(registry, timeout=5):
        ensure_managed_directory(folder)
        action = "restore" if destination is not None else "verify"
        plan = {
            "jobId": folder.name,
            "projectId": manifest["projectId"],
            "projectPath": manifest.get("originalPath", ""),
            "storageRoots": [str(root) for root in roots.roots],
            "sourceRoots": [str(root) for root in settings.data_roots],
            "request": {
                "action": action,
                "archivePath": str(archive),
                "destinationPath": str(destination) if destination is not None else None,
                "operationId": operation,
            },
        }
        state = {
            "id": folder.name,
            "projectId": manifest["projectId"],
            "action": action,
            "status": "starting",
            "createdAt": _now(),
            "sessionName": f"histopilot-archive-{identity_hash[:20]}",
            "logPath": str(folder / "worker.log"),
            "result": None,
            "error": None,
        }
        write_json(folder / "plan.json", plan)
        write_json(folder / "state.json", state)
        executor.launch(
            state["sessionName"],
            Path(__file__).parent / "workers" / "portability.py",
            folder / "plan.json",
        )
    return {
        **state,
        "statePath": str(folder / "state.json"),
        "planPath": str(folder / "plan.json"),
        "cancelPath": str(folder / "cancel.requested"),
    }


def register_archive_commands(app):
    def execute(archive, destination, workspace, data_root, config):
        try:
            settings = load_settings(
                config, workspace=workspace, data_roots=tuple(data_root) if data_root else None
            )
            job = launch_archive_recovery(settings, archive, destination)
            typer.echo(json.dumps(job, indent=2))
            typer.echo(f"Reconnect: tmux attach -t {job['sessionName']}")
            typer.echo(f"Cancel: touch {job['cancelPath']}")
            typer.echo(
                "Read the state JSON for the verified completion result before opening the restored folder."
            )
        except (ValueError, OSError, RuntimeError, zipfile.BadZipFile) as error:
            typer.echo(f"Cannot start archive recovery: {error}", err=True)
            raise typer.Exit(1) from error

    @app.command("restore-study")
    def restore_study(
        archive: Path = typer.Argument(..., exists=True, dir_okay=False),
        destination: Path = typer.Argument(..., help="New destination folder; never overwritten."),
        workspace: Path | None = typer.Option(None),
        data_root: list[Path] | None = typer.Option(
            None, help="Permitted archive/destination root; repeatable."
        ),
        config: Path | None = typer.Option(None),
    ):
        """Verify and restore an archive in tmux, without requiring the original project."""
        execute(archive, destination, workspace, data_root, config)

    @app.command("verify-study")
    def verify_study(
        archive: Path = typer.Argument(..., exists=True, dir_okay=False),
        workspace: Path | None = typer.Option(None),
        data_root: list[Path] | None = typer.Option(None),
        config: Path | None = typer.Option(None),
    ):
        """Verify every archive checksum in tmux without starting the web service."""
        execute(archive, None, workspace, data_root, config)
