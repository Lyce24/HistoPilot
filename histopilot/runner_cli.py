"""`histopilot runner`: the machine-wide Task Center runner, without the web service."""

import json
import os
from pathlib import Path

import typer


def _use_state_dir(state_dir: Path | None) -> None:
    if state_dir is not None:
        from histopilot.taskcenter.paths import STATE_ENV

        os.environ[STATE_ENV] = str(state_dir.expanduser().resolve())


def runner_status() -> dict:
    """Read the task store and probe the runner lock directly (no HTTP)."""
    from histopilot.taskcenter.launcher import runner_alive
    from histopilot.taskcenter.paths import autostart_enabled, state_dir
    from histopilot.taskcenter.runner import code_current, foreign_checkout
    from histopilot.taskcenter.store import TaskStore

    store = TaskStore()
    runner = store.runner()
    return {
        "alive": runner_alive(store.path.parent / "runner.lock"),
        "stateDir": str(state_dir()),
        "store": str(store.path),
        "runner": runner,
        "codeCurrent": code_current(runner),
        "otherCheckout": foreign_checkout(runner),
        "autostart": autostart_enabled(),
        "paused": store.settings()["paused"],
        "counts": store.counts(),
    }


def register_runner_commands(app: typer.Typer) -> None:
    runner = typer.Typer(no_args_is_help=True, help="Machine-wide Task Center runner.")
    app.add_typer(runner, name="runner")
    state_option = typer.Option(
        None, "--state-dir", help="Task Center state directory (default: $XDG_STATE_HOME)."
    )

    @runner.command("run")
    def run(
        state_dir: Path | None = state_option,
        once: bool = typer.Option(False, "--once", hidden=True, help="Run a single pass."),
    ) -> None:
        """Run the runner in the foreground (this is what the tmux session executes)."""
        import threading

        from histopilot.storage.project_lock import StorageError

        _use_state_dir(state_dir)
        from histopilot.diagnostics import enable_stack_dumps
        from histopilot.taskcenter import paths
        from histopilot.taskcenter.runner import Runner, RunnerBusy, stop_signals

        enable_stack_dumps(paths.state_dir() / "runner-stacks.log")
        from histopilot.taskcenter.store import TaskStore

        # Installed before startup: a stop while the runner reconciles (which may probe the
        # training runtime for a while) ends it gracefully instead of killing it midway.
        with stop_signals(threading.Event()) as stop:
            try:
                instance = Runner(TaskStore())
                instance.start(stop=stop)
            except RunnerBusy as error:
                typer.echo(f"{error} Nothing to do.")
                return
            except (StorageError, OSError) as error:
                typer.echo(f"Cannot start the Task Center runner: {error}", err=True)
                raise typer.Exit(1) from error
            if once:
                try:
                    if not stop.is_set():
                        instance.tick()
                finally:
                    instance.shutdown()
                return
            instance.run_forever(stop=stop)

    @runner.command("start")
    def start(state_dir: Path | None = state_option) -> None:
        """Start the runner in tmux unless it is already running."""
        _use_state_dir(state_dir)
        from histopilot.taskcenter.launcher import ensure_runner

        typer.echo(json.dumps(ensure_runner(), indent=2))

    @runner.command("status")
    def status(
        state_dir: Path | None = state_option,
        json_output: bool = typer.Option(False, "--json", help="Emit machine-readable status."),
    ) -> None:
        """Show whether the runner is alive and how many tasks are in each state."""
        from histopilot.storage.project_lock import StorageError

        _use_state_dir(state_dir)
        try:
            report = runner_status()
        except (StorageError, OSError) as error:
            typer.echo(f"Cannot read the Task Center store: {error}", err=True)
            raise typer.Exit(1) from error
        if json_output:
            typer.echo(json.dumps(report, indent=2))
            return
        row = report["runner"] or {}
        state = "running" if report["alive"] else "stopped"
        typer.echo(f"Runner     {state}" + (f" (pid {row['pid']})" if report["alive"] else ""))
        if report["alive"] and report["otherCheckout"]:
            typer.echo(
                f"           runs the code of another checkout ({report['otherCheckout']}); "
                "restart it from this one to use this checkout's code"
            )
        elif report["alive"] and not report["codeCurrent"]:
            typer.echo("           code changed since it started; restart it to pick up changes")
        typer.echo(f"Heartbeat  {row.get('heartbeatAt') or 'never'}")
        typer.echo(f"State      {report['stateDir']}")
        typer.echo("Queue      " + ("paused" if report["paused"] else "admitting"))
        states = [f"{count} {name}" for name, count in report["counts"].items() if count]
        typer.echo("Tasks      " + (" · ".join(states) if states else "none"))

    @runner.command("stop")
    def stop(state_dir: Path | None = state_option) -> None:
        """Stop the runner. Running tasks keep running and are adopted on the next start."""
        _use_state_dir(state_dir)
        from histopilot.taskcenter.launcher import stop_runner

        typer.echo(json.dumps(stop_runner(), indent=2))
