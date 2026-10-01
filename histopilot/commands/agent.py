"""`histopilot agent serve`: the MCP server for AI agents, over stdio.

On stdio, standard output belongs to the MCP protocol, so this command prints nothing there;
problems go to standard error with the contract's exit codes.
"""

import os
from pathlib import Path

import typer

from histopilot.client import DEFAULT_URL, ClientError


def register(app: typer.Typer) -> None:
    group = typer.Typer(no_args_is_help=True, help="Agent tools: an MCP server over stdio.")
    app.add_typer(group, name="agent", rich_help_panel="Service")

    @group.command("serve")
    def serve(
        url: str = typer.Option(DEFAULT_URL, "--url", envvar="HISTOPILOT_URL", help="Service URL."),
        token_file: Path | None = typer.Option(
            None, "--token-file", help="File holding a scoped token (default: HISTOPILOT_TOKEN)."
        ),
    ) -> None:
        """Serve HistoPilot's agent tools over stdio with a scoped token, never full access."""
        try:
            token = (
                token_file.expanduser().read_text(encoding="utf-8").strip()
                if token_file
                else os.environ.get("HISTOPILOT_TOKEN", "").strip()
            )
        except OSError as error:
            typer.echo(f"Cannot read {token_file}: {error.strerror or error}", err=True)
            raise typer.Exit(2) from error
        if any(character.isspace() or not character.isprintable() for character in token):
            # Never echo it: a token file of two lines may still hold a working secret.
            typer.echo(
                "The token must be one line without spaces. Check the token file or "
                "HISTOPILOT_TOKEN.",
                err=True,
            )
            raise typer.Exit(2)
        try:
            from histopilot.agent import server
        except ImportError as error:
            typer.echo(
                "The agent tools need the optional `agent` extra, best in an environment of "
                "its own: `UV_PROJECT_ENVIRONMENT=.venv-agent uv sync --locked --extra agent`.",
                err=True,
            )
            raise typer.Exit(2) from error
        try:
            session = server.connect(
                url, token, host_header=os.environ.get("HISTOPILOT_HOST_HEADER") or None
            )
        except ValueError as error:
            typer.echo(str(error), err=True)
            raise typer.Exit(2) from error
        except ClientError as error:
            typer.echo(f"{error.message} [{error.code}]", err=True)
            raise typer.Exit(error.exit_code) from error
        server.build(session).run("stdio")
