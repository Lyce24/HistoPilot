"""`histopilot login`: the sign-in link of a service started with `--login`."""

from urllib.parse import urlsplit

import typer

from histopilot.api import login as secrets
from histopilot.client.errors import usage_error

from .common import Result, command


def register(app: typer.Typer) -> None:
    group = typer.Typer(no_args_is_help=True, help="Sign-in for a service started with --login.")
    app.add_typer(group, name="login", rich_help_panel="Service")

    def port_of(ctx) -> int:
        return urlsplit(ctx.url).port or 80

    @command(group, "url")
    def url(ctx):
        """Print the link that signs a browser in; only this OS user can read it."""
        port = port_of(ctx)
        secret = secrets.read_secret(port)
        if not secret:
            raise usage_error(
                f"No sign-in is set up for port {port}; start the service with `--login`.",
                code="LOGIN_NOT_SET_UP",
            )
        link = f"{ctx.url}/?login={secret}"
        return Result({"url": link}, text=link)

    @command(group, "rotate", commit=True)
    def rotate(ctx):
        """Replace the secret: every browser must open the new link to sign in again."""
        port = port_of(ctx)
        return ctx.commit(
            {"port": port, "action": "rotate"},
            lambda: {"url": f"{ctx.url}/?login={secrets.rotate_secret(port)}"},
            summary=f"Replace the sign-in secret of the service on port {port}.",
        )
