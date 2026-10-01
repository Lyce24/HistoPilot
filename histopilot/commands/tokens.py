"""`histopilot token`: scoped access tokens for AI agents (admin changes)."""

import typer

from histopilot.api.route_classes import (
    DEFAULT_TOKEN_SCOPES,
    MAX_TOKEN_DAYS,
    TOKEN_DAYS,
    TOKEN_SCOPES,
)
from histopilot.client.api import segment
from histopilot.client.errors import usage_error
from histopilot.client.paging import page

from . import output
from .common import Result, command


def register(app: typer.Typer) -> None:
    group = typer.Typer(no_args_is_help=True, help="Scoped access tokens for AI agents.")
    app.add_typer(group, name="token", rich_help_panel="Service")

    @command(group, "create", commit=True)
    def create(
        ctx,
        scope: list[str] = typer.Option(
            list(DEFAULT_TOKEN_SCOPES),
            "--scope",
            help=", ".join(TOKEN_SCOPES) + "; repeatable. Each includes the ones before it.",
        ),
        name: str = typer.Option("", "--name", help="What the token is for, such as a chat app."),
        days: int = typer.Option(
            TOKEN_DAYS, "--days", min=1, max=MAX_TOKEN_DAYS, help="Lifetime in days."
        ),
    ):
        """Create a token for one project; it is printed once and stored only as a hash."""
        unknown = sorted(set(scope) - set(TOKEN_SCOPES))
        if unknown:
            raise usage_error(
                f"Unknown scope {', '.join(unknown)}; choose from {', '.join(TOKEN_SCOPES)}."
            )
        project = ctx.project()
        body = {"projectId": project, "scopes": scope, "name": name, "days": days}
        warning = (
            " With commit scope, the agent's changes still wait for a person's approval."
            if "commit" in scope
            else ""
        )
        return ctx.commit(
            {"routeClass": "admin", **body},
            lambda: ctx.client.request("POST", "/tokens", body=body),
            summary=f"Create a {', '.join(scope)} token for {project}, valid {days} days.{warning}",
        )

    @command(group, "list", lists=True)
    def list_(ctx):
        """List tokens: project, scopes, expiry, last use. Secrets are never shown."""
        project = ctx.optional_project()
        rows = ctx.client.get("/tokens", query={"project": project})["tokens"]
        shown, bounds = page(rows, offset=ctx.offset, limit=ctx.limit)
        columns = [
            ("ID", "id"),
            ("NAME", "name"),
            ("PROJECT", "projectId"),
            ("SCOPES", lambda row: ",".join(row.get("scopes") or [])),
            ("STATE", "state"),
            ("EXPIRES", "expiresAt"),
            ("LAST USED", "lastUsedAt"),
        ]
        return Result(shown, page=bounds, text=lambda: output.table(shown, columns))

    @command(group, "revoke", commit=True)
    def revoke(
        ctx, token: str = typer.Argument(..., help="Token ID from `histopilot token list`.")
    ):
        """Revoke a token; it stops working on its next request."""
        return ctx.commit(
            {"routeClass": "admin", "tokenId": token},
            lambda: ctx.client.request("POST", f"/tokens/{segment(token)}/revoke"),
            summary=f"Revoke token {token}.",
        )
