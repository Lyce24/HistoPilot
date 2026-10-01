"""`histopilot confirm`: changes AI agents asked for, approved or declined by a person.

Approving claims the request, so it runs once however many people approve it, then replays
it with this session: the service checks it again, so a preview that went stale since the
agent asked is refused as usual. The request then records how the replay ended.
"""

import typer

from histopilot.client import ClientError
from histopilot.client.api import segment
from histopilot.client.paging import page

from . import output
from .common import Result, command


def register(app: typer.Typer) -> None:
    group = typer.Typer(no_args_is_help=True, help="Changes AI agents asked a person to approve.")
    app.add_typer(group, name="confirm", rich_help_panel="Service")

    @command(group, "list", lists=True)
    def list_(
        ctx,
        everything: bool = typer.Option(False, "--all", help="Include decided and expired ones."),
    ):
        """List agents' requests waiting for approval."""
        rows = ctx.client.get(
            "/agent-requests",
            query={"project": ctx.optional_project(), "state": None if everything else "pending"},
        )["requests"]
        shown, bounds = page(rows, offset=ctx.offset, limit=ctx.limit)
        columns = [
            ("ID", "id"),
            ("STATE", "state"),
            ("REQUEST", lambda row: f"{row['method']} {row['path']}"),
            ("TOKEN", "tokenName"),
            ("EXPIRES", "expiresAt"),
        ]
        return Result(
            shown, page=bounds, text=lambda: output.table(shown, columns, empty="Nothing waiting.")
        )

    @command(group, "show")
    def show(ctx, request: str = typer.Argument(..., help="Request ID.")):
        """Show what an agent asked to change."""
        data = ctx.client.get(f"/agent-requests/{segment(request)}")

        def text():
            target = data["path"] + (f"?{data['query']}" if data.get("query") else "")
            head = output.fields(
                [
                    ("Request", data["id"]),
                    ("State", data["state"]),
                    ("Token", data.get("tokenName") or data.get("tokenId")),
                    ("Asks", f"{data['method']} {target}"),
                    ("Created", data.get("createdAt")),
                    ("Expires", data.get("expiresAt")),
                ]
            )
            return f"{head}\n\nBody:\n{output.pretty(data.get('body'))}"

        return Result(data, text=text)

    @command(group, "approve", commit=True)
    def approve(ctx, request: str = typer.Argument(..., help="Request ID.")):
        """Approve an agent's request: it runs now, as you."""
        entry = ctx.client.get(f"/agent-requests/{segment(request)}")
        if entry["state"] != "pending":
            # Only a pending request may run; the service settles each one once.
            raise ClientError(
                f"This agent request is already {entry['state']}; nothing was run.",
                code="AGENT_REQUEST_SETTLED",
                kind="refused",
            )
        target = entry["path"] + (f"?{entry['query']}" if entry.get("query") else "")
        path = f"/agent-requests/{segment(request)}"

        def settle(outcome: str, status: int | None) -> None:
            body = {"outcome": outcome, "status": status}
            ctx.client.request("POST", f"{path}/resolve", body=body)

        def send():
            ctx.client.request("POST", f"{path}/claim")
            try:
                result = ctx.client.request(entry["method"], target, body=entry.get("body"))
            except Exception as error:
                # Refused: nothing changed. No answer or a fault: it may have run.
                status = getattr(error, "status", None)
                refused = status is not None and status < 500
                settle("failed" if refused else "unknown", status)
                raise
            settle("approved", 200)
            return result

        return ctx.commit(
            {
                "request": entry["id"],
                "method": entry["method"],
                "path": target,
                "body": entry.get("body"),
            },
            send,
            summary=f"Approve {entry['tokenName'] or entry['tokenId']}'s request: "
            f"{entry['method']} {target}.",
        )

    @command(group, "decline", commit=True)
    def decline(ctx, request: str = typer.Argument(..., help="Request ID.")):
        """Decline an agent's request; nothing changes."""
        return ctx.commit(
            {"request": request, "outcome": "declined"},
            lambda: ctx.client.request(
                "POST", f"/agent-requests/{segment(request)}/resolve", body={"outcome": "declined"}
            ),
            summary=f"Decline agent request {request}.",
        )
