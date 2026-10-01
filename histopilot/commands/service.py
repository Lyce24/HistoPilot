"""Commands outside the noun-verb grid: status, api, schema and use."""

import json
from pathlib import Path

import typer

from histopilot.api.route_classes import ADMIN, COMMIT, READ, route_class
from histopilot.client import resources, specs
from histopilot.client.api import API
from histopilot.client.errors import usage_error

from . import output
from .common import Result, command

# `histopilot schema KIND`: every spec kind, and request bodies `histopilot api` may send.
SCHEMAS = {
    **specs.MODELS,
    "project": ("histopilot.schemas.workspace", "ProjectRequest"),
    "predictor": ("histopilot.schemas.predictors", "PredictorSelection"),
    "run": ("histopilot.schemas.predictors", "EvaluationRunSelection"),
    "cleanup": ("histopilot.schemas.lifecycle", "CleanupSelection"),
    "archive": ("histopilot.schemas.operations", "PortabilityRequest"),
}


def register(app: typer.Typer) -> None:
    @command(app, "status", rich_help_panel="Service")
    def status(ctx):
        """Summarize the service, its Task Center runner and the default project."""
        from histopilot.version_info import git_revision

        data = resources.status(ctx.client)
        access = data["access"]
        data["project"] = access["projectId"] if access else ctx.optional_project()
        data["cliRevision"] = git_revision()
        warnings = []
        service, local = data["serviceRevision"], data["cliRevision"]
        if service and local and service != local:
            warnings.append(
                {
                    "code": "VERSION_SKEW",
                    "message": f"The service runs revision {service[:10]}, this CLI {local[:10]}; "
                    "commands follow this CLI's contract.",
                    "severity": "warning",
                }
            )
        elif data["features"] is None:
            warnings.append(
                {
                    "code": "SERVICE_VERSION_UNKNOWN",
                    "message": "This service predates the version route; newer commands may fail.",
                    "severity": "warning",
                }
            )
        runner, tasks = data["runner"], data["tasks"]

        def text():
            live = sum(
                tasks.get(state, 0)
                for state in ("blocked", "queued", "starting", "running", "stopping")
            )
            return output.fields(
                [
                    ("Service", f"{data['url']} · HistoPilot {data['version']}"),
                    ("Workspace", data["workspace"]),
                    *_project_lines(data),
                    (
                        "Runner",
                        runner.get("state") or ("running" if runner.get("alive") else "stopped"),
                    ),
                    ("Queue", ("paused · " if data["queuePaused"] else "") + f"{live} live tasks"),
                    ("Failures", f"{data['recentFailures'] or 0} in the last 24 hours"),
                    ("Extraction", "ready" if data["extraction"]["ready"] else "not ready"),
                ]
            )

        return Result(data, warnings=warnings, text=text)

    @command(app, "api", rich_help_panel="Service", commit=True)
    def api(
        ctx,
        method: str = typer.Argument(..., help="GET, POST, PUT, PATCH or DELETE."),
        path: str = typer.Argument(..., help="Path under /api/v1, such as /projects."),
        data: str | None = typer.Option(
            None, "--data", "-d", help="JSON request body, or @FILE to read it from a file."
        ),
        query: list[str] = typer.Option([], "--query", "-q", help="KEY=VALUE; repeatable."),
    ):
        """Send one raw request. Commit and admin routes ask first, like every command."""
        method = method.upper()
        if method not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
            raise usage_error(f"Unsupported method {method}.")
        if not path.startswith("/"):
            raise usage_error("Give the path from its first slash, such as /projects.")
        target = path if path.startswith("/api/") else API + path
        body = _json_body(data)
        values = {}
        for item in query:
            key, separator, value = item.partition("=")
            if not separator or not key:
                raise usage_error(f"Write queries as KEY=VALUE, not {item!r}.")
            values[key] = value
        # A route missing from the table is unknown to the service too; a GET changes nothing.
        kind = route_class(method, target) or (READ if method == "GET" else COMMIT)
        if kind in (COMMIT, ADMIN):
            return ctx.commit(
                {"request": f"{method} {target}", "routeClass": kind, "body": body},
                lambda: ctx.client.request(method, target, query=values, body=body),
                summary=f"{method} {target} changes state ({kind}).",
            )
        return Result(ctx.client.request(method, target, query=values, body=body))

    @command(app, "schema", rich_help_panel="Service")
    def schema(
        ctx,
        kind: str | None = typer.Argument(None, help="Spec kind; omit to list the kinds."),
    ):
        """Print the JSON Schema of a spec kind: the fields its spec file may hold."""
        if kind is None:
            rows = [{"kind": key, "model": name} for key, (_, name) in SCHEMAS.items()]
            return Result(
                rows, text=lambda: output.table(rows, [("KIND", "kind"), ("MODEL", "model")])
            )
        return Result(specs.schema(kind, SCHEMAS))

    @command(app, "use", rich_help_panel="Service")
    def use(
        ctx,
        identity: str | None = typer.Argument(
            None, metavar="PROJECT", help="Project ID; omit to show the saved one."
        ),
        clear: bool = typer.Option(False, "--clear", help="Forget the saved project."),
    ):
        """Save a default project for this service URL, or show the saved one."""
        from histopilot.client import local

        if clear:
            local.save_project(ctx.url, None)
            return Result({"url": ctx.url, "project": None}, text="No default project.")
        if identity is None:
            saved = local.saved_project(ctx.url)
            return Result(
                {"url": ctx.url, "project": saved},
                text=saved or "No default project; save one with `histopilot use PROJECT`.",
            )
        summary = resources.project(ctx.client, identity).get("project") or {}
        local.save_project(ctx.url, identity)
        data = {"url": ctx.url, "project": identity, "name": summary.get("name")}
        return Result(
            data, text=f"Default project for {ctx.url}: {summary.get('name') or identity}"
        )


def _project_lines(data: dict) -> list[tuple[str, str]]:
    access = data["access"]
    if access:
        return [
            ("Project", f"{access['projectName']} ({access['projectId']})"),
            ("Access", f"agent token · this project only · exposure {access['exposure']}"),
        ]
    return [("Project", data["project"] or "none saved; see `histopilot use`")]


def _json_body(value: str | None):
    if value is None:
        return None
    try:
        if value.startswith("@"):
            return json.loads(Path(value[1:]).expanduser().read_text(encoding="utf-8"))
        return json.loads(value)
    except OSError as error:
        raise usage_error(f"Cannot read {value[1:]}: {error.strerror or error}") from error
    except ValueError as error:
        raise usage_error(f"The request body is not valid JSON: {error}") from error
