"""Options every command shares, confirmation, and the loop from results to exit codes.

See docs/cli-contract.md: `--json` prints one envelope on stdout; progress and prompts go
to stderr; each error kind has its own exit code.
"""

import functools
import hashlib
import inspect
import json
import os
import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import typer
from typer.core import TyperGroup

from histopilot.client import DEFAULT_URL, Client, ClientError, resources, specs
from histopilot.client.errors import usage_error
from histopilot.client.paging import DEFAULT_LIMIT, MAX_LIMIT

from . import output

# Tests plug in an in-process service: a callable from the service URL to a transport.
TRANSPORT_FACTORY: Callable[[str], Any] | None = None

# Options of every command that writes a spec file.
OUTPUT = typer.Option(None, "--output", "-o", help="Write a file (default: print it).")
FORCE = typer.Option(False, "--force", help="Replace an existing file.")


@dataclass
class Result:
    data: Any
    warnings: list[dict] = field(default_factory=list)
    page: dict | None = None
    # How people see it; the JSON envelope never depends on this.
    text: str | Callable[[], str] | None = None


def spec_result(
    kind: str,
    body: dict,
    *,
    note: str = "",
    destination: Path | None,
    force: bool,
    warnings=(),
    extra: dict | None = None,
) -> Result:
    """A spec file, printed or written to ``destination``. With --json the data is the body,
    or ``{"path", "spec"}`` once written; ``extra`` fields sit beside the spec either way."""
    text = specs.render(kind, body, note=note)
    if destination is not None:
        saved = output.write_file(text.encode("utf-8"), destination, force=force)
        data = {"path": str(saved), "spec": body, **(extra or {})}
        return Result(data, warnings=list(warnings), text=f"Wrote {saved}.")
    data = {"spec": body, **extra} if extra else body
    return Result(data, warnings=list(warnings), text=text.rstrip("\n"))


class _DryRun(Exception):
    def __init__(self, preview, warnings, summary):
        super().__init__("dry run")
        self.preview, self.warnings, self.summary = preview, warnings, summary


def interactive() -> bool:
    return sys.stdin.isatty() and sys.stderr.isatty()


class Context:
    def __init__(
        self,
        *,
        url: str | None,
        project: str | None,
        json_mode: bool,
        timeout: float | None,
        yes: bool = False,
        dry_run: bool = False,
        preview_hash: str | None = None,
        operation_id: str | None = None,
        wait: bool = False,
        limit: int = DEFAULT_LIMIT,
        offset: int = 0,
    ):
        self.url = (url or DEFAULT_URL).rstrip("/")
        self._project = project
        self.json_mode = json_mode
        self.timeout = timeout
        self.yes = yes
        self.dry_run = dry_run
        self.preview_hash = preview_hash
        self.operation_id = operation_id
        self.wait = wait
        self.limit = limit
        self.offset = offset
        self._client: Client | None = None
        self._token_project: str | None = None

    @property
    def client(self) -> Client:
        if self._client is None:
            from histopilot.client import local

            try:
                transport = TRANSPORT_FACTORY(self.url) if TRANSPORT_FACTORY else None
                self._client = Client(
                    self.url,
                    transport=transport,
                    timeout=self.timeout,
                    host_header=os.environ.get("HISTOPILOT_HOST_HEADER") or None,
                    journal=local.journal(),
                    token=os.environ.get("HISTOPILOT_TOKEN") or None,
                )
            except ValueError as error:
                raise usage_error(str(error)) from error
        return self._client

    def optional_project(self) -> str | None:
        """--project or HISTOPILOT_PROJECT, else a scoped token's one project, else the
        project saved by `histopilot use`."""
        if self._project:
            return self._project
        if self.client.scoped:
            if self._token_project is None:
                self._token_project = self.client.get("/access")["projectId"]
            return self._token_project
        from histopilot.client import local

        return local.saved_project(self.url)

    def project(self) -> str:
        project = self.optional_project()
        if not project:
            raise usage_error(
                "Name a project with --project or HISTOPILOT_PROJECT, or save one with "
                "`histopilot use PROJECT`.",
                code="PROJECT_REQUIRED",
            )
        return project

    def experiment(self, name: str) -> str:
        """The ID of the experiment named by its ID or by its name, as `experiment list`
        shows them."""
        return resources.experiment_id(self.client, self.project(), name)

    def commit(
        self,
        preview: dict,
        send: Callable[[], Any],
        *,
        findings=(),
        summary: str | None = None,
    ) -> Result:
        """Review, confirm, then send: the path of every commit and admin command.

        The preview carries a ``previewHash``: the service's when it has one, otherwise a
        hash of the preview itself. ``--preview-hash`` must match it.
        """
        preview = dict(preview)
        if not preview.get("previewHash"):
            preview["previewHash"] = preview_digest(preview)
        if self.preview_hash and self.preview_hash != preview["previewHash"]:
            raise ClientError(
                "The preview changed since it was reviewed; review it again.",
                code="PREVIEW_CHANGED",
                kind="conflict",
                data={"preview": preview, "result": None},
            )
        self.confirm(preview, findings=findings, summary=summary)
        warnings = [item for item in findings if item.get("severity") != "error"]
        client, answered = self.client, self.client.answer_count
        try:
            result = send()
        except (ClientError, KeyboardInterrupt) as error:
            # Refused, parked for a person, or stopped while waiting for the work it
            # started: the preview still comes back, and so does the commit's answer.
            committed = client.last_answer if client.answer_count > answered else None
            if isinstance(error, KeyboardInterrupt):
                raise ClientError(
                    "Interrupted.",
                    code="INTERRUPTED",
                    kind="interrupted",
                    data={"preview": preview, "result": committed},
                ) from None
            if error.code == "CONFIRMATION_PENDING":
                error.data = {"preview": preview, "result": None, "request": error.data}
            elif error.data is None:
                error.data = {"preview": preview, "result": committed}
            raise
        return Result(
            {"preview": preview, "result": result},
            warnings=warnings,
            text=lambda: _committed_text(summary, result),
        )

    def confirm(self, preview, *, findings=(), summary: str | None = None) -> None:
        """Gate a commit: blocking findings refuse, --dry-run stops, and a person or
        --yes must agree. See docs/cli-contract.md#confirmation."""
        blocking = [item for item in findings if item.get("severity") == "error"]
        pending = {"preview": preview, "result": None}
        if blocking:
            raise ClientError(
                "The preview has blocking findings; nothing was changed.",
                code="PREVIEW_BLOCKED",
                kind="refused",
                findings=blocking,
                data=pending,
            )
        notes = [item for item in findings if item.get("severity") != "error"]
        if self.dry_run:
            raise _DryRun(preview, notes, summary)
        if self.yes:
            return
        if self.json_mode or not interactive():
            raise ClientError(
                "This change needs confirmation. Review data.preview, then rerun with --yes.",
                code="CONFIRMATION_REQUIRED",
                kind="unconfirmed",
                data=pending,
            )
        output.note(summary or output.pretty(preview))
        for item in notes:
            output.note(finding_line(item))
        try:
            agreed = typer.confirm("Proceed?", default=False, err=True)
        except typer.Abort as stop:
            # Ctrl-C at the prompt interrupts the command; the end of input answers no.
            if isinstance(stop.__context__, KeyboardInterrupt):
                raise KeyboardInterrupt from None
            agreed = False
        if not agreed:
            raise ClientError(
                "Not confirmed; nothing was changed.",
                code="CONFIRMATION_DECLINED",
                kind="unconfirmed",
                data=pending,
            )


def finding_line(item: dict) -> str:
    label = "Warning" if item.get("severity") == "warning" else "Note"
    return f"{label}: {item.get('message')} [{item.get('code')}]"


def _committed_text(summary: str | None, result) -> str:
    """What a person reads after a change: what was asked, and what the service answered."""
    done = "Done."
    if isinstance(result, dict):
        state = result.get("state") or result.get("status") or result.get("runState")
        shown = [value for value in (result.get("id"), state) if value]
        if shown:
            done = f"Done: {' · '.join(str(value) for value in shown)}."
    return f"{summary}\n{done}" if summary else done


def _dry_run_text(stop: "_DryRun") -> str:
    lines = [stop.summary] if stop.summary else []
    lines += [output.pretty(stop.preview), "Nothing was changed (--dry-run)."]
    if stop.preview.get("previewHash"):
        lines.append(
            f"Confirm exactly this preview with --preview-hash {stop.preview['previewHash']}."
        )
    return "\n".join(lines)


def preview_digest(preview) -> str:
    """A hash of what a person reviewed, for previews the service does not hash itself."""
    canonical = json.dumps(preview, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# Shared options -----------------------------------------------------------------------

_COMMON = [
    (
        "url",
        str | None,
        typer.Option(
            None,
            "--url",
            envvar="HISTOPILOT_URL",
            help=f"Running service, loopback only (default: {DEFAULT_URL}).",
            show_default=False,
        ),
    ),
    (
        "project",
        str | None,
        typer.Option(
            None,
            "--project",
            envvar="HISTOPILOT_PROJECT",
            help="Project ID (default: the one saved by `histopilot use`).",
            show_default=False,
        ),
    ),
    (
        "json_mode",
        bool,
        typer.Option(False, "--json", help="Print one JSON envelope for scripts and agents."),
    ),
    (
        "timeout",
        float | None,
        typer.Option(
            None,
            "--timeout",
            min=0.1,
            metavar="SECONDS",
            help="Bound the whole command; running work goes on when it expires.",
        ),
    ),
]
_COMMIT = [
    ("yes", bool, typer.Option(False, "--yes", help="Confirm without a prompt.")),
    ("dry_run", bool, typer.Option(False, "--dry-run", help="Preview only; change nothing.")),
    (
        "preview_hash",
        str | None,
        typer.Option(
            None,
            "--preview-hash",
            metavar="HASH",
            help="Commit only if a fresh preview still has this hash.",
        ),
    ),
    (
        "operation_id",
        str | None,
        typer.Option(None, "--operation-id", metavar="ID", help="Reuse an operation ID."),
    ),
]
_WAIT = [("wait", bool, typer.Option(False, "--wait", help="Wait until the work settles."))]
_LIST = [
    (
        "limit",
        int,
        typer.Option(DEFAULT_LIMIT, "--limit", min=1, max=MAX_LIMIT, help="Items to return."),
    ),
    ("offset", int, typer.Option(0, "--offset", min=0, help="Items to skip.")),
]


def command(
    group: typer.Typer,
    name: str,
    *,
    commit: bool = False,
    waits: bool = False,
    lists: bool = False,
    **settings,
):
    """Register ``function(ctx, **arguments)`` with the shared options appended."""

    def decorate(function):
        shared = _COMMON + (_COMMIT if commit else []) + (_WAIT if waits else [])
        shared += _LIST if lists else []
        signature = inspect.signature(function)
        own = [value for key, value in signature.parameters.items() if key != "ctx"]
        extra = [
            inspect.Parameter(key, inspect.Parameter.KEYWORD_ONLY, default=default, annotation=hint)
            for key, hint, default in shared
        ]

        @functools.wraps(function)
        def wrapper(**arguments):
            ctx = Context(**{key: arguments.pop(key) for key, _, _ in shared})
            run(ctx, lambda: function(ctx, **arguments))

        wrapper.__signature__ = signature.replace(parameters=own + extra)
        wrapper.__annotations__ = {
            **{key: value for key, value in function.__annotations__.items() if key != "ctx"},
            **{key: hint for key, hint, _ in shared},
        }
        group.command(name, **settings)(wrapper)
        return function

    return decorate


# A command that ends this way keeps its answered operations journaled, so that running it
# again replays them: the work it started, or the draft it saved, is not made twice.
REPLAYED_KINDS = frozenset({"timeout", "interrupted", "unconfirmed"})


def run(ctx: Context, body: Callable[[], Result]) -> None:
    replayed = False
    try:
        try:
            result = body()
        except _DryRun as stop:
            replayed = True
            data = {"preview": stop.preview, "result": None}
            result = Result(data, warnings=stop.warnings, text=_dry_run_text(stop))
        # Printing is part of the command: a fault there exits like any other.
        _emit(ctx, result)
    except ClientError as error:
        replayed = error.kind in REPLAYED_KINDS
        raise typer.Exit(_fail(ctx, error)) from None
    except KeyboardInterrupt:
        replayed = True
        interrupted = ClientError("Interrupted.", code="INTERRUPTED", kind="interrupted")
        raise typer.Exit(_fail(ctx, interrupted)) from None
    except typer.Exit:
        raise
    except Exception as error:  # a CLI fault; the service's faults arrive as ClientError
        if os.environ.get("HISTOPILOT_DEBUG"):
            traceback.print_exc()
        fault = ClientError(
            f"Unexpected CLI failure: {error!r}", code="INTERNAL_ERROR", kind="internal"
        )
        raise typer.Exit(_fail(ctx, fault)) from None
    finally:
        if ctx._client is not None and not replayed:
            ctx._client.settle_answered()


def _emit(ctx: Context, result: Result) -> None:
    if ctx.json_mode:
        output.emit(
            output.dump(output.envelope_ok(result.data, warnings=result.warnings, page=result.page))
        )
        return
    for warning in result.warnings:
        output.note(finding_line(warning))
    text = result.text() if callable(result.text) else result.text
    output.emit(output.pretty(result.data) if text is None else text)
    if result.page and result.page.get("hasMore"):
        following = result.page["offset"] + result.page["limit"]
        output.note(f"More items follow; add --offset {following} to see them.")


def _fail(ctx: Context, error: ClientError) -> int:
    if ctx.json_mode:
        output.emit(output.dump(output.envelope_error(error, error.data)))
        return error.exit_code
    output.note(f"Error: {error.message} [{error.code}]")
    for finding in error.findings:
        where = f" ({finding['field']})" if finding.get("field") else ""
        output.note(f"  - {finding.get('message')}{where} [{finding.get('code')}]")
    if isinstance(error.data, dict) and error.data.get("preview") is not None:
        output.note(output.pretty(error.data["preview"]))
    return error.exit_code


class HistoPilotGroup(TyperGroup):
    """With --json, even a usage error prints one JSON envelope on standard output."""

    def main(self, args=None, prog_name=None, complete_var=None, standalone_mode=True, **extra):
        argv = list(sys.argv[1:] if args is None else args)
        if not standalone_mode or "--json" not in argv:
            return super().main(args, prog_name, complete_var, standalone_mode, **extra)
        try:
            result = super().main(argv, prog_name, complete_var, False, **extra)
        except typer.Abort:
            error = ClientError("Interrupted.", code="INTERRUPTED", kind="interrupted")
            output.emit(output.dump(output.envelope_error(error)))
            sys.exit(error.exit_code)
        except typer.TyperException as problem:
            text = problem.format_message() if hasattr(problem, "format_message") else str(problem)
            error = usage_error(text)
            output.emit(output.dump(output.envelope_error(error)))
            sys.exit(error.exit_code)
        sys.exit(result if isinstance(result, int) else 0)
