"""Local application launcher. No compute frameworks are imported by the CLI."""

import json
import os
import threading
import webbrowser
from dataclasses import asdict
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4

import typer

from histopilot.config import Settings, load_settings
from histopilot.contracts.experiment import ExperimentSpec
from histopilot.doctor import system_report
from histopilot.service_lock import service_lock

app = typer.Typer(
    no_args_is_help=True, help="Local-first PFM–MIL workflows for computational pathology."
)


def create_dev_app():
    """Uvicorn reload factory; the launcher retains the workspace lock."""
    from histopilot.api.app import create_app

    values = json.loads(os.environ["HISTOPILOT_SERVICE_SETTINGS"])
    return create_app(Settings(**values))


@app.command()
def serve(
    workspace: Path | None = typer.Option(
        None, help="Directory for local application metadata/artifacts."
    ),
    data_root: list[Path] | None = typer.Option(
        None, help="Read-only source directory visible in the picker; repeatable."
    ),
    config: Path | None = typer.Option(
        None, help="TOML configuration; defaults to ~/.histopilot/config.toml."
    ),
    host: str | None = typer.Option(
        None, help="Loopback host; network hosting requires a future auth layer."
    ),
    port: int | None = typer.Option(None, help="Local HTTP port (default 8787)."),
    dev: bool = typer.Option(
        False, help="Enable API reload and the local Vite development origin."
    ),
    browser: bool = typer.Option(
        True, "--browser/--no-browser", help="Open the local URL in a browser."
    ),
) -> None:
    """Start one local control service and serve the packaged browser application."""
    import uvicorn

    from histopilot.api.app import create_app

    try:
        settings = load_settings(
            config,
            workspace=workspace,
            data_roots=tuple(data_root) if data_root else None,
            host=host,
            port=port,
            dev=dev,
        )
        for root in settings.data_roots:
            if not root.is_dir():
                raise ValueError(f"Data root does not exist or is not a directory: {root}")
        if not dev and not (settings.static_dir / "index.html").is_file():
            raise ValueError(
                "The frontend bundle is missing. Run npm ci and npm run build in web/, then python scripts/bundle_web.py."
            )
        with service_lock(settings.workspace):
            url = f"http://{settings.host}:{settings.port}"
            typer.echo(
                f"HistoPilot · local control service\nWorkspace  {settings.workspace}\nBrowser    {url}\nCompute    TRIDENT extraction; ABMIL k-fold training in Model development"
            )
            if settings.data_roots:
                typer.echo("Sources    Read-only folders available in the data/slide picker:")
                for root in settings.data_roots:
                    typer.echo(f"           {root}")
            else:
                typer.echo(
                    "Sources    No source folders configured; the data/slide picker is empty.\n"
                    "           Restart with --data-root /path/to/data (repeat for more folders),\n"
                    "           or set [storage].data_roots in your service TOML configuration.\n"
                    "           Experiment storage is available in the workspace above."
                )
            if browser:
                timer = threading.Timer(1, webbrowser.open, args=[url], kwargs={"new": 2})
                timer.daemon = True
                timer.start()
            if dev:
                os.environ["HISTOPILOT_SERVICE_SETTINGS"] = json.dumps(
                    asdict(settings), default=str
                )
                uvicorn.run(
                    "histopilot.cli:create_dev_app",
                    factory=True,
                    host=settings.host,
                    port=settings.port,
                    reload=True,
                    reload_dirs=[str(Path(__file__).parent)],
                )
            else:
                uvicorn.run(create_app(settings), host=settings.host, port=settings.port, workers=1)
    except (ValueError, OSError, RuntimeError) as exc:
        typer.echo(f"Cannot start HistoPilot: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command()
def doctor(
    json_output: bool = typer.Option(
        False, "--json", help="Emit a machine-readable environment report."
    ),
) -> None:
    """Inspect control-service dependencies without initializing CUDA or loading weights."""
    report = system_report()
    if json_output:
        typer.echo(json.dumps(report, indent=2))
        return
    typer.echo(f"Python {report['python']} · {report['platform']} · SQLite {report['sqlite']}")
    for name, package in report["packages"].items():
        typer.echo(f"{name:18} {package['version'] or 'not installed'}")
    typer.echo(f"\n{report['note']}")


@app.command("run")
def run_manifest(
    manifest: Path = typer.Argument(..., exists=True, dir_okay=False),
    validate_only: bool = typer.Option(
        False, help="Validate manifest shape without attempting execution."
    ),
) -> None:
    """Read the same v1 experiment specification exported by the service."""
    import yaml
    from pydantic import ValidationError

    try:
        content = manifest.read_text(encoding="utf-8")
        values = (
            yaml.safe_load(content)
            if manifest.suffix.lower() in {".yaml", ".yml"}
            else json.loads(content)
        )
        spec = ExperimentSpec.model_validate(values)
    except (OSError, ValueError, ValidationError, yaml.YAMLError) as exc:
        typer.echo(f"Invalid experiment manifest: {exc}", err=True)
        raise typer.Exit(1) from exc
    if validate_only:
        typer.echo(spec.model_dump_json(indent=2))
        typer.echo(
            "Manifest shape is valid. Dataset/artifact availability and execution readiness are not checked.",
            err=True,
        )
        return
    typer.echo(
        "Execution is not connected in this skeleton. The manifest is valid, but no job was submitted.",
        err=True,
    )
    raise typer.Exit(2)


@app.command()
def jobs(
    url: str = typer.Option("http://127.0.0.1:8787", help="Running local control service URL."),
) -> None:
    """Read job status from the same API as the browser."""
    parsed = urlsplit(url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"localhost", "127.0.0.1"}
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise typer.BadParameter(
            "Use an HTTP loopback service URL without a path, credentials, or query."
        )
    base = url.rstrip("/")
    try:
        with urlopen(f"{base}/api/v1/session", timeout=5) as response:
            token = json.load(response)["token"]
        request = Request(f"{base}/api/v1/jobs", headers={"X-HistoPilot-Token": token})
        with urlopen(request, timeout=5) as response:
            typer.echo(json.dumps(json.load(response), indent=2))
    except (URLError, OSError, KeyError, ValueError) as exc:
        typer.echo(f"Cannot read jobs from the local service: {exc}", err=True)
        raise typer.Exit(1) from exc


def _feature_api(url: str, path: str, payload: dict | None = None) -> dict | None:
    """Use the browser's authenticated local API without opening external URLs."""
    parsed = urlsplit(url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"localhost", "127.0.0.1"}
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise typer.BadParameter("Use an HTTP loopback service URL without credentials or a path.")
    base = url.rstrip("/")
    try:
        with urlopen(f"{base}/api/v1/session", timeout=5) as response:
            token = json.load(response)["token"]
        request = Request(
            f"{base}/api/v1{path}",
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"X-HistoPilot-Token": token, "Content-Type": "application/json"},
            method="POST" if payload is not None else "GET",
        )
        with urlopen(request, timeout=45) as response:
            return json.load(response)
    except HTTPError as exc:
        try:
            message = json.loads(exc.read(65536)).get("detail", str(exc))
        except (ValueError, AttributeError):
            message = str(exc)
        typer.echo(f"Local operation failed: {message}", err=True)
        raise typer.Exit(1) from exc
    except (URLError, OSError, KeyError, ValueError) as exc:
        typer.echo(f"Cannot contact the local service: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command("pack-features")
def pack_features(
    feature_set_id: str = typer.Argument(..., help="Frozen feature configuration ID."),
    project: str = typer.Option(..., help="Saved project ID."),
    output: Path | None = typer.Option(
        None, help="New or empty destination folder on the service."
    ),
    existing_pack: Path | None = typer.Option(
        None, help="Verify and register an existing pack folder against these features."
    ),
    dtype: str = typer.Option(
        "preserve", help="preserve (default) or float16 (reduced precision)."
    ),
    validate_only: bool = typer.Option(
        False, help="Run a full content-validation job without a pack."
    ),
    preview: bool = typer.Option(
        False, help="Inspect inputs and estimated space without starting a job."
    ),
    operation_id: str | None = typer.Option(
        None, help="Reuse an ID to retry the same submission safely."
    ),
    url: str = typer.Option("http://127.0.0.1:8787", help="Running local control service URL."),
) -> None:
    """Validate, create, or verify an existing pack in a durable CPU worker."""
    from histopilot.schemas.feature_packs import FeaturePackSpec

    if existing_pack and (output or validate_only or dtype != "preserve"):
        raise typer.BadParameter(
            "--existing-pack cannot be combined with --output, --validate-only or dtype conversion."
        )
    try:
        spec = FeaturePackSpec(
            featureSetId=feature_set_id,
            action="attach" if existing_pack else "validate" if validate_only else "pack",
            outputPath=str(output) if output else None,
            existingPath=str(existing_pack) if existing_pack else None,
            dtype=dtype,
        ).model_dump(mode="json")
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    route = f"/projects/{quote(project, safe='')}/feature-packs"
    checked = _feature_api(url, route + "/preview", spec)
    if preview or not checked["canRun"]:
        typer.echo(json.dumps(checked, indent=2))
        if not checked["canRun"]:
            raise typer.Exit(1)
        return
    result = _feature_api(
        url,
        route,
        {
            **spec,
            "previewHash": checked["previewHash"],
            "operationId": operation_id or f"packing:{uuid4()}",
        },
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("feature-jobs")
def feature_jobs(
    project: str = typer.Option(..., help="Saved project ID."),
    job: str | None = typer.Option(
        None, help="Read one packing/validation job, including its log."
    ),
    cancel: bool = typer.Option(False, help="Cancel the specified job cooperatively."),
    url: str = typer.Option("http://127.0.0.1:8787", help="Running local control service URL."),
) -> None:
    """List feature jobs or inspect/cancel one through the local API."""
    if cancel and not job:
        raise typer.BadParameter("--cancel requires --job.")
    route = f"/projects/{quote(project, safe='')}/feature-packs"
    if job:
        route += f"/{quote(job, safe='')}"
    if cancel:
        route += "/cancel"
    typer.echo(json.dumps(_feature_api(url, route, {} if cancel else None), indent=2))


@app.command("verify-feature-pack")
def verify_feature_pack(
    path: Path = typer.Argument(..., exists=True, file_okay=False),
) -> None:
    """Verify all packed bytes and metadata locally, including a relocated pack."""
    from histopilot.storage.packed import validate_pack

    try:
        manifest = validate_pack(path, full=True)
    except (OSError, ValueError) as exc:
        typer.echo(f"Pack verification failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(manifest, indent=2))


@app.command("train-batch")
def train_batch(
    batch: str = typer.Argument(..., help="Frozen development batch configuration ID."),
    project: str = typer.Option(..., help="Saved project ID."),
    resume: bool = typer.Option(False, help="Resume unfinished runs, keeping completed folds."),
    operation_id: str | None = typer.Option(None, help="Reuse an ID to retry this action safely."),
    url: str = typer.Option("http://127.0.0.1:8787", help="Running local control service URL."),
) -> None:
    """Launch a frozen ABMIL k-fold batch through the same API as the browser."""
    route = f"/projects/{quote(project, safe='')}/mil-experiments/batches/{quote(batch, safe='')}"
    result = _feature_api(
        url,
        route + ("/resume" if resume else "/launch"),
        {"operationId": operation_id or f"training:{uuid4()}"},
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("training-status")
def training_status(
    batch: str = typer.Argument(..., help="Frozen development batch configuration ID."),
    project: str = typer.Option(..., help="Saved project ID."),
    cancel: bool = typer.Option(False, help="Stop this batch, preserving available checkpoints."),
    results: bool = typer.Option(False, help="Read completed development OOF results."),
    url: str = typer.Option("http://127.0.0.1:8787", help="Running local control service URL."),
) -> None:
    """Inspect a training batch, read OOF results, or request cancellation."""
    if cancel and results:
        raise typer.BadParameter("Choose either --cancel or --results.")
    route = f"/projects/{quote(project, safe='')}/mil-experiments/batches/{quote(batch, safe='')}"
    suffix = "/cancel" if cancel else "/results" if results else "/execution"
    payload = {"operationId": f"cancel:{uuid4()}"} if cancel else None
    typer.echo(json.dumps(_feature_api(url, route + suffix, payload), indent=2))


if __name__ == "__main__":
    app()
