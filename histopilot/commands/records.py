"""List and show for every project record kind, plus each kind's own reads and downloads."""

from pathlib import Path

import typer

from histopilot import resolvers
from histopilot.client import records as store
from histopilot.client import resources
from histopilot.client.api import segment
from histopilot.client.paging import page, served
from histopilot.client.resources import project_path

from . import output
from .common import FORCE, Result, command


def _manifest(row: dict) -> dict:
    return row.get("manifest") if isinstance(row.get("manifest"), dict) else {}


def _name(row: dict):
    manifest = _manifest(row)
    spec = row.get("spec") if isinstance(row.get("spec"), dict) else {}
    # A batch keeps its name in its frozen spec.
    batch = (manifest.get("spec") or {}).get("batchName")
    return output.name(
        row.get("name") or manifest.get("name") or row.get("title") or spec.get("name") or batch
    )


def _model(key: str):
    return lambda row: (row.get("model") or {}).get(key)


def _seeds(row: dict) -> str:
    manifest = _manifest(row)
    if manifest.get("trainingSeeds") is not None:
        splits = manifest.get("splitSeeds")
        return f"{len(manifest['trainingSeeds'])} × {1 if splits is None else len(splits)} seeds"
    return f"{manifest.get('trainingSeed')}/{manifest.get('splitSeed')}"


def _batch_experiment(row: dict):
    manifest = _manifest(row)
    return (manifest.get("experiment") or {}).get("name") or (manifest.get("spec") or {}).get(
        "experimentName"
    )


def _kind(row: dict):
    manifest = row.get("manifest") if isinstance(row.get("manifest"), dict) else {}
    return manifest.get("kind") or row.get("kind")


def _tag(row: dict):
    return (row.get("versionLabel") or {}).get("tag")


def _sentence_case(text: str) -> str:
    """Lower the first letter unless it starts an acronym such as TRIDENT."""
    return text if text[1:2].isupper() else text[:1].lower() + text[1:]


def columns(kind: store.Kind) -> list:
    if kind.noun == "predictor":
        # Predictor names omit their batch: two models' predictors read alike by name.
        return [
            ("ID", "id"),
            ("EXPERIMENT", _model("experiment")),
            ("BATCH", _model("batch")),
            ("#", lambda row: _manifest(row).get("candidateNumber")),
            ("METHOD", lambda row: resolvers.predictor_method_label(_model("method")(row))),
            ("SEEDS", _seeds),
            ("UPDATED", lambda row: row.get("updatedAt") or row.get("createdAt")),
        ]
    shown = [("ID", "id"), ("NAME", _name)]
    if kind.noun == "batch":
        shown.append(("EXPERIMENT", _batch_experiment))
    if kind.noun == "run":
        shown.append(("BATCH", _model("batch")))
    if kind.tagged:
        shown.append(("TAG", _tag))
    if kind.noun in ("configuration", "draft"):
        shown.append(("KIND", _kind))
    if kind.runs:
        shown += [("RUN STATE", "runState"), ("STATUS", store.status_of)]
    return shown + [("UPDATED", lambda row: row.get("updatedAt") or row.get("createdAt"))]


def name_help(noun: str) -> str:
    """How a record of this kind is named on the command line."""
    return "ID, or @tag of a tagged version." if store.KINDS[noun].tagged else "ID."


# One row per experiment configuration and seed group that can produce a predictor.
CHOICE_COLUMNS = [
    ("EXPERIMENT", "experimentName"),
    ("BATCH", "batchName"),
    ("#", "candidateNumber"),
    ("SEEDS", lambda row: f"{row.get('trainingSeed')}/{row.get('splitSeed')}"),
    ("RUNS", lambda row: f"{row.get('completedRuns')}/{row.get('totalRuns')}"),
    ("METHODS", lambda row: ", ".join(row.get("eligibleMethods") or [])),
    ("WHY NOT", "reason"),
]


def _listed(
    ctx,
    kind: store.Kind,
    *,
    inactive: bool = False,
    experiment: str | None = None,
    cohort: str | None = None,
) -> Result:
    project = ctx.project()
    items = store.records(
        ctx.client, project, kind.noun, inactive=inactive, describe=kind.noun in store.DESCRIBED
    )
    if experiment is not None:
        wanted = resources.experiment_id(ctx.client, project, experiment, existing=True)
        items = [item for item in items if store.experiment_of(kind.noun, item) == wanted]
    if cohort is not None:
        wanted = store.existing(ctx.client, project, "cohort", cohort)
        items = [item for item in items if _manifest(item).get("cohortId") == wanted]
    rows, bounds = page(items, offset=ctx.offset, limit=ctx.limit)
    return Result(rows, page=bounds, text=lambda: output.table(rows, columns(kind)))


INACTIVE = typer.Option(False, "--include-inactive", help="Include archived and trashed records.")
EXPERIMENT = typer.Option(None, "--experiment", help="Only this experiment's; its ID or name.")
COHORT = typer.Option(None, "--cohort", help="Only runs on this cohort; its ID or @tag.")


def _register_list(group: typer.Typer, kind: store.Kind) -> None:
    """`list`, with `--include-inactive` where the kind keeps inactive records, and the
    filters a batch, predictor or run list takes."""
    listing = f"List {_sentence_case(kind.help)}"
    if kind.noun == "run":

        @command(group, "list", lists=True, help=listing)
        def list_runs(
            ctx,
            inactive: bool = INACTIVE,
            experiment: str | None = EXPERIMENT,
            cohort: str | None = COHORT,
        ):
            return _listed(ctx, kind, inactive=inactive, experiment=experiment, cohort=cohort)

    elif kind.noun == "predictor":

        @command(group, "list", lists=True, help=listing)
        def list_predictors(ctx, inactive: bool = INACTIVE, experiment: str | None = EXPERIMENT):
            return _listed(ctx, kind, inactive=inactive, experiment=experiment)

    elif kind.noun == "batch":

        @command(group, "list", lists=True, help=listing)
        def list_batches(ctx, experiment: str | None = EXPERIMENT):
            return _listed(ctx, kind, experiment=experiment)

    elif kind.inactive:

        @command(group, "list", lists=True, help=listing)
        def list_all(ctx, inactive: bool = INACTIVE):
            return _listed(ctx, kind, inactive=inactive)

    else:

        @command(group, "list", lists=True, help=listing)
        def list_active(ctx):
            return _listed(ctx, kind)


def _shown(noun: str, data: dict) -> str:
    """A record's text view: what tells a predictor or a run apart first, then every field."""
    model = data.get("model") or {}
    manifest = _manifest(data)
    if noun == "predictor":
        lines = output.fields(
            [
                ("Predictor", manifest.get("name")),
                ("ID", data.get("id")),
                ("Experiment", model.get("experiment")),
                ("Model", model.get("description")),
                ("Architecture", model.get("architecture")),
                ("Members", len(manifest.get("checkpoints") or []) or None),
                ("Classes", ", ".join((manifest.get("target") or {}).get("classes") or []) or None),
                ("State", data.get("lifecycleState")),
                ("Created", data.get("createdAt")),
            ]
        )
        return f"{lines}\n\nAdd --json for every field."
    if noun == "run":
        execution = data.get("execution") or {}
        lines = output.fields(
            [
                ("Run", manifest.get("name")),
                ("ID", data.get("id")),
                ("Purpose", manifest.get("purpose")),
                ("Model", model.get("description")),
                ("Predictor", manifest.get("predictorId")),
                ("Cohort", manifest.get("cohortId")),
                ("Unit", manifest.get("splitUnit")),
                ("Run state", data.get("runState") or "not started"),
                ("Status", execution.get("status")),
                ("Created", data.get("createdAt")),
            ]
        )
        return f"{lines}\n\n`histopilot run summary` reads its predictions; add --json for every field."
    return output.record(data)


def register_kind(app: typer.Typer, kind: store.Kind) -> typer.Typer:
    group = typer.Typer(no_args_is_help=True, help=kind.help)
    app.add_typer(group, name=kind.noun, rich_help_panel="Records")
    _register_list(group, kind)

    @command(group, "show")
    def show(ctx, name: str = typer.Argument(..., metavar="RECORD", help=name_help(kind.noun))):
        """Show one record."""
        data = store.record(
            ctx.client,
            ctx.project(),
            kind.noun,
            name,
            describe=kind.noun in store.DESCRIBED,
        )
        return Result(data, text=lambda: _shown(kind.noun, data))

    return group


def _metric(key: str):
    def read(row: dict):
        value = (row.get("metrics") or {}).get(key)
        return None if value is None else f"{value:.3f}"

    return read


def _batch_results_text(data: dict) -> str:
    """One row per configuration and seed group: its out-of-fold metrics, as `experiment
    results` averages them, and the validation score the selection reads."""
    columns = [
        ("CONFIGURATION", lambda row: resolvers.short_record_id(row.get("candidateId") or "")),
        ("SEEDS", lambda row: f"{row.get('trainingSeed')}/{row.get('splitSeed')}"),
        ("RUNS", lambda row: f"{row.get('completedRuns')}/{row.get('totalRuns')}"),
        ("UNITS", lambda row: (row.get("metrics") or {}).get("count")),
        ("AUROC", _metric("auroc")),
        ("AUPRC", _metric("auprc")),
        ("BAL ACC", _metric("balancedAccuracy")),
        ("ACCURACY", _metric("accuracy")),
        ("LOSS", _metric("loss")),
        (
            "SELECTION",
            lambda row: (
                None if row.get("selectionScore") is None else f"{row['selectionScore']:.3f}"
            ),
        ),
        ("SELECTED", "selected"),
    ]
    table = output.table(data.get("candidates") or [], columns, empty="No results yet.")
    note = data.get("selectionNote")
    return f"Batch {data.get('batchId')} · {data.get('status')}\n\n{table}" + (
        f"\n\n{note}" if note else ""
    )


def _download_command(group: typer.Typer, noun: str, route: str, files: str) -> None:
    @command(group, "download")
    def download(
        ctx,
        name: str = typer.Argument(..., metavar="RECORD", help="ID."),
        filename: str = typer.Argument(..., help=files),
        destination: Path = typer.Option(..., "--output", "-o", help="Where to save the file."),
        force: bool = FORCE,
    ):
        """Save one of the record's files, checked against its recorded SHA-256."""
        project = ctx.project()
        document = store.record(ctx.client, project, noun, name)
        derived = store.derived_download(document, filename)
        expected = None if derived else store.recorded_sha256(document, filename)
        path = project_path(project) + route.format(
            id=segment(document["id"]), file=segment(filename)
        )
        content, check = store.download(ctx.client, path, expected=expected)
        saved = output.write_file(content, destination, force=force)
        data = {"id": document["id"], "file": filename, "path": str(saved), **check}
        verdict = "SHA-256 verified" if check["verified"] else "no recorded SHA-256 to check"
        return Result(data, text=f"Saved {saved} ({check['bytes']} bytes; {verdict}).")


def register(app: typer.Typer) -> dict[str, typer.Typer]:
    groups = {noun: register_kind(app, kind) for noun, kind in store.KINDS.items()}

    @command(groups["dataset"], "records", lists=True)
    def dataset_records(
        ctx, name: str = typer.Argument(..., metavar="DATASET", help="ID or @tag.")
    ):
        """List a dataset's canonical records: one per slide, with its attributes."""
        project = ctx.project()
        identity = store.resolve(ctx.client, project, "dataset", name)
        listed = ctx.client.get(
            f"{project_path(project)}/datasets/{segment(identity)}/records",
            query={"offset": ctx.offset, "limit": min(ctx.limit, 1000)},
        )
        rows = listed.get("records") or []
        if isinstance(rows, dict):
            # A project shared at the metadata level withholds per-case records.
            return Result(
                rows,
                text=f"Withheld: {rows.get('count')} per-case records. This project is shared "
                "with agents at the metadata level.",
            )
        bounds = served(
            rows, offset=ctx.offset, limit=min(ctx.limit, 1000), total=listed.get("total")
        )
        table = [
            ("SLIDE", "slideId"),
            ("PATIENT", "patientId"),
            ("PATIENT SOURCE", "patientIdSource"),
        ]
        return Result(rows, page=bounds, text=lambda: output.table(rows, table))

    @command(groups["features"], "validation")
    def features_validation(
        ctx, name: str = typer.Argument(..., metavar="FEATURES", help="ID or @tag.")
    ):
        """Show the validation evidence of a feature source."""
        project = ctx.project()
        identity = store.resolve(ctx.client, project, "features", name)
        data = ctx.client.get(f"{project_path(project)}/features/{segment(identity)}/validation")
        return Result(data, text=lambda: output.record(data))

    @command(groups["batch"], "results")
    def batch_results(ctx, name: str = typer.Argument(..., metavar="BATCH", help="ID.")):
        """Show a development batch's results: each seed group's out-of-fold metrics.

        Beside them, the validation score that selects the reported configuration."""
        path = f"{project_path(ctx.project())}/mil-experiments/batches/{segment(name)}/results"
        data = ctx.client.get(path)
        return Result(data, text=lambda: _batch_results_text(data))

    @command(groups["extraction"], "catalog")
    def extraction_catalog(ctx):
        """Show the encoders, options and TRIDENT runtime that extraction can use."""
        data = ctx.client.get(f"{project_path(ctx.project())}/extractions/catalog")
        return Result(data, text=lambda: output.record(data))

    @command(groups["predictor"], "choices")
    def predictor_choices(ctx):
        """List the experiment results that can produce a predictor."""
        data = ctx.client.get(f"{project_path(ctx.project())}/predictors/choices")
        rows = data.get("items") or []
        return Result(data, text=lambda: output.table(rows, CHOICE_COLUMNS))

    _download_command(
        groups["run"],
        "run",
        "/evaluation-runs/{id}/artifacts/{file}",
        "predictions.json, metrics.json, summary.json, slide-predictions.csv or "
        "patient-predictions.csv.",
    )
    _download_command(
        groups["analysis"],
        "analysis",
        "/clinical-analyses/{id}/artifacts/{file}",
        "Artifact file name.",
    )
    _download_command(
        groups["interpretation"],
        "interpretation",
        "/interpretations/{id}/artifacts/{file}",
        "Artifact file name.",
    )
    return groups
