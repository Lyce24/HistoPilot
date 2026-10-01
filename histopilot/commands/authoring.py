"""Records authored from spec files: templates, then create with preview and confirmation.

Targets & splits, cohorts and dataset imports are saved as drafts, previewed, and frozen
under a version tag. Feature sources and bundles are previewed and frozen under a tag;
reference standards, clinical analyses and seed ensembles are previewed and saved; Apply
models batches, packing, extraction and attention studies are previewed and queued. Each
commit shows its preview first; see docs/cli-contract.md.
"""

from collections.abc import Callable
from pathlib import Path

import typer

from histopilot import templates
from histopilot.client import authoring, records, resolve, specs
from histopilot.client.api import segment
from histopilot.client.errors import usage_error
from histopilot.client.resources import project_path
from histopilot.client.wait import wait

from . import output
from .common import FORCE, OUTPUT, Result, command, spec_result

TAG = typer.Option(..., "--tag", help="The version's tag, unique for its kind in the project.")
NOTE = typer.Option("", "--note", help="A note kept with the version.")
SOURCE = typer.Option(..., "--from", help="Spec file (YAML or JSON).")
PLACEHOLDERS = "Fill in the @tag and … placeholders; `histopilot schema {kind}` lists every field."
COHORT_VARIANTS = {"labeled": "cohort", "unlabeled": "unlabeled-cohort"}


def _read(ctx, path: Path, kind: str) -> tuple[dict, list[dict]]:
    """A checked spec, and one note per `@tag` it names (see specs.prepare)."""
    return specs.prepare(kind, specs.read(path, kind), records.resolver(ctx.client, ctx.project()))


def _template_command(group: typer.Typer, kind: str) -> None:
    @command(group, "template")
    def template(ctx, destination: Path | None = OUTPUT, force: bool = FORCE):
        """Write a starting spec file with every science field written out."""
        return spec_result(
            kind,
            templates.spec_starter(kind),
            note=PLACEHOLDERS.format(kind=kind),
            destination=destination,
            force=force,
        )


def _draft_command(group: typer.Typer, noun: str, label: str, verb: str = "create") -> None:
    @command(
        group,
        verb,
        commit=True,
        help=f"Save a {label} draft from a spec file, preview it, and freeze it under a tag.",
    )
    def create(
        ctx,
        source: Path = SOURCE,
        tag: str = TAG,
        note: str = NOTE,
        name: str | None = typer.Option(None, "--name", help="Draft name (default: the tag)."),
    ):
        spec, notes = _read(ctx, source, noun)
        project = ctx.project()
        draft = authoring.save_draft(ctx.client, project, noun, name or tag, spec)
        preview = authoring.preview_draft(ctx.client, project, noun, draft)
        return ctx.commit(
            {"draftId": draft["id"], **preview},
            lambda: authoring.freeze_draft(
                ctx.client,
                project,
                noun,
                draft,
                preview,
                tag=tag,
                note=note,
                operation_id=ctx.operation_id,
            ),
            findings=[*notes, *authoring.review(preview, authoring.DRAFTS[noun].gate)],
            summary=f"Freeze {label} {tag!r} from draft {draft['id']}.",
        )


def _commit_selection(
    ctx,
    noun: str,
    selection: dict,
    summary: str,
    *,
    notes=(),
    label: dict | None = None,
    after: Callable[[str, dict], object] | None = None,
    describe: Callable[[dict], str] | None = None,
) -> Result:
    """Preview a selection, then commit it: `after(project, saved)` shapes the result, and
    `describe(preview)` shows people what they confirm."""
    project = ctx.project()
    preview = authoring.preview_selection(ctx.client, project, noun, selection)

    def send():
        saved = authoring.commit_selection(
            ctx.client,
            project,
            noun,
            selection,
            preview,
            operation_id=ctx.operation_id,
            label=label,
        )
        return after(project, saved) if after else saved

    return ctx.commit(
        preview,
        send,
        findings=[*notes, *authoring.review(preview, authoring.SELECTIONS[noun].gate)],
        summary=f"{summary}\n{describe(preview)}" if describe else summary,
    )


def _commit_spec(ctx, noun: str, source: Path, summary: str, **options) -> Result:
    selection, notes = _read(ctx, source, noun)
    return _commit_selection(ctx, noun, selection, summary, notes=notes, **options)


def _settled(ctx, noun: str) -> Callable[[str, dict], object]:
    """With --wait, the queued record read until its work settles; otherwise as saved."""

    def after(project: str, saved: dict):
        if not ctx.wait:
            return saved
        return wait(ctx.client, lambda: records.record(ctx.client, project, noun, saved["id"]))

    return after


def _versioned_command(group: typer.Typer, noun: str, label: str, verb: str = "create") -> None:
    @command(group, verb, commit=True, help=f"Preview a {label} spec, then freeze it under a tag.")
    def create(ctx, source: Path = SOURCE, tag: str = TAG, note: str = NOTE):
        return _commit_spec(
            ctx, noun, source, f"Freeze {label} {tag!r}.", label={"tag": tag, "note": note}
        )


def _job_command(group: typer.Typer, noun: str, record: str, label: str, verb: str) -> None:
    @command(group, verb, commit=True, waits=True, help=f"Preview a {label} spec, then queue it.")
    def create(ctx, source: Path = SOURCE):
        return _commit_spec(ctx, noun, source, f"Queue this {label}.", after=_settled(ctx, record))


APPLY_COLUMNS = [
    ("PREDICTOR", "predictorId"),
    ("NAME", "predictorName"),
    ("METHOD", "method"),
    ("QUEUED", "eligible"),
    (
        "WHY NOT",
        lambda row: "; ".join(
            item.get("message") or ""
            for item in row.get("findings") or []
            if item.get("severity") == "error"
        ),
    ),
]


def _apply_text(preview: dict) -> str:
    """Which runs an Apply models spec queues, one row per predictor."""
    head = output.fields(
        [
            ("Cohort", preview.get("cohortId")),
            (
                "Runs",
                f"{preview.get('eligibleCount')} to queue, {preview.get('blockedCount')} blocked",
            ),
        ]
    )
    return f"{head}\n\n{output.table(preview.get('items') or [], APPLY_COLUMNS)}"


def _target_note(target: dict, field: str) -> dict:
    """What `targets infer` found: a binary task still needs its positive class."""
    if target["task"] == "binary_classification":
        return {
            "code": "POSITIVE_CLASS_REQUIRED",
            "message": "Set target.positiveClass to one of: " + ", ".join(target["classes"]) + ".",
            "severity": "warning",
        }
    if not target["task"]:
        return {
            "code": "TARGET_NOT_INFERRED",
            "message": f"No task was inferred: {field!r} has fewer than two values in training, "
            "or too many to list. Set the task, classes and labels yourself.",
            "severity": "warning",
        }
    return {
        "code": "TARGET_INFERRED",
        "message": f"{len(target['classes'])} classes from {field!r}.",
        "severity": "info",
    }


def register(groups: dict[str, typer.Typer]) -> None:
    # Targets & splits and cohorts: drafts frozen under a version tag.
    _template_command(groups["targets"], "targets")
    _draft_command(groups["targets"], "targets", "Targets & splits")

    @command(groups["targets"], "infer")
    def targets_infer(
        ctx,
        source: Path = typer.Argument(..., help="Targets & splits spec file."),
        field: str = typer.Option(..., "--field", help="The dataset attribute to predict."),
        destination: Path | None = OUTPUT,
        force: bool = FORCE,
    ):
        """Set the training target from a field, as the Targets & splits page does.

        The field's distinct training values become the classes: two make a binary task,
        more a multiclass one. Value order never names the positive class: choose it."""
        document = specs.read(source, "targets")
        project = ctx.project()
        resolved, notes = specs.resolve_tags(document, records.resolver(ctx.client, project))
        body = {**document, **resolve.target(ctx.client, project, resolved, field)}
        finding = _target_note(body["target"], field)
        return spec_result(
            "targets",
            body,
            note=finding["message"],
            destination=destination,
            force=force,
            warnings=[*notes, finding],
        )

    @command(groups["cohort"], "template")
    def cohort_template(
        ctx,
        variant: str = typer.Option("labeled", "--variant", help="labeled or unlabeled."),
        destination: Path | None = OUTPUT,
        force: bool = FORCE,
    ):
        """Write a starting cohort spec, labeled or unlabeled, with every science field."""
        if variant not in COHORT_VARIANTS:
            raise usage_error(f"Choose --variant from: {', '.join(COHORT_VARIANTS)}.")
        return spec_result(
            "cohort",
            templates.spec_starter(COHORT_VARIANTS[variant]),
            note=PLACEHOLDERS.format(kind="cohort"),
            destination=destination,
            force=force,
        )

    _draft_command(groups["cohort"], "cohort", "cohort")

    # Reference standards and clinical analyses: previewed, then saved.
    _template_command(groups["reference"], "reference")

    @command(groups["reference"], "create", commit=True)
    def reference_create(ctx, source: Path = SOURCE):
        """Map a dataset column to a cohort's classes as a reference standard."""
        return _commit_spec(ctx, "reference", source, "Save this reference standard.")

    _template_command(groups["analysis"], "clinical-analysis")

    @command(groups["analysis"], "create", commit=True)
    def analysis_create(ctx, source: Path = SOURCE):
        """Save a clinical utility analysis of a scored run."""
        return _commit_spec(
            ctx, "clinical-analysis", source, "Save this clinical utility analysis."
        )

    # Apply models: which predictors run on which cohort, previewed, then queued.
    apply_group = groups["apply"]

    @command(apply_group, "template")
    def apply_template(
        ctx,
        experiments: list[str] = typer.Option(
            [], "--experiment", help="Apply this experiment's ready predictors; repeatable."
        ),
        method: str | None = typer.Option(
            None,
            "--method",
            help="seed_ensemble, both, ensemble or refit (default: seed ensembles when any is "
            "ready, else both).",
        ),
        cohort: str | None = typer.Option(
            None,
            "--cohort",
            help="Cohort ID or @tag (default: the testing set the predictors' development "
            "reserved).",
        ),
        predictor: str | None = typer.Option(
            None, "--predictor", help="Apply this one predictor, with its own method."
        ),
        name: str = typer.Option(
            "", "--name", help="Batch name (default: Evaluation, or Inference when unlabeled)."
        ),
        destination: Path | None = OUTPUT,
        force: bool = FORCE,
    ):
        """Write an Apply models spec: cohort, predictors, scope and inference, all explicit.

        With --experiment or --predictor it is filled in as Apply models proposes: the method,
        every ready predictor, the reserved testing cohort and the cohort's inference settings,
        with notes on each choice."""
        if not (experiments or predictor):
            if method or cohort or name:
                raise usage_error("--method, --cohort and --name need --experiment or --predictor.")
            return spec_result(
                "apply",
                templates.spec_starter("apply"),
                note="Name the cohort and the predictors to apply.",
                destination=destination,
                force=force,
            )
        resolution = resolve.apply(
            ctx.client,
            ctx.project(),
            experiments,
            method=method,
            cohort=cohort,
            predictor=predictor,
            name=name,
        )
        findings = resolution["notes"] + resolution["findings"]
        return spec_result(
            "apply",
            resolution["selection"],
            note="\n".join(item["message"] for item in findings),
            destination=destination,
            force=force,
            warnings=findings,
        )

    @command(apply_group, "preview")
    def apply_preview(ctx, source: Path = SOURCE):
        """Review which runs an Apply models spec would queue."""
        selection, notes = _read(ctx, source, "apply")
        preview = authoring.preview_selection(ctx.client, ctx.project(), "apply", selection)
        found = [*notes, *authoring.review(preview, authoring.SELECTIONS["apply"].gate)]
        return Result(preview, warnings=found, text=lambda: _apply_text(preview))

    @command(apply_group, "run", commit=True, waits=True)
    def apply_run(ctx, source: Path = SOURCE):
        """Queue the runs of an Apply models spec after reviewing them."""
        return _commit_spec(
            ctx,
            "apply",
            source,
            "Queue these runs.",
            after=_settled(ctx, "apply"),
            describe=_apply_text,
        )

    # Seed ensembles: one configuration's folds from every seed, as one predictor.
    @command(groups["predictor"], "seed-ensemble", commit=True)
    def seed_ensemble(
        ctx,
        experiment: str = typer.Option(..., "--experiment", help="Experiment ID or name."),
        batch: str = typer.Option(..., "--batch", help="Development batch ID."),
        candidate: str = typer.Option(..., "--candidate", help="Configuration (candidate) ID."),
        name: str = typer.Option(..., "--name", help="Predictor name."),
    ):
        """Publish a seed-ensemble predictor from a completed configuration; nothing trains."""
        selection = specs.check(
            "seed-ensemble",
            {
                "experimentId": ctx.experiment(experiment),
                "batchId": batch,
                "candidateId": candidate,
                "name": name,
            },
        )
        return _commit_selection(
            ctx, "seed-ensemble", selection, f"Publish the seed ensemble {name!r}."
        )


def register_preparation(groups: dict[str, typer.Typer]) -> None:
    """Dataset imports, features, bundles, packs, extraction, interpretation and labels."""
    dataset = groups["dataset"]

    @command(dataset, "inspect")
    def dataset_inspect(
        ctx,
        table: Path = typer.Argument(..., help="CSV or XLSX table on the service's machine."),
        sheet: str | None = typer.Option(None, "--sheet", help="Worksheet of an XLSX file."),
    ):
        """Read a table's sheets, columns and example values before importing it."""
        source = {"path": str(table), **({"sheet": sheet} if sheet else {})}
        data = ctx.client.request(
            "POST", f"{project_path(ctx.project())}/imports/inspect", body={"source": source}
        )

        def text():
            summaries = data.get("columnSummaries") or {}
            rows = [{"column": header, **summaries.get(header, {})} for header in data["headers"]]
            head = output.fields(
                [
                    ("Sheet", data.get("sheet")),
                    ("Sheets", ", ".join(data.get("sheets") or []) or None),
                    ("Rows", data.get("rowCount")),
                ]
            )
            columns = [
                ("COLUMN", "column"),
                ("DISTINCT", "distinctCount"),
                ("MISSING", "missingCount"),
                (
                    "EXAMPLES",
                    lambda row: ", ".join(str(item) for item in row.get("examples") or []),
                ),
            ]
            return f"{head}\n\n{output.table(rows, columns)}"

        warnings = [item for item in data.get("findings") or [] if item.get("severity") != "error"]
        return Result(data, warnings=warnings, text=text)

    _template_command(dataset, "import")
    _draft_command(dataset, "import", "dataset", verb="import")

    @command(groups["features"], "template")
    def features_template(
        ctx,
        extraction: str | None = typer.Option(
            None,
            "--from-extraction",
            help="A succeeded extraction job: attach its outputs exactly as written.",
        ),
        destination: Path | None = OUTPUT,
        force: bool = FORCE,
    ):
        """Write a features spec with every field written out, or one for an extraction's
        outputs: that folder only, its encoder and its job, with no slide list."""
        if extraction:
            body = resolve.features_from_extraction(ctx.client, ctx.project(), extraction)
            note = (
                f"The outputs of extraction {extraction}, read as written. Attach them with "
                "`histopilot features attach --from FILE --tag TAG`."
            )
        else:
            body, note = templates.spec_starter("features"), PLACEHOLDERS.format(kind="features")
        return spec_result("features", body, note=note, destination=destination, force=force)

    _versioned_command(groups["features"], "features", "feature source", verb="attach")
    _template_command(groups["bundle"], "feature-bundle")
    _versioned_command(groups["bundle"], "feature-bundle", "feature bundle")
    _template_command(groups["pack"], "feature-pack")
    _job_command(groups["pack"], "feature-pack", "pack", "packing job", "create")
    _template_command(groups["extraction"], "extraction")
    _job_command(groups["extraction"], "extraction", "extraction", "feature extraction", "run")
    _template_command(groups["interpretation"], "interpretation")

    @command(groups["interpretation"], "run", commit=True, waits=True)
    def interpretation_run(ctx, source: Path = SOURCE):
        """Preview an attention study, save it, and launch it."""

        def after(project, saved):
            path = f"{project_path(project)}/interpretations/{segment(saved['id'])}/launch"
            launched = ctx.client.operation("POST", path, {}, prefix="interpretation-launch")
            if not ctx.wait:
                return {"saved": saved, "launch": launched}
            return _settled(ctx, "interpretation")(project, saved)

        return _commit_spec(
            ctx, "interpretation", source, "Save and launch this attention study.", after=after
        )

    for noun, route in (
        ("dataset", "/datasets/{id}/label"),
        ("targets", "/configurations/{id}/label"),
        ("features", "/configurations/{id}/label"),
        ("cohort", "/configurations/{id}/label"),
        ("configuration", "/configurations/{id}/label"),
    ):
        _label_command(groups[noun], noun, route)


def _label_command(group: typer.Typer, noun: str, route: str) -> None:
    @command(group, "label", commit=True)
    def label(
        ctx,
        name: str = typer.Argument(..., metavar="RECORD", help="ID or @tag."),
        tag: str = typer.Option(..., "--tag", help="New tag, 1 to 80 characters."),
        note: str = typer.Option(..., "--note", help="New note; pass '' to clear it."),
    ):
        """Rename a version: its tag and note. Scientific contents and hashes never change.

        Both --tag and --note are required, because the service clears whichever is left out."""
        project = ctx.project()
        current = records.record(ctx.client, project, noun, name)
        label_now = current.get("versionLabel") or {}
        path = project_path(project) + route.replace("{id}", segment(current["id"]))
        preview = {
            "record": current["id"],
            "from": {"tag": label_now.get("tag"), "note": label_now.get("note")},
            "to": {"tag": tag, "note": note},
        }
        body = {"tag": tag, "note": note, "expectedRevision": label_now.get("revision", 0)}
        return ctx.commit(
            preview,
            lambda: ctx.client.request("PUT", path, body=body),
            summary=f"Relabel {current['id']}: {label_now.get('tag')!r} → {tag!r}.",
        )
