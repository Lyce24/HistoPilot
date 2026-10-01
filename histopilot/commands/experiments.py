"""`histopilot experiment`: model-development experiments and their results."""

from pathlib import Path

import typer

from histopilot import templates
from histopilot.client import ClientError, authoring, records, resolve, resources, specs
from histopilot.client import experiments as designs
from histopilot.client.errors import usage_error
from histopilot.client.paging import page
from histopilot.client.states import run_state
from histopilot.client.wait import wait

from . import output
from .common import FORCE, OUTPUT, Result, command, spec_result

STATES = ("active", "archived", "trashed", "all")


def _score(summary: dict | None) -> str | None:
    if not summary or summary.get("mean") is None:
        return None
    spread = f" ± {summary['sd']:.3f}" if summary.get("sd") is not None else ""
    return f"{summary['mean']:.3f}{spread} (n={summary.get('n')})"


def _interval(bounds: dict | None, *, signed: bool = False) -> str | None:
    if not bounds or bounds.get("lower") is None or bounds.get("upper") is None:
        return None
    shown = "{:+.3f}" if signed else "{:.3f}"
    lower, upper = shown.format(bounds["lower"]), shown.format(bounds["upper"])
    return f"{lower} to {upper}" if signed else f"{lower}–{upper}"


# Paired comparisons name these metrics, in this order, when the service reports them.
PAIRED_METRICS = (("auroc", "AUROC"), ("balancedAccuracy", "balanced accuracy"))


def _results_text(data: dict) -> str:
    """The Results page in text: one row per configuration, then per-class recall and the
    paired comparisons, each with the interval the service computed."""
    level = (data.get("policy") or {}).get("confidenceLevel") or 0.95
    ci = f"{round(level * 100):g}% CI"
    names = {batch.get("batchId"): batch.get("name") for batch in data.get("batches") or []}
    rows, recalls = [], []
    for batch in data.get("batches") or []:
        for item in batch.get("configurations") or []:
            intervals = item.get("intervals") or {}
            mean = (intervals.get("seedAverage") or {}).get("intervals") or {}
            rows.append(
                {
                    "batch": batch.get("name"),
                    "number": item.get("number"),
                    "model": item.get("model"),
                    "inputMode": item.get("inputMode"),
                    "seeds": f"{item.get('seedCount')}/{item.get('plannedSeedCount')}",
                    "auroc": _score((item.get("seedAverage") or {}).get("auroc")),
                    "interval": _interval(mean.get("auroc")),
                    "ensemble": (item.get("ensemble") or {}).get("auroc"),
                    "reported": item.get("selected"),
                }
            )
            classes = [
                f"{row.get('label')} {row['recall']['mean']:.3f}"
                for row in item.get("perClass") or []
                if (row.get("recall") or {}).get("mean") is not None
            ]
            if classes:
                recalls.append(
                    f"  {batch.get('name')} #{item.get('number')}: {' · '.join(classes)}"
                )
    if not rows:
        return "No results yet."
    columns = [
        ("BATCH", "batch"),
        ("#", "number"),
        ("MODEL", "model"),
        ("INPUT", "inputMode"),
        ("SEEDS", "seeds"),
        ("AUROC MEAN ± SD", "auroc"),
        (ci, "interval"),
        (
            "SEED ENSEMBLE",
            lambda row: None if row["ensemble"] is None else f"{row['ensemble']:.3f}",
        ),
        ("REPORTED", "reported"),
    ]
    lines = [output.table(rows, columns)]
    if recalls:
        lines += ["", "Recall by class, mean over training seeds:", *recalls]
    pairs = []
    for pair in data.get("comparisons") or []:
        if not pair.get("available", True):
            continue
        bounds = (pair.get("oofInterval") or {}).get("intervals") or {}
        parts = []
        for key, label in PAIRED_METRICS:
            difference = ((pair.get("oof") or {}).get(key) or {}).get("difference")
            if difference is None:
                continue
            spread = _interval(bounds.get(key), signed=True)
            parts.append(f"{label} {difference:+.3f}" + (f" ({spread})" if spread else ""))
        if parts:
            left = names.get(pair.get("leftBatchId")) or pair.get("leftBatchId")
            right = names.get(pair.get("rightBatchId")) or pair.get("rightBatchId")
            pairs.append(f"  {left} − {right}: {' · '.join(parts)}")
    if pairs:
        unit = ((data.get("design") or {}).get("resamplingUnit") or "unit") + "s"
        lines += ["", f"Paired differences, out of fold ({ci}, resampling {unit}):", *pairs]
    return "\n".join(lines)


def _saved(ctx, project: str, record: dict, verb: str, notes: list[dict]) -> Result:
    """A saved design's preview, as `experiment create` and `update` report it."""
    review = designs.preview(ctx.client, project, record["id"])
    data = {"experimentId": record["id"], "revision": review["revision"], "preview": review}
    warnings = notes + [item for item in review["findings"] if item.get("severity") == "warning"]

    def text():
        return f"{verb} {record['name']} ({record['id']}).\n\n{_preview_text(review)}"

    return Result(data, warnings=warnings, text=text)


def register(app: typer.Typer) -> typer.Typer:
    group = typer.Typer(no_args_is_help=True, help="Model-development experiments.")
    app.add_typer(group, name="experiment", rich_help_panel="Records")
    _register_authoring(group)

    @command(group, "list", lists=True)
    def list_(
        ctx,
        state: str = typer.Option("active", "--state", help="active, archived, trashed or all."),
    ):
        """List experiments, newest first."""
        if state not in STATES:
            raise usage_error(f"Choose --state from: {', '.join(STATES)}.")
        items = resources.experiments(ctx.client, ctx.project(), state=state)
        rows, bounds = page(items, offset=ctx.offset, limit=ctx.limit)
        columns = [
            ("ID", "id"),
            ("NAME", "name"),
            ("RUN STATE", "runState"),
            ("STAGE", "stage"),
            ("BATCHES", lambda row: len(row.get("batches") or [])),
            ("UPDATED", "updatedAt"),
        ]
        return Result(rows, page=bounds, text=lambda: output.table(rows, columns))

    @command(group, "show")
    def show(ctx, experiment: str = typer.Argument(..., help="Experiment ID or name.")):
        """Show an experiment: its stage, run state, batches and predictors."""
        data = resources.experiment(ctx.client, ctx.project(), ctx.experiment(experiment))

        def text():
            lines = output.fields(
                [
                    ("Experiment", f"{data.get('name')} ({data.get('id')})"),
                    ("Stage", data.get("stage")),
                    ("Run state", data.get("runState") or "not started"),
                    ("Status", data.get("status")),
                    ("Why", data.get("statusReason")),
                    ("Setup", data.get("setupStatus")),
                    ("Updated", data.get("updatedAt")),
                ]
            )
            batches = [
                {**batch, "runState": run_state(batch.get("status"))}
                for batch in data.get("batches") or []
            ]
            table = output.table(
                batches,
                [
                    ("BATCH", "id"),
                    ("NAME", "name"),
                    ("RUN STATE", "runState"),
                    ("STATUS", "status"),
                ],
                empty="No batches yet.",
            )
            return f"{lines}\n\n{table}"

        return Result(data, text=text)

    @command(group, "results")
    def results(ctx, experiment: str = typer.Argument(..., help="Experiment ID or name.")):
        """Show cross-validated results: each configuration's seed-averaged AUROC and interval.

        Also the seed ensemble, recall by class and the paired comparisons between batches."""
        data = resources.experiment_results(ctx.client, ctx.project(), ctx.experiment(experiment))
        findings = data.get("findings") or []
        return Result(
            data,
            warnings=[item for item in findings if item.get("severity") == "warning"],
            text=lambda: _results_text(data),
        )

    return group


def _read_design(ctx, path: Path) -> tuple[dict, list[dict]]:
    """A checked design, and one note per `@tag` it names (see experiments.prepare)."""
    design = specs.read(path, designs.KIND)
    return designs.prepare(design, records.resolver(ctx.client, ctx.project()))


def _preview_text(preview: dict) -> str:
    rows = [
        {
            "plan": plan["planId"],
            "ready": plan["canFreeze"],
            "runs": plan.get("runCount"),
            "errors": sum(item.get("severity") == "error" for item in plan["findings"]),
            "warnings": sum(item.get("severity") == "warning" for item in plan["findings"]),
        }
        for plan in preview["plans"]
    ]
    table = output.table(
        rows,
        [
            ("PLAN", "plan"),
            ("READY", "ready"),
            ("RUNS", "runs"),
            ("ERRORS", "errors"),
            ("WARNINGS", "warnings"),
        ],
        empty="No batch plans yet.",
    )
    blocking = [item for item in preview["findings"] if item.get("severity") == "error"]
    lines = [f"  - {item.get('planId')}: {item.get('message')}" for item in blocking]
    return "\n".join([table, *lines])


def _register_authoring(group: typer.Typer) -> None:
    @command(group, "template")
    def template(
        ctx,
        preset: str = typer.Option(
            templates.DEFAULT_PRESET, "--preset", help=", ".join(templates.PRESETS) + "."
        ),
        name: str = typer.Option("New experiment", "--name", help="Experiment name."),
        compare_models: list[str] = typer.Option(
            [],
            "--compare-model",
            help="Compare this model against the preset's recipe, the reference arm; repeatable.",
        ),
        compare_inputs: list[str] = typer.Option(
            [],
            "--compare-input",
            help="image, multimodal or clinical; repeatable (default: image).",
        ),
        clinical_fields: list[str] = typer.Option(
            [],
            "--clinical-field",
            help="FIELD:numeric or FIELD:categorical; the reference arm becomes clinical + "
            "image. Repeatable.",
        ),
        destination: Path | None = OUTPUT,
        force: bool = FORCE,
    ):
        """Write a complete experiment design file to edit, with every science field explicit.

        With --compare-model, --compare-input or --clinical-field, its batch is a controlled
        comparison built as the batch editor builds one: the preset's recipe first, as the
        reference, then each model and input, changing only what that model owns."""
        design = designs.template(
            preset,
            name=name,
            compare_models=compare_models,
            compare_inputs=compare_inputs,
            clinical_fields=clinical_fields,
        )
        return spec_result(
            designs.KIND,
            design,
            note="Replace the @dataset-tag, @targets-tag and featureBundleId placeholders, "
            "then run `histopilot experiment create --from FILE`.",
            destination=destination,
            force=force,
        )

    @command(group, "create")
    def create(
        ctx,
        source: Path = typer.Option(..., "--from", help="Experiment design file (YAML or JSON)."),
    ):
        """Create an experiment from a design file and save its inputs and batch plans.

        Nothing is frozen: preview it, then `histopilot experiment freeze`."""
        design, notes = _read_design(ctx, source)
        project = ctx.project()
        record = designs.create(ctx.client, project, design)
        return _saved(ctx, project, record, "Created", notes)

    @command(group, "update")
    def update(
        ctx,
        experiment: str = typer.Argument(
            ..., help="Experiment ID or name; its setup must not be frozen."
        ),
        source: Path = typer.Option(..., "--from", help="Experiment design file (YAML or JSON)."),
    ):
        """Save a design file over an experiment draft instead of creating another.

        It replaces the name, notes, tags, predictor policy, inputs and batch plans; a frozen
        setup is refused."""
        design, notes = _read_design(ctx, source)
        project = ctx.project()
        record = designs.revise(ctx.client, project, ctx.experiment(experiment), design)
        return _saved(ctx, project, record, "Updated", notes)

    @command(group, "preview")
    def preview(ctx, experiment: str = typer.Argument(..., help="Experiment ID or name.")):
        """Review every batch plan: findings, run counts, and whether the setup can freeze."""
        review = designs.preview(ctx.client, ctx.project(), ctx.experiment(experiment))
        return Result(review, text=lambda: _preview_text(review))

    @command(group, "freeze", commit=True)
    def freeze(ctx, experiment: str = typer.Argument(..., help="Experiment ID or name.")):
        """Freeze the experiment's inputs and batch plans after reviewing them."""
        project, experiment = ctx.project(), ctx.experiment(experiment)
        review = designs.preview(ctx.client, project, experiment)
        return ctx.commit(
            review,
            lambda: designs.freeze(
                ctx.client,
                project,
                experiment,
                revision=review["revision"],
                operation_id=ctx.operation_id,
            ),
            findings=review["findings"],
            summary=f"Freeze {len(review['plans'])} batch plan(s) of {experiment}.\n"
            + _preview_text(review),
        )

    @command(group, "start", commit=True, waits=True)
    def start(ctx, experiment: str = typer.Argument(..., help="Experiment ID or name.")):
        """Start a frozen experiment: queue its training, collection and predictor tasks."""
        project, experiment = ctx.project(), ctx.experiment(experiment)
        record = resources.experiment(ctx.client, project, experiment)
        if not record.get("frozenSetupId"):
            raise ClientError(
                "Freeze the experiment first: `histopilot experiment freeze`.",
                code="EXPERIMENT_SETUP_REQUIRED",
                kind="refused",
            )
        preview = {
            "experimentId": experiment,
            "revision": record["revision"],
            "frozenSetupId": record["frozenSetupId"],
            "batchPlans": [plan["id"] for plan in record.get("batchPlans") or []],
        }

        def send():
            started = designs.start(
                ctx.client,
                project,
                experiment,
                revision=record["revision"],
                operation_id=ctx.operation_id,
            )
            if not ctx.wait:
                return started
            return wait(ctx.client, lambda: resources.experiment(ctx.client, project, experiment))

        return ctx.commit(preview, send, summary=f"Start {record.get('name')} ({experiment}).")

    @command(group, "wait")
    def wait_(ctx, experiment: str = typer.Argument(..., help="Experiment ID or name.")):
        """Wait until an experiment settles: exit 0 when it completes, 9 when it does not."""
        project, experiment = ctx.project(), ctx.experiment(experiment)
        final = wait(ctx.client, lambda: resources.experiment(ctx.client, project, experiment))
        return Result(final, text=f"{final.get('name')}: {final.get('status')}")

    @command(group, "export-design")
    def export_design(
        ctx,
        experiment: str = typer.Argument(..., help="Experiment ID or name."),
        destination: Path | None = OUTPUT,
        force: bool = FORCE,
    ):
        """Write an experiment's design file; `experiment create --from` accepts it back."""
        design = designs.export(
            resources.experiment(ctx.client, ctx.project(), ctx.experiment(experiment))
        )
        return spec_result(
            designs.KIND,
            design,
            destination=destination,
            force=force,
            extra={"designHash": designs.design_hash(design)},
        )

    @command(group, "apply-config", commit=True)
    def apply_config(
        ctx,
        experiment: str = typer.Argument(..., help="Experiment ID or name."),
        batch: str = typer.Option(
            ..., "--batch", help="Development batch ID, or its name in the experiment."
        ),
        candidate: str = typer.Option(
            ..., "--candidate", help="Configuration (candidate) ID, or its number in the batch."
        ),
        cohort: str | None = typer.Option(
            None,
            "--cohort",
            help="Cohort ID or @tag (default: the testing set its development reserved).",
        ),
        destination: Path | None = OUTPUT,
        force: bool = FORCE,
    ):
        """Apply one configuration, as Results' "Apply this configuration" does.

        Writes an Apply models spec for the configuration's seed ensemble, or for its single
        fold ensemble when it has one seed group. A seed ensemble not built yet is built first
        from the verified fold checkpoints, after confirmation; nothing trains. Review the
        spec, then `histopilot apply run --from FILE`."""
        project, experiment = ctx.project(), ctx.experiment(experiment)
        plan = resolve.configuration(ctx.client, project, experiment, batch, candidate)
        if plan["action"] in ("choose", "none"):
            raise ClientError(
                plan["detail"]
                + (
                    " Write a spec for one of them with `histopilot apply template --predictor ID`."
                    if plan["action"] == "choose"
                    else ""
                ),
                code="CONFIGURATION_AMBIGUOUS"
                if plan["action"] == "choose"
                else "CONFIGURATION_NOT_READY",
                kind="refused",
                data={"plan": plan},
            )

        def spec_for(predictor: str) -> dict:
            resolution = resolve.apply(
                ctx.client, project, [experiment], cohort=cohort, predictor=predictor
            )
            body = resolution["selection"]
            notes = resolution["notes"] + resolution["findings"]
            text = specs.render("apply", body, note="\n".join(item["message"] for item in notes))
            path = None
            if destination is not None:
                path = str(output.write_file(text.encode("utf-8"), destination, force=force))
            return {
                "predictorId": predictor,
                "spec": body,
                "path": path,
                "notes": notes,
                "text": text,
            }

        def shown(data: dict) -> str:
            if data["path"]:
                return f"{plan['detail']}\nWrote {data['path']}."
            return f"# {plan['detail']}\n{data['text'].rstrip()}"

        if plan["action"] == "apply":
            data = spec_for(plan["predictorId"])
            notes, rendered = data.pop("notes"), dict(data)
            data.pop("text")
            return Result({"plan": plan, **data}, warnings=notes, text=lambda: shown(rendered))

        selection = plan["selection"]
        preview = authoring.preview_selection(ctx.client, project, "seed-ensemble", selection)

        def build():
            built = authoring.commit_selection(
                ctx.client,
                project,
                "seed-ensemble",
                selection,
                preview,
                operation_id=ctx.operation_id,
            )
            return {"builtPredictor": built, **spec_for(built["id"])}

        committed = ctx.commit(
            preview,
            build,
            findings=authoring.review(preview, authoring.SELECTIONS["seed-ensemble"].gate),
            summary=f"Build the seed ensemble {selection['name']!r}. {plan['detail']}",
        )
        result = dict(committed.data["result"])
        notes, rendered = result.pop("notes"), dict(result)
        result.pop("text")
        return Result(
            {"plan": plan, "preview": committed.data["preview"], **result},
            warnings=committed.warnings + notes,
            text=lambda: shown(rendered),
        )

    @command(group, "export-results")
    def export_results(
        ctx,
        experiment: str = typer.Argument(..., help="Experiment ID or name."),
        destination: Path = typer.Option(..., "--output", "-o", help="CSV file to write."),
        force: bool = FORCE,
    ):
        """Save every seed's out-of-fold result and every test fold as CSV, byte for byte the
        Results page's "Folds and seeds" download."""
        from histopilot import exports

        experiment = ctx.experiment(experiment)
        results = resources.experiment_results(ctx.client, ctx.project(), experiment)
        rows = exports.result_rows(results)
        content = exports.csv_text(rows).encode("utf-8")
        saved = output.write_file(content, destination, force=force)
        data = {
            "experiment": experiment,
            "path": str(saved),
            "rows": len(rows) - 1,
            "bytes": len(content),
        }
        return Result(data, text=f"Saved {len(rows) - 1} rows to {saved}.")

    register_batches(group)


def register_batches(group: typer.Typer) -> None:
    batches = typer.Typer(
        no_args_is_help=True, help="An experiment's batch plans, before its setup is frozen."
    )
    group.add_typer(batches, name="batches")

    @command(batches, "list", lists=True)
    def batches_list(ctx, experiment: str = typer.Argument(..., help="Experiment ID or name.")):
        """List an experiment's batch plans: name, model, mode and training seeds."""
        record = resources.experiment(ctx.client, ctx.project(), ctx.experiment(experiment))
        items = [
            {
                "id": plan["id"],
                "batchName": plan["spec"].get("batchName"),
                "model": (plan["spec"].get("recipe") or {}).get("model"),
                "mode": plan["spec"].get("mode"),
                "trainingSeeds": plan["spec"].get("trainingSeeds"),
            }
            for plan in record.get("batchPlans") or []
        ]
        rows, bounds = page(items, offset=ctx.offset, limit=ctx.limit)
        columns = [
            ("ID", "id"),
            ("NAME", "batchName"),
            ("MODEL", "model"),
            ("MODE", "mode"),
            ("SEEDS", lambda row: ", ".join(str(seed) for seed in row["trainingSeeds"] or [])),
        ]
        return Result(
            rows,
            page=bounds,
            text=lambda: output.table(rows, columns, empty="No batch plans yet."),
        )

    @command(batches, "add")
    def batches_add(
        ctx,
        experiment: str = typer.Argument(..., help="Experiment ID or name."),
        source: Path = typer.Option(
            ..., "--from", help="An experiment design file, such as `experiment template` writes."
        ),
        only: list[str] = typer.Option(
            [], "--batch", help="Add only this batch of the file; repeatable."
        ),
    ):
        """Add a design file's batches to an experiment's plans, then preview them.

        The experiment keeps its own inputs; only the file's batches are read. Nothing is
        frozen."""
        design = specs.read(source, designs.KIND)
        offered = design.get("batches") or []
        missing = sorted(set(only) - {batch.get("id") for batch in offered})
        if missing:
            raise usage_error(f"{source} has no batch {', '.join(missing)}.")
        chosen = [batch for batch in offered if not only or batch.get("id") in only]
        designs.check_batches(chosen, name=design.get("name") or "x")
        project, experiment = ctx.project(), ctx.experiment(experiment)
        record = resources.experiment(ctx.client, project, experiment)
        plans = record.get("batchPlans") or []
        clash = sorted({plan["id"] for plan in plans} & {batch["id"] for batch in chosen})
        if clash:
            raise usage_error(
                f"{experiment} already has the batch plan {', '.join(clash)}; give the new "
                "batch another id.",
                code="BATCH_PLAN_EXISTS",
            )
        added = [designs.plan_for(record, batch) for batch in chosen]
        saved = designs.save_plans(ctx.client, project, record, [*plans, *added])
        review = designs.preview(ctx.client, project, experiment)
        data = {
            "experimentId": experiment,
            "revision": saved["revision"],
            "added": [plan["id"] for plan in added],
            "preview": review,
        }
        warnings = [item for item in review["findings"] if item.get("severity") == "warning"]
        return Result(data, warnings=warnings, text=lambda: _preview_text(review))

    @command(batches, "remove")
    def batches_remove(
        ctx,
        experiment: str = typer.Argument(..., help="Experiment ID or name."),
        plan: str = typer.Argument(..., help="Batch plan ID."),
    ):
        """Remove one batch plan from an experiment whose setup is not frozen."""
        project, experiment = ctx.project(), ctx.experiment(experiment)
        record = resources.experiment(ctx.client, project, experiment)
        plans = record.get("batchPlans") or []
        if plan not in {item["id"] for item in plans}:
            raise ClientError(
                f"{experiment} has no batch plan {plan!r}.",
                code="BATCH_PLAN_NOT_FOUND",
                kind="not-found",
            )
        saved = designs.save_plans(
            ctx.client, project, record, [item for item in plans if item["id"] != plan]
        )
        remaining = [item["id"] for item in saved.get("batchPlans") or []]
        data = {
            "experimentId": experiment,
            "revision": saved["revision"],
            "removed": plan,
            "remaining": remaining,
        }
        return Result(data, text=f"Removed {plan}; {len(remaining)} batch plan(s) remain.")
