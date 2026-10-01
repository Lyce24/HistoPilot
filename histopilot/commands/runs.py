"""`histopilot run …`: scores, agreement, calibration, subgroups, cases and exports."""

from pathlib import Path

import typer

from histopilot import resolvers
from histopilot.client import ClientError, resolve, runs
from histopilot.client.errors import usage_error
from histopilot.client.paging import served

from . import output
from .common import FORCE, Result, command

UNIT = typer.Option("selected", "--unit", help="selected, slide or patient.")
COMPARISON = typer.Option(
    None,
    "--comparison",
    metavar="RUN",
    help="Another run on the same cohort: show its call beside each case; with --outcome "
    "disagreement, only the cases where the two disagree.",
)
REFERENCE = typer.Option(
    None,
    "--reference",
    metavar="REFERENCE",
    help="Score against a reference standard (default: the cohort's labels; for an "
    "unlabeled run, the first fitting reference).",
)


def _unit(value: str) -> str:
    if value not in runs.UNITS:
        raise usage_error(f"Choose --unit from: {', '.join(runs.UNITS)}.")
    return value


def _labels(ctx, run: str, reference: str | None) -> tuple[str | None, list[dict]]:
    """The reference to score against: the one named, else the browser's default."""
    if reference is not None:
        return reference, []
    chosen, note = resolve.default_reference(ctx.client, ctx.project(), run)
    return chosen, [note] if note else []


def _share(count, total) -> str:
    return f"{count} ({count / total:.1%})" if total else str(count)


def _summary_text(data: dict) -> str:
    """What a run predicted, how sure it was, how its members agreed and, with a comparison,
    how often another run's decisions agree with its own."""
    total = data.get("count") or 0
    predicted = " · ".join(
        f"{row.get('label')} {_share(row.get('count') or 0, total)}"
        for row in data.get("predicted") or []
    )
    confidence = data.get("confidence") or {}
    quantiles = confidence.get("quantiles") or {}
    members = data.get("ensemble") or {}
    pairs = [
        ("Run", data.get("name")),
        ("ID", data.get("evaluationId")),
        ("Model", (data.get("model") or {}).get("description")),
        ("Purpose", f"{data.get('purpose')} · {total} {data.get('unit')}s"),
        ("Predicted", predicted or None),
        (
            "Confidence",
            f"mean {confidence['mean']:.3f} · median {quantiles.get('median', 0):.3f} · "
            f"10th percentile {quantiles.get('p10', 0):.3f}"
            if confidence.get("mean") is not None
            else None,
        ),
        (
            "Members",
            f"{members.get('memberCount')} models, unanimous on {members.get('unanimous')} of "
            f"{members.get('records')}"
            if members.get("memberCount")
            else None,
        ),
    ]
    other = data.get("comparison")
    table = ""
    if other:
        weighted = other.get("weightedKappa")
        pairs += [
            ("Compared with", other.get("name")),
            ("Its ID", other.get("evaluationId")),
            ("Its model", (other.get("model") or {}).get("description")),
            (
                "Agreement",
                f"{other['agreement']:.3f} on {other.get('count')} · kappa {other['kappa']:.3f}"
                + (f" · weighted kappa {weighted:.3f}" if weighted is not None else "")
                + f" · {other.get('disagreements')} disagree"
                if other.get("agreement") is not None and other.get("kappa") is not None
                else None,
            ),
        ]
        classes = data.get("classOrder") or []
        matrix = other.get("matrix") or []
        if len(matrix) == len(classes):
            rows = [
                {"class": label, **{name: count for name, count in zip(classes, row, strict=True)}}
                for label, row in zip(classes, matrix, strict=True)
            ]
            shown = [("THIS RUN \\ OTHER", "class"), *[(name, name) for name in classes]]
            table = "\n\n" + output.table(rows, shown)
    return output.fields(pairs) + table + "\n\nAdd --json for every field."


METRIC_COLUMNS = [
    ("UNIT", "unit"),
    ("N", "count"),
    ("AUROC", "auroc"),
    ("AUPRC", "auprc"),
    ("ACCURACY", "accuracy"),
    ("BALANCED", "balancedAccuracy"),
    ("MACRO F1", "macroF1"),
    ("LOSS", "loss"),
]


CASE_COLUMNS = [
    ("CASE", "id"),
    ("SLIDES", lambda row: len(row.get("slideIds") or [])),
    ("LABEL", "label"),
    ("PREDICTED", "predictedLabel"),
    ("CONFIDENCE", "confidence"),
    ("MARGIN", "margin"),
    ("OUTCOME", "outcome"),
    (
        "MEMBERS AGREE",
        lambda row: (
            f"{row['memberAgreement']['agree']}/{row['memberAgreement']['total']}"
            if row.get("memberAgreement")
            else None
        ),
    ),
]


def _metrics_text(data: dict) -> str:
    """The run's metrics per scored unit, with each confusion matrix."""
    classes = ", ".join(data.get("classOrder") or [])
    lines = [
        output.fields(
            [
                ("Scored unit", data.get("unit")),
                ("Positive class", data.get("positiveClass")),
                ("Threshold", data.get("decisionThreshold")),
            ]
        ),
        "",
    ]
    scored = []
    for unit in ("slide", "patient"):
        block = data.get(unit) or {}
        if block.get("available"):
            scored.append({"unit": unit, **block})
            lines.append(
                f"{unit} confusion matrix (rows true {classes}; columns predicted): "
                f"{block.get('confusionMatrix')}"
            )
        elif block.get("reason"):
            lines.append(f"{unit}: {block['reason']}")
    return "\n".join([*lines[:2], output.table(scored, METRIC_COLUMNS), "", *lines[2:]])


def register(group: typer.Typer) -> None:
    @command(group, "metrics")
    def metrics(
        ctx, run: str = typer.Argument(..., help="Run ID."), reference: str | None = REFERENCE
    ):
        """Show a completed run's metrics, against its cohort's labels or a reference."""
        reference, notes = _labels(ctx, run, reference)
        data = runs.metrics(ctx.client, ctx.project(), run, reference=reference)
        return Result(data, warnings=notes, text=lambda: _metrics_text(data))

    @command(group, "label-sources")
    def label_sources(ctx, run: str = typer.Argument(..., help="Run ID.")):
        """The labels a run can be scored against, and the one used without --reference."""
        sources = resolve.label_sources(ctx.client, ctx.project(), run)
        chosen = resolvers.label_source(sources)
        rows = [
            {**item, "default": chosen is not None and item["id"] == chosen["id"]}
            for item in sources
        ]
        return Result(
            rows,
            text=lambda: (
                output.table(
                    [{**row, "id": row["id"] or "(cohort labels)"} for row in rows],
                    [("ID", "id"), ("NAME", "name"), ("DEFAULT", "default")],
                )
                if rows
                else "No labels: the cohort is unlabeled and no reference standard fits this run."
            ),
        )

    @command(group, "agreement")
    def agreement(ctx, run: str = typer.Argument(..., help="Run ID."), unit: str = UNIT):
        """Agreement between the run's decisions and every label source of its cohort."""
        data = runs.agreement(ctx.client, ctx.project(), run, unit=_unit(unit))
        return Result(data, text=lambda: output.record(data))

    @command(group, "recalibration")
    def recalibration(
        ctx,
        run: str = typer.Argument(..., help="Run ID."),
        unit: str = UNIT,
        reference: str | None = REFERENCE,
    ):
        """Calibration as predicted and after recalibration on development predictions."""
        reference, notes = _labels(ctx, run, reference)
        data = runs.recalibration(
            ctx.client, ctx.project(), run, unit=_unit(unit), reference=reference
        )
        return Result(data, warnings=notes, text=lambda: output.record(data))

    @command(group, "subgroups")
    def subgroups(
        ctx,
        run: str = typer.Argument(..., help="Run ID."),
        attribute: str = typer.Option(..., "--attribute", help="A data dictionary key."),
        unit: str = UNIT,
        reference: str | None = REFERENCE,
    ):
        """Performance within each value of an attribute, over the records the run scores."""
        reference, notes = _labels(ctx, run, reference)
        data = runs.subgroups(
            ctx.client,
            ctx.project(),
            run,
            attribute=attribute,
            unit=_unit(unit),
            reference=reference,
        )
        return Result(data, warnings=notes, text=lambda: output.record(data))

    @command(group, "cases", lists=True)
    def cases(
        ctx,
        run: str = typer.Argument(..., help="Run ID."),
        unit: str = UNIT,
        outcome: str = typer.Option("all", "--outcome", help=", ".join(runs.OUTCOMES) + "."),
        sort: str = typer.Option("confidence_desc", "--sort", help=", ".join(runs.SORTS) + "."),
        search: str | None = typer.Option(None, "--search", help="Case identifier text."),
        min_confidence: float | None = typer.Option(None, "--min-confidence", min=0, max=1),
        max_confidence: float | None = typer.Option(None, "--max-confidence", min=0, max=1),
        max_margin: float | None = typer.Option(None, "--max-margin", min=0, max=1),
        disagreement: bool = typer.Option(
            False, "--member-disagreement", help="Only cases whose members disagree."
        ),
        reference: str | None = REFERENCE,
        comparison: str | None = COMPARISON,
    ):
        """Review cases: the most or least confident, errors, disagreements."""
        if outcome not in runs.OUTCOMES:
            raise usage_error(f"Choose --outcome from: {', '.join(runs.OUTCOMES)}.")
        if sort not in runs.SORTS:
            raise usage_error(f"Choose --sort from: {', '.join(runs.SORTS)}.")
        reference, notes = _labels(ctx, run, reference)
        query = {
            "unit": _unit(unit),
            "outcome": outcome,
            "sort": sort,
            "search": search,
            "minConfidence": min_confidence,
            "maxConfidence": max_confidence,
            "maxMargin": max_margin,
            "memberDisagreement": disagreement or None,
            "referenceId": reference,
            "comparisonId": comparison,
            "offset": ctx.offset,
            "limit": min(ctx.limit, 100),
        }
        data = runs.cases(ctx.client, ctx.project(), run, query)
        rows = data.get("items") or []
        columns = CASE_COLUMNS + (
            [("OTHER RUN", lambda row: (row.get("comparison") or {}).get("predictedLabel"))]
            if comparison
            else []
        )
        bounds = served(
            rows, offset=ctx.offset, limit=query["limit"], has_more=bool(data.get("hasMore"))
        )
        return Result(
            rows,
            warnings=notes,
            page=bounds,
            text=lambda: output.table(rows, columns, empty="No cases match."),
        )

    @command(group, "export-cases")
    def export_cases(
        ctx,
        run: str = typer.Argument(..., help="Run ID."),
        destination: Path = typer.Option(..., "--output", "-o", help="CSV file to write."),
        unit: str = UNIT,
        reference: str | None = REFERENCE,
        comparison: str | None = COMPARISON,
        force: bool = FORCE,
    ):
        """Save the case review table as CSV."""
        reference, notes = _labels(ctx, run, reference)
        query = {"unit": _unit(unit), "referenceId": reference, "comparisonId": comparison}
        content = runs.cases_csv(ctx.client, ctx.project(), run, query)
        saved = output.write_file(content, destination, force=force)
        data = {"run": run, "path": str(saved), "bytes": len(content)}
        return Result(data, warnings=notes, text=f"Saved {saved}.")

    @command(group, "summary")
    def summary(
        ctx,
        run: str = typer.Argument(..., help="Run ID."),
        unit: str = UNIT,
        attribute: str | None = typer.Option(None, "--attribute", help="Cross-tabulate by this."),
        comparison: str | None = typer.Option(
            None, "--comparison", metavar="RUN", help="Another run on the same cohort."
        ),
    ):
        """A label-free summary of the run's predictions, for any run, labeled or not.

        With --comparison, how often the two runs' decisions agree on the same cases (kappa,
        and with three or more classes a weighted kappa that reads the class order as a
        scale), as two models applied to one cohort compare without labels."""
        project = ctx.project()
        data = runs.inference_summary(
            ctx.client, project, run, unit=_unit(unit), attribute=attribute, comparison=comparison
        )
        data = runs.with_models(ctx.client, project, data)
        return Result(data, text=lambda: _summary_text(data))

    @command(group, "export-predictions")
    def export_predictions(
        ctx,
        run: str = typer.Argument(..., help="Run ID."),
        destination: Path = typer.Option(..., "--output", "-o", help="CSV file to write."),
        unit: str = UNIT,
        attributes: str | None = typer.Option(
            None, "--attributes", help="Comma-separated attributes; omit for all, '' for none."
        ),
        force: bool = FORCE,
    ):
        """Save one row per slide or patient: predicted class, probabilities, confidence."""
        chosen = None if attributes is None else [item for item in attributes.split(",") if item]
        content = runs.predictions_csv(
            ctx.client, ctx.project(), run, unit=_unit(unit), attributes=chosen
        )
        saved = output.write_file(content, destination, force=force)
        return Result(
            {"run": run, "path": str(saved), "bytes": len(content)}, text=f"Saved {saved}."
        )

    @command(group, "compare")
    def compare(
        ctx,
        left: str = typer.Argument(..., help="Run ID."),
        right: str = typer.Argument(..., help="Run ID on the same cohort."),
    ):
        """A paired comparison of two scored runs, by patient.

        Runs of slide-level experiments and runs on unlabeled cohorts compare by their
        decisions instead: `histopilot run summary LEFT --comparison RIGHT`."""
        try:
            data = runs.compare(ctx.client, ctx.project(), left, right)
        except ClientError as error:
            if error.code in ("COMPARISON_REQUIRES_LABELS", "PATIENT_ANALYSIS_DISABLED"):
                error.message += (
                    f" How often the two runs agree: `histopilot run summary {left} "
                    f"--comparison {right}`."
                )
            raise
        return Result(data, text=lambda: output.record(data))
