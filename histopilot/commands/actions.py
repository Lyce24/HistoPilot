"""Science actions on one record: launch, resume, cancel and publish, each confirmed first."""

import typer

from histopilot.client import records
from histopilot.client.api import segment
from histopilot.client.resources import experiment_path, project_path
from histopilot.client.wait import wait

from .common import command
from .records import name_help

# noun → route under the project, the actions it has, and whether they take an operation ID.
ACTIONS = {
    "batch": ("/mil-experiments/batches/{id}", ("launch", "resume", "cancel"), True),
    "run": ("/evaluation-runs/{id}", ("launch", "resume", "cancel"), True),
    "apply": ("/evaluation-runs/bulk/{id}", ("cancel",), True),
    "refit": ("/predictors/refits/{id}", ("launch", "resume", "cancel", "publish"), True),
    "interpretation": ("/interpretations/{id}", ("launch", "resume", "cancel"), True),
    "extraction": ("/extractions/{id}", ("resume", "cancel"), False),
    "pack": ("/feature-packs/{id}", ("cancel",), False),
}
HELP = {
    "launch": "Start this record's work in the Task Center.",
    "resume": "Resume stopped or failed work; finished parts are kept.",
    "cancel": "Cancel this record's unfinished work.",
    "publish": "Publish the completed refit as a predictor.",
}
STARTS = ("launch", "resume")


def _action(group: typer.Typer, noun: str, action: str, route: str, operation: bool) -> None:
    @command(group, action, commit=True, waits=action in STARTS, help=HELP[action])
    def run(ctx, name: str = typer.Argument(..., metavar="RECORD", help=name_help(noun))):
        project = ctx.project()
        identity = records.resolve(ctx.client, project, noun, name)
        current = records.record(ctx.client, project, noun, identity)
        path = project_path(project) + route.replace("{id}", segment(identity))
        preview = {
            "action": action,
            "record": {
                "id": identity,
                "status": records.status_of(current),
                "runState": current["runState"],
            },
        }

        def send():
            if operation:
                result = ctx.client.operation(
                    "POST",
                    f"{path}/{action}",
                    {},
                    prefix=f"{noun}-{action}",
                    operation_id=ctx.operation_id,
                )
            else:
                result = ctx.client.request("POST", f"{path}/{action}")
            if not (ctx.wait and action in STARTS):
                return result
            return wait(ctx.client, lambda: records.record(ctx.client, project, noun, identity))

        return ctx.commit(preview, send, summary=f"{action.capitalize()} {noun} {name}.")


def _predictor_actions(group: typer.Typer) -> None:
    for action in ("resume", "cancel"):
        _experiment_predictors(group, action)


def _experiment_predictors(group: typer.Typer, action: str) -> None:
    @command(
        group,
        f"{action}-predictors",
        commit=True,
        help=f"{action.capitalize()} an experiment's automatic predictor work.",
    )
    def run(ctx, experiment: str = typer.Argument(..., help="Experiment ID or name.")):
        experiment = ctx.experiment(experiment)
        path = f"{experiment_path(ctx.project(), experiment)}/predictors/{action}"
        return ctx.commit(
            {"action": f"{action}-predictors", "experimentId": experiment},
            lambda: ctx.client.operation(
                "POST", path, {}, prefix=f"predictors-{action}", operation_id=ctx.operation_id
            ),
            summary=f"{action.capitalize()} predictor work of {experiment}.",
        )


def register(groups: dict[str, typer.Typer], experiment: typer.Typer) -> None:
    for noun, (route, actions, operation) in ACTIONS.items():
        for action in actions:
            _action(groups[noun], noun, action, route, operation)
    _predictor_actions(experiment)
