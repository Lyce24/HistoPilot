"""The noun-verb commands over `histopilot.client`; see docs/cli-contract.md."""

import typer

from . import (
    actions,
    agent,
    authoring,
    confirm,
    experiments,
    login,
    operations,
    projects,
    records,
    runs,
    service,
    tasks,
    tokens,
)


def register_commands(app: typer.Typer) -> None:
    service.register(app)
    agent.register(app)
    tokens.register(app)
    confirm.register(app)
    login.register(app)
    projects.register(app)
    experiment = experiments.register(app)
    groups = records.register(app)
    runs.register(groups["run"])
    authoring.register(groups)
    authoring.register_preparation(groups)
    actions.register(groups, experiment)
    tasks.register(app)
    operations.register(app)
