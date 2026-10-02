"""Objectives, and filing an experiment under one with a role and a series.

An objective is named by its ID or by its name; names are unique in a project without
regard to case or repeated spaces. Filing replaces an experiment's whole filing, so changing
one part reads the saved filing first and sends it as `expected`.
"""

from .api import Client, segment
from .errors import ClientError
from .resources import experiment_path, listed_items, project_path

FILING_FIELDS = ("objectiveId", "role", "series", "seriesMember")


def objectives_path(project: str, identity: str = "") -> str:
    base = f"{project_path(project)}/objectives"
    return f"{base}/{segment(identity)}" if identity else base


def objectives(client: Client, project: str) -> list[dict]:
    return listed_items(client.get(objectives_path(project)), "items")


def _folded(name: str) -> str:
    return " ".join(name.split()).casefold()


def objective_id(client: Client, project: str, name: str) -> str:
    """The objective ``name`` stands for: an ID as it is, else the objective of that name."""
    if name.startswith("objective-"):
        return name
    for item in objectives(client, project):
        if _folded(item["name"]) == _folded(name):
            return item["id"]
    raise ClientError(
        f"No objective named {name!r} in this project; `histopilot objective list` shows them.",
        code="OBJECTIVE_NOT_FOUND",
        kind="not-found",
    )


def objective(client: Client, project: str, identity: str) -> dict:
    return client.get(objectives_path(project, identity))


def filing(record: dict) -> dict:
    """An experiment's filing as the service lists it: objective ID, role, series, member."""
    return {key: record.get(key) for key in FILING_FIELDS}


def file_experiment(client: Client, project: str, record: dict, wanted: dict) -> dict:
    """File ``record`` as ``wanted`` (all four fields), if its filing is still the one read."""
    return client.request(
        "PUT",
        f"{experiment_path(project, record['id'])}/objective",
        body={**wanted, "expected": filing(record)},
    )
