"""The client's private state on this machine: saved default projects and the journal."""

import json
import os
from pathlib import Path

from .journal import OperationJournal


def cli_dir() -> Path:
    """`<state directory>/cli`, private to this user like the Task Center's own state."""
    from histopilot.taskcenter.paths import state_dir

    path = state_dir() / "cli"
    path.mkdir(mode=0o700, exist_ok=True)
    return path


def journal() -> OperationJournal:
    return OperationJournal(cli_dir() / "operations")


def _defaults_path() -> Path:
    return cli_dir() / "defaults.json"


def _read_defaults() -> dict:
    try:
        value = json.loads(_defaults_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def saved_project(url: str) -> str | None:
    """The default project `histopilot use` saved for this service URL."""
    project = (_read_defaults().get("projects") or {}).get(url)
    return project if isinstance(project, str) and project else None


def save_project(url: str, project: str | None) -> None:
    defaults = _read_defaults()
    projects = dict(defaults.get("projects") or {})
    if project:
        projects[url] = project
    else:
        projects.pop(url, None)
    defaults["projects"] = projects
    path = _defaults_path()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(defaults, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)
