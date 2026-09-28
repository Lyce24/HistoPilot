"""Where extraction and feature-pack outputs may be written inside a project.

Outputs may be siblings of project data (``trident/``, ``feature-packs/``), never the
project folder itself, its metadata or its record folders. One list serves every writer
so the checks cannot drift apart again.
"""

from pathlib import Path

PROTECTED_PROJECT_ENTRIES = (
    "datasets",
    "configurations",
    "extractions",
    "packing",
    "training",
    "compute-jobs",
    "predictor-builds",
    "evaluation-batches",
    "jobs",
    "drafts",
    ".staging",
    ".trash",
    ".git",
    ".codex",
    ".histopilot-write.lock",
    ".histopilot-lifecycle.lock",
    "histopilot-lifecycle.json",
    "histopilot-project.json",
    "histopilot-state.sqlite",
    "histopilot-state.sqlite-wal",
    "histopilot-state.sqlite-shm",
    "project.sqlite3",
    "histopilot.sqlite3",
)


def overlaps(first: str | Path, second: str | Path) -> bool:
    first, second = Path(first), Path(second)
    return first.is_relative_to(second) or second.is_relative_to(first)


def protected_output(path: str | Path, project_folder: str | Path) -> bool:
    """Whether ``path`` is the project folder or overlaps one of its protected entries."""
    path, project_folder = Path(path), Path(project_folder)
    return path == project_folder or any(
        overlaps(path, project_folder / name) for name in PROTECTED_PROJECT_ENTRIES
    )
