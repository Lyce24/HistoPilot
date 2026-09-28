"""Deterministic Task Center identifiers; folder arguments are absolute resolved paths."""

import hashlib


def _digest(*parts: str) -> str:
    return hashlib.sha256("\0".join(str(part) for part in parts).encode()).hexdigest()[:32]


def owner_key(kind: str, identity: str, project_folder: str) -> str:
    return "owner-" + _digest(kind, identity, project_folder)


def task_id(kind: str, *parts: str) -> str:
    return "task-" + _digest(kind, *parts)


def fold_task_id(batch_folder: str, run_id: str) -> str:
    return task_id("mil-fold", str(batch_folder), run_id)


def collect_task_id(batch_folder: str, final: bool) -> str:
    return task_id("mil-collect", str(batch_folder), "final" if final else "progress")


def compute_task_id(compute_folder: str) -> str:
    return task_id("compute-job", str(compute_folder))


def coordinator_task_id(coordinator_folder: str) -> str:
    return task_id("predictor-coordinator", str(coordinator_folder))


def bulk_submit_task_id(project_folder: str, batch_id: str) -> str:
    return task_id("bulk-submit", str(project_folder), batch_id)
