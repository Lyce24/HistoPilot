"""Names and paths that stored tasks, records and pinned archives depend on.

Task rows keep their adapter name and argv. Pinned compute archives are started by
module path and read for their Task Center protocol. Renaming or moving any of these
strands work that is already queued or archived, so a change here needs a migration or a
compatibility shim, not only an edit.
"""

import importlib
import importlib.util
from pathlib import Path

import pytest

from histopilot.application import compute_jobs, feature_packs, portability_jobs
from histopilot.taskcenter.adapters import ADAPTERS
from histopilot.workers.training_process import compute_snapshot

PACKAGE = Path(compute_jobs.__file__).resolve().parents[1]

# Adapter names stored in task rows.
STORED_ADAPTERS = {
    "generic",
    "mil-fold",
    "mil-collect",
    "compute-job",
    "predictor-coordinator",
    "bulk-submit",
    "extraction",
    "extraction-validation",
    "packing",
    "archive",
}

# Started with ``python -m``; queued tasks and pinned archives store these names.
ENTRY_MODULES = (
    "histopilot.workers.managed_fold",
    "histopilot.workers.managed_collect",
    "histopilot.workers.compute_job",
    "histopilot.workers.experiment_predictors",
    "histopilot.workers.pack_features",
    "histopilot.workers.verify_extraction",
    "histopilot.taskcenter.jobs",
)


def test_every_stored_adapter_name_still_resolves():
    assert STORED_ADAPTERS <= set(ADAPTERS)
    for target in ADAPTERS.values():
        module, _, name = target.partition(":")
        assert hasattr(importlib.import_module(module), name)


@pytest.mark.parametrize("module", ENTRY_MODULES)
def test_worker_entry_modules_keep_their_names(module):
    assert importlib.util.find_spec(module) is not None


def test_worker_files_started_by_path_exist():
    for path in (
        feature_packs.WORKER,
        portability_jobs.WORKER,
        PACKAGE / "adapters" / "trident" / "runner.py",
    ):
        assert Path(path).is_file()


def test_archived_modules_declare_the_protocol_where_launches_read_it():
    for relative in ("workers/compute_job.py", "application/experiment_predictors.py"):
        assert compute_jobs.archive_protocol(PACKAGE / relative) == 1


def test_the_code_fingerprint_names_only_existing_files():
    snapshot = compute_snapshot()
    assert snapshot["files"]
    assert all((PACKAGE / name).is_file() for name in snapshot["files"])
