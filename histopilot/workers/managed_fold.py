"""Task Center entry point for one fold of a frozen training batch.

Pinned with the batch: ``python -u -m histopilot.workers.managed_fold <batch plan.json> <runId>``
runs from ``<batch>/compute``. The runner chooses the device through
``CUDA_VISIBLE_DEVICES``, so the run plan (and its fit receipt) is identical on every attempt.
"""

import sys
from pathlib import Path

from histopilot.storage.project_lock import ensure_managed_directory
from histopilot.workers.packing_process import write_json
from histopilot.workers.train_batch import execute_plan, run_fold_worker
from histopilot.workers.training_process import compute_snapshot, read_json


def run(plan_path: Path, run_id: str) -> None:
    if not plan_path.is_absolute():
        raise SystemExit("The batch plan path must be absolute; run paths derive from it.")
    batch = read_json(plan_path)
    if batch.get("code") not in (None, compute_snapshot()):
        raise SystemExit(
            "Training code changed after this execution was prepared. Clone a new batch."
        )
    run = next((row for row in batch["runs"] if row["id"] == run_id), None)
    if run is None:
        raise SystemExit(f"Run {run_id} is not part of this batch plan.")
    folder = plan_path.parent / "runs" / run_id
    ensure_managed_directory(folder)
    # Only the device kind enters the plan; the physical GPU is the runner's choice.
    gpu = 0 if batch["resources"].get("gpuIds") else None
    write_json(folder / "plan.json", execute_plan(batch, run, gpu))
    run_fold_worker(folder / "plan.json")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python -m histopilot.workers.managed_fold PLAN.json RUN_ID")
    run(Path(sys.argv[1]), sys.argv[2])
