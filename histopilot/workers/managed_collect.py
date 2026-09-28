"""Task Center entry point that collects a batch's finished folds into results.json.

Pinned with the batch: ``python -u -m histopilot.workers.managed_collect <batch plan.json>
[--final]`` runs from ``<batch>/compute`` so OOF files carry the batch's own code identity.
It never writes ``state.json``; the runner's adapter applies ``collect-result.json``.
"""

import sys
import traceback
from contextlib import ExitStack
from pathlib import Path

from histopilot.storage.io import content_hash, read_json_bounded, utc_now, write_json_atomic
from histopilot.storage.project_lock import StorageError
from histopilot.workers.packing_process import output_lock
from histopilot.workers.train_batch import collect_results
from histopilot.workers.training_process import ACTIVE

BUSY_EXIT = 75  # another collector owns this batch output


def final_status(state: dict, *, cancel_requested: bool) -> str:
    statuses = [run.get("status") for run in state.get("runs", [])]
    if cancel_requested:
        return "cancelled"
    if "interrupted" in statuses:
        return "interrupted"
    if "failed" in statuses:
        return "failed"
    if "cancelled" in statuses:
        return "cancelled"
    if statuses and all(status == "completed" for status in statuses):
        return "completed"
    return "interrupted"


def collect(plan_path: Path, *, final: bool) -> int:
    folder = plan_path.parent
    with ExitStack() as stack:
        try:
            stack.enter_context(output_lock(folder))
        except StorageError as error:
            if error.code != "OUTPUT_BUSY":
                raise
            print(f"{utc_now()} Results collection deferred: {error}", flush=True)
            return BUSY_EXIT
        receipt = {"status": None, "final": final, "at": None, "error": None}
        try:
            plan = read_json_bounded(plan_path)
            state = read_json_bounded(folder / "state.json")
            if state.get("planHash") != content_hash(plan):
                raise ValueError("The batch plan changed after launch; results were not collected.")
            if not final and state.get("status") not in ACTIVE:
                # A finished batch's results carry its final status; never regress them.
                receipt.update(status=state.get("status"), skipped=True, at=utc_now())
            else:
                status = (
                    final_status(state, cancel_requested=(folder / "cancel.json").exists())
                    if final
                    else "running"
                )
                collect_results(plan, {**state, "status": status}, folder)
                receipt.update(status=status, at=utc_now())
        except Exception as error:
            traceback.print_exc()
            receipt.update(at=utc_now(), error=str(error) or type(error).__name__)
            write_json_atomic(folder / "collect-result.json", receipt)
            return 1
        write_json_atomic(folder / "collect-result.json", receipt)
        print(f"{utc_now()} Results collected ({receipt['status']}, final={final}).", flush=True)
        return 0


if __name__ == "__main__":
    arguments = sys.argv[1:]
    final = "--final" in arguments
    paths = [item for item in arguments if item != "--final"]
    if len(paths) != 1 or len(arguments) - len(paths) > 1:
        raise SystemExit("Usage: python -m histopilot.workers.managed_collect PLAN.json [--final]")
    raise SystemExit(collect(Path(paths[0]), final=final))
