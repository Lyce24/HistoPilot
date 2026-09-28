"""Live-code Task Center task entrypoints: ``python -m histopilot.taskcenter.jobs <job> ...``.

These run the current checkout (not a pinned archive) and exit 0 on success.
"""

import argparse
import sys
import time
from pathlib import Path

# Member submission takes the project lifecycle lock once per member; other launches may
# hold it longer than its 5 s wait. A replay only submits missing members, so it retries
# briefly and then, under the busy contract, exits 75: the runner requeues it with a
# backoff instead of keeping a task waiting on a lock.
BUSY_EXIT = 75
BUSY_WAIT_SECONDS = 60
BUSY_RETRY_DELAY = 2.0


class Busy(Exception):
    """The workspace stayed busy; the task yields (exit 75) and runs again later."""


def bulk_submit(project_folder, project_id, batch_id, data_roots=(), *, sleep=time.sleep) -> None:
    """Submit an evaluation batch's missing members, exactly as the API would inline."""
    from histopilot.application.bulk_evaluations import BulkEvaluationService
    from histopilot.storage.filesystem import LocalFilesystem
    from histopilot.storage.project_lock import StorageError
    from histopilot.storage.scientific import ScientificStore

    store = ScientificStore(Path(project_folder), project_id)
    filesystem = LocalFilesystem(tuple(Path(root) for root in data_roots))
    batch = store.get_configuration(batch_id, include_inactive=True)
    if batch["manifest"].get("kind") != "evaluation-batch":
        raise StorageError("Evaluation batch not found.", "EVALUATION_BATCH_NOT_FOUND", 404)
    service = BulkEvaluationService(store, filesystem)
    deadline = time.monotonic() + BUSY_WAIT_SECONDS
    while True:
        try:
            service._submit(batch)
            return
        except StorageError as error:
            if error.code != "PROJECT_BUSY":
                raise
            if time.monotonic() >= deadline:
                raise Busy(str(error)) from error
            print(f"Workspace busy; retrying member submission: {error}", flush=True)
        sleep(BUSY_RETRY_DELAY)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m histopilot.taskcenter.jobs")
    jobs = parser.add_subparsers(dest="job", required=True)
    bulk = jobs.add_parser("bulk-submit", help="Submit the members of an evaluation batch.")
    bulk.add_argument("--project-folder", required=True)
    bulk.add_argument("--project-id", required=True)
    bulk.add_argument("--batch", required=True)
    bulk.add_argument("--data-root", action="append", default=[])
    args = parser.parse_args(argv)
    if args.job == "bulk-submit":
        try:
            bulk_submit(args.project_folder, args.project_id, args.batch, args.data_root)
        except Busy as error:
            print(f"PROJECT_BUSY: {error} The Task Center retries later.", flush=True)
            return BUSY_EXIT
    return 0


if __name__ == "__main__":
    sys.exit(main())
