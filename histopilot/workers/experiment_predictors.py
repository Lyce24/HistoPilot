"""Persistent experiment-owned predictor coordinator, independent of the API.

A busy project is never a failure: the coordinator keeps looping while another request
holds the project lock, and after ``BUSY_WAIT_SECONDS`` without progress it exits with
EX_TEMPFAIL (75) and leaves its state untouched, so the Task Center requeues it with a
backoff. Finished items are kept; the next attempt continues with the rest.
"""

import os
import sys
import time
import traceback
from pathlib import Path

from histopilot.application.experiment_predictors import ExperimentPredictorService
from histopilot.storage.filesystem import LocalFilesystem
from histopilot.storage.io import read_json_bounded, utc_now, write_json_atomic
from histopilot.storage.lifecycle import lifecycle_guard
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import ScientificStore
from histopilot.workers.compute_archive import prepare_compute_archive
from histopilot.workers.packing_process import output_lock
from histopilot.workers.training_process import process_identity

FINISHED = {"completed", "cancelled", "attention"}
BUSY_EXIT = 75
BUSY_CODES = {"PROJECT_BUSY", "OUTPUT_BUSY"}
BUSY_WAIT_SECONDS = 120
POLL_SECONDS = 3


def _busy(error) -> bool:
    return getattr(error, "code", None) in BUSY_CODES


def run(plan_path, *, sleep=time.sleep, clock=time.monotonic) -> int:
    plan_path = Path(plan_path).absolute()
    plan = read_json_bounded(plan_path)
    store = ScientificStore(Path(plan["projectFolder"]), plan["projectId"])
    filesystem = LocalFilesystem(tuple(Path(root) for root in plan["dataRoots"]))
    service = ExperimentPredictorService(store, filesystem)
    identity = plan["experimentId"]
    folder = service.folder(identity)
    if plan_path != folder / "plan.json":
        raise ValueError("The coordinator plan is outside its frozen experiment folder.")
    prepare_compute_archive(folder, plan["executionContract"]["code"])
    # API Python stays lightweight; only isolated refit workers use this captured
    # training interpreter. Its identity and versions are checked before launch.
    os.environ["HISTOPILOT_TRAINING_PYTHON"] = plan["executionContract"]["runtime"]["python"]
    try:
        with output_lock(folder):
            return _coordinate(service, store, identity, folder, sleep=sleep, clock=clock)
    except StorageError as error:
        if not _busy(error):
            raise
        print(f"{utc_now()} {error.code}: {error} The Task Center retries later.", flush=True)
        return BUSY_EXIT


def _coordinate(service, store, identity, folder, *, sleep, clock) -> int:
    with lifecycle_guard(store.folder):
        _plan, state = service._read(identity)
        state.update(process=process_identity(os.getpid()), updatedAt=utc_now())
        write_json_atomic(folder / "state.json", state)
    busy_since = None
    try:
        while True:
            try:
                status = service.advance(identity)
            except StorageError as error:
                if not _busy(error):
                    raise
                busy_since = clock() if busy_since is None else busy_since
                if clock() - busy_since >= BUSY_WAIT_SECONDS:
                    raise
                sleep(POLL_SECONDS)
                continue
            busy_since = None
            if status["status"] in FINISHED:
                return 0
            sleep(POLL_SECONDS)
    except Exception as error:
        if _busy(error):
            raise  # the caller exits 75; the saved state stays as it was
        traceback.print_exc()
        with lifecycle_guard(store.folder):
            _plan, state = service._read(identity)
            state.update(
                status="attention",
                error={
                    "code": getattr(error, "code", "EXPERIMENT_PREDICTOR_FAILED"),
                    "message": str(error),
                },
                updatedAt=utc_now(),
            )
            write_json_atomic(folder / "state.json", state)
        return 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1]))
