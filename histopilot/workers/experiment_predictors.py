"""Persistent experiment-owned predictor coordinator, independent of the API."""

import os
import sys
import time
import traceback
from pathlib import Path

from histopilot.application.experiment_predictors import ExperimentPredictorService
from histopilot.storage.filesystem import LocalFilesystem
from histopilot.storage.lifecycle import lifecycle_guard
from histopilot.storage.scientific import ScientificStore
from histopilot.workers.compute_archive import prepare_compute_archive
from histopilot.workers.packing_process import output_lock, write_json
from histopilot.workers.training_process import now, process_identity, read_json


def run(plan_path):
    plan_path = Path(plan_path).absolute()
    plan = read_json(plan_path)
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
    with output_lock(folder):
        with lifecycle_guard(store.folder):
            _plan, state = service._read(identity)
            state.update(process=process_identity(os.getpid()), updatedAt=now())
            write_json(folder / "state.json", state)
        try:
            while True:
                status = service.advance(identity)
                if status["status"] in {"completed", "cancelled", "attention"}:
                    return
                time.sleep(3)
        except Exception as error:
            traceback.print_exc()
            with lifecycle_guard(store.folder):
                _plan, state = service._read(identity)
                state.update(
                    status="attention",
                    error={
                        "code": getattr(error, "code", "EXPERIMENT_PREDICTOR_FAILED"),
                        "message": str(error),
                    },
                    updatedAt=now(),
                )
                write_json(folder / "state.json", state)


if __name__ == "__main__":
    run(sys.argv[1])
