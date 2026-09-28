"""One persistent refit, evaluation or interpretation worker, run as a Task Center task.

The runner owns admission and the resource lease and chooses the GPU
(``HISTOPILOT_TASK_GPU``). Under the Task Center (``HISTOPILOT_TASK_MANAGED=1``) the worker
also checks that the record is still assigned to its task and follows the busy contract.
"""

import os
import signal
import sys
import time
import traceback
from pathlib import Path

from histopilot.application.feature_bundles import _hash
from histopilot.storage.lifecycle import lifecycle_guard
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import ScientificStore
from histopilot.workers.compute_archive import prepare_compute_archive
from histopilot.workers.packing_process import output_lock, write_json
from histopilot.workers.train_batch import _check_inputs
from histopilot.workers.training_process import (
    now,
    process_identity,
    read_json,
    stop_owned_processes,
)

# Archives whose worker defines this constant can run under the Task Center runner.
TASK_CENTER_PROTOCOL = 1
MANAGED_STATUSES = {"queued", "running", "interrupted"}
# The Task Center busy contract: a managed worker that cannot get the workspace or its
# output lock within a short wait exits with EX_TEMPFAIL and leaves its record untouched;
# the runner frees the slot and requeues the task with a backoff.
BUSY_EXIT = 75
BUSY_CODES = {"PROJECT_BUSY", "OUTPUT_BUSY"}
BUSY_WAIT_SECONDS = 60
# The check after compute guards finished work; losing it costs a full rerun, so it waits
# longer before giving the slot back.
FINISHED_BUSY_WAIT_SECONDS = 15 * 60


class Busy(Exception):
    """The workspace stayed busy; the managed worker yields its slot (exit 75)."""


def _managed():
    return os.environ.get("HISTOPILOT_TASK_MANAGED") == "1"


def _verify_inputs(plan, folder, *, managed, wait=BUSY_WAIT_SECONDS):
    """``verify_plan_inputs``; a managed worker retries while the workspace is busy, then
    raises ``Busy``."""
    deadline = time.monotonic() + (wait if managed else 0)
    while True:
        try:
            return verify_plan_inputs(plan)
        except StorageError as error:
            if error.code != "PROJECT_BUSY" or not managed:
                raise
            if time.monotonic() >= deadline:
                raise Busy(str(error)) from error
        if (folder / "cancel.requested").exists():
            raise KeyboardInterrupt("Compute cancellation requested.")
        time.sleep(1)


def verify_plan_inputs(plan):
    """Check immutable references and source/pack stamps before and after compute."""
    store = ScientificStore(Path(plan["projectFolder"]), plan["projectId"])
    with lifecycle_guard(store.folder):
        record = store.get_configuration(plan["recordId"])
        if record["contentHash"] != plan["recordContentHash"]:
            raise ValueError("The owning compute record changed.")
        store.lifecycle.assert_document_usable(record)
        manifest = record["manifest"]
        if plan["kind"] == "refit":
            template = manifest.get("planTemplate")
            if (
                manifest.get("kind") != "predictor-refit"
                or not template
                or any(plan.get(key) != value for key, value in template.items())
            ):
                raise ValueError("The refit execution differs from its immutable build plan.")
        elif plan["kind"] == "evaluation":
            predictor = store.get_configuration(manifest["predictorId"])
            cohort = store.get_configuration(manifest["cohortId"])
            feature = store.get_configuration(manifest["features"]["feature"]["id"])
            selected = {row["slideId"] for row in cohort["manifest"]["memberships"]}
            files = {
                row["slideId"]: row
                for row in feature["manifest"]["files"]
                if row["slideId"] in selected
            }
            expected_data = {
                "memberships": cohort["manifest"]["memberships"],
                "featureDim": manifest["features"]["dimensions"],
                "featureFiles": files,
                "sourceStamps": {row["path"]: row for row in files.values()},
            }
            if (
                manifest.get("kind") != "model-evaluation"
                # Inference plans skip label metrics; the saved record decides that.
                or plan.get("purpose")
                != ("inference" if manifest.get("purpose") == "inference" else None)
                or plan["checkpoints"] != predictor["manifest"]["checkpoints"]
                or plan["target"] != manifest["target"]
                or plan["target"] != predictor["manifest"]["target"]
                or plan["inference"] != manifest["inference"]
                or plan["method"] != predictor["manifest"].get("method", "ensemble")
                or any(plan["data"].get(key) != value for key, value in expected_data.items())
            ):
                raise ValueError(
                    "Evaluation inputs differ from the immutable predictor and test cohort."
                )
            packed = plan["inference"]["loadingPolicy"] == "packed"
            pack_path = None
            if packed:
                # Independent cohorts freeze membership and labels. The saved
                # evaluation owns its reviewed feature bundle and loading choice,
                # including overrides of a historical cohort's original pack.
                binding = manifest["features"]["bundle"]
                bundle = store.get_configuration(binding["id"])
                packs = [
                    item
                    for item in bundle["manifest"].get("packs", [])
                    if item["id"] == manifest["inference"]["packArtifactId"]
                ]
                if (
                    bundle["contentHash"] != binding["contentHash"]
                    or bundle["manifest"].get("kind") != "feature-bundle"
                    or bundle["manifest"]["feature"]["id"] != feature["id"]
                    or len(packs) != 1
                    or packs[0]["featureSetId"] != feature["id"]
                    or packs[0]["sourceContentHash"] != manifest["features"]["sourceContentHash"]
                ):
                    raise ValueError("The saved evaluation pack binding changed.")
                pack_path = packs[0]["outputPath"]
            if (
                plan["data"].get("loadingPolicy") != ("mmap" if packed else "native")
                or (packed and plan["data"].get("packPath") != pack_path)
                or (not packed and (plan["data"].get("packPath") or plan["data"].get("packStamps")))
            ):
                raise ValueError("The test feature loading contract changed.")
        elif plan["kind"] == "interpretation":
            predictor = store.get_configuration(manifest["predictorId"])
            if (
                manifest.get("kind") != "model-interpretation"
                or plan["checkpoints"] != predictor["manifest"]["checkpoints"]
                or plan["target"] != predictor["manifest"]["target"]
                or plan["target"] != manifest["target"]
                or plan["method"] != predictor["manifest"].get("method", "ensemble")
                or any(
                    plan.get(key) != manifest.get(key)
                    for key in ("slides", "featureContract", "references", "resources")
                )
                or plan["data"]
                != {
                    "sourceStamps": {
                        path: stamp
                        for slide in manifest["slides"]
                        for path, stamp in slide["sourceStamps"].items()
                    }
                }
            ):
                raise ValueError(
                    "Interpretation differs from its immutable predictor and slide evidence."
                )
            from histopilot.storage.attention_inputs import verify_sources

            verify_sources(plan["slides"])
        for expected in plan.get("references", []):
            actual = store.get_configuration(expected["id"])
            if actual["contentHash"] != expected["contentHash"]:
                raise ValueError("A frozen compute input changed.")
    _check_inputs(plan["data"])
    if plan.get("checkpoints"):
        from histopilot.application.predictors import checkpoint_snapshot

        for expected in plan["checkpoints"]:
            actual = checkpoint_snapshot(expected["path"], store.folder)
            if any(actual[key] != expected[key] for key in ("path", "bytes", "sha256")):
                raise ValueError("A predictor checkpoint changed.")


def execute(path):
    path = Path(path).absolute()
    folder = path.parent
    managed = _managed()
    state = read_json(folder / "state.json")
    plan = read_json(path)
    # All loader children inherit this private session; cancellation and cleanup
    # can still recognize them if the main worker is abruptly terminated.
    if os.getsid(0) != os.getpid():
        os.setsid()

    def stopped(_signum, _frame):
        raise KeyboardInterrupt("Compute cancellation requested.")

    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGINT, stopped)
    busy = False
    with output_lock(folder):
        if managed:
            # Fence: another launch may have reassigned or finished this record since
            # the task was queued. Never touch a state that belongs to someone else.
            state = read_json(folder / "state.json")
            task = os.environ.get("HISTOPILOT_TASK_ID")
            if (
                not task
                or state.get("taskId") != task
                or state.get("status") not in MANAGED_STATUSES
            ):
                print(
                    "This compute record is not assigned to Task Center task "
                    f"{task or '(none)'} in a runnable state; nothing to do.",
                    flush=True,
                )
                return state
        try:
            if _hash(plan) != state["planHash"]:
                raise ValueError("The immutable execution plan changed.")
            prepare_compute_archive(folder, plan["code"])
            state.update(
                process=process_identity(),
                processGroupId=os.getpid(),
                status="queued",
                updatedAt=now(),
            )
            write_json(folder / "state.json", state)
            resources = plan["resources"]
            # The runner admitted this task and holds its lease; it chose the device.
            if (folder / "cancel.requested").exists():
                raise KeyboardInterrupt("Compute cancellation requested.")
            assigned = os.environ.get("HISTOPILOT_TASK_GPU", "")
            gpu = int(assigned) if assigned else None
            os.environ.update(
                CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu),
                OMP_NUM_THREADS=str(resources["cpuThreadsPerRun"]),
                MKL_NUM_THREADS=str(resources["cpuThreadsPerRun"]),
                OPENBLAS_NUM_THREADS=str(resources["cpuThreadsPerRun"]),
            )
            try:
                _verify_inputs(plan, folder, managed=managed)
            except Busy:
                # Nothing ran yet: the record stays queued for the requeued attempt.
                state.update(status="queued", updatedAt=now())
                write_json(folder / "state.json", state)
                raise
            state.update(status="running", gpu=gpu, updatedAt=now())
            write_json(folder / "state.json", state)
            execution = {**plan, "device": "cpu" if gpu is None else "cuda"}
            if plan["kind"] == "refit":
                from histopilot.training.refit import train_refit

                checkpoint = folder / "last.ckpt"
                result = train_refit(
                    execution, folder, checkpoint_path=checkpoint if checkpoint.exists() else None
                )
            elif plan["kind"] == "evaluation":
                from histopilot.training.inference import evaluate

                result = evaluate(execution, folder)
            elif plan["kind"] == "interpretation":
                from histopilot.training.attention import interpret

                result = interpret(execution, folder)
            else:
                raise ValueError("Unsupported compute job kind.")
            if (folder / "cancel.requested").exists():
                raise KeyboardInterrupt("Compute cancellation requested.")
            _verify_inputs(plan, folder, managed=managed, wait=FINISHED_BUSY_WAIT_SECONDS)
            write_json(folder / "result.json", result)
            state.update(status="completed", result=result, error=None)
        except Busy:
            busy = True
            raise
        except (KeyboardInterrupt, SystemExit) as error:
            state.update(
                status="cancelled" if (folder / "cancel.requested").exists() else "interrupted",
                error=str(error),
                result=None,
            )
        except Exception as error:
            traceback.print_exc()
            state.update(
                status="cancelled" if (folder / "cancel.requested").exists() else "failed",
                error=str(error),
                result=None,
            )
        finally:
            try:
                if state.get("process"):
                    stop_owned_processes(state["process"], exclude_pid=os.getpid())
            except Exception as error:
                traceback.print_exc()
                state.update(status="failed", error=str(error), result=None)
            if not busy:
                state.update(updatedAt=now())
                write_json(folder / "state.json", state)
    return state


def main(path) -> int:
    try:
        execute(path)
    except Busy as error:
        print(f"{now()} PROJECT_BUSY: {error} The Task Center retries later.", flush=True)
        return BUSY_EXIT
    except StorageError as error:
        if not _managed():
            raise
        traceback.print_exc()
        print(f"{error.code}: {error}", file=sys.stderr, flush=True)
        # An output held by another worker is the busy contract; anything else failed.
        return BUSY_EXIT if error.code in BUSY_CODES else 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
