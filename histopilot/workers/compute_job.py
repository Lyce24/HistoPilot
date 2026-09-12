"""One persistent refit/evaluation worker, sharing training resource reservations."""

import os
import signal
import sys
import time
import traceback
from pathlib import Path

from histopilot.application.feature_bundles import _hash
from histopilot.storage.lifecycle import lifecycle_guard
from histopilot.storage.scientific import ScientificStore
from histopilot.workers.compute_archive import prepare_compute_archive
from histopilot.workers.packing_process import output_lock, write_json
from histopilot.workers.train_batch import _capacity, _check_inputs, _leases, available_device
from histopilot.workers.training_process import cpu_slots_per_run, now, process_identity, read_json


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
            if (
                plan["data"].get("loadingPolicy") != ("mmap" if packed else "native")
                or (
                    packed
                    and plan["data"].get("packPath") != cohort["manifest"]["pack"]["outputPath"]
                )
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
    lease_path = None
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
    with output_lock(folder):
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
            while True:
                if (folder / "cancel.requested").exists():
                    raise KeyboardInterrupt("Compute cancellation requested.")
                with _leases() as (registry, active):
                    available, gpu = available_device(resources, active, _capacity())
                    if available:
                        lease_path = registry / f"lease-{os.getpid()}.json"
                        write_json(
                            lease_path,
                            {
                                "process": state["process"],
                                "gpu": gpu,
                                "cpus": cpu_slots_per_run(resources),
                                "ramGb": resources["ramGbPerRun"],
                                "runsPerGpu": resources["runsPerGpu"],
                                "batchId": plan["recordId"],
                                "runId": plan["recordId"],
                            },
                        )
                        break
                state.update(
                    waitingReason="Waiting for requested CPU, RAM, or GPU capacity.",
                    updatedAt=now(),
                )
                write_json(folder / "state.json", state)
                time.sleep(1)
            os.environ.update(
                CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu),
                OMP_NUM_THREADS=str(resources["cpuThreadsPerRun"]),
                MKL_NUM_THREADS=str(resources["cpuThreadsPerRun"]),
                OPENBLAS_NUM_THREADS=str(resources["cpuThreadsPerRun"]),
            )
            verify_plan_inputs(plan)
            state.pop("waitingReason", None)
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
            verify_plan_inputs(plan)
            write_json(folder / "result.json", result)
            state.update(status="completed", result=result, error=None)
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
            state.update(updatedAt=now())
            write_json(folder / "state.json", state)
            if lease_path:
                with _leases():
                    lease_path.unlink(missing_ok=True)
    return state


if __name__ == "__main__":
    execute(sys.argv[1])
