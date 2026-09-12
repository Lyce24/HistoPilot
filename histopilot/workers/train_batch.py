"""Persistent local scheduler and isolated single-fold compute entry point."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from histopilot.application.feature_bundles import _hash
from histopilot.storage.project_lock import (
    StorageError,
    _reject_symlink_components,
    ensure_managed_directory,
    writer_lock,
)
from histopilot.workers.packing_process import output_lock, write_json
from histopilot.workers.training_process import (
    ResourceTelemetry,
    append_event,
    classify_training_failure,
    compute_snapshot,
    cpu_slots_per_run,
    device_health_failure,
    now,
    process_alive,
    process_identity,
    read_json,
    save_state,
)


def _capacity():
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    memory = {
        line.split(":")[0]: int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
    }
    return cpus, memory["MemAvailable"] / 1024**3


@contextmanager
def _leases():
    folder = Path(tempfile.gettempdir()) / f"histopilot-training-{os.getuid()}"
    ensure_managed_directory(folder)
    info = folder.stat()
    if info.st_uid != os.getuid() or info.st_mode & 0o022:
        raise StorageError("Training resource registry is unsafe.", "TRAINING_REGISTRY_UNSAFE", 403)
    with writer_lock(folder, timeout=5):
        active = []
        for path in folder.glob("lease-*.json"):
            value = read_json(path)
            if process_alive(value.get("process")):
                try:
                    status = Path(f"/proc/{value['process']['pid']}/status").read_text()
                    value["rssGb"] = next(
                        int(line.split()[1]) * 1024 / 1024**3
                        for line in status.splitlines()
                        if line.startswith("VmRSS:")
                    )
                except (OSError, ValueError, StopIteration):
                    value["rssGb"] = 0
                active.append(value)
            else:
                path.unlink()
        yield folder, active


def available_device(
    resources: dict, active: list[dict], capacity: tuple
) -> tuple[bool, int | None]:
    """Account for all HistoPilot batches, including workers orphaned by a scheduler crash."""
    cpus, ram = capacity
    needed = cpu_slots_per_run(resources)
    if sum(item["cpus"] for item in active) + needed > cpus:
        return False, None
    # MemAvailable already excludes resident pages. Reserve only each worker's
    # remaining requested headroom, avoiding charging resident RAM twice.
    if (
        sum(max(0, item["ramGb"] - item.get("rssGb", 0)) for item in active)
        + resources["ramGbPerRun"]
        > ram
    ):
        return False, None
    if not resources["gpuIds"]:
        return True, None
    for gpu in resources["gpuIds"]:
        same_gpu = [item for item in active if item["gpu"] == gpu]
        limit = min([resources["runsPerGpu"], *(item["runsPerGpu"] for item in same_gpu)])
        if len(same_gpu) < limit:
            return True, gpu
    return False, None


def _run_plan(batch: dict, run: dict, gpu: int | None) -> dict:
    candidate = next(item for item in batch["configurations"] if item["id"] == run["candidateId"])
    split = next(item for item in batch["splitPlans"] if item["id"] == run["splitPlanId"])
    return {
        "runId": run["id"],
        "batchId": batch["batchId"],
        "batchContentHash": batch["batchContentHash"],
        "code": batch.get("code"),
        "runtime": batch["runtime"],
        "candidateId": run["candidateId"],
        "trainingSeed": run["trainingSeed"],
        "splitPlan": split,
        "recipe": candidate["recipe"],
        "target": batch["target"],
        "resources": batch["resources"],
        "device": "cpu" if gpu is None else "cuda",
        "data": {**batch["data"], "memberships": batch["memberships"][run["splitPlanId"]]},
    }


def _check_inputs(data):
    from histopilot.storage.pack_import import pack_file_stamps
    from histopilot.storage.packed import _check_sources

    _check_sources({"sourceStamps": data["sourceStamps"]})
    if data.get("packPath") and pack_file_stamps(Path(data["packPath"])) != data["packStamps"]:
        raise ValueError("The verified feature pack changed after the batch was launched.")


def _prediction_records(path, target):
    value = json.loads(Path(path).read_text())
    if isinstance(value, dict) and value.get("classOrder") != target["classes"]:
        raise ValueError("Prediction class order differs from the frozen target.")
    return value if isinstance(value, list) else value["records"]


def collect_results(batch: dict, state: dict, folder: Path):
    groups = defaultdict(list)
    splits = {row["id"]: row for row in batch["splitPlans"]}
    for run in state["runs"]:
        groups[
            (run["candidateId"], run["trainingSeed"], splits[run["splitPlanId"]]["seed"])
        ].append(run)
    candidates, oof = [], []
    for (candidate, training_seed, split_seed), runs in groups.items():
        completed = [run for run in runs if run["status"] == "completed"]
        item = {
            "candidateId": candidate,
            "trainingSeed": training_seed,
            "splitSeed": split_seed,
            "completedRuns": len(completed),
            "totalRuns": len(runs),
            "complete": len(completed) == len(runs),
            "metrics": None,
            "oofPath": None,
        }
        if item["complete"]:
            from histopilot.training.module import classification_metrics

            records, expected = [], {}
            for run in runs:
                records.extend(
                    _prediction_records(run["result"]["predictions"]["assessment"], batch["target"])
                )
                for row in batch["memberships"][run["splitPlanId"]]:
                    if row["partition"] == "test":
                        if row["slideId"] in expected:
                            raise ValueError(
                                "An assessment slide appears in multiple folds of one k-fold seed."
                            )
                        expected[row["slideId"]] = row
            actual = Counter(row["slideId"] for row in records)
            if set(actual) != set(expected) or any(count != 1 for count in actual.values()):
                raise ValueError(
                    "OOF predictions must cover each assessment-eligible slide exactly once."
                )
            for record in records:
                expected_row = expected[record["slideId"]]
                if (
                    record["patientId"] != expected_row["patientId"]
                    or record["label"] != expected_row["label"]
                    or type(record.get("labelIndex")) is not int
                    or record["labelIndex"]
                    != batch["target"]["classes"].index(expected_row["label"])
                ):
                    raise ValueError(
                        "An OOF prediction identity or label differs from frozen memberships."
                    )
            summary = classification_metrics(records, batch["target"])
            key = hashlib.sha256(f"{candidate}/{training_seed}/{split_seed}".encode()).hexdigest()[
                :24
            ]
            path = folder / f"oof-{key}.json"
            write_json(
                path,
                {
                    "batchId": batch["batchId"],
                    "candidateId": candidate,
                    "trainingSeed": training_seed,
                    "splitSeed": split_seed,
                    "protocolId": batch["protocolId"],
                    "classOrder": batch["target"]["classes"],
                    "records": records,
                    "summary": summary,
                    "purpose": "development_assessment",
                },
            )
            item.update(
                metrics=summary["selected"],
                metricDetails=summary,
                oofPath=str(path),
                assessmentSlideCount=len(expected),
            )
            oof.append(
                {
                    "path": str(path),
                    "candidateId": candidate,
                    "trainingSeed": training_seed,
                    "splitSeed": split_seed,
                    "slideCount": len(expected),
                }
            )
        candidates.append(item)
    write_json(
        folder / "results.json",
        {
            "batchId": batch["batchId"],
            "status": state["status"],
            "candidates": candidates,
            "oof": oof,
            "selectionNote": "OOF metrics describe development assessment. Comparing hyperparameters on these folds does not provide an independent final-test estimate.",
        },
    )


def run_fold_worker(plan_path: Path):
    plan = read_json(plan_path)
    folder = plan_path.parent
    try:
        if plan.get("code") is not None and plan["code"] != compute_snapshot():
            raise ValueError(
                "Training code changed after this execution was prepared. Clone a new batch."
            )
        _check_inputs(plan["data"])
        from histopilot.training.fold import train_fold

        checkpoint = folder / "last.ckpt"
        result = train_fold(
            plan, folder, checkpoint_path=checkpoint if checkpoint.exists() else None
        )
        _check_inputs(plan["data"])
        write_json(folder / "result.json", result)
    except BaseException as error:
        write_json(
            folder / "failure.json",
            {
                "error": str(error),
                "type": type(error).__name__,
                "category": classify_training_failure(str(error)),
                "traceback": traceback.format_exc(),
                "at": now(),
            },
        )
        raise


def run_batch(plan_path: Path):
    batch, folder = read_json(plan_path), plan_path.parent
    with output_lock(folder):
        state = read_json(folder / "state.json")
        if state["planHash"] != _hash(batch):
            raise ValueError("Batch execution plan changed before worker startup.")
        state.update(status="running", process=process_identity())
        save_state(folder, state)
        print(
            f"{now()} Starting {batch['batchId']} with {len(state['runs'])} planned folds.",
            flush=True,
        )
        running = {}
        cancelled_at = None
        halted_at = None
        signalled = set()
        telemetry = ResourceTelemetry(folder)

        def halt_dispatch(message):
            nonlocal halted_at
            if halted_at is not None:
                return
            halted_at = time.monotonic()
            finding = {
                "severity": "error",
                "code": "TRAINING_GPU_DISPATCH_HALTED",
                "message": message
                + " Further dispatch stopped; unfinished runs can be resumed after the GPU is healthy. Existing checkpoints are retained.",
            }
            state["dispatchHalted"] = finding
            state.setdefault("findings", []).append(finding)
            append_event(folder / "events.jsonl", {"at": now(), **finding})
            for pending in state["runs"]:
                if pending["status"] == "queued":
                    pending.update(status="interrupted", error=finding["message"])

        def request_cancel(_signum, _frame):
            write_json(folder / "cancel.json", {"requestedAt": now(), "source": "signal"})

        signal.signal(signal.SIGTERM, request_cancel)
        signal.signal(signal.SIGINT, request_cancel)
        try:
            if batch.get("code") is not None and batch["code"] != compute_snapshot():
                raise ValueError(
                    "Training code changed after this execution was prepared. Clone a new batch."
                )
            _check_inputs(batch["data"])
            # A lost scheduler can leave a completed child result. Adopt only this exact run.
            for run in state["runs"]:
                result_path = Path(run["outputPath"]) / "result.json"
                if run["status"] != "completed" and result_path.exists():
                    result = read_json(result_path)
                    if result.get("runId") == run["id"] and result.get("state") == "succeeded":
                        run.update(
                            status="completed",
                            result=result,
                            metrics=result.get("metrics"),
                            checkpointPath=result.get("bestCheckpointPath"),
                        )
            while any(run["status"] in {"queued", "running"} for run in state["runs"]):
                observation = telemetry.record(state)
                if observation:
                    health_error = device_health_failure(
                        state.get("provenance", batch["runtime"]),
                        observation,
                        batch["resources"]["gpuIds"],
                    )
                    if health_error:
                        halt_dispatch(health_error)
                cancelled = (folder / "cancel.json").exists()
                if cancelled:
                    cancelled_at = cancelled_at or time.monotonic()
                    state["cancelRequested"] = True
                    for run in state["runs"]:
                        if run["status"] == "queued":
                            run["status"] = "cancelled"
                if cancelled or halted_at is not None:
                    stop_started = cancelled_at if cancelled_at is not None else halted_at
                    for process, _stream, _lease in running.values():
                        if process.poll() is None:
                            try:
                                if process.pid not in signalled:
                                    os.killpg(process.pid, signal.SIGTERM)
                                    signalled.add(process.pid)
                                elif time.monotonic() - stop_started > 30:
                                    os.killpg(process.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                for run in state["runs"]:
                    if run["id"] not in running:
                        continue
                    process, stream, lease_path = running[run["id"]]
                    code = process.poll()
                    if code is None:
                        continue
                    stream.close()
                    with _leases():
                        lease_path.unlink(missing_ok=True)
                    result_path = Path(run["outputPath"]) / "result.json"
                    if code == 0 and result_path.exists():
                        result = read_json(result_path)
                        if result.get("runId") != run["id"] or result.get("state") != "succeeded":
                            raise ValueError("Training child returned a mismatched result.")
                        run.update(
                            status="completed",
                            result=result,
                            metrics=result.get("metrics"),
                            checkpointPath=result.get("bestCheckpointPath"),
                        )
                    else:
                        failure_path = Path(run["outputPath"]) / "failure.json"
                        failure = read_json(failure_path) if failure_path.exists() else {}
                        if failure.get("at", "") < run.get("startedAt", ""):
                            failure = {}  # A prior attempt's failure is not this child's exit.
                        category = failure.get("category") or classify_training_failure(
                            failure.get("error", "")
                        )
                        run.update(
                            status="cancelled"
                            if cancelled
                            else "interrupted"
                            if halted_at is not None
                            else "failed",
                            error=failure.get("error")
                            or f"Training process exited with code {code}. See run.log.",
                            failureCategory=category,
                        )
                        append_event(
                            folder / "events.jsonl",
                            {
                                "at": now(),
                                "runId": run["id"],
                                "attempt": run.get("attempt"),
                                "exitCode": code,
                                "category": category,
                                "error": run["error"],
                            },
                        )
                        if category == "cuda_device_failure" and run.get("gpu") is not None:
                            halt_dispatch(f"Run {run['id']} reported a fatal CUDA/device error.")
                    run.update(finishedAt=now(), exitCode=code)
                    print(f"{now()} {run['id']} {run['status']} (exit {code}).", flush=True)
                    del running[run["id"]]
                    collect_results(batch, state, folder)
                if not cancelled and halted_at is None:
                    for run in state["runs"]:
                        if (
                            run["status"] != "queued"
                            or len(running) >= batch["resources"]["maxConcurrentRuns"]
                        ):
                            continue
                        with _leases() as (registry, active):
                            available, gpu = available_device(
                                batch["resources"], active, _capacity()
                            )
                            if not available:
                                state["waitingReason"] = (
                                    "Waiting for requested CPU, RAM, or GPU capacity."
                                )
                                break
                            state.pop("waitingReason", None)
                            run_folder = Path(run["outputPath"])
                            ensure_managed_directory(run_folder)
                            run_plan = _run_plan(batch, run, gpu)
                            write_json(run_folder / "plan.json", run_plan)
                            _reject_symlink_components(run_folder / "run.log")
                            stream = (run_folder / "run.log").open("ab", buffering=0)
                            environment = dict(os.environ)
                            environment.update(
                                OMP_NUM_THREADS=str(batch["resources"]["cpuThreadsPerRun"]),
                                MKL_NUM_THREADS=str(batch["resources"]["cpuThreadsPerRun"]),
                                OPENBLAS_NUM_THREADS=str(batch["resources"]["cpuThreadsPerRun"]),
                                CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu),
                                PYTHONUNBUFFERED="1",
                            )
                            started_at = now()
                            try:
                                process = subprocess.Popen(
                                    [
                                        batch["runtime"]["python"],
                                        "-u",
                                        str(Path(__file__).resolve()),
                                        "--fold",
                                        str(run_folder / "plan.json"),
                                    ],
                                    stdout=stream,
                                    stderr=subprocess.STDOUT,
                                    env=environment,
                                    start_new_session=True,
                                )
                            except BaseException:
                                stream.close()
                                raise
                            lease_path = registry / f"lease-{process.pid}.json"
                            # Track ownership immediately: PID inspection and durable lease
                            # publication can fail after the child already started.
                            running[run["id"]] = (process, stream, lease_path)
                            identity = process_identity(process.pid)
                            run.update(
                                status="running",
                                process=identity,
                                gpu=gpu,
                                startedAt=started_at,
                                logPath=str(run_folder / "run.log"),
                            )
                            write_json(
                                lease_path,
                                {
                                    "process": identity,
                                    "gpu": gpu,
                                    "cpus": cpu_slots_per_run(batch["resources"]),
                                    "ramGb": batch["resources"]["ramGbPerRun"],
                                    "runsPerGpu": batch["resources"]["runsPerGpu"],
                                    "batchId": batch["batchId"],
                                    "runId": run["id"],
                                },
                            )
                            print(
                                f"{now()} {run['id']} started, training seed {run['trainingSeed']}, device {'cpu' if gpu is None else f'cuda:{gpu}'}, log {run_folder / 'run.log'}",
                                flush=True,
                            )
                save_state(folder, state)
                if running or any(run["status"] == "queued" for run in state["runs"]):
                    time.sleep(0.5)
            state["status"] = (
                "cancelled"
                if cancelled_at
                else "interrupted"
                if halted_at is not None
                else "failed"
                if any(run["status"] == "failed" for run in state["runs"])
                else "completed"
            )
            state["finishedAt"] = now()
            collect_results(batch, state, folder)
        except BaseException as error:
            traceback.print_exc()
            state.update(
                status="failed",
                findings=[
                    {"severity": "error", "code": "TRAINING_WORKER_FAILED", "message": str(error)}
                ],
            )
            # Request shutdown for every owned process group and wait for its leader.
            # Descendant-aware reservation release requires separate reconciliation.
            for process, stream, lease_path in running.values():
                try:
                    if process.poll() is None:
                        try:
                            os.killpg(process.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass  # The child can exit between poll and signal.
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            try:
                                os.killpg(process.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                            process.wait(timeout=10)
                    with _leases():
                        lease_path.unlink(missing_ok=True)
                except Exception as cleanup_error:
                    # Keep a reservation if shutdown cannot be confirmed, and
                    # still attempt to stop every other child owned by this batch.
                    traceback.print_exc()
                    state["findings"].append(
                        {
                            "severity": "error",
                            "code": "TRAINING_CLEANUP_FAILED",
                            "message": f"Could not finish cleaning up worker {process.pid}: {cleanup_error}",
                        }
                    )
                finally:
                    stream.close()
            for run in state["runs"]:
                if run["status"] in {"queued", "running"}:
                    run.update(status="failed", error=str(error))
        finally:
            save_state(folder, state)
            print(f"{now()} Batch {state['status']}: {state['runCounts']}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--fold":
        run_fold_worker(Path(sys.argv[2]))
    elif len(sys.argv) == 2:
        run_batch(Path(sys.argv[1]))
    else:
        raise SystemExit("Usage: train_batch.py [--fold] PLAN.json")
