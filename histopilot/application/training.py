"""Launch frozen development plans; mutable execution never rewrites scientific inputs."""

import hashlib
import os
import signal
import subprocess

from histopilot.adapters.native.runtime import training_runtime
from histopilot.application.development import development_plans
from histopilot.application.feature_bundles import FeatureBundleService, _hash
from histopilot.application.feature_packs import FeaturePackService
from histopilot.application.mil_inputs import MILInputService
from histopilot.schemas.development import DevelopmentBatchSpec
from histopilot.storage.lifecycle import LifecycleStore, lifecycle_guard
from histopilot.storage.project_lock import StorageError, ensure_managed_directory, writer_lock
from histopilot.workers.compute_archive import prepare_compute_archive
from histopilot.workers.packing_process import write_json
from histopilot.workers.training_process import (
    ACTIVE,
    TmuxTrainingExecutor,
    append_event,
    compute_snapshot,
    cpu_slots_per_run,
    gpu_snapshot,
    host_snapshot,
    now,
    process_alive,
    read_json,
    read_progress,
    resource_plan,
    save_state,
)


def membership_plan_id(row: dict) -> str:
    metadata = {
        key: row[key]
        for key in ("planId", "seed", "fold", "phase", "outerFold", "innerFold", "repeat", "domain")
        if key in row
    }
    metadata.setdefault("planId", f"seed:{row.get('seed', 0)}/fold:{row.get('fold')}")
    return _hash(metadata)


class TrainingService:
    def __init__(self, store, filesystem, executor=None, runtime=None):
        self.store, self.filesystem = store, filesystem
        self.executor = executor or TmuxTrainingExecutor()
        self.runtime = runtime or training_runtime

    def _batch(self, identity, *, include_inactive=False):
        configuration = self.store.get_configuration(identity, include_inactive=include_inactive)
        if configuration["manifest"].get("kind") != "mil-batch":
            raise StorageError("Choose a frozen development batch.", "INVALID_BATCH", 422)
        return configuration

    def _folder(self, identity, *, include_inactive=False):
        # IDs are resolved by the scientific store before path construction.
        self._batch(identity, include_inactive=include_inactive)
        return self.store.folder / "training" / identity

    def execution(self, identity, *, include_inactive=False, include_progress=True):
        folder = (
            self._folder(identity, include_inactive=True)
            if include_inactive
            else self._folder(identity)
        )
        if not (folder / "state.json").exists():
            return None
        state = read_json(folder / "state.json")
        if state["status"] in ACTIVE and not self.executor.running(state["sessionName"]):
            children = [run for run in state["runs"] if process_alive(run.get("process"))]
            if not process_alive(state.get("process")) and not children:
                # Return a reconciled view. Only the scheduler writes active state.
                state["status"] = (
                    "cancelled" if (folder / "cancel.json").exists() else "interrupted"
                )
                for run in state["runs"]:
                    if run["status"] in ACTIVE:
                        run["status"] = state["status"]
                from histopilot.workers.training_process import counts

                state["runCounts"] = counts(state["runs"])
            elif not process_alive(state.get("process")):
                state["findings"] = [
                    {
                        "severity": "warning",
                        "code": "ORPHAN_TRAINING_RUN",
                        "message": "A training child is still running. Wait for it to finish before resuming this batch.",
                    }
                ]
        state["cancelRequested"] = (folder / "cancel.json").exists()
        if include_progress:
            for run in state["runs"]:
                progress, warning = read_progress(folder / "runs" / run["id"] / "progress.json")
                if progress is not None or warning:
                    run["progress"] = progress
                if warning:
                    run["progressWarning"] = warning
        return state

    def list(self, *, include_inactive=False):
        lifecycle = LifecycleStore(self.store.folder, self.store.project_id).read()["records"]
        items = []
        for batch in self.store.list_configurations("mil-batch", include_inactive=True):
            state = self.execution(batch["id"], include_inactive=True)
            if state is not None and (
                include_inactive
                or state["status"] in ACTIVE
                or lifecycle.get(f"configuration:{batch['id']}", {}).get("state", "active")
                == "active"
            ):
                items.append(state)
        return {"items": items}

    def results(self, identity):
        state = self.execution(identity)
        path = self._folder(identity) / "results.json"
        if path.exists():
            return {**read_json(path), "status": state["status"] if state else "planned"}
        return {"status": state["status"] if state else "planned", "candidates": [], "oof": []}

    def _prepare(self, batch):
        manifest = batch["manifest"]
        spec = DevelopmentBatchSpec.model_validate(manifest["spec"])
        runtime = self.runtime()
        if not runtime["available"]:
            raise StorageError(runtime["findings"][0]["message"], "TRAINING_RUNTIME_UNAVAILABLE")
        resources = spec.resources.model_dump()
        if resources["gpuIds"] and (
            not runtime["cudaAvailable"] or max(resources["gpuIds"]) >= runtime["gpuCount"]
        ):
            raise StorageError(
                "The requested GPU is unavailable. Choose available GPU IDs or CPU execution (an empty GPU list).",
                "TRAINING_GPU_UNAVAILABLE",
            )
        host = host_snapshot()
        if cpu_slots_per_run(resources) > host["cpuCount"]:
            raise StorageError(
                "CPU threads plus overlapping training and validation data workers exceed the available CPU capacity per run.",
                "TRAINING_RESOURCES_UNAVAILABLE",
            )
        if resources["ramGbPerRun"] > host["availableRamGb"]:
            raise StorageError(
                "Requested RAM per run exceeds currently available memory.",
                "TRAINING_RESOURCES_UNAVAILABLE",
            )
        binding = MILInputService(self.store, self.filesystem).preview(spec.inputs)
        if not binding["canPlan"]:
            raise StorageError(
                "Training inputs are no longer current: "
                + "; ".join(item["message"] for item in binding["findings"]),
                "TRAINING_INPUTS_STALE",
            )
        original_binding = manifest["resolvedInputs"]
        if any(
            binding[key] != original_binding.get(key)
            for key in ("resolvedLoadingPolicy", "packArtifactId", "featureSetId", "bundleId")
        ):
            raise StorageError(
                "The loading policy differs from the frozen batch.", "TRAINING_INPUTS_STALE"
            )
        protocol = self.store.get_configuration(spec.inputs.protocolId)["manifest"]
        if (
            protocol["spec"]["split"].get("version") != 4
            or protocol["spec"]["split"]["mode"] != "kfold"
        ):
            raise StorageError(
                "Training currently supports Stage 2 development-only k-fold protocols.",
                "TRAINING_SPLIT_UNSUPPORTED",
                422,
            )
        target = protocol["spec"]["target"]
        if target["task"] not in {"binary_classification", "multiclass_classification"}:
            raise StorageError(
                "ABMIL currently supports binary and multiclass classification.",
                "TRAINING_TASK_UNSUPPORTED",
                422,
            )
        if any(item["recipe"]["model"].lower() != "abmil" for item in manifest["configurations"]):
            raise StorageError(
                "Only ABMIL is connected to the training worker.", "TRAINING_MODEL_UNSUPPORTED", 422
            )
        if development_plans(protocol) != manifest["splitPlans"]:
            raise StorageError(
                "Frozen split plans do not match this batch.", "TRAINING_SPLITS_CHANGED"
            )
        groups = {item["id"]: [] for item in manifest["splitPlans"]}
        for row in protocol["memberships"]:
            key = membership_plan_id(row)
            if (
                key not in groups
                or row.get("phase") == "final"
                or row.get("pool") == "external_test"
            ):
                raise StorageError(
                    "Unexpected final-test or split membership.", "TRAINING_SPLITS_CHANGED"
                )
            groups[key].append(row)
        for rows in groups.values():
            if set(row["partition"] for row in rows) != {"train", "val", "test"}:
                raise StorageError(
                    "Every fold requires training, validation, and development assessment rows.",
                    "TRAINING_PARTITION_MISSING",
                )
            validation_classes = {row["label"] for row in rows if row["partition"] == "val"}
            if any(
                item["recipe"]["checkpointMetric"] == "validation_auroc"
                for item in manifest["configurations"]
            ) and validation_classes != set(target["classes"]):
                raise StorageError(
                    "Validation AUROC requires every target class in each validation fold. Choose validation loss or revise the protocol.",
                    "TRAINING_METRIC_UNAVAILABLE",
                )
        feature = self.store.get_configuration(binding["featureSetId"])
        bundles = FeatureBundleService(self.store, self.filesystem)
        bundle = bundles.get(spec.inputs.featureBundleId)
        files = {row["slideId"]: row for row in feature["manifest"]["files"]}
        required = {row["slideId"] for row in protocol["memberships"]}
        files = {key: value for key, value in files.items() if key in required}
        dimensions = {row["dimensions"] for row in files.values()}
        if len(dimensions) != 1 or set(files) != required:
            raise StorageError(
                "All development slides need one compatible feature dimension.",
                "TRAINING_FEATURES_INVALID",
            )
        pack_path, pack_stamps = None, None
        if binding["packArtifactId"]:
            packing = FeaturePackService(self.store, self.filesystem)
            packing.resolve_artifact(binding["featureSetId"], binding["packArtifactId"])
            pack = packing.artifact(binding["packArtifactId"])
            pack_path, pack_stamps = pack["outputPath"], pack["packStamps"]
        source_stamps = {row["path"]: row for row in files.values()}
        data = {
            "featureFiles": files,
            "loadingPolicy": binding["resolvedLoadingPolicy"],
            "packPath": pack_path,
            "packStamps": pack_stamps,
            "sourceStamps": source_stamps,
            "featureDim": next(iter(dimensions)),
        }
        plan = {
            "version": 1,
            "batchId": batch["id"],
            "batchContentHash": batch["contentHash"],
            "protocolId": spec.inputs.protocolId,
            "featureBundleId": spec.inputs.featureBundleId,
            "runtime": runtime,
            "code": compute_snapshot(),
            "protocolContentHash": self.store.get_configuration(spec.inputs.protocolId)[
                "contentHash"
            ],
            "featureBundleContentHash": bundle["contentHash"],
            "target": target,
            "resources": resources,
            "configurations": manifest["configurations"],
            "splitPlans": manifest["splitPlans"],
            "runs": manifest["runs"],
            "memberships": groups,
            "data": data,
        }
        return plan, bundles._freshness_guard(feature, bundle["manifest"])

    @staticmethod
    def _operation(folder, operation_id, action):
        path = folder / "operations.json"
        operations = read_json(path) if path.exists() else {}
        if operation_id in operations and operations[operation_id] != action:
            raise StorageError(
                "This operation ID was already used for another training action.",
                "OPERATION_CONFLICT",
            )
        return path, operations, operation_id in operations

    def launch(self, identity, operation_id, *, resume=False):
        # Serialize lifecycle changes across preview, input checks and worker launch.
        # This guard is distinct from the batch lock and is reentrant for store reads.
        with lifecycle_guard(self.store.folder, timeout=5):
            lifecycle = LifecycleStore(self.store.folder, self.store.project_id)
            lifecycle.assert_usable(
                [f"project:{self.store.project_id}", f"configuration:{identity}"]
            )
            lifecycle.assert_document_usable(self._batch(identity))
            return self._launch(identity, operation_id, resume=resume)

    def _launch(self, identity, operation_id, *, resume=False):
        batch = self._batch(identity)
        folder = self._folder(identity)
        ensure_managed_directory(folder)
        with writer_lock(folder, timeout=5):
            action = "resume" if resume else "launch"
            path, operations, replay = self._operation(folder, operation_id, action)
            previous = self.execution(identity)
            if replay:
                return previous
            owner = batch["manifest"]["spec"].get("experimentId")
            experiment_record = None
            if owner:
                from histopilot.application.model_experiments import ModelExperimentService

                experiment_record = ModelExperimentService(
                    self.store, self.filesystem, training=self
                ).require_training_action(owner, identity, resume=resume)
            if any(
                item["manifest"].get("batchId") == identity
                for item in self.store.list_configurations(
                    "frozen-predictor", include_inactive=True
                )
            ):
                raise StorageError(
                    "This batch supplies a frozen predictor. Create a new experiment and batch to continue training; its pinned checkpoints must remain unchanged.",
                    "BATCH_HAS_FROZEN_PREDICTOR",
                    409,
                )
            if previous and previous["status"] in ACTIVE:
                raise StorageError(
                    "This batch already has active training workers.", "TRAINING_ACTIVE"
                )
            # A failed tmux start has durable plan/state but no accepted launch
            # receipt. Retrying the same launch safely resumes that exact plan.
            if (
                previous
                and not resume
                and previous["status"] == "failed"
                and any(
                    item.get("code") == "TRAINING_LAUNCH_FAILED"
                    for item in previous.get("findings", [])
                )
            ):
                resume = True
            if previous and not resume:
                raise StorageError(
                    "This batch was already launched. Resume its unfinished runs or clone a new batch.",
                    "TRAINING_ALREADY_LAUNCHED",
                )
            if resume and (not previous or previous["status"] == "completed"):
                raise StorageError(
                    "Only interrupted, failed, or cancelled batches can be resumed.",
                    "TRAINING_NOT_RESUMABLE",
                )
            plan, freshness = self._prepare(batch)
            prepared_runtime = plan["runtime"]
            provenance = {"at": now(), "host": host_snapshot(), **gpu_snapshot()}
            session = "hp-train-" + hashlib.sha256(str(folder).encode()).hexdigest()[:16]
            if resume:
                original_plan = read_json(folder / "plan.json")
                if _hash(original_plan) != previous.get("planHash"):
                    raise StorageError(
                        "The saved execution plan changed after launch; it cannot be resumed.",
                        "TRAINING_PLAN_CHANGED",
                    )
                if (
                    original_plan.get("outputPath") != str(folder)
                    or original_plan.get("sessionName") != session
                ):
                    raise StorageError(
                        "The saved execution belongs to a different output location or training session. Restore its original location or clone a new batch; existing checkpoints and results contain fixed paths.",
                        "TRAINING_LOCATION_CHANGED",
                    )
                if original_plan["runtime"]["versions"] != plan["runtime"]["versions"]:
                    raise StorageError(
                        "Training code or dependency versions changed. Clone a new batch to keep its results separate from the original execution.",
                        "TRAINING_RUNTIME_CHANGED",
                    )
                # Execution provenance may change after reboot or a driver repair;
                # retain the exact scientific execution plan and record the new attempt.
                plan = original_plan
            else:
                plan.update(outputPath=str(folder), sessionName=session)
            submission = (experiment_record or {}).get("payload", {}).get("submission")
            if submission:
                from histopilot.application.model_experiments import (
                    execution_contract,
                    require_execution_contract,
                )

                # Resumes retain their archived original worker code while the
                # currently installed interpreter still has to match the common
                # experiment contract. Live batches never pass through here.
                contract = execution_contract({**plan, "runtime": prepared_runtime})
                require_execution_contract(submission.get("executionContract"), contract)
            package_root = prepare_compute_archive(folder, plan.get("code", {}))
            freshness()
            if self.executor.running(session):
                raise StorageError("A worker for this batch is still running.", "TRAINING_ACTIVE")
            prior_runs = {row["id"]: row for row in previous["runs"]} if previous else {}
            runs = []
            for row in plan["runs"]:
                prior = prior_runs.get(row["id"])
                runs.append(
                    prior
                    if prior and prior["status"] == "completed"
                    else {
                        **row,
                        "status": "queued",
                        "outputPath": str(folder / "runs" / row["id"]),
                        "attempt": (prior.get("attempt", 0) if prior else 0) + 1,
                    }
                )
            state = {
                "batchId": identity,
                "status": "queued",
                "sessionName": session,
                "outputPath": str(folder),
                "logPath": str(folder / "batch.log"),
                "createdAt": previous["createdAt"] if previous else now(),
                "runs": runs,
                "findings": [],
                "cancelRequested": False,
                "runtime": plan["runtime"],
                "planHash": _hash(plan),
                "provenance": provenance,
                "provenancePath": str(folder / "attempts.jsonl"),
                "resourcePlan": resource_plan(plan["resources"], provenance["host"], len(runs)),
                "computePath": str(package_root),
                "computeVersion": plan["code"]["sha256"],
            }
            if previous and previous.get("telemetry"):
                state["telemetry"] = previous["telemetry"]
            if plan["code"] != compute_snapshot():
                state["findings"].append(
                    {
                        "severity": "warning",
                        "code": "TRAINING_PINNED_CODE",
                        "message": "This execution resumes its verified original worker code. Improvements added after its launch, including telemetry, apply only to new executions.",
                    }
                )
            if previous and previous.get("provenance"):
                prior_provenance = previous["provenance"]
                previous_drivers = {
                    (gpu["uuid"], gpu["driverVersion"]) for gpu in prior_provenance.get("gpus", [])
                }
                current_drivers = {
                    (gpu["uuid"], gpu["driverVersion"]) for gpu in provenance["gpus"]
                }
                if (
                    prior_provenance["host"]["bootId"] != provenance["host"]["bootId"]
                    or previous_drivers != current_drivers
                ):
                    state["findings"].append(
                        {
                            "severity": "warning",
                            "code": "TRAINING_HOST_CHANGED",
                            "message": "Host boot or GPU driver/device provenance differs from the previous attempt; both are retained in attempts.jsonl.",
                        }
                    )
            (folder / "cancel.json").unlink(missing_ok=True)
            if not resume:
                write_json(folder / "plan.json", plan)
            append_event(
                folder / "attempts.jsonl",
                {
                    "action": action,
                    "operationId": operation_id,
                    "planHash": state["planHash"],
                    "provenance": provenance,
                },
            )
            save_state(folder, state)
            try:
                self.executor.launch(
                    session,
                    plan["runtime"]["python"],
                    folder / "plan.json",
                    folder / "batch.log",
                    package_root=package_root,
                )
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                # tmux can create its session before a timeout loses the reply.
                # The worker owns active state; never overwrite evidence it wrote.
                try:
                    session_running = self.executor.running(session)
                    # Session inspection may block while the worker finishes.
                    # Read its evidence afterwards, not before the tmux probe.
                    current = read_json(folder / "state.json")
                    started = (
                        session_running
                        or current.get("process") is not None
                        or any(process_alive(run.get("process")) for run in current["runs"])
                        or current["status"] not in ACTIVE
                    )
                except (OSError, RuntimeError, subprocess.SubprocessError) as inspection_error:
                    raise StorageError(
                        "Launch acknowledgement was lost. Check execution status before retrying.",
                        "TRAINING_LAUNCH_UNCERTAIN",
                    ) from inspection_error
                if started:
                    operations[operation_id] = action
                    write_json(path, operations)
                    return self.execution(identity)
                state.update(
                    status="failed",
                    findings=[
                        {
                            "severity": "error",
                            "code": "TRAINING_LAUNCH_FAILED",
                            "message": str(error),
                        }
                    ],
                )
                save_state(folder, state)
                raise StorageError(
                    f"Training worker could not start: {error}", "TRAINING_LAUNCH_FAILED"
                ) from error
            operations[operation_id] = action
            write_json(path, operations)
            return state

    def cancel(self, identity, operation_id):
        with lifecycle_guard(self.store.folder, timeout=5):
            return self._cancel(identity, operation_id)

    def _cancel(self, identity, operation_id):
        # Cancellation remains reachable while reconciling hidden or orphaned work.
        folder = self._folder(identity, include_inactive=True)
        if not folder.exists():
            raise StorageError("This batch has not been launched.", "TRAINING_NOT_LAUNCHED")
        with writer_lock(folder, timeout=5):
            path, operations, replay = self._operation(folder, operation_id, "cancel")
            state = self.execution(identity, include_inactive=True)
            busy = state and (
                state["status"] in ACTIVE
                or process_alive(state.get("process"))
                or any(process_alive(run.get("process")) for run in state["runs"])
            )
            if not replay and busy:
                write_json(
                    folder / "cancel.json", {"requestedAt": now(), "operationId": operation_id}
                )
                if not process_alive(state.get("process")):
                    # A scheduler can disappear while its independently isolated children
                    # survive. Signal only the recorded process groups whose identity still
                    # matches; a cancel marker alone cannot reach an orphaned child.
                    for run in state["runs"]:
                        process = run.get("process")
                        if process_alive(process):
                            try:
                                os.killpg(process["pid"], signal.SIGTERM)
                            except ProcessLookupError:
                                pass
            operations[operation_id] = "cancel"
            write_json(path, operations)
        return self.execution(identity, include_inactive=True)
