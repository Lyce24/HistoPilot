"""Launch frozen development plans; mutable execution never rewrites scientific inputs."""

import hashlib
import os
import signal
from pathlib import Path

from histopilot.adapters.native.runtime import training_runtime
from histopilot.application.development import development_plans
from histopilot.application.feature_bundles import FeatureBundleService, _hash
from histopilot.application.feature_packs import FeaturePackService
from histopilot.application.mil_inputs import MILInputService
from histopilot.application.protocols import FilterEvaluator, ProtocolService
from histopilot.application.task_records import LEGACY_CODE, LEGACY_MESSAGE, refuse_legacy
from histopilot.domain.features import representation_kind
from histopilot.models import catalog
from histopilot.schemas.development import DevelopmentBatchSpec
from histopilot.schemas.training_controls import (
    sampling_memberships,
    validate_selection_metric,
    validate_training_controls,
)
from histopilot.storage.lifecycle import LifecycleStore, lifecycle_guard
from histopilot.storage.project_lock import StorageError, ensure_managed_directory, writer_lock
from histopilot.taskcenter import ids as task_ids
from histopilot.taskcenter.model import ACTIVE as TASK_ACTIVE
from histopilot.taskcenter.model import LIVE as TASK_LIVE
from histopilot.workers.compute_archive import prepare_compute_archive
from histopilot.workers.packing_process import write_json
from histopilot.workers.training_process import (
    ACTIVE,
    append_event,
    compute_snapshot,
    cpu_slots_per_run,
    gpu_snapshot,
    host_snapshot,
    now,
    owned_processes,
    read_json,
    read_progress,
    save_state,
    stop_owned_processes,
)
from histopilot.workers.training_process import (
    confirmed_process_alive as process_alive,
)


def membership_plan_id(row: dict) -> str:
    metadata = {
        key: row[key]
        for key in ("planId", "seed", "fold", "phase", "outerFold", "innerFold", "repeat", "domain")
        if key in row
    }
    metadata.setdefault("planId", f"seed:{row.get('seed', 0)}/fold:{row.get('fold')}")
    return _hash(metadata)


def run_processes(run: dict) -> list:
    process = run.get("process")
    if process_alive(process):
        return [process]
    # Fold workers have always launched with start_new_session=True, including
    # saved runs that predate the explicit processGroupId metadata.
    return owned_processes(process, process["pid"] if process else None)


MANAGED = "task-center"
TASK_GROUP = "mil-batch"


def session_name(folder) -> str:
    return "tc-" + hashlib.sha256(str(folder).encode()).hexdigest()[:16]


def task_folder(folder) -> str:
    """Task Center ids name a batch by its absolute resolved folder."""
    return str(Path(folder).resolve())


class TrainingService:
    def __init__(self, store, filesystem, *, runtime=None, task_center=None):
        self.store, self.filesystem = store, filesystem
        self.runtime = runtime or training_runtime
        self._task_center = task_center
        self._default_task_center = task_center is None

    @property
    def task_center(self):
        if self._task_center is None:
            from histopilot.taskcenter.client import default_client

            self._task_center = default_client()
        return self._task_center

    def _task_group(self, identity, state):
        """The batch's Task Center group, or ``{"error": ...}`` when the store is unreadable."""
        project = (state.get("taskGroup") or {}).get("projectFolder") or str(self.store.folder)
        try:
            return self.task_center.group(TASK_GROUP, identity, project)
        except (StorageError, OSError) as error:
            return {"error": str(error)}

    def _task_summary(self, group):
        if group is None or "error" in group:
            return None
        counts = group["counts"]
        try:
            runner_alive = bool(self.task_center.runner_alive())
        except (StorageError, OSError):
            runner_alive = False
        return {
            "runnerAlive": runner_alive,
            "queued": counts["blocked"] + counts["queued"],
            "running": sum(counts[state] for state in TASK_ACTIVE),
            "held": group["held"],
            "waitingReason": group["waitingReason"],
            "ownerKey": group["owner"]["key"] if group["owner"] else None,
        }

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
        if state.get("executor") != MANAGED:
            return self._legacy_execution(folder, state, include_progress=include_progress)
        # The task store says whether work is still queued or running. Unknown task state
        # never reads as idle.
        group = self._task_group(identity, state)
        known = "error" not in group
        children = [run for run in state["runs"] if run_processes(run)]
        if children:
            # A terminal receipt can precede cleanup, or a fold's loader descendants can
            # survive it.
            if state["status"] not in ACTIVE:
                state["status"] = "running"
            if known and not group["live"]:
                state["findings"] = [
                    *[
                        item
                        for item in state.get("findings", [])
                        if item.get("code") != "ORPHAN_TRAINING_RUN"
                    ],
                    {
                        "severity": "warning",
                        "code": "ORPHAN_TRAINING_RUN",
                        "message": "Training workers are still running without their scheduler. Cancel this batch and wait for its workers to stop before resuming.",
                    },
                ]
        elif state["status"] in ACTIVE and self._managed_idle(group, folder):
            # Return a reconciled view. Only the Task Center writes active state.
            self._settle(state, "cancelled" if (folder / "cancel.json").exists() else "interrupted")
        state["taskCenter"] = self._task_summary(group)
        if not known:
            state["findings"] = [
                *state.get("findings", []),
                {
                    "severity": "warning",
                    "code": "TASK_CENTER_UNAVAILABLE",
                    "message": "The Task Center cannot be read, so this batch shows its last "
                    f"saved status. {group.get('error', '')}".strip(),
                },
            ]
        return self._with_progress(folder, state, include_progress)

    def _legacy_execution(self, folder, state, *, include_progress):
        """A batch launched before the Task Center: its saved state, never probed.

        Nothing can run it any more, so an unfinished one reads as interrupted.
        """
        if state["status"] in ACTIVE:
            self._settle(state, "interrupted")
            state["findings"] = [
                *state.get("findings", []),
                {"severity": "warning", "code": LEGACY_CODE, "message": LEGACY_MESSAGE},
            ]
        return self._with_progress(folder, state, include_progress)

    @staticmethod
    def _settle(state, status):
        from histopilot.workers.training_process import counts

        state["status"] = status
        for run in state["runs"]:
            if run["status"] in ACTIVE:
                run["status"] = status
        state["runCounts"] = counts(state["runs"])

    @staticmethod
    def _with_progress(folder, state, include_progress):
        state["cancelRequested"] = (folder / "cancel.json").exists()
        if include_progress:
            for run in state["runs"]:
                progress, warning = read_progress(folder / "runs" / run["id"] / "progress.json")
                if progress is not None or warning:
                    run["progress"] = progress
                if warning:
                    run["progressWarning"] = warning
        return state

    @staticmethod
    def _managed_idle(group, folder) -> bool:
        """No task can still move this batch: nothing is live, or a cancel left no fold."""
        if "error" in group:
            return False
        if not group["live"]:
            return True
        # A cancelled batch whose final collection waits (for example behind a hold)
        # reads as cancelled; that collection only rewrites results with the same status.
        return (folder / "cancel.json").exists() and not any(
            task["kind"] == "mil-fold" and task["state"] in TASK_LIVE for task in group["tasks"]
        )

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
        spec = DevelopmentBatchSpec.model_validate(manifest["spec"], context={"legacy": True})
        runtime = self.runtime()
        if not runtime["available"]:
            raise StorageError(runtime["findings"][0]["message"], "TRAINING_RUNTIME_UNAVAILABLE")
        # Design section 4.7: a frozen setup's legacy resources are read but ignored. The
        # device kind follows this machine and threads and loader workers the Task Center
        # defaults; the Task Center places and paces every run. The loader worker count
        # does not change augmentation random streams.
        resources = self._default_resources(runtime)
        host = host_snapshot()
        if cpu_slots_per_run(resources) > host["cpuCount"]:
            raise StorageError(
                "CPU threads plus overlapping training and validation data workers exceed the available CPU capacity per run.",
                "TRAINING_RESOURCES_UNAVAILABLE",
            )
        # The Task Center waits for free memory; refuse only a run that can never fit.
        if resources["ramGbPerRun"] > host["totalRamGb"]:
            raise StorageError(
                "Requested RAM per run exceeds this workstation's memory.",
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
                "MIL training supports binary and multiclass classification.",
                "TRAINING_TASK_UNSUPPORTED",
                422,
            )
        if any(
            not catalog.is_supported(item["recipe"]["model"])
            for item in manifest["configurations"]
        ):
            raise StorageError(
                f"Choose one of: {catalog.choices()}.", "TRAINING_MODEL_UNSUPPORTED", 422
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
        cohort_columns = {
            item["recipe"].get("cohortColumn", "cohort")
            for item in manifest["configurations"]
            if item["recipe"].get("samplingStrategy")
            in {"cohort_balanced", "cohort_label_balanced"}
        }
        cohort_values = {}
        if cohort_columns:
            _, _, dataset_rows = ProtocolService(self.store, self.filesystem)._load_dataset(
                protocol["datasetId"]
            )
            required_slides = {row["slideId"] for row in protocol["memberships"]}
            cohort_values = {
                column: {
                    row["slideId"]: FilterEvaluator.field(row, column)
                    for row in dataset_rows
                    if row["slideId"] in required_slides
                }
                for column in sorted(cohort_columns)
            }
        for rows in groups.values():
            if set(row["partition"] for row in rows) != {"train", "val", "test"}:
                raise StorageError(
                    "Every fold requires training, validation, and development assessment rows.",
                    "TRAINING_PARTITION_MISSING",
                )
            for item in manifest["configurations"]:
                try:
                    selected_rows = sampling_memberships(rows, item["recipe"], cohort_values)
                    validate_training_controls(
                        item["recipe"],
                        target,
                        selected_rows,
                        split_unit=protocol["spec"].get("splitUnit"),
                    )
                    if spec.candidateSelection == "best_validation":
                        validate_selection_metric(spec.selectionMetric, target, selected_rows)
                except ValueError as error:
                    code = (
                        "TRAINING_METRIC_UNAVAILABLE"
                        if "Validation AUROC" in str(error)
                        else "TRAINING_RECIPE_UNAVAILABLE"
                    )
                    raise StorageError(str(error), code, 422) from error
        feature = self.store.get_configuration(binding["featureSetId"])
        feature_kind = representation_kind(feature["manifest"])
        if any(
            item["recipe"].get("inputMode", "image") != "clinical"
            and catalog.feature_kind(item["recipe"].get("model")) != feature_kind
            for item in manifest["configurations"]
        ):
            raise StorageError(
                f"This bundle holds {feature_kind} features. Choose one of: "
                f"{catalog.choices(feature_kind)}.",
                "TRAINING_FEATURE_KIND_MISMATCH",
                422,
            )
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
        from histopilot.application.clinical_inputs import development_clinical_values

        clinical_values = development_clinical_values(
            self.store, self.filesystem, protocol,
            [item["recipe"] for item in manifest["configurations"]],
        )
        data = {
            "featureFiles": files,
            "loadingPolicy": binding["resolvedLoadingPolicy"],
            "packPath": pack_path,
            "packStamps": pack_stamps,
            "sourceStamps": source_stamps,
            "featureDim": next(iter(dimensions)),
            **({"cohortValues": cohort_values} if cohort_values else {}),
            **({"clinicalValues": clinical_values} if clinical_values else {}),
        }
        from histopilot.schemas.nnmil import resolve_nnmil_recipe

        resolutions = []
        try:
            for candidate in manifest["configurations"]:
                for plan_id, rows in groups.items():
                    _, resolution = resolve_nnmil_recipe(candidate["recipe"], rows, files)
                    if resolution:
                        resolutions.append({"candidateId": candidate["id"],
                                            "splitPlanId": plan_id, **resolution})
        except ValueError as error:
            raise StorageError(str(error), "MIL_BAG_PLANNING_INVALID", 422) from error
        if resolutions != manifest.get("nnmilPlanning", []):
            raise StorageError("Fitting features differ from the frozen MIL bag preview.",
                               "TRAINING_INPUTS_STALE", 409)
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
            **({"splitUnit": protocol["spec"]["splitUnit"]} if "splitUnit" in protocol["spec"] else {}),
            **({"groupByPatient": True} if protocol["spec"]["split"].get("groupByPatient") else {}),
            **({"selectionMetric": spec.selectionMetric} if spec.selectionMetric else {}),
            **({"candidateSelection": spec.candidateSelection} if spec.candidateSelection else {}),
            "resources": resources,
            "configurations": manifest["configurations"],
            "splitPlans": manifest["splitPlans"],
            "runs": manifest["runs"],
            "memberships": groups,
            "data": data,
            **({"nnmilPlanning": resolutions} if resolutions else {}),
        }
        return plan, bundles._freshness_guard(feature, bundle["manifest"])

    def _default_resources(self, runtime):
        """Per-run settings for specs without resources; the Task Center sets parallelism."""
        defaults = {"cpuThreadsPerRun": 2, "dataLoaderWorkers": 2}
        try:
            defaults.update(self.task_center.store.settings()["defaults"])
        except (StorageError, OSError, AttributeError, KeyError, TypeError):
            pass  # an unavailable store keeps the documented defaults
        return {
            "maxConcurrentRuns": 1,
            "gpuIds": [0]
            if runtime.get("cudaAvailable") and runtime.get("gpuCount", 0) > 0
            else [],
            "runsPerGpu": 1,
            "cpuThreadsPerRun": int(defaults["cpuThreadsPerRun"]),
            "dataLoaderWorkers": int(defaults["dataLoaderWorkers"]),
            "ramGbPerRun": 8.0,
        }

    # -- Task Center ---------------------------------------------------------------------

    def _task_owner(self, identity, batch):
        spec = batch["manifest"]["spec"]
        batch_name = spec.get("batchName") or identity
        experiment = spec.get("experimentId")
        common = {
            "projectId": self.store.project_id,
            "projectFolder": str(self.store.folder),
            "workspace": None,
            "labels": {
                **({"experimentId": experiment} if experiment else {}),
                "batchName": batch_name,
            },
        }
        if experiment:
            title = (batch["manifest"].get("experiment") or {}).get("name") or spec.get(
                "experimentName"
            )
            return {"kind": "experiment", "id": experiment, "title": title or experiment, **common}
        return {"kind": TASK_GROUP, "id": identity, "title": batch_name, **common}

    def _task_graph(self, identity, batch, folder, plan):
        """One task per fold in plan order, then the final results collection."""
        from histopilot.taskcenter import estimator

        spec = batch["manifest"]["spec"]
        batch_name = spec.get("batchName") or identity
        experiment = spec.get("experimentId")
        resources = plan["resources"]
        threads = int(resources["cpuThreadsPerRun"])
        workers = int(resources["dataLoaderWorkers"])
        lane = "gpu" if resources.get("gpuIds") else "cpu"
        key_folder = task_folder(folder)
        python = plan["runtime"]["python"]
        plan_path = str(folder / "plan.json")
        compute = str(folder / "compute")
        environment = {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": str(threads),
            "MKL_NUM_THREADS": str(threads),
            "OPENBLAS_NUM_THREADS": str(threads),
        }
        data = plan["data"]
        configurations = {row["id"]: row for row in plan["configurations"]}
        splits = {row["id"]: row for row in plan["splitPlans"]}
        group = {"kind": TASK_GROUP, "id": identity}
        workloads, tasks = {}, []
        for index, run in enumerate(plan["runs"]):
            candidate = configurations[run["candidateId"]]
            split = splits[run["splitPlanId"]]
            pair = (run["candidateId"], run["splitPlanId"])
            if pair not in workloads:
                try:
                    workloads[pair] = estimator.fold_workloads(
                        candidate["recipe"],
                        plan["memberships"][run["splitPlanId"]],
                        data["featureFiles"],
                        data["loadingPolicy"],
                    )
                except (ValueError, KeyError, TypeError):
                    workloads[pair] = None  # unmeasurable folds keep a conservative default
            workload = workloads[pair]
            vram = 0.0
            # An enqueue-time guess: the runner replaces RAM and VRAM with the estimator's
            # evidence-based request before admitting the fold.
            if lane == "gpu":
                try:
                    vram = round(estimator.gpu_estimate(workload), 3) if workload else 3.0
                except (ValueError, KeyError, TypeError, ZeroDivisionError):
                    vram = 3.0
            fold = split.get("fold")
            fold_number = fold + 1 if isinstance(fold, int) else fold
            run_folder = folder / "runs" / run["id"]
            tasks.append(
                {
                    "id": task_ids.fold_task_id(key_folder, run["id"]),
                    "kind": "mil-fold",
                    "adapter": "mil-fold",
                    "title": f"{batch_name} · Config {candidate.get('number', '?')} · "
                    f"Fold {fold_number} · Train seed {run['trainingSeed']} · "
                    f"Split seed {split.get('seed')}",
                    "group": group,
                    "planOrder": index,
                    "labels": {
                        "batchId": identity,
                        "batchName": batch_name,
                        "runId": run["id"],
                        "candidateId": run["candidateId"],
                        "configurationNumber": candidate.get("number"),
                        "fold": fold_number,
                        "trainingSeed": run["trainingSeed"],
                        "splitSeed": split.get("seed"),
                        "model": candidate["recipe"].get("model", "abmil"),
                        **({"experimentId": experiment} if experiment else {}),
                    },
                    "request": {
                        "lane": lane,
                        "cpuThreads": threads,
                        "dataWorkers": workers,
                        "ramGb": 6.0,
                        "vramGb": vram,
                        "sharedFiles": [data["packPath"]] if data.get("packPath") else [],
                        "workloadKey": workload["key"] if workload else None,
                        "workload": workload,
                    },
                    "command": {
                        "argv": [
                            python,
                            "-u",
                            "-m",
                            "histopilot.workers.managed_fold",
                            plan_path,
                            run["id"],
                        ],
                        "cwd": compute,
                        "env": environment,
                        "log": str(run_folder / "run.log"),
                        "progress": str(run_folder / "progress.json"),
                        "result": str(run_folder / "result.json"),
                    },
                    "adapterData": {
                        "batchFolder": key_folder,
                        "runId": run["id"],
                        "batchId": identity,
                        "final": False,
                    },
                }
            )
        tasks.append(
            {
                "id": task_ids.collect_task_id(key_folder, True),
                "kind": "mil-collect",
                "adapter": "mil-collect",
                "title": f"{batch_name} · Final results",
                "group": group,
                # Partial collections take the slot just before it (see the fold adapter).
                "planOrder": len(plan["runs"]) + 1,
                "priority": "interactive",
                "exclusiveKey": "collect:" + key_folder,
                "labels": {
                    "batchId": identity,
                    "batchName": batch_name,
                    **({"experimentId": experiment} if experiment else {}),
                },
                "dependsOn": [{"task": task["id"], "condition": "terminal"} for task in tasks],
                "request": {"lane": "cpu", "cpuThreads": 1, "dataWorkers": 0, "ramGb": 3.0},
                "command": {
                    "argv": [
                        python,
                        "-u",
                        "-m",
                        "histopilot.workers.managed_collect",
                        plan_path,
                        "--final",
                    ],
                    "cwd": compute,
                    "env": {
                        **environment,
                        "OMP_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1",
                    },
                    "log": str(folder / "batch.log"),
                    "progress": None,
                    "result": str(folder / "collect-result.json"),
                },
                "adapterData": {"batchFolder": key_folder, "batchId": identity, "final": True},
            }
        )
        return tasks

    def _submit_tasks(self, identity, batch, folder, plan, state, *, resume):
        """Enqueue the batch's task graph, or requeue its unfinished tasks on resume."""
        client = self.task_center
        project = state["taskGroup"]["projectFolder"]
        key_folder = task_folder(folder)
        if resume:
            client.requeue_group(
                TASK_GROUP, identity, project, kinds=("mil-fold", "mil-collect"), reason="resume"
            )
            pending = [
                task_ids.fold_task_id(key_folder, run["id"])
                for run in state["runs"]
                if run["status"] != "completed"
            ]
            # A fold recorded as succeeded while its run is not completed must run again.
            client.store.requeue(pending, reason="resume", include_succeeded=True)
            final = task_ids.collect_task_id(key_folder, True)
            client.store.requeue([final], reason="resume", include_succeeded=True)
            # A final collection still waiting from the previous attempt (for example
            # behind a hold) must wait for the requeued folds again.
            if any(client.store.get(task) is not None for task in pending):
                client.store.transition(
                    final,
                    from_states=("queued",),
                    to_state="blocked",
                    waiting_reason=None,
                    detail={"reason": "resume"},
                )
            present = {task["id"] for task in client.group(TASK_GROUP, identity, project)["tasks"]}
            expected = {task_ids.fold_task_id(key_folder, run["id"]) for run in plan["runs"]}
            if expected <= present:
                return
        client.enqueue(
            self._task_owner(identity, batch), self._task_graph(identity, batch, folder, plan)
        )

    def _task_accepted(self, identity, folder, state, *, resume) -> bool:
        try:
            group = self.task_center.group(
                TASK_GROUP, identity, state["taskGroup"]["projectFolder"]
            )
        except (StorageError, OSError):
            return False
        if group["live"]:
            return True
        # Tasks may already have finished and recorded their outcome in state.json.
        return bool(group["tasks"]) and not resume

    def _require_idle_tasks(self, identity):
        """A managed (re)launch never overlaps a running task or a queued fold of the batch."""
        group = self.task_center.group(TASK_GROUP, identity, str(self.store.folder))
        if any(
            task["state"] in TASK_ACTIVE
            or (task["kind"] == "mil-fold" and task["state"] in TASK_LIVE)
            for task in group["tasks"]
        ):
            raise StorageError(
                "Tasks of this batch are still queued or running in the Task Center.",
                "TRAINING_ACTIVE",
            )

    def _wake_runner(self):
        """The first submission starts the runner; failures only delay queued work."""
        if not self._default_task_center:
            return
        try:
            from histopilot.taskcenter.launcher import ensure_runner

            ensure_runner()
        except Exception:  # never fail an accepted launch because tmux misbehaved
            pass

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
            if previous and previous.get("executor") != MANAGED:
                refuse_legacy()
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
            # A failed enqueue has durable plan/state but no accepted launch receipt.
            # Retrying the same launch safely resumes that exact plan.
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
            original_plan = read_json(folder / "plan.json") if resume else None
            session = session_name(folder)
            if resume:
                if original_plan.get("executionMode") != MANAGED:
                    refuse_legacy()
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
                plan.update(executionMode=MANAGED, outputPath=str(folder), sessionName=session)
            submission = (experiment_record or {}).get("payload", {}).get("submission")
            pinned = None
            if submission:
                from histopilot.application.model_experiments import (
                    execution_contract,
                    pinned_compute,
                    require_execution_contract,
                )

                if not resume:
                    # A batch launched after its siblings, as when a submission is
                    # retried, copies their archived code rather than the live checkout.
                    pinned = pinned_compute(self.store, submission)
                    if pinned:
                        plan["code"] = pinned[0]
                # Resumes retain their archived original worker code while the
                # currently installed interpreter still has to match the common
                # experiment contract. Live batches never pass through here.
                contract = execution_contract({**plan, "runtime": prepared_runtime})
                require_execution_contract(submission.get("executionContract"), contract)
            package_root = prepare_compute_archive(
                folder, plan.get("code", {}), source_root=pinned[1] if pinned else None
            )
            if not (package_root / "histopilot" / "workers" / "managed_fold.py").is_file():
                # Code pinned before the Task Center has no fold entry point to run.
                refuse_legacy()
            freshness()
            self._require_idle_tasks(identity)
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
                "computePath": str(package_root),
                "computeVersion": plan["code"]["sha256"],
                # Parallelism belongs to the Task Center, so there is no per-batch plan.
                "executor": MANAGED,
                "taskGroup": {
                    "kind": TASK_GROUP,
                    "id": identity,
                    "projectFolder": str(self.store.folder),
                },
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
                self._submit_tasks(identity, batch, folder, plan, state, resume=resume)
            except (OSError, ValueError, KeyError, TypeError) as error:
                # A committed enqueue owns this batch; never overwrite worker evidence.
                if not self._task_accepted(identity, folder, state, resume=resume):
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
                        f"Training tasks could not be queued: {error}",
                        "TRAINING_LAUNCH_FAILED",
                    ) from error
                operations[operation_id] = action
                write_json(path, operations)
                return self.execution(identity)
            operations[operation_id] = action
            write_json(path, operations)
            self._wake_runner()
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
            if state and not replay:
                if state.get("executor") != MANAGED:
                    refuse_legacy()
                self._cancel_managed(identity, folder, state, operation_id)
            operations[operation_id] = "cancel"
            write_json(path, operations)
        return self.execution(identity, include_inactive=True)

    def _cancel_managed(self, identity, folder, state, operation_id):
        """Cancel pending folds at once; the runner stops running ones after cancel.json."""
        project = (state.get("taskGroup") or {}).get("projectFolder") or str(self.store.folder)
        try:
            group = self.task_center.group(TASK_GROUP, identity, project)
        except (StorageError, OSError):
            group = None
        live_runs = [run for run in state["runs"] if run_processes(run)]
        if not (state["status"] in ACTIVE or live_runs or (group and group["live"])):
            return
        # The marker precedes every stop request: stopped folds are then classified as
        # cancelled and queued folds are never started.
        write_json(folder / "cancel.json", {"requestedAt": now(), "operationId": operation_id})
        cancelled = set()
        if group is not None:
            try:
                # A second pass catches folds the runner started during the first one.
                for _ in range(2):
                    result = self.task_center.cancel_group(
                        TASK_GROUP, identity, project, exclude_kinds=("mil-collect",)
                    )
                    cancelled.update(result["cancelled"])
                group = self.task_center.group(TASK_GROUP, identity, project)
            except (StorageError, OSError):
                group = None
        key_folder = task_folder(folder)
        if cancelled:
            self._record_cancelled_before_start(folder, cancelled)
        running = (
            {task["id"] for task in group["tasks"] if task["state"] in TASK_ACTIVE}
            if group is not None
            else set()
        )
        # Workers no task tracks (or that an unreadable store cannot stop) are signalled here.
        self._stop_orphans(
            [
                run["process"]
                for run in live_runs
                if task_ids.fold_task_id(key_folder, run["id"]) not in running
            ]
        )
        self._wake_runner()

    @staticmethod
    def _record_cancelled_before_start(folder, tasks) -> None:
        """Queued runs whose fold tasks (by id) will not start are recorded as cancelled."""
        key_folder = task_folder(folder)
        state = read_json(folder / "state.json")
        at, changed = now(), False
        for run in state["runs"]:
            if run["status"] == "queued" and task_ids.fold_task_id(key_folder, run["id"]) in tasks:
                run.update(status="cancelled", finishedAt=at, error="Cancelled before start.")
                changed = True
        if changed:
            save_state(folder, state)

    def cancel_runs(self, identity, run_ids, operation_id):
        """Cancel single runs of a Task Center batch while its other runs continue.

        No batch cancel marker is written: the batch finishes as cancelled and a resume
        runs the cancelled runs again.
        """
        with lifecycle_guard(self.store.folder, timeout=5):
            return self._cancel_runs(identity, list(dict.fromkeys(run_ids)), operation_id)

    def _cancel_runs(self, identity, run_ids, operation_id):
        folder = self._folder(identity, include_inactive=True)
        if not (folder / "state.json").exists():
            raise StorageError("This batch has not been launched.", "TRAINING_NOT_LAUNCHED")
        if not run_ids:
            raise StorageError("Choose at least one run to cancel.", "TRAINING_RUNS_REQUIRED", 422)
        # The receipt names the runs, so a reused operation ID for other runs conflicts.
        digest = hashlib.sha256("\0".join(sorted(run_ids)).encode()).hexdigest()[:16]
        action = f"cancel-runs:{digest}"
        with writer_lock(folder, timeout=5):
            state = read_json(folder / "state.json")
            if state.get("executor") != MANAGED:
                refuse_legacy()
            path, operations, replay = self._operation(folder, operation_id, action)
            if not replay:
                known = {run["id"] for run in state["runs"]}
                unknown = [run_id for run_id in run_ids if run_id not in known]
                if unknown:
                    raise StorageError(
                        f"Run {unknown[0]} is not part of this batch.",
                        "TRAINING_RUN_NOT_FOUND",
                        404,
                    )
                self._cancel_managed_runs(folder, run_ids)
                operations[operation_id] = action
                write_json(path, operations)
        return self.execution(identity, include_inactive=True)

    def _cancel_managed_runs(self, folder, run_ids):
        """Pending folds are cancelled at once; the runner stops running ones."""
        key_folder = task_folder(folder)
        tasks = [task_ids.fold_task_id(key_folder, run_id) for run_id in run_ids]
        store = self.task_center.store
        try:
            cancelled = set(store.cancel_pending(tasks))
            # Both steps are state-guarded: a fold started in between is stopped instead.
            store.request_stop([task for task in tasks if task not in cancelled], "cancel")
            current = {task: store.get(task) for task in tasks}
        except OSError as error:
            raise StorageError(
                f"The Task Center store is unavailable: {error}", "TASK_CENTER_UNAVAILABLE", 503
            ) from error
        # A fold with no active task never starts again: its requeue (after a pause, for
        # example) needs this lock and then finds the run cancelled. A running fold is
        # recorded when it stops.
        idle = cancelled | {
            task for task, row in current.items() if row is None or row["state"] not in TASK_ACTIVE
        }
        self._record_cancelled_before_start(folder, idle)
        self._wake_runner()

    @staticmethod
    def _stop_orphans(processes):
        errors = []
        # Signal all groups before waiting: a stubborn first group
        # must not prevent cancellation from reaching later folds.
        for process in processes:
            try:
                os.killpg(process["pid"], signal.SIGTERM)
            except ProcessLookupError:
                pass
            except OSError as error:
                errors.append(error)
        for process in processes:
            try:
                stop_owned_processes(process)
            except (OSError, ValueError) as error:
                errors.append(error)
        if errors:
            raise StorageError(
                "Cancellation was requested for every orphan worker, but some workers could not be confirmed stopped. Their reservations are retained.",
                "TRAINING_CLEANUP_FAILED",
            ) from errors[0]
