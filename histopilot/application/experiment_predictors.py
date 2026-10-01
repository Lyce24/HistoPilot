"""Durable predictor creation belonging to one frozen experiment submission.

Only submit/resume starts the persistent coordinator. Status is observational.
Each source is a complete configuration/training-seed/split-seed fold group;
folds are never counted as separate predictors. Individual publications and
refit launches have stable operation IDs, including across coordinator crashes.
"""

from __future__ import annotations

import hashlib
import sys
from copy import deepcopy
from pathlib import Path

from histopilot.adapters.native.runtime import training_runtime
from histopilot.application.compute_jobs import (
    archive_protocol,
    host_gpu_argv,
    submit_task,
    task_pending,
    task_view,
)
from histopilot.application.experiment_policy import (
    has_predictor_intent,
    predictor_work_expected,
    submission_policies,
    verify_batch_policy,
)
from histopilot.application.predictor_builds import PredictorBuildService
from histopilot.application.predictors import PredictorService
from histopilot.application.refits import RefitService
from histopilot.application.task_records import (
    LEGACY_CODE,
    LEGACY_MESSAGE,
    TaskCenterAccess,
    refuse_legacy,
)
from histopilot.schemas.model_experiments import ExperimentPredictorPolicy
from histopilot.schemas.predictors import ApplyPredictorBuilds, LaunchRefit, PredictorBuildSelection
from histopilot.storage.io import content_hash, read_json_bounded, utc_now, write_json_atomic
from histopilot.storage.lifecycle import lifecycle_guard
from histopilot.storage.project_lock import (
    StorageError,
    ensure_managed_directory,
    reject_symlink_components,
)
from histopilot.taskcenter import ids
from histopilot.taskcenter.model import LIVE
from histopilot.workers.compute_archive import prepare_compute_archive

ACTIVE = {"queued", "waiting", "running", "cancelling"}
TERMINAL = {"completed", "cancelled", "skipped"}
# A lock another request held: the step changed nothing and is retried, never failed.
BUSY_CODES = {"PROJECT_BUSY", "OUTPUT_BUSY"}
# Archives whose coordinator defines this constant run as Task Center tasks.
TASK_CENTER_PROTOCOL = 1


def source_items(experiment_id, batches, policy=None, *, policies=None):
    """Expand methods once per seed group, independently of the fold count."""
    items = []
    for batch in batches:
        selected = policies[batch["id"]] if policies is not None else policy
        if policies is not None:
            verify_batch_policy(selected, batch["manifest"].get("spec", {}))
        if selected["method"] == "skip":
            continue
        methods = ("ensemble", "refit") if selected["method"] == "both" else (selected["method"],)
        groups = PredictorService._groups(batch["manifest"])
        numbers = {row["id"]: row["number"] for row in batch["manifest"]["configurations"]}
        for (candidate, training_seed, split_seed), runs in sorted(groups.items()):
            source = {
                "experimentId": experiment_id,
                "batchId": batch["id"],
                "candidateId": candidate,
                "trainingSeed": training_seed,
                "splitSeed": split_seed,
            }
            for method in methods:
                items.append(
                    {
                        "key": content_hash([source, method]),
                        "source": source,
                        "method": method,
                        **(
                            {
                                "refitPercentile": selected["refitPercentile"]
                                if method == "refit"
                                else None
                            }
                            if policies is not None
                            else {}
                        ),
                        "configurationNumber": numbers[candidate],
                        "foldCount": len(runs),
                        "runIds": sorted(row["id"] for row in runs),
                    }
                )
    return items


class TaskCenterExperimentExecutor(TaskCenterAccess):
    """Queue the coordinator as a Task Center service task behind its fold batches.

    It depends on every fold task of its batches succeeding and on each batch's final
    results collection finishing; batches launched before the Task Center add nothing.
    """

    def running(self, session):
        task = self.client.by_session(session)
        return task is not None and task["state"] in LIVE

    def dependencies(self, plan):
        """Wait for every batch's final results; the coordinator itself decides per batch.

        A cancelled or failed batch marks only its own items, as it always has, so one
        stopped fold never blocks the predictors of the experiment's other batches.
        """
        return [
            {"task": task["id"], "condition": "terminal"}
            for batch in plan["batchIds"]
            for task in self.client.store.list(
                group=("mil-batch", batch),
                project_folder=plan["projectFolder"],
                kinds=("mil-collect",),
                limit=None,
            )
            if task["adapterData"].get("final")
        ]

    def launch(self, session, python, plan, log, *, package_root, task_id, owner, title):
        frozen = read_json_bounded(plan)
        folder = Path(plan).parent
        spec = {
            "id": task_id,
            "kind": "predictor-coordinator",
            "adapter": "predictor-coordinator",
            "title": title,
            "group": {"kind": "predictor-coordinator", "id": frozen["experimentId"]},
            "sessionName": session,
            "labels": {"experimentId": frozen["experimentId"]},
            "request": {"lane": "cpu", "service": True, "cpuThreads": 1, "ramGb": 0.5},
            "command": {
                "argv": host_gpu_argv(
                    [python, "-u", "-m", "histopilot.workers.experiment_predictors", str(plan)]
                ),
                "cwd": str(package_root),
                "env": {"PYTHONDONTWRITEBYTECODE": "1"},
                "log": str(log),
            },
            "adapterData": {
                "coordinatorFolder": str(folder),
                "experimentId": frozen["experimentId"],
            },
            "dependsOn": self.dependencies(frozen),
        }
        submit_task(
            self.client,
            owner,
            spec,
            reason="launch",
            active_message="This experiment coordinator already exists.",
            active_code="EXPERIMENT_ACTIVE",
        )
        self.wake()


class ExperimentPredictorService:
    def __init__(
        self,
        store,
        filesystem,
        *,
        training=None,
        builds=None,
        refits=None,
        runtime=None,
        task_center=None,
    ):
        self.store, self.filesystem = store, filesystem
        self.executor = TaskCenterExperimentExecutor(task_center)
        if training is None:
            from histopilot.application.training import TrainingService

            training = TrainingService(store, filesystem)
        self.training = training
        self.builds = builds or PredictorBuildService(store, filesystem)
        self.refits = refits or RefitService(store, filesystem)
        self.runtime = runtime or training_runtime

    @property
    def task_center(self):
        return self.executor.client

    @staticmethod
    def _legacy(state):
        """A coordinator started before the Task Center (a cancel before any start is not)."""
        return state.get("executor") != "task-center" and state.get("attempt", 0) > 0

    def _task(self, state):
        """The Task Center view of a coordinator; None for one from before the Task Center."""
        if state.get("executor") != "task-center":
            return None
        try:
            client = self.task_center
        except (StorageError, OSError) as error:
            return {
                "id": state.get("taskId"),
                "state": None,
                "unknown": True,
                "error": str(error),
                "runnerAlive": None,
            }
        return task_view(client, state.get("taskId"))

    def _coordinator_running(self, state):
        """A queued Task Center coordinator counts as running."""
        return state.get("executor") == "task-center" and task_pending(self._task(state))

    def folder(self, identity):
        folder = (
            self.store.folder
            / "experiment-predictors"
            / hashlib.sha256(identity.encode()).hexdigest()
        )
        reject_symlink_components(folder)
        return folder

    def _submission(self, identity, *, inactive=False):
        record = self.store.get_draft(identity, include_inactive=inactive)
        submission = record["payload"].get("submission")
        if (
            record["payload"].get("type") != "model-experiment"
            or not submission
            or not has_predictor_intent(submission)
        ):
            raise StorageError(
                "This historical experiment has no automatic predictor plan. Build its predictors explicitly, or copy it into a new experiment.",
                "EXPERIMENT_PREDICTORS_LEGACY",
                409,
            )
        if not predictor_work_expected(submission):
            raise StorageError(
                "This experiment started without predictors. Copy it to change that choice.",
                "EXPERIMENT_PREDICTOR_POLICY_LOCKED",
                409,
            )
        return record, submission

    def _read(self, identity):
        folder = self.folder(identity)
        plan, state = (
            read_json_bounded(folder / "plan.json"),
            read_json_bounded(folder / "state.json"),
        )
        _record, submission = self._submission(identity, inactive=True)
        if (
            state.get("planHash") != content_hash(plan)
            or plan.get("experimentId") != identity
            or plan.get("projectId") != self.store.project_id
            or plan.get("projectFolder") != str(self.store.folder)
            or plan.get("submissionOperationId") != submission["operationId"]
            or (
                plan.get("policies") != submission_policies(submission)
                if "predictorPolicies" in submission
                else plan.get("policy") != submission["predictorPolicy"]
            )
            or plan.get("batchIds") != submission["batchIds"]
            or plan.get("executionContract") != submission["executionContract"]
            or state.get("status") not in ACTIVE | TERMINAL | {"attention", "interrupted"}
            or len(state.get("items", [])) != len(plan["items"])
            or any(
                any(item.get(key) != value for key, value in expected.items())
                for item, expected in zip(state["items"], plan["items"], strict=True)
            )
        ):
            raise StorageError(
                "The frozen experiment predictor plan changed.",
                "EXPERIMENT_PREDICTOR_PLAN_CHANGED",
                409,
            )
        return plan, state

    @staticmethod
    def public(state, *, summary=False):
        legacy = ExperimentPredictorService._legacy(state)
        items = state.get("items", [])
        counts = {
            "total": len(items),
            "ensemble": sum(row["method"] == "ensemble" for row in items),
            "refit": sum(row["method"] == "refit" for row in items),
            "completed": sum(row["status"] == "completed" for row in items),
            "waiting": sum(row["status"] == "waiting" for row in items),
            "active": sum(row["status"] in {"queued", "running"} for row in items),
            "failed": sum(row["status"] == "failed" for row in items),
            "cancelled": sum(row["status"] == "cancelled" for row in items),
            "skipped": sum(row["status"] == "skipped" for row in items),
        }
        return {
            "status": state["status"],
            "counts": counts,
            "items": []
            if summary
            else [
                {
                    key: deepcopy(value)
                    for key, value in item.items()
                    if key not in {"buildRequest", "launchedAttempt"}
                }
                for item in items
            ],
            "error": state.get("error"),
            "updatedAt": state.get("updatedAt"),
            "sessionName": state.get("sessionName"),
            "logPath": state.get("logPath"),
            "retryable": not legacy
            and (
                state["status"] in {"attention", "interrupted"}
                or (state["status"] == "cancelled" and state.get("executor") == "task-center")
            ),
            "cancellable": not legacy and state["status"] not in TERMINAL | {"cancelling"},
            **(
                {
                    "executor": "task-center",
                    "waitingReason": state.get("waitingReason"),
                    "runnerAlive": state.get("runnerAlive"),
                }
                if state.get("executor") == "task-center"
                else {}
            ),
        }

    def status(self, identity, *, summary=False):
        folder = self.folder(identity)
        if not (folder / "state.json").exists():
            return None
        _plan, state = self._read(identity)
        if self._legacy(state):
            # Nothing runs a coordinator from before the Task Center any more.
            if state["status"] in ACTIVE:
                state = {
                    **state,
                    "status": "interrupted",
                    "error": {"code": LEGACY_CODE, "message": LEGACY_MESSAGE},
                }
        elif state["status"] in ACTIVE:
            from histopilot.application.lifecycle import confirmed_live

            task = self._task(state)
            if task:
                state = {**state, "runnerAlive": task.get("runnerAlive")}
            if task and task["state"] in {"blocked", "queued"}:
                state = {
                    **state,
                    "waitingReason": task["waitingReason"]
                    or (
                        "Waiting for the experiment's fold batches to finish."
                        if task["state"] == "blocked"
                        else "Queued in the Task Center."
                    ),
                }
            if not confirmed_live(state.get("process")) and not self._coordinator_running(state):
                if (folder / "cancel.requested").exists():
                    pending = False
                    for item in state["items"]:
                        if item["status"] in TERMINAL:
                            continue
                        if item["method"] == "refit" and item["recordId"]:
                            execution = self.refits.execution(item["recordId"])
                            if execution["status"] in {"queued", "running"}:
                                pending = True
                                continue
                        item.update(status="cancelled", error=None)
                    state = {
                        **state,
                        "status": "cancelling" if pending else "cancelled",
                        "error": None,
                    }
                else:
                    state = {
                        **state,
                        "status": "interrupted",
                        "error": {
                            "code": "EXPERIMENT_PREDICTORS_INTERRUPTED",
                            "message": "Predictor coordination stopped. Resume to continue from saved work.",
                        },
                    }
            elif (folder / "cancel.requested").exists():
                state = {**state, "status": "cancelling"}
        return self.public(state, summary=summary)

    def pending(self, identity, error, *, summary=False):
        """Show the frozen denominator even if coordinator startup never succeeded."""
        _record, submission = self._submission(identity, inactive=True)
        batches = [
            self.store.get_configuration(batch, include_inactive=True)
            for batch in submission["batchIds"]
        ]
        return self.public(
            {
                "status": "interrupted",
                "error": error,
                "items": [
                    {
                        **item,
                        "status": "waiting",
                        "recordId": None,
                        "predictorId": None,
                        "epochBudget": None,
                        "execution": None,
                        "error": None,
                    }
                    for item in source_items(
                        identity,
                        batches,
                        submission.get("predictorPolicy"),
                        policies=submission_policies(submission)
                        if "predictorPolicies" in submission
                        else None,
                    )
                ],
            },
            summary=summary,
        )

    def _new_plan(self, identity, submission):
        batches = [self.store.get_configuration(item) for item in submission["batchIds"]]
        mapped = "predictorPolicies" in submission
        policies = submission_policies(submission) if mapped else None
        policy = (
            None
            if mapped
            else ExperimentPredictorPolicy.model_validate(
                submission["predictorPolicy"]
            ).model_dump()
        )
        return {
            "version": 2 if mapped else 1,
            "experimentId": identity,
            "projectId": self.store.project_id,
            "projectFolder": str(self.store.folder),
            "submissionOperationId": submission["operationId"],
            "batchIds": submission["batchIds"],
            **({"policies": policies} if mapped else {"policy": policy}),
            "executionContract": submission["executionContract"],
            "name": submission["experiment"]["name"],
            "dataRoots": [str(path) for path in self.filesystem.roots],
            "items": source_items(identity, batches, policy, policies=policies),
        }

    def launch(self, identity, operation_id, *, resume=False):
        with lifecycle_guard(self.store.folder):
            record, submission = self._submission(identity)
            self.store.lifecycle.assert_document_usable(record)
            if (
                self.store.lifecycle.read()["records"].get(f"draft:{identity}", {}).get("state")
                == "archived"
            ):
                raise StorageError(
                    "Restore this experiment to Active before resuming predictors.",
                    "EXPERIMENT_ARCHIVED",
                    409,
                )
            if submission["status"] != "submitted":
                raise StorageError(
                    "Finish starting the experiment before creating predictors.",
                    "EXPERIMENT_SUBMISSION_REQUIRED",
                    409,
                )
            folder = self.folder(identity)
            ensure_managed_directory(folder)
            if (folder / f"cancel-{content_hash(operation_id)}.json").exists():
                raise StorageError(
                    "This operation belongs to a predictor cancellation.", "OPERATION_CONFLICT", 409
                )
            reopen = False
            if (folder / "state.json").exists():
                plan, previous = self._read(identity)
                if operation_id in previous["operations"]:
                    if previous["operations"][operation_id] != resume:
                        raise StorageError(
                            "This operation belongs to another predictor action.",
                            "OPERATION_CONFLICT",
                            409,
                        )
                    return self.status(identity)
                if self._legacy(previous):
                    refuse_legacy()
                current = self.status(identity)

                def accepted_noop():
                    previous["operations"][operation_id] = resume
                    write_json_atomic(folder / "state.json", previous)
                    return current

                # Task Center experiments stay resumable after a cancel, like their batches.
                reopen = (
                    resume
                    and current["status"] == "cancelled"
                    and previous.get("executor") == "task-center"
                )
                if current["status"] in ACTIVE | TERMINAL and not reopen:
                    return accepted_noop()
                if not resume:
                    # An accepted launch must never implicitly resume failures.
                    return accepted_noop()
                from histopilot.application.lifecycle import confirmed_live

                if confirmed_live(previous.get("process")) or self._coordinator_running(previous):
                    return accepted_noop()
            else:
                previous = None
                plan = self._new_plan(identity, submission)
            # Archive the exact code captured by the first training batch, which
            # remains usable even after the main application checkout changes.
            source = self.store.folder / "training" / plan["batchIds"][0] / "compute" / "histopilot"
            archive = prepare_compute_archive(
                folder,
                plan["executionContract"]["code"],
                source_root=source if source.is_dir() else None,
            )
            # A coordinator pinned before the Task Center cannot run as a task.
            coordinator = archive / "histopilot" / "application" / "experiment_predictors.py"
            if archive_protocol(coordinator) is None:
                refuse_legacy()
            task_id = ids.coordinator_task_id(str(folder))
            if previous is None:
                write_json_atomic(folder / "plan.json", plan)
            items = (
                deepcopy(previous["items"])
                if previous
                else [
                    {
                        **row,
                        "status": "waiting",
                        "recordId": None,
                        "predictorId": None,
                        "epochBudget": None,
                        "execution": None,
                        "error": None,
                    }
                    for row in plan["items"]
                ]
            )
            reopened = {"failed", "cancelled"} if reopen else {"failed"}
            for item in items:
                if item["status"] in reopened:
                    item.update(status="waiting", error=None)
            state = {
                "planHash": content_hash(plan),
                "status": "queued",
                "process": None,
                "sessionName": "hp-predictors-"
                + hashlib.sha256(str(folder).encode()).hexdigest()[:16],
                "logPath": str(folder / "worker.log"),
                "items": items,
                "error": None,
                "attempt": previous["attempt"] + 1 if previous else 1,
                "operations": {
                    **(previous["operations"] if previous else {}),
                    operation_id: resume,
                },
                "updatedAt": utc_now(),
                "executor": "task-center",
                "taskId": task_id,
            }
            (folder / "cancel.requested").unlink(missing_ok=True)
            write_json_atomic(folder / "state.json", state)
            try:
                self.executor.launch(
                    state["sessionName"],
                    sys.executable,
                    folder / "plan.json",
                    folder / "worker.log",
                    package_root=archive,
                    task_id=task_id,
                    owner={
                        "kind": "experiment",
                        "id": identity,
                        "title": (record.get("name") or plan["name"] or identity)[:200],
                        "projectId": self.store.project_id,
                        "projectFolder": str(self.store.folder),
                        "workspace": None,
                    },
                    title=f"Predictors · {record.get('name') or plan['name']}"[:200],
                )
            except Exception as error:
                try:
                    running = self.executor.running(state["sessionName"])
                    _plan, current = self._read(identity)
                    if running or current.get("process") or current["status"] != "queued":
                        return self.public(current)
                except Exception as inspection_error:
                    raise StorageError(
                        "Predictor launch acknowledgement was lost. Inspect its status before retrying.",
                        "EXPERIMENT_PREDICTORS_LAUNCH_UNCERTAIN",
                        409,
                    ) from inspection_error
                state.update(
                    status="interrupted",
                    error={"code": "EXPERIMENT_PREDICTORS_LAUNCH_FAILED", "message": str(error)},
                )
                write_json_atomic(folder / "state.json", state)
            return self.public(state)

    def cancel(self, identity, operation_id):
        with lifecycle_guard(self.store.folder):
            _record, submission = self._submission(identity)
            if submission["status"] != "submitted":
                raise StorageError(
                    "Finish starting the experiment before cancelling predictor creation.",
                    "EXPERIMENT_SUBMISSION_REQUIRED",
                    409,
                )
            state = self.status(identity)
            if state is not None and self._legacy(self._read(identity)[1]):
                refuse_legacy()
            if state is not None and state["status"] in TERMINAL:
                return state
            folder = self.folder(identity)
            ensure_managed_directory(folder)
            if state is None:
                # Startup may fail before an archive or coordinator exists.
                # Cancellation still freezes the decision without requiring a
                # working training environment or starting any process.
                plan = self._new_plan(identity, submission)
                stored = {
                    "planHash": content_hash(plan),
                    "status": "cancelled",
                    "process": None,
                    "sessionName": None,
                    "logPath": None,
                    "attempt": 0,
                    "operations": {},
                    "items": [
                        {
                            **item,
                            "status": "cancelled",
                            "recordId": None,
                            "predictorId": None,
                            "epochBudget": None,
                            "execution": None,
                            "error": None,
                        }
                        for item in plan["items"]
                    ],
                    "error": None,
                    "updatedAt": utc_now(),
                }
                write_json_atomic(folder / "plan.json", plan)
                write_json_atomic(folder / "state.json", stored)
            plan, stored = self._read(identity)
            if operation_id in stored["operations"]:
                raise StorageError(
                    "This operation belongs to another predictor action.", "OPERATION_CONFLICT", 409
                )
            receipt_path = folder / f"cancel-{content_hash(operation_id)}.json"
            if receipt_path.exists():
                receipt = read_json_bounded(receipt_path)
                if (
                    receipt.get("operationId") != operation_id
                    or type(receipt.get("attempt")) is not int
                ):
                    raise StorageError(
                        "The cancellation receipt changed.",
                        "EXPERIMENT_PREDICTOR_PLAN_CHANGED",
                        409,
                    )
                if receipt["attempt"] != stored["attempt"] or receipt.get("status") == "applied":
                    return self.status(identity)
            else:
                receipt = {
                    "operationId": operation_id,
                    "attempt": stored["attempt"],
                    "status": "requested",
                }
                write_json_atomic(receipt_path, receipt)
            # This marker is checked before every publication and refit launch.
            write_json_atomic(
                folder / "cancel.requested", {"operationId": operation_id, "at": utc_now()}
            )
            self._cancel_items(plan, stored)
            write_json_atomic(folder / "state.json", stored)
            self._cancel_pending_task(stored)
            write_json_atomic(receipt_path, {**receipt, "status": "applied"})
            return self.public(stored)

    def _cancel_pending_task(self, state):
        """A coordinator still waiting in the Task Center never needs to start.

        A running one sees ``cancel.requested`` within one iteration and exits itself.
        """
        task = self._task(state)
        if task and task["state"] in {"blocked", "queued"}:
            self.task_center.store.cancel_pending([task["id"]])

    def _cancel_items(self, plan, state):
        pending = False
        for item in state["items"]:
            if item["status"] in TERMINAL:
                continue
            if item["method"] == "refit" and item["recordId"]:
                execution = self.refits.execution(item["recordId"])
                if execution["status"] in {"queued", "running"}:
                    execution = self.refits.cancel(
                        item["recordId"],
                        "experiment-cancel-"
                        + content_hash(
                            [plan["submissionOperationId"], item["key"], state["attempt"]]
                        ),
                    )
                item["execution"] = execution
                if execution["status"] in {"queued", "running"}:
                    item["status"] = execution["status"]
                    pending = True
                    continue
            item.update(status="cancelled", error=None)
        state.update(
            status="cancelling" if pending else "cancelled", error=None, updatedAt=utc_now()
        )

    def _runtime_contract(self, plan):
        # Refit execution must use the same code and package versions as CV.
        # Launching from the pinned coordinator archive also pins downstream
        # ComputeJobService's source snapshot and worker archive.
        from histopilot.application.model_experiments import (
            execution_contract,
            require_execution_contract,
        )
        from histopilot.workers.training_process import compute_snapshot

        runtime = self.runtime()
        if not runtime.get("available"):
            raise StorageError(
                "The training runtime the experiment started with is unavailable.",
                "TRAINING_RUNTIME_UNAVAILABLE",
                409,
            )
        require_execution_contract(
            plan["executionContract"],
            execution_contract({"code": compute_snapshot(), "runtime": runtime}),
        )

    def _build(self, plan, item, save):
        request = PredictorBuildSelection(
            selections=[item["source"]],
            method=item["method"],
            refitPercentile=item.get(
                "refitPercentile", (plan.get("policy") or {}).get("refitPercentile")
            )
            or 50.0,
            namePrefix=plan["name"][:100],
        )
        operation = "experiment-predictor-" + content_hash(
            [plan["submissionOperationId"], item["key"]]
        )
        # Persist the exact reviewed request before dispatch: a publication may
        # succeed just before its acknowledgement is lost.
        if not item.get("buildRequest"):
            preview = self.builds.preview(request)
            if not preview["canBuild"]:
                findings = [
                    finding
                    for row in preview["items"]
                    for finding in row["findings"]
                    if finding["severity"] == "error"
                ]
                raise StorageError(
                    "; ".join(row["message"] for row in findings),
                    findings[0]["code"] if findings else "PREDICTOR_BUILD_BLOCKED",
                    409,
                    findings=findings,
                )
            item["buildRequest"] = {
                **request.model_dump(),
                "previewHash": preview["previewHash"],
                "operationId": operation,
            }
            save()
        return self.builds.apply(ApplyPredictorBuilds.model_validate(item["buildRequest"]))

    def advance(self, identity):
        """One worker iteration; never called by GET endpoints.

        Batch and refit status is read once per pass without the lifecycle guard; each
        source item then takes the guard only to check cancellation, act and save, so a
        large parameter sweep stays inspectable and cancellable. Every publication and
        refit launch re-reads what it acts on under the guard.
        """
        with lifecycle_guard(self.store.folder):
            plan, state = self._read(identity)
            keys = [row["key"] for row in state["items"] if row["status"] not in TERMINAL]
        result = self.public(state)
        if not keys:
            return result
        observed = self._observe(plan, state)
        for key in keys:
            result = self._advance_one(identity, key, observed)
            if result["status"] in TERMINAL:
                break
        return result

    def _observe(self, plan, state):
        return {
            "batches": {
                key: self.training.execution(key, include_progress=False)
                for key in plan["batchIds"]
            },
            "refits": {
                item["recordId"]: self.refits.execution(item["recordId"])
                for item in state["items"]
                if item["method"] == "refit" and item["recordId"] and item["status"] not in TERMINAL
            },
        }

    def _advance_one(self, identity, item_key, observed=None):
        """Serialize one source item with cancellation and other publications.

        Every ready refit is launched as its own task; the Task Center decides when each
        runs. A busy project lock leaves the item as it was for the next pass.
        """
        observed = observed or {}
        with lifecycle_guard(self.store.folder):
            plan, state = self._read(identity)
            if "batches" not in observed:
                observed = self._observe(plan, state)
            record, _submission = self._submission(identity)
            self.store.lifecycle.assert_document_usable(record)
            folder = self.folder(identity)
            if (folder / "cancel.requested").exists():
                self._cancel_items(plan, state)
                write_json_atomic(folder / "state.json", state)
                return self.public(state)
            batches = observed["batches"]
            for item in state["items"]:
                if item["key"] != item_key:
                    continue
                active_execution = (
                    observed["refits"].get(item["recordId"], {}) if item["recordId"] else {}
                )
                if item["status"] == "failed" and active_execution.get("status") in {
                    "queued",
                    "running",
                }:
                    item.update(
                        status=active_execution["status"], execution=active_execution, error=None
                    )
                if item["status"] in TERMINAL | {"failed"}:
                    continue
                try:
                    batch = batches[item["source"]["batchId"]]
                    if not batch or batch["status"] in {"planned", "queued", "running"}:
                        continue
                    if batch["status"] == "cancelled":
                        item["status"] = "cancelled"
                        continue
                    if batch["status"] != "completed":
                        raise StorageError(
                            "Resume this experiment's unfinished fold batch before retrying predictors.",
                            "EXPERIMENT_FOLDS_INCOMPLETE",
                            409,
                        )
                    batch_record = self.store.get_configuration(item["source"]["batchId"])
                    if (
                        batch_record["manifest"]["spec"].get("candidateSelection")
                        == "best_validation"
                    ):
                        from histopilot.candidate_selection import validation_selection

                        training_folder = self.store.folder / "training" / item["source"]["batchId"]
                        selected = validation_selection(
                            read_json_bounded(training_folder / "plan.json"),
                            read_json_bounded(training_folder / "state.json"),
                        )
                        if not selected or not selected["ready"]:
                            raise StorageError(
                                "Complete valid validation scores are required for configuration selection.",
                                "VALIDATION_SELECTION_UNAVAILABLE",
                                409,
                            )
                        if selected["selectedCandidateId"] != item["source"]["candidateId"]:
                            item.update(status="skipped", selectionEvidence=selected, error=None)
                            continue
                    if not item["recordId"]:
                        # _build stores its review in memory; save it even if an
                        # acknowledgement fails after external publication.
                        try:
                            built = self._build(
                                plan, item, lambda: write_json_atomic(folder / "state.json", state)
                            )
                        finally:
                            state["updatedAt"] = utc_now()
                            write_json_atomic(folder / "state.json", state)
                        result = built["items"][0]
                        if result["status"] == "failed":
                            raise StorageError(
                                result["error"]["message"], result["error"]["code"], 409
                            )
                        item["recordId"] = result["recordId"]
                        if result["recordKind"] == "frozen-predictor":
                            item.update(
                                status="completed", predictorId=result["recordId"], error=None
                            )
                            continue
                    if item["method"] == "refit":
                        refit = self.refits.get(item["recordId"])
                        item["epochBudget"] = refit["manifest"]["epochBudget"]
                        # Re-read under the guard: this item acts on it.
                        execution = self.refits.execution(item["recordId"])
                        item["execution"] = execution
                        status = execution["status"]
                        if status == "completed":
                            published = self.refits.publish(
                                item["recordId"],
                                "experiment-refit-publish-"
                                + content_hash([plan["submissionOperationId"], item["key"]]),
                            )
                            item.update(status="completed", predictorId=published["id"], error=None)
                        elif status in {"queued", "running"}:
                            item["status"] = status
                        elif (
                            status == "cancelled"
                            and item.get("launchedAttempt") == state["attempt"]
                        ):
                            # Cancelled during this attempt; a resumed attempt relaunches
                            # refits cancelled by an earlier one.
                            item["status"] = "cancelled"
                        else:
                            # Failure retries require a new explicit coordinator
                            # attempt; polling never silently retries failed work.
                            if (
                                status in {"failed", "interrupted"}
                                and item.get("launchedAttempt") == state["attempt"]
                            ):
                                raise StorageError(
                                    execution.get("error")
                                    or "Refit stopped. Resume predictor creation to continue.",
                                    "EXPERIMENT_REFIT_STOPPED",
                                    409,
                                )
                            self._runtime_contract(plan)
                            operation = "experiment-refit-launch-" + content_hash(
                                [plan["submissionOperationId"], item["key"], state["attempt"]]
                            )
                            item["launchedAttempt"] = state["attempt"]
                            write_json_atomic(folder / "state.json", state)
                            execution = self.refits.launch(
                                item["recordId"],
                                LaunchRefit(operationId=operation),
                                resume=status != "not_started",
                            )
                            item.update(status=execution["status"], execution=execution)
                except (StorageError, OSError, ValueError, RuntimeError) as error:
                    if getattr(error, "code", None) in BUSY_CODES:
                        # Another request held a lock; nothing was changed. Every step is
                        # idempotent by operation id, so the next pass simply retries.
                        continue
                    item.update(
                        status="failed",
                        error={
                            "code": getattr(error, "code", "EXPERIMENT_PREDICTOR_FAILED"),
                            "message": str(error),
                        },
                    )
                    # Reconcile a lost child launch acknowledgement before
                    # deciding that the item failed or dispatching another one.
                    if item["method"] == "refit" and item["recordId"]:
                        try:
                            execution = self.refits.execution(item["recordId"])
                            if execution["status"] in {"queued", "running"}:
                                item.update(
                                    status=execution["status"], execution=execution, error=None
                                )
                        except (StorageError, OSError, ValueError, RuntimeError):
                            pass
                finally:
                    state["updatedAt"] = utc_now()
                    write_json_atomic(folder / "state.json", state)
            statuses = {row["status"] for row in state["items"]}
            state["status"] = (
                "completed"
                if statuses <= {"completed", "skipped"}
                else "cancelled"
                if statuses <= TERMINAL
                else "running"
                if statuses & {"queued", "running"}
                else "waiting"
                if "waiting" in statuses
                else "attention"
            )
            state["error"] = next(
                (row["error"] for row in state["items"] if row["status"] == "failed"), None
            )
            state["updatedAt"] = utc_now()
            write_json_atomic(folder / "state.json", state)
            return self.public(state)
