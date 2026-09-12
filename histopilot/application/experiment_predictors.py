"""Durable predictor creation belonging to one frozen experiment submission.

Only submit/resume starts the persistent coordinator. Status is observational.
Each source is a complete configuration/training-seed/split-seed fold group;
folds are never counted as separate predictors. Individual publications and
refit launches have stable operation IDs, including across coordinator crashes.
"""

from __future__ import annotations

import hashlib
import shlex
import subprocess
import sys
from copy import deepcopy

from histopilot.adapters.native.runtime import training_runtime
from histopilot.application.feature_bundles import _hash
from histopilot.application.predictor_builds import PredictorBuildService
from histopilot.application.predictors import PredictorService
from histopilot.application.refits import RefitService
from histopilot.schemas.model_experiments import ExperimentPredictorPolicy
from histopilot.schemas.predictors import ApplyPredictorBuilds, LaunchRefit, PredictorBuildSelection
from histopilot.storage.lifecycle import lifecycle_guard
from histopilot.storage.project_lock import (
    StorageError,
    _reject_symlink_components,
    ensure_managed_directory,
)
from histopilot.workers.compute_archive import prepare_compute_archive
from histopilot.workers.packing_process import write_json
from histopilot.workers.training_process import TmuxTrainingExecutor, now, read_json

ACTIVE = {"queued", "waiting", "running", "cancelling"}
TERMINAL = {"completed", "cancelled"}


def source_items(experiment_id, batches, policy):
    """Expand methods once per seed group, independently of the fold count."""
    if policy["method"] == "skip":
        return []
    methods = ("ensemble", "refit") if policy["method"] == "both" else (policy["method"],)
    items = []
    for batch in batches:
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
                        "key": _hash([source, method]),
                        "source": source,
                        "method": method,
                        "configurationNumber": numbers[candidate],
                        "foldCount": len(runs),
                        "runIds": sorted(row["id"] for row in runs),
                    }
                )
    return items


class TmuxExperimentExecutor(TmuxTrainingExecutor):
    def launch(self, session, python, plan, log, *, package_root):
        subprocess.run(["tmux", "ls"], capture_output=True, timeout=10)
        if self.running(session):
            raise StorageError("This experiment coordinator already exists.", "EXPERIMENT_ACTIVE")
        _reject_symlink_components(log)
        command = "cd " + shlex.quote(str(package_root)) + " && "
        command += shlex.join(
            [
                "env",
                "PYTHONDONTWRITEBYTECODE=1",
                python,
                "-u",
                "-m",
                "histopilot.workers.experiment_predictors",
                str(plan),
            ]
        )
        command += " >> " + shlex.quote(str(log)) + " 2>&1"
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", session, command],
            capture_output=True,
            check=True,
            timeout=15,
        )


class ExperimentPredictorService:
    def __init__(
        self,
        store,
        filesystem,
        *,
        executor=None,
        training=None,
        builds=None,
        refits=None,
        runtime=None,
    ):
        self.store, self.filesystem = store, filesystem
        self.executor = executor or TmuxExperimentExecutor()
        if training is None:
            from histopilot.application.training import TrainingService

            training = TrainingService(store, filesystem)
        self.training = training
        self.builds = builds or PredictorBuildService(store, filesystem)
        self.refits = refits or RefitService(store, filesystem)
        self.runtime = runtime or training_runtime

    def folder(self, identity):
        folder = (
            self.store.folder
            / "experiment-predictors"
            / hashlib.sha256(identity.encode()).hexdigest()
        )
        _reject_symlink_components(folder)
        return folder

    def _submission(self, identity, *, inactive=False):
        record = self.store.get_draft(identity, include_inactive=inactive)
        submission = record["payload"].get("submission")
        if (
            record["payload"].get("type") != "model-experiment"
            or not submission
            or not submission.get("predictorPolicy")
        ):
            raise StorageError(
                "This historical experiment has no automatic predictor plan. Build its predictors explicitly, or copy it into a new experiment.",
                "EXPERIMENT_PREDICTORS_LEGACY",
                409,
            )
        if submission["predictorPolicy"]["method"] == "skip":
            raise StorageError(
                "This experiment was submitted without predictors. Copy it to change that choice.",
                "EXPERIMENT_PREDICTOR_POLICY_LOCKED",
                409,
            )
        return record, submission

    def _read(self, identity):
        folder = self.folder(identity)
        plan, state = read_json(folder / "plan.json"), read_json(folder / "state.json")
        _record, submission = self._submission(identity, inactive=True)
        if (
            state.get("planHash") != _hash(plan)
            or plan.get("experimentId") != identity
            or plan.get("projectId") != self.store.project_id
            or plan.get("projectFolder") != str(self.store.folder)
            or plan.get("submissionOperationId") != submission["operationId"]
            or plan.get("policy") != submission["predictorPolicy"]
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
            "retryable": state["status"] in {"attention", "interrupted"},
            "cancellable": state["status"] not in TERMINAL | {"cancelling"},
        }

    def status(self, identity, *, summary=False):
        folder = self.folder(identity)
        if not (folder / "state.json").exists():
            return None
        _plan, state = self._read(identity)
        if state["status"] in ACTIVE:
            from histopilot.application.lifecycle import _confirmed_live

            if not _confirmed_live(state.get("process")) and not self.executor.running(
                state["sessionName"]
            ):
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
                    for item in source_items(identity, batches, submission["predictorPolicy"])
                ],
            },
            summary=summary,
        )

    def _new_plan(self, identity, submission):
        batches = [self.store.get_configuration(item) for item in submission["batchIds"]]
        policy = ExperimentPredictorPolicy.model_validate(
            submission["predictorPolicy"]
        ).model_dump()
        return {
            "version": 1,
            "experimentId": identity,
            "projectId": self.store.project_id,
            "projectFolder": str(self.store.folder),
            "submissionOperationId": submission["operationId"],
            "batchIds": submission["batchIds"],
            "policy": policy,
            "executionContract": submission["executionContract"],
            "name": submission["experiment"]["name"],
            "dataRoots": [str(path) for path in self.filesystem.roots],
            "items": source_items(identity, batches, policy),
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
                    "Finish submitting the experiment before creating predictors.",
                    "EXPERIMENT_SUBMISSION_REQUIRED",
                    409,
                )
            folder = self.folder(identity)
            ensure_managed_directory(folder)
            if (folder / f"cancel-{_hash(operation_id)}.json").exists():
                raise StorageError(
                    "This operation belongs to a predictor cancellation.", "OPERATION_CONFLICT", 409
                )
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
                current = self.status(identity)

                def accepted_noop():
                    previous["operations"][operation_id] = resume
                    write_json(folder / "state.json", previous)
                    return current

                if current["status"] in ACTIVE | TERMINAL:
                    return accepted_noop()
                if not resume:
                    # An accepted launch must never implicitly resume failures.
                    return accepted_noop()
                from histopilot.application.lifecycle import _confirmed_live

                if _confirmed_live(previous.get("process")) or self.executor.running(
                    previous["sessionName"]
                ):
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
            if previous is None:
                write_json(folder / "plan.json", plan)
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
            for item in items:
                if item["status"] == "failed":
                    item.update(status="waiting", error=None)
            state = {
                "planHash": _hash(plan),
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
                "updatedAt": now(),
            }
            (folder / "cancel.requested").unlink(missing_ok=True)
            write_json(folder / "state.json", state)
            try:
                self.executor.launch(
                    state["sessionName"],
                    sys.executable,
                    folder / "plan.json",
                    folder / "worker.log",
                    package_root=archive,
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
                write_json(folder / "state.json", state)
            return self.public(state)

    def cancel(self, identity, operation_id):
        with lifecycle_guard(self.store.folder):
            _record, submission = self._submission(identity)
            if submission["status"] != "submitted":
                raise StorageError(
                    "Finish submission before cancelling predictor creation.",
                    "EXPERIMENT_SUBMISSION_REQUIRED",
                    409,
                )
            state = self.status(identity)
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
                    "planHash": _hash(plan),
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
                    "updatedAt": now(),
                }
                write_json(folder / "plan.json", plan)
                write_json(folder / "state.json", stored)
            plan, stored = self._read(identity)
            if operation_id in stored["operations"]:
                raise StorageError(
                    "This operation belongs to another predictor action.", "OPERATION_CONFLICT", 409
                )
            receipt_path = folder / f"cancel-{_hash(operation_id)}.json"
            if receipt_path.exists():
                receipt = read_json(receipt_path)
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
                write_json(receipt_path, receipt)
            # This marker is checked before every publication and refit launch.
            write_json(folder / "cancel.requested", {"operationId": operation_id, "at": now()})
            self._cancel_items(plan, stored)
            write_json(folder / "state.json", stored)
            write_json(receipt_path, {**receipt, "status": "applied"})
            return self.public(stored)

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
                        + _hash([plan["submissionOperationId"], item["key"], state["attempt"]]),
                    )
                item["execution"] = execution
                if execution["status"] in {"queued", "running"}:
                    item["status"] = execution["status"]
                    pending = True
                    continue
            item.update(status="cancelled", error=None)
        state.update(status="cancelling" if pending else "cancelled", error=None, updatedAt=now())

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
                "The submitted training runtime is unavailable.",
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
            refitPercentile=plan["policy"]["refitPercentile"] or 50.0,
            namePrefix=plan["name"][:100],
        )
        operation = "experiment-predictor-" + _hash([plan["submissionOperationId"], item["key"]])
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

        Release the lifecycle guard between source groups: a large parameter
        sweep must remain inspectable and cancellable while checkpoints are
        verified. Every publication reacquires it and checks cancellation again.
        """
        with lifecycle_guard(self.store.folder):
            _plan, state = self._read(identity)
            keys = [row["key"] for row in state["items"] if row["status"] not in TERMINAL]
        result = self.public(state)
        for key in keys:
            result = self._advance_one(identity, key)
            if result["status"] in TERMINAL:
                break
        return result

    def _advance_one(self, identity, item_key):
        """Serialize one source item with cancellation and other publications."""
        with lifecycle_guard(self.store.folder):
            plan, state = self._read(identity)
            record, _submission = self._submission(identity)
            self.store.lifecycle.assert_document_usable(record)
            folder = self.folder(identity)
            if (folder / "cancel.requested").exists():
                self._cancel_items(plan, state)
                write_json(folder / "state.json", state)
                return self.public(state)
            batches = {
                key: self.training.execution(key, include_progress=False)
                for key in plan["batchIds"]
            }
            refit_executions = {
                item["recordId"]: self.refits.execution(item["recordId"])
                for item in state["items"]
                if item["method"] == "refit" and item["recordId"] and item["status"] not in TERMINAL
            }
            active_refit = any(
                row["status"] in {"queued", "running"} for row in refit_executions.values()
            )
            for item in state["items"]:
                if item["key"] != item_key:
                    continue
                active_execution = refit_executions.get(item["recordId"], {})
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
                    if not item["recordId"]:
                        # _build stores its review in memory; save it even if an
                        # acknowledgement fails after external publication.
                        try:
                            built = self._build(
                                plan, item, lambda: write_json(folder / "state.json", state)
                            )
                        finally:
                            state["updatedAt"] = now()
                            write_json(folder / "state.json", state)
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
                        execution = refit_executions.get(item["recordId"]) or self.refits.execution(
                            item["recordId"]
                        )
                        item["execution"] = execution
                        status = execution["status"]
                        if status == "completed":
                            published = self.refits.publish(
                                item["recordId"],
                                "experiment-refit-publish-"
                                + _hash([plan["submissionOperationId"], item["key"]]),
                            )
                            item.update(status="completed", predictorId=published["id"], error=None)
                        elif status in {"queued", "running"}:
                            item["status"] = status
                            active_refit = True
                        elif active_refit:
                            item["status"] = "waiting"
                        elif status == "cancelled":
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
                            operation = "experiment-refit-launch-" + _hash(
                                [plan["submissionOperationId"], item["key"], state["attempt"]]
                            )
                            item["launchedAttempt"] = state["attempt"]
                            write_json(folder / "state.json", state)
                            execution = self.refits.launch(
                                item["recordId"],
                                LaunchRefit(operationId=operation),
                                resume=status != "not_started",
                            )
                            item.update(status=execution["status"], execution=execution)
                            active_refit = execution["status"] in {"queued", "running"}
                except (StorageError, OSError, ValueError, RuntimeError) as error:
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
                                active_refit = True
                        except (StorageError, OSError, ValueError, RuntimeError):
                            active_refit = True
                finally:
                    state["updatedAt"] = now()
                    write_json(folder / "state.json", state)
            statuses = {row["status"] for row in state["items"]}
            state["status"] = (
                "completed"
                if statuses <= {"completed"}
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
            state["updatedAt"] = now()
            write_json(folder / "state.json", state)
            return self.public(state)
