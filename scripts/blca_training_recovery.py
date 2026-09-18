"""Opt-in, one-fold interruption/recovery exercise for the new BLCA study only.

The owning coordinator calls advance from its polling loop. This module never
starts another coordinator and never cancels the experiment or predictor plan.
One identity-checked fold leader receives SIGTERM after a resumable checkpoint;
all recovery uses HistoPilot's native services and original frozen batch.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import signal
import stat
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from histopilot.application.training import run_processes
from histopilot.workers.packing_process import write_json
from histopilot.workers.training_process import (
    confirmed_process_alive,
    process_identity,
)

TRAINING_OPERATION = "blca-interruption-training-resume-v1"
PREDICTOR_OPERATION = "blca-interruption-predictors-resume-v1"


def _now():
    return datetime.now(UTC).isoformat()


def _read(path):
    return json.loads(Path(path).read_text())


def _hash(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("Recovery evidence requires an existing regular owned file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _probe_checkpoint(python, path):
    """Decode a bounded stable snapshot with the captured training interpreter.

    A later epoch may atomically replace last.ckpt during torch startup. That does
    not invalidate proof that an earlier complete checkpoint existed before the
    interruption. An in-place write during the actual copy does invalidate it.
    """
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        return None
    program = (
        "import json,sys,torch; c=torch.load(sys.argv[1],map_location='cpu',weights_only=True); "
        "print(json.dumps({'epoch':c.get('epoch'),'globalStep':c.get('global_step'),"
        "'optimizerStatePresent':bool(c.get('optimizer_states')),"
        "'rngStatePresent':all(k in c.get('trainingRngState',{}) for k in ('python','numpy','torch','cuda'))}))"
    )
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return None
    digest = hashlib.sha256()
    with os.fdopen(descriptor, "rb") as source, tempfile.NamedTemporaryFile(suffix=".ckpt") as snapshot:
        before = os.fstat(source.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= 512 * 1024**2:
            return None
        copied = 0
        for block in iter(lambda: source.read(1024 * 1024), b""):
            copied += len(block)
            if copied > before.st_size:
                return None
            snapshot.write(block)
            digest.update(block)
        after = os.fstat(source.fileno())
        if copied != before.st_size or (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size, after.st_mtime_ns, after.st_ctime_ns
        ):
            return None
        snapshot.flush()
        try:
            result = subprocess.run([python, "-c", program, snapshot.name], capture_output=True,
                                    text=True, timeout=30, check=False)
        except subprocess.TimeoutExpired:
            return None  # An overloaded runtime may recover by the next coordinator poll.
    if result.returncode:
        return None  # A concurrently replaced checkpoint may be incomplete; wait for the next poll.
    try:
        evidence = json.loads(result.stdout)
    except (TypeError, ValueError):
        return None
    if (type(evidence.get("epoch")) is not int or evidence["epoch"] < 0
            or type(evidence.get("globalStep")) is not int or evidence["globalStep"] < 1
            or not evidence.get("optimizerStatePresent") or not evidence.get("rngStatePresent")):
        return None
    return {**evidence, "path": str(path), "sha256": digest.hexdigest(), "bytes": before.st_size,
            "mtimeNs": before.st_mtime_ns, "stableSnapshotVerified": True}


def _owned_leader(run, folder):
    identity = run.get("process")
    if not identity or not confirmed_process_alive(identity):
        return False
    pid = identity["pid"]
    try:
        if process_identity(pid) != identity or os.getpgid(pid) != pid:
            return False
        arguments = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except (FileNotFoundError, ProcessLookupError):
        return False  # Normal fold completion can race checkpoint probing.
    expected = str(folder / "runs" / run["id"] / "plan.json").encode()
    return b"--fold" in arguments and expected in arguments


def _open_pidfd(pid):
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid, 0)
    # Some standalone Python distributions omit os.pidfd_open even though the
    # Linux host and libc support it. Use libc's typed wrapper, never a raw PID
    # signal fallback or an architecture-dependent syscall number.
    library = ctypes.CDLL(None, use_errno=True)
    if not hasattr(library, "pidfd_open"):
        raise RuntimeError("The interruption exercise requires Linux process descriptors")
    function = library.pidfd_open
    function.argtypes = (ctypes.c_int, ctypes.c_uint)
    function.restype = ctypes.c_int
    result = function(pid, 0)
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def _signal_owned_leader(run, folder):
    """Pin the process before rechecking ownership, preventing PID-reuse signals."""
    if not hasattr(signal, "pidfd_send_signal"):
        raise RuntimeError("The interruption exercise requires Linux process-descriptor signalling")
    try:
        descriptor = _open_pidfd(run["process"]["pid"])
    except ProcessLookupError:
        return False
    try:
        if not _owned_leader(run, folder):
            return False
        try:
            signal.pidfd_send_signal(descriptor, signal.SIGTERM, None, 0)
        except ProcessLookupError:
            return False
        return True
    finally:
        os.close(descriptor)


def _completed_proof(runs, folder):
    proof = []
    for run in runs:
        if run["status"] != "completed":
            continue
        run_folder = folder / "runs" / run["id"]
        receipt = _read(run_folder / "result.json")
        files = {"result.json": _hash(run_folder / "result.json")}
        for key in ("bestCheckpointPath", "lastCheckpointPath"):
            path = Path(receipt[key])
            if path.parent.resolve() != run_folder.resolve():
                raise RuntimeError("Completed checkpoint escaped its original run folder")
            files[path.name] = _hash(path)
        proof.append({"runId": run["id"], "attempt": run.get("attempt"), "files": files})
    return proof


def _verify_completed(proof, runs, folder):
    current = {run["id"]: run for run in runs}
    for row in proof:
        run = current.get(row["runId"], {})
        if run.get("status") != "completed" or run.get("attempt") != row["attempt"]:
            raise RuntimeError("A previously completed fold was rerun or lost")
        for name, digest in row["files"].items():
            if _hash(folder / "runs" / row["runId"] / name) != digest:
                raise RuntimeError("A previously completed fold's receipt or checkpoint changed")


def _training_stopped(service, training):
    return not (
        confirmed_process_alive(training.get("process"))
        or any(run_processes(run) for run in training["runs"])
        or service.executor.running(training["sessionName"])
    )


def _predictor_stopped(service, identity):
    _, state = service._read(identity)
    return not (confirmed_process_alive(state.get("process"))
                or service.executor.running(state["sessionName"]))


def advance(study, training, predictor_work):
    """Return True while handling the expected interruption; False allows normal polling.

    The caller must opt in with study.exercise_recovery. Evidence is separate from
    state.json, so the main coordinator remains its only writer. Unexpected
    failures propagate; they are never hidden as an exercised recovery.
    """
    if not getattr(study, "exercise_recovery", False):
        return False
    state = study.state
    project = Path(state["projectPath"]).resolve()
    if project.name != "blca-e2e-20260917":
        raise RuntimeError("Interruption exercise is restricted to the new BLCA project")
    batch_id, experiment_id = state["batchId"], state["experimentId"]
    if training["batchId"] != batch_id:
        raise RuntimeError("Training status does not belong to the owned BLCA batch")
    folder = project / "training" / batch_id
    plan_path = folder / "plan.json"
    evidence_path = Path(study.path).parent / "evidence/interruption-resume.json"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence = _read(evidence_path) if evidence_path.exists() else None
    if evidence is not None and any(evidence.get(key) != state[key]
                                   for key in ("projectId", "batchId", "experimentId")):
        raise RuntimeError("Recovery evidence belongs to different study bindings")
    skipped_candidates = []
    if evidence and evidence.get("phase") == "signal_not_dispatched":
        # A pidfd reported that the process had already exited, so no signal was
        # delivered. Retain its audit and wait for another owned live candidate.
        skipped_candidates = [*evidence.get("skippedCandidates", []),
                              {"runId": evidence["runId"], "at": evidence["updatedAt"],
                               "reason": "Fold exited before signal delivery"}]
        evidence = None
    training_service = study.experiments.training
    predictors = study.experiments._predictors()

    def save(**values):
        evidence.update(values, updatedAt=_now())
        write_json(evidence_path, evidence)

    if evidence is None:
        if training["status"] not in {"queued", "running"}:
            if training["status"] == "completed":
                raise RuntimeError("Training finished before the requested interruption exercise")
            return False
        if not any(row["status"] == "completed" for row in training["runs"]):
            return False
        active = [row for row in training["runs"] if row["status"] == "running"]
        if len(active) != 1:
            return False
        run = active[0]
        if not _owned_leader(run, folder):
            return False
        plan = _read(plan_path)
        checkpoint = _probe_checkpoint(plan["runtime"]["python"], folder / "runs" / run["id"] / "last.ckpt")
        if checkpoint is None:
            return False
        completed = _completed_proof(training["runs"], folder)
        if not _owned_leader(run, folder):
            return False
        evidence = {"schemaVersion": 1, "projectId": state["projectId"], "batchId": batch_id,
            "experimentId": experiment_id, "createdAt": _now(), "phase": "signal_intent",
            "runId": run["id"], "attemptBefore": run.get("attempt", 1),
            "process": run["process"], "planHashBefore": training["planHash"],
            "planFileSha256Before": _hash(plan_path), "checkpointBeforeInterruption": checkpoint,
            "completedBefore": completed, "completedFoldsPreserved": len(completed),
            "interruptedAfterCheckpoint": False, "resumedFromCheckpoint": False,
            "planHashUnchanged": False, "completedReceiptsUnchanged": False,
            "allFoldsCompleted": False, "trainingResumeAccepted": False,
            "predictorResumeAccepted": False, "complete": False}
        if skipped_candidates:
            evidence["skippedCandidates"] = skipped_candidates
        save(signalDispatchStartedAt=_now())
        # Durable intent precedes the signal. An uncertain dispatch is never
        # repeated after a coordinator restart; later state must prove failure.
        if not _signal_owned_leader(run, folder):
            save(phase="signal_not_dispatched", signalDeliveryConfirmedAbsent=True)
            return False
        save(phase="interrupted", signalDispatchedAt=_now(), interruptedAfterCheckpoint=True)
        return True

    if evidence.get("complete"):
        return False
    if _hash(plan_path) != evidence["planFileSha256Before"] or training["planHash"] != evidence["planHashBefore"]:
        raise RuntimeError("The frozen training plan changed during the recovery exercise")
    _verify_completed(evidence["completedBefore"], training["runs"], folder)
    current_runs = {run["id"]: run for run in training["runs"]}
    selected = current_runs[evidence["runId"]]
    unexpected = [run for run in training["runs"] if run["id"] != selected["id"]
                  and run["status"] in {"failed", "cancelled", "interrupted"}]
    if unexpected or training["status"] in {"cancelled", "interrupted"}:
        raise RuntimeError("An unplanned training failure occurred during the one-fold exercise")

    # Reconcile a lost acknowledgement before choosing the next action. Native
    # operation receipts are authoritative and make repeated calls idempotent.
    operations_path = folder / "operations.json"
    operations = _read(operations_path) if operations_path.exists() else {}
    if operations.get(TRAINING_OPERATION) == "resume" and not evidence["trainingResumeAccepted"]:
        save(trainingResumeAccepted=True, phase="training_resumed")
    if evidence["trainingResumeAccepted"] and not evidence["predictorResumeAccepted"]:
        _, native = predictors._read(experiment_id)
        operations = native.get("operations", {})
        if operations.get(PREDICTOR_OPERATION) is True:
            save(predictorResumeAccepted=True, predictorResumedAttempt=native["attempt"])

    if not evidence["trainingResumeAccepted"]:
        if training["status"] == "completed":
            raise RuntimeError("The selected fold completed without the intended interruption")
        if training["status"] != "failed":
            return True
        if selected["status"] != "failed":
            raise RuntimeError("Batch failed without the selected fold interruption")
        fresh = training_service.execution(batch_id)
        if fresh["status"] != "failed" or not _training_stopped(training_service, fresh):
            return True
        checkpoint = _probe_checkpoint(_read(plan_path)["runtime"]["python"],
                                       folder / "runs" / selected["id"] / "last.ckpt")
        if checkpoint is None:
            raise RuntimeError("Interrupted fold has no valid optimizer/RNG checkpoint for recovery")
        save(checkpointAtResume=checkpoint, observedFailedAttempt=True,
             interruptedAfterCheckpoint=True, trainingResumeRequestedAt=_now())
        resumed = training_service.launch(batch_id, TRAINING_OPERATION, resume=True)
        if resumed["planHash"] != evidence["planHashBefore"]:
            raise RuntimeError("Native resume changed the execution plan")
        save(trainingResumeAccepted=True, phase="training_resumed")
        return True

    if training["status"] == "failed":
        raise RuntimeError("The resumed training attempt failed unexpectedly")
    if predictor_work["status"] == "cancelled":
        raise RuntimeError("Predictor coordination was cancelled and cannot be resumed by this exercise")
    if predictor_work["status"] in {"attention", "interrupted"}:
        if evidence["predictorResumeAccepted"]:
            raise RuntimeError("The resumed predictor coordinator requires unexpected recovery")
        if not _predictor_stopped(predictors, experiment_id):
            return True
        save(predictorResumeRequestedAt=_now())
        resumed = predictors.launch(experiment_id, PREDICTOR_OPERATION, resume=True)
        if resumed["status"] not in {"queued", "running", "waiting", "completed"}:
            raise RuntimeError("Predictor resume was not accepted")
        save(predictorResumeAccepted=True)
        return True

    if training["status"] == "completed" and predictor_work["status"] == "completed":
        receipt = _read(folder / "runs" / selected["id"] / "result.json")
        expected = str(folder / "runs" / selected["id"] / "last.ckpt")
        if (receipt.get("resumedFrom") != expected or selected.get("attempt") != evidence["attemptBefore"] + 1
                or receipt.get("resumePolicy") != "replay_interrupted_epoch_from_last_completed_epoch"):
            raise RuntimeError("Completed fold does not prove native checkpoint recovery")
        if not all(run["status"] == "completed" for run in training["runs"]):
            raise RuntimeError("Completion status does not cover every fold")
        save(phase="complete", complete=True, resumedFromCheckpoint=True, planHashUnchanged=True,
             completedReceiptsUnchanged=True, allFoldsCompleted=True, finishedAt=_now(),
             resumedAttempt=selected["attempt"], totalFolds=len(training["runs"]))
        return False
    return True
