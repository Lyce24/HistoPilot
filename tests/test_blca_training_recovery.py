"""Exercise recovery orchestration with mocked process boundaries; never signal workers."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import blca_training_recovery as helper


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def recovery(tmp_path, monkeypatch):
    project = tmp_path / "blca-e2e-20260917"
    folder = project / "training/configuration-test"
    write(folder / "plan.json", {"runtime": {"python": "/captured/python"}})
    runs = []
    for identity, status in (("done", "completed"), ("active", "running"), ("later", "queued")):
        directory = folder / "runs" / identity
        directory.mkdir(parents=True)
        (directory / "best.ckpt").write_bytes(b"best")
        (directory / "last.ckpt").write_bytes(b"resumable")
        write(directory / "result.json", {"bestCheckpointPath": str(directory / "best.ckpt"),
                                         "lastCheckpointPath": str(directory / "last.ckpt")})
        runs.append({"id": identity, "status": status, "attempt": 1,
                     "process": {"pid": 9001, "startTicks": 3, "bootId": "test"} if identity == "active" else None})
    training = {"batchId": "configuration-test", "status": "running", "planHash": "immutable",
                "runs": runs, "sessionName": "hp-test", "process": None}
    calls, signals = [], []
    work = {"status": "running"}

    class Training:
        def execution(self, _):
            return copy.deepcopy(training)

        def launch(self, identity, operation, *, resume):
            calls.append(("training", identity, operation, resume))
            training["status"] = "running"
            training["runs"][1].update(status="running", attempt=2)
            write(folder / "operations.json", {operation: "resume"})
            return copy.deepcopy(training)

    class Predictors:
        def __init__(self):
            self.native = {"operations": {}, "attempt": 1}

        def _read(self, _):
            return {}, self.native

        def launch(self, identity, operation, *, resume):
            calls.append(("predictor", identity, operation, resume))
            work["status"] = "running"
            self.native["operations"][operation] = resume
            self.native["attempt"] += 1
            return copy.deepcopy(work)

    predictors = Predictors()
    study = SimpleNamespace(exercise_recovery=True, path=tmp_path / "state.json",
        state={"projectPath": str(project), "projectId": "project-test", "batchId": "configuration-test",
               "experimentId": "experiment-test"},
        experiments=SimpleNamespace(training=Training(), _predictors=lambda: predictors))
    monkeypatch.setattr(helper, "_owned_leader", lambda *args: True)
    monkeypatch.setattr(helper, "_probe_checkpoint", lambda _python, path: {
        "epoch": 1, "globalStep": 20, "path": str(path), "sha256": helper._hash(path),
        "optimizerStatePresent": True, "rngStatePresent": True})
    monkeypatch.setattr(helper, "_training_stopped", lambda *args: True)
    monkeypatch.setattr(helper, "_predictor_stopped", lambda *args: True)
    monkeypatch.setattr(helper, "_open_pidfd", lambda _: helper.os.open("/dev/null", helper.os.O_RDONLY))
    # Standalone Python builds may omit this optional Linux API. These tests
    # mock the process boundary and must never depend on real signal delivery.
    monkeypatch.setattr(helper.signal, "pidfd_send_signal",
                        lambda fd, sig, info, flags: signals.append((fd, sig)), raising=False)
    return study, training, work, folder, calls, signals


def test_full_native_resume_sequence_preserves_completed_fold_and_signals_once(recovery):
    study, training, work, folder, calls, signals = recovery
    assert helper.advance(study, training, work)
    assert len(signals) == 1
    assert helper.advance(study, training, work)
    assert len(signals) == 1
    training["status"] = "failed"
    training["runs"][1]["status"] = "failed"
    training["runs"][2]["status"] = "completed"
    work["status"] = "attention"
    assert helper.advance(study, training, work)
    assert calls == [("training", "configuration-test", helper.TRAINING_OPERATION, True)]
    assert helper.advance(study, training, work)
    assert calls[-1] == ("predictor", "experiment-test", helper.PREDICTOR_OPERATION, True)
    training["status"] = "completed"
    training["runs"][1]["status"] = "completed"
    work["status"] = "completed"
    write(folder / "runs/active/result.json", {
        "resumedFrom": str(folder / "runs/active/last.ckpt"),
        "resumePolicy": "replay_interrupted_epoch_from_last_completed_epoch"})
    assert helper.advance(study, training, work) is False
    report = helper._read(study.path.parent / "evidence/interruption-resume.json")
    assert all(report[key] for key in ("complete", "interruptedAfterCheckpoint", "resumedFromCheckpoint",
                                      "planHashUnchanged", "completedReceiptsUnchanged", "allFoldsCompleted"))
    assert report["completedFoldsPreserved"] == 1 and report["totalFolds"] == 3
    assert helper.advance(study, training, work) is False
    assert len(signals) == 1 and len(calls) == 2


def test_opt_in_checkpoint_and_ownership_are_required(recovery, monkeypatch):
    study, training, work, _, _, signals = recovery
    study.exercise_recovery = False
    assert helper.advance(study, training, work) is False
    study.exercise_recovery = True
    monkeypatch.setattr(helper, "_probe_checkpoint", lambda *args: None)
    assert helper.advance(study, training, work) is False
    assert not signals
    monkeypatch.setattr(helper, "_owned_leader", lambda *args: False)
    assert helper.advance(study, training, work) is False
    assert not signals


@pytest.mark.parametrize("change", ["plan", "completed_receipt", "unexpected_failure"])
def test_scientific_changes_and_unplanned_failures_stop_recovery(recovery, change):
    study, training, work, folder, calls, signals = recovery
    helper.advance(study, training, work)
    if change == "plan":
        write(folder / "plan.json", {"changed": True})
    elif change == "completed_receipt":
        write(folder / "runs/done/result.json", {"changed": True})
    else:
        training["runs"][2]["status"] = "failed"
    with pytest.raises(RuntimeError):
        helper.advance(study, training, work)
    assert len(signals) == 1 and not calls


def test_waits_for_predictor_process_exit_before_spending_resume_operation(recovery, monkeypatch):
    study, training, work, _, calls, _ = recovery
    helper.advance(study, training, work)
    training["status"] = "failed"
    training["runs"][1]["status"] = "failed"
    helper.advance(study, training, work)
    work["status"] = "attention"
    monkeypatch.setattr(helper, "_predictor_stopped", lambda *args: False)
    assert helper.advance(study, training, work)
    assert len(calls) == 1


def test_recorded_signal_intent_is_never_repeated(recovery):
    study, training, work, _, _, signals = recovery
    helper.advance(study, training, work)
    path = study.path.parent / "evidence/interruption-resume.json"
    evidence = helper._read(path)
    evidence["phase"] = "signal_intent"
    evidence["interruptedAfterCheckpoint"] = False
    write(path, evidence)
    assert helper.advance(study, training, work)
    assert len(signals) == 1


def test_checkpoint_probe_checks_snapshot_even_if_later_epoch_replaces_source(tmp_path, monkeypatch):
    path = tmp_path / "last.ckpt"
    path.write_bytes(b"checkpoint")
    payload = {"epoch": 2, "globalStep": 4, "optimizerStatePresent": True, "rngStatePresent": True}
    monkeypatch.setattr(helper.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=0, stdout=json.dumps(payload)))
    original = helper._probe_checkpoint("python", path)
    assert original["epoch"] == 2

    def changed(*args, **kwargs):
        path.write_bytes(b"replacement checkpoint")
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload))

    monkeypatch.setattr(helper.subprocess, "run", changed)
    observed = helper._probe_checkpoint("python", path)
    assert observed["sha256"] == original["sha256"]
    assert observed["stableSnapshotVerified"]


def test_owned_process_requires_exact_plan_and_private_process_group(tmp_path, monkeypatch):
    run = {"id": "fold-one", "process": {"pid": 12345, "startTicks": 1, "bootId": "test"}}
    monkeypatch.setattr(helper, "confirmed_process_alive", lambda _: True)
    monkeypatch.setattr(helper, "process_identity", lambda _: run["process"])
    monkeypatch.setattr(helper.os, "getpgid", lambda _: 12345)
    command = [b"python", b"--fold", str(tmp_path / "runs/fold-one/plan.json").encode()]
    monkeypatch.setattr(Path, "read_bytes", lambda _: b"\0".join(command))
    assert helper._owned_leader(run, tmp_path)
    command[-1] = b"/another/study/plan.json"
    assert not helper._owned_leader(run, tmp_path)


@pytest.mark.parametrize("boundary", ["identity", "group", "command"])
def test_disappearing_fold_is_an_unavailable_candidate_not_coordinator_failure(tmp_path, monkeypatch, boundary):
    run = {"id": "fold-one", "process": {"pid": 12345, "startTicks": 1, "bootId": "test"}}
    monkeypatch.setattr(helper, "confirmed_process_alive", lambda _: True)
    monkeypatch.setattr(helper, "process_identity", lambda _: run["process"])
    monkeypatch.setattr(helper.os, "getpgid", lambda _: 12345)

    def gone(*args):
        raise ProcessLookupError("process exited")

    if boundary == "identity":
        monkeypatch.setattr(helper, "process_identity", gone)
    elif boundary == "group":
        monkeypatch.setattr(helper.os, "getpgid", gone)
    else:
        monkeypatch.setattr(Path, "read_bytes", gone)
    assert helper._owned_leader(run, tmp_path) is False


def test_pidfd_exit_records_no_delivery_and_allows_next_candidate(recovery, monkeypatch):
    study, training, work, _, calls, signals = recovery

    def gone(*args):
        raise ProcessLookupError("fold exited before delivery")

    monkeypatch.setattr(helper.signal, "pidfd_send_signal", gone)
    assert helper.advance(study, training, work) is False
    evidence = helper._read(study.path.parent / "evidence/interruption-resume.json")
    assert evidence["phase"] == "signal_not_dispatched" and not evidence["interruptedAfterCheckpoint"]
    monkeypatch.setattr(helper.signal, "pidfd_send_signal", lambda *args: signals.append(args))
    assert helper.advance(study, training, work)
    assert len(signals) == 1 and not calls
    evidence = helper._read(study.path.parent / "evidence/interruption-resume.json")
    assert len(evidence["skippedCandidates"]) == 1


def test_pidfd_is_not_signalled_if_identity_changes_after_open(recovery, monkeypatch):
    _, training, _, folder, _, signals = recovery
    monkeypatch.setattr(helper, "_owned_leader", lambda *args: False)
    assert helper._signal_owned_leader(training["runs"][1], folder) is False
    assert not signals


def test_probe_timeout_is_transient(tmp_path, monkeypatch):
    path = tmp_path / "last.ckpt"
    path.write_bytes(b"checkpoint")

    def timeout(*args, **kwargs):
        raise helper.subprocess.TimeoutExpired("checkpoint probe", 30)

    monkeypatch.setattr(helper.subprocess, "run", timeout)
    assert helper._probe_checkpoint("python", path) is None


def test_lost_predictor_ack_is_reconciled_from_native_operation_receipt(recovery):
    study, training, work, _, calls, _ = recovery
    helper.advance(study, training, work)
    training["status"] = "failed"
    training["runs"][1]["status"] = "failed"
    helper.advance(study, training, work)
    work["status"] = "attention"
    helper.advance(study, training, work)
    path = study.path.parent / "evidence/interruption-resume.json"
    evidence = helper._read(path)
    evidence["predictorResumeAccepted"] = False  # native resume committed before coordinator acknowledgement
    write(path, evidence)
    assert helper.advance(study, training, work)
    assert helper._read(path)["predictorResumeAccepted"]
    assert helper._read(path)["predictorResumedAttempt"] == 2
    assert len(calls) == 2
