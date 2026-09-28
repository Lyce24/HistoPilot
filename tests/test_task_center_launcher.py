"""Runner launcher and CLI: never touch real tmux or the user's state directory."""

import json
import os
import shlex
import signal
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from typer.testing import CliRunner

from histopilot.cli import app
from histopilot.taskcenter import TaskStore, launcher, paths
from histopilot.taskcenter import runner as runner_module
from histopilot.taskcenter.runner import Runner
from histopilot.workers.training_process import process_identity


def fake_host():
    return {"cpuCount": 8, "totalRamGb": 32.0, "availableRamGb": 16.0, "gpus": []}


@pytest.fixture
def tmux(monkeypatch):
    calls = []

    def run(command, **kwargs):
        if command[1] == "has-session":
            return subprocess.CompletedProcess(command, 0 if tmux.session else 1, "", "")
        calls.append((command, kwargs))
        failure = tmux.failure.pop(0) if isinstance(tmux.failure, list) else tmux.failure
        if failure is not None:
            raise failure
        return subprocess.CompletedProcess(command, 0, "", "")

    tmux.failure = None
    tmux.session = False
    tmux.calls = calls
    monkeypatch.setattr(launcher.shutil, "which", lambda name: "/usr/bin/tmux")
    monkeypatch.setattr(launcher.subprocess, "run", run)
    return tmux


def test_autostart_is_disabled_in_tests(tmux):
    assert os.environ[paths.AUTOSTART_ENV] == "0"
    assert launcher.ensure_runner() == {"started": False, "reason": "disabled"}
    assert tmux.calls == []


def test_ensure_runner_starts_one_tmux_session(tmux, monkeypatch):
    monkeypatch.setenv(paths.AUTOSTART_ENV, "1")
    state = paths.state_dir()
    assert launcher.ensure_runner(python="/opt/python") == {
        "started": True,
        "session": f"hp-runner-{os.getuid()}",
    }
    [(command, options)] = tmux.calls
    assert command[:5] == ["/usr/bin/tmux", "new-session", "-d", "-s", f"hp-runner-{os.getuid()}"]
    script = command[5]
    root = launcher._repository_root()
    assert script.startswith(f"cd {root} && env PYTHONUNBUFFERED=1 ")
    assert f"HISTOPILOT_STATE_DIR={state} " in script
    assert "TMPDIR=" in script
    assert "/opt/python -u -m histopilot.cli runner run" in script
    assert script.endswith(f">> {state / 'runner.log'} 2>&1")
    assert options["check"] is True and options["timeout"] == 15

    duplicate = subprocess.CalledProcessError(
        1, command, stderr="duplicate session: hp-runner-1000"
    )
    tmux.failure = duplicate
    assert launcher.ensure_runner() == {"started": True, "note": "already starting"}
    # A session left by a runner that released its lock and exited: start again once it is gone.
    tmux.calls.clear()
    tmux.failure = [duplicate, None]
    assert launcher.ensure_runner() == {"started": True, "session": f"hp-runner-{os.getuid()}"}
    assert len(tmux.calls) == 2
    # A session that stays without taking the lock is reported as starting, not restarted.
    tmux.calls.clear()
    tmux.failure, tmux.session = duplicate, True
    monkeypatch.setattr(launcher, "SESSION_WAIT_SECONDS", 0.0)
    assert launcher.ensure_runner() == {"started": True, "note": "already starting"}
    assert len(tmux.calls) == 1
    tmux.session = False
    tmux.failure = subprocess.CalledProcessError(1, command, stderr="no server running")
    assert launcher.ensure_runner()["reason"] == "tmux failed: no server running"
    tmux.failure = subprocess.TimeoutExpired(command, 15)
    assert launcher.ensure_runner()["started"] is False
    monkeypatch.setattr(launcher.shutil, "which", lambda name: None)
    assert launcher.ensure_runner() == {"started": False, "reason": "tmux unavailable"}


def test_live_runner_is_detected_through_its_lock(tmux, monkeypatch):
    monkeypatch.setenv(paths.AUTOSTART_ENV, "1")
    assert launcher.runner_alive() is False
    runner = Runner(TaskStore(), host_probe=fake_host, log=lambda message: None)
    runner.start()
    try:
        assert launcher.runner_alive() is True
        assert launcher.ensure_runner() == {"started": False, "alive": True}
        assert tmux.calls == []
    finally:
        runner.close()
    assert launcher.runner_alive() is False


HOLD_LOCK = """
import fcntl, os, sys, time
descriptor = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o600)
fcntl.flock(descriptor, fcntl.LOCK_EX)
print("locked", flush=True)
time.sleep(60)
"""


def test_stop_runner_signals_only_a_verified_runner_holding_the_lock(tmux):
    assert launcher.stop_runner() == {"stopped": False, "reason": "no runner recorded"}
    store = TaskStore()
    me = process_identity()
    store.write_runner(pid=me["pid"], start_ticks=me["startTicks"], boot_id=me["bootId"])
    # The recorded process is alive, but no runner holds the lock: nothing is signalled.
    assert launcher.stop_runner() == {"stopped": False, "reason": "not running"}
    lock = paths.state_dir() / "runner.lock"
    process = subprocess.Popen(
        [sys.executable, "-c", HOLD_LOCK, str(lock)], stdout=subprocess.PIPE, text=True
    )
    try:
        assert process.stdout.readline().strip() == "locked"
        store.write_runner(pid=999_999, start_ticks=1, boot_id="an-earlier-boot")
        assert launcher.stop_runner() == {"stopped": False, "reason": "not running"}
        identity = process_identity(process.pid)
        store.write_runner(
            pid=identity["pid"], start_ticks=identity["startTicks"], boot_id=identity["bootId"]
        )
        assert launcher.stop_runner(wait=5) == {
            "stopped": True,
            "pid": process.pid,
            "exited": True,
        }
        assert process.wait(timeout=5) == -15
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()


def test_runner_cli_run_status_start_and_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_module, "_LOADED_AT", float("inf"))  # edits by others
    state = tmp_path / "cli-state"
    cli = CliRunner()
    monkeypatch.setattr("histopilot.taskcenter.capacity.gpu_snapshot", lambda: {"gpus": []})
    result = cli.invoke(app, ["runner", "run", "--state-dir", str(state), "--once"])
    assert result.exit_code == 0, result.output
    assert os.environ[paths.STATE_ENV] == str(state.resolve())
    assert (state / "task-center.sqlite").exists()
    row = TaskStore().runner()
    assert row["pid"] == os.getpid() and row["protocol"] == 1

    result = cli.invoke(app, ["runner", "status", "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["alive"] is False and report["codeCurrent"] is True
    assert report["stateDir"] == str(state.resolve())
    assert report["counts"]["queued"] == 0 and report["paused"] is False
    result = cli.invoke(app, ["runner", "status"])
    assert "Runner     stopped" in result.output and "Tasks      none" in result.output

    result = cli.invoke(app, ["runner", "start"])
    assert json.loads(result.output) == {"started": False, "reason": "disabled"}
    result = cli.invoke(app, ["runner", "stop"])
    assert json.loads(result.output)["stopped"] is False

    holder = Runner(TaskStore(), host_probe=fake_host, log=lambda message: None)
    holder.start()
    monkeypatch.setattr(Runner, "lock_wait", 0.05)
    try:
        result = cli.invoke(app, ["runner", "run", "--once"])
        assert result.exit_code == 0 and "Nothing to do" in result.output
        assert json.loads(cli.invoke(app, ["runner", "status", "--json"]).output)["alive"] is True
    finally:
        holder.close()


def test_serve_starts_the_runner_unless_disabled(tmp_path, monkeypatch):
    config = tmp_path / "config.toml"
    config.write_text('[storage]\nworkspace = "state"\ndata_roots = []\n')
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: None)
    monkeypatch.setenv("HISTOPILOT_SERVICE_SETTINGS", "{}")
    calls = []

    def ensure_runner(**kwargs):
        calls.append(kwargs)
        return {"started": False, "reason": "tmux unavailable"}

    monkeypatch.setattr(launcher, "ensure_runner", ensure_runner)
    arguments = ["serve", "--config", str(config), "--dev", "--no-browser"]
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 0, result.output
    assert calls == [{}]
    assert "Task Center runner not started: tmux unavailable" in result.output
    result = CliRunner().invoke(app, [*arguments, "--no-runner"])
    assert result.exit_code == 0, result.output
    assert calls == [{}]

    def broken(**kwargs):
        raise RuntimeError("state directory is unsafe")

    monkeypatch.setattr(launcher, "ensure_runner", broken)
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 0, result.output
    assert "Task Center runner not started: state directory is unsafe" in result.output


def env_assignments(script: str) -> tuple[dict, list[str]]:
    """The NAME=value pairs of the runner command's ``env`` and the command after them."""
    words = shlex.split(script.split(" && ", 1)[1].split(" >> ", 1)[0])
    assert words[0] == "env"
    pairs, index = {}, 1
    while "=" in words[index] and not words[index].startswith("/"):
        name, value = words[index].split("=", 1)
        pairs[name] = value
        index += 1
    return pairs, words[index:]


def test_ensure_runner_forwards_the_starting_environment(tmux, monkeypatch):
    monkeypatch.setenv(paths.AUTOSTART_ENV, "1")
    monkeypatch.setenv("HISTOPILOT_TRAINING_PYTHON", "/opt/train env/bin/python")
    monkeypatch.setenv("HISTOPILOT_TRIDENT_ROOT", "it's here; $(rm -rf ~)")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/lib/wsl/lib:/opt/cuda/lib64")
    monkeypatch.setenv("PATH", "/opt/tools/bin:/usr/bin")
    monkeypatch.setenv("HISTOPILOT_TASK_ID", "task-1")  # set inside task processes
    monkeypatch.setenv("HISTOPILOT_TASK_MANAGED", "1")
    state = paths.state_dir()
    launcher.ensure_runner(python="/opt/python")
    [(command, _)] = tmux.calls
    pairs, rest = env_assignments(command[5])
    assert pairs["HISTOPILOT_TRAINING_PYTHON"] == "/opt/train env/bin/python"
    assert pairs["HISTOPILOT_TRIDENT_ROOT"] == "it's here; $(rm -rf ~)"
    assert pairs["LD_LIBRARY_PATH"] == "/usr/lib/wsl/lib:/opt/cuda/lib64"
    assert pairs["PATH"] == "/opt/tools/bin:/usr/bin"
    assert pairs[paths.STATE_ENV] == str(state) and pairs["PYTHONUNBUFFERED"] == "1"
    for name in (paths.AUTOSTART_ENV, "HISTOPILOT_TASK_ID", "HISTOPILOT_TASK_MANAGED"):
        assert name not in pairs
    assert rest == ["/opt/python", "-u", "-m", "histopilot.cli", "runner", "run"]
    monkeypatch.delenv("LD_LIBRARY_PATH")
    tmux.calls.clear()
    launcher.ensure_runner(python="/opt/python")
    assert "LD_LIBRARY_PATH" not in env_assignments(tmux.calls[0][0][5])[0]


def test_restart_runner_waits_for_the_old_runner(monkeypatch):
    calls = []
    monkeypatch.setattr(
        launcher, "ensure_runner", lambda: calls.append("ensure") or {"started": True}
    )
    stopped = {"stopped": True, "pid": 7, "exited": False}
    monkeypatch.setattr(launcher, "stop_runner", lambda wait: calls.append(wait) or dict(stopped))
    assert launcher.restart_runner() == {"started": False, "reason": "disabled"}
    monkeypatch.setenv(paths.AUTOSTART_ENV, "1")
    result = launcher.restart_runner(wait=3)
    assert result["pending"] is True and result["started"] is False and calls == [3]
    stopped["exited"] = True
    assert launcher.restart_runner(wait=3) == {"started": True, "stopped": stopped}
    assert calls == [3, 3, "ensure"]


def test_serve_restarts_a_runner_whose_code_changed(tmp_path, monkeypatch):
    config = tmp_path / "config.toml"
    config.write_text('[storage]\nworkspace = "state"\ndata_roots = []\n')
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: None)
    monkeypatch.setenv("HISTOPILOT_SERVICE_SETTINGS", "{}")
    monkeypatch.setenv(paths.AUTOSTART_ENV, "1")
    monkeypatch.setattr(
        launcher, "ensure_runner", lambda **kwargs: {"started": False, "alive": True}
    )
    restarts = []

    def restart_runner(wait):
        restarts.append(wait)
        return outcome

    monkeypatch.setattr(launcher, "restart_runner", restart_runner)
    arguments = ["serve", "--config", str(config), "--dev", "--no-browser"]
    source = tmp_path / "module.py"
    source.write_text("x = 1\n")
    store = TaskStore()
    store.write_runner(code_files=[str(source)], code_hash=runner_module.code_hash([str(source)]))
    outcome = {"started": True, "session": "hp-runner"}
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 0, result.output
    assert restarts == [] and "Task Center runner is running" in result.output

    source.write_text("x = 2\n")  # the runner's code changed on disk
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 0, result.output
    assert len(restarts) == 1
    assert "Task Center runner restarted because its code changed" in result.output

    waited = []
    monkeypatch.setattr(
        launcher, "await_runner_exit", lambda timeout: waited.append(timeout) or True
    )
    outcome = {"started": False, "pending": True, "note": "finishing"}
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 0, result.output
    assert "restarts once its current step finishes" in result.output
    for thread in threading.enumerate():
        if thread.name == "runner-restart":
            thread.join(timeout=5)
    assert waited == [600.0]

    # A runner that cannot be stopped keeps its old code; that is not a restart.
    outcome = {"started": False, "alive": True, "stopped": {"stopped": False, "reason": "denied"}}
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 0, result.output
    assert "restarted because" not in result.output
    assert "code changed but it could not be restarted: denied" in result.output

    monkeypatch.setenv(paths.AUTOSTART_ENV, "0")
    restarts.clear()
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 0 and restarts == []


def test_a_runner_started_from_another_checkout_is_never_current(tmp_path, monkeypatch):
    """The state directory is per OS user, so every checkout shares one runner."""
    config = tmp_path / "config.toml"
    config.write_text('[storage]\nworkspace = "state"\ndata_roots = []\n')
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: None)
    monkeypatch.setenv("HISTOPILOT_SERVICE_SETTINGS", "{}")
    monkeypatch.setenv(paths.AUTOSTART_ENV, "1")
    monkeypatch.setattr(
        launcher, "ensure_runner", lambda **kwargs: {"started": False, "alive": True}
    )
    restarts = []
    monkeypatch.setattr(
        launcher, "restart_runner", lambda wait: restarts.append(wait) or {"started": True}
    )
    source = tmp_path / "module.py"
    source.write_text("x = 1\n")
    store = TaskStore()
    store.write_runner(
        code_files=[str(source)],
        code_hash=runner_module.code_hash([str(source)]),
        checkout_root=paths.checkout_root(),
    )
    assert runner_module.code_current(store.runner())
    assert runner_module.foreign_checkout(store.runner()) is None
    other = str(tmp_path / "another-checkout")
    store.write_runner(checkout_root=other)
    row = store.runner()
    assert row["checkoutRoot"] == other and not runner_module.code_current(row)
    assert runner_module.foreign_checkout(row) == other
    arguments = ["serve", "--config", str(config), "--dev", "--no-browser"]
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 0, result.output
    assert len(restarts) == 1
    assert f"started from another checkout ({other})" in result.output


def test_a_runner_row_without_its_checkout_is_placed_by_its_code_files(tmp_path):
    """Runners from before ``checkoutRoot`` was recorded still reveal their checkout."""
    here = paths.checkout_root()
    own = {"codeFiles": [str(Path(here, "histopilot", "taskcenter", "runner.py"))]}
    assert runner_module.foreign_checkout(own) is None
    other = tmp_path / "histopilot" / "HistoPilot-main"  # the checkout's own name may repeat it
    row = {"codeFiles": [str(other / "histopilot" / "taskcenter" / "runner.py")]}
    assert runner_module.foreign_checkout(row) == str(other)
    assert not runner_module.code_current({**row, "codeHash": "anything"})
    # Code files outside any histopilot package (tests) name no checkout.
    assert runner_module.foreign_checkout({"codeFiles": [str(tmp_path / "module.py")]}) is None


def test_a_started_runner_records_its_checkout(tmp_path):
    runner = Runner(TaskStore(), host_probe=fake_host, log=lambda message: None)
    runner.start()
    try:
        assert TaskStore().runner()["checkoutRoot"] == paths.checkout_root()
        assert Path(paths.checkout_root(), "histopilot", "taskcenter", "runner.py").is_file()
    finally:
        runner.shutdown()


def test_a_stop_signal_during_runner_startup_ends_it_gracefully(tmp_path, monkeypatch):
    monkeypatch.setattr("histopilot.taskcenter.capacity.gpu_snapshot", lambda: {"gpus": []})
    fallback = []
    previous = signal.signal(signal.SIGTERM, lambda *_: fallback.append("SIGTERM"))
    started = Runner.start

    def start(self, **options):
        os.kill(os.getpid(), signal.SIGTERM)  # e.g. `histopilot runner stop` during startup
        started(self, **options)
        ticks.append(self._stopping())

    ticks = []
    monkeypatch.setattr(Runner, "start", start)
    monkeypatch.setattr(Runner, "tick", lambda self: ticks.append("tick"))
    try:
        result = CliRunner().invoke(app, ["runner", "run", "--state-dir", str(tmp_path / "s")])
    finally:
        signal.signal(signal.SIGTERM, previous)
    assert result.exit_code == 0, result.output
    assert fallback == []  # the runner's own handler received it
    assert ticks == [True]  # startup saw the stop; the loop never ticked
    assert TaskStore().runner()["state"] == "stopped"
    assert launcher.runner_alive() is False
