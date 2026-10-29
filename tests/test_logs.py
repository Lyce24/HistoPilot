"""Structured logs: the formats, request and task IDs on every record, outputs that never
block, and how the service, the runner and workers route their records."""

import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from support.logs import collect as collect
from support.task_center import fake_host
from typer.testing import CliRunner

from histopilot.config import Settings
from histopilot.observability import logs
from histopilot.taskcenter import capacity

ROOT = Path(__file__).resolve().parents[1]
TIMESTAMP = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z")


# -- formats -------------------------------------------------------------------------------


def test_json_lines_name_the_request_and_task_they_were_made_in(collect):
    lines = collect("histopilot.test.json")
    log = logging.getLogger("histopilot.test.json")
    with logs.bound(task="task-1", request="request-1"):
        try:
            raise ValueError("bad input")
        except ValueError:
            log.exception("fold %s failed", "f1", extra={"adapter": "mil-fold", "msg2": 1})
        log.info("forged", extra={"ts": "yesterday", "level": "DEBUG"})
    log.info("outside")
    failed, forged, outside = lines
    assert TIMESTAMP.fullmatch(failed["ts"])
    assert failed["level"] == "ERROR" and failed["logger"] == "histopilot.test.json"
    assert failed["msg"] == "fold f1 failed" and failed["component"] == "test"
    assert failed["request_id"] == "request-1" and failed["task_id"] == "task-1"
    assert failed["adapter"] == "mil-fold" and failed["msg2"] == 1
    assert "Traceback" in failed["exc"] and "ValueError: bad input" in failed["exc"]
    # A record's own fields never replace the standard ones.
    assert TIMESTAMP.fullmatch(forged["ts"]) and forged["level"] == "INFO"
    assert outside["request_id"] is None and outside["task_id"] is None


def test_text_lines_are_one_readable_line_per_record():
    record = logging.LogRecord(
        "histopilot.test", logging.INFO, __file__, 1, "%s ready", ("fold",), None
    )
    record.task_id, record.request_id = "task-1", None
    record.reason, record.note = "ok", "two words"
    line = logs.TextFormatter().format(record)
    assert re.fullmatch(
        r"\S+Z INFO histopilot\.test: fold ready task_id=task-1 reason=ok note=\"two words\"",
        line,
    )


@pytest.mark.parametrize(
    ("offered", "kept"),
    [
        ("4f2c9a1e-7b7d-4c55-9a51-0f0c4b2f9e1d", True),
        ("trace.span:01-x_y", True),
        ("a" * 128, True),
        ("a" * 129, False),
        ("two words", False),
        ("line\nbreak", False),
        ('quote"', False),
        ("", False),
        (None, False),
    ],
)
def test_clients_choose_request_ids_only_from_a_safe_alphabet(offered, kept):
    assert logs.valid_request_id(offered) == (offered if kept else None)


# -- outputs -------------------------------------------------------------------------------


def test_the_service_writes_json_lines_and_only_warnings_to_stderr(tmp_path, capsys):
    target = tmp_path / "logs" / "service-8787.jsonl"
    target.parent.mkdir()
    routing = logs.configure("service", file=target, stderr_warnings=True)
    try:
        log = logging.getLogger("histopilot.test.outputs")
        with logs.bound(request="request-2"):
            log.info("routine")
            log.warning("look at this")
    finally:
        routing.close()
    written = [json.loads(line) for line in target.read_text().splitlines()]
    assert [(line["msg"], line["request_id"]) for line in written] == [
        ("routine", "request-2"),
        ("look at this", "request-2"),
    ]
    assert all(line["component"] == "service" for line in written)
    shown = capsys.readouterr().err
    assert "look at this" in shown and "request_id=request-2" in shown
    assert "routine" not in shown
    # Closing undoes the routing.
    assert routing.handler not in logging.getLogger().handlers and logs.dropped() == {}


def test_stderr_carries_everything_as_json_when_a_container_asks(capsys, monkeypatch):
    monkeypatch.setenv(logs.STDERR_ENV, "1")
    monkeypatch.setenv(logs.FORMAT_ENV, "json")
    routing = logs.configure("runner")
    try:
        logging.getLogger("histopilot.test.container").info("routine", extra={"pid": 7})
    finally:
        routing.close()
    [line] = [json.loads(text) for text in capsys.readouterr().err.splitlines()]
    assert line["msg"] == "routine" and line["component"] == "runner" and line["pid"] == 7


def test_a_terminal_nobody_reads_never_holds_up_the_thread_that_logs(tmp_path, monkeypatch):
    """The service once hung writing its access log to a terminal nobody drained."""
    release = threading.Event()

    class Stuck:
        def write(self, text):
            release.wait(30)
            return len(text)

        def flush(self):
            pass

    monkeypatch.setattr(logs, "QUEUE_RECORDS", 5)
    monkeypatch.setenv(logs.STDERR_ENV, "1")
    monkeypatch.setattr(sys, "stderr", Stuck())
    target = tmp_path / "service.jsonl"
    routing = logs.configure("service", file=target)
    try:
        log = logging.getLogger("histopilot.test.stuck")
        started = time.monotonic()
        for index in range(50):
            log.info("record %d", index)
        assert time.monotonic() - started < 5
        # The stuck writer holds one record and its queue five; the rest are dropped.
        assert routing.dropped()["stderr"] >= 44
    finally:
        release.set()
        routing.close()
    first = json.loads(target.read_text().splitlines()[0])
    assert first["msg"] == "record 0"


# -- workers -------------------------------------------------------------------------------

WORKER = """
import logging, threading
from histopilot.observability.logs import configure_worker
configure_worker()
log = logging.getLogger("histopilot.workers.test")
log.info("step %d", 1)
thread = threading.Thread(target=lambda: log.warning("from a thread"))
thread.start(); thread.join()
logging.getLogger("other.library").info("library chatter")
# A library with a console handler of its own, as Lightning has.
console = logging.StreamHandler()
console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
library = logging.getLogger("console.library")
library.addHandler(console)
library.setLevel(logging.INFO)
library.info("library line")
"""


def run_worker(**environment):
    variables = {
        name: value
        for name, value in os.environ.items()
        if name not in (logs.TASK_ENV, logs.REQUEST_ENV, logs.FORMAT_ENV, logs.LEVEL_ENV)
    }
    completed = subprocess.run(
        [sys.executable, "-c", WORKER],
        cwd=ROOT,
        env={**variables, **environment},
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return completed.stderr.splitlines()


def test_a_worker_logs_to_its_task_log_with_the_ids_the_runner_gave_it():
    lines = run_worker(HISTOPILOT_TASK_ID="task-7", HISTOPILOT_REQUEST_ID="request-7")
    # Other libraries print as before: once, and at their own, quieter level.
    assert len(lines) == 3 and lines[2] == "INFO: library line", lines
    for line, message in zip(lines[:2], ("step 1", "from a thread"), strict=True):
        assert f"histopilot.workers.test: {message}" in line
        assert "task_id=task-7" in line and "request_id=request-7" in line
    json_lines = run_worker(
        HISTOPILOT_TASK_ID="task-8",
        HISTOPILOT_REQUEST_ID="not a request id",
        HISTOPILOT_LOG_FORMAT="json",
    )
    records = [json.loads(line) for line in json_lines if line.startswith("{")]
    assert [record["msg"] for record in records] == ["step 1", "from a thread"]
    assert {(record["task_id"], record["request_id"]) for record in records} == {("task-8", None)}
    assert {record["component"] for record in records} == {"worker"}


@pytest.mark.parametrize(
    "module",
    [
        "workers/managed_fold.py",
        "workers/managed_collect.py",
        "workers/compute_job.py",
        "workers/experiment_predictors.py",
        "workers/pack_features.py",
        "workers/verify_extraction.py",
        "workers/portability.py",
        "taskcenter/jobs.py",
    ],
)
def test_every_worker_entry_point_routes_its_logs(module):
    source = (ROOT / "histopilot" / module).read_text(encoding="utf-8")
    entry = source.split('if __name__ == "__main__":', 1)[1]
    assert "configure_worker()" in entry


# -- the service and the runner ------------------------------------------------------------


def test_serve_routes_uvicorn_through_the_service_logs(tmp_path, monkeypatch):
    from histopilot.cli import app

    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<!doctype html>")
    calls = []

    def run(application, **options):
        calls.append(options)
        logging.getLogger("uvicorn.error").info(
            "Application startup complete.", extra={"color_message": "\x1b[32mstartup\x1b[0m"}
        )

    monkeypatch.setattr("uvicorn.run", run)
    monkeypatch.setattr(
        "histopilot.cli.load_settings",
        lambda *args, **kwargs: Settings(
            workspace=tmp_path / "workspace", static_dir=static, port=8794
        ),
    )
    result = CliRunner().invoke(app, ["serve", "--no-browser", "--no-runner"])
    assert result.exit_code == 0, result.output
    [options] = calls
    # Uvicorn writes nothing itself: no console handlers, no access log of its own.
    assert options["log_config"] is None and options["access_log"] is False
    target = Path(os.environ["HISTOPILOT_STATE_DIR"]) / "logs" / "service-8794.jsonl"
    records = [json.loads(text) for text in target.read_text().splitlines()]
    [line] = [record for record in records if record["logger"].startswith("uvicorn")]
    assert line["logger"] == "uvicorn.error" and line["msg"] == "Application startup complete."
    assert "color_message" not in line  # uvicorn's copy with terminal colour codes
    # The routing ends with the service; nothing stays attached to this process.
    assert logs.dropped() == {}


def test_runner_run_writes_its_records_to_runner_jsonl(monkeypatch):
    from histopilot.cli import app
    from histopilot.taskcenter.runner import Runner

    monkeypatch.setattr(capacity, "gpu_snapshot", lambda: {"gpus": []})
    monkeypatch.setattr(capacity, "host_snapshot", lambda: fake_host()())
    tick = Runner.tick

    def logged_tick(self):
        logging.getLogger("histopilot.taskcenter.runner").info("one tick")
        return tick(self)

    monkeypatch.setattr(Runner, "tick", logged_tick)
    result = CliRunner().invoke(app, ["runner", "run", "--once"])
    assert result.exit_code == 0, result.output
    target = Path(os.environ["HISTOPILOT_STATE_DIR"]) / "logs" / "runner.jsonl"
    records = [json.loads(text) for text in target.read_text().splitlines()]
    assert "one tick" in [record["msg"] for record in records]
    assert {record["component"] for record in records} == {"runner"}
    assert logs.dropped() == {}
