"""GET /api/v1/metrics: Prometheus text that parses, aggregates from the Task Center store,
GPUs only when nvidia-smi answers, and the service's access rules."""

import re
import subprocess
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from support.cli import Service

from histopilot.api import audit, create_app, login
from histopilot.application import system_compute
from histopilot.config import Settings
from histopilot.observability.metrics import Exposition, HttpMetrics
from histopilot.taskcenter.store import TaskStore

BASE = "http://127.0.0.1:8787"
METRICS = "/api/v1/metrics"
SAMPLE = re.compile(r"([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})? (\S+)")
LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"(,|$)')
SUFFIXES = {"counter": ("",), "gauge": ("",), "histogram": ("_bucket", "_sum", "_count")}


def parse(text: str) -> dict:
    """{family: {"type", "help", "samples": [(name, labels, value)]}}, checking the text
    format's rules on the way: HELP then TYPE once per family, its samples right after
    them, well-formed labels, and histograms whose buckets add up."""
    assert text.endswith("\n")
    families, current = {}, None
    for line in text.splitlines():
        if line.startswith("# HELP "):
            name, _, summary = line.removeprefix("# HELP ").partition(" ")
            assert name not in families and summary
            families[name] = {"help": summary, "type": None, "samples": []}
            current = name
        elif line.startswith("# TYPE "):
            name, kind = line.removeprefix("# TYPE ").split(" ")
            assert name == current and families[name]["type"] is None
            families[name]["type"] = kind
        else:
            match = SAMPLE.fullmatch(line)
            assert match, line
            name, rendered, value = match.groups()
            kind = families[current]["type"]
            assert name in {current + suffix for suffix in SUFFIXES[kind]}, line
            labels = {}
            if rendered:
                pairs = LABEL.findall(rendered)
                assert "".join(f'{k}="{v}"{c}' for k, v, c in pairs) == rendered, line
                labels = {key: item for key, item, _ in pairs}
            families[current]["samples"].append((name, labels, float(value)))
    for name, family in families.items():
        if family["type"] == "counter":
            assert name.endswith("_total")
        if family["type"] == "histogram":
            check_histogram(name, family["samples"])
    return families


def check_histogram(name, samples):
    series = {}
    for sample, labels, value in samples:
        key = tuple(sorted((k, v) for k, v in labels.items() if k != "le"))
        entry = series.setdefault(key, {"buckets": [], "sum": None, "count": None})
        if sample == f"{name}_bucket":
            entry["buckets"].append((float(labels["le"]), value))
        else:
            entry[sample.removeprefix(f"{name}_")] = value
    for entry in series.values():
        bounds = [bound for bound, _ in entry["buckets"]]
        counts = [count for _, count in entry["buckets"]]
        assert bounds == sorted(bounds) and bounds[-1] == float("inf")
        assert counts == sorted(counts) and counts[-1] == entry["count"]
        assert entry["sum"] is not None


def value(families, metric, /, **labels):
    family = metric.removesuffix("_bucket").removesuffix("_sum").removesuffix("_count")
    found = [
        number
        for sample, have, number in families[family]["samples"]
        if sample == metric and have == labels
    ]
    assert len(found) == 1, (metric, labels, families[family]["samples"])
    return found[0]


@pytest.fixture(autouse=True)
def no_gpus(monkeypatch):
    """No test reaches the machine's nvidia-smi: neither PATH nor WSL's copy has one."""

    def absent(*_args, **_kwargs):
        raise AssertionError("nvidia-smi must not run in tests")

    monkeypatch.setattr(system_compute.shutil, "which", lambda _name: None)
    monkeypatch.setattr(system_compute, "Path", lambda *_: SimpleNamespace(is_file=lambda: False))
    monkeypatch.setattr(
        system_compute,
        "subprocess",
        SimpleNamespace(run=absent, TimeoutExpired=subprocess.TimeoutExpired),
    )


@pytest.fixture
def http(tmp_path):
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<!doctype html>")
    settings = Settings(workspace=tmp_path / "workspace", static_dir=static)
    with TestClient(create_app(settings), base_url=BASE) as client:
        yield client


def scrape(http, **headers) -> dict:
    response = http.get(METRICS, headers=headers)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "text/plain; version=0.0.4; charset=utf-8"
    return parse(response.text)


# -- the format ----------------------------------------------------------------------------


def test_the_exposition_escapes_labels_and_refuses_malformed_families():
    out = Exposition()
    out.gauge("demo", "Help with a \\ and\na newline.", [({"name": 'a "b"\\c\nd'}, 1.5)])
    out.counter("demo_total", "Counted.", [({}, 3)])
    out.histogram("demo_seconds", "Timed.", (0.5, 1.0), [({"kind": "x"}, 3, 2.5, [1, 2])])
    text = out.render()
    assert 'demo{name="a \\"b\\"\\\\c\\nd"} 1.5' in text
    assert "# HELP demo Help with a \\\\ and\\na newline." in text
    families = parse(text)
    assert value(families, "demo_seconds_bucket", kind="x", le="+Inf") == 3
    assert value(families, "demo_seconds_bucket", kind="x", le="0.5") == 1
    with pytest.raises(ValueError):
        out.gauge("demo", "Again.", [])
    with pytest.raises(ValueError):
        out.counter("demo_count", "A counter must end in _total.", [])
    with pytest.raises(ValueError):
        out.gauge("bad name", "Invalid.", [])


def test_request_metrics_count_by_route_template():
    metrics = HttpMetrics(buckets=(0.1, 1.0))
    metrics.observe("GET", "/api/v1/projects", 200, 0.05)
    metrics.observe("GET", "/api/v1/projects", 200, 0.5)
    metrics.observe("BREW", "/{path:path}", 405, 2.0)
    listed, odd = metrics.snapshot()
    assert listed[0] == {"method": "GET", "route": "/api/v1/projects", "status": "200"}
    assert listed[1] == 2 and listed[2] == pytest.approx(0.55) and listed[3] == [1, 2]
    assert odd == ({"method": "OTHER", "route": "/{path:path}", "status": "405"}, 1, 2.0, [0, 0])


# -- what the service reports --------------------------------------------------------------


def owner(tmp_path, identity):
    return {
        "kind": "experiment",
        "id": identity,
        "projectId": f"project-{identity}",
        "projectFolder": str(tmp_path / "projects" / identity),
        "title": f"Experiment {identity}",
    }


def spec(tmp_path, identity, *, adapter="generic", depends=None):
    return {
        "id": identity,
        "kind": adapter,
        "adapter": adapter,
        "title": f"Task {identity}",
        "request": {"lane": "cpu"},
        "command": {
            "argv": [sys.executable, "-c", "pass"],
            "cwd": str(tmp_path),
            "log": str(tmp_path / "logs" / f"{identity}.log"),
        },
        **({"dependsOn": [{"task": depends}]} if depends else {}),
    }


def test_the_queue_and_its_history_come_from_the_store(task_center, tmp_path, http):
    store: TaskStore = task_center.store
    client = task_center.client
    client.enqueue(
        owner(tmp_path, "done"),
        [spec(tmp_path, "ok"), spec(tmp_path, "oom"), spec(tmp_path, "crash", adapter="mil-fold")],
    )
    held = client.enqueue(
        owner(tmp_path, "held"), [spec(tmp_path, "h1"), spec(tmp_path, "h2", depends="h1")]
    )
    store.hold_owner(held["owner"]["key"], True)
    client.enqueue(owner(tmp_path, "live"), [spec(tmp_path, "waits"), spec(tmp_path, "runs")])
    store.set_waiting_reasons({"waits": "Waiting for RAM"})
    task_center.start("runs")
    task_center.finish("ok")
    task_center.finish("oom", "failed", returncode=1, reason="oom")
    task_center.finish("crash", "failed", returncode=1, reason="error")
    for task_id, seconds in (("ok", 42.0), ("oom", 3.0), ("crash", 7200.0)):
        store.record_measurement({"taskId": task_id, "kind": "generic", "wallSeconds": seconds})

    families = scrape(http)
    assert value(families, "histopilot_task_store_up") == 1
    states = {
        labels["state"]: count for _, labels, count in families["histopilot_tasks"]["samples"]
    }
    assert states == {
        "blocked": 1,
        "queued": 2,
        "starting": 0,
        "running": 1,
        "stopping": 0,
        "succeeded": 1,
        "failed": 2,
        "cancelled": 0,
        "interrupted": 0,
    }
    assert value(families, "histopilot_tasks_held") == 2
    assert value(families, "histopilot_tasks_waiting") == 1
    assert value(families, "histopilot_queue_paused") == 0
    finished = "histopilot_tasks_finished_total"
    assert value(families, finished, adapter="generic", state="succeeded") == 1
    assert value(families, finished, adapter="generic", state="failed") == 1
    assert value(families, finished, adapter="mil-fold", state="failed") == 1
    failures = "histopilot_task_failures_total"
    assert value(families, failures, adapter="generic", reason="oom") == 1
    assert value(families, failures, adapter="mil-fold", reason="error") == 1
    duration = "histopilot_task_duration_seconds"
    assert value(families, f"{duration}_count", adapter="generic") == 2
    assert value(families, f"{duration}_sum", adapter="generic") == 45.0
    assert value(families, f"{duration}_bucket", adapter="generic", le="5.0") == 1
    assert value(families, f"{duration}_bucket", adapter="generic", le="60.0") == 2
    assert value(families, f"{duration}_bucket", adapter="mil-fold", le="3600.0") == 0
    assert value(families, f"{duration}_bucket", adapter="mil-fold", le="7200.0") == 1
    assert value(families, "histopilot_runner_up") == 0
    assert "histopilot_runner_heartbeat_age_seconds" not in families

    # A retry moves the task back to the queue; what already happened stays counted.
    store.requeue(["oom"], reason="retry")
    store.update_settings({"paused": True})
    store.write_runner(heartbeat_at="2000-01-01T00:00:00+00:00", state="stopped")
    families = scrape(http)
    assert value(families, failures, adapter="generic", reason="oom") == 1
    assert value(families, "histopilot_tasks", state="failed") == 1
    assert value(families, "histopilot_queue_paused") == 1
    assert value(families, "histopilot_runner_heartbeat_age_seconds") > 86400


def test_metrics_never_name_a_project_task_or_path(task_center, tmp_path, http):
    task_center.client.enqueue(owner(tmp_path, "secret-owner"), [spec(tmp_path, "secret-task")])
    task_center.finish("secret-task", "failed", returncode=1, reason="Bad Reason /tmp/x")
    session = http.get("/api/v1/session").json()["token"]
    http.get("/api/v1/projects/secret-project/datasets", headers={"X-HistoPilot-Token": session})
    text = http.get(METRICS).text
    for leaked in ("secret-owner", "secret-task", "secret-project", str(tmp_path), "Bad Reason"):
        assert leaked not in text
    families = parse(text)
    # A reason that is not a code-defined name is reported as "other".
    assert value(families, "histopilot_task_failures_total", adapter="generic", reason="other") == 1
    assert (
        value(
            families,
            "histopilot_http_requests_total",
            method="GET",
            route="/api/v1/projects/{identity}/datasets",
            status="404",
        )
        == 1
    )


def test_request_metrics_and_build_info(http):
    http.get("/api/v1/health")
    http.get("/api/v1/health")
    http.get("/api/v1/projects")  # refused: no session token
    families = scrape(http)
    requests = "histopilot_http_requests_total"
    assert value(families, requests, method="GET", route="/api/v1/health", status="200") == 2
    assert value(families, requests, method="GET", route="/api/v1/projects", status="401") == 1
    timed = "histopilot_http_request_duration_seconds_count"
    assert value(families, timed, method="GET", route="/api/v1/health", status="200") == 2
    [(_, labels, one)] = families["histopilot_build_info"]["samples"]
    assert one == 1 and labels["version"] and labels["revision"]
    assert value(families, "histopilot_service_start_time_seconds") > 1.6e9


def test_gpus_are_left_out_without_nvidia_smi(http):
    families = scrape(http)
    assert not [name for name in families if name.startswith("histopilot_gpu_")]


def test_each_gpu_reports_utilization_and_memory(http, monkeypatch):
    monkeypatch.setattr(system_compute.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    rows = [
        "0, Test GPU A, GPU-a, 555.0, 75, 24576, 6144, 18432, 60, 120.5, 300.0",
        "1, Test GPU B, GPU-b, 555.0, [N/A], 16384, 0, 16384, [N/A], [N/A], [N/A]",
    ]
    calls = []

    def run(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="\n".join(rows), stderr="")

    monkeypatch.setattr(system_compute.subprocess, "run", run)  # this module's stand-in
    families = scrape(http)
    scrape(http)  # within the cache window: nvidia-smi is not asked again
    assert len(calls) == 1
    assert value(families, "histopilot_gpu_info", gpu="0", name="Test GPU A") == 1
    assert value(families, "histopilot_gpu_utilization_ratio", gpu="0") == 0.75
    assert value(families, "histopilot_gpu_memory_used_bytes", gpu="0") == 6144 * 2**20
    assert value(families, "histopilot_gpu_memory_total_bytes", gpu="1") == 16384 * 2**20
    # An unknown reading is left out, never reported as zero.
    assert [
        labels["gpu"] for _, labels, _ in families["histopilot_gpu_utilization_ratio"]["samples"]
    ] == ["0"]


def test_an_unreadable_store_leaves_the_other_metrics(http, monkeypatch):
    def unreadable(self, _bounds):
        raise OSError("disk gone")

    monkeypatch.setattr(TaskStore, "statistics", unreadable)
    families = scrape(http)
    assert value(families, "histopilot_task_store_up") == 0
    assert "histopilot_tasks" not in families and "histopilot_build_info" in families


# -- who may read them ---------------------------------------------------------------------


def test_a_scraper_on_loopback_needs_no_session_and_agents_never_read_metrics(
    tmp_path, monkeypatch
):
    with Service(tmp_path, monkeypatch) as service:
        http = service.http
        session = http.headers.pop("X-HistoPilot-Token")
        assert http.get(METRICS).status_code == 200
        assert http.get(METRICS, headers={"X-HistoPilot-Token": session}).status_code == 200
        wrong = http.get(METRICS, headers={"X-HistoPilot-Token": "not-the-session"})
        assert wrong.status_code == 401 and wrong.json()["code"] == "SESSION_TOKEN_REQUIRED"
        crossed = http.get(METRICS, headers={"Sec-Fetch-Site": "cross-site"})
        assert crossed.status_code == 403 and crossed.json()["code"] == "CROSS_SITE_REFUSED"
        assert http.get(METRICS, headers={"Host": "metrics.example"}).status_code == 400

        http.headers["X-HistoPilot-Token"] = session
        project = service.create_project("Shared")
        exposed = http.put(
            f"/api/v1/projects/{project}/ai-exposure",
            json={"level": "full", "expectedLevel": "none"},
        )
        assert exposed.status_code == 200, exposed.text
        token = http.post(
            "/api/v1/tokens", json={"projectId": project, "scopes": ["read"], "name": "agent"}
        ).json()
        agent = http.get(
            METRICS, headers={"X-HistoPilot-Token": token["token"], "X-Request-ID": "agent-1"}
        )
        assert agent.status_code == 403 and agent.json()["code"] == "SCOPE_MISSING"
        [entry] = [
            row
            for row in audit.read(service.settings.workspace / "projects" / "Shared")
            if row.get("route") == METRICS
        ]
        assert entry["actor"]["kind"] == "agent" and entry["status"] == 403
        # The audit line names the request, as the service's own access line does.
        assert entry["requestId"] == "agent-1"


def test_with_sign_in_on_a_scraper_presents_the_secret(tmp_path):
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<!doctype html>")
    settings = Settings(workspace=tmp_path / "workspace", static_dir=static, login=True)
    with TestClient(create_app(settings), base_url=BASE) as http:
        secret = login.read_secret(8787)
        refused = http.get(METRICS)
        assert refused.status_code == 401 and refused.json()["code"] == "LOGIN_REQUIRED"
        wrong = http.get(METRICS, headers={"Authorization": "Bearer not-the-secret"})
        assert wrong.status_code == 401
        assert http.get(METRICS, headers={"Authorization": f"Bearer {secret}"}).status_code == 200
        assert http.get(METRICS, headers={"X-HistoPilot-Login": secret}).status_code == 200
        token = http.get("/api/v1/session", headers={"X-HistoPilot-Login": secret}).json()["token"]
        assert http.get(METRICS, headers={"X-HistoPilot-Token": token}).status_code == 200
