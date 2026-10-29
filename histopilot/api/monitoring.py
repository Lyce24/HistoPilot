"""Request IDs, access lines and the Prometheus metrics route.

Every request gets an ID: a well-formed ``X-Request-ID`` from the client, else a new one.
It is echoed on the response, stamps every record logged while the request is served
(and the tasks it queues), and closes the request with one access line: the method, the
route template, the status, the duration and the kind of credential presented. Never a
path's values, the query, headers, cookies or the body.

``GET /api/v1/metrics`` answers in the Prometheus text format, from the Task Center store
at scrape time (so the values survive restarts), from nvidia-smi (cached for a few
seconds), and from this process's own request counts. Who may read it is decided in
``histopilot/api/security.py``.
"""

import logging
import threading
import time

from fastapi import APIRouter
from fastapi.responses import Response

from histopilot import __version__
from histopilot.api.route_classes import TOKEN_PREFIX, route_template
from histopilot.observability import logs
from histopilot.observability.metrics import CONTENT_TYPE, Exposition, HttpMetrics, word
from histopilot.taskcenter.model import STATES, parse_iso

LOG = logging.getLogger(__name__)
ACCESS = logging.getLogger("histopilot.api.access")
HEADER = b"x-request-id"
# Run time of a finished task attempt, in seconds: a result collection takes seconds, a
# training fold or an extraction hours.
TASK_BUCKETS = (
    1.0,
    5.0,
    15.0,
    30.0,
    60.0,
    120.0,
    300.0,
    600.0,
    1800.0,
    3600.0,
    7200.0,
    14400.0,
    43200.0,
    86400.0,
)
GPU_SECONDS = 5.0


def route_of(scope) -> str:
    """The template of the route that served the request, never its concrete path."""
    path = getattr(scope.get("route"), "path", None)
    if isinstance(path, str):
        return path
    # Refused before routing (Host, Origin, session or scope checks): the route table
    # still names an API route.
    return route_template(scope.get("method", ""), scope.get("path", "")) or "unmatched"


def credential(headers) -> str:
    """The kind of credential a request presented, never the credential itself."""
    for name, value in headers:
        if name == b"x-histopilot-token" and value:
            return "token" if value.startswith(TOKEN_PREFIX.encode()) else "session"
    return "none"


class RequestLog:
    """ASGI middleware: the request's ID, its access line and its request metrics."""

    def __init__(self, app, *, metrics: HttpMetrics):
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = scope.get("headers") or []
        offered = [value for name, value in headers if name == HEADER]
        identity = None
        if len(offered) == 1:
            identity = logs.valid_request_id(offered[0].decode("latin-1"))
        identity = identity or logs.new_request_id()
        # Left set for the rest of the request's task: when an error escapes the app, the
        # server's own report of it is made there and carries the ID too.
        logs.REQUEST_ID.set(identity)
        echoed = (HEADER, identity.encode("ascii"))
        status = 500
        started = time.perf_counter()

        async def answer(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message = {**message, "headers": [*message.get("headers", ()), echoed]}
            await send(message)

        try:
            await self.app(scope, receive, answer)
        finally:
            seconds = time.perf_counter() - started
            method, route = scope.get("method", ""), route_of(scope)
            self.metrics.observe(method, route, status, seconds)
            ACCESS.info(
                "%s %s %s",
                method,
                route,
                status,
                extra={
                    "method": method,
                    "route": route,
                    "status": status,
                    "duration_ms": round(seconds * 1000, 1),
                    "auth": credential(headers),
                },
            )


class GpuCounters:
    """nvidia-smi's per-GPU counters, probed at most every few seconds and by one request
    at a time; none when the tool is missing or fails."""

    def __init__(self, seconds: float = GPU_SECONDS):
        self.seconds = seconds
        self._lock = threading.Lock()
        self._read: tuple[float, list] | None = None

    def devices(self) -> list[dict]:
        from histopilot.application import system_compute

        with self._lock:
            if self._read is not None and time.monotonic() - self._read[0] < self.seconds:
                return self._read[1]
            reading = system_compute.read_gpus()
            devices = reading["devices"] if reading.get("status") == "available" else []
            self._read = (time.monotonic(), devices)
            return devices


def _task_metrics(out: Exposition, store, alive) -> None:
    statistics = store.statistics(TASK_BUCKETS)
    out.gauge(
        "histopilot_tasks",
        "Tasks in the Task Center store by state.",
        [({"state": state}, statistics["states"].get(state, 0)) for state in STATES],
    )
    out.gauge(
        "histopilot_tasks_held",
        "Blocked or queued tasks whose owner is on hold.",
        [({}, statistics["held"])],
    )
    out.gauge(
        "histopilot_tasks_waiting",
        "Queued tasks the runner left waiting: for capacity, a related task, a busy output "
        "or a fenced GPU. Held owners' tasks count as held instead.",
        [({}, statistics["waiting"])],
    )
    out.gauge(
        "histopilot_queue_paused",
        "1 while the queue is paused: running tasks go on, none start.",
        [({}, statistics["paused"])],
    )
    finished: dict[tuple[str, str], int] = {}
    failures: dict[tuple[str, str], int] = {}
    for row in statistics["outcomes"]:
        adapter = word(row["adapter"])
        key = (adapter, row["state"])
        finished[key] = finished.get(key, 0) + row["count"]
        if row["state"] == "failed":
            # A task that failed before it started (its adapter or command was refused)
            # records no exit reason; its exit says "error".
            reason = (adapter, word(row["reason"] or "error"))
            failures[reason] = failures.get(reason, 0) + row["count"]
    out.counter(
        "histopilot_tasks_finished_total",
        "Task attempts that ended, by adapter and how they ended, from the store's history.",
        [({"adapter": a, "state": s}, n) for (a, s), n in sorted(finished.items())],
    )
    out.counter(
        "histopilot_task_failures_total",
        "Task attempts that failed, by adapter and exit reason (error, oom, cuda_failure, …).",
        [({"adapter": a, "reason": r}, n) for (a, r), n in sorted(failures.items())],
    )
    durations: dict[str, list] = {}
    for row in statistics["durations"]:
        entry = durations.setdefault(word(row["adapter"]), [0, 0.0, [0] * len(TASK_BUCKETS)])
        entry[0] += row["count"]
        entry[1] += row["seconds"]
        entry[2] = [have + more for have, more in zip(entry[2], row["buckets"], strict=True)]
    out.histogram(
        "histopilot_task_duration_seconds",
        "Run time of each task attempt that ended, by adapter, from the store's measurements.",
        TASK_BUCKETS,
        [({"adapter": a}, n, total, counts) for a, (n, total, counts) in sorted(durations.items())],
    )
    out.gauge(
        "histopilot_runner_up",
        "1 while a Task Center runner holds the runner lock.",
        [({}, alive)],
    )
    heartbeat = parse_iso(statistics["heartbeatAt"])
    if heartbeat is not None:
        out.gauge(
            "histopilot_runner_heartbeat_age_seconds",
            "Seconds since the runner last wrote its heartbeat (every 2 s while it runs).",
            [({}, max(0.0, time.time() - heartbeat.timestamp()))],
        )


def _gpu_metrics(out: Exposition, devices: list[dict]) -> None:
    if not devices:
        return
    labelled = [({"gpu": str(device["index"])}, device) for device in devices]
    out.gauge(
        "histopilot_gpu_info",
        "Each GPU nvidia-smi reports, with its model name.",
        [({**labels, "name": device["name"]}, 1) for labels, device in labelled],
    )
    out.gauge(
        "histopilot_gpu_utilization_ratio",
        "GPU busy time over nvidia-smi's last sample period, from 0 to 1.",
        [
            (labels, device["utilizationPercent"] / 100)
            for labels, device in labelled
            if device.get("utilizationPercent") is not None
        ],
    )
    out.gauge(
        "histopilot_gpu_memory_used_bytes",
        "GPU memory in use, by any process.",
        [
            (labels, device["memoryUsedBytes"])
            for labels, device in labelled
            if device.get("memoryUsedBytes") is not None
        ],
    )
    out.gauge(
        "histopilot_gpu_memory_total_bytes",
        "GPU memory in total.",
        [
            (labels, device["memoryTotalBytes"])
            for labels, device in labelled
            if device.get("memoryTotalBytes") is not None
        ],
    )


def render(center, http: HttpMetrics, gpus: GpuCounters, *, revision, started) -> str:
    from histopilot.taskcenter import launcher

    out = Exposition()
    tasks = Exposition()
    try:
        store = center.store
        _task_metrics(tasks, store, launcher.runner_alive(store.path.parent / "runner.lock"))
    except Exception as error:  # noqa: BLE001 - an unreadable store leaves the other metrics
        LOG.warning("The Task Center store could not be read for metrics: %s", error)
        tasks = None
    out.gauge(
        "histopilot_task_store_up",
        "1 when the Task Center store could be read for this scrape.",
        [({}, tasks is not None)],
    )
    _gpu_metrics(out, gpus.devices())
    series = http.snapshot()
    out.counter(
        "histopilot_http_requests_total",
        "Requests this service process answered, by method, route template and status.",
        [(labels, count) for labels, count, _seconds, _buckets in series],
    )
    out.histogram(
        "histopilot_http_request_duration_seconds",
        "Time to answer a request, by method, route template and status.",
        http.buckets,
        [(labels, count, seconds, buckets) for labels, count, seconds, buckets in series],
    )
    dropped = logs.dropped()
    if dropped:
        out.counter(
            "histopilot_log_records_dropped_total",
            "Log records a sink dropped because it fell behind, by sink (file or stderr).",
            [({"sink": sink}, count) for sink, count in sorted(dropped.items())],
        )
    out.gauge(
        "histopilot_build_info",
        "The code this service runs: package version and git revision.",
        [({"version": __version__, "revision": revision or "unknown"}, 1)],
    )
    out.gauge(
        "histopilot_service_start_time_seconds",
        "When this service process started, in seconds since the Unix epoch.",
        [({}, started)],
    )
    return out.render() + (tasks.render() if tasks is not None else "")


def monitoring_router(center, http: HttpMetrics, *, revision: str | None) -> APIRouter:
    router = APIRouter(prefix="/api/v1")
    gpus = GpuCounters()
    started = time.time()

    @router.get("/metrics")
    def metrics():
        """Aggregate service, queue, GPU and request metrics in the Prometheus text format."""
        content = render(center, http, gpus, revision=revision, started=started)
        return Response(content, media_type=CONTENT_TYPE)

    return router
