"""The Prometheus text exposition format (0.0.4), written by hand, and the service's own
request metrics.

Every value is an aggregate. Label values are task states, adapter and exit-reason names,
methods, route templates, status codes and GPU indices: never a project, record, slide,
path or token.
"""

import math
import re
import threading

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
# Request durations in seconds: a poll answers in milliseconds, a large project's first
# listing can take seconds.
HTTP_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
METHODS = frozenset({"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"})
_NAME = re.compile(r"[a-zA-Z_:][a-zA-Z0-9_:]*")
_LABEL = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
_WORD = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}")


def word(value, fallback: str = "other") -> str:
    """A name the code defines, such as an adapter or an exit reason, as a label value;
    ``fallback`` for anything else, so stored text never widens a label or names a record."""
    return value if isinstance(value, str) and _WORD.fullmatch(value) else fallback


def _number(value) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    value = float(value)
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    return repr(value)


def _escape(value: str, *, quoted=True) -> str:
    value = value.replace("\\", "\\\\").replace("\n", "\\n")
    return value.replace('"', '\\"') if quoted else value


class Exposition:
    """Metric families in the text format. Each family is written once, with its HELP and
    TYPE lines before its samples."""

    def __init__(self):
        self._lines: list[str] = []
        self._names: set[str] = set()

    def _family(self, name: str, kind: str, summary: str) -> None:
        if not _NAME.fullmatch(name) or name in self._names:
            raise ValueError(f"Invalid or repeated metric family: {name}")
        self._names.add(name)
        self._lines.append(f"# HELP {name} {_escape(summary, quoted=False)}")
        self._lines.append(f"# TYPE {name} {kind}")

    def _sample(self, name: str, labels: dict, value) -> None:
        if any(not _LABEL.fullmatch(key) for key in labels):
            raise ValueError(f"Invalid label name in {name}: {sorted(labels)}")
        pairs = ",".join(f'{key}="{_escape(str(item))}"' for key, item in labels.items())
        series = f"{name}{{{pairs}}}" if pairs else name
        self._lines.append(f"{series} {_number(value)}")

    def gauge(self, name: str, summary: str, samples) -> None:
        """``samples``: (labels, value) pairs."""
        self._family(name, "gauge", summary)
        for labels, value in samples:
            self._sample(name, labels, value)

    def counter(self, name: str, summary: str, samples) -> None:
        if not name.endswith("_total"):
            raise ValueError(f"A counter's name ends in _total: {name}")
        self._family(name, "counter", summary)
        for labels, value in samples:
            self._sample(name, labels, value)

    def histogram(self, name: str, summary: str, bounds, series) -> None:
        """``series``: (labels, count, sum, cumulative counts at ``bounds``)."""
        self._family(name, "histogram", summary)
        for labels, count, total, cumulative in series:
            for bound, observed in zip(bounds, cumulative, strict=True):
                self._sample(f"{name}_bucket", {**labels, "le": _number(float(bound))}, observed)
            self._sample(f"{name}_bucket", {**labels, "le": "+Inf"}, count)
            self._sample(f"{name}_sum", labels, total)
            self._sample(f"{name}_count", labels, count)

    def render(self) -> str:
        return "\n".join(self._lines) + "\n"


class HttpMetrics:
    """Requests this service process answered, by method, route template and status.

    Held in memory: they start again from zero when the service restarts, which
    Prometheus reads as a counter reset."""

    def __init__(self, buckets=HTTP_BUCKETS):
        self.buckets = tuple(float(bound) for bound in buckets)
        self._lock = threading.Lock()
        # (method, route, status) -> [count, seconds, cumulative counts per bucket]
        self._series: dict[tuple[str, str, str], list] = {}

    def observe(self, method: str, route: str, status: int, seconds: float) -> None:
        key = (method if method in METHODS else "OTHER", route, str(status))
        with self._lock:
            entry = self._series.get(key)
            if entry is None:
                entry = self._series[key] = [0, 0.0, [0] * len(self.buckets)]
            entry[0] += 1
            entry[1] += seconds
            for index, bound in enumerate(self.buckets):
                if seconds <= bound:
                    entry[2][index] += 1

    def snapshot(self) -> list[tuple[dict, int, float, list[int]]]:
        """(labels, count, seconds, cumulative bucket counts) per series, in a stable order."""
        with self._lock:
            return [
                ({"method": method, "route": route, "status": status}, count, total, list(counts))
                for (method, route, status), (count, total, counts) in sorted(self._series.items())
            ]
