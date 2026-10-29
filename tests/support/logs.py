"""Structured records of HistoPilot's loggers, as the JSON formatter writes them.

``collect(name)`` attaches a handler that stamps each record in the thread that logs it,
as the service's own handlers do, so request and task IDs read as they would in the log
files. Records of child loggers arrive too.
"""

import json
import logging

import pytest

from histopilot.observability import logs


class Collector(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []
        self.addFilter(logs.stamp)
        self.setFormatter(logs.JsonFormatter("test"))

    def emit(self, record):
        self.lines.append(json.loads(self.format(record)))


@pytest.fixture
def collect():
    """``collect(name)``: the records of logger ``name`` from now on, at INFO and above."""
    attached = []

    def start(name: str) -> list[dict]:
        handler, logger = Collector(), logging.getLogger(name)
        attached.append((logger, handler, logger.level))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return handler.lines

    yield start
    for logger, handler, level in attached:
        logger.removeHandler(handler)
        logger.setLevel(level)
