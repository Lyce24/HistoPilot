"""Read distribution metadata without importing optional compute libraries."""

import platform
import sqlite3
from importlib.metadata import PackageNotFoundError, version


def system_report() -> dict:
    packages = {}
    for name in (
        "histopilot",
        "fastapi",
        "sqlalchemy",
        "torch",
        "trident",
        "openslide-python",
        "nvidia-ml-py",
    ):
        try:
            packages[name] = {"installed": True, "version": version(name)}
        except PackageNotFoundError:
            packages[name] = {"installed": False, "version": None}
    return {
        "python": platform.python_version(),
        "platform": platform.system(),
        "sqlite": sqlite3.sqlite_version,
        "packages": packages,
        "compute": {
            "enabled": False,
            "scope": "control-service",
            "cuda": "not probed",
            "gpus": "not probed",
        },
        "note": (
            "This report inspects control-service package metadata only. Isolated workers "
            "implement extraction, packing, ABMIL training, evaluation and attention. "
            "Check runtime readiness in each module; package presence here does not establish "
            "worker dependencies, GPU availability or checkpoint access."
        ),
    }
