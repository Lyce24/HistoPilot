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
        "compute": {"enabled": False, "cuda": "not probed", "gpus": "not probed"},
        "note": "Package presence is not backend readiness. GPU, weight, and model-access checks are future worker capabilities.",
    }
