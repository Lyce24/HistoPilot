"""Discover an optional compute interpreter without importing Torch into the API."""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from threading import Lock

from histopilot.workers.training_process import gpu_snapshot, host_snapshot

_CACHE = {}
_LOCK = Lock()
_PROBE = """
import json, sys, platform, importlib.metadata as m
import torch, lightning, h5py, pyarrow, sklearn
print(json.dumps({'available': True, 'python': sys.executable,
 'pythonVersion': sys.version, 'platform': platform.platform(), 'cudaVersion': torch.version.cuda,
 'versions': {k:m.version(k) for k in ['torch','lightning','torchmetrics','numpy','h5py','pyarrow','scikit-learn']},
 'cudaAvailable': torch.cuda.is_available(), 'gpuCount': torch.cuda.device_count(),
 'gpuNames': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}))
"""


def training_python() -> str:
    configured = os.environ.get("HISTOPILOT_TRAINING_PYTHON")
    if configured:
        return str(Path(configured).expanduser().absolute())
    checkout = Path(__file__).resolve().parents[3]
    candidate = checkout / ".venv-training" / "bin" / "python"
    return str(candidate) if candidate.is_file() else sys.executable


def training_runtime(*, refresh=False) -> dict:
    executable = training_python()
    with _LOCK:
        prior = _CACHE.get(executable)
        if not refresh and prior and time.monotonic() - prior[0] < 30:
            return prior[1]
        result = {
            "available": False,
            "python": executable,
            "versions": {},
            "cudaAvailable": False,
            "gpuCount": 0,
            "findings": [],
            "host": host_snapshot(),
            **gpu_snapshot(),
        }
        try:
            probe = subprocess.run(
                [executable, "-c", _PROBE],
                capture_output=True,
                text=True,
                timeout=45,
            )
            if probe.returncode:
                raise RuntimeError(probe.stderr.strip().splitlines()[-1])
            result.update(json.loads(probe.stdout.strip().splitlines()[-1]))
            if shutil.which("tmux") is None:
                raise RuntimeError("Install tmux to run persistent training workers.")
        except (OSError, ValueError, IndexError, RuntimeError, subprocess.TimeoutExpired) as error:
            result["available"] = False
            result["findings"] = [
                {
                    "severity": "error",
                    "code": "TRAINING_RUNTIME_UNAVAILABLE",
                    "message": f"Training runtime unavailable: {error}. Install HistoPilot's training extra or set HISTOPILOT_TRAINING_PYTHON to that environment's Python.",
                }
            ]
        _CACHE[executable] = (time.monotonic(), result)
        return result
