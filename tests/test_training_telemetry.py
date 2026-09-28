"""GPU failures and driver observations are explicit, bounded, and auditable."""

import subprocess
from types import SimpleNamespace

import pytest

from histopilot.workers import training_process as process


@pytest.mark.parametrize(
    "message, expected",
    [
        (
            "CUDA unknown error - this may be due to an incorrectly set up environment",
            "cuda_device_failure",
        ),
        ("CUDA error: an illegal memory access was encountered", "cuda_device_failure"),
        ("Failed to initialize NVML: Driver/library version mismatch", "cuda_device_failure"),
        ("CUDA out of memory. Tried to allocate 12 GiB", "out_of_memory"),
        ("OutOfMemoryError", "out_of_memory"),
        ("Cannot read feature file", "training_error"),
    ],
)
def test_failure_classification_does_not_confuse_oom_with_device_loss(message, expected):
    assert process.classify_training_failure(message) == expected


def test_gpu_probe_preserves_unknown_fields_and_uses_short_timeout(monkeypatch):
    calls = []
    monkeypatch.setattr(process.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")

    def probe(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout='0, GPU-a, "GPU, large", 560.35, 24576, 1024, [N/A], N/A\n')

    monkeypatch.setattr(process.subprocess, "run", probe)
    value = process.gpu_snapshot()
    assert value["gpus"][0]["name"] == "GPU, large"
    assert value["gpus"][0]["totalMemoryGb"] == 24
    assert value["gpus"][0]["usedMemoryGb"] == 1
    assert value["gpus"][0]["freeMemoryGb"] is None
    assert value["gpus"][0]["utilizationPercent"] is None
    assert calls[0][1]["timeout"] == 3


def test_gpu_probe_failure_remains_a_diagnostic_not_fabricated_zero_memory(monkeypatch):
    monkeypatch.setattr(process.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")

    def unavailable(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("nvidia-smi", 3)

    monkeypatch.setattr(process.subprocess, "run", unavailable)
    value = process.gpu_snapshot()
    assert value["gpus"] == []
    assert "timed out" in value["gpuProbeError"]
