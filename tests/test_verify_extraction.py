"""Artifact validation persists once inside the worker instead of on status polling."""

import json
import subprocess
import sys

import pytest

from histopilot.storage.project_lock import StorageError
from histopilot.workers import verify_extraction


@pytest.fixture
def validation_job(tmp_path):
    output = tmp_path / "outputs"
    geojson = output / "contours_geojson"
    geojson.mkdir(parents=True)
    slides = [{"name": f"slide-{index}"} for index in range(17)]
    for slide in slides:
        (geojson / f"{slide['name']}.geojson").write_text(
            json.dumps({"type": "FeatureCollection", "features": []})
        )
    job = {
        "id": "extraction-" + "a" * 32,
        "outputPath": str(output),
        "spec": {"options": {"task": "seg"}},
        "slides": slides,
    }
    path = tmp_path / "job.json"
    path.write_text(json.dumps(job))
    return path, tmp_path / "validation.json", job


def test_validation_command_inspects_multiple_chunks_and_persists_result(validation_job):
    job_path, validation_path, job = validation_job
    command = [
        sys.executable,
        "-m",
        "histopilot.workers.verify_extraction",
        str(job_path),
        str(validation_path),
    ]
    run = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stderr
    assert len(run.stdout.splitlines()) == 2
    result = json.loads(validation_path.read_text())
    assert result["jobId"] == job["id"]
    assert result["completedSlides"] == 17
    assert result["missingSlides"] == 0
    assert result["unvalidatedSlides"] == 0
    assert result["inspectionComplete"]
    assert len(result["findings"]) == 17
    assert all(item["severity"] == "warning" for item in result["findings"])
    assert json.loads(job_path.read_text()) == job


def test_invalid_artifact_exits_nonzero_but_retains_precise_coverage(validation_job):
    job_path, validation_path, job = validation_job
    from pathlib import Path

    (Path(job["outputPath"]) / "contours_geojson" / "slide-0.geojson").write_text("invalid")
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "histopilot.workers.verify_extraction",
            str(job_path),
            str(validation_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert run.returncode == 1
    result = json.loads(validation_path.read_text())
    assert result["completedSlides"] == 16
    assert result["missingSlides"] == 1
    assert result["unvalidatedSlides"] == 0
    assert result["inspectionComplete"]


def test_chunk_timeout_preserves_unvalidated_counts(validation_job, monkeypatch):
    job_path, validation_path, job = validation_job
    sizes = []

    def inspect(chunk):
        sizes.append(len(chunk["slides"]))
        return {
            "completedSlides": 0,
            "missingSlides": 0,
            "unvalidatedSlides": len(chunk["slides"]),
            "inspectionComplete": False,
            "findings": [
                {"severity": "error", "code": "ARTIFACT_SCAN_LIMIT", "message": "Timed out."}
            ],
        }

    monkeypatch.setattr(verify_extraction, "inspect_outputs", inspect)
    result = verify_extraction.validate_job(job_path, validation_path)
    assert sizes == [16, 1]
    assert result["unvalidatedSlides"] == 17
    assert result["missingSlides"] == 0
    assert not result["inspectionComplete"]


def test_validation_failure_is_persisted_as_unvalidated(validation_job, monkeypatch):
    job_path, validation_path, job = validation_job

    def fail(_job):
        raise RuntimeError("Reader unavailable")

    monkeypatch.setattr(verify_extraction, "inspect_outputs", fail)
    result = verify_extraction.validate_job(job_path, validation_path)
    assert not result["inspectionComplete"]
    assert result["unvalidatedSlides"] == 17
    assert result["findings"][0]["code"] == "OUTPUT_VALIDATION_FAILED"
    assert json.loads(validation_path.read_text()) == result


def test_validation_output_cannot_overwrite_job_or_follow_symlinks(validation_job):
    job_path, validation_path, job = validation_job
    with pytest.raises(ValueError, match="separate file"):
        verify_extraction.validate_job(job_path, job_path)
    validation_path.symlink_to(job_path)
    with pytest.raises(StorageError):
        verify_extraction.validate_job(job_path, validation_path)
    assert json.loads(job_path.read_text()) == job


def test_excess_findings_are_explicitly_truncated(validation_job, monkeypatch):
    job_path, validation_path, job = validation_job
    monkeypatch.setattr(verify_extraction, "MAX_FINDINGS", 3)
    result = verify_extraction.validate_job(job_path, validation_path)
    assert len(result["findings"]) == 3
    assert result["findingsTruncated"] == 14
    assert result["completedSlides"] == 17
