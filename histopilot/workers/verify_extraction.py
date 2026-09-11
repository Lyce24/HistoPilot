"""Validate extraction artifacts in bounded chunks inside the durable worker session.

Run with the HistoPilot service interpreter, which owns h5py independently of the
TRIDENT environment: python -m histopilot.workers.verify_extraction JOB VALIDATION.
"""

import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from histopilot.application.extraction_artifacts import inspect_outputs
from histopilot.storage.project_lock import _reject_symlink_components, fsync_directory
from histopilot.storage.scientific import ScientificStore

CHUNK_SIZE = 16
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_FINDINGS = 1000


def _write_validation(path: Path, value: dict) -> None:
    _reject_symlink_components(path)
    content = json.dumps(value, allow_nan=False, indent=2).encode() + b"\n"
    if len(content) > MAX_JSON_BYTES:
        raise ValueError("Extraction validation exceeds its storage limit.")
    descriptor, temporary = tempfile.mkstemp(prefix=".validation-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def validate_job(job_path: Path, validation_path: Path) -> dict:
    """Persist one completion inspection; status polling only needs the resulting JSON."""
    job = json.loads(ScientificStore._read_file(job_path, MAX_JSON_BYTES))
    if (
        not isinstance(job, dict)
        or not isinstance(job.get("id"), str)
        or not isinstance(job.get("slides"), list)
        or not job["slides"]
    ):
        raise ValueError("Extraction job must identify a nonempty frozen slide list.")
    _reject_symlink_components(validation_path)
    if (
        validation_path.parent.resolve() != job_path.parent.resolve()
        or validation_path.resolve() == job_path.resolve()
    ):
        raise ValueError("Validation must be a separate file beside the immutable job record.")
    slides = job["slides"]
    result = {
        "jobId": job["id"],
        "startedAt": datetime.now(UTC).isoformat(),
        "completedSlides": 0,
        "missingSlides": 0,
        "unvalidatedSlides": 0,
        "inspectionComplete": True,
        "findings": [],
        "findingsTruncated": 0,
    }
    for start in range(0, len(slides), CHUNK_SIZE):
        chunk = slides[start : start + CHUNK_SIZE]
        try:
            coverage = inspect_outputs({**job, "slides": chunk})
        except Exception as error:
            # Preserve a durable, honest outcome even if validation itself fails.
            coverage = {
                "completedSlides": 0,
                "missingSlides": 0,
                "unvalidatedSlides": len(chunk),
                "inspectionComplete": False,
                "findings": [
                    {
                        "severity": "error",
                        "code": "OUTPUT_VALIDATION_FAILED",
                        "message": f"Could not inspect {len(chunk)} slides: {error}",
                    }
                ],
            }
        for key in ("completedSlides", "missingSlides", "unvalidatedSlides"):
            result[key] += coverage[key]
        result["inspectionComplete"] &= coverage["inspectionComplete"]
        if "featurePath" in coverage:
            result["featurePath"] = coverage["featurePath"]
        findings = coverage.get("findings", [])
        available = MAX_FINDINGS - len(result["findings"])
        result["findings"].extend(findings[:available])
        result["findingsTruncated"] += max(0, len(findings) - available)
        print(
            f"[validation] Inspected {min(start + CHUNK_SIZE, len(slides))}/{len(slides)} slides: "
            f"{result['completedSlides']} complete, {result['missingSlides']} invalid or missing, "
            f"{result['unvalidatedSlides']} unvalidated.",
            flush=True,
        )
    result["finishedAt"] = datetime.now(UTC).isoformat()
    _write_validation(validation_path, result)
    return result


def main() -> int:
    if len(sys.argv) != 3:
        print(
            "Usage: python -m histopilot.workers.verify_extraction JOB VALIDATION", file=sys.stderr
        )
        return 2
    try:
        result = validate_job(Path(sys.argv[1]), Path(sys.argv[2]))
    except Exception as error:
        print(f"[validation] Failed to validate extraction: {error}", file=sys.stderr)
        return 1
    return 0 if result["inspectionComplete"] and not result["missingSlides"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
