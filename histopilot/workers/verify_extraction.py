"""Validate extraction artifacts in bounded chunks inside the durable worker session.

Run with the HistoPilot service interpreter, which owns h5py independently of the
TRIDENT environment: python -m histopilot.workers.verify_extraction JOB VALIDATION
[PROGRESS]. As a Task Center task (``extraction-validation``) it also records its task
attempt in the report and writes PROGRESS after every chunk.
"""

import json
import os
import sys
from pathlib import Path

from histopilot.application.extraction_artifacts import complete_coverage, inspect_outputs
from histopilot.storage.io import read_file_bounded, utc_now, write_json_atomic
from histopilot.storage.project_lock import reject_symlink_components

CHUNK_SIZE = 16
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_FINDINGS = 1000


def _task_identity() -> dict:
    """The Task Center attempt this process runs, so a report is never mistaken for another's."""
    attempt = os.environ.get("HISTOPILOT_TASK_ATTEMPT", "")
    if not os.environ.get("HISTOPILOT_TASK_ID"):
        return {}
    return {
        "taskId": os.environ["HISTOPILOT_TASK_ID"],
        "taskAttempt": int(attempt) if attempt.isdecimal() else None,
    }


def validate_job(job_path: Path, validation_path: Path, progress_path: Path | None = None) -> dict:
    """Persist one completion inspection; status polling only needs the resulting JSON."""
    job = json.loads(read_file_bounded(job_path, MAX_JSON_BYTES))
    if (
        not isinstance(job, dict)
        or not isinstance(job.get("id"), str)
        or not isinstance(job.get("slides"), list)
        or not job["slides"]
    ):
        raise ValueError("Extraction job must identify a nonempty frozen slide list.")
    reject_symlink_components(validation_path)
    if (
        validation_path.parent.resolve() != job_path.parent.resolve()
        or validation_path.resolve() == job_path.resolve()
    ):
        raise ValueError("Validation must be a separate file beside the immutable job record.")
    slides = job["slides"]
    identity = _task_identity()
    if identity:
        # The job log is shared with the extraction task; mark where this phase starts.
        print(f"[{utc_now()}] Starting Artifact validation worker", flush=True)
    result = {
        "jobId": job["id"],
        **identity,
        "startedAt": utc_now(),
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
        inspected = min(start + CHUNK_SIZE, len(slides))
        print(
            f"[validation] Inspected {inspected}/{len(slides)} slides: "
            f"{result['completedSlides']} complete, {result['missingSlides']} invalid or missing, "
            f"{result['unvalidatedSlides']} unvalidated.",
            flush=True,
        )
        if progress_path is not None:
            write_json_atomic(
                progress_path,
                {
                    "stage": "validation",
                    "phase": "Output validation",
                    "completed": inspected,
                    "total": len(slides),
                    "completedSlides": inspected,
                    "totalSlides": len(slides),
                    "unit": "slides",
                    "message": f"{inspected} of {len(slides)} slides inspected.",
                },
                limit=MAX_JSON_BYTES,
            )
    result["finishedAt"] = utc_now()
    result["expectedSlides"] = len(slides)
    result["complete"] = complete_coverage(result, len(slides))
    write_json_atomic(validation_path, result, limit=MAX_JSON_BYTES)
    return result


def main() -> int:
    if len(sys.argv) not in {3, 4}:
        print(
            "Usage: python -m histopilot.workers.verify_extraction JOB VALIDATION [PROGRESS]",
            file=sys.stderr,
        )
        return 2
    progress = Path(sys.argv[3]) if len(sys.argv) == 4 else None
    managed = bool(_task_identity())
    try:
        result = validate_job(Path(sys.argv[1]), Path(sys.argv[2]), progress)
    except Exception as error:
        print(f"[validation] Failed to validate extraction: {error}", file=sys.stderr)
        if managed:
            print(f"[{utc_now()}] Artifact validation failed (exit 1)", flush=True)
        return 1
    code = 0 if result["inspectionComplete"] and not result["missingSlides"] else 1
    if managed:
        state = "succeeded" if code == 0 else "failed"
        print(f"[{utc_now()}] Artifact validation {state} (exit {code})", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
