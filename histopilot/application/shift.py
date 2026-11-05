"""Dataset shift for a run of Apply models: its cohort's slides against the slides its
predictor was developed on, in feature space (``histopilot/shift.py`` has the method).

An analysis is a frozen ``shift-analysis`` record, content-addressed by what it compares:
the training batch whose development slides the predictor learned from (pinned by that
batch's plan hash) and the cohort as the run reads it (cohort and feature record, pinned by
content hash). Every run with the same development batch and cohort features shares one
analysis, computed once. Its job is a Task Center compute task that reads features and never
a label; performance by atypicality joins the per-slide distances with a scored run's labels
here, in the control service, exactly as subgroup performance does.
"""

import hashlib
from itertools import pairwise

from histopilot import shift as method
from histopilot.application.case_review import CaseReviewService
from histopilot.application.compute_jobs import ComputeJobService
from histopilot.application.evaluation_runs import EvaluationRunService, run_purpose
from histopilot.application.predictors import read_evidence
from histopilot.application.run_evidence import (
    development_flag,
    development_patients,
    labels_withheld,
)
from histopilot.cv_summary import evaluation_metrics
from histopilot.storage.io import content_hash, decode_json_object, read_file_bounded
from histopilot.storage.lifecycle import lifecycle_guard
from histopilot.storage.project_lock import StorageError
from histopilot.storage.receipts import receipt_names
from histopilot.storage.scientific import MAX_CONFIGURATION_BYTES, encode_document

KIND = "shift-analysis"
# A CPU task: it reads features and runs numpy, so it never waits for, or holds, a GPU.
RESOURCES = {
    "maxConcurrentRuns": 1,
    "gpuIds": [],
    "runsPerGpu": 1,
    "cpuThreadsPerRun": 2,
    "dataLoaderWorkers": 0,
    "ramGbPerRun": 4.0,
}
MAX_REPORT_BYTES = 64 * 1024 * 1024
# Atypicality bins for performance: thirds of the scored units, least atypical first.
THIRDS = ("Least atypical third", "Middle third", "Most atypical third")


def _error(message, code, status=409):
    return StorageError(message, code, status)


def training_inputs(store, batch_id):
    """The development feature files a training batch read, from its verified plan."""
    folder = store.folder / "training" / batch_id
    try:
        plan = read_evidence(folder / "plan.json", folder)
        state = read_evidence(folder / "state.json", folder)
    except StorageError as error:
        raise _error(
            "The predictor's training plan is unavailable, so its development slides are unknown.",
            "SHIFT_INPUTS_UNAVAILABLE",
        ) from error
    if state.get("planHash") != content_hash(plan) or plan.get("batchId") != batch_id:
        raise _error("The predictor's training plan changed.", "SHIFT_INPUTS_UNAVAILABLE")
    data = plan.get("data") or {}
    if not isinstance(data.get("featureFiles"), dict) or not data["featureFiles"]:
        raise _error(
            "The predictor's training plan lists no feature files.", "SHIFT_INPUTS_UNAVAILABLE"
        )
    return state["planHash"], data["featureFiles"], data.get("featureDim")


def cohort_inputs(store, run_manifest, development):
    """The cohort's feature files as the run reads them, without development slides."""
    cohort = store.get_configuration(run_manifest["cohortId"])
    feature = store.get_configuration(run_manifest["features"]["feature"]["id"])
    selected = {row["slideId"] for row in cohort["manifest"]["memberships"]}
    files = {
        row["slideId"]: row for row in feature["manifest"]["files"] if row["slideId"] in selected
    }
    shared = sorted(set(files) & set(development))
    for slide in shared:
        del files[slide]
    return cohort, feature, files, shared


class ShiftService:
    def __init__(self, store, filesystem, *, runtime=None, task_center=None):
        self.store, self.filesystem = store, filesystem
        self.evaluations = EvaluationRunService(store, filesystem)
        self.jobs = ComputeJobService(store, filesystem, runtime=runtime, task_center=task_center)
        self._task_center = task_center

    # Inputs and identity -------------------------------------------------------------

    def inputs(self, evaluation_id):
        run = self.evaluations.record(evaluation_id)
        manifest = run["manifest"]
        predictor = self.store.get_configuration(manifest["predictorId"])
        batch_id = predictor["manifest"].get("batchId")
        if not batch_id:
            raise _error(
                "This predictor names no training batch, so its development slides are unknown.",
                "SHIFT_INPUTS_UNAVAILABLE",
            )
        plan_hash, development, development_dim = training_inputs(self.store, batch_id)
        cohort, feature, files, shared = cohort_inputs(self.store, manifest, development)
        dimensions = manifest["features"]["dimensions"]
        findings = []
        if development_dim != dimensions:
            findings.append(
                {
                    "severity": "error",
                    "code": "SHIFT_FEATURES_INCOMPATIBLE",
                    "message": f"Development features have {development_dim} dimensions and "
                    f"the cohort's have {dimensions}.",
                }
            )
        if len(development) < 2 or len(files) < 2:
            findings.append(
                {
                    "severity": "error",
                    "code": "SHIFT_INPUTS_UNAVAILABLE",
                    "message": "Both the development slides and the cohort need at least two "
                    "slides with features.",
                }
            )
        if shared:
            findings.append(
                {
                    "severity": "warning",
                    "code": "SHIFT_SHARED_SLIDES",
                    "message": f"{len(shared)} cohort slides were also development slides; "
                    "they are compared as development slides only.",
                }
            )
        experiment = predictor["manifest"].get("experiment") or {}
        label = (cohort.get("versionLabel") or {}).get("tag") or cohort["id"][:22]
        # Names are labels beside frozen content, never part of it: renaming an experiment or
        # retagging a cohort must not orphan its analysis. The title names the Task Center task.
        title = f"Dataset shift · {label} against {experiment.get('name') or batch_id}"
        record = {
            "kind": KIND,
            "schemaVersion": 1,
            "datasetId": manifest["datasetId"],
            "method": method.METHOD,
            "featureDim": dimensions,
            "development": {
                "batchId": batch_id,
                "planHash": plan_hash,
                "slides": len(development),
            },
            "cohort": {
                "cohortId": cohort["id"],
                "cohortContentHash": cohort["contentHash"],
                "featureId": feature["id"],
                "featureContentHash": feature["contentHash"],
                "slides": len(files),
                "sharedSlidesExcluded": len(shared),
            },
        }
        return {
            "run": run,
            "title": title,
            "record": record,
            "identity": "configuration-"
            + hashlib.sha256(encode_document(record, MAX_CONFIGURATION_BYTES)).hexdigest(),
            "development": development,
            "cohort": files,
            "findings": findings,
        }

    def _existing(self, identity):
        try:
            document = self.store.get_configuration(identity)
        except StorageError as error:
            if error.code == "CONFIGURATION_NOT_FOUND":
                return None
            raise
        return document if document["manifest"].get("kind") == KIND else None

    # Reads ----------------------------------------------------------------------------

    def status(self, evaluation_id):
        """The run's analysis: whether it exists, its job, and its report once complete."""
        inputs = self.inputs(evaluation_id)
        document = self._existing(inputs["identity"])
        execution = self.jobs.status(inputs["identity"]) if document else None
        report = (
            self.report(inputs["identity"])
            if execution and execution["status"] == "completed"
            else None
        )
        return {
            "evaluationId": evaluation_id,
            "analysisId": inputs["identity"],
            "title": inputs["title"],
            "exists": document is not None,
            "record": document,
            "execution": execution,
            "findings": inputs["findings"],
            "canRun": not any(row["severity"] == "error" for row in inputs["findings"]),
            "report": report,
        }

    def report(self, identity):
        """The job's verified ``shift.json``: its receipt names it and its checksum matches."""
        execution = self.jobs.status(identity)
        if execution["status"] != "completed":
            raise _error("The dataset shift analysis has not finished.", "SHIFT_NOT_COMPLETED")
        receipt = (execution.get("result") or {}).get("artifacts", {}).get("shift.json")
        path = self.jobs.folder(identity) / "shift.json"
        if not receipt or not receipt_names(receipt.get("path"), path):
            raise _error("The dataset shift report is missing.", "SHIFT_RESULT_CHANGED")
        content = read_file_bounded(path, MAX_REPORT_BYTES)
        if (
            len(content) != receipt["bytes"]
            or hashlib.sha256(content).hexdigest() != receipt["sha256"]
        ):
            raise _error("The dataset shift report changed.", "SHIFT_RESULT_CHANGED")
        return decode_json_object(content)

    # Preview and start ----------------------------------------------------------------

    def preview(self, evaluation_id):
        current = self.status(evaluation_id)
        inputs = self.inputs(evaluation_id)
        summary = {
            "evaluationId": evaluation_id,
            "analysisId": current["analysisId"],
            "exists": current["exists"],
            "status": (current["execution"] or {}).get("status", "not_started"),
            "development": inputs["record"]["development"],
            "cohort": inputs["record"]["cohort"],
            "featureDim": inputs["record"]["featureDim"],
            "findings": current["findings"],
            "canRun": current["canRun"]
            and (current["execution"] or {}).get("status")
            not in {"queued", "running", "completed"},
        }
        return {**summary, "previewHash": content_hash(summary)}

    def start(self, evaluation_id, preview_hash, operation_id):
        """Publish the analysis record if it is new, then queue its job."""
        with lifecycle_guard(self.store.folder):
            # An accepted request is answered again as it was, whatever has changed since.
            inputs = self.inputs(evaluation_id)
            if self._existing(inputs["identity"]) is not None:
                replay = self.jobs.replay_launch(
                    inputs["identity"], operation_id, resume=False, record_kind=KIND
                )
                if replay is not None:
                    return replay
            preview = self.preview(evaluation_id)
            if preview["previewHash"] != preview_hash:
                raise _error(
                    "The analysis changed since its preview. Preview it again.", "PREVIEW_STALE"
                )
            if not preview["canRun"]:
                problem = next(
                    (row for row in preview["findings"] if row["severity"] == "error"), None
                )
                if problem:
                    raise _error(problem["message"], problem["code"])
                return self.jobs.status(preview["analysisId"])
            identity = inputs["identity"]
            if self._existing(identity) is None:
                published = self.store.publish_configuration(
                    manifest=inputs["record"], operation_id=f"shift-{operation_id}"
                )
                if published["id"] != identity:
                    raise _error("The analysis record changed.", "SHIFT_INPUTS_UNAVAILABLE")
            state = self.jobs.status(identity)
            plan = {
                "kind": "shift",
                "resources": RESOURCES,
                "data": {
                    "featureDim": inputs["record"]["featureDim"],
                    "development": {"featureFiles": inputs["development"]},
                    "cohort": {"featureFiles": inputs["cohort"]},
                    # The worker re-verifies every source's file stamps before and after.
                    "sourceStamps": {
                        row["path"]: row
                        for files in (inputs["development"], inputs["cohort"])
                        for row in files.values()
                    },
                },
            }
            return self.jobs.launch(
                identity,
                plan,
                operation_id,
                resume=state["status"] != "not_started",
                task_title=inputs["title"],
            )

    # Performance by atypicality -------------------------------------------------------

    def performance(self, evaluation_id, *, unit=None, reference_id=None):
        """A labeled run's metrics within thirds of its units' atypicality, and within and
        beyond the development range. Descriptive: small groups are unstable."""
        current = self.status(evaluation_id)
        if current["report"] is None:
            raise _error("Analyze dataset shift for this run first.", "SHIFT_NOT_COMPLETED")
        manifest = current_manifest = self.evaluations.record(evaluation_id)["manifest"]
        if not reference_id and run_purpose(current_manifest) == "inference":
            raise _error(
                "This run predicts an unlabeled cohort. Score it against a reference standard "
                "to see its performance by shift.",
                "PERFORMANCE_REQUIRES_LABELS",
            )
        cases = CaseReviewService(self.store, self.filesystem)
        _evaluation, _cohort, records, unit, checksum = cases._evidence(
            evaluation_id, unit, reference_id
        )
        if labels_withheld(manifest) or reference_id:
            shared = development_patients(manifest)
            records = [row for row in records if development_flag(row, shared) is not True]
        slides = {
            row["slideId"]: row for row in current["report"]["slides"] if row["set"] == "cohort"
        }
        scored = []
        for row in records:
            known = [slides[slide] for slide in row["slideIds"] if slide in slides]
            if known and row.get("labelIndex") is not None:
                scored.append(
                    (
                        sum(item["percentile"] for item in known) / len(known),
                        any(item["beyond"] for item in known),
                        row,
                    )
                )
        scored.sort(key=lambda item: (item[0], item[2]["slideIds"][0]))
        target = manifest["target"]
        threshold = manifest["inference"]["decisionThreshold"]
        bins = []
        if scored:
            size = len(scored)
            edges = [0, size // 3 + (size % 3 > 0), 2 * size // 3 + (size % 3 > 1), size]
            for label, (start, end) in zip(THIRDS, pairwise(edges), strict=True):
                group = scored[start:end]
                if group:
                    bins.append(
                        {
                            "group": label,
                            "percentileRange": [group[0][0], group[-1][0]],
                            **evaluation_metrics([item[2] for item in group], target, threshold),
                        }
                    )
        within = [item[2] for item in scored if not item[1]]
        beyond = [item[2] for item in scored if item[1]]
        return {
            "evaluationId": evaluation_id,
            "analysisId": current["analysisId"],
            "unit": unit,
            "classOrder": target["classes"],
            "positiveClass": target.get("positiveClass"),
            "decisionThreshold": threshold,
            "overall": evaluation_metrics([item[2] for item in scored], target, threshold),
            "thirds": bins,
            "range": [
                {
                    "group": "Within the development range",
                    **evaluation_metrics(within, target, threshold),
                },
                {
                    "group": "Beyond the development range",
                    **evaluation_metrics(beyond, target, threshold),
                },
            ],
            "source": {"predictionsSha256": checksum},
        }
