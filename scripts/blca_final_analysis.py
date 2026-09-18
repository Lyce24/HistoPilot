"""Resume descriptive BLCA reporting and a prespecified, bounded attention review.

Run with the control environment and an existing real project state. This script
never trains, starts a server, changes an evaluation cutoff, or invents evidence.
Attention uses one native durable compute job for the selected slides.
Repeated invocations poll the same saved job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from histopilot.application.case_review import CaseReviewService
from histopilot.application.clinical import ClinicalService
from histopilot.application.evaluation_runs import EvaluationRunService
from histopilot.application.interpretation import InterpretationService
from histopilot.schemas.case_review import CaseReviewQuery
from histopilot.schemas.clinical import ClinicalSelection, SaveClinicalAnalysis
from histopilot.schemas.interpretation import (
    InterpretationGallerySource,
    InterpretationSelection,
    SaveInterpretation,
)
from histopilot.schemas.predictors import CompareEvaluations
from histopilot.storage.filesystem import LocalFilesystem
from histopilot.storage.project_lock import StorageError
from histopilot.storage.scientific import ScientificStore
from histopilot.workers.packing_process import write_json

POLICY = {
    "version": 1,
    "primaryMethod": "ensemble",
    "secondaryMethod": "refit",
    "unit": "slide",
    "decisionThreshold": 0.5,
    "calibrationBins": 5,
    "curveThresholdMin": 0.05,
    "curveThresholdMax": 0.95,
    "curveThresholdSteps": 19,
    "selectionSeed": 20260917,
    "attentionMaximumSlides": 8,
    "attentionResources": {"gpuIds": [0], "cpuThreadsPerRun": 4, "dataLoaderWorkers": 0,
                           "ramGbPerRun": 8.0, "maxConcurrentRuns": 1},
    "attentionBuckets": [
        {"name": "confident_false_positive", "maximum": 1},
        {"name": "confident_false_negative", "maximum": 1},
        {"name": "confident_correct_positive", "maximum": 1},
        {"name": "confident_correct_negative", "maximum": 1},
        {"name": "near_threshold_positive", "maximum": 2},
        {"name": "near_threshold_negative", "maximum": 2},
    ],
    "selectionRule": "Buckets in stated order, without replacement; deterministic SHA256 ties; empty buckets remain empty.",
    "attentionMeaning": "Class-independent whole-bag pooling attention; not tumor localization or causal explanation.",
    "nearThresholdMeaning": "Distance from the frozen 0.5 cutoff, not estimated epistemic uncertainty.",
    "history": "Grade-2 cases were used in earlier model searches; this is exploratory workflow validation, not independent confirmation.",
    "identity": "Slide/case identifiers are acknowledged fallback groups, not verified patients. Patient-level uncertainty and paired comparisons must be unavailable.",
    "clinicalScope": "Descriptive grading discrimination and calibration. Decision curves are software outputs without a prespecified clinical intervention or patient-benefit claim.",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_state(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 1024 * 1024:
        raise ValueError("An existing regular state JSON file of at most 1 MB is required.")
    state = json.loads(path.read_text())
    if not isinstance(state, dict):
        raise ValueError("The run state must be an object.")
    return state


def update_state(path, **changes):
    state = read_state(path)
    state.update(changes)
    write_json(Path(path), state)
    return state


def atomic_bytes(path, content):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Refuse a symlink output.")
    descriptor, temporary = tempfile.mkstemp(prefix=".blca-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def prepare(state_path):
    """Record the immutable analysis policy before fitting/evaluation when possible."""
    state_path = Path(state_path)
    state = read_state(state_path)
    policy_path = state_path.parent / "final-analysis-plan.json"
    if policy_path.exists():
        saved = read_state(policy_path)
        if saved.get("policy") != POLICY or saved.get("policySha256") != digest(POLICY):
            raise ValueError("The recorded BLCA analysis policy changed; preserve the original run.")
    else:
        saved = {
            "policy": POLICY,
            "policySha256": digest(POLICY),
            "recordedAt": datetime.now(UTC).isoformat(),
            "evaluationIdsAtRegistration": state.get("evaluationIds", {}),
            "registrationNote": "Policy registration time is recorded; its existence alone does not prove predictions were previously unseen.",
        }
        write_json(policy_path, saved)
    update_state(state_path, finalAnalysisPolicyPath=str(policy_path),
                 finalAnalysisPolicySha256=saved["policySha256"])
    return saved


def select_attention(items, *, positive_index, threshold=0.5):
    """Select up to eight different cases using the fixed policy and actual scores."""
    if threshold != POLICY["decisionThreshold"]:
        raise ValueError("The evaluation cutoff differs from the registered review policy.")
    if len({row["id"] for row in items}) != len(items):
        raise ValueError("Duplicate case identities cannot enter attention selection.")

    def score(row):
        return row["probabilities"][positive_index]

    def tie(row):
        return digest([POLICY["selectionSeed"], row["id"]])

    buckets = [
        ("confident_false_positive", 1,
         lambda row: row["outcome"] == "false_positive", lambda row: -score(row)),
        ("confident_false_negative", 1,
         lambda row: row["outcome"] == "false_negative", score),
        ("confident_correct_positive", 1,
         lambda row: row["outcome"] == "correct" and row["labelIndex"] == positive_index,
         lambda row: -score(row)),
        ("confident_correct_negative", 1,
         lambda row: row["outcome"] == "correct" and row["labelIndex"] != positive_index,
         score),
        ("near_threshold_positive", 2, lambda row: row["labelIndex"] == positive_index,
         lambda row: abs(score(row) - threshold)),
        ("near_threshold_negative", 2,
         lambda row: row["labelIndex"] is not None and row["labelIndex"] != positive_index,
         lambda row: abs(score(row) - threshold)),
    ]
    selected, seen, counts = [], set(), {}
    for name, count, predicate, order in buckets:
        chosen = sorted((row for row in items if row["id"] not in seen and predicate(row)),
                        key=lambda row: (order(row), tie(row)))[:count]
        counts[name] = len(chosen)
        for row in chosen:
            if len(row["slides"]) != 1 or not row["slides"][0].get("slidePath"):
                raise ValueError("Each selected slide must resolve to one frozen original image.")
            selected.append({
                "slideId": row["id"], "slidePath": row["slides"][0]["slidePath"],
                "reason": name, "label": row["label"], "outcome": row["outcome"],
                "positiveProbability": score(row),
                "distanceToThreshold": abs(score(row) - threshold),
            })
            seen.add(row["id"])
    return {"items": selected, "bucketCounts": counts}


def _services(state):
    project = Path(state["projectPath"])
    if not project.is_dir() or not (project / "histopilot-state.sqlite").is_file():
        raise ValueError("The real HistoPilot project must already exist.")
    # BLCA's TRIDENT job root is two levels above its encoder directory: patch
    # geometry config lives in the resolution folder, and segmentation config
    # lives in the job root. Both frozen bindings must be readable by the gallery.
    filesystem = LocalFilesystem((project, Path(state["slideRoot"]), Path(state["featurePath"]).parent.parent))
    store = ScientificStore(project, state["projectId"])
    return store, filesystem


def _advance_attention(service, state, state_path, selection, clinical_id, *, launch):
    """Save before launch, then replay the same launch receipt after interruption."""
    identity = state.get("attentionBatchId")
    if identity:
        saved = service.get(identity)
        if ({row["slideId"] for row in saved["manifest"]["slides"]}
                != {row["slideId"] for row in selection["items"]}
                or saved["manifest"]["evaluationId"] != state["evaluationIds"]["ensemble"]):
            raise ValueError("The saved attention job differs from the fixed review selection.")
    else:
        source = InterpretationGallerySource(
            featureBundleId=state["bundleId"], slideFolder=state["slideRoot"],
            predictorId=state["predictorIds"]["ensemble"],
        )
        resolved, folder, rows, _ = service.gallery.rows(source, require_current=True)
        by_path = {row["slidePath"]: row for row in rows}
        slides = []
        for row in selection["items"]:
            if row["slidePath"] not in by_path:
                raise ValueError("A selected image no longer belongs to the frozen source gallery.")
            slides.append(service.gallery.input(resolved, by_path[row["slidePath"]]))
        choice = InterpretationSelection(
            name="BLCA prespecified exploratory review", predictorId=source.predictorId,
            evaluationId=state["evaluationIds"]["ensemble"], clinicalAnalysisId=clinical_id,
            featureBundleId=source.featureBundleId, slideFolder=str(folder),
            encoderId=resolved["contract"]["encoderId"], slides=slides,
            resources=POLICY["attentionResources"],
        )
        # These are separate native requests. Each must resolve/revalidate its
        # gallery with its own inspection deadline, as the HTTP flow does.
        preview = service.preview(choice)
        if not preview["canSave"]:
            raise ValueError(f"Attention review blocked: {preview['findings']}")
        saved = service.save(SaveInterpretation(
            **choice.model_dump(), previewHash=preview["previewHash"],
            operationId=f"blca-attention-save-{digest(selection)}",
        ))
        identity = saved["id"]
        update_state(state_path, attentionBatchId=identity,
                     interpretationIds={row["slideId"]: identity for row in selection["items"]})
        saved = service.get(identity)
    status = saved["execution"]["status"]
    if status == "not_started" and launch:
        # A retry after a lost acknowledgement reuses this exact launch operation.
        service.launch(identity, f"blca-attention-launch-{digest(selection)}")
        saved = service.get(identity)
        status = saved["execution"]["status"]
    translated = {
        "completed": "complete", "not_started": "attention_prepared",
        "failed": "attention_needs_review", "cancelled": "attention_needs_review",
        "interrupted": "attention_needs_review",
    }.get(status, "attention_running")
    return saved, translated


def _export_attention(service, saved, selection, primary_cases, folder):
    content = service.artifact(saved["id"], "attention.json")
    evidence = json.loads(content)
    expected = {row["id"]: row for row in primary_cases}
    observed = {row["slideId"]: row for row in evidence["slides"]}
    if set(observed) != {row["slideId"] for row in selection["items"]}:
        raise ValueError("Attention results do not cover exactly the fixed selection.")
    # Full-bag attention must preserve the frozen ensemble's full-bag predictions.
    for index, row in enumerate(selection["items"]):
        slide = row["slideId"]
        if any(not math.isclose(a, b, abs_tol=2e-5, rel_tol=2e-5) for a, b in
               zip(observed[slide]["probabilities"], expected[slide]["probabilities"], strict=True)):
            raise ValueError("Whole-bag attention predictions differ from frozen evaluation.")
        top = service.top_attention(saved["id"], slide, limit=3)
        if top["total"] != observed[slide]["patchCount"] or not top["patches"]:
            raise ValueError("Attention patch coverage differs from the saved whole bag.")
        write_json(folder / f"attention-{index:02d}-top.json", top)
        atomic_bytes(folder / f"attention-{index:02d}-overview.png", service.image(saved["id"], slide, max_size=1024))
        atomic_bytes(folder / f"attention-{index:02d}-top-patch.png",
                     service.patch_image(saved["id"], slide, top["patches"][0]["index"], max_size=512))
    atomic_bytes(folder / "attention-evidence.json", content)
    return {"verifiedSlides": len(observed),
            "verifiedPatches": sum(row["patchCount"] for row in observed.values()),
            "predictionAgreementTolerance": {"absolute": 2e-5, "relative": 2e-5}}


def finalize(state_path, *, launch_attention=True):
    """Publish real reports and advance the one native attention job; never wait."""
    state_path = Path(state_path)
    registration = prepare(state_path)
    state = read_state(state_path)
    methods = ("ensemble", "refit")
    if any(method not in state.get("evaluationIds", {}) for method in methods):
        return update_state(state_path, finalAnalysisStatus="waiting_for_evaluations")
    store, filesystem = _services(state)
    evaluations = EvaluationRunService(store, filesystem)
    clinical = ClinicalService(store, filesystem)
    cases = CaseReviewService(store, filesystem)
    interpretations = InterpretationService(store, filesystem)
    documents = {method: evaluations.get(state["evaluationIds"][method]) for method in methods}
    if any(document["execution"]["status"] != "completed" for document in documents.values()):
        return update_state(state_path, finalAnalysisStatus="waiting_for_evaluations")
    reports, analysis_ids, primary_cases, provenance = {}, {}, [], {}
    for method, evaluation in documents.items():
        manifest = evaluation["manifest"]
        if (manifest["predictorId"] != state["predictorIds"][method]
                or manifest["cohortId"] != state["cohortId"]
                or manifest["target"]["unit"] != "slide"
                or manifest["inference"]["decisionThreshold"] != POLICY["decisionThreshold"]):
            raise ValueError("The evaluated predictor, cohort, unit or cutoff differs from this run.")
        choice = ClinicalSelection(
            evaluationId=evaluation["id"], name=f"BLCA exploratory grading — {method}",
            unit="slide", bins=POLICY["calibrationBins"],
            thresholdMin=POLICY["curveThresholdMin"], thresholdMax=POLICY["curveThresholdMax"],
            thresholdSteps=POLICY["curveThresholdSteps"],
        )
        reviewed = clinical.preview(choice)
        if not reviewed["canSave"]:
            raise ValueError(f"Clinical report blocked: {reviewed['findings']}")
        saved = clinical.save(SaveClinicalAnalysis(
            **choice.model_dump(), previewHash=reviewed["previewHash"],
            operationId=f"blca-final-analysis-{method}-{evaluation['id']}",
        ))
        report = saved["manifest"]["report"]
        if (report["counts"]["fallbackPatientIds"] != report["counts"]["slides"]
                or report["uncertainty"]["method"] != "unavailable"):
            raise ValueError("This BLCA workflow expects explicit slide fallback and unavailable patient intervals.")
        reports[method] = report
        analysis_ids[method] = saved["id"]
        provenance[method] = saved["manifest"]["source"]
        for filename in ("report.json", "calibration.csv", "roc.csv", "precision-recall.csv", "operating-curves.csv"):
            content, _ = clinical.artifact(saved["id"], filename)
            atomic_bytes(state_path.parent / f"{method}-{filename}", content)
        if method == "ensemble":
            offset = 0
            while True:
                page = cases.query(evaluation["id"], CaseReviewQuery(unit="slide", offset=offset, limit=100))
                if page["total"] > 1000 or not page["supportsAttention"]:
                    raise ValueError("The primary case review exceeds this bounded study or lacks attention.")
                if page["source"]["predictionsSha256"] != provenance[method]["predictionsSha256"]:
                    raise ValueError("Clinical analysis and case review read different predictions.")
                primary_cases.extend(page["items"])
                if not page["hasMore"]:
                    break
                offset += 100
    try:
        evaluations.compare(CompareEvaluations(
            leftEvaluationId=state["evaluationIds"]["ensemble"],
            rightEvaluationId=state["evaluationIds"]["refit"],
        ))
    except StorageError as error:
        if error.code != "COMPARISON_EVIDENCE_INVALID" or "verified patient" not in str(error):
            raise
        comparison_check = {"status": "correctly_rejected", "code": error.code, "reason": str(error)}
    else:
        raise ValueError("Patient comparison unexpectedly accepted unverified slide groups.")
    target = documents["ensemble"]["manifest"]["target"]
    selection = select_attention(primary_cases, positive_index=target["classes"].index(target["positiveClass"]))
    selection.update(policySha256=registration["policySha256"],
                     evaluationId=state["evaluationIds"]["ensemble"],
                     predictionsSha256=provenance["ensemble"]["predictionsSha256"])
    selection_path = state_path.parent / "attention-selection.json"
    if selection_path.exists() and read_state(selection_path) != selection:
        raise ValueError("The frozen attention case selection or prediction evidence changed.")
    write_json(selection_path, selection)
    state = update_state(state_path, clinicalAnalysisIds=analysis_ids,
                         attentionSelectionPath=str(selection_path),
                         patientComparisonCheck=comparison_check)
    saved_attention, status = _advance_attention(
        interpretations, state, state_path, selection, analysis_ids["ensemble"],
        launch=launch_attention,
    )
    verification = (_export_attention(interpretations, saved_attention, selection, primary_cases, state_path.parent)
                    if status == "complete" else {})
    attention_summary = {"selectedSlides": len(selection["items"]),
                         "bucketCounts": selection["bucketCounts"],
                         "savedInterpretations": 1,
                         "completed": int(status == "complete"),
                         "failed": int(status == "attention_needs_review"), **verification}
    summary = {
        "status": status, "scope": POLICY["history"], "identity": POLICY["identity"],
        "clinicalScope": POLICY["clinicalScope"], "primaryMethod": "ensemble",
        "methods": {method: {"counts": report["counts"], "metrics": report["metrics"],
                             "operatingPoint": report["operatingPoint"],
                             "uncertainty": report["uncertainty"], "warnings": report["warnings"]}
                    for method, report in reports.items()},
        "attention": attention_summary, "patientComparison": comparison_check,
    }
    write_json(state_path.parent / "final-analysis-summary.json", summary)
    return update_state(state_path, finalAnalysisStatus=status,
                        finalAnalysisSummaryPath=str(state_path.parent / "final-analysis-summary.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--no-launch-attention", action="store_true")
    args = parser.parse_args()
    result = prepare(args.state) if args.prepare_only else finalize(args.state, launch_attention=not args.no_launch_attention)
    # CLI output contains aggregate status only; private identities remain in .local.
    print(json.dumps({"policySha256": result.get("policySha256", result.get("finalAnalysisPolicySha256")),
                      "status": "policy_prepared" if args.prepare_only else result["finalAnalysisStatus"]}))


if __name__ == "__main__":
    main()
