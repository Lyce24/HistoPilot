"""A model card for a frozen predictor, assembled from the records that made and tested it.

The card adds no new evidence. Every number on it is one HistoPilot already reports, on the
experiment's Results or on a run of Apply models, and every section names the record it
comes from, so a reader can check it. What the card adds is the frame a methods section or
a model registry needs around those numbers: what the model is, what it learned from, how it
was evaluated and how far its evaluation cohorts sit from its development data, what it is
for and what it is not for.
"""

from histopilot import __version__
from histopilot.application.evaluation_runs import EvaluationRunService
from histopilot.application.predictors import PredictorService
from histopilot.application.shift import ShiftService
from histopilot.application.training import TrainingService
from histopilot.storage.project_lock import StorageError

METRICS = (("auroc", "AUROC"), ("auprc", "AUPRC"), ("balancedAccuracy", "Balanced accuracy"))
RESEARCH_USE = (
    "Research use only. This model is not a medical device and has not been validated for "
    "diagnosis, treatment or any decision about a patient."
)


def _intervals(details):
    """Point estimates and 95% intervals of a scored metrics block, whatever its version."""
    patient = details.get("patient") or details.get("selected") or {}
    intervals = dict(patient.get("confidenceIntervals") or {})
    uncertainty = (details.get("patientAnalysis") or {}).get("uncertainty") or {}
    for name, value in (uncertainty.get("intervals") or {}).items():
        intervals.setdefault(name, value)
    return patient, intervals


def _metric_rows(details):
    patient, intervals = _intervals(details)
    rows = []
    for key, label in METRICS:
        if patient.get(key) is None:
            continue
        interval = intervals.get(key) or {}
        rows.append(
            {
                "metric": label,
                "value": patient[key],
                "lower": interval.get("lower"),
                "upper": interval.get("upper"),
            }
        )
    return rows, patient.get("count")


def _optional(call):
    try:
        return call()
    except StorageError:
        return None


def model_card(store, filesystem, predictor_id):
    predictor = PredictorService(store, filesystem).get(predictor_id)
    manifest = predictor["manifest"]
    experiment = manifest.get("experiment") or {}
    recipe = manifest.get("recipe") or {}
    target = manifest["target"]
    batch_id = manifest.get("batchId")
    development, development_count = [], None
    seeds = []
    if batch_id:
        results = _optional(lambda: TrainingService(store, filesystem).results(batch_id)) or {}
        for candidate in results.get("candidates", []):
            if candidate.get("candidateId") != manifest.get("candidateId"):
                continue
            seeds.append(candidate.get("trainingSeed"))
            if candidate.get("trainingSeed") == manifest.get("trainingSeed") or not development:
                development, development_count = _metric_rows(candidate.get("metricDetails") or {})
    runs = EvaluationRunService(store, filesystem)
    shifts = ShiftService(store, filesystem)
    evaluations = []
    for item in runs.list()["items"]:
        if item["manifest"].get("predictorId") != predictor_id:
            continue
        result = (item.get("execution") or {}).get("result") or {}
        rows, count = _metric_rows(result.get("metrics") or {})
        calibration = _optional(lambda item=item: runs.recalibration(item["id"]))
        shift = _optional(lambda item=item: shifts.status(item["id"]))
        report = (shift or {}).get("report") or {}
        evaluations.append(
            {
                "runId": item["id"],
                "name": item["manifest"].get("name"),
                "cohortId": item["manifest"].get("cohortId"),
                "cohort": (item["manifest"].get("cohort") or {}).get("name")
                or item["manifest"].get("cohortId"),
                "status": (item.get("execution") or {}).get("status"),
                "patients": count,
                "metrics": rows,
                # The cohort's risks as predicted, before any recalibration.
                "calibration": ((calibration or {}).get("cohort") or {}).get("original"),
                "shift": report.get("summary"),
                "shiftAnalysisId": (shift or {}).get("analysisId") if report else None,
            }
        )
    checkpoints = manifest.get("checkpoints") or []
    runtime = manifest.get("runtime") or {}
    limitations = [
        (
            f"Trained on {development_count or 'the'} development "
            f"{target.get('unit', 'patient')}s of one dataset; how it performs on other "
            "populations, scanners or staining is known only from the evaluations listed here."
        ),
    ]
    if len(set(seeds)) <= 1:
        limitations.append(
            "One training seed: how much the model would change if retrained is unmeasured."
        )
    for row in evaluations:
        shift = row["shift"]
        if shift and shift.get("verdict") in {"moderate", "strong"}:
            limitations.append(
                f"{row['cohort']}: {shift['verdictText']} Treat its metrics as specific to "
                "that cohort."
            )
    return {
        "schemaVersion": 1,
        "generatedBy": f"HistoPilot {__version__}",
        "predictor": {
            "id": predictor["id"],
            "contentHash": predictor["contentHash"],
            "name": manifest.get("name"),
            "method": manifest.get("method"),
            "lifecycleState": predictor.get("lifecycleState"),
        },
        "model": {
            "architecture": recipe.get("model"),
            "inputMode": recipe.get("inputMode", "image"),
            "aggregation": manifest.get("aggregation"),
            "patientAggregation": manifest.get("patientAggregation"),
            "trainingSeed": manifest.get("trainingSeed"),
            "splitSeed": manifest.get("splitSeed"),
            "seedsTrained": sorted({seed for seed in seeds if seed is not None}),
            "checkpoints": [
                {"sha256": row.get("sha256"), "bytes": row.get("bytes")} for row in checkpoints
            ],
            "recipe": {
                key: recipe[key]
                for key in (
                    "learningRate",
                    "weightDecay",
                    "epochs",
                    "maxEpochs",
                    "bagSize",
                    "batchSize",
                    "optimizer",
                )
                if key in recipe
            },
        },
        "task": {
            "field": target.get("field"),
            "kind": target.get("task"),
            "unit": target.get("unit"),
            "classes": target.get("classes"),
            "positiveClass": target.get("positiveClass"),
        },
        "development": {
            "experimentId": manifest.get("experimentId"),
            "experiment": experiment.get("name"),
            "batchId": batch_id,
            "datasetId": manifest.get("datasetId"),
            "units": development_count,
            "evidence": "Out-of-fold predictions of cross-validation on the development set.",
            "metrics": development,
        },
        "evaluations": evaluations,
        "intendedUse": RESEARCH_USE,
        "limitations": limitations,
        "provenance": {
            "trainingPlanHash": manifest.get("trainingPlanHash"),
            "runtime": runtime.get("versions"),
            "python": runtime.get("pythonVersion") or runtime.get("python"),
        },
    }


def _value(row):
    if row.get("lower") is None:
        return f"{row['value']:.3f}"
    return f"{row['value']:.3f} ({row['lower']:.3f}–{row['upper']:.3f})"


def render_markdown(card):
    """The card as Markdown, for a methods supplement or a registry entry."""
    predictor, model, task = card["predictor"], card["model"], card["task"]
    lines = [
        f"# Model card: {predictor['name'] or predictor['id']}",
        "",
        f"*{card['generatedBy']} · predictor `{predictor['id']}`*",
        "",
        "## Model",
        "",
        (
            f"- Architecture: {model['architecture']} ({model['inputMode']} input), "
            f"{predictor['method'] or 'single'} predictor"
        ),
        f"- Checkpoints: {len(model['checkpoints'])}, each pinned by SHA-256",
        (
            f"- Training seed {model['trainingSeed']}, split seed {model['splitSeed']}; seeds "
            f"trained for this configuration: {', '.join(map(str, model['seedsTrained'])) or '—'}"
        ),
        "",
        "## Task",
        "",
        (
            f"- Predicts `{task['field']}` per {task['unit']}: "
            f"{' vs '.join(task['classes'] or [])} (positive class {task['positiveClass']})"
        ),
        "",
        "## Development",
        "",
        f"- Experiment: {card['development']['experiment']} (`{card['development']['experimentId']}`)",
        f"- {card['development']['evidence']} {card['development']['units'] or '—'} units.",
        "",
    ]
    if card["development"]["metrics"]:
        lines += ["| Metric | Value (95% CI) |", "| --- | --- |"]
        lines += [f"| {row['metric']} | {_value(row)} |" for row in card["development"]["metrics"]]
        lines.append("")
    lines += ["## Evaluations", ""]
    if not card["evaluations"]:
        lines += ["No run of Apply models has used this predictor yet.", ""]
    for row in card["evaluations"]:
        lines += [f"### {row['cohort']}", "", f"Run `{row['runId']}` · {row['status']}", ""]
        if row["metrics"]:
            lines += ["| Metric | Value (95% CI) |", "| --- | --- |"]
            lines += [f"| {item['metric']} | {_value(item)} |" for item in row["metrics"]]
            lines.append("")
        calibration = row.get("calibration") or {}
        if calibration.get("brierScore") is not None or calibration.get("ece") is not None:
            parts = [
                f"Brier score {calibration['brierScore']:.3f}"
                if calibration.get("brierScore") is not None
                else "",
                f"ECE {calibration['ece']:.3f}" if calibration.get("ece") is not None else "",
            ]
            lines += [f"Calibration as predicted: {', '.join(p for p in parts if p)}.", ""]
        if row.get("shift"):
            from histopilot.shift import describe

            lines += [f"Dataset shift: {describe(row['shift'])}", ""]
    lines += ["## Intended use", "", card["intendedUse"], "", "## Limitations", ""]
    lines += [f"- {item}" for item in card["limitations"]]
    lines += ["", "## Provenance", ""]
    provenance = card["provenance"]
    lines += [
        f"- Predictor content hash: `{predictor['contentHash']}`",
        f"- Training plan hash: `{provenance['trainingPlanHash']}`",
    ]
    for name, version in sorted((provenance.get("runtime") or {}).items()):
        lines.append(f"- {name} {version}")
    return "\n".join(lines) + "\n"
