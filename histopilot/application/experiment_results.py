"""Experiment results: the per-fold, per-seed and seed-averaged evidence of every batch.

Read-only. Each batch's frozen execution plan, run receipts and out-of-fold (OOF)
predictions are read as the workers left them; nothing is written. The summary is
cached in memory per file identity, so a finished experiment is computed once.

Every configuration reports:
- per fold: the held-out test fold metrics each run recorded, with its checkpoint epoch
  (a leave-one-site-out fold names its site; a held-out design has one plan per seed);
- per seed: the OOF metrics of one training seed (every assessed slide scored once, by
  the fold model that did not train on it), plus the spread of its folds. Every slide is
  assessed in k-fold, predefined-fold and all-site designs; a held-out assessment, or
  leave-one-site-out over selected sites, assesses only its held-out slides;
- seed average: the mean ± SD of the per-seed OOF metrics, with a 95% interval that
  resamples the design's independent units and scores all seeds on each draw;
- seed ensemble: the OOF metrics of the seeds' averaged predictions.

Batches of one experiment share their folds, so the selected configurations of two
batches are compared on the same draws (OOF) and on the same folds (fold means).
"""

from __future__ import annotations

import statistics
import threading
from collections import OrderedDict
from pathlib import Path

from histopilot import cv_summary as cv
from histopilot.application.model_experiments import ModelExperimentService
from histopilot.storage.io import read_json_bounded
from histopilot.storage.project_lock import StorageError

DEFAULT_RESAMPLES = 2000
DEFAULT_SEED = 42
CONFIDENCE_LEVEL = 0.95
# Robustness notes. They describe the evidence; they never change a result.
WEAK_RECALL = 0.5
SEED_SD_NOTE = 0.02
FOLD_RANGE_NOTE = 0.10
EARLY_EPOCH_NOTE = 2
_CACHE: OrderedDict[tuple, dict] = OrderedDict()
_CACHE_SIZE = 16
_CACHE_LOCK = threading.Lock()
_PENDING: dict[tuple, threading.Event] = {}


def experiment_results(store, filesystem, experiment_id: str) -> dict:
    record = ModelExperimentService(store, filesystem).get(experiment_id)
    batches = [batch for batch in record["batches"] if batch.get("state", "active") != "trashed"]
    key = (
        str(store.folder),
        experiment_id,
        tuple(_batch_stamp(store.folder / "training" / batch["id"], batch) for batch in batches),
    )
    while True:
        with _CACHE_LOCK:
            if key in _CACHE:
                _CACHE.move_to_end(key)
                return _CACHE[key]
            pending = _PENDING.get(key)
            if pending is None:
                pending = _PENDING[key] = threading.Event()
                break
        # Parallel first reads (tab switch plus poll) wait for one computation.
        pending.wait(timeout=120)
    try:
        summary = summarize_experiment(
            record, [_load(store.folder / "training" / batch["id"], batch) for batch in batches]
        )
        with _CACHE_LOCK:
            _CACHE[key] = summary
            while len(_CACHE) > _CACHE_SIZE:
                _CACHE.popitem(last=False)
        return summary
    finally:
        with _CACHE_LOCK:
            _PENDING.pop(key, None)
        pending.set()


def _stamp(path: Path):
    try:
        info = path.stat()
    except FileNotFoundError:
        return (path.name, None, None)
    return (path.name, info.st_mtime_ns, info.st_size)


def _batch_stamp(folder: Path, batch: dict) -> tuple:
    files = [
        folder / "plan.json",
        folder / "state.json",
        folder / "results.json",
        *sorted(folder.glob("oof-*.json")),
    ]
    return (
        batch["id"],
        batch.get("status"),
        batch.get("state"),
        tuple(_stamp(path) for path in files),
    )


def _comparison(batch: dict) -> dict | None:
    return (batch.get("manifest") or {}).get("spec", {}).get("comparison")


def _reported(
    plan: dict | None, results: dict | None, comparison: dict | None = None
) -> tuple[str | None, str]:
    """The configuration a batch reports: its declared reference arm or its validation
    choice, never an OOF ranking."""
    configurations = (plan or {}).get("configurations", [])
    if comparison is not None:
        reference = next(
            (row for row in configurations if row["number"] == comparison.get("reference", 1)),
            None,
        )
        if reference is not None:
            return reference["id"], "reference"
    selection = (results or {}).get("selection") or {}
    if selection.get("ready") and selection.get("selectedCandidateId"):
        return selection["selectedCandidateId"], "validation"
    if len(configurations) == 1:
        return configurations[0]["id"], "single"
    if configurations:
        return min(configurations, key=lambda row: row["number"])["id"], "first"
    return None, "first"


def _load(folder: Path, batch: dict) -> dict:
    """The files one batch summary needs; missing files mean the batch has not got there.

    OOF predictions are read for the reported configuration only: a large grid keeps its
    recorded per-seed metrics without reading every configuration's predictions.
    """
    loaded = {
        "batch": batch,
        "plan": None,
        "state": batch.get("execution"),
        "results": None,
        "oof": {},
    }
    if (folder / "plan.json").exists():
        loaded["plan"] = read_json_bounded(folder / "plan.json")
    if (folder / "results.json").exists():
        loaded["results"] = read_json_bounded(folder / "results.json")
        root = folder.resolve()
        comparison = _comparison(batch)
        reported, _source = _reported(loaded["plan"], loaded["results"], comparison)
        for candidate in loaded["results"].get("candidates", []):
            path = candidate.get("oofPath")
            wanted = comparison is not None or candidate["candidateId"] == reported
            if not candidate.get("complete") or not path or not wanted:
                continue
            resolved = Path(path).resolve()
            # Results name their own files; anything outside the batch folder is ignored.
            if (
                resolved.parent != root
                or not resolved.name.startswith("oof-")
                or not resolved.exists()
            ):
                continue
            try:
                records = read_json_bounded(resolved)["records"]
            except (StorageError, ValueError, KeyError, OSError):
                continue
            loaded["oof"][
                (candidate["candidateId"], candidate["trainingSeed"], candidate["splitSeed"])
            ] = records
    return loaded


def summarize_experiment(record: dict, loaded: list[dict]) -> dict:
    """Pure assembly from loaded batch files; ``experiment_results`` adds IO and caching."""
    policy = _policy(loaded)
    batches, internals = [], []
    for item in loaded:
        summary, hidden = _summarize_batch(item, policy)
        batches.append(summary)
        internals.append(hidden)
    target = next((item["plan"]["target"] for item in loaded if item["plan"]), None)
    return {
        "experimentId": record["id"],
        "target": _target(target),
        "design": _design(loaded),
        "policy": {**policy, "confidenceLevel": CONFIDENCE_LEVEL, "method": cv.METHOD},
        "primaryMetric": "auroc",
        "batches": batches,
        "comparisons": _comparisons(batches, internals, policy),
        "findings": [finding for batch in batches for finding in batch["findings"]],
    }


def _policy(loaded: list[dict]) -> dict:
    """One resampling policy per experiment, so every batch and comparison shares draws."""
    for item in loaded:
        for configuration in (item["plan"] or {}).get("configurations", []):
            analysis = configuration["recipe"].get("analysis")
            if isinstance(analysis, dict):
                return {
                    "resamples": int(analysis.get("bootstrapResamples", DEFAULT_RESAMPLES)),
                    "seed": int(analysis.get("bootstrapSeed", DEFAULT_SEED)),
                }
    return {"resamples": DEFAULT_RESAMPLES, "seed": DEFAULT_SEED}


def _target(target: dict | None) -> dict | None:
    if not target:
        return None
    return {
        name: target.get(name) for name in ("task", "unit", "classes", "positiveClass", "field")
    }


def _design(loaded: list[dict]) -> dict | None:
    plan = next((item["plan"] for item in loaded if item["plan"]), None)
    if plan is None:
        return None
    splits = plan["splitPlans"]
    split_seeds = sorted({split.get("seed", 0) for split in splits})
    folds = max(sum(split.get("seed", 0) == seed for split in splits) for seed in split_seeds)
    slide_counts = sorted({split.get("slideCount") for split in splits if split.get("slideCount")})
    # How units were assessed: one site or cohort per plan, one held-out set per seed, or folds.
    strategy = (
        "leave_one_domain_out"
        if any(split.get("domain") is not None for split in splits)
        else "held_out"
        if all(str(split.get("planId", "")).endswith("/held_out") for split in splits)
        else "folds"
    )
    return {
        "strategy": strategy,
        "splitUnit": "slide" if plan.get("splitUnit") == "slide" else "patient",
        "groupByPatient": bool(plan.get("groupByPatient")),
        "folds": folds,
        "splitSeeds": split_seeds,
        "slideCount": slide_counts[-1] if slide_counts else None,
        "resamplingUnit": "slide" if plan.get("splitUnit") == "slide" else "patient",
    }


def _summarize_batch(item: dict, policy: dict) -> tuple[dict, dict]:
    batch, plan, state, results = item["batch"], item["plan"], item["state"], item["results"]
    spec = batch.get("manifest", {}).get("spec", {})
    counts = (state or {}).get("runCounts") or {}
    summary = {
        "batchId": batch["id"],
        "name": batch.get("name") or spec.get("batchName") or batch["id"],
        "state": batch.get("state", "active"),
        "status": batch.get("status", "planned"),
        "progress": {
            "completedRuns": counts.get("completed", 0),
            "totalRuns": counts.get("total")
            or batch.get("manifest", {}).get("summary", {}).get("runCount", 0),
        },
        "selection": None,
        "selectedCandidateId": None,
        "configurations": [],
        "findings": [],
    }
    hidden = {"draws": {}}
    if plan is None:
        return summary, hidden
    target = plan["target"]
    selection = (results or {}).get("selection")
    configurations = plan["configurations"]
    comparison = _comparison(batch)
    # Never pick by OOF: without a validation choice the first configuration is shown.
    selected_id, source = _reported(plan, results, comparison)
    summary["selection"] = {
        "source": source,
        "metric": (selection or {}).get("metric"),
        "ready": bool((selection or {}).get("ready")),
        "scores": {
            row["candidateId"]: row.get("score") for row in (selection or {}).get("candidates", [])
        },
    }
    summary["selectedCandidateId"] = selected_id
    for configuration in sorted(configurations, key=lambda row: row["number"]):
        result, draws = _summarize_configuration(
            item,
            configuration,
            target,
            policy,
            detailed=comparison is not None or configuration["id"] == selected_id,
        )
        result["selected"] = configuration["id"] == selected_id
        result["validationScore"] = summary["selection"]["scores"].get(configuration["id"])
        summary["configurations"].append(result)
        hidden["draws"][configuration["id"]] = draws
    chosen = next((row for row in summary["configurations"] if row["selected"]), None)
    if chosen is not None:
        summary["findings"] = _findings(summary, chosen, target, len(configurations))
    if comparison is not None and chosen is not None:
        summary["comparison"] = _arm_contrasts(summary, hidden, chosen, comparison, policy)
    return summary, hidden


def _arm_contrasts(summary, hidden, reference, comparison, policy) -> dict:
    """Reference minus every other arm on shared draws, Holm-adjusted on the primary metric."""
    metric = comparison.get("primaryMetric", "auroc")
    contrasts = []
    for arm in summary["configurations"]:
        if arm["candidateId"] == reference["candidateId"]:
            continue
        left = hidden["draws"].get(reference["candidateId"]) or {}
        right = hidden["draws"].get(arm["candidateId"]) or {}
        row = _compare((summary, reference, left), (summary, arm, right), policy)
        p_value = None
        a, b = left.get("draws") or {}, right.get("draws") or {}
        if (
            row.get("available")
            and a.get("identities")
            and a.get("identities") == b.get("identities")
            and metric in a.get("seedAverage", {})
        ):
            p_value = cv.bootstrap_p_value(
                a["seedAverage"][metric] - b["seedAverage"][metric],
                a["seedAverage"]["valid"] & b["seedAverage"]["valid"],
            )
        contrasts.append(
            {
                "armId": arm["candidateId"],
                "armNumber": arm["number"],
                "model": arm["model"],
                "inputMode": arm["inputMode"],
                "difference": "reference_minus_arm",
                "available": row["available"],
                **(
                    {"reason": "Both arms need complete OOF results for every seed."}
                    if "reason" in row
                    else {}
                ),
                **{key: row[key] for key in ("oof", "oofInterval", "folds") if key in row},
                "pValue": p_value,
            }
        )
    for row, adjusted in zip(contrasts, cv.holm([row["pValue"] for row in contrasts]), strict=True):
        row["pValueHolm"] = adjusted
    return {
        "referenceId": reference["candidateId"],
        "referenceNumber": reference["number"],
        "primaryMetric": metric,
        "adjustment": "holm",
        "contrasts": contrasts,
    }


def _summarize_configuration(
    item: dict, configuration: dict, target: dict, policy: dict, *, detailed: bool = True
) -> tuple[dict, dict]:
    plan, state, results = item["plan"], item["state"], item["results"]
    recipe = configuration["recipe"]
    classes = target["classes"]
    threshold = recipe.get("decisionThreshold", 0.5)
    aggregation = recipe.get("patientAggregation", "mean_probabilities")
    # Folds are numbered in plan order when a plan records no fold number: older split
    # plans, and the single plan per seed of a held-out assessment.
    splits = {
        row["id"]: {**row, "fold": index if row.get("fold") is None else row["fold"]}
        for index, row in enumerate(plan["splitPlans"])
    }
    actual = {row["id"]: row for row in (state or {}).get("runs", [])}
    recorded = {
        (row["candidateId"], row["trainingSeed"], row["splitSeed"]): row
        for row in (results or {}).get("candidates", [])
    }
    grouped: dict[int, dict[int, list[dict]]] = {}
    # Older execution plans carry no run list; their state lists every planned run.
    for planned in plan.get("runs") or (state or {}).get("runs", []):
        if planned["candidateId"] != configuration["id"] or planned["splitPlanId"] not in splits:
            continue
        split = splits[planned["splitPlanId"]]
        run = actual.get(planned["id"], {**planned, "status": "planned"})
        grouped.setdefault(split.get("seed", 0), {}).setdefault(planned["trainingSeed"], []).append(
            _fold(split, run, target)
        )
    split_rows, seed_rows, ensemble_groups, fold_metrics = [], [], [], {}
    for split_seed in sorted(grouped):
        seeds = []
        for training_seed in sorted(grouped[split_seed]):
            folds = sorted(grouped[split_seed][training_seed], key=lambda row: row["fold"])
            key = (configuration["id"], training_seed, split_seed)
            candidate = recorded.get(key, {})
            records = item["oof"].get(key) if detailed else None
            oof = None
            if candidate.get("complete"):
                if records is not None:
                    oof = _strip(
                        cv.point_metrics(
                            _scoring_rows(records, target, aggregation), target, threshold
                        )
                    )
                elif (
                    isinstance(candidate.get("metrics"), dict)
                    and candidate["metrics"].get("available") is not False
                ):
                    oof = _recorded(candidate["metrics"], classes)
            seed = {
                "trainingSeed": training_seed,
                "splitSeed": split_seed,
                "complete": bool(candidate.get("complete")),
                "completedRuns": sum(row["status"] == "completed" for row in folds),
                "totalRuns": len(folds),
                "oof": oof,
                "folds": folds,
                "foldStats": {
                    name: cv.describe(row["metrics"][name] for row in folds if row["metrics"])
                    for name in cv.METRICS
                },
            }
            seeds.append(seed)
            seed_rows.append(seed)
            if oof is not None and records is not None:
                ensemble_groups.append(records)
            for row in folds:
                if row["metrics"]:
                    fold_metrics.setdefault(row["splitPlanId"], []).append(row["metrics"])
        split_rows.append(
            {
                "splitSeed": split_seed,
                "folds": [
                    {
                        "fold": row["fold"],
                        **({"domain": row["domain"]} if "domain" in row else {}),
                        "splitPlanId": row["splitPlanId"],
                        "testCount": row["testCount"],
                    }
                    for row in sorted(
                        next(iter(grouped[split_seed].values())), key=lambda row: row["fold"]
                    )
                ],
                "seeds": seeds,
            }
        )
    complete = [row for row in seed_rows if row["oof"] is not None]
    all_folds = [fold["metrics"] for row in seed_rows for fold in row["folds"] if fold["metrics"]]
    ensemble = None
    ensemble_rows = cv.seed_ensemble(
        ensemble_groups, recipe.get("ensembleAggregation", "mean_probability")
    )
    if ensemble_rows is not None:
        ensemble = _strip(
            cv.point_metrics(_scoring_rows(ensemble_rows, target, aggregation), target, threshold)
        )
    if detailed:
        intervals, draws = _intervals(ensemble_groups, ensemble_rows, target, plan, recipe, policy)
    else:
        unit = "slide" if plan.get("splitUnit") == "slide" else "patient"
        intervals, draws = (
            {
                "unit": unit,
                "resamples": policy["resamples"],
                "seed": policy["seed"],
                "note": cv.LIMITATION.format(unit=unit),
                "available": False,
                "reason": "Intervals, the seed ensemble and per-class ranking are computed for the reported configuration.",
            },
            {},
        )
    return {
        "candidateId": configuration["id"],
        "number": configuration["number"],
        "model": recipe.get("model"),
        "inputMode": recipe.get("inputMode", "image"),
        "splitSeeds": split_rows,
        "seedCount": len(complete),
        "plannedSeedCount": len(seed_rows),
        "seedAverage": {
            name: cv.describe(row["oof"][name] for row in complete) for name in cv.METRICS
        },
        "foldAverage": {name: cv.describe(fold[name] for fold in all_folds) for name in cv.METRICS},
        "foldCount": len(all_folds),
        "plannedFoldCount": sum(len(row["folds"]) for row in seed_rows),
        "perClass": _per_class(complete, classes),
        "confusion": _confusion(complete, classes),
        "ensemble": ensemble,
        "intervals": intervals,
        "complete": bool(seed_rows) and len(complete) == len(seed_rows),
    }, {"draws": draws, "foldMetrics": fold_metrics}


def _fold(split: dict, run: dict, target: dict) -> dict:
    receipt = run.get("metrics") or (run.get("result") or {}).get("metrics") or {}
    assessment = receipt.get("assessment") or {}
    selected = assessment.get("selected")
    result = run.get("result") or {}
    metrics = None
    if (
        run.get("status") == "completed"
        and isinstance(selected, dict)
        and selected.get("available") is not False
    ):
        metrics = _recorded(selected, target["classes"])
    return {
        "fold": split.get("fold", 0),
        # A leave-one-site-out fold assesses one site or cohort.
        **({"domain": split["domain"]} if split.get("domain") is not None else {}),
        "splitPlanId": split["id"],
        "runId": run.get("id"),
        "status": run.get("status", "planned"),
        "testCount": (split.get("partitions") or {}).get("test"),
        "metrics": metrics,
        "bestEpoch": result.get("bestEpoch"),
        "epochsCompleted": result.get("epochsCompleted"),
        "validationScore": result.get("bestValidationScore"),
        "checkpointMetric": result.get("checkpointMetric"),
    }


def _recorded(metrics: dict, classes: list[str]) -> dict:
    """A worker's saved metric block, with per-class rates recovered from its confusion matrix."""
    value = {name: metrics.get(name) for name in cv.METRICS}
    value.update(
        count=metrics.get("count"),
        classCounts=metrics.get("classCounts"),
        confusionMatrix=metrics.get("confusionMatrix"),
        perClass=cv.per_class_from_confusion(metrics.get("confusionMatrix"), classes)
        if metrics.get("confusionMatrix")
        else None,
    )
    return value


def _strip(metrics: dict) -> dict:
    return {
        name: value
        for name, value in metrics.items()
        if name not in {"available", "missingClasses"}
    }


def _scoring_rows(records: list[dict], target: dict, aggregation: str) -> list[dict]:
    return cv.patient_rows(records, aggregation) if target["unit"] == "patient" else records


def _per_class(complete: list[dict], classes: list[str]) -> list[dict]:
    rows = []
    for index, label in enumerate(classes):
        entries = [row["oof"]["perClass"][index] for row in complete if row["oof"].get("perClass")]
        rows.append(
            {
                "label": label,
                "support": entries[0]["support"] if entries else None,
                **{
                    name: cv.describe(entry.get(name) for entry in entries)
                    for name in ("recall", "precision", "f1", "auroc", "auprc")
                },
            }
        )
    return rows


def _confusion(complete: list[dict], classes: list[str]) -> dict | None:
    matrices = [
        row["oof"]["confusionMatrix"] for row in complete if row["oof"].get("confusionMatrix")
    ]
    if not matrices:
        return None
    size = len(classes)
    mean = [
        [statistics.fmean(matrix[i][j] for matrix in matrices) for j in range(size)]
        for i in range(size)
    ]
    rates = []
    for i in range(size):
        rows = [
            [matrix[i][j] / sum(matrix[i]) if sum(matrix[i]) else None for j in range(size)]
            for matrix in matrices
        ]
        rates.append(
            [
                statistics.fmean(row[j] for row in rows)
                if all(row[j] is not None for row in rows)
                else None
                for j in range(size)
            ]
        )
    return {"meanCounts": mean, "rowRates": rates, "seeds": len(matrices)}


def _intervals(
    groups: list[list[dict]], ensemble_rows, target: dict, plan: dict, recipe: dict, policy: dict
) -> tuple[dict, dict]:
    """Seed-average and seed-ensemble intervals over the design's independent units."""
    unit = "slide" if plan.get("splitUnit") == "slide" else "patient"
    base = {
        "unit": unit,
        "resamples": policy["resamples"],
        "seed": policy["seed"],
        "note": cv.LIMITATION.format(unit=unit),
    }
    if not groups:
        return {
            **base,
            "available": False,
            "reason": "No training seed has complete OOF predictions yet.",
        }, {}
    aggregation = recipe.get("patientAggregation", "mean_probabilities")
    threshold = recipe.get("decisionThreshold", 0.5)
    try:
        members = [_scoring_rows(rows, target, aggregation) for rows in groups]
        key = "slideId" if target["unit"] == "slide" else "patientId"
        aligned = cv._aligned(members, key)
        if aligned is None:
            raise ValueError("Training seeds scored different assessment units.")
        unit_key = "slideId" if unit == "slide" else "patientId"
        if unit == "patient" and any(
            not row.get("patientId") or row.get("patientIdSource") == "slide_fallback"
            for row in aligned[0]
        ):
            raise ValueError("Patient resampling requires verified patient identities.")
        identities = sorted({row[unit_key] for row in aligned[0]})
        bootstrap = cv.UnitBootstrap(identities, policy["resamples"], policy["seed"])
        index = cv.unit_indices(aligned[0], identities, unit_key)
        seeds = bootstrap.mean_of(
            [(cv.WeightedScorer(rows, target, threshold), index) for rows in aligned]
        )
        result = {
            **base,
            "units": len(identities),
            "identities": identities,
            "seedAverage": cv.interval_summary(
                seeds, resamples=policy["resamples"], level=CONFIDENCE_LEVEL
            ),
        }
        draws = {"seedAverage": seeds, "identities": identities, "unit": unit}
        if ensemble_rows is not None:
            ensemble = sorted(
                _scoring_rows(ensemble_rows, target, aggregation), key=lambda row: row[key]
            )
            values = bootstrap.mean_of(
                [
                    (
                        cv.WeightedScorer(ensemble, target, threshold),
                        cv.unit_indices(ensemble, identities, unit_key),
                    )
                ]
            )
            result["ensemble"] = cv.interval_summary(
                values, resamples=policy["resamples"], level=CONFIDENCE_LEVEL
            )
        result.pop("identities")
        return {**result, "available": result["seedAverage"].get("available", False)}, draws
    except ValueError as error:
        return {**base, "available": False, "reason": str(error)}, {}


def _comparisons(batches: list[dict], internals: list[dict], policy: dict) -> list[dict]:
    """Every pair of batches, on their selected configurations: OOF draws and shared folds."""
    rows = []
    selected = []
    for summary, hidden in zip(batches, internals, strict=True):
        candidate = summary["selectedCandidateId"]
        configuration = next(
            (row for row in summary["configurations"] if row["candidateId"] == candidate), None
        )
        selected.append(
            (summary, configuration, hidden["draws"].get(candidate) if candidate else None)
        )
    for left_index in range(len(selected)):
        for right_index in range(left_index + 1, len(selected)):
            left, right = selected[left_index], selected[right_index]
            rows.append(_compare(left, right, policy))
    return rows


def _compare(left, right, policy: dict) -> dict:
    (left_batch, left_config, left_hidden), (right_batch, right_config, right_hidden) = left, right
    row = {
        "leftBatchId": left_batch["batchId"],
        "rightBatchId": right_batch["batchId"],
        "leftCandidateId": left_config and left_config["candidateId"],
        "rightCandidateId": right_config and right_config["candidateId"],
        "difference": "left_minus_right",
    }
    if (
        not left_config
        or not right_config
        or not left_config["seedCount"]
        or not right_config["seedCount"]
    ):
        return {**row, "available": False, "reason": "Both batches need complete OOF results."}
    row["oof"] = {
        name: _difference(
            left_config["seedAverage"].get(name), right_config["seedAverage"].get(name)
        )
        for name in cv.METRICS
    }
    left_draws = (left_hidden or {}).get("draws", {})
    right_draws = (right_hidden or {}).get("draws", {})
    if (
        left_draws.get("identities")
        and left_draws.get("identities") == right_draws.get("identities")
        and left_draws.get("unit") == right_draws.get("unit")
    ):
        a, b = left_draws["seedAverage"], right_draws["seedAverage"]
        difference = {name: a[name] - b[name] for name in cv.BOOTSTRAP_METRICS}
        difference["valid"] = a["valid"] & b["valid"]
        row["oofInterval"] = {
            "unit": left_draws["unit"],
            "units": len(left_draws["identities"]),
            **cv.interval_summary(
                a, resamples=policy["resamples"], level=CONFIDENCE_LEVEL, difference=difference
            ),
        }
    else:
        row["oofInterval"] = {
            "available": False,
            "reason": "The two batches scored different assessment units, so draws cannot be paired.",
        }
    left_folds, right_folds = (
        (left_hidden or {}).get("foldMetrics", {}),
        (right_hidden or {}).get("foldMetrics", {}),
    )
    row["folds"] = {
        name: cv.fold_differences(
            {key: [value[name] for value in values] for key, values in left_folds.items()},
            {key: [value[name] for value in values] for key, values in right_folds.items()},
            higher_is_better=name not in cv.LOWER_IS_BETTER,
        )
        for name in cv.METRICS
    }
    return {**row, "available": True}


def _difference(left: dict | None, right: dict | None) -> dict | None:
    if not left or not right:
        return None
    return {
        "left": left["mean"],
        "right": right["mean"],
        "difference": left["mean"] - right["mean"],
    }


def _findings(
    summary: dict, configuration: dict, target: dict, configuration_count: int
) -> list[dict]:
    name, batch_id = summary["name"], summary["batchId"]
    primary = "macro AUROC" if target["task"] != "binary_classification" else "AUROC"
    notes = []

    def note(severity, code, message):
        notes.append({"severity": severity, "code": code, "message": message, "batchId": batch_id})

    if not configuration["complete"]:
        done, planned = configuration["seedCount"], configuration["plannedSeedCount"]
        note(
            "info",
            "PARTIAL_RESULTS",
            f"{name}: {done} of {planned} training seed{'s' if planned != 1 else ''} have complete OOF results. "
            "Seed averages cover completed seeds only.",
        )
    for row in configuration["perClass"]:
        recall = row.get("recall")
        if recall and recall["mean"] < WEAK_RECALL and row.get("support"):
            spread = f" ± {recall['sd']:.3f}" if recall.get("sd") is not None else ""
            note(
                "warning",
                "CLASS_RECALL_LOW",
                f"{name}: {row['label']} recall is {recall['mean']:.3f}{spread} "
                f"({row['support']} {target['unit']}s). The model seldom predicts this class correctly.",
            )
    average = configuration["seedAverage"].get("auroc")
    if configuration["seedCount"] == 1:
        note(
            "info",
            "SINGLE_TRAINING_SEED",
            f"{name}: one training seed, so seed-to-seed variation is unknown. "
            "Repeat with more training seeds before comparing close results.",
        )
    elif average and average.get("sd") is not None and average["sd"] >= SEED_SD_NOTE:
        note(
            "warning",
            "SEED_VARIATION",
            f"{name}: {primary} varies across training seeds ({average['mean']:.3f} ± {average['sd']:.3f}, "
            f"range {average['min']:.3f}–{average['max']:.3f}). A single seed is not representative.",
        )
    folds = configuration["foldAverage"].get("auroc")
    if folds and folds["n"] > 1 and folds["max"] - folds["min"] >= FOLD_RANGE_NOTE:
        note(
            "info",
            "FOLD_VARIATION",
            f"{name}: test-fold {primary} ranges from {folds['min']:.3f} to {folds['max']:.3f}. "
            "Individual folds are small; the pooled OOF value is the stable estimate.",
        )
    epochs = [
        fold["bestEpoch"]
        for split in configuration["splitSeeds"]
        for seed in split["seeds"]
        for fold in seed["folds"]
        if isinstance(fold.get("bestEpoch"), int)
    ]
    if len(epochs) >= 3 and statistics.median(epochs) <= EARLY_EPOCH_NOTE:
        note(
            "info",
            "EARLY_CHECKPOINTS",
            f"{name}: half of the folds kept a checkpoint from epoch {EARLY_EPOCH_NOTE} or earlier "
            f"(median {statistics.median(epochs):g}). Validation folds may be too small to guide stopping.",
        )
    # A declared comparison reports its reference arm by design, not for lack of a choice.
    if configuration_count > 1 and summary["selection"]["source"] not in {
        "validation",
        "reference",
    }:
        note(
            "warning",
            "SELECTION_UNAVAILABLE",
            f"{name}: no validation-based configuration choice is available; configuration "
            f"{configuration['number']} is shown. Compare configurations by validation, not OOF.",
        )
    return notes


def experiment_headlines(store, filesystem) -> dict:
    """Seed-mean OOF headline of every batch, from its small ``results.json`` only.

    The list page shows one number per experiment; the full summary (fold receipts,
    OOF predictions, intervals) is computed only when an experiment is opened.
    """
    items = []
    for record in ModelExperimentService(store, filesystem).list(summary=True)["items"]:
        batches = []
        for batch in record["batches"]:
            if batch.get("state", "active") == "trashed":
                continue
            headline = _headline(store.folder / "training" / batch["id"], batch)
            if headline is not None:
                batches.append(headline)
        if batches:
            items.append({"experimentId": record["id"], "batches": batches})
    return {"items": items}


def _headline(folder: Path, batch: dict) -> dict | None:
    path = folder / "results.json"
    if not path.exists():
        return None
    try:
        stored = read_json_bounded(path)
    except (StorageError, ValueError, OSError):
        return None
    candidates = stored.get("candidates", [])
    identities = list(dict.fromkeys(row["candidateId"] for row in candidates))
    selection = stored.get("selection") or {}
    chosen = selection.get("selectedCandidateId") if selection.get("ready") else None
    if chosen is None and identities:
        numbers = {row["candidateId"]: row.get("number") for row in selection.get("candidates", [])}
        chosen = min(
            identities,
            key=lambda identity: (numbers.get(identity) or 10**9, identities.index(identity)),
        )
    rows = [row for row in candidates if row["candidateId"] == chosen]
    complete = [
        row
        for row in rows
        if row.get("complete")
        and isinstance(row.get("metrics"), dict)
        and row["metrics"].get("available") is not False
    ]
    if not complete:
        return None
    details = next(
        (row["metricDetails"] for row in complete if isinstance(row.get("metricDetails"), dict)), {}
    )
    return {
        "batchId": batch["id"],
        "name": batch.get("name") or batch["id"],
        "status": batch.get("status"),
        "seeds": len(complete),
        "plannedSeeds": len(rows),
        "configurations": len(identities),
        "task": "binary_classification"
        if details.get("positiveClass")
        else "multiclass_classification",
        "metrics": {
            name: cv.describe(row["metrics"].get(name) for row in complete)
            for name in ("auroc", "balancedAccuracy", "macroF1", "accuracy")
        },
    }
