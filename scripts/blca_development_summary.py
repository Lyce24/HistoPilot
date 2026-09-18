"""Read authenticated BLCA development evidence without launching or changing work.

The JSON and OOF CSV remain beside the private run state. Console output contains
aggregate slide-level results only. These are exploratory development results,
not independent test performance or verified patient-level evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import sys
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from histopilot.application.clinical import (  # noqa: E402
    _operating_point,
    _ranking_curves,
    _validate_records,
)
from histopilot.application.feature_bundles import _hash  # noqa: E402
from histopilot.application.predictors import PredictorService, read_evidence  # noqa: E402
from histopilot.application.refits import epoch_budget  # noqa: E402
from histopilot.application.training_exports import training_oof_csv  # noqa: E402
from histopilot.application.training_history import training_history  # noqa: E402
from histopilot.scoring import class_ranking_score  # noqa: E402
from histopilot.storage.filesystem import LocalFilesystem  # noqa: E402
from histopilot.storage.project_lock import StorageError  # noqa: E402
from histopilot.storage.scientific import ScientificStore  # noqa: E402
from scripts.blca_final_analysis import atomic_bytes, read_state  # noqa: E402

EXPECTED_DEVELOPMENT_SLIDES = 62
EXPECTED_EXTERNAL_SLIDES = 76
EXPECTED_FOLDS = 5
EXPECTED_SEED = 42


class EvidenceStore(ScientificStore):
    """Read a bounded stable DB+WAL copy without even creating project sidecars.

    SQLite mode=ro can still create WAL/SHM files. A temporary copy avoids those
    writes while retaining committed WAL transactions (immutable=1 would omit
    them). Native configuration checksum and lifecycle checks remain in use.
    """

    def _snapshot(self, destination):
        paths = [self.path, Path(str(self.path) + "-wal")]

        def stamps():
            result = []
            for index, path in enumerate(paths):
                value = self._regular(path, missing_ok=index > 0)
                result.append(None if value is None else
                              (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns))
            return result

        for _ in range(3):
            before = stamps()
            # Source writes only occur through native SQLite. Stable inode/size/
            # modification/change times establish one coherent copying interval.
            try:
                content = [None if stamp is None else self._read_file(path, 64 * 1024**2)
                           for path, stamp in zip(paths, before, strict=True)]
            except (StorageError, FileNotFoundError):
                # A native SQLite connection may checkpoint and remove its WAL
                # between inspection and opening. Retry the whole snapshot only
                # when source stamps establish that concurrent change.
                if stamps() != before:
                    continue
                raise
            if stamps() == before:
                for path, data in zip(paths, content, strict=True):
                    if data is not None:
                        (destination / path.name).write_bytes(data)
                return destination / self.path.name
        raise ValueError("Scientific metadata kept changing during the bounded read-only snapshot; retry later.")

    def get_configuration(self, identity, *, include_inactive=False):
        if not include_inactive:
            self.lifecycle.assert_usable([f"configuration:{identity}"])
        with tempfile.TemporaryDirectory(prefix="blca-evidence-") as temporary:
            copied = self._snapshot(Path(temporary))
            connection = sqlite3.connect(copied.as_uri() + "?mode=rw", uri=True, timeout=2)
            try:
                connection.row_factory = sqlite3.Row
                connection.execute("PRAGMA query_only=ON")
                connection.execute("PRAGMA trusted_schema=OFF")
                return self._with_version_label(connection, "configuration", self._configuration(connection, identity))
            finally:
                connection.close()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def coverage(records, memberships, external, *, expected_count=EXPECTED_DEVELOPMENT_SLIDES):
    """Assert exact, disjoint, once-per-slide coverage using frozen memberships."""
    expected = {}
    for row in memberships:
        _require(row.get("phase") != "final" and row.get("pool") != "external_test",
                 "External rows cannot enter development evidence.")
        identity = row["slideId"]
        frozen = {key: row.get(key) for key in ("slideId", "patientId", "patientIdSource", "label")}
        _require(identity not in expected or expected[identity] == frozen,
                 "Development identity or label changes across folds.")
        expected[identity] = frozen
    identities = [row["slideId"] for row in records]
    _require(len(expected) == expected_count and len(identities) == expected_count
             and len(set(identities)) == expected_count and set(identities) == set(expected),
             "OOF predictions must cover every development slide exactly once.")
    external_ids = {row["slideId"] for row in external}
    _require(not external_ids.intersection(expected), "External slides overlap development.")
    for row in records:
        frozen = expected[row["slideId"]]
        _require(all(row.get(key) == frozen[key] for key in ("patientId", "label")),
                 "OOF identity and label differ from frozen development.")
        if "patientIdSource" in row:
            _require(row["patientIdSource"] == frozen["patientIdSource"],
                     "OOF patient identity provenance changed.")
    return expected


def pooled_metrics(records, target, threshold):
    """Reuse clinical scoring, retaining native log-probability ranking precision."""
    classes = target["classes"]
    _require(target["task"] == "binary_classification" and len(classes) == 2,
             "This BLCA report expects the frozen binary target.")
    _validate_records(records, classes)
    _require(records and all(row["labelIndex"] is not None for row in records),
             "OOF assessment requires labels for every slide.")
    positive = classes.index(target["positiveClass"])
    labels = [row["labelIndex"] == positive for row in records]
    probabilities = [row["probabilities"][positive] for row in records]
    scores = [class_ranking_score(row, positive) for row in records]
    _, _, auc, ap = _ranking_curves(labels, scores, probabilities)
    point = _operating_point(sorted(p for p, y in zip(probabilities, labels, strict=True) if y),
                             sorted(p for p, y in zip(probabilities, labels, strict=True) if not y),
                             threshold)
    confusion = [[0, 0], [0, 0]]
    for row, probability in zip(records, probabilities, strict=True):
        prediction = positive if probability >= threshold else 1 - positive
        confusion[row["labelIndex"]][prediction] += 1
    losses = []
    for row in records:
        index = row["labelIndex"]
        if row.get("logProbabilities") is not None:
            losses.append(-row["logProbabilities"][index])
        elif row["probabilities"][index] > 0:
            losses.append(-math.log(row["probabilities"][index]))
        else:
            losses = []  # No arbitrary epsilon can recover an underflowed log probability.
            break
    f1s = []
    for index in range(2):
        tp = confusion[index][index]
        denominator = 2 * tp + confusion[index][1 - index] + confusion[1 - index][index]
        f1s.append(2 * tp / denominator if denominator else 0.0)
    return {
        "unit": "slide", "count": len(records), "classOrder": classes,
        "positiveClass": target["positiveClass"], "decisionThreshold": threshold,
        "classCounts": dict(Counter(row["label"] for row in records)),
        "confusionMatrix": confusion, "auroc": auc, "auprc": ap,
        "accuracy": point["accuracy"], "balancedAccuracy": point["balancedAccuracy"],
        "macroF1": math.fsum(f1s) / 2,
        "sensitivity": point["sensitivity"], "specificity": point["specificity"],
        "ppv": point["ppv"], "npv": point["npv"],
        "brierScore": math.fsum((p - int(y)) ** 2 for p, y in zip(probabilities, labels, strict=True)) / len(records),
        "logLoss": math.fsum(losses) / len(records) if losses else None,
        "confidenceIntervals": None,
    }


def _fold_evidence(store, batch_id, run, split, recipe, folder):
    run_folder = folder / "runs" / run["id"]
    receipt = read_evidence(run_folder / "result.json", run_folder)
    plan = read_evidence(run_folder / "plan.json", run_folder)
    history = training_history(store, batch_id, run["id"])
    _require(not history.get("warning") and not history["truncated"] and history["rows"],
             "Complete validation history is required to verify best checkpoint epochs.")
    _require(receipt.get("checkpointMetric") == "validation_loss"
             and receipt.get("checkpointUnit") == "slide",
             "The fold did not select its checkpoint by slide validation loss.")
    completed, best_epoch = receipt["epochsCompleted"], receipt["bestEpoch"]
    _require(type(best_epoch) is int and 1 <= best_epoch <= completed <= recipe["maxEpochs"]
             and completed == history["totalRows"], "Fold epoch receipts disagree with history.")
    values = [row["validation"]["loss"] for row in history["rows"]]
    _require(all(type(value) in {int, float} and math.isfinite(value) for value in values),
             "Validation history needs a finite loss at every epoch.")
    # Lightning's epoch mean first casts the scalar to float32, then multiplies
    # and divides by validation count in float32 (see training_patience.py).
    raw = json.loads(ScientificStore._read_file(run_folder / "history.json", 32 * 1024**2))
    _require(len(raw) == len(values), "Validation history changed during reporting.")
    counts = [row["validation"].get("count") for row in raw]
    _require(all(type(count) is int and count > 0 for count in counts),
             "Validation sample counts are required to replay checkpoint selection.")
    rounded = [float(np.float32(np.float32(np.float32(value) * np.float32(count)) / np.float32(count)))
               for value, count in zip(values, counts, strict=True)]
    _require(best_epoch == rounded.index(min(rounded)) + 1
             and math.isclose(receipt["bestValidationScore"], rounded[best_epoch - 1],
                              rel_tol=1e-6, abs_tol=1e-8),
             "Best checkpoint epoch or score does not match validation history.")
    counts = Counter(row["partition"] for row in plan["data"]["memberships"])
    return {
        "runId": run["id"], "splitPlanId": split["id"], "fold": split["fold"],
        "trainingSeed": run["trainingSeed"], "splitSeed": split["seed"],
        "slideCounts": dict(counts), "bestEpoch": best_epoch, "epochsCompleted": completed,
        "bestValidationScore": receipt["bestValidationScore"],
        "bestValidationMetrics": history["rows"][best_epoch - 1]["validation"],
        "checkpointMetric": receipt["checkpointMetric"], "checkpointUnit": receipt["checkpointUnit"],
        "receiptHash": _hash(receipt), "runPlanHash": _hash(plan),
    }


def verify_refit_budget(manifest, folds):
    expected = epoch_budget([row["bestEpoch"] for row in folds], 50.0)
    budget = manifest["epochBudget"]
    evidence = {row["runId"]: row["bestEpoch"] for row in budget["foldBestEpochs"]}
    _require(len(evidence) == len(budget["foldBestEpochs"]) == len(folds)
             and evidence == {row["runId"]: row["bestEpoch"] for row in folds},
             "Refit best-epoch evidence differs from completed folds.")
    _require(budget["percentile"] == 50.0 and budget["epochs"] == expected
             and budget["interpolation"] == "linear" and budget["rounding"] == "ceil"
             and budget["epochIndexing"] == "one_based",
             "The frozen refit budget differs from the median fold best epoch.")
    _require(manifest["recipe"]["maxEpochs"] == expected
             and manifest["recipe"]["minEpochs"] == expected
             and manifest["recipe"]["earlyStopping"] is False,
             "The refit recipe differs from its fixed epoch budget.")
    return expected


def _predictor_evidence(store, filesystem, state, batch, candidate, folds, members):
    identities = state.get("predictorIds", {})
    result = {"status": "awaiting_saved_predictors", "ensemble": None, "refit": None}
    service = PredictorService(store, filesystem)
    for method in ("ensemble", "refit"):
        if not identities.get(method):
            continue
        predictor = store.get_configuration(identities[method])
        manifest = predictor["manifest"]
        _require(manifest.get("kind") == "frozen-predictor" and manifest.get("method") == method
                 and manifest.get("batchId") == batch["id"]
                 and manifest.get("candidateId") == candidate["id"]
                 and manifest.get("trainingSeed") == EXPECTED_SEED
                 and manifest.get("splitSeed") == EXPECTED_SEED,
                 "The saved predictor belongs to a different development group.")
        service.verify_checkpoints(predictor)
        checkpoints = manifest["checkpoints"]
        if method == "ensemble":
            by_run = {row["runId"]: row for row in checkpoints}
            _require(len(by_run) == len(checkpoints) == len(folds)
                     and set(by_run) == {row["runId"] for row in folds},
                     "The ensemble must contain each of the five completed folds once.")
            for row in folds:
                _require(all(by_run[row["runId"]][key] == row[key]
                             for key in ("receiptHash", "runPlanHash", "splitPlanId")),
                         "The ensemble differs from authenticated fold evidence.")
            result[method] = {"predictorId": predictor["id"], "checkpointCount": len(checkpoints),
                              "checkpointsVerified": True}
            continue
        expected = verify_refit_budget(manifest, folds)
        _require(len(checkpoints) == 1, "The refit must publish one checkpoint.")
        refit_id = manifest["refitId"]
        refit = store.get_configuration(refit_id)
        folder = store.folder / "compute-jobs" / refit_id
        receipt = read_evidence(folder / "result.json", folder)
        plan = read_evidence(folder / "plan.json", folder)
        status = read_evidence(folder / "state.json", folder)
        _require(status.get("status") == "completed" and status.get("result") == receipt
                 and receipt.get("state") == "succeeded" and receipt.get("runId") == refit_id
                 and receipt.get("epochsCompleted") == expected
                 and _hash(receipt) == checkpoints[0]["receiptHash"]
                 and _hash(plan) == checkpoints[0]["runPlanHash"] == status.get("planHash")
                 and plan.get("recordContentHash") == refit["contentHash"]
                 and all(plan.get(key) == value for key, value in refit["manifest"]["planTemplate"].items()),
                 "Refit execution receipts differ from its frozen plan or checkpoint.")
        training = plan["data"]["memberships"]
        _require(len(training) == len(members) == manifest["trainingSlideCount"]
                 and {row["slideId"] for row in training} == set(members),
                 "Refit must train on every development slide exactly once.")
        for row in training:
            _require(row["partition"] == "train" and row["phase"] == "refit"
                     and row["pool"] == "development"
                     and all(row[key] == members[row["slideId"]][key] for key in ("patientId", "label")),
                     "Refit contains changed identities, labels, or external rows.")
        result[method] = {"predictorId": predictor["id"], "refitId": refit_id,
                          "epochsCompleted": expected, "medianBestEpoch": expected,
                          "epochBudget": manifest["epochBudget"], "trainingSlideCount": len(training),
                          "checkpointsVerified": True, "receiptHash": _hash(receipt), "runPlanHash": _hash(plan)}
    if all(result[key] is not None for key in ("ensemble", "refit")):
        result["status"] = "verified"
    return result


def summarize(state_path):
    """Save actual completed OOF evidence; pending predictors remain explicitly pending."""
    state_path = Path(state_path).resolve(strict=True)
    state = read_state(state_path)
    project = Path(state["projectPath"])
    _require((project / "histopilot-state.sqlite").is_file(), "The real project must already exist.")
    store = EvidenceStore(project, state["projectId"])
    filesystem = LocalFilesystem((project,))
    batch = store.get_configuration(state["batchId"])
    manifest = batch["manifest"]
    _require(manifest.get("kind") == "mil-batch" and len(manifest["configurations"]) == 1,
             "This report requires the single fixed BLCA development candidate.")
    candidate = manifest["configurations"][0]
    recipe = candidate["recipe"]
    _require(recipe["model"] == "abmil" and recipe["checkpointMetric"] == "validation_loss"
             and recipe["decisionThreshold"] == 0.5,
             "The frozen BLCA recipe differs from the prespecified scoring policy.")
    splits = {row["id"]: row for row in manifest["splitPlans"]}
    runs = manifest["runs"]
    _require(len(splits) == len(manifest["splitPlans"]) == len(runs) == EXPECTED_FOLDS
             and {row["fold"] for row in splits.values()} == set(range(EXPECTED_FOLDS))
             and {row["seed"] for row in splits.values()} == {EXPECTED_SEED}
             and {row["trainingSeed"] for row in runs} == {EXPECTED_SEED}
             and {row["candidateId"] for row in runs} == {candidate["id"]}
             and {row["splitPlanId"] for row in runs} == set(splits),
             "The frozen BLCA plan must have exactly five folds and split/training seed 42.")
    _require(manifest["spec"]["inputs"]["protocolId"] == state["protocolId"],
             "The batch differs from the study protocol.")
    protocol = store.get_configuration(state["protocolId"])["manifest"]
    cohort = store.get_configuration(state["cohortId"])["manifest"]
    target = protocol["spec"]["target"]
    _require(target["unit"] == "slide", "BLCA fallback identities require slide-level reporting.")
    _require(len(cohort["memberships"]) == len({row["slideId"] for row in cohort["memberships"]})
             == EXPECTED_EXTERNAL_SLIDES,
             "The frozen external cohort must retain all 76 grade-2 slides.")
    # This validator follows frozen memberships through every native fold receipt,
    # prediction artifact, and the OOF analysisInputHash before releasing the CSV.
    csv_data = training_oof_csv(store, batch["id"], candidate["id"], EXPECTED_SEED, EXPECTED_SEED, "slide")
    key = hashlib.sha256(f"{candidate['id']}/{EXPECTED_SEED}/{EXPECTED_SEED}".encode()).hexdigest()[:24]
    folder = project / "training" / batch["id"]
    oof_path = folder / f"oof-{key}.json"
    document = read_evidence(oof_path, folder)
    records = document["records"]
    members = coverage(records, protocol["memberships"], cohort["memberships"])
    _require(all(row["patientIdSource"] == "slide_fallback" for row in members.values()),
             "The report's identity limitation no longer matches frozen provenance.")
    folds = [_fold_evidence(store, batch["id"], run, splits[run["splitPlanId"]], recipe, folder)
             for run in sorted(runs, key=lambda row: splits[row["splitPlanId"]]["fold"])]
    predictors = _predictor_evidence(store, filesystem, state, batch, candidate, folds, members)
    metrics = pooled_metrics(records, target, recipe["decisionThreshold"])
    saved_metrics = document.get("summary", {}).get("slide")
    if saved_metrics is not None:
        _require(document.get("analysisSummaryHash") == _hash(document["summary"]),
                 "Saved native OOF summary changed.")
        for key in ("accuracy", "balancedAccuracy", "macroF1", "auroc", "auprc"):
            _require(saved_metrics.get(key) is None and metrics[key] is None
                     or saved_metrics.get(key) is not None and metrics[key] is not None
                     and math.isclose(saved_metrics[key], metrics[key], rel_tol=1e-6, abs_tol=1e-8),
                     f"Recomputed OOF {key} differs from the native summary.")
    native_summary = document.get("summary", {})
    patient_analysis = native_summary.get("patientAnalysis", {})
    _require(native_summary.get("patient", {}).get("available") is False
             and not native_summary.get("patient", {}).get("confidenceIntervals")
             and patient_analysis.get("uncertainty", {}).get("available") is not True,
             "Fallback grouping must not produce available patient-level metrics or uncertainty.")
    output = {
        "generatedAt": datetime.now(UTC).isoformat(), "status": predictors["status"],
        "batchId": batch["id"], "candidateId": candidate["id"], "protocolId": state["protocolId"],
        "foldCount": len(folds), "splitSeed": EXPECTED_SEED, "trainingSeed": EXPECTED_SEED,
        "developmentSlideCount": len(members), "externalSlideCount": len(cohort["memberships"]),
        "assessmentCoverage": "exactly_once", "externalOverlapCount": 0,
        "verifiedPatientCount": 0, "identitySource": "slide_fallback", "folds": folds,
        "frozenAnalysisPolicy": recipe.get("analysis"),
        "nativePatientAnalysis": patient_analysis,
        "oof": {"path": str(oof_path), "analysisInputHash": document["analysisInputHash"],
                "documentHash": _hash(document), "metrics": metrics},
        "predictors": predictors,
        "limitations": [
            "Pooled OOF metrics describe internal development assessment, not independent test performance.",
            "De ID identifies slides/cases; verified patient grouping is unavailable, so no patient CIs are reported.",
            "The grade-2 cohort was used in historical model searches; this run validates an exploratory workflow.",
            "Fold validation losses selected checkpoints and are not unbiased assessment metrics.",
        ],
    }
    atomic_bytes(state_path.parent / "development-oof.csv", csv_data)
    atomic_bytes(state_path.parent / "development-summary.json",
                 (json.dumps(output, indent=2, allow_nan=False) + "\n").encode())
    return output


def aggregate(summary):
    """Allow-list console fields to keep case/run identities in private artifacts."""
    return {key: summary[key] for key in ("status", "foldCount", "splitSeed", "trainingSeed",
                                        "developmentSlideCount", "externalSlideCount", "assessmentCoverage",
                                        "externalOverlapCount", "verifiedPatientCount")} | {
        "bestEpochsByFold": [row["bestEpoch"] for row in summary["folds"]],
        "epochsCompletedByFold": [row["epochsCompleted"] for row in summary["folds"]],
        "pooledOofSlideMetrics": summary["oof"]["metrics"],
        "refitEpochs": summary["predictors"]["refit"]["epochsCompleted"] if summary["predictors"]["refit"] else None,
        "limitations": summary["limitations"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(aggregate(summarize(args.state)), indent=2, allow_nan=False))
