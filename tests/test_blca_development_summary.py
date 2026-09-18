"""Development reporting respects native OOF evidence and the frozen refit policy."""

import copy
import importlib.util
import json
import math
import sqlite3
from pathlib import Path

import pytest
from test_training_oof_exports import execution, exports

from histopilot.application.clinical import clinical_report
from histopilot.application.training_exports import training_oof_csv
from histopilot.schemas.clinical import ClinicalSelection
from histopilot.storage.project_lock import StorageError
from histopilot.workers.packing_process import write_json

__all__ = ["execution", "exports"]

spec = importlib.util.spec_from_file_location(
    "blca_development_summary", Path(__file__).resolve().parents[1] / "scripts/blca_development_summary.py"
)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)

TARGET = {"task": "binary_classification", "unit": "slide", "classes": ["high", "low"],
          "positiveClass": "high"}


def records():
    return [{"slideId": f"s{i}", "patientId": f"s{i}", "patientIdSource": "slide_fallback",
             "labelIndex": label, "label": TARGET["classes"][label],
             "probabilities": [p, 1 - p], "logProbabilities": [math.log(p), math.log1p(-p)]}
            for i, (label, p) in enumerate([(1, .1), (1, .4), (0, .35), (0, .8)])]


def test_pooled_oof_metrics_match_native_slide_report_with_reversed_class_order():
    rows = records()
    original = copy.deepcopy(rows)
    actual = helper.pooled_metrics(rows, TARGET, .5)
    report = clinical_report({"classOrder": TARGET["classes"], "records": rows}, TARGET,
                             {"decisionThreshold": .5, "patientAggregation": "mean"},
                             ClinicalSelection(evaluationId="configuration-" + "a" * 64, unit="slide"))
    for local, native in (("auroc", "rocAuc"), ("auprc", "averagePrecision"),
                          ("brierScore", "brierScore"), ("logLoss", "logLoss")):
        assert actual[local] == pytest.approx(report["metrics"][native])
    for field in ("accuracy", "balancedAccuracy", "sensitivity", "specificity", "ppv", "npv"):
        assert actual[field] == pytest.approx(report["operatingPoint"][field])
    assert actual["confusionMatrix"] == [[1, 1], [0, 2]]
    assert actual["macroF1"] == pytest.approx((2 / 3 + 4 / 5) / 2)
    assert actual["confidenceIntervals"] is None
    assert report["uncertainty"]["method"] == "unavailable"
    assert rows == original


def test_log_probabilities_preserve_auc_order_when_probabilities_round_to_one():
    rows = records()[:2]
    for row, label, score in zip(rows, (1, 0), (100, 200), strict=True):
        row.update(labelIndex=label, label=TARGET["classes"][label],
                   probabilities=[1.0, math.exp(-score)], logProbabilities=[0.0, -float(score)])
    result = helper.pooled_metrics(rows, TARGET, .5)
    assert result["auroc"] == result["auprc"] == 1.0
    assert result["logLoss"] == 50.0
    rounded = copy.deepcopy(rows)
    for row in rounded:
        row["probabilities"] = [1.0, 0.0]
        row.pop("logProbabilities")
    assert helper.pooled_metrics(rounded, TARGET, .5)["auroc"] == .5
    assert helper.pooled_metrics(rounded, TARGET, .5)["logLoss"] is None


def test_coverage_accepts_native_authenticated_fivefold_oof_without_mutating_project(exports, monkeypatch):
    service, args, folder, path = exports
    batch = service.store.get_configuration(args[0])
    protocol = service.store.get_configuration(batch["manifest"]["spec"]["inputs"]["protocolId"])
    before = {file: file.read_bytes() for file in service.store.folder.rglob("*") if file.is_file()}
    read_only = helper.EvidenceStore(service.store.folder, service.store.project_id)
    monkeypatch.setattr(read_only, "initialize", lambda: pytest.fail("Reporting must never initialize/recover the project"))
    csv_data = training_oof_csv(read_only, *args, "slide")
    document = json.loads(path.read_text())
    result = helper.coverage(document["records"], protocol["manifest"]["memberships"], [], expected_count=30)
    assert len(result) == 30
    assert csv_data.decode().count("\n") == 31
    assert before == {file: file.read_bytes() for file in service.store.folder.rglob("*") if file.is_file()}


def test_read_only_snapshot_includes_committed_wal_without_creating_or_changing_sidecars(exports):
    service, args, _, _ = exports
    manifest = {**service.store.get_configuration(args[0])["manifest"], "marker": "committed-in-wal"}
    with sqlite3.connect(service.store.path) as anchor:
        anchor.execute("SELECT count(*) FROM configurations").fetchone()
        document = service.store.publish_configuration(
            manifest=manifest,
            operation_id="snapshot-wal-test",
        )
        wal = Path(str(service.store.path) + "-wal")
        assert wal.stat().st_size > 0
        before = {path: path.read_bytes() for path in service.store.folder.rglob("*") if path.is_file()}
        read_only = helper.EvidenceStore(service.store.folder, service.store.project_id)
        actual = read_only.get_configuration(document["id"])
        assert actual["contentHash"] == document["contentHash"]
        assert actual["manifest"]["marker"] == "committed-in-wal"
        assert before == {path: path.read_bytes() for path in service.store.folder.rglob("*") if path.is_file()}


def test_read_only_snapshot_retries_when_checkpointed_wal_disappears_during_copy(exports, monkeypatch):
    service, args, _, _ = exports
    document = service.store.get_configuration(args[0])
    read_only = helper.EvidenceStore(service.store.folder, service.store.project_id)
    wal = Path(str(service.store.path) + "-wal")
    wal.write_bytes(b"already-checkpointed-wal")
    original = read_only._read_file

    def concurrent_cleanup(path, maximum):
        if Path(path) == wal:
            wal.unlink()
            raise StorageError("A required storage file is missing.", "STORAGE_CORRUPT")
        return original(path, maximum)

    monkeypatch.setattr(read_only, "_read_file", concurrent_cleanup)
    assert read_only.get_configuration(document["id"])["contentHash"] == document["contentHash"]


@pytest.mark.parametrize("damage", ["duplicate", "missing", "extra", "external", "label", "source", "fold_label"])
def test_coverage_rejects_non_oof_identity_and_external_leakage(damage):
    rows = records()
    members = [{key: row[key] for key in ("slideId", "patientId", "patientIdSource", "label")}
               for row in rows]
    external = []
    if damage == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    elif damage == "missing":
        rows.pop()
    elif damage == "extra":
        rows.append({**rows[0], "slideId": "foreign"})
    elif damage == "external":
        external = [members[0]]
    elif damage == "label":
        rows[0]["label"] = "other"
    elif damage == "source":
        rows[0]["patientIdSource"] = "verified"
    else:
        members.append({**members[0], "label": "other"})
    with pytest.raises(ValueError):
        helper.coverage(rows, members, external, expected_count=4)


def refit():
    folds = [{"runId": f"run{i}", "bestEpoch": epoch} for i, epoch in enumerate([3, 5, 8, 10, 12])]
    manifest = {"epochBudget": {"percentile": 50.0, "epochs": 8, "foldBestEpochs": copy.deepcopy(folds),
                               "interpolation": "linear", "rounding": "ceil", "epochIndexing": "one_based"},
                "recipe": {"maxEpochs": 8, "minEpochs": 8, "earlyStopping": False}}
    return manifest, folds


def test_refit_budget_uses_median_best_epochs_not_stopped_epochs():
    manifest, folds = refit()
    for row in folds:
        row["epochsCompleted"] = row["bestEpoch"] + 8
    assert helper.verify_refit_budget(manifest, folds) == 8


@pytest.mark.parametrize("damage", ["epoch", "percentile", "folds", "duplicate", "recipe", "stopping"])
def test_changed_refit_epoch_policy_is_rejected(damage):
    manifest, folds = refit()
    if damage in {"epoch", "percentile"}:
        manifest["epochBudget"]["epochs" if damage == "epoch" else "percentile"] = 10
    elif damage == "folds":
        manifest["epochBudget"]["foldBestEpochs"][0]["bestEpoch"] = 20
    elif damage == "duplicate":
        manifest["epochBudget"]["foldBestEpochs"].append(manifest["epochBudget"]["foldBestEpochs"][0])
    elif damage == "recipe":
        manifest["recipe"]["maxEpochs"] = 40
    else:
        manifest["recipe"]["earlyStopping"] = True
    with pytest.raises(ValueError):
        helper.verify_refit_budget(manifest, folds)


@pytest.mark.parametrize("best_epoch,accepted", [(2, True), (3, False)])
def test_best_epoch_history_matches_first_strict_float32_optimum(tmp_path, monkeypatch, best_epoch, accepted):
    run = {"id": "run1", "trainingSeed": 42}
    folder = tmp_path / "runs" / "run1"
    folder.mkdir(parents=True)
    receipt = {"bestEpoch": best_epoch, "epochsCompleted": 3, "checkpointMetric": "validation_loss",
               "checkpointUnit": "slide", "bestValidationScore": .4000000059604645}
    write_json(folder / "result.json", receipt)
    write_json(folder / "plan.json", {"data": {"memberships": [{"partition": "train"}]}})
    history = {"truncated": False, "totalRows": 3, "rows": [
        {"epoch": i + 1, "validation": {"loss": loss}} for i, loss in enumerate([.8, .4, .4000000001])]}
    write_json(folder / "history.json", [{"epoch": row["epoch"] - 1,
                                         "validation": {**row["validation"], "count": 6}}
                                        for row in history["rows"]])
    monkeypatch.setattr(helper, "training_history", lambda *_: history)
    args = (None, "batch", run, {"id": "split1", "fold": 0, "seed": 42}, {"maxEpochs": 40}, tmp_path)
    if accepted:
        result = helper._fold_evidence(*args)
        assert result["bestEpoch"] == 2
        assert result["bestValidationMetrics"]["loss"] == .4
    else:
        with pytest.raises(ValueError, match="validation history"):
            helper._fold_evidence(*args)
