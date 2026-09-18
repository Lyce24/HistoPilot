"""Patience reports exact recorded validation checks and effective stopping policy."""

import json
from types import SimpleNamespace

import pytest

from histopilot.application.development import _hash, _plan_metadata
from histopilot.application.training_history import training_history
from histopilot.application.training_patience import patience_summary


def history(values, metric="loss", count=3):
    return [
        {"epoch": index, "validation": {metric: value, "count": count, "available": True}}
        for index, value in enumerate(values)
    ]


def recipe(**kwargs):
    return {
        "earlyStopping": True,
        "patience": 3,
        "minEpochs": 1,
        "checkpointMetric": "validation_loss",
        "earlyStoppingMinDelta": 0.0,
        "maxEpochs": 100,
        **kwargs,
    }


def summary(values, **kwargs):
    return patience_summary(None, {}, {}, recipe(**kwargs), history(values))


def test_patience_waits_for_first_validation_and_disabled_policy_needs_no_history():
    assert summary([])["remaining"] is None
    disabled = patience_summary(None, {}, {}, recipe(earlyStopping=False))
    assert disabled["status"] == "disabled" and not disabled["enabled"]
    assert summary([0.5], patience=0)["status"] == "unavailable"


def test_strict_loss_improvement_resets_only_after_sufficient_change():
    result = summary([0.8, 0.7, 0.7, 0.72])
    assert result["remaining"] == 1 and result["waitCount"] == 2 and result["epoch"] == 4
    assert summary([0.8, 0.7, 0.7, 0.72, 0.69])["remaining"] == 3
    assert summary([0.8, 0.795, 0.79], earlyStoppingMinDelta=0.02)["remaining"] == 1


def test_auc_monitor_and_latest_checkpoint_policy_keep_early_stopping():
    result = patience_summary(
        None,
        {},
        {},
        recipe(
            model="nnmil", nnmilCheckpointSelection="latest", checkpointMetric="validation_auroc"
        ),
        history([0.7, 0.71, 0.70, 0.69], "auroc"),
    )
    assert result["remaining"] == 1 and result["status"] == "tracking"


def test_float32_logging_roundtrip_prevents_false_improvement():
    # The first value logs to 0.12345679849386215 when count=3; both values
    # become the same logged float32 despite differing Python floats.
    result = summary([0.123456789, 0.123456788], patience=2)
    assert result["remaining"] == 1
    assert result["waitCount"] == 1


def test_minimum_epoch_floor_does_not_reset_patience():
    early = summary([0.6, 0.7, 0.8], patience=1, minEpochs=5)
    assert early["remaining"] == 0 and early["minEpochs"] == 5
    assert summary([0.6, 0.7, 0.8, 0.5, 0.4], patience=1, minEpochs=5)["remaining"] == 1


@pytest.mark.parametrize(
    "change",
    [
        {"loss": None},
        {"loss": float("nan")},
        {"available": False},
        {"count": None},
        {"count": 0},
        {"count": True},
    ],
)
def test_missing_or_invalid_monitor_never_guesses_remaining(change):
    rows = history([0.8, 0.7])
    rows[0]["validation"].update(change)
    result = patience_summary(None, {}, {}, recipe(), rows)
    assert result["status"] == "unavailable" and result["remaining"] is None


def test_fixed_budget_disables_patience_even_when_requested_early_stopping_is_true():
    result = summary([0.8, 0.8], fixedEpochBudget=10)
    assert result["status"] == "disabled" and result["minEpochs"] == 10


@pytest.mark.parametrize("positives,disabled", [(1, True), (3, False)])
def test_validation_positive_fallback_uses_frozen_patient_membership(positives, disabled):
    rows = [
        {"patientId": f"p{i}", "partition": "val", "label": "yes", "fold": 0, "seed": 42}
        for i in range(positives)
    ]
    # An additional slide from an existing patient does not add a positive patient.
    rows.append(rows[0].copy())
    protocol = {
        "manifest": {
            "kind": "protocol",
            "spec": {"target": {"task": "binary_classification", "positiveClass": "yes"}},
            "memberships": rows,
        }
    }
    store = SimpleNamespace(get_configuration=lambda _: protocol)
    manifest = {"spec": {"inputs": {"protocolId": "p"}}}
    run = {"splitPlanId": _hash(_plan_metadata(rows[0]))}
    result = patience_summary(
        store, manifest, run, recipe(minValidationPositives=3, fixedEpochBudget=10), history([0.8])
    )
    assert (result["status"] == "disabled") == disabled
    assert result["enabled"] is not disabled


def test_api_replays_full_history_before_chart_truncation(tmp_path):
    rows = history([0.7] * 2100)
    rows[10]["validation"]["loss"] = 0.1
    batch = {
        "manifest": {
            "kind": "mil-batch",
            "runs": [{"id": "run", "candidateId": "candidate"}],
            "configurations": [
                {"id": "candidate", "recipe": recipe(patience=3000, maxEpochs=5000)}
            ],
        }
    }
    store = SimpleNamespace(folder=tmp_path, get_configuration=lambda _: batch)
    path = tmp_path / "training/batch/runs/run/history.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(rows))
    before = path.read_bytes()
    result = training_history(store, "batch", "run")
    assert result["truncated"] and len(result["rows"]) == 2000
    assert result["stopping"]["waitCount"] == 2089
    assert result["stopping"]["remaining"] == 911
    assert path.read_bytes() == before
