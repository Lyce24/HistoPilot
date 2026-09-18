"""Replay early-stopping state from complete epoch evidence without loading Torch."""

import math

import numpy as np

from histopilot.application.development import _hash, _plan_metadata
from histopilot.schemas.training_controls import resolve_stopping
from histopilot.storage.project_lock import StorageError


def stopping_recipe(store, manifest, run, recipe):
    """Apply the same explicit/small-validation epoch budget as the frozen fold."""
    target, rows = {}, []
    if recipe.get("minValidationPositives") is not None:
        protocol_id = manifest["spec"]["inputs"]["protocolId"]
        protocol = store.get_configuration(protocol_id)["manifest"]
        if protocol.get("kind") != "protocol":
            raise ValueError("The stopping policy requires its frozen protocol.")
        target = protocol["spec"]["target"]
        rows = [
            row
            for row in protocol["memberships"]
            if _hash(_plan_metadata(row)) == run["splitPlanId"]
        ]
        if not rows:
            raise ValueError("The stopping policy requires its frozen fold membership.")
    effective, _ = resolve_stopping(recipe, target, rows)
    return effective


def patience_summary(store, manifest, run, recipe, rows=None):
    """Counts remaining non-improving validation checks, including before minEpochs.

    This runtime checks validation once per completed epoch. Its Lightning mean
    logger stores a Python float as float32, multiplies by the validation count,
    then divides in float32. Replay those operations and the strict min-delta
    comparison so an apparent improvement below float32 precision does not reset
    patience. Full rows are supplied before API display truncation.
    """
    result = {
        "enabled": None,
        "patience": None,
        "remaining": None,
        "waitCount": None,
        "minEpochs": None,
        "epoch": len(rows) if rows is not None else None,
        "status": "unavailable",
        "reason": "Patience appears after a completed validation epoch.",
    }
    try:
        effective = stopping_recipe(store, manifest, run, recipe)
        enabled = effective.get("earlyStopping", True)
        patience = effective.get("patience")
        minimum = effective.get("minEpochs", 1)
        if type(enabled) is not bool or type(minimum) is not int or minimum < 0:
            raise ValueError("The recorded early-stopping settings are invalid.")
        result.update(enabled=enabled, minEpochs=minimum)
        if not enabled:
            return {
                **result,
                "status": "disabled",
                "reason": "Early stopping is disabled."
                if not effective.get("fixedEpochBudget")
                else "Training uses a fixed epoch budget; early stopping is disabled.",
            }
        if type(patience) is not int or patience < 1:
            raise ValueError("The recorded patience budget is unavailable.")
        result["patience"] = patience
        if not rows:
            return result
        metric = effective.get("checkpointMetric", "validation_loss")
        if metric not in {"validation_loss", "validation_auroc", "validation_accuracy"}:
            raise ValueError("The recorded stopping metric is unsupported.")
        delta = effective.get("earlyStoppingMinDelta", 0.0)
        if type(delta) not in {int, float} or not math.isfinite(delta) or delta < 0:
            raise ValueError("The recorded minimum improvement is invalid.")
        minimize = metric == "validation_loss"
        best, waiting = math.inf if minimize else -math.inf, 0
        for row in rows:
            validation = row["validation"]
            value, count = (
                validation.get(metric.removeprefix("validation_")),
                validation.get("count"),
            )
            if (
                validation.get("available") is False
                or type(value) not in {int, float}
                or not math.isfinite(value)
                or type(count) is not int
                or count < 1
            ):
                raise ValueError(
                    "Complete validation metrics and sample counts are needed to report patience."
                )
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                number = np.float32(count)
                current = np.float32(np.float32(np.float32(value) * number) / number)
                adjusted = np.float32(current - np.float32(-delta if minimize else delta))
            if not np.isfinite(current):
                raise ValueError(
                    "The recorded validation monitor is not finite in the training precision."
                )
            improved = adjusted < best if minimize else adjusted > best
            if improved:
                best, waiting = current, 0
            else:
                waiting += 1
        return {
            **result,
            "status": "tracking",
            "remaining": max(0, patience - waiting),
            "waitCount": waiting,
            "reason": None,
        }
    except (KeyError, TypeError, ValueError, OverflowError, StorageError) as error:
        return {**result, "reason": str(error)}
