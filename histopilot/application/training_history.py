"""Bounded, read-only epoch history for one run in a frozen development batch."""

import json
import math

from histopilot.application.training_patience import patience_summary
from histopilot.storage.project_lock import StorageError, _reject_symlink_components
from histopilot.storage.scientific import ScientificStore

MAX_HISTORY_BYTES = 32 * 1024**2
MAX_HISTORY_ROWS = 100000
DISPLAY_HISTORY_ROWS = 2000
METRICS = ("loss", "accuracy", "auroc", "auprc", "balancedAccuracy", "macroF1")


def _finite(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("History numbers must be finite.")
    return number


def _number(value):
    if value is not None and (type(value) not in {int, float} or not math.isfinite(value)):
        raise ValueError("Invalid history metric.")
    return value


def training_history(store, batch_id: str, run_id: str) -> dict:
    """Authorize by frozen membership before constructing paths; never load Torch.

    History stores zero-based epochs. This API returns one-based epochs, matching
    the progress snapshot. Missing or invalid optional history never hides state.
    """
    batch = store.get_configuration(batch_id)
    manifest = batch["manifest"]
    if manifest.get("kind") != "mil-batch":
        raise StorageError("Choose a frozen development batch.", "INVALID_BATCH", 422)
    run = next((row for row in manifest["runs"] if row["id"] == run_id), None)
    if run is None:
        raise StorageError("This run does not belong to the batch.", "TRAINING_RUN_NOT_FOUND", 404)
    # Frozen IDs are generated internally, but do not trust corrupt stored metadata
    # to turn a read-only metrics endpoint into arbitrary filesystem access.
    for identity in (batch_id, run_id):
        if not identity or identity in {".", ".."} or any(c in identity for c in ("/", "\\", "\0")):
            raise StorageError("The frozen run identity is invalid.", "TRAINING_RUN_INVALID", 409)
    candidate = next(
        (item for item in manifest["configurations"] if item["id"] == run["candidateId"]), None
    )
    recipe = candidate.get("recipe", {}) if candidate else {}
    if not isinstance(recipe, dict):
        raise StorageError("The frozen run recipe is invalid.", "TRAINING_RUN_INVALID", 409)
    maximum_epochs = recipe.get("maxEpochs")
    if type(maximum_epochs) is not int or not 1 <= maximum_epochs <= MAX_HISTORY_ROWS:
        raise StorageError("The frozen run has no valid epoch budget.", "TRAINING_RUN_INVALID", 409)
    path = store.folder / "training" / batch_id / "runs" / run_id / "history.json"
    response = {"runId": run_id, "rows": [], "totalRows": 0, "truncated": False}
    # Older records without stopping configuration retain their original shape.
    report_stopping = "earlyStopping" in recipe or "patience" in recipe
    if report_stopping:
        response["stopping"] = patience_summary(store, manifest, run, recipe)
    try:
        _reject_symlink_components(path)
        if not path.exists():
            return response
        rows = json.loads(
            ScientificStore._read_file(path, MAX_HISTORY_BYTES),
            parse_float=_finite,
            parse_constant=_finite,
        )
        if not isinstance(rows, list) or len(rows) > maximum_epochs:
            raise ValueError("Invalid history length.")
        displayed = []
        units = set()
        for index, row in enumerate(rows):
            if (
                not isinstance(row, dict)
                or type(row.get("epoch")) is not int
                or row["epoch"] != index
            ):
                raise ValueError("History epochs must be contiguous.")
            validation = row.get("validation", {})
            if not isinstance(validation, dict):
                raise ValueError("Invalid validation history.")
            unit = row.get("checkpointUnit")
            if unit not in {None, "slide", "patient"}:
                raise ValueError("Invalid validation scoring unit.")
            if unit is not None:
                units.add(unit)
                if len(units) > 1:
                    raise ValueError("Validation scoring units changed within one run.")
            values = {
                "epoch": index + 1,
                "trainingLoss": _number(row.get("trainingLoss")),
                "validation": {key: _number(validation.get(key)) for key in METRICS},
                "learningRate": _number(row.get("learningRate")),
                "checkpointUnit": unit,
            }
            if index >= len(rows) - DISPLAY_HISTORY_ROWS:
                displayed.append(values)
        return {
            **response,
            "rows": displayed,
            "totalRows": len(rows),
            "truncated": len(rows) > DISPLAY_HISTORY_ROWS,
            **(
                {"stopping": patience_summary(store, manifest, run, recipe, rows)}
                if report_stopping
                else {}
            ),
        }
    except (OSError, ValueError, TypeError, OverflowError, RecursionError):
        return {
            **response,
            "warning": "Epoch history is unavailable because its file is invalid or cannot be read safely. Run status and saved results remain available.",
        }
