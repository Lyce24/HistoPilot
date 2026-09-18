"""Training-only, versioned nnMIL planning without importing the compute runtime.

The half-median heuristic follows Luo et al., arXiv:2511.14907. Unlike the
upstream dataset-wide planner, resolution uses only the actual fitting slides.
"""

import hashlib
import json
import math
from statistics import median

from histopilot.schemas.development import TrainingRecipe


def _hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _quantile(values, fraction):
    position = (len(values) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def resolve_nnmil_recipe(recipe, rows, feature_files):
    """Return the numeric recipe and fingerprint, leaving candidate intent intact."""
    is_nnmil = recipe.get("model", "abmil").lower() == "nnmil"
    automatic = recipe.get("bagSizeMode", "fixed") == "training_median"
    if not is_nnmil and not automatic:
        return recipe, None
    training = sorted(
        (row for row in rows if row["partition"] == "train"), key=lambda row: row["slideId"]
    )
    if not training:
        raise ValueError("Automatic MIL planning requires eligible fitting slides.")
    identities = [row["slideId"] for row in training]
    if len(set(identities)) != len(identities):
        raise ValueError("A fitting slide occurs more than once in the frozen membership.")
    counts, dimensions, evidence = [], set(), []
    for row in training:
        if row.get("phase") == "final" or row.get("pool") == "external_test":
            raise ValueError("External or final-test slides cannot determine training bags.")
        entry = feature_files.get(row["slideId"])
        if not isinstance(entry, dict):
            raise ValueError(f"Missing fitting features for {row['slideId']}.")
        count, dimension = entry.get("patchCount"), entry.get("dimensions")
        if type(count) is not int or count < 1:
            raise ValueError(f"Fitting slide {row['slideId']} needs a positive patch count.")
        if type(dimension) is not int or dimension < 1:
            raise ValueError(f"Fitting slide {row['slideId']} needs valid feature dimensions.")
        counts.append(count)
        dimensions.add(dimension)
        evidence.append({"slideId": row["slideId"], "patientId": row.get("patientId"),
                         "label": row.get("label"), "features": entry})
    if len(dimensions) != 1:
        raise ValueError("Fitting features must have one compatible feature dimension.")
    counts.sort()
    center = median(counts)
    fraction = recipe.get("bagSizeFraction", 0.5)
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not (
        math.isfinite(fraction) and 0 < fraction <= 1
    ):
        raise ValueError("Automatic bag fraction must be greater than zero and at most one.")
    defaults = TrainingRecipe.model_fields
    cap = (
        max(1, math.floor(center * fraction)) if automatic
        else recipe.get("bagSize", defaults["bagSize"].default)
    )
    if cap is not None and (type(cap) is not int or not 1 <= cap <= 1000000):
        raise ValueError("Resolved training bags exceed the supported 1–1,000,000 patch range.")
    if automatic and recipe.get("bagCurriculum", False):
        raise ValueError("Choose automatic bags or a bag curriculum.")
    dimension = next(iter(dimensions))
    width = min(recipe.get("attentionDim", defaults["attentionDim"].default), dimension)
    divisor = recipe.get("nnmilWindowStrideDivisor", 4)
    if type(width) is not int or width < 1 or type(divisor) is not int or divisor < 1:
        raise ValueError("nnMIL requires positive attention width and window stride divisor.")
    stride = max(1, width // divisor)
    windows = (
        math.ceil((dimension - width) / stride) + 1
        if is_nnmil and recipe.get("nnmilFeatureSampling", True)
        else 1
    )
    summary = {
        "version": 1,
        "mode": "training_median" if automatic else "fixed",
        "fraction": fraction if automatic else None,
        "rounding": "floor", "minimumPatches": 1,
        "trainingSlideCount": len(training),
        "trainingPatientCount": len({row["patientId"] for row in training}),
        "medianPatchCount": center, "minPatchCount": counts[0], "maxPatchCount": counts[-1],
        "patchCountP05": _quantile(counts, 0.05), "patchCountP25": _quantile(counts, 0.25),
        "patchCountP75": _quantile(counts, 0.75), "patchCountP95": _quantile(counts, 0.95),
        "bagSize": cap, "featureDimension": dimension, "windowCount": windows,
        "paddedSlides": sum(count < cap for count in counts) if cap is not None else 0,
        "truncatedSlides": sum(count > cap for count in counts) if cap is not None else 0,
        "inputMemoryMiB": recipe.get("batchSize", defaults["batchSize"].default)
        * (cap or counts[-1]) * dimension * 4 / 1024**2,
        "fingerprint": _hash({"version": 1, "fittingSlides": evidence}),
    }
    return {**recipe, "bagSize": cap}, summary


def resolve_nnmil_plan(plan):
    """Resolve direct calls and verify already frozen per-run planning evidence."""
    if plan["recipe"].get("model", "abmil").lower() != "nnmil" and (
        plan["recipe"].get("bagSizeMode", "fixed") != "training_median"
    ):
        if "effectiveRecipe" in plan or "nnmilPlanning" in plan:
            raise ValueError("Unexpected automatic MIL planning evidence for this recipe.")
        return plan
    effective, summary = resolve_nnmil_recipe(
        plan["recipe"], plan["data"]["memberships"], plan["data"]["featureFiles"]
    )
    if summary is None:
        if "effectiveRecipe" in plan or "nnmilPlanning" in plan:
            raise ValueError("Unexpected automatic MIL planning evidence for this recipe.")
        return plan
    for key, expected in (("effectiveRecipe", effective), ("nnmilPlanning", summary)):
        if key in plan and plan[key] != expected:
            raise ValueError(f"Frozen {key} differs from the fitting data and recipe.")
    return {**plan, "effectiveRecipe": effective, "nnmilPlanning": summary}
