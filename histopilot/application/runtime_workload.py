"""Memory-relevant MIL workload metadata, without reading feature tensors."""

import hashlib
import json

from histopilot.schemas.development import TrainingRecipe
from histopilot.schemas.nnmil import resolve_nnmil_recipe

# Learning rate, stopping rules and reporting metrics do not determine tensor sizes.
# Patch caps are compared separately so nearby folds can share an observation.
MEMORY_FIELDS = (
    "model",
    "optimizer",
    "batchSize",
    "evalBatchSize",
    "precision",
    "embedDim",
    "attentionDim",
    "numFcLayers",
    "gatedAttention",
    "dropout",
    "inputDropout",
    "gradientCheckpointing",
    "accumulateGradBatches",
    "weightDecayPolicy",
    "lossType",
    "classWeighting",
    "classWeights",
    "nnmilFeatureSampling",
    "nnmilWindowStrideDivisor",
    "nnmilWindowAggregation",
    "nnmilBatchSampler",
    "instanceDropout",
    "featureNoiseStd",
    "bagCurriculum",
)


def _recipe(recipe):
    parsed = TrainingRecipe.model_validate(recipe, context={"legacy": True})
    # Frozen legacy serialization intentionally omits newer defaults. Runtime
    # identity needs their effective values, including omitted evaluation fields.
    values = {key: getattr(parsed, key) for key in TrainingRecipe.model_fields}
    values["model"] = values["model"].lower()
    values["evalBatchSize"] = values["evalBatchSize"] or values["batchSize"]
    return values


def workload_key(
    recipe, protocol_hash, feature_bundle_hash, feature_dim, loading_policy, class_count
):
    values = _recipe(recipe)
    identity = {
        "version": 1,
        "recipe": {key: values[key] for key in MEMORY_FIELDS},
        "protocol": protocol_hash,
        "bundle": feature_bundle_hash,
        "featureDimension": feature_dim,
        "loadingPolicy": loading_policy,
        "classCount": class_count,
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def workload_for_recipe(
    recipe, rows, files, protocol_hash, feature_bundle_hash, loading_policy, class_count
):
    """Bound padded train/evaluation inputs for one fold from frozen headers.

    Assessment bag sizes are read only to budget memory; they cannot set the
    fitting-only automatic patch cap or any scientific training parameter.
    """
    values = _recipe(recipe)
    rows = [
        row
        for row in rows
        if row.get("partition") in {"train", "val", "test"}
        and row.get("phase") != "final"
        and row.get("pool") != "external_test"
    ]
    counts = {role: [] for role in ("train", "val", "test")}
    dimensions = set()
    seen = set()
    for row in rows:
        identity = row["slideId"]
        if identity in seen:
            raise ValueError("A slide occurs more than once in a runtime planning fold.")
        seen.add(identity)
        entry = files.get(identity)
        if not isinstance(entry, dict):
            raise ValueError("Runtime planning requires feature headers for every eligible slide.")
        count, dimension = entry.get("patchCount"), entry.get("dimensions")
        if type(count) is not int or count < 1 or type(dimension) is not int or dimension < 1:
            raise ValueError(
                "Runtime planning requires positive patch counts and feature dimensions."
            )
        counts[row["partition"]].append(count)
        dimensions.add(dimension)
    if not counts["train"] or not (counts["val"] or counts["test"]) or len(dimensions) != 1:
        raise ValueError(
            "Runtime planning needs fitting/evaluation slides with compatible features."
        )
    feature_dim = next(iter(dimensions))
    effective, _ = resolve_nnmil_recipe(values, rows, files)
    cap = effective["bagSize"]
    if effective["bagCurriculum"]:
        cap = effective["bagCurriculumEnd"]
    training_patches = min(max(counts["train"]), cap) if cap else max(counts["train"])
    evaluation_patches = max(counts["val"] + counts["test"])
    if effective["evalBagSize"]:
        evaluation_patches = min(evaluation_patches, effective["evalBagSize"])
    return {
        "key": workload_key(
            values, protocol_hash, feature_bundle_hash, feature_dim, loading_policy, class_count
        ),
        **{
            key: values[key]
            for key in (
                "model",
                "batchSize",
                "evalBatchSize",
                "precision",
                "embedDim",
                "attentionDim",
                "numFcLayers",
                "gradientCheckpointing",
            )
        },
        "featureDimension": feature_dim,
        "trainingPatches": training_patches,
        "evaluationPatches": evaluation_patches,
        "sourcePatches": max(counts["train"] + counts["val"] + counts["test"]),
        "loadingPolicy": loading_policy,
        "classCount": class_count,
        "protocolHash": protocol_hash,
        "featureBundleHash": feature_bundle_hash,
    }
