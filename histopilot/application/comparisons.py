"""Controlled comparisons: arms of one batch that differ only in the factor being ablated.

A batch that declares a comparison trains every arm on the same frozen folds and training
seeds. Its arms may differ in the model, the options that model owns, and its inputs
(image, clinical or both, and the clinical fields). Any other recipe difference would mix
the ablated factor with optimisation choices, so it blocks the batch.
"""

from histopilot.models import catalog
from histopilot.schemas.development import EXPERIMENTAL_DEFAULTS

FACTOR_KEYS = frozenset({"model", "inputMode", "clinicalFields"})

# Architecture hyperparameters that only the named model reads. Keys under a model's
# catalog option prefix (for example nnmil*) are owned by that model as well.
MODEL_OPTIONS = {
    "abmil": frozenset({"attentionDim", "gatedAttention"}),
    "nnmil": frozenset({"attentionDim", "gatedAttention"}),
    "mean_pool": frozenset(),
    "max_pool": frozenset(),
    "slide_linear": frozenset(),
    "slide_mlp": frozenset(),
}

# Settings of the image path that a clinical-only arm never reads.
IMAGE_KEYS = frozenset(
    {
        "embedDim",
        "attentionDim",
        "numFcLayers",
        "gatedAttention",
        "dropout",
        "inputDropout",
        "gradientCheckpointing",
        "aggregatorLearningRate",
        "bagSize",
        "bagSizeMode",
        "bagSizeFraction",
        "bagCurriculum",
        "bagCurriculumStart",
        "bagCurriculumEnd",
        "bagCurriculumWarmupEpochs",
        "evalBagSize",
        "instanceDropout",
        "featureNoiseStd",
    }
)


def _owned(recipe):
    model = catalog.normalize(recipe.get("model"))
    keys = set(MODEL_OPTIONS.get(model, ()))
    spec = catalog.spec(model)
    if spec and spec.option_prefix:
        keys |= {key for key in recipe if key.startswith(spec.option_prefix)}
    return keys


def confounders(reference, arm):
    """Recipe settings, outside the compared factor, on which two arms disagree."""
    left, right = {**EXPERIMENTAL_DEFAULTS, **reference}, {**EXPERIMENTAL_DEFAULTS, **arm}
    ignored = set(FACTOR_KEYS) | _owned(left) | _owned(right)
    if "clinical" in {left.get("inputMode", "image"), right.get("inputMode", "image")}:
        ignored |= IMAGE_KEYS
        ignored |= {key for key in set(left) | set(right) if key.startswith("nnmil")}
    keys = (set(left) | set(right)) - ignored
    return sorted(key for key in keys if left.get(key) != right.get(key))


def comparison_findings(spec, recipes):
    """Blocking findings for a declared comparison over the batch's resolved recipes."""
    comparison = spec.comparison
    if comparison is None:
        return []
    if len(recipes) != len(spec.configurations):
        return [
            {
                "severity": "error",
                "code": "COMPARISON_DUPLICATE_ARMS",
                "message": "Two arms of this comparison are identical. Remove the duplicate.",
            }
        ]
    reference = recipes[comparison.reference - 1]
    findings = []
    for number, recipe in enumerate(recipes, start=1):
        differences = number != comparison.reference and confounders(reference, recipe)
        if differences:
            findings.append(
                {
                    "severity": "error",
                    "code": "COMPARISON_CONFOUNDED",
                    "message": f"Arm {number} differs from the reference arm in "
                    f"{', '.join(differences)}. A comparison may change only the model and "
                    "its inputs; align these settings so the difference measures one factor.",
                }
            )
    return findings
