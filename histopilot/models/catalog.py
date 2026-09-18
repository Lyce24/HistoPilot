"""The supported model architectures and what each one can actually do.

This module is pure data. It must import without Torch, CUDA, Pydantic or any
optional backend so the FastAPI service, the schema validators and the training
workers all read one description instead of repeating literal model names.

Before this catalog existed the same facts were written out separately in
``application/training.py``, ``application/predictors.py``,
``application/development.py``, ``application/runtime_advisor.py``,
``training/module.py``, ``training/fold.py``, ``training/attention.py`` and the
browser. Adding an architecture meant finding every one of them. Add an entry
here instead; construction still lives with the Torch modules.

``featureKind`` records which extraction output an architecture consumes: patch
bags with coordinates, or one embedding per slide from a slide encoder.
"""

from dataclasses import dataclass

PATCH = "patch"
SLIDE = "slide"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One architecture's identity and the capabilities other layers branch on."""

    name: str
    label: str
    summary: str
    feature_kind: str = PATCH
    # Attention extraction reads per-instance weights; pooling has none to read.
    supports_attention: bool = False
    # Structured models return a dict from ``prediction_output`` rather than a
    # logits tensor from ``__call__``; inference must ask for the right one.
    structured_output: bool = False
    # Only architectures with their own checkpoint policy honour a recipe's
    # explicit selection; everything else uses the best validation checkpoint.
    honors_checkpoint_selection: bool = False
    # Some architectures draw several feature windows per slide per epoch, so
    # each presentation needs its own draw identity and augmentation stream.
    windowed_sampling: bool = False
    # Recipe fields an architecture owns, so validators can reject them for the
    # architectures that ignore them.
    option_prefix: str | None = None


CATALOG: dict[str, ModelSpec] = {
    spec.name: spec
    for spec in (
        ModelSpec(
            name="abmil",
            label="ABMIL",
            summary="Gated attention over patch embeddings.",
            supports_attention=True,
        ),
        ModelSpec(
            name="nnmil",
            label="nnMIL",
            summary="Full-dimensional pooling with windowed feature sampling.",
            supports_attention=True,
            structured_output=True,
            honors_checkpoint_selection=True,
            windowed_sampling=True,
            option_prefix="nnmil",
        ),
        ModelSpec(
            name="mean_pool",
            label="Mean pooling MIL",
            summary="Mean of patch embeddings with a classifier head.",
        ),
        ModelSpec(
            name="max_pool",
            label="Max pooling MIL",
            summary="Element-wise maximum of patch embeddings with a classifier head.",
        ),
        ModelSpec(
            name="slide_linear",
            label="Slide-embedding linear probe",
            summary="Linear read-out of one slide encoder embedding per slide.",
            feature_kind=SLIDE,
        ),
        ModelSpec(
            name="slide_mlp",
            label="Slide-embedding MLP probe",
            summary="One hidden layer over a slide encoder embedding.",
            feature_kind=SLIDE,
        ),
    )
}

NAMES: frozenset[str] = frozenset(CATALOG)
PATCH_MODELS: frozenset[str] = frozenset(
    name for name, spec in CATALOG.items() if spec.feature_kind == PATCH
)
SLIDE_MODELS: frozenset[str] = frozenset(
    name for name, spec in CATALOG.items() if spec.feature_kind == SLIDE
)
OPTION_PREFIXES: frozenset[str] = frozenset(
    spec.option_prefix for spec in CATALOG.values() if spec.option_prefix
)

DEFAULT_MODEL = "abmil"


def normalize(model: str | None) -> str:
    """Recipes have always compared lowercased names; keep that exact meaning."""
    return (model or DEFAULT_MODEL).lower()


def spec(model: str | None) -> ModelSpec | None:
    """Return the architecture description, or None for an unsupported name."""
    return CATALOG.get(normalize(model))


def is_supported(model: str | None) -> bool:
    return normalize(model) in CATALOG


def feature_kind(model: str | None) -> str:
    """Which extraction output this architecture consumes; patch when unknown."""
    found = spec(model)
    return found.feature_kind if found else PATCH


def supports_attention(model: str | None, input_mode: str | None = "image") -> bool:
    """Clinical-only predictors have no image attention regardless of architecture."""
    found = spec(model)
    return bool(found and found.supports_attention and input_mode != "clinical")


def structured_output(model: str | None) -> bool:
    """True when the architecture itself returns a dict rather than a logits tensor."""
    found = spec(model)
    return bool(found and found.structured_output)


def uses_structured_output(model: str | None, input_mode: str | None = "image") -> bool:
    """Combined and clinical arms also return structured output from any model."""
    return structured_output(model) or (input_mode or "image") != "image"


def windowed_sampling(model: str | None) -> bool:
    """True when one slide yields several feature windows per training epoch."""
    found = spec(model)
    return bool(found and found.windowed_sampling)


def honors_checkpoint_selection(model: str | None) -> bool:
    found = spec(model)
    return bool(found and found.honors_checkpoint_selection)


def owns_options(model: str | None, prefix: str) -> bool:
    """True when the named architecture is the one that reads ``prefix`` fields."""
    found = spec(model)
    return bool(found and found.option_prefix == prefix)


def choices(feature_kind: str | None = None) -> str:
    """Name the supported architectures so no message can list a stale set."""
    labels = [
        item.label
        for item in CATALOG.values()
        if feature_kind is None or item.feature_kind == feature_kind
    ]
    return ", ".join(labels)


def describe() -> list[dict]:
    """Serialize the catalog for the browser in the API's camelCase convention."""
    return [
        {
            "name": item.name,
            "label": item.label,
            "summary": item.summary,
            "featureKind": item.feature_kind,
            "supportsAttention": item.supports_attention,
            "structuredOutput": item.structured_output,
            "honorsCheckpointSelection": item.honors_checkpoint_selection,
            "windowedSampling": item.windowed_sampling,
            "optionPrefix": item.option_prefix,
        }
        for item in CATALOG.values()
    ]
