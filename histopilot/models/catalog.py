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

An architecture's own settings are ``options``: recipe fields named with its
option prefix, each with its type, default, range and help. The recipe schema
declares the same fields (``tests/test_model_options.py`` keeps the two equal),
the browser renders them from the published description, and the Task Center
keys memory evidence on the ones marked ``memory``.
"""

from dataclasses import dataclass

PATCH = "patch"
SLIDE = "slide"

INTEGER = "integer"
NUMBER = "number"
BOOLEAN = "boolean"
CHOICE = "choice"

# Shared architecture fields; each model lists the ones it reads.
SHARED_FIELDS = (
    "embedDim",
    "attentionDim",
    "numFcLayers",
    "gatedAttention",
    "dropout",
    "inputDropout",
)


@dataclass(frozen=True, slots=True)
class OptionSpec:
    """One architecture-owned recipe setting."""

    key: str
    label: str
    kind: str
    default: bool | int | float | str
    help: str
    minimum: int | float | None = None
    maximum: int | float | None = None
    maximum_exclusive: bool = False
    choices: tuple[tuple[str, str], ...] = ()
    # Changes tensor shapes, so measured memory of one value says little about another.
    memory: bool = False
    # Read only while another option of the same model has one of these values.
    enabled_when: tuple[str, tuple] | None = None

    def describe(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "kind": self.kind,
            "default": self.default,
            "help": self.help,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "maximumExclusive": self.maximum_exclusive,
            "choices": [{"value": value, "label": label} for value, label in self.choices],
            "memory": self.memory,
            "enabledWhen": (
                {"key": self.enabled_when[0], "values": list(self.enabled_when[1])}
                if self.enabled_when
                else None
            ),
        }


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
    # Shared architecture fields (SHARED_FIELDS) this architecture reads.
    reads: tuple[str, ...] = ()
    # Shared fields it reads only while one of its options has one of these values,
    # as (field, option key, values).
    reads_enabled_when: tuple[tuple[str, str, tuple], ...] = ()
    # Settings it owns, all named with ``option_prefix``.
    options: tuple[OptionSpec, ...] = ()
    # It adds its own loss terms while fitting, so it needs the slide labels.
    auxiliary_loss: bool = False
    # Fitting needs at least one class logit per target class (no one-logit BCE).
    requires_class_logits: bool = False
    # How one member's per-patch map is formed, when it is not one softmax over patches.
    attention_note: str | None = None
    reference: str | None = None


_PROJECTION = ("embedDim", "numFcLayers", "inputDropout")

_CLAM_OPTIONS = (
    OptionSpec(
        "clamAttentionDim",
        "Attention dimensions",
        INTEGER,
        256,
        "Hidden width of the attention network. CLAM's small model uses 256, its big model 384.",
        minimum=1,
        maximum=8192,
        memory=True,
    ),
    OptionSpec(
        "clamGatedAttention",
        "Gated attention",
        BOOLEAN,
        True,
        "Multiply the attention scores by a sigmoid gate, as CLAM does by default.",
    ),
    OptionSpec(
        "clamRecipeAttentionDropout",
        "Attention dropout follows Dropout",
        BOOLEAN,
        False,
        "Off, as published: the attention branches drop 25% whenever Dropout is above 0, "
        "whatever its value. On, they use the Dropout setting.",
    ),
    OptionSpec(
        "clamInstanceSamples",
        "Patches per instance class (B)",
        INTEGER,
        8,
        "The B most and B least attended patches of each slide train the instance "
        "classifiers. A slide with fewer than 2B patches uses half of them.",
        minimum=1,
        maximum=4096,
    ),
    OptionSpec(
        "clamBagWeight",
        "Slide loss weight",
        NUMBER,
        0.7,
        "Weight of the slide-level loss; the instance-clustering loss gets the rest. "
        "1 turns instance clustering off.",
        minimum=0,
        maximum=1,
    ),
    OptionSpec(
        "clamInstanceLoss",
        "Instance loss",
        CHOICE,
        "svm",
        "Loss of the instance classifiers. CLAM's published runs use the smooth top-1 SVM "
        "(margin 1, temperature 1).",
        choices=(("svm", "Smooth top-1 SVM"), ("ce", "Cross entropy")),
    ),
    OptionSpec(
        "clamSubtyping",
        "Mutually exclusive subtypes",
        BOOLEAN,
        False,
        "Also teach every other class's instance classifier that this slide's most attended "
        "patches are negative. CLAM uses this for subtyping tasks.",
    ),
)

CATALOG: dict[str, ModelSpec] = {
    spec.name: spec
    for spec in (
        ModelSpec(
            name="abmil",
            label="ABMIL",
            summary="Gated attention over patch embeddings.",
            supports_attention=True,
            reads=(
                "embedDim",
                "attentionDim",
                "numFcLayers",
                "gatedAttention",
                "dropout",
                "inputDropout",
            ),
            reference="Ilse, Tomczak and Welling, ICML 2018",
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
            reads=("attentionDim", "dropout", "inputDropout"),
            attention_note="Each member averages normalized attention across its feature windows.",
            reference="Luo et al., arXiv:2511.14907",
        ),
        ModelSpec(
            name="clam_sb",
            label="CLAM-SB",
            summary="Gated attention with instance-level clustering of the most and least "
            "attended patches.",
            supports_attention=True,
            option_prefix="clam",
            reads=("embedDim", "numFcLayers", "dropout", "inputDropout"),
            options=_CLAM_OPTIONS,
            auxiliary_loss=True,
            reference="Lu et al., Nature Biomedical Engineering 2021",
        ),
        ModelSpec(
            name="clam_mb",
            label="CLAM-MB",
            summary="One attention branch and slide classifier per class, with instance-level "
            "clustering.",
            supports_attention=True,
            option_prefix="clam",
            reads=("embedDim", "numFcLayers", "dropout", "inputDropout"),
            options=_CLAM_OPTIONS,
            auxiliary_loss=True,
            requires_class_logits=True,
            attention_note="Each member's map is the attention branch of its predicted class, "
            "as CLAM's heatmaps use.",
            reference="Lu et al., Nature Biomedical Engineering 2021",
        ),
        ModelSpec(
            name="transmil",
            label="TransMIL",
            summary="Nyström Transformer over patch tokens with a class token and pyramid "
            "position encoding.",
            supports_attention=True,
            option_prefix="transmil",
            reads=_PROJECTION,
            options=(
                OptionSpec(
                    "transmilLayers",
                    "Transformer layers",
                    INTEGER,
                    2,
                    "Nyström self-attention layers. The position encoding follows the first "
                    "layer, as in TransMIL's two-layer model; a one-layer model encodes "
                    "positions before its layer.",
                    minimum=1,
                    maximum=12,
                    memory=True,
                ),
                OptionSpec(
                    "transmilHeads",
                    "Attention heads",
                    INTEGER,
                    8,
                    "Heads per layer. The embedding dimensions must divide evenly by them.",
                    minimum=1,
                    maximum=64,
                    memory=True,
                ),
                OptionSpec(
                    "transmilLandmarks",
                    "Nyström landmarks",
                    INTEGER,
                    256,
                    "Landmark tokens that approximate full self-attention; TransMIL uses half "
                    "the embedding dimensions.",
                    minimum=1,
                    maximum=4096,
                    memory=True,
                ),
                OptionSpec(
                    "transmilPinvIterations",
                    "Pseudo-inverse iterations",
                    INTEGER,
                    6,
                    "Moore-Penrose iterations of the Nyström approximation.",
                    minimum=1,
                    maximum=32,
                ),
                OptionSpec(
                    "transmilDropout",
                    "Attention dropout",
                    NUMBER,
                    0.1,
                    "Dropout on each attention layer's output.",
                    minimum=0,
                    maximum=1,
                    maximum_exclusive=True,
                ),
                OptionSpec(
                    "transmilPositionEncoding",
                    "Pyramid position encoding (PPEG)",
                    BOOLEAN,
                    True,
                    "Depthwise 3×3, 5×5 and 7×7 convolutions over the patch tokens laid out on "
                    "a square grid in bag order.",
                ),
                OptionSpec(
                    "transmilResidualConv",
                    "Value residual convolution",
                    BOOLEAN,
                    True,
                    "Add a depthwise convolution over the attention values along the sequence "
                    "(kernel 33), as TransMIL's Nyström layers do.",
                ),
            ),
            attention_note="Each member's map is the class token's softmax attention to the "
            "patches in the last layer, computed exactly and averaged over heads; the copies "
            "that square the token grid count toward their source patches.",
            reference="Shao et al., NeurIPS 2021",
        ),
        ModelSpec(
            name="dsmil",
            label="DSMIL",
            summary="A max-instance classifier plus attention to each class's critical patch.",
            supports_attention=True,
            option_prefix="dsmil",
            reads=_PROJECTION,
            reads_enabled_when=(
                ("embedDim", "dsmilProjection", (True,)),
                ("numFcLayers", "dsmilProjection", (True,)),
            ),
            options=(
                OptionSpec(
                    "dsmilProjection",
                    "Shared input layer",
                    BOOLEAN,
                    False,
                    "Project patches with the input layer under Model architecture before the "
                    "instance classifier, as standardized benchmark libraries do. Off, as "
                    "published, DSMIL classifies the patch features directly.",
                    memory=True,
                ),
                OptionSpec(
                    "dsmilQueryDim",
                    "Query dimensions",
                    INTEGER,
                    128,
                    "Width of the query network that compares patches with the critical patch.",
                    minimum=1,
                    maximum=8192,
                    memory=True,
                ),
                OptionSpec(
                    "dsmilNonlinearQuery",
                    "Nonlinear query network",
                    BOOLEAN,
                    True,
                    "Two layers with ReLU and tanh; off, a single linear layer.",
                ),
                OptionSpec(
                    "dsmilPassingValue",
                    "Value projection",
                    BOOLEAN,
                    False,
                    "Project patch embeddings before pooling them; off, they are pooled as "
                    "they are.",
                ),
                OptionSpec(
                    "dsmilValueDropout",
                    "Value dropout",
                    NUMBER,
                    0.0,
                    "Dropout before the value projection.",
                    minimum=0,
                    maximum=1,
                    maximum_exclusive=True,
                    enabled_when=("dsmilPassingValue", (True,)),
                ),
                OptionSpec(
                    "dsmilMaxLossWeight",
                    "Max-instance loss weight",
                    NUMBER,
                    0.5,
                    "Weight of the max-instance classifier's loss; the slide loss gets the "
                    "rest. DSMIL weights both equally.",
                    minimum=0,
                    maximum=1,
                ),
                OptionSpec(
                    "dsmilAveragePrediction",
                    "Average both classifiers",
                    BOOLEAN,
                    False,
                    "Predict with the mean of the slide and max-instance classifiers' "
                    "probabilities; off, the slide classifier alone.",
                ),
            ),
            auxiliary_loss=True,
            attention_note="Each member's map is the attention to its predicted class's "
            "critical patch.",
            reference="Li, Li and Eliceiri, CVPR 2021",
        ),
        ModelSpec(
            name="dtfd",
            label="DTFD-MIL",
            summary="Attention over random pseudo-bags, distilled into a second attention tier.",
            supports_attention=True,
            option_prefix="dtfd",
            reads=("embedDim", "inputDropout"),
            options=(
                OptionSpec(
                    "dtfdPseudoBags",
                    "Pseudo-bags per slide",
                    INTEGER,
                    4,
                    "Random groups each slide's patches are split into: 4 in the authors' code, "
                    "5 and 8 in the paper's CAMELYON16 and TCGA lung runs. Testing uses one "
                    "fixed split per slide size, so predictions are reproducible.",
                    minimum=1,
                    maximum=256,
                ),
                OptionSpec(
                    "dtfdTestSplits",
                    "Test-time pseudo-bag splits",
                    INTEGER,
                    1,
                    "Random splits whose predicted probabilities are averaged when testing, each "
                    "from a fixed seed, so predictions repeat. The authors' code uses 1; with "
                    "MaxMinS one split moved a slide's probability by up to 0.33, so 8 gives "
                    "steadier clinical predictions.",
                    minimum=1,
                    maximum=64,
                ),
                OptionSpec(
                    "dtfdDistill",
                    "Feature distillation",
                    CHOICE,
                    "afs",
                    "What each pseudo-bag passes to the second tier.",
                    choices=(
                        ("afs", "Attention-pooled feature (AFS)"),
                        ("maxmins", "Highest- and lowest-scoring patches (MaxMinS)"),
                        ("maxs", "Highest-scoring patches (MaxS)"),
                    ),
                ),
                OptionSpec(
                    "dtfdDistillPatches",
                    "Patches distilled per pseudo-bag",
                    INTEGER,
                    1,
                    "For MaxMinS and MaxS: patches taken from each end of the pseudo-bag's "
                    "ranking.",
                    minimum=1,
                    maximum=1024,
                    enabled_when=("dtfdDistill", ("maxmins", "maxs")),
                ),
                OptionSpec(
                    "dtfdResidualBlocks",
                    "Residual blocks",
                    INTEGER,
                    0,
                    "Residual blocks after the input layer (the authors' numLayer_Res, 0 by "
                    "default).",
                    minimum=0,
                    maximum=8,
                    memory=True,
                ),
                OptionSpec(
                    "dtfdAttentionDim",
                    "Attention dimensions",
                    INTEGER,
                    128,
                    "Hidden width of both tiers' gated attention.",
                    minimum=1,
                    maximum=8192,
                    memory=True,
                ),
                OptionSpec(
                    "dtfdDropout",
                    "Classifier dropout",
                    NUMBER,
                    0.0,
                    "Dropout before both tiers' classifiers.",
                    minimum=0,
                    maximum=1,
                    maximum_exclusive=True,
                ),
            ),
            auxiliary_loss=True,
            attention_note="Each member's map is the weight of each patch in the slide "
            "embedding: its attention within its pseudo-bag times the second tier's attention "
            "to that pseudo-bag. With MaxMinS or MaxS distillation only the distilled patches "
            "carry weight.",
            reference="Zhang et al., CVPR 2022",
        ),
        ModelSpec(
            name="acmil",
            label="ACMIL",
            summary="Several attention branches with masking of top patches and a diversity "
            "penalty.",
            supports_attention=True,
            option_prefix="acmil",
            reads=("embedDim", "inputDropout"),
            options=(
                OptionSpec(
                    "acmilBranches",
                    "Attention branches",
                    INTEGER,
                    5,
                    "Branches with their own attention and classifier. One branch with no "
                    "masking is gated ABMIL.",
                    minimum=1,
                    maximum=64,
                    memory=True,
                ),
                OptionSpec(
                    "acmilMaskedPatches",
                    "Top patches eligible for masking",
                    INTEGER,
                    10,
                    "While fitting, part of each branch's most attended patches is hidden "
                    "(stochastic top-k instance masking). 0 turns masking off.",
                    minimum=0,
                    maximum=100000,
                ),
                OptionSpec(
                    "acmilMaskDrop",
                    "Masked fraction",
                    NUMBER,
                    0.6,
                    "Fraction of those top patches hidden at random each step. ACMIL's authors "
                    "report that 0 is sometimes more stable.",
                    minimum=0,
                    maximum=1,
                    maximum_exclusive=True,
                ),
                OptionSpec(
                    "acmilAttentionDim",
                    "Attention dimensions",
                    INTEGER,
                    128,
                    "Hidden width of the gated attention.",
                    minimum=1,
                    maximum=8192,
                    memory=True,
                ),
                OptionSpec(
                    "acmilDropout",
                    "Classifier dropout",
                    NUMBER,
                    0.0,
                    "Dropout before the branch and slide classifiers.",
                    minimum=0,
                    maximum=1,
                    maximum_exclusive=True,
                ),
            ),
            auxiliary_loss=True,
            attention_note="Each member's map is the mean of its branches' attention, which "
            "is what its slide classifier pools.",
            reference="Zhang et al., ECCV 2024",
        ),
        ModelSpec(
            name="mean_pool",
            label="Mean pooling MIL",
            summary="Mean of patch embeddings with a classifier head.",
            reads=("embedDim", "numFcLayers", "dropout", "inputDropout"),
        ),
        ModelSpec(
            name="max_pool",
            label="Max pooling MIL",
            summary="Element-wise maximum of patch embeddings with a classifier head.",
            reads=("embedDim", "numFcLayers", "dropout", "inputDropout"),
        ),
        ModelSpec(
            name="slide_linear",
            label="Slide-embedding linear probe",
            summary="Linear read-out of one slide encoder embedding per slide.",
            feature_kind=SLIDE,
            reads=("inputDropout",),
        ),
        ModelSpec(
            name="slide_mlp",
            label="Slide-embedding MLP probe",
            summary="One hidden layer over a slide encoder embedding.",
            feature_kind=SLIDE,
            reads=("embedDim", "dropout", "inputDropout"),
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
# Every catalogued option once; two architectures may share one set (CLAM-SB and -MB).
OPTIONS: dict[str, OptionSpec] = {
    option.key: option for spec in CATALOG.values() for option in spec.options
}

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


def auxiliary_loss(model: str | None) -> bool:
    """True when fitting adds the architecture's own loss terms to the slide loss."""
    found = spec(model)
    return bool(found and found.auxiliary_loss)


def options(model: str | None) -> tuple[OptionSpec, ...]:
    found = spec(model)
    return found.options if found else ()


def option_defaults(model: str | None = None) -> dict:
    """Default values of one architecture's options, or of every catalogued option."""
    chosen = OPTIONS.values() if model is None else options(model)
    return {option.key: option.default for option in chosen}


def memory_options(model: str | None) -> tuple[str, ...]:
    """The architecture's options that change tensor shapes."""
    return tuple(option.key for option in options(model) if option.memory)


def reads_enabled(model: str | None, field: str, recipe: dict) -> bool:
    """Whether the model reads a shared field under this recipe's own options."""
    found = spec(model)
    if found is None or field not in found.reads:
        return False
    for name, key, values in found.reads_enabled_when:
        if name == field and recipe.get(key, OPTIONS[key].default) not in values:
            return False
    return True


def foreign_options(model: str | None) -> tuple[str, ...]:
    """Catalogued options that belong to other architectures than ``model``."""
    own = {option.key for option in options(model)}
    return tuple(key for key in OPTIONS if key not in own)


def choices(feature_kind: str | None = None) -> str:
    """Name the supported architectures so no message can list a stale set."""
    labels = [
        item.label
        for item in CATALOG.values()
        if feature_kind is None or item.feature_kind == feature_kind
    ]
    return ", ".join(labels)


def attention_choices() -> str:
    """Name the architectures whose predictors have attention maps."""
    return ", ".join(item.label for item in CATALOG.values() if item.supports_attention)


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
            "reads": list(item.reads),
            "readsEnabledWhen": {
                field: {"key": key, "values": list(values)}
                for field, key, values in item.reads_enabled_when
            },
            "options": [option.describe() for option in item.options],
            "auxiliaryLoss": item.auxiliary_loss,
            "requiresClassLogits": item.requires_class_logits,
            "attentionNote": item.attention_note,
            "reference": item.reference,
        }
        for item in CATALOG.values()
    ]
