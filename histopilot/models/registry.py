"""Build the architecture a recipe names. Torch lives here, never in the catalog.

``catalog`` answers "what can this architecture do" for the service and the
schemas without importing Torch. This module answers "how is it constructed" and
is imported only by training workers. A new architecture needs one catalog entry
and one builder here; no other layer should branch on a model name.
"""

from histopilot.models.catalog import normalize


def _shared(recipe):
    """Options every patch-bag architecture accepts, with the recipe defaults."""
    return {
        "embed_dim": recipe.get("embedDim", 512),
        "num_fc_layers": recipe.get("numFcLayers", 1),
        "dropout": recipe.get("dropout", 0.25),
        "input_dropout": recipe.get("inputDropout", 0.0),
        "gradient_checkpointing": recipe.get("gradientCheckpointing", False),
    }


def _abmil(feature_dim, output_dim, recipe):
    from histopilot.models.abmil import ABMIL

    return ABMIL(
        feature_dim,
        output_dim,
        **_shared(recipe),
        attention_dim=recipe.get("attentionDim", 384),
        gated_attention=recipe.get("gatedAttention", True),
    )


def _nnmil(feature_dim, output_dim, recipe):
    from histopilot.models.nnmil import NNMIL

    options = _shared(recipe)
    return NNMIL(
        feature_dim,
        output_dim,
        attention_dim=recipe.get("attentionDim", 256),
        dropout=options["dropout"],
        input_dropout=options["input_dropout"],
        gradient_checkpointing=options["gradient_checkpointing"],
        feature_sampling=recipe.get("nnmilFeatureSampling", True),
        window_stride_divisor=recipe.get("nnmilWindowStrideDivisor", 4),
        window_shuffle=recipe.get("nnmilWindowShuffle", True),
        window_seed=recipe.get("nnmilWindowSeed", 42),
        window_aggregation=recipe.get("nnmilWindowAggregation", "mean_logits"),
    )


def _pooling(pooling):
    def build(feature_dim, output_dim, recipe):
        from histopilot.models.pooling import PoolingMIL

        return PoolingMIL(feature_dim, output_dim, pooling=pooling, **_shared(recipe))

    return build


def _slide_probe(hidden):
    def build(feature_dim, output_dim, recipe):
        from histopilot.models.slide import SlideProbe

        return SlideProbe(
            feature_dim,
            output_dim,
            hidden_dim=recipe.get("embedDim", 512) if hidden else 0,
            dropout=recipe.get("dropout", 0.25),
            input_dropout=recipe.get("inputDropout", 0.0),
        )

    return build


BUILDERS = {
    "abmil": _abmil,
    "nnmil": _nnmil,
    "mean_pool": _pooling("mean"),
    "max_pool": _pooling("max"),
    "slide_linear": _slide_probe(hidden=False),
    "slide_mlp": _slide_probe(hidden=True),
}


def build(model, feature_dim, output_dim, recipe):
    """Construct the recipe's architecture, rejecting names with no builder."""
    name = normalize(model)
    builder = BUILDERS.get(name)
    if builder is None:
        raise ValueError(f"Unsupported MIL model: {name}")
    return builder(feature_dim, output_dim, recipe)
