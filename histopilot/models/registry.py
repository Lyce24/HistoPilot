"""Build the architecture a recipe names. Torch lives here, never in the catalog.

``catalog`` answers "what can this architecture do" for the service and the
schemas without importing Torch. This module answers "how is it constructed" and
is imported only by training workers. A new architecture needs one catalog entry
and one builder here; no other layer should branch on a model name.
"""

from histopilot.models.catalog import OPTIONS, normalize


def _option(recipe, key):
    """A catalogued architecture option, defaulting to the catalog's value."""
    return recipe.get(key, OPTIONS[key].default)


def _projection(recipe, layers=True):
    """The shared input-layer options of architectures without their own dropout there."""
    return {
        "embed_dim": recipe.get("embedDim", 512),
        **({"num_fc_layers": recipe.get("numFcLayers", 1)} if layers else {}),
        "input_dropout": recipe.get("inputDropout", 0.0),
        "gradient_checkpointing": recipe.get("gradientCheckpointing", False),
    }


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


def _clam(multi_branch):
    def build(feature_dim, output_dim, recipe):
        from histopilot.models.clam import CLAM

        return CLAM(
            feature_dim,
            output_dim,
            multi_branch=multi_branch,
            **_shared(recipe),
            recipe_attention_dropout=_option(recipe, "clamRecipeAttentionDropout"),
            attention_dim=_option(recipe, "clamAttentionDim"),
            gated_attention=_option(recipe, "clamGatedAttention"),
            instance_samples=_option(recipe, "clamInstanceSamples"),
            bag_weight=_option(recipe, "clamBagWeight"),
            instance_loss=_option(recipe, "clamInstanceLoss"),
            subtyping=_option(recipe, "clamSubtyping"),
        )

    return build


def _transmil(feature_dim, output_dim, recipe):
    from histopilot.models.transmil import TransMIL

    return TransMIL(
        feature_dim,
        output_dim,
        **_projection(recipe),
        layers=_option(recipe, "transmilLayers"),
        heads=_option(recipe, "transmilHeads"),
        landmarks=_option(recipe, "transmilLandmarks"),
        pinv_iterations=_option(recipe, "transmilPinvIterations"),
        dropout=_option(recipe, "transmilDropout"),
        position_encoding=_option(recipe, "transmilPositionEncoding"),
        residual_conv=_option(recipe, "transmilResidualConv"),
    )


def _dsmil(feature_dim, output_dim, recipe):
    from histopilot.models.dsmil import DSMIL

    return DSMIL(
        feature_dim,
        output_dim,
        **_projection(recipe),
        projection=_option(recipe, "dsmilProjection"),
        query_dim=_option(recipe, "dsmilQueryDim"),
        nonlinear_query=_option(recipe, "dsmilNonlinearQuery"),
        passing_value=_option(recipe, "dsmilPassingValue"),
        value_dropout=_option(recipe, "dsmilValueDropout"),
        max_loss_weight=_option(recipe, "dsmilMaxLossWeight"),
        average_prediction=_option(recipe, "dsmilAveragePrediction"),
    )


def _dtfd(feature_dim, output_dim, recipe):
    from histopilot.models.dtfd import DTFDMIL

    return DTFDMIL(
        feature_dim,
        output_dim,
        **_projection(recipe, layers=False),
        residual_blocks=_option(recipe, "dtfdResidualBlocks"),
        pseudo_bags=_option(recipe, "dtfdPseudoBags"),
        test_splits=_option(recipe, "dtfdTestSplits"),
        distill=_option(recipe, "dtfdDistill"),
        distill_patches=_option(recipe, "dtfdDistillPatches"),
        attention_dim=_option(recipe, "dtfdAttentionDim"),
        dropout=_option(recipe, "dtfdDropout"),
    )


def _acmil(feature_dim, output_dim, recipe):
    from histopilot.models.acmil import ACMIL

    return ACMIL(
        feature_dim,
        output_dim,
        **_projection(recipe, layers=False),
        branches=_option(recipe, "acmilBranches"),
        masked_patches=_option(recipe, "acmilMaskedPatches"),
        mask_drop=_option(recipe, "acmilMaskDrop"),
        attention_dim=_option(recipe, "acmilAttentionDim"),
        dropout=_option(recipe, "acmilDropout"),
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
    "clam_sb": _clam(multi_branch=False),
    "clam_mb": _clam(multi_branch=True),
    "transmil": _transmil,
    "dsmil": _dsmil,
    "dtfd": _dtfd,
    "acmil": _acmil,
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
