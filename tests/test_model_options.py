"""Architecture options: one description in the catalog, the same fields everywhere else.

The browser renders each model's options from the published catalog, the recipe schema
validates them, the server templates give the browser their defaults, comparisons and
arms treat them as the model's own, and the Task Center keys memory on some of them.
"""

import hashlib
import json
from typing import get_args

import pytest
from pydantic import ValidationError

from histopilot import resolvers, templates
from histopilot.application.comparisons import confounders
from histopilot.models import catalog
from histopilot.schemas.development import EXPERIMENTAL_DEFAULTS, TrainingRecipe
from histopilot.taskcenter import estimator

NEW_MODELS = ("clam_sb", "clam_mb", "transmil", "dsmil", "dtfd", "acmil")
OPTIONS = sorted(catalog.OPTIONS.values(), key=lambda option: option.key)


def recipe(**values):
    return TrainingRecipe.model_validate(values).model_dump()


def test_every_new_architecture_is_catalogued_with_options_under_its_prefix():
    for name in NEW_MODELS:
        spec = catalog.spec(name)
        assert spec.options and spec.option_prefix and spec.supports_attention
        assert all(option.key.startswith(spec.option_prefix) for option in spec.options)
        assert set(spec.reads) <= set(catalog.SHARED_FIELDS)
    # CLAM's two variants share one set of options; no other key belongs to two models.
    owners = {}
    for spec in catalog.CATALOG.values():
        for option in spec.options:
            owners.setdefault(option.key, set()).add(spec.option_prefix)
    assert all(len(prefixes) == 1 for prefixes in owners.values())
    assert catalog.options("clam_sb") == catalog.options("clam_mb")


@pytest.mark.parametrize("option", OPTIONS, ids=lambda option: option.key)
def test_schema_field_matches_the_catalogued_option(option):
    field = TrainingRecipe.model_fields[option.key]
    assert field.default == option.default
    assert getattr(TrainingRecipe(), option.key) == option.default
    assert EXPERIMENTAL_DEFAULTS[option.key] == option.default
    assert templates.EXPERIMENTAL_DEFAULTS[option.key] == option.default

    def accepts(value):
        return getattr(TrainingRecipe.model_validate({option.key: value}), option.key) == value

    if option.kind == catalog.CHOICE:
        assert set(get_args(field.annotation)) == {value for value, _ in option.choices}
        assert all(accepts(value) for value, _ in option.choices)
        with pytest.raises(ValidationError):
            TrainingRecipe.model_validate({option.key: "unlisted"})
    elif option.kind == catalog.BOOLEAN:
        assert field.annotation is bool and accepts(not option.default)
    else:
        assert accepts(option.minimum)
        for outside in (option.minimum - 1, option.maximum + 1):
            with pytest.raises(ValidationError):
                TrainingRecipe.model_validate({option.key: outside})
        if option.maximum_exclusive:
            with pytest.raises(ValidationError):
                TrainingRecipe.model_validate({option.key: option.maximum})
        else:
            assert accepts(option.maximum)
        if option.kind == catalog.INTEGER:
            with pytest.raises(ValidationError):
                TrainingRecipe.model_validate({option.key: option.minimum + 0.5})


def test_default_options_leave_the_serialized_recipe_and_its_identity_unchanged():
    dumped = TrainingRecipe.model_validate({"model": "transmil"}).model_dump()
    assert not set(catalog.OPTIONS) & set(dumped)
    explicit = TrainingRecipe.model_validate({**dumped, **catalog.option_defaults()}).model_dump()
    assert explicit == dumped
    tuned = TrainingRecipe.model_validate({"model": "transmil", "transmilLayers": 4}).model_dump()
    assert tuned["transmilLayers"] == 4
    assert TrainingRecipe.model_validate(tuned).model_dump() == tuned


def test_the_published_description_carries_options_for_the_browser():
    described = {item["name"]: item for item in catalog.describe()}
    transmil = described["transmil"]
    assert transmil["reads"] == ["embedDim", "numFcLayers", "inputDropout"]
    layers = next(item for item in transmil["options"] if item["key"] == "transmilLayers")
    assert layers == {
        "key": "transmilLayers",
        "label": "Transformer layers",
        "kind": "integer",
        "default": 2,
        "help": layers["help"],
        "minimum": 1,
        "maximum": 12,
        "maximumExclusive": False,
        "choices": [],
        "memory": True,
        "enabledWhen": None,
    }
    loss = next(
        item for item in described["clam_sb"]["options"] if item["key"] == "clamInstanceLoss"
    )
    assert [choice["value"] for choice in loss["choices"]] == ["svm", "ce"]
    assert described["dsmil"]["auxiliaryLoss"] and not described["transmil"]["auxiliaryLoss"]
    assert described["clam_mb"]["requiresClassLogits"]
    assert described["abmil"]["options"] == [] and described["abmil"]["reads"]
    patches = next(
        item for item in described["dtfd"]["options"] if item["key"] == "dtfdDistillPatches"
    )
    assert patches["enabledWhen"] == {"key": "dtfdDistill", "values": ["maxmins", "maxs"]}
    for spec in catalog.CATALOG.values():
        keys = {option.key for option in spec.options}
        for option in spec.options:
            if option.enabled_when:
                assert option.enabled_when[0] in keys


def test_cross_field_rules_reject_choices_the_architecture_cannot_fit():
    with pytest.raises(ValidationError, match="CLAM-MB has one attention branch per class"):
        TrainingRecipe.model_validate({"model": "clam_mb", "lossType": "bce"})
    TrainingRecipe.model_validate({"model": "clam_sb", "lossType": "bce"})
    TrainingRecipe.model_validate({"model": "clam_mb", "lossType": "focal"})
    with pytest.raises(ValidationError, match="divide evenly by its attention heads"):
        TrainingRecipe.model_validate({"model": "transmil", "embedDim": 500})
    TrainingRecipe.model_validate({"model": "transmil", "embedDim": 500, "transmilHeads": 5})
    # Another architecture's options are inert, so they are not checked against this one.
    TrainingRecipe.model_validate({"model": "abmil", "embedDim": 500})


def test_comparison_arms_own_their_architecture_options():
    reference = recipe(model="abmil")
    assert confounders(reference, recipe(model="transmil", transmilLayers=4)) == []
    assert confounders(reference, recipe(model="clam_mb", clamBagWeight=0.5)) == []
    assert (
        confounders(recipe(model="dsmil", dsmilQueryDim=64), recipe(model="dtfd", dtfdPseudoBags=8))
        == []
    )
    # An option the arm's model does not own is a real difference between two arms.
    assert confounders(reference, recipe(model="mean_pool", transmilLayers=4)) == ["transmilLayers"]
    fields = [{"field": "age", "kind": "numeric"}]
    clinical = recipe(inputMode="clinical", clinicalFields=fields)
    assert confounders(recipe(model="acmil", acmilBranches=3), clinical) == []


def test_an_arm_returns_other_architectures_options_to_their_defaults():
    base = {**recipe(model="transmil", transmilLayers=4, transmilHeads=4), "learningRate": 0.001}
    arm = resolvers.with_arm_model(base, "clam_sb")
    assert arm["model"] == "clam_sb" and arm["learningRate"] == 0.001
    assert arm["transmilLayers"] == 2 and arm["transmilHeads"] == 8
    # Options the base never set stay unset, so older recipes keep their shape.
    assert "clamBagWeight" not in arm
    assert resolvers.with_arm_model(base, "transmil") is base
    tuned_clam = resolvers.with_arm_model({**base, "clamBagWeight": 0.5}, "clam_mb")
    assert tuned_clam["clamBagWeight"] == 0.5


def test_foreign_options_are_everything_another_architecture_owns():
    assert set(catalog.foreign_options("abmil")) == set(catalog.OPTIONS)
    own = {option.key for option in catalog.options("dsmil")}
    assert set(catalog.foreign_options("dsmil")) == set(catalog.OPTIONS) - own
    assert catalog.option_defaults("acmil") == {
        "acmilBranches": 5,
        "acmilMaskedPatches": 10,
        "acmilMaskDrop": 0.6,
        "acmilAttentionDim": 128,
        "acmilDropout": 0.0,
    }


def profile(recipe):
    rows = [
        {"slideId": "a", "patientId": "a", "partition": "train"},
        {"slideId": "b", "patientId": "b", "partition": "val"},
    ]
    files = {
        "a": {"patchCount": 4000, "dimensions": 1024},
        "b": {"patchCount": 9000, "dimensions": 1024},
    }
    return estimator.fold_workloads(recipe, rows, files, "mmap")


def test_existing_workload_keys_do_not_change():
    workload = profile({"model": "abmil"})
    assert "modelOptions" not in workload
    identity = {
        "version": estimator.KEY_VERSION,
        **{field: workload.get(field) for field in estimator.MEMORY_FIELDS},
        "featureDimension": workload["featureDimension"],
        "loadingPolicy": workload["loadingPolicy"],
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert workload["key"] == hashlib.sha256(canonical.encode()).hexdigest()[:24]
    # Stale options of another architecture neither enter nor change the key.
    assert profile({"model": "abmil", "transmilLayers": 6})["key"] == workload["key"]


def test_shape_changing_options_key_memory_and_scale_the_estimate():
    base = profile({"model": "transmil"})
    assert base["modelOptions"] == {
        "transmilLayers": 2,
        "transmilHeads": 8,
        "transmilLandmarks": 256,
    }
    deeper = profile({"model": "transmil", "transmilLayers": 4})
    assert deeper["key"] != base["key"]
    assert estimator.gpu_estimate(deeper) > estimator.gpu_estimate(base)
    # Dropout does not change tensor shapes, so it shares the memory class.
    assert profile({"model": "transmil", "transmilDropout": 0.3})["key"] == base["key"]
    # A workload recorded without its options is estimated with the catalog defaults.
    recorded = {key: value for key, value in base.items() if key != "modelOptions"}
    assert estimator.gpu_estimate(recorded) == estimator.gpu_estimate(base)
    for name in NEW_MODELS:
        workload = profile({"model": name})
        assert set(workload["modelOptions"]) == set(catalog.memory_options(name))
        assert estimator.gpu_estimate(workload) > 1


def test_published_defaults_and_shared_fields_that_depend_on_an_option():
    defaults = catalog.option_defaults()
    assert defaults["dsmilProjection"] is False
    assert defaults["clamRecipeAttentionDropout"] is False
    assert (defaults["dtfdTestSplits"], defaults["dtfdResidualBlocks"]) == (1, 0)
    # DSMIL reads the input-layer settings only with its shared input layer on.
    assert not catalog.reads_enabled("dsmil", "embedDim", {})
    assert not catalog.reads_enabled("dsmil", "numFcLayers", {"dsmilProjection": False})
    assert catalog.reads_enabled("dsmil", "numFcLayers", {"dsmilProjection": True})
    assert catalog.reads_enabled("dsmil", "inputDropout", {})
    # DTFD-MIL and ACMIL have the published single input layer, not the shared stack.
    assert not catalog.reads_enabled("dtfd", "numFcLayers", {})
    assert catalog.reads_enabled("acmil", "embedDim", {})
    described = {item["name"]: item for item in catalog.describe()}
    assert described["dsmil"]["readsEnabledWhen"] == {
        "embedDim": {"key": "dsmilProjection", "values": [True]},
        "numFcLayers": {"key": "dsmilProjection", "values": [True]},
    }
    assert described["dtfd"]["reads"] == described["acmil"]["reads"] == ["embedDim", "inputDropout"]
    assert described["transmil"]["readsEnabledWhen"] == {}
