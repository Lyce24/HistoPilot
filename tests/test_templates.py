"""Server-owned templates: the browser's copy matches, and every starter is valid science.

The browser reads web/src/lib/templates.json instead of repeating these values, and the
CLI and agent tools read the same module, so the two can no longer drift apart.
"""

import runpy
from pathlib import Path

import pytest

from histopilot import templates
from histopilot.schemas.development import DevelopmentBatchSpec, TrainingRecipe
from histopilot.schemas.evaluations import InferenceSettings
from histopilot.schemas.protocols import SplitSpec

ROOT = Path(__file__).resolve().parents[1]


def test_the_browsers_copy_matches_the_templates_byte_for_byte():
    generator = runpy.run_path(str(ROOT / "scripts" / "export_templates.py"))
    assert generator["TARGET"].read_text(encoding="utf-8") == generator["render"](), (
        "Regenerate it with: python scripts/export_templates.py"
    )


@pytest.mark.parametrize("preset", ["default", *templates.RECIPE_PRESETS])
def test_every_recipe_preset_is_a_valid_training_recipe(preset):
    TrainingRecipe.model_validate(templates.recipe(preset))


@pytest.mark.parametrize("preset", [item["id"] for item in templates.BATCH_PRESETS])
def test_every_batch_preset_is_a_valid_batch(preset):
    inputs = {"protocolId": "p", "featureBundleId": "b", "loadingPolicy": "auto"}
    spec = templates.batch(preset, inputs=inputs, experiment_name="Study")
    if preset == "blank":
        spec["batchName"] = "Named later"
    DevelopmentBatchSpec.model_validate(spec)


def test_the_training_design_is_version_4_with_pools():
    split = SplitSpec.model_validate(templates.STARTERS["trainingSplit"])
    assert split.version == 4 and split.pools is not None


def test_apply_writes_out_the_fields_whose_service_defaults_differ():
    assert templates.APPLY["scope"] == "selected"
    # A partial inference object would fall back to mean aggregation and a 0.5 threshold.
    assert set(templates.APPLY["inference"]) == set(InferenceSettings.model_fields)


def test_starters_state_the_units_and_inclusions_the_service_defaults_otherwise():
    starters = templates.STARTERS
    assert starters["targetSplit"]["splitUnit"] == "slide"
    assert starters["targetSplit"]["target"]["unit"] == "slide"
    assert starters["datasetImport"]["includeMissingSlides"] is True
    assert starters["datasetImport"]["recursive"] is True
    assert templates.BATCH_DEFAULTS["predictorPolicy"] == {
        "method": "ensemble",
        "refitPercentile": None,
    }


def test_describe_returns_copies():
    first = templates.describe()
    first["recipes"]["default"]["model"] = "changed"
    assert templates.describe()["recipes"]["default"]["model"] == "abmil"
