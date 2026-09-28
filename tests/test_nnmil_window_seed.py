"""nnMIL can draw one feature-window permutation per training seed."""

import pytest

from histopilot.schemas.development import TrainingRecipe
from histopilot.schemas.nnmil import window_seed


def test_recipes_keep_their_fixed_window_seed_unless_asked():
    recipe = TrainingRecipe(model="nnmil", attentionDim=256).model_dump()
    # Stored recipes omit the new option, so their hashes and seeds are unchanged.
    assert "nnmilWindowSeedFromTraining" not in recipe
    assert [window_seed(recipe, seed) for seed in (42, 43)] == [42, 42]
    per_seed = TrainingRecipe(
        model="nnmil", attentionDim=256, nnmilWindowSeedFromTraining=True
    ).model_dump()
    assert per_seed["nnmilWindowSeedFromTraining"] is True
    assert [window_seed(per_seed, seed) for seed in (42, 43)] == [42, 43]


def test_only_nnmil_draws_window_seeds():
    with pytest.raises(ValueError, match="require nnMIL"):
        TrainingRecipe(model="abmil", nnmilWindowSeedFromTraining=True)
