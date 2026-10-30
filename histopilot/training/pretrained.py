"""Start a fit from the pretrained weights its frozen batch pinned.

Only fitting reads the file: a fold or refit that starts fresh loads it into the model,
and a resumed one continues from its own checkpoint. Either way the file must still match
its pin (storage/model_weights.py), so the run's provenance stays true. Saved models
rebuild from their own checkpoints and never need the pretrained file again.
"""

import io

import torch

from histopilot.models import catalog
from histopilot.storage.model_weights import read_pinned
from histopilot.storage.project_lock import StorageError


def load_pinned_weights(model, recipe, pins, *, load=True):
    """Verify the recipe's pinned checkpoint and, with ``load``, put it into ``model``.

    Returns the pin ({path, bytes, sha256}), or None when the recipe names no checkpoint.
    """
    path = catalog.initial_checkpoint(recipe)
    if path is None:
        return None
    pinned, content = read_pinned(pins, path)
    if load:
        state = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
        # OceanPath's own loader also accepts these two wrappers around the state dict.
        for wrapper in ("state_dict", "model_state_dict"):
            if isinstance(state, dict) and isinstance(state.get(wrapper), dict):
                state = state[wrapper]
        try:
            model.load_pretrained(state)
        except (RuntimeError, TypeError, ValueError, AttributeError) as error:
            raise StorageError(
                f"The aggregator checkpoint {pinned['path']} does not fit this model's "
                f"settings: {error}",
                "MODEL_CHECKPOINT_INCOMPATIBLE",
                422,
            ) from error
    return pinned
