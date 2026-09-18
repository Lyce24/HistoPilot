"""Probes over one embedding per slide from a slide encoder.

A slide encoder (TITAN, PRISM, CHIEF, GigaPath and the mean-pooled variants)
already reduces a whole slide to a single vector, so there is no bag to attend
over. These probes are the honest baseline for such features: a linear read-out,
or one hidden layer when a little capacity is wanted.

The bag arrives with exactly one instance so that the loaders, masks, batching
and metrics stay identical to the patch models. Masked instances would mean a
slide with no embedding, which the feature inventory already rejects.
"""

import torch
from torch import nn

from histopilot.models.abmil import prepare_bags


class SlideProbe(nn.Module):
    """Classify one slide embedding; ``hidden_dim`` of zero is a linear probe."""

    def __init__(
        self,
        in_dim,
        num_classes,
        *,
        hidden_dim=0,
        dropout=0.25,
        input_dropout=0.0,
    ):
        super().__init__()
        if min(in_dim, num_classes) < 1 or hidden_dim < 0:
            raise ValueError("A slide probe requires positive dimensions and output logits.")
        if not 0 <= dropout < 1 or not 0 <= input_dropout < 1:
            raise ValueError("Dropout probabilities must be in [0, 1).")
        self.in_dim = in_dim
        self.input_dropout = nn.Dropout(input_dropout)
        self.head = (
            nn.Linear(in_dim, num_classes)
            if not hidden_dim
            else nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    @property
    def classifier(self):
        """Expose the common training head without changing saved ``head.*`` keys.

        Optimizer construction identifies classifier parameters for a separate
        head learning rate. A property avoids registering this module twice and
        keeps already serialized probe state dictionaries loadable.
        """
        return self.head

    def forward(self, features, mask=None, *, return_attention=False):
        if return_attention:
            raise ValueError("Slide-embedding probes have no per-patch attention weights.")
        features, mask = prepare_bags(features, mask, self.in_dim)
        if features.shape[1] != 1 or not bool(mask.all()):
            raise ValueError(
                "A slide-embedding probe expects exactly one present embedding per slide."
            )
        embedding = self.input_dropout(features)[:, 0]
        with torch.autocast(device_type=features.device.type, enabled=False):
            return self.head(embedding.float())
