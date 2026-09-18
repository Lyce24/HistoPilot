"""Masked mean/max MIL baselines using the same patch encoder as ABMIL."""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from histopilot.models.abmil import patch_projection, prepare_bags


class PoolingMIL(nn.Module):
    def __init__(
        self,
        in_dim,
        num_classes,
        *,
        pooling="mean",
        embed_dim=512,
        num_fc_layers=1,
        dropout=0.25,
        input_dropout=0.0,
        gradient_checkpointing=False,
    ):
        super().__init__()
        if min(in_dim, num_classes, embed_dim, num_fc_layers) < 1:
            raise ValueError("Pooling MIL requires positive dimensions, layers, and output logits.")
        if pooling not in {"mean", "max"}:
            raise ValueError("Pooling must be mean or max.")
        if not 0 <= dropout < 1 or not 0 <= input_dropout < 1:
            raise ValueError("Dropout probabilities must be in [0, 1).")
        self.in_dim = in_dim
        self.pooling = pooling
        self.gradient_checkpointing = gradient_checkpointing
        self.input_dropout = nn.Dropout(input_dropout)
        self.patch_embed = patch_projection(in_dim, embed_dim, num_fc_layers, dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, features, mask=None, *, return_attention=False):
        if return_attention:
            raise ValueError("Mean/max pooling models do not have learned attention weights.")
        features, mask = prepare_bags(features, mask, self.in_dim)
        features = self.input_dropout(features)
        embeddings = (
            checkpoint(self.patch_embed, features, use_reentrant=False)
            if self.gradient_checkpointing and self.training
            else self.patch_embed(features)
        )
        with torch.autocast(device_type=features.device.type, enabled=False):
            embeddings = embeddings.float()
            if self.pooling == "mean":
                pooled = embeddings.masked_fill(~mask.unsqueeze(-1), 0).sum(dim=1)
                pooled = pooled / mask.sum(dim=1, keepdim=True)
            else:
                pooled = embeddings.masked_fill(~mask.unsqueeze(-1), float("-inf")).amax(dim=1)
        return self.classifier(pooled)
