"""Native attention MIL classifier with variable-size, masked bags.

Architecture attribution: Ilse, Tomczak and Welling, *Attention-based Deep
Multiple Instance Learning* (ICML 2018). Design reviewed against the local
OceanPath ABMIL and training modules; this implementation has no OceanPath
runtime dependency. Torch remains confined to optional compute modules.
"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class ABMIL(nn.Module):
    """Patch projection, gated attention pooling, and a multiclass logit head."""

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        *,
        embed_dim: int = 512,
        attention_dim: int = 384,
        num_fc_layers: int = 1,
        gated_attention: bool = True,
        dropout: float = 0.25,
        input_dropout: float = 0.0,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        if min(in_dim, embed_dim, attention_dim, num_fc_layers) < 1 or num_classes < 2:
            raise ValueError("ABMIL requires positive dimensions/layers and at least two classes.")
        if not 0 <= dropout < 1 or not 0 <= input_dropout < 1:
            raise ValueError("Dropout probabilities must be in [0, 1).")
        self.in_dim = in_dim
        self.gradient_checkpointing = gradient_checkpointing
        self.input_dropout = nn.Dropout(input_dropout)
        layers = []
        for layer in range(num_fc_layers):
            layers.extend(
                [
                    nn.Linear(in_dim if layer == 0 else embed_dim, embed_dim),
                    nn.ReLU(),
                ]
            )
            # Match OceanPath's projection: regularize between FC layers, while
            # keeping the final projected embedding intact for attention.
            if layer < num_fc_layers - 1 and dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.patch_embed = nn.Sequential(*layers)
        self.attention_tanh = nn.Sequential(
            nn.Linear(embed_dim, attention_dim), nn.Tanh(), nn.Dropout(dropout)
        )
        self.attention_gate = (
            nn.Sequential(nn.Linear(embed_dim, attention_dim), nn.Sigmoid(), nn.Dropout(dropout))
            if gated_attention
            else None
        )
        self.attention_score = nn.Linear(attention_dim, 1)
        self.classifier = nn.Linear(embed_dim, num_classes)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def _attention_logits(self, embeddings):
        attention = self.attention_tanh(embeddings)
        if self.attention_gate is not None:
            attention = attention * self.attention_gate(embeddings)
        return self.attention_score(attention).squeeze(-1)

    def forward(self, features, mask=None, *, return_attention=False):
        if features.ndim != 3 or features.shape[-1] != self.in_dim or features.shape[1] == 0:
            raise ValueError("Features must have shape [bags, nonempty patches, input dimensions].")
        if mask is None:
            mask = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
        elif mask.shape != features.shape[:2]:
            raise ValueError("Attention masks must match the bags and patch dimensions.")
        else:
            mask = mask.to(device=features.device, dtype=torch.bool)
        if not bool(mask.any(dim=1).all()):
            raise ValueError("Every bag must contain at least one unmasked patch.")
        # Padding must not affect projections or produce 0*NaN during pooling.
        features = features.masked_fill(~mask.unsqueeze(-1), 0).float()
        embeddings = self.patch_embed(self.input_dropout(features))
        attention = (
            checkpoint(self._attention_logits, embeddings, use_reentrant=False)
            if self.gradient_checkpointing and self.training
            else self._attention_logits(embeddings)
        )
        # Keep normalization and accumulation stable under CPU/CUDA autocast.
        with torch.autocast(device_type=features.device.type, enabled=False):
            attention = attention.float().masked_fill(~mask, float("-inf"))
            weights = torch.softmax(attention, dim=1)
            pooled = torch.bmm(weights.unsqueeze(1), embeddings.float()).squeeze(1)
        logits = self.classifier(pooled)
        if return_attention:
            return {"logits": logits, "embedding": pooled, "attention": weights}
        return logits
