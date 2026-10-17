"""TransMIL: a Nyström Transformer over patch tokens with pyramid position encoding.

Method: Shao et al., *TransMIL: Transformer based Correlated Multiple Instance Learning
for Whole Slide Image Classification*, NeurIPS 2021. Reviewed against
szc19990412/TransMIL ``models/TransMIL.py`` and lucidrains/nystrom-attention
``nystrom_attention.py`` (main, 2026-10-16), which TransMIL imports; this module has
no upstream runtime dependency.

Each slide is its own sequence, as in the reference's one-slide batches: padding never
enters attention, and the pseudo-inverse's scaling, which the reference takes over the
whole batch, depends on that slide alone. The class token is followed by the patch
tokens in bag order, then copies of the first ones to fill a square grid for the
position encoding. The Nyström core always runs in float32. Layer count, heads,
landmarks, iterations and dropout are configurable; the position encoding follows the
first layer (before the only layer of a one-layer model), and every head has
``embed_dim / heads`` dimensions.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from histopilot.models.abmil import patch_projection, prepare_bags

RESIDUAL_KERNEL = 33


def moore_penrose_pinv(values, iterations):
    """Iterative Moore-Penrose pseudo-inverse of each square matrix, as nystrom-attention."""
    absolute = values.abs()
    scale = absolute.sum(dim=-1).max() * absolute.sum(dim=-2).max()
    inverse = values.transpose(-1, -2) / scale
    identity = torch.eye(values.shape[-1], device=values.device, dtype=values.dtype)
    for _ in range(iterations):
        product = values @ inverse
        inverse = (
            0.25
            * inverse
            @ (13 * identity - (product @ (15 * identity - (product @ (7 * identity - product)))))
        )
    return inverse


class NystromAttention(nn.Module):
    def __init__(self, dim, heads, landmarks, pinv_iterations, residual, dropout):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.landmarks = landmarks
        self.pinv_iterations = pinv_iterations
        self.scale = self.head_dim**-0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))
        self.residual = (
            nn.Conv2d(
                heads,
                heads,
                (RESIDUAL_KERNEL, 1),
                padding=(RESIDUAL_KERNEL // 2, 0),
                groups=heads,
                bias=False,
            )
            if residual
            else None
        )

    def _heads(self, values):
        batch, length, _ = values.shape
        return values.reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, tokens):
        batch, length, _ = tokens.shape
        landmarks = self.landmarks
        # Zero tokens in front make the length a multiple of the landmark count.
        remainder = length % landmarks
        if remainder:
            tokens = F.pad(tokens, (0, 0, landmarks - remainder, 0), value=0)
        query, key, value = (self._heads(part) for part in self.to_qkv(tokens).chunk(3, dim=-1))
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            query, key, value = query.float() * self.scale, key.float(), value.float()
            group = math.ceil(length / landmarks)
            shape = (batch, self.heads, landmarks, group, self.head_dim)
            query_landmarks = query.reshape(shape).sum(dim=3) / group
            key_landmarks = key.reshape(shape).sum(dim=3) / group
            kernel_1 = torch.softmax(query @ key_landmarks.transpose(-1, -2), dim=-1)
            kernel_2 = torch.softmax(query_landmarks @ key_landmarks.transpose(-1, -2), dim=-1)
            kernel_3 = torch.softmax(query_landmarks @ key.transpose(-1, -2), dim=-1)
            output = (kernel_1 @ moore_penrose_pinv(kernel_2, self.pinv_iterations)) @ (
                kernel_3 @ value
            )
            if self.residual is not None:
                output = output + self.residual(value)
        output = output.transpose(1, 2).reshape(batch, -1, self.heads * self.head_dim)
        return self.to_out(output.to(tokens.dtype))[:, -length:]

    def class_attention(self, tokens):
        """The first token's exact softmax attention over all tokens, averaged over heads."""
        query, key, _ = self.to_qkv(tokens).chunk(3, dim=-1)
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            query = self._heads(query[:, :1]).float() * self.scale
            key = self._heads(key).float()
            return torch.softmax(query @ key.transpose(-1, -2), dim=-1).mean(dim=1)[:, 0]


class TransLayer(nn.Module):
    def __init__(self, dim, heads, landmarks, pinv_iterations, residual, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attention = NystromAttention(dim, heads, landmarks, pinv_iterations, residual, dropout)

    def forward(self, tokens):
        return tokens + self.attention(self.norm(tokens))


class PPEG(nn.Module):
    """Pyramid position encoding: depthwise 7×7, 5×5 and 3×3 convolutions plus identity."""

    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, tokens, side):
        batch, _, channels = tokens.shape
        class_token, patches = tokens[:, :1], tokens[:, 1:]
        grid = patches.transpose(1, 2).reshape(batch, channels, side, side)
        grid = self.proj(grid) + grid + self.proj1(grid) + self.proj2(grid)
        return torch.cat((class_token, grid.flatten(2).transpose(1, 2)), dim=1)


class TransMIL(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        *,
        embed_dim: int = 512,
        num_fc_layers: int = 1,
        input_dropout: float = 0.0,
        layers: int = 2,
        heads: int = 8,
        landmarks: int = 256,
        pinv_iterations: int = 6,
        dropout: float = 0.1,
        position_encoding: bool = True,
        residual_conv: bool = True,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        if min(in_dim, num_classes, embed_dim, num_fc_layers, layers, heads, landmarks) < 1:
            raise ValueError("TransMIL requires positive dimensions, layers, heads and landmarks.")
        if pinv_iterations < 1:
            raise ValueError("TransMIL needs at least one pseudo-inverse iteration.")
        if embed_dim % heads:
            raise ValueError("TransMIL embedding dimensions must divide evenly by its heads.")
        if not 0 <= dropout < 1 or not 0 <= input_dropout < 1:
            raise ValueError("Dropout probabilities must be in [0, 1).")
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.gradient_checkpointing = gradient_checkpointing
        self.input_dropout = nn.Dropout(input_dropout)
        self.patch_embed = patch_projection(in_dim, embed_dim, num_fc_layers, 0.0)
        self.class_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.layers = nn.ModuleList(
            TransLayer(embed_dim, heads, landmarks, pinv_iterations, residual_conv, dropout)
            for _ in range(layers)
        )
        self.position = PPEG(embed_dim) if position_encoding else None
        self.position_before = 1 if layers > 1 else 0
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def _slide(self, features, attention=False):
        """One slide's logits and class embedding, and on request its per-patch map."""
        embeddings = self.patch_embed(self.input_dropout(features))[None]
        count = embeddings.shape[1]
        side = math.ceil(math.sqrt(count))
        extra = side * side - count
        tokens = torch.cat(
            (
                self.class_token.to(embeddings.dtype),
                embeddings,
                embeddings[:, :extra],
            ),
            dim=1,
        )
        last = len(self.layers) - 1
        weights = None
        for index, layer in enumerate(self.layers):
            if self.position is not None and index == self.position_before:
                tokens = self.position(tokens, side)
            if attention and index == last:
                attended = layer.attention.class_attention(layer.norm(tokens))[0, 1:]
                weights = attended[:count].clone()
                weights[:extra] = weights[:extra] + attended[count:]
                weights = weights / weights.sum()
            tokens = (
                checkpoint(layer, tokens, use_reentrant=False)
                if self.gradient_checkpointing and self.training
                else layer(tokens)
            )
        embedding = self.norm(tokens)[0, 0]
        return self.classifier(embedding), embedding, weights

    def forward(self, features, mask=None, *, return_attention=False):
        features, mask = prepare_bags(features, mask, self.in_dim)
        slides = [
            self._slide(
                features[index].index_select(0, mask[index].nonzero().squeeze(1)),
                return_attention,
            )
            for index in range(len(features))
        ]
        logits = torch.stack([slide[0] for slide in slides])
        if not return_attention:
            return logits
        attention = torch.zeros(mask.shape, dtype=torch.float32, device=features.device)
        for index, slide in enumerate(slides):
            attention[index, mask[index]] = slide[2]
        return {
            "logits": logits,
            "embedding": torch.stack([slide[1] for slide in slides]),
            "attention": attention,
        }
