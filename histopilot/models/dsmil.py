"""DSMIL: dual-stream MIL with a max-instance stream and critical-instance attention.

Method: Li, Li and Eliceiri, *Dual-stream Multiple Instance Learning Network for Whole
Slide Image Classification with Self-supervised Contrastive Learning*, CVPR 2021.
Reviewed against binli123/dsmil-wsi ``dsmil.py`` and ``train_tcga.py`` (master,
2026-10-16); this module has no upstream runtime dependency.

As in the reference, the streams classify the patch features directly; ``projection``
first applies the shared patch projection (Linear and ReLU layers), as standardized
benchmark libraries do. HistoPilot adds masked padding and minibatches. The fitting
loss weights the slide and max-instance streams 0.5/0.5 like the reference, with the
recipe's loss: one-logit BCE is the reference's binary setting, while several logits
use the recipe's loss where the reference applies BCE to every class. Averaged
prediction is the mean of the two streams' probabilities.
"""

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from histopilot.models.abmil import mean_probability_logits, patch_projection, prepare_bags


class DSMIL(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        *,
        projection: bool = False,
        embed_dim: int = 512,
        num_fc_layers: int = 1,
        input_dropout: float = 0.0,
        query_dim: int = 128,
        nonlinear_query: bool = True,
        passing_value: bool = False,
        value_dropout: float = 0.0,
        max_loss_weight: float = 0.5,
        average_prediction: bool = False,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        if min(in_dim, num_classes, embed_dim, num_fc_layers, query_dim) < 1:
            raise ValueError("DSMIL requires positive dimensions, layers and output logits.")
        if not 0 <= input_dropout < 1 or not 0 <= value_dropout < 1:
            raise ValueError("Dropout probabilities must be in [0, 1).")
        if not 0 <= max_loss_weight <= 1:
            raise ValueError("The DSMIL max-instance loss weight must be between 0 and 1.")
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.query_dim = query_dim
        self.max_loss_weight = max_loss_weight
        self.average_prediction = average_prediction
        self.gradient_checkpointing = gradient_checkpointing
        self.input_dropout = nn.Dropout(input_dropout)
        width = embed_dim if projection else in_dim
        self.patch_embed = (
            patch_projection(in_dim, embed_dim, num_fc_layers, 0.0) if projection else nn.Identity()
        )
        self.instance_classifier = nn.Linear(width, num_classes)
        self.query = (
            nn.Sequential(
                nn.Linear(width, query_dim),
                nn.ReLU(),
                nn.Linear(query_dim, query_dim),
                nn.Tanh(),
            )
            if nonlinear_query
            else nn.Linear(width, query_dim)
        )
        self.value = (
            nn.Sequential(nn.Dropout(value_dropout), nn.Linear(width, width), nn.ReLU())
            if passing_value
            else nn.Identity()
        )
        # The reference's one-dimensional convolution: every class reads every class's
        # pooled embedding, so it is the slide classifier.
        self.classifier = nn.Conv1d(num_classes, num_classes, kernel_size=width)

    def _streams(self, features, mask):
        features, mask = prepare_bags(features, mask, self.in_dim)
        features = self.input_dropout(features)
        embeddings = (
            checkpoint(self.patch_embed, features, use_reentrant=False)
            if self.gradient_checkpointing and self.training
            else self.patch_embed(features)
        )
        instance_logits = self.instance_classifier(embeddings)
        with torch.autocast(device_type=features.device.type, enabled=False):
            embeddings = embeddings.float()
            scores = instance_logits.float().masked_fill(~mask[..., None], float("-inf"))
            # Each class's critical instance is its highest-scoring present patch.
            max_logits, critical = scores.max(dim=1)
            critical_embeddings = torch.gather(
                embeddings, 1, critical[..., None].expand(-1, -1, embeddings.shape[-1])
            )
            queries = self.query(embeddings)
            attention = queries @ self.query(critical_embeddings).transpose(1, 2)
            attention = (attention / math.sqrt(self.query_dim)).masked_fill(
                ~mask[..., None], float("-inf")
            )
            weights = torch.softmax(attention, dim=1)
            pooled = weights.transpose(1, 2) @ self.value(embeddings)
            bag_logits = self.classifier(pooled).squeeze(-1)
        return bag_logits, max_logits, weights, pooled

    def _prediction(self, bag_logits, max_logits):
        if self.average_prediction:
            return mean_probability_logits(bag_logits, max_logits)
        return bag_logits

    def forward(self, features, mask=None, *, return_attention=False):
        bag_logits, max_logits, weights, pooled = self._streams(features, mask)
        logits = self._prediction(bag_logits, max_logits)
        if not return_attention:
            return logits
        # One map per class; a one-logit binary model has a single one.
        branch = (
            torch.zeros(len(logits), dtype=torch.long, device=logits.device)
            if self.num_classes == 1
            else logits.argmax(dim=-1)
        )
        rows = torch.arange(len(logits), device=logits.device)
        return {
            "logits": logits,
            "embedding": pooled[rows, branch],
            "attention": weights.transpose(1, 2)[rows, branch],
        }

    def training_output(self, features, mask, labels, criterion):
        """The slide stream's logits plus the weighted max-instance loss."""
        bag_logits, max_logits, _, _ = self._streams(features, mask)
        return {
            "logits": bag_logits,
            "auxiliaryLoss": self.max_loss_weight * criterion(max_logits, labels),
            "bagLossWeight": 1 - self.max_loss_weight,
        }
