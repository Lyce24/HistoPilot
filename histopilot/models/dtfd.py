"""DTFD-MIL: double-tier feature distillation over random pseudo-bags.

Method: Zhang et al., *DTFD-MIL: Double-Tier Feature Distillation Multiple Instance
Learning for Histopathology Whole Slide Image Classification*, CVPR 2022. Reviewed
against hrzhang1123/DTFD-MIL ``Main_DTFD_MIL.py`` and ``Model/`` (main, 2026-10-16);
this module has no upstream runtime dependency.

Tier one splits a slide's patches into random pseudo-bags, pools each with gated
attention and classifies it with the slide's label; tier two attends over what each
pseudo-bag distils. Each tier learns from its own loss only, as the paper's objective
states (Eq. 14) and the reference's two optimizers do: it updates tier one before tier
two's loss exists and clears tier one's gradients before the next slide. Here one
optimizer steps both tiers, and tier two reads the distilled features detached, which
gives each tier the same gradients. The input layer is the reference's: a bias-free
Linear and ReLU, optionally followed by its residual blocks. The reference draws new
random splits when testing and averages ``num_MeanInference`` of them (one by default);
here the ``test_splits`` splits come from fixed seeds, so the same slide always gets the
same prediction. A slide with fewer patches than pseudo-bags uses one pseudo-bag per
patch, where the reference adds empty ones. Distilled patches are ranked by the last
class's probability as in the reference, or by the positive logit of a one-logit head.
"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from histopilot.models.abmil import mean_probability_logits, prepare_bags

DISTILL = ("afs", "maxmins", "maxs")


class _GatedAttention(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.attention_v = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh())
        self.attention_u = nn.Sequential(nn.Linear(dim, hidden), nn.Sigmoid())
        self.attention_w = nn.Linear(hidden, 1)

    def forward(self, values):
        return self.attention_w(self.attention_v(values) * self.attention_u(values)).squeeze(-1)


class _Classifier(nn.Module):
    def __init__(self, dim, classes, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(dim, classes)

    def forward(self, values):
        return self.fc(self.dropout(values))


class _ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(),
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(),
        )

    def forward(self, values):
        return values + self.block(values)


class DimReduction(nn.Module):
    """The reference's input layer: bias-free Linear and ReLU, then residual blocks."""

    def __init__(self, in_dim, embed_dim, residual_blocks=0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, embed_dim, bias=False)
        self.relu1 = nn.ReLU()
        self.resBlocks = nn.Sequential(*(_ResidualBlock(embed_dim) for _ in range(residual_blocks)))

    def forward(self, values):
        return self.resBlocks(self.relu1(self.fc1(values)))


def split_sizes(count, groups):
    """``numpy.array_split`` sizes: the first ``count % groups`` groups get one more."""
    base, extra = divmod(count, groups)
    return [base + (1 if index < extra else 0) for index in range(min(groups, count))]


class DTFDMIL(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        *,
        embed_dim: int = 512,
        residual_blocks: int = 0,
        input_dropout: float = 0.0,
        pseudo_bags: int = 4,
        test_splits: int = 1,
        distill: str = "afs",
        distill_patches: int = 1,
        attention_dim: int = 128,
        dropout: float = 0.0,
        gradient_checkpointing: bool = False,
        evaluation_seed: int = 42,
    ):
        super().__init__()
        if min(in_dim, num_classes, embed_dim, pseudo_bags, test_splits, attention_dim) < 1:
            raise ValueError("DTFD-MIL requires positive dimensions, pseudo-bags and splits.")
        if residual_blocks < 0:
            raise ValueError("DTFD-MIL residual blocks cannot be negative.")
        if distill not in DISTILL or distill_patches < 1:
            raise ValueError("DTFD-MIL distillation must be afs, maxmins or maxs.")
        if not 0 <= dropout < 1 or not 0 <= input_dropout < 1:
            raise ValueError("Dropout probabilities must be in [0, 1).")
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.pseudo_bags = pseudo_bags
        self.test_splits = test_splits
        self.distill = distill
        self.distill_patches = distill_patches
        self.evaluation_seed = evaluation_seed
        self.gradient_checkpointing = gradient_checkpointing
        self.input_dropout = nn.Dropout(input_dropout)
        self.patch_embed = DimReduction(in_dim, embed_dim, residual_blocks)
        self.tier1_attention = _GatedAttention(embed_dim, attention_dim)
        self.tier1_classifier = _Classifier(embed_dim, num_classes, dropout)
        self.tier2_attention = _GatedAttention(embed_dim, attention_dim)
        self.classifier = _Classifier(embed_dim, num_classes, dropout)

    def _orders(self, count, device):
        """One random split while fitting; ``test_splits`` seeded splits when testing."""
        if self.training:
            return [torch.randperm(count, device=device)]
        return [
            torch.randperm(
                count,
                generator=torch.Generator(device="cpu").manual_seed(self.evaluation_seed + split),
            ).to(device)
            for split in range(self.test_splits)
        ]

    def _rank(self, weighted):
        """Class activation of each patch from tier one's classifier, as the reference ranks."""
        activation = weighted @ self.tier1_classifier.fc.weight.transpose(0, 1)
        return activation[:, 0] if self.num_classes == 1 else torch.softmax(activation, -1)[:, -1]

    def _split(self, embeddings, scores, order, patch_weights=False):
        """One split's tier-one logits, slide logits, embedding and, on request, the
        weight each patch carries in that embedding."""
        count = len(embeddings)
        tier1_logits, distilled, sources, source_weights = [], [], [], []
        for group in order.split(split_sizes(count, self.pseudo_bags)):
            members = embeddings.index_select(0, group)
            attention = torch.softmax(scores.index_select(0, group), dim=0)
            weighted = attention[:, None] * members
            pooled = weighted.sum(dim=0)
            tier1_logits.append(self.tier1_classifier(pooled))
            if self.distill == "afs":
                distilled.append(pooled[None])
                sources.append(group)
                source_weights.append(attention)
                continue
            ranked = torch.sort(self._rank(weighted), descending=True, stable=True).indices
            keep = min(self.distill_patches, len(group))
            chosen = (
                torch.cat((ranked[:keep], ranked[-keep:]))
                if self.distill == "maxmins"
                else ranked[:keep]
            )
            distilled.append(members.index_select(0, chosen))
            sources.append(group.index_select(0, chosen))
        # Tier two's loss must not train tier one (see the module docstring).
        distilled = torch.cat(distilled).detach()
        tier2 = torch.softmax(self.tier2_attention(distilled), dim=0)
        embedding = tier2 @ distilled
        logits = self.classifier(embedding)
        weights = None
        if patch_weights:
            # Every distilled row is one patch, or one pseudo-bag's attention-weighted patches.
            contributions = (
                torch.cat([tier2[index] * values for index, values in enumerate(source_weights)])
                if self.distill == "afs"
                else tier2
            )
            weights = torch.zeros(count, dtype=torch.float32, device=embeddings.device)
            weights = weights.index_add(0, torch.cat(sources), contributions)
        return torch.stack(tier1_logits), logits, embedding, weights

    def _slide(self, embeddings, patch_weights=False):
        """A slide's outputs; several test splits average probabilities, embeddings and maps."""
        scores = self.tier1_attention(embeddings)
        views = [
            self._split(embeddings, scores, order, patch_weights)
            for order in self._orders(len(embeddings), embeddings.device)
        ]
        if len(views) == 1:
            return views[0]
        logits = mean_probability_logits(*(view[1] for view in views))
        embedding = torch.stack([view[2] for view in views]).mean(dim=0)
        weights = torch.stack([view[3] for view in views]).mean(dim=0) if patch_weights else None
        return views[0][0], logits, embedding, weights

    def _forward(self, features, mask, patch_weights=False):
        features, mask = prepare_bags(features, mask, self.in_dim)
        features = self.input_dropout(features)
        embeddings = (
            checkpoint(self.patch_embed, features, use_reentrant=False)
            if self.gradient_checkpointing and self.training
            else self.patch_embed(features)
        )
        slides = []
        with torch.autocast(device_type=features.device.type, enabled=False):
            embeddings = embeddings.float()
            for index in range(len(features)):
                present = mask[index].nonzero().squeeze(1)
                tier1, logits, embedding, weights = self._slide(
                    embeddings[index].index_select(0, present), patch_weights
                )
                if weights is not None:
                    weights = torch.zeros(
                        mask.shape[1], dtype=torch.float32, device=features.device
                    ).index_copy(0, present, weights)
                slides.append((tier1, logits, embedding, weights))
        return slides

    def forward(self, features, mask=None, *, return_attention=False):
        slides = self._forward(features, mask, patch_weights=return_attention)
        logits = torch.stack([slide[1] for slide in slides])
        if not return_attention:
            return logits
        return {
            "logits": logits,
            "embedding": torch.stack([slide[2] for slide in slides]),
            "attention": torch.stack([slide[3] for slide in slides]),
        }

    def training_output(self, features, mask, labels, criterion):
        """Slide logits plus each slide's mean tier-one loss over its pseudo-bags."""
        slides = self._forward(features, mask)
        tier1 = torch.stack(
            [
                criterion(slide[0], labels[index].expand(len(slide[0]))).mean()
                for index, slide in enumerate(slides)
            ]
        )
        return {
            "logits": torch.stack([slide[1] for slide in slides]),
            "auxiliaryLoss": tier1,
            "bagLossWeight": 1.0,
        }
