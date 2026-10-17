"""ACMIL: attention-challenging MIL with multiple branches and top-instance masking.

Method: Zhang et al., *Attention-Challenging Multiple Instance Learning for Whole Slide
Image Classification*, ECCV 2024. Reviewed against dazhangyu123/ACMIL
``architecture/transformer.py`` (``ACMIL_GA``) and ``Step3_WSI_classification_ACMIL.py``
(main, 2026-10-16); this module has no upstream runtime dependency.

Each branch has its own gated attention and classifier. While fitting, stochastic
top-k instance masking hides part of every branch's most attended patches, and the
loss adds the branches' classification losses and their mean pairwise attention
cosine similarity to the slide loss, all with weight one as in the reference. The
slide classifier reads patches pooled with the branches' mean attention. The input
layer is the reference's bias-free Linear and ReLU (ACMIL reuses DTFD-MIL's). HistoPilot
adds masked padding and minibatches and keeps at least one patch visible in every branch.
"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from histopilot.models.abmil import prepare_bags
from histopilot.models.dtfd import DimReduction


class _Classifier(nn.Module):
    def __init__(self, dim, classes, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(dim, classes)

    def forward(self, values):
        return self.fc(self.dropout(values))


class ACMIL(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        *,
        embed_dim: int = 512,
        input_dropout: float = 0.0,
        branches: int = 5,
        masked_patches: int = 10,
        mask_drop: float = 0.6,
        attention_dim: int = 128,
        dropout: float = 0.0,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        if min(in_dim, num_classes, embed_dim, branches, attention_dim) < 1:
            raise ValueError("ACMIL requires positive dimensions and branches.")
        if masked_patches < 0 or not 0 <= mask_drop < 1:
            raise ValueError(
                "ACMIL masking needs a nonnegative patch count and a fraction in [0, 1)."
            )
        if not 0 <= dropout < 1 or not 0 <= input_dropout < 1:
            raise ValueError("Dropout probabilities must be in [0, 1).")
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.branches = branches
        self.masked_patches = masked_patches
        self.mask_drop = mask_drop
        self.gradient_checkpointing = gradient_checkpointing
        self.input_dropout = nn.Dropout(input_dropout)
        self.patch_embed = DimReduction(in_dim, embed_dim)
        self.attention_v = nn.Sequential(nn.Linear(embed_dim, attention_dim), nn.Tanh())
        self.attention_u = nn.Sequential(nn.Linear(embed_dim, attention_dim), nn.Sigmoid())
        self.attention_w = nn.Linear(attention_dim, branches)
        self.branch_classifiers = nn.ModuleList(
            _Classifier(embed_dim, num_classes, dropout) for _ in range(branches)
        )
        self.classifier = _Classifier(embed_dim, num_classes, dropout)

    def _attention_scores(self, embeddings):
        return self.attention_w(self.attention_v(embeddings) * self.attention_u(embeddings))

    def _hidden(self, scores, mask):
        """Stochastic top-k instance masking: which present patches each branch cannot see."""
        hidden = torch.zeros(scores.shape, dtype=torch.bool, device=scores.device)
        for index in range(len(scores)):
            present = mask[index].nonzero().squeeze(1)
            top = min(self.masked_patches, len(present))
            # Never hide every present patch of a branch.
            drop = min(int(top * self.mask_drop), len(present) - 1)
            if drop < 1:
                continue
            candidates = scores[index].index_select(1, present).topk(top, dim=-1).indices
            chosen = torch.rand(candidates.shape, device=scores.device).argsort(dim=-1)[:, :drop]
            positions = present[candidates.gather(1, chosen)]
            hidden[index] = hidden[index].scatter(1, positions, True)
        return hidden

    def _forward(self, features, mask):
        features, mask = prepare_bags(features, mask, self.in_dim)
        embeddings = self.patch_embed(self.input_dropout(features))
        scores = (
            checkpoint(self._attention_scores, embeddings, use_reentrant=False)
            if self.gradient_checkpointing and self.training
            else self._attention_scores(embeddings)
        )
        with torch.autocast(device_type=features.device.type, enabled=False):
            embeddings = embeddings.float()
            scores = scores.float().transpose(1, 2)
            if self.training and self.masked_patches and self.mask_drop:
                # The reference's finite fill, so hidden patches stay in the softmax at ~0.
                scores = scores.masked_fill(self._hidden(scores.detach(), mask), -1e9)
            weights = torch.softmax(scores.masked_fill(~mask[:, None, :], float("-inf")), dim=-1)
            branch_embeddings = torch.bmm(weights, embeddings)
            slide_weights = weights.mean(dim=1)
            slide_embedding = torch.bmm(slide_weights[:, None, :], embeddings).squeeze(1)
        branch_logits = torch.stack(
            [
                head(branch_embeddings[:, index])
                for index, head in enumerate(self.branch_classifiers)
            ],
            dim=1,
        )
        return (
            self.classifier(slide_embedding),
            branch_logits,
            weights,
            slide_weights,
            slide_embedding,
        )

    def forward(self, features, mask=None, *, return_attention=False):
        logits, _, _, slide_weights, slide_embedding = self._forward(features, mask)
        if not return_attention:
            return logits
        return {"logits": logits, "embedding": slide_embedding, "attention": slide_weights}

    def training_output(self, features, mask, labels, criterion):
        """Slide logits plus the branches' mean loss and their attention similarity."""
        logits, branch_logits, weights, _, _ = self._forward(features, mask)
        count = len(logits)
        auxiliary = torch.zeros(count, dtype=torch.float32, device=logits.device)
        if self.branches > 1:
            auxiliary = auxiliary + criterion(
                branch_logits.flatten(0, 1), labels.repeat_interleave(self.branches)
            ).view(count, self.branches).mean(dim=1)
            pairs = self.branches * (self.branches - 1) / 2
            for first in range(self.branches):
                for second in range(first + 1, self.branches):
                    auxiliary = (
                        auxiliary
                        + torch.cosine_similarity(weights[:, first], weights[:, second], dim=-1)
                        / pairs
                    )
        return {"logits": logits, "auxiliaryLoss": auxiliary, "bagLossWeight": 1.0}
