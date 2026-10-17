"""CLAM: clustering-constrained attention MIL, single- and multi-branch.

Method: Lu et al., *Data-efficient and weakly supervised computational pathology on
whole-slide images*, Nature Biomedical Engineering 2021. Reviewed against
mahmoodlab/CLAM ``models/model_clam.py`` and ``utils/core_utils.py`` (master,
2026-10-16) and oval-group/smooth-topk for the instance SVM loss; this module has no
upstream runtime dependency.

HistoPilot adds masked padding and minibatches, a one-logit binary head for the
single-branch model, and a configurable number of projection layers. As in the reference,
the attention branches drop 25% whenever dropout is on, whatever its value;
``recipe_attention_dropout`` uses the recipe's dropout there instead. A slide with fewer
than 2B patches samples half of them, where the reference fails below B patches and
labels shared patches both positive and negative below 2B.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from histopilot.models.abmil import prepare_bags


def smooth_top1_svm(logits, targets, *, alpha=1.0, tau=1.0, threshold=1e3):
    """Per-instance smooth top-1 SVM (smooth-topk ``SmoothTop1SVM``), CLAM's ``svm`` loss.

    Rows whose top two scores differ by at least ``tau * log(threshold)`` use the hard
    hinge, exactly as the reference switches, so the gradient there is the hinge's.
    """
    classes = logits.shape[-1]
    margin = alpha * (1 - F.one_hot(targets, classes).to(logits.dtype))
    shifted = logits + margin - logits.gather(1, targets[:, None])
    smooth = tau * torch.logsumexp(shifted / tau, dim=1)
    hard = shifted.max(dim=1).values
    top = logits.topk(min(2, classes), dim=1).values
    large = (top[:, 0] - top[:, -1]) >= tau * math.log(threshold)
    return torch.where(large, hard, smooth)


def _projection(in_dim, embed_dim, num_fc_layers, dropout):
    """CLAM's patch encoder: every layer is Linear, ReLU, then dropout."""
    layers = []
    for layer in range(num_fc_layers):
        layers.extend(
            [
                nn.Linear(in_dim if layer == 0 else embed_dim, embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        )
    return nn.Sequential(*layers)


def _valid_rows(values, mask_row):
    """The bag's present rows in their original order."""
    return values.index_select(0, mask_row.nonzero().squeeze(1))


class CLAM(nn.Module):
    """Attention pooling with instance-level clustering of each slide's extreme patches.

    ``multi_branch`` gives every class its own attention branch and one-logit slide
    classifier (CLAM-MB); otherwise one branch feeds a single classifier (CLAM-SB).
    ``training_output`` adds the instance-clustering loss, weighted ``1 - bag_weight``.
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        *,
        multi_branch: bool = False,
        embed_dim: int = 512,
        attention_dim: int = 256,
        num_fc_layers: int = 1,
        gated_attention: bool = True,
        dropout: float = 0.25,
        recipe_attention_dropout: bool = False,
        input_dropout: float = 0.0,
        instance_samples: int = 8,
        bag_weight: float = 0.7,
        instance_loss: str = "svm",
        subtyping: bool = False,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        if min(in_dim, embed_dim, attention_dim, num_fc_layers, num_classes, instance_samples) < 1:
            raise ValueError("CLAM requires positive dimensions, layers, samples and logits.")
        if multi_branch and num_classes < 2:
            raise ValueError(
                "CLAM-MB has one attention branch per class; use cross entropy or focal loss, "
                "not one-logit BCE."
            )
        if not 0 <= dropout < 1 or not 0 <= input_dropout < 1:
            raise ValueError("Dropout probabilities must be in [0, 1).")
        if not 0 <= bag_weight <= 1:
            raise ValueError("The CLAM slide loss weight must be between 0 and 1.")
        if instance_loss not in {"svm", "ce"}:
            raise ValueError("The CLAM instance loss must be svm or ce.")
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.multi_branch = multi_branch
        # A one-logit binary head still clusters instances for both classes.
        self.instance_classes = max(2, num_classes)
        self.instance_samples = instance_samples
        self.bag_weight = bag_weight
        self.instance_loss = instance_loss
        self.subtyping = subtyping
        self.gradient_checkpointing = gradient_checkpointing
        branches = num_classes if multi_branch else 1
        # The reference's Attn_Net(_Gated) appends Dropout(0.25) whenever dropout is truthy.
        attention_dropout = dropout if recipe_attention_dropout else (0.25 if dropout else 0.0)
        self.input_dropout = nn.Dropout(input_dropout)
        self.patch_embed = _projection(in_dim, embed_dim, num_fc_layers, dropout)
        self.attention_a = nn.Sequential(
            nn.Linear(embed_dim, attention_dim), nn.Tanh(), nn.Dropout(attention_dropout)
        )
        self.attention_b = (
            nn.Sequential(
                nn.Linear(embed_dim, attention_dim), nn.Sigmoid(), nn.Dropout(attention_dropout)
            )
            if gated_attention
            else None
        )
        self.attention_c = nn.Linear(attention_dim, branches)
        self.classifier = (
            nn.ModuleList(nn.Linear(embed_dim, 1) for _ in range(num_classes))
            if multi_branch
            else nn.Linear(embed_dim, num_classes)
        )
        self.instance_classifiers = nn.ModuleList(
            nn.Linear(embed_dim, 2) for _ in range(self.instance_classes)
        )

    def _attention_scores(self, embeddings):
        attention = self.attention_a(embeddings)
        if self.attention_b is not None:
            attention = attention * self.attention_b(embeddings)
        return self.attention_c(attention)

    def _forward(self, features, mask):
        features, mask = prepare_bags(features, mask, self.in_dim)
        embeddings = self.patch_embed(self.input_dropout(features))
        scores = (
            checkpoint(self._attention_scores, embeddings, use_reentrant=False)
            if self.gradient_checkpointing and self.training
            else self._attention_scores(embeddings)
        )
        with torch.autocast(device_type=features.device.type, enabled=False):
            scores = scores.float().transpose(1, 2).masked_fill(~mask[:, None, :], float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            pooled = torch.bmm(weights, embeddings.float())
        if self.multi_branch:
            logits = torch.cat(
                [head(pooled[:, index]) for index, head in enumerate(self.classifier)], dim=-1
            )
        else:
            logits = self.classifier(pooled[:, 0])
        return logits, weights, pooled, embeddings, mask

    def _branch(self, logits):
        """Each slide's map: the single branch, or the predicted class's branch."""
        if not self.multi_branch:
            return torch.zeros(len(logits), dtype=torch.long, device=logits.device)
        return logits.argmax(dim=-1)

    def forward(self, features, mask=None, *, return_attention=False):
        logits, weights, pooled, _, _ = self._forward(features, mask)
        if not return_attention:
            return logits
        branch = self._branch(logits)
        rows = torch.arange(len(logits), device=logits.device)
        return {
            "logits": logits,
            "embedding": pooled[rows, branch],
            "attention": weights[rows, branch],
        }

    def _instance_criterion(self, logits, targets):
        if self.instance_loss == "ce":
            return F.cross_entropy(logits, targets)
        return smooth_top1_svm(logits, targets).mean()

    def _slide_instance_loss(self, weights, embeddings, label):
        """CLAM's ``inst_eval``/``inst_eval_out`` over one slide's present patches."""
        count = len(embeddings)
        inside = min(self.instance_samples, count // 2)
        outside = min(self.instance_samples, count)
        total = embeddings.new_zeros((), dtype=torch.float32)
        for index, head in enumerate(self.instance_classifiers):
            branch = weights[index if self.multi_branch else 0]
            if index == label:
                if not inside:
                    continue
                chosen = torch.cat(
                    (branch.topk(inside).indices, branch.topk(inside, largest=False).indices)
                )
                targets = torch.cat(
                    (
                        torch.ones(inside, dtype=torch.long, device=branch.device),
                        torch.zeros(inside, dtype=torch.long, device=branch.device),
                    )
                )
            elif self.subtyping:
                chosen = branch.topk(outside).indices
                targets = torch.zeros(outside, dtype=torch.long, device=branch.device)
            else:
                continue
            instances = embeddings.index_select(0, chosen)
            total = total + self._instance_criterion(head(instances).float(), targets)
        if self.subtyping:
            total = total / self.instance_classes
        return total

    def training_output(self, features, mask, labels, criterion):
        """Slide logits plus each slide's weighted instance-clustering loss."""
        logits, weights, _, embeddings, mask = self._forward(features, mask)
        instance = torch.stack(
            [
                self._slide_instance_loss(
                    _valid_rows(weights[index].transpose(0, 1), mask[index]).transpose(0, 1),
                    _valid_rows(embeddings[index], mask[index]),
                    int(labels[index]),
                )
                for index in range(len(logits))
            ]
        )
        return {
            "logits": logits,
            "auxiliaryLoss": (1 - self.bag_weight) * instance,
            "bagLossWeight": self.bag_weight,
        }
