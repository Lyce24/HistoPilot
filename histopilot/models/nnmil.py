"""Native nnMIL feature-subspace attention with full-dimensional pooling.

Method: Luo et al., arXiv:2511.14907. The architecture and window rule were
reviewed against Luoxd1996/nnMIL commit e92e8a0 (2026-09-06); this module has
no upstream runtime dependency. HistoPilot adds masked padding, stable
accumulation, persistent feature order and one-logit classification support.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from histopilot.models.abmil import prepare_bags


class NNMIL(nn.Module):
    """Sample attention input coordinates; pool the original feature vectors.

    Training samples one feature subset per forward call, shared across its
    minibatch as in the reference implementation. Evaluation visits fixed,
    overlapping feature windows sequentially without retaining K bag-sized
    attention tensors. ``feature_sampling=False`` keeps the gated architecture
    and uses a single full-dimensional view for both fitting and evaluation.
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        *,
        attention_dim: int = 256,
        dropout: float = 0.25,
        input_dropout: float = 0.0,
        gradient_checkpointing: bool = False,
        feature_sampling: bool = True,
        window_stride_divisor: int = 4,
        window_shuffle: bool = True,
        window_seed: int = 42,
        window_aggregation: str = "mean_logits",
    ):
        super().__init__()
        if any(
            type(value) is not int or value < 1 for value in (in_dim, num_classes, attention_dim)
        ):
            raise ValueError(
                "nnMIL requires positive integer feature, output and attention dimensions."
            )
        if not 0 <= dropout < 1 or not 0 <= input_dropout < 1:
            raise ValueError("Dropout probabilities must be in [0, 1).")
        if type(window_stride_divisor) is not int or window_stride_divisor < 1:
            raise ValueError("nnMIL window stride divisor must be a positive integer.")
        if type(window_seed) is not int or not 0 <= window_seed < 2**63:
            raise ValueError("nnMIL window seed must be an integer in [0, 2**63).")
        if window_aggregation not in {"mean_logits", "mean_probabilities"}:
            raise ValueError("nnMIL window aggregation must be mean_logits or mean_probabilities.")
        self.in_dim = in_dim
        self.attention_dim = attention_dim
        self.num_classes = num_classes
        self.feature_sampling = feature_sampling
        self.window_stride_divisor = window_stride_divisor
        self.window_aggregation = window_aggregation
        self.gradient_checkpointing = gradient_checkpointing
        self.input_dropout = nn.Dropout(input_dropout)
        self.attention_tanh = nn.Linear(in_dim, attention_dim)
        self.attention_gate = nn.Linear(in_dim, attention_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_score = nn.Linear(attention_dim, 1)
        self.classifier = nn.Linear(in_dim, num_classes)
        # Generate on CPU once, and persist the actual order. CUDA generators
        # need not reproduce CPU randperm; checkpoint loading always restores
        # the exact inference windows that were used for model selection.
        generator = torch.Generator(device="cpu").manual_seed(window_seed)
        order = (
            torch.randperm(in_dim, generator=generator) if window_shuffle else torch.arange(in_dim)
        )
        self.register_buffer("feature_order", order)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        order = state_dict.get(f"{prefix}feature_order")
        if order is not None and (
            order.dtype != torch.int64
            or order.shape != (self.in_dim,)
            or not torch.equal(torch.sort(order.detach().cpu()).values, torch.arange(self.in_dim))
        ):
            raise ValueError(
                "nnMIL checkpoint feature order must be a permutation of its dimensions."
            )
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def evaluation_windows(self):
        keep = min(self.attention_dim, self.in_dim)
        if not self.feature_sampling or keep == self.in_dim:
            return (None,)
        stride = max(1, keep // self.window_stride_divisor)
        final_start = self.in_dim - keep
        starts = list(range(0, final_start + 1, stride))
        if starts[-1] != final_start:
            starts.append(final_start)
        return tuple(self.feature_order[start : start + keep] for start in starts)

    def _attention_logits(self, features, indices):
        if indices is None:
            a = self.attention_tanh(features)
            b = self.attention_gate(features)
        else:
            selected = features.index_select(-1, indices)
            a = F.linear(
                selected,
                self.attention_tanh.weight.index_select(1, indices),
                self.attention_tanh.bias,
            )
            b = F.linear(
                selected,
                self.attention_gate.weight.index_select(1, indices),
                self.attention_gate.bias,
            )
        return self.attention_score(
            self.attention_dropout(torch.tanh(a)) * self.attention_dropout(torch.sigmoid(b))
        ).squeeze(-1)

    def _view(self, features, attention_features, mask, indices):
        attention = (
            checkpoint(self._attention_logits, attention_features, indices, use_reentrant=False)
            if self.gradient_checkpointing and self.training
            else self._attention_logits(attention_features, indices)
        )
        with torch.autocast(device_type=features.device.type, enabled=False):
            attention = attention.float().masked_fill(~mask, float("-inf"))
            weights = torch.softmax(attention, dim=1)
            pooled = torch.bmm(weights.unsqueeze(1), features.float()).squeeze(1)
        return self.classifier(pooled), pooled, weights

    def _log_probabilities(self, logits):
        logits = logits.double()
        if self.num_classes == 1:
            # Canonical order [negative, positive]. Runtime serialization maps
            # this to the user's immutable target class order.
            return torch.cat((F.logsigmoid(-logits), F.logsigmoid(logits)), dim=-1)
        return F.log_softmax(logits, dim=-1)

    def forward(self, features, mask=None, *, return_attention=False, return_uncertainty=False):
        features, mask = prepare_bags(features, mask, self.in_dim)
        # Optional input dropout regularizes attention; the pooled vectors stay
        # in the original full-dimensional feature space.
        attention_features = self.input_dropout(features)
        if self.training:
            indices = (
                torch.randperm(self.in_dim, device=features.device)[
                    : min(self.attention_dim, self.in_dim)
                ]
                if self.feature_sampling
                else None
            )
            logits, embedding, weights = self._view(features, attention_features, mask, indices)
            if return_attention or return_uncertainty:
                return {"logits": logits, "embedding": embedding, "attention": weights}
            return logits

        windows = self.evaluation_windows()
        logits_sum = log_probability_sum = entropy_sum = mean_p = probability_m2 = None
        embedding_sum = attention_sum = None
        for count, indices in enumerate(windows, start=1):
            logits, embedding, weights = self._view(features, attention_features, mask, indices)
            logits_sum = logits.double() if logits_sum is None else logits_sum + logits.double()
            if return_attention:
                embedding_sum = (
                    embedding.double()
                    if embedding_sum is None
                    else embedding_sum + embedding.double()
                )
                attention_sum = weights if attention_sum is None else attention_sum + weights
            if return_uncertainty or self.window_aggregation == "mean_probabilities":
                log_p = self._log_probabilities(logits)
                log_probability_sum = (
                    log_p
                    if log_probability_sum is None
                    else torch.logaddexp(log_probability_sum, log_p)
                )
                if return_uncertainty:
                    p = log_p.exp()
                    entropy = -(p * log_p).sum(-1)
                    entropy_sum = entropy if entropy_sum is None else entropy_sum + entropy
                    if mean_p is None:
                        mean_p = p
                        probability_m2 = torch.zeros_like(p)
                    else:
                        delta = p - mean_p
                        mean_p = mean_p + delta / count
                        probability_m2 = probability_m2 + delta * (p - mean_p)

        count = len(windows)
        if self.window_aggregation == "mean_probabilities":
            mean_log_p = log_probability_sum - math.log(count)
            # Equivalent logits keep every existing CE/BCE prediction caller
            # correct, including validation, ensembles, attention and refits.
            final_logits = (
                (mean_log_p[:, 1] - mean_log_p[:, 0]).unsqueeze(-1)
                if self.num_classes == 1
                else mean_log_p
            )
        else:
            final_logits = logits_sum / count
        if not return_attention and not return_uncertainty:
            return final_logits
        result = {"logits": final_logits}
        if return_attention:
            result.update(
                embedding=(embedding_sum / count).float(), attention=attention_sum / count
            )
        if return_uncertainty:
            mean_log_p = log_probability_sum - math.log(count)
            entropy_mean_p = -(mean_log_p.exp() * mean_log_p).sum(-1)
            mean_entropy = entropy_sum / count
            result["window_uncertainty"] = {
                "windowCount": count,
                "entropyOfMeanProbability": entropy_mean_p,
                "meanWindowEntropy": mean_entropy,
                "mutualInformation": (entropy_mean_p - mean_entropy).clamp_min(0),
                # Population variance describes the finite, deterministic set
                # of views; unlike unbiased sample variance it is defined for K=1.
                "probabilityVariance": probability_m2.clamp_min(0) / count,
            }
        return result
