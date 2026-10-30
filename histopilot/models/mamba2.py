"""OceanPath's Mamba-2 MIL aggregator in plain PyTorch, with a classifier head.

Architecture: OceanPath ``oceanpath.models.mambamil.Mamba2MIL`` (MIT licence), a
COBRA-style slide encoder (Lenz et al., 2025) built on Mamba-2 state space duality (Dao
and Gu, ICML 2024): patch embedding (Linear, LayerNorm), Mamba-2 blocks with a residual
around each, a Linear and LayerNorm, and gated attention pooling. OceanPath runs the
blocks with ``mamba_ssm.modules.mamba2.Mamba2`` and its fused CUDA/Triton kernels; this
module computes the same block with tensor operations only, on CPU or GPU:

* ``in_proj`` splits into the gate z, the convolved stream xBC and the step dt;
* a depthwise causal convolution and SiLU over xBC, which splits into x, B and C;
* the selective scan as the chunked SSD algorithm: within each chunk a masked
  "attention" C·B with exp(cumulative dA) decay, and between chunks a recurrence over
  chunk states, with the D skip, exactly the decomposition the kernels compute;
* RMSNorm of y·SiLU(z) (``norm_before_gate=False``) and ``out_proj``.

Parameter names and shapes are those of OceanPath's aggregator, so its state dicts,
including the self-supervised checkpoints it pretrains, load into ``aggregator`` with
``strict=True``. Parameters are created in OceanPath's order with the same initializers,
so a seed gives the same random initialization as OceanPath's ``build_classifier``.
``tests/test_oceanpath_parity.py`` checks outputs and gradients against golden outputs
of OceanPath's own model, recorded by ``scripts/oceanpath_parity_fixtures.py``.

Padding: OceanPath masks padding only in attention pooling; the scan is causal and its
batches put padding last, so padding never changes a real patch. This module keeps that
contract for any mask by moving each slide's valid patches first, in order, before the
blocks. With coordinates (and Hilbert ordering on) patches are first sorted along a
Hilbert curve as OceanPath does; attention is returned in input order either way.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from histopilot.models.abmil import prepare_bags

HILBERT_ORDER = 16
# Positions per scan chunk. The chunking is exact algebra, so it changes rounding only;
# mamba-ssm's kernels use 256, and 64 halves this port's step time and memory on a CPU.
CHUNK_SIZE = 64


# --- Hilbert ordering (OceanPath ``models/components.py``) --------------------------------


def hilbert_index_2d(x, y, order=HILBERT_ORDER):
    """Hilbert curve index of integer coordinates in [0, 2**order - 1]."""
    x, y = x.long().clone(), y.long().clone()
    d = torch.zeros_like(x)
    s = 1 << (order - 1)
    while s > 0:
        rx = ((x & s) > 0).long()
        ry = ((y & s) > 0).long()
        d += s * s * ((3 * rx) ^ ry)
        rotate = ry == 0
        flip = rotate & (rx == 1)
        x = torch.where(flip, s - 1 - x, x)
        y = torch.where(flip, s - 1 - y, y)
        x, y = torch.where(rotate, y, x), torch.where(rotate, x, y)
        s >>= 1
    return d


def normalize_coords_to_grid(coords, mask=None, order=HILBERT_ORDER):
    """Per-slide min-max scaling of coordinates onto the curve's grid, aspect kept.

    Computed in float32 as OceanPath does, so every patch lands on the same grid cell.
    """
    grid_max = (1 << order) - 1
    x, y = coords[..., 0].float(), coords[..., 1].float()
    if mask is not None:
        invalid = ~mask.bool()
        x_min = x.masked_fill(invalid, float("inf")).min(dim=1, keepdim=True).values
        y_min = y.masked_fill(invalid, float("inf")).min(dim=1, keepdim=True).values
        x_max = x.masked_fill(invalid, float("-inf")).max(dim=1, keepdim=True).values
        y_max = y.masked_fill(invalid, float("-inf")).max(dim=1, keepdim=True).values
    else:
        x_min, y_min = x.min(dim=1, keepdim=True).values, y.min(dim=1, keepdim=True).values
        x_max, y_max = x.max(dim=1, keepdim=True).values, y.max(dim=1, keepdim=True).values
    span = torch.max((x_max - x_min).clamp(min=1.0), (y_max - y_min).clamp(min=1.0))
    x_norm = ((x - x_min) / span * grid_max).long().clamp(0, grid_max)
    y_norm = ((y - y_min) / span * grid_max).long().clamp(0, grid_max)
    if mask is not None:
        x_norm, y_norm = x_norm.masked_fill(invalid, 0), y_norm.masked_fill(invalid, 0)
    return x_norm, y_norm


def hilbert_permutation(coords, mask=None, order=HILBERT_ORDER):
    """The stable order of each slide's patches along the curve, padding last."""
    index = hilbert_index_2d(*normalize_coords_to_grid(coords, mask, order), order)
    if mask is not None:
        index = index.masked_fill(~mask.bool(), 1 << (2 * order))
    return index.argsort(dim=1, stable=True)


def valid_first_permutation(mask):
    """Each slide's valid patches first in bag order, then its padding; None if already so."""
    if bool((mask[:, 1:] <= mask[:, :-1]).all()):
        return None
    return (~mask).to(torch.int8).argsort(dim=1, stable=True)


# --- Mamba-2 ---------------------------------------------------------------------------


def compute_dtype(tensor):
    """Float32 for half precision and float32 inputs; float64 stays float64."""
    return torch.promote_types(tensor.dtype, torch.float32)


def running_sum(x):
    """Inclusive cumulative sum over the last dimension by log-step doubling.

    ``torch.cumsum`` has no deterministic CUDA kernel, so deterministic training would
    refuse it on a GPU; these shifted additions are deterministic on every device.
    """
    length, shift = x.shape[-1], 1
    while shift < length:
        x = x + F.pad(x[..., : length - shift], (shift, 0))
        shift *= 2
    return x


def ssd_scan(x, dt, A, B, C, chunk_size=CHUNK_SIZE):
    """Mamba-2's selective scan, h_t = exp(dt_t A) h_(t-1) + dt_t B_t x_t and y_t = C_t h_t,
    by the chunked state space duality algorithm of ``mamba_ssm``'s kernels.

    x: [batch, length, heads, head dim]; dt: [batch, length, heads], after softplus;
    A: [heads]; B, C: [batch, length, groups, state]. Returns y without the D skip.
    Computed in float32, or float64 for float64 inputs. Positions past the end are padded
    with dt = 0, which neither decays nor adds to the state.
    """
    batch, length, heads, head_dim = x.shape
    groups, state = B.shape[2], B.shape[3]
    dtype = compute_dtype(x)
    x, dt, A, B, C = (value.to(dtype) for value in (x, dt, A, B, C))
    pad = (-length) % chunk_size
    if pad:
        x, B, C = (F.pad(value, (0, 0, 0, 0, 0, pad)) for value in (x, B, C))
        dt = F.pad(dt, (0, 0, 0, pad))
    chunks = (length + pad) // chunk_size
    x = x.reshape(batch, chunks, chunk_size, heads, head_dim)
    dt = dt.reshape(batch, chunks, chunk_size, heads)
    B = B.reshape(batch, chunks, chunk_size, groups, state)
    C = C.reshape(batch, chunks, chunk_size, groups, state)
    # Heads share their group's B and C: [batch, chunks, position, group, heads per group].
    per_group = heads // groups
    cumulative = running_sum((dt * A).transpose(2, 3)).transpose(2, 3)  # [b, c, position, h]
    # Within chunks: y_l += sum over s <= l of (C_l . B_s) exp(dA_l - dA_s) dt_s x_s.
    scores = torch.einsum("bclgn,bcsgn->bcgls", C, B)
    decay = cumulative.permute(0, 1, 3, 2)  # [batch, chunks, heads, position]
    decay = torch.exp(torch.clamp(decay[..., :, None] - decay[..., None, :], max=0))
    causal = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=x.device).tril()
    weights = (
        decay.masked_fill(~causal, 0).reshape(
            batch, chunks, groups, per_group, chunk_size, chunk_size
        )
        * scores[:, :, :, None]
        * dt.permute(0, 1, 3, 2).reshape(batch, chunks, groups, per_group, 1, chunk_size)
    )
    within = torch.einsum(
        "bcgjls,bcsgjp->bclgjp",
        weights,
        x.reshape(batch, chunks, chunk_size, groups, per_group, head_dim),
    )
    # Each chunk's final state from its own positions.
    to_end = torch.exp(torch.clamp(cumulative[:, :, -1:] - cumulative, max=0)) * dt
    states = torch.einsum(
        "bcsgn,bcsgj,bcsgjp->bcgjpn",
        B,
        to_end.reshape(batch, chunks, chunk_size, groups, per_group),
        x.reshape(batch, chunks, chunk_size, groups, per_group, head_dim),
    )
    # Between chunks: the state entering each chunk, carried with its total decay.
    carry = torch.exp(cumulative[:, :, -1]).reshape(batch, chunks, groups, per_group, 1, 1)
    entering, running = [], torch.zeros_like(states[:, 0])
    for index in range(chunks):
        entering.append(running)
        running = carry[:, index] * running + states[:, index]
    entering = torch.stack(entering, dim=1)  # [batch, chunks, groups, per group, p, n]
    between = torch.einsum("bclgn,bcgjpn->bclgjp", C, entering) * torch.exp(cumulative).reshape(
        batch, chunks, chunk_size, groups, per_group, 1
    )
    y = (within + between).reshape(batch, chunks * chunk_size, heads, head_dim)
    return y[:, :length]


class GatedRMSNorm(nn.Module):
    """RMSNorm of x * SiLU(z) over groups of channels, at least in float32
    (``norm_before_gate=False``)."""

    def __init__(self, dimensions, eps=1e-5, group_size=None):
        super().__init__()
        self.eps = eps
        self.group_size = group_size or dimensions
        self.weight = nn.Parameter(torch.ones(dimensions))

    def forward(self, x, z):
        dtype, compute = x.dtype, compute_dtype(x)
        x = x.to(compute) * F.silu(z.to(compute))
        grouped = x.reshape(*x.shape[:-1], -1, self.group_size)
        grouped = grouped * torch.rsqrt(grouped.square().mean(dim=-1, keepdim=True) + self.eps)
        return (grouped.reshape(x.shape) * self.weight.to(compute)).to(dtype)


class Mamba2(nn.Module):
    """``mamba_ssm`` Mamba-2 block with OceanPath's arguments (one group, no biases on the
    projections, gated RMSNorm, dt limit off), its parameter names and initializers."""

    def __init__(
        self,
        d_model,
        d_state=64,
        d_conv=4,
        expand=2,
        headdim=64,
        ngroups=1,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        chunk_size=CHUNK_SIZE,
    ):
        super().__init__()
        self.d_model, self.d_state, self.d_conv = d_model, d_state, d_conv
        self.d_inner = expand * d_model
        if self.d_inner % headdim or ngroups < 1 or (self.d_inner // headdim) % ngroups:
            raise ValueError("Mamba-2 head dimensions must divide expand × embedding dimensions.")
        self.headdim, self.ngroups, self.chunk_size = headdim, ngroups, chunk_size
        self.nheads = self.d_inner // headdim
        projected = 2 * self.d_inner + 2 * ngroups * d_state + self.nheads
        self.in_proj = nn.Linear(d_model, projected, bias=False)
        conv_dim = self.d_inner + 2 * ngroups * d_state
        self.conv1d = nn.Conv1d(
            conv_dim, conv_dim, kernel_size=d_conv, groups=conv_dim, padding=d_conv - 1, bias=True
        )
        # dt from a log-uniform range, stored as softplus^-1 so softplus(dt_bias) = dt.
        dt = torch.exp(
            torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        A = torch.empty(self.nheads, dtype=torch.float32).uniform_(*A_init_range)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.nheads))
        for parameter in (self.dt_bias, self.A_log, self.D):
            parameter._no_weight_decay = True
        self.norm = GatedRMSNorm(self.d_inner, eps=1e-5, group_size=self.d_inner // ngroups)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, u):
        batch, length, _ = u.shape
        z, xBC, dt = torch.split(
            self.in_proj(u),
            [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1,
        )
        # Depthwise causal convolution: left padding only, the first `length` outputs.
        xBC = F.silu(self.conv1d(xBC.transpose(1, 2))[..., :length].transpose(1, 2))
        x, B, C = torch.split(
            xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1
        )
        with torch.autocast(device_type=u.device.type, enabled=False):
            dtype = compute_dtype(x)
            x = x.to(dtype).reshape(batch, length, self.nheads, self.headdim)
            step = F.softplus(dt.to(dtype) + self.dt_bias.to(dtype))
            y = ssd_scan(
                x,
                step,
                -torch.exp(self.A_log.to(dtype)),
                B.to(dtype).reshape(batch, length, self.ngroups, self.d_state),
                C.to(dtype).reshape(batch, length, self.ngroups, self.d_state),
                self.chunk_size,
            )
            y = y + x * self.D.to(dtype)[:, None]
            y = self.norm(y.reshape(batch, length, self.d_inner), z.to(dtype))
        return self.out_proj(y)


class Mamba2Layer(nn.Module):
    """One Mamba-2 block with a residual around it, as COBRA's Eq. 3 (``blocks.N.mamba``)."""

    def __init__(self, d_model, d_state, d_conv, expand, headdim, chunk_size=CHUNK_SIZE):
        super().__init__()
        self.mamba = Mamba2(d_model, d_state, d_conv, expand, headdim, chunk_size=chunk_size)

    def forward(self, x):
        return x + self.mamba(x)


class GlobalGatedAttention(nn.Module):
    """Gated attention scores: tanh and sigmoid branches multiplied, then projected."""

    def __init__(self, L, D, dropout=0.0, num_classes=1):
        super().__init__()
        self.attention_a = nn.Sequential(nn.Linear(L, D), nn.Tanh(), nn.Dropout(dropout))
        self.attention_b = nn.Sequential(nn.Linear(L, D), nn.Sigmoid(), nn.Dropout(dropout))
        self.attention_c = nn.Linear(D, num_classes)

    def forward(self, x):
        return self.attention_c(self.attention_a(x) * self.attention_b(x))


class Mamba2MILAggregator(nn.Module):
    """OceanPath's ``Mamba2MIL``: its parameters, names and forward pass."""

    def __init__(
        self,
        in_dim,
        embed_dim=384,
        num_layers=2,
        d_state=64,
        d_conv=4,
        expand=2,
        headdim=64,
        attn_dim=256,
        dropout=0.1,
        use_cobra_inference=False,
        gradient_checkpointing=False,
        use_hilbert=True,
        hilbert_order=HILBERT_ORDER,
        chunk_size=CHUNK_SIZE,
    ):
        super().__init__()
        self.in_dim, self.embed_dim = in_dim, embed_dim
        self.use_cobra_inference = use_cobra_inference
        self.gradient_checkpointing = gradient_checkpointing
        self.use_hilbert, self.hilbert_order = use_hilbert, hilbert_order
        self.patch_embed = nn.Sequential(nn.Linear(in_dim, embed_dim), nn.LayerNorm(embed_dim))
        self.blocks = nn.ModuleList(
            Mamba2Layer(embed_dim, d_state, d_conv, expand, headdim, chunk_size)
            for _ in range(num_layers)
        )
        self.post_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.global_attn = GlobalGatedAttention(embed_dim, attn_dim, dropout, num_classes=1)

    def forward(self, features, mask, coords=None):
        """Slide embeddings [B, E], and attention logits [B, N] in input order (padding -inf).

        ``mask`` is boolean and nonempty per slide. Without coordinates (or with Hilbert
        ordering off), patches keep bag order with any padding moved last.
        """
        embedded = self.patch_embed(features)
        order = (
            hilbert_permutation(coords, mask, self.hilbert_order)
            if self.use_hilbert and coords is not None
            else valid_first_permutation(mask)
        )
        if order is not None:
            embedded = embedded.gather(1, order[..., None].expand_as(embedded))
            mask = mask.gather(1, order)
        encoded = embedded
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                encoded = checkpoint(block, encoded, use_reentrant=False)
            else:
                encoded = block(encoded)
        encoded = self.norm(self.post_proj(encoded))
        logits = self.global_attn(encoded).squeeze(-1).masked_fill(~mask, float("-inf"))
        with torch.autocast(device_type=features.device.type, enabled=False):
            dtype = compute_dtype(logits)
            weights = torch.softmax(logits.to(dtype), dim=-1)
            pooled = embedded if self.use_cobra_inference and not self.training else encoded
            embedding = torch.bmm(weights.unsqueeze(1), pooled.to(dtype)).squeeze(1)
        if order is not None:
            logits = torch.empty_like(logits).scatter_(1, order, logits)
        return embedding, logits


class OceanPathMamba2MIL(nn.Module):
    """OceanPath's Mamba-2 MIL aggregator and a linear classifier (its ``WSIClassifier``).

    ``aggregator`` holds exactly OceanPath's aggregator parameters; ``classifier`` is the
    head that the recipe's learning rate trains. ``aggregator_learning_rate`` (set by the
    registry for checkpoint initialisation) gives the aggregator its own rate.
    """

    def __init__(
        self,
        in_dim,
        num_classes,
        *,
        embed_dim=384,
        num_layers=2,
        d_state=64,
        d_conv=4,
        expand=2,
        headdim=64,
        attention_dim=256,
        dropout=0.1,
        hilbert=True,
        cobra_pooling=False,
        input_dropout=0.0,
        gradient_checkpointing=False,
        chunk_size=CHUNK_SIZE,
    ):
        super().__init__()
        if min(in_dim, num_classes, embed_dim, num_layers, d_state, d_conv, expand) < 1:
            raise ValueError("Mamba-2 MIL requires positive dimensions, layers and logits.")
        if min(headdim, attention_dim, chunk_size) < 1:
            raise ValueError("Mamba-2 MIL requires positive dimensions, layers and logits.")
        if not 0 <= dropout < 1 or not 0 <= input_dropout < 1:
            raise ValueError("Dropout probabilities must be in [0, 1).")
        self.in_dim, self.num_classes = in_dim, num_classes
        self.reads_coordinates = hilbert
        self.aggregator_learning_rate = None
        self.input_dropout = nn.Dropout(input_dropout)
        self.aggregator = Mamba2MILAggregator(
            in_dim,
            embed_dim=embed_dim,
            num_layers=num_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            attn_dim=attention_dim,
            dropout=dropout,
            use_cobra_inference=cobra_pooling,
            gradient_checkpointing=gradient_checkpointing,
            use_hilbert=hilbert,
            chunk_size=chunk_size,
        )
        # OceanPath's WSIClassifier builds its head, then resets it once more.
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.classifier.reset_parameters()

    def load_pretrained(self, state):
        """An OceanPath aggregator state dict, strictly: every name and shape must match."""
        if not isinstance(state, dict) or not all(
            isinstance(value, torch.Tensor) for value in state.values()
        ):
            raise ValueError("expected a state dict of tensors")
        self.aggregator.load_state_dict(state, strict=True)

    def freeze_aggregator(self):
        """Fix the aggregator's weights, as OceanPath's ``freeze_aggregator``; the head trains."""
        for parameter in self.aggregator.parameters():
            parameter.requires_grad_(False)

    def forward(self, features, mask=None, *, coords=None, return_attention=False):
        features, mask = prepare_bags(features, mask, self.in_dim)
        if coords is not None:
            if coords.shape != (*features.shape[:2], 2):
                raise ValueError("Coordinates must be one XY pair per patch.")
            coords = coords.to(features.device)
        embedding, logits = self.aggregator(self.input_dropout(features), mask, coords)
        output = self.classifier(embedding)
        if return_attention:
            with torch.autocast(device_type=features.device.type, enabled=False):
                weights = torch.softmax(logits.float(), dim=-1)
            return {"logits": output, "embedding": embedding, "attention": weights}
        return output
