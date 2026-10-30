"""Write tests/fixtures/oceanpath_parity.json: golden outputs of OceanPath's Mamba-2 MIL.

The fixture lets tests/test_oceanpath_parity.py check HistoPilot's plain-PyTorch port
(histopilot/models/mamba2.py) against OceanPath's own model without OceanPath, mamba-ssm
or a GPU. This script builds OceanPath's classifier (``build_classifier("mamba2mil")``),
fills it with weights from tests/support/oceanpath_parity.py, runs it on the fused
mamba-ssm kernels and records slide embeddings, logits, attention, a training loss and
every gradient norm, with the name, shape and HistoPilot counterpart of every tensor. It
also records the random initialization OceanPath draws from a fixed seed, and, given
``--checkpoint``, the outputs of one real pretrained aggregator on a synthetic bag.

Run it with OceanPath's environment (mamba-ssm and causal-conv1d built for CUDA), in one
process on a free GPU:

    OCEANPATH/.venv/bin/python scripts/oceanpath_parity_fixtures.py --oceanpath OCEANPATH \\
        --checkpoint /path/to/aggregator_weights_lejepa.pt            # write
    ... the same arguments ... --check                               # compare, exit 1 if off

On CUDA, matrix products run in IEEE float32: TF32 is off in PyTorch, and the Triton
kernels are compiled into a private cache with TRITON_F32_DEFAULT=ieee. The fixture
therefore holds float32 results of OceanPath's algorithm rather than of a reduced-precision
mode. The GPU kernels are not bitwise deterministic, so --check compares with a tolerance.

``--device cpu`` runs OceanPath's model with mamba-ssm's own reference implementations
(``causal_conv1d_ref``, the sequential ``selective_scan_ref`` and ``rms_norm_ref``, which
mamba-ssm's tests hold its kernels to) in place of the fused functions. The fixture names
the operations it was recorded with.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "tests" / "fixtures" / "oceanpath_parity.json"
SOURCE_FILES = (
    "src/oceanpath/models/__init__.py",
    "src/oceanpath/models/base.py",
    "src/oceanpath/models/components.py",
    "src/oceanpath/models/mambamil.py",
    "src/oceanpath/models/wsi_classifier.py",
)
CONFIGS = {
    "small": {
        "embed_dim": 32,
        "num_layers": 2,
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
        "headdim": 16,
        "attn_dim": 16,
    },
    "narrow": {
        "embed_dim": 24,
        "num_layers": 1,
        "d_state": 8,
        "d_conv": 2,
        "expand": 1,
        "headdim": 8,
        "attn_dim": 8,
    },
    # configs/model/mamba2mil.yaml, the shape of the pretrained aggregators.
    "oceanpath": {
        "embed_dim": 384,
        "num_layers": 2,
        "d_state": 64,
        "d_conv": 4,
        "expand": 2,
        "headdim": 64,
        "attn_dim": 256,
    },
}
# id, configuration, feature dimension, classes, bag lengths, coordinates (Hilbert order),
# COBRA pooling at evaluation, training labels (None: no training step).
CASES = (
    ("small-chunk-boundary", "small", 24, 2, [300], False, False, [1]),
    ("small-hilbert", "small", 24, 3, [600], True, False, [2]),
    ("small-padded-batch-hilbert", "small", 24, 3, [520, 257, 3], True, False, [0, 2, 1]),
    ("small-cobra-pooling", "small", 24, 2, [280], True, True, None),
    ("narrow-padded-batch", "narrow", 24, 2, [513, 40], False, False, [0, 1]),
    ("oceanpath-shape", "oceanpath", 1536, 2, [700], True, False, [1]),
)
INIT_SEED = 7
REAL_BAG = {"inputSeed": 20261029, "featureDim": 1536, "lengths": [2048]}


def floats(tensor):
    """Shortest decimal of each float32 value, so the fixture stays small and exact."""
    import numpy as np

    return [float(str(np.float32(value))) for value in tensor.detach().float().cpu().reshape(-1)]


def mapped(name):
    """HistoPilot's name for an OceanPath WSIClassifier tensor."""
    return "classifier." + name[len("head.") :] if name.startswith("head.") else name


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def use_reference_operations():
    """Run mamba-ssm's Mamba2 through its reference functions instead of CUDA and Triton."""
    from causal_conv1d.causal_conv1d_interface import causal_conv1d_ref
    from mamba_ssm.modules import mamba2
    from mamba_ssm.ops import selective_scan_interface
    from mamba_ssm.ops.triton import ssd_combined
    from mamba_ssm.ops.triton.layernorm_gated import rms_norm_ref

    def rmsnorm(x, weight, bias, z=None, eps=1e-6, group_size=None, norm_before_gate=True):
        return rms_norm_ref(x, weight, bias, z, eps, group_size, norm_before_gate, upcast=True)

    def split_conv1d_scan(*args, seq_idx=None, **kwargs):
        if seq_idx is not None:
            raise ValueError("The reference path has no packed sequences.")
        return ssd_combined.mamba_split_conv1d_scan_ref(*args, **kwargs)

    ssd_combined.causal_conv1d_fn = causal_conv1d_ref
    ssd_combined.rmsnorm_fn = rmsnorm
    selective_scan_interface.selective_scan_fn = selective_scan_interface.selective_scan_ref
    mamba2.mamba_split_conv1d_scan_combined = split_conv1d_scan


def record(oceanpath, checkpoint, device_name):
    import numpy as np
    import torch
    from torch.nn import functional as F

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device_name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("OceanPath's fused Mamba-2 kernels need a CUDA GPU (or use --device cpu).")
    import causal_conv1d
    import mamba_ssm
    import triton
    from oceanpath.models import build_aggregator, build_classifier
    from support.oceanpath_parity import coordinates, features, mask, tensor_values

    if device_name == "cpu":
        use_reference_operations()
    device = torch.device(device_name)
    commit = subprocess.run(
        ["git", "-C", str(oceanpath), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    fixture = {
        "version": 1,
        "generator": "scripts/oceanpath_parity_fixtures.py",
        "sources": {
            "OceanPath": f"https://github.com/Lyce24/OceanPath {commit[:7]}",
            "files": {name: digest(oceanpath / name) for name in SOURCE_FILES},
            "mamba-ssm": mamba_ssm.__version__,
            "causal-conv1d": causal_conv1d.__version__,
            "triton": triton.__version__,
            "torch": torch.__version__.split("+")[0],
        },
        "operations": (
            "fused mamba-ssm CUDA and Triton kernels; float32 with TF32 off and "
            "TRITON_F32_DEFAULT=ieee"
            if device_name == "cuda"
            else "mamba-ssm reference functions on CPU (causal_conv1d_ref, selective_scan_ref, "
            "rms_norm_ref); float32"
        ),
        "configurations": CONFIGS,
        "cases": [],
    }

    def inputs(case):
        values = torch.from_numpy(features(case)).to(device)
        valid = torch.from_numpy(mask(case).astype(np.float32)).to(device)
        coords = torch.from_numpy(coordinates(case)).to(device) if case["coordinates"] else None
        return values, valid, coords

    for index, (name, config, dim, classes, lengths, hilbert, cobra, labels) in enumerate(CASES):
        case = {
            "id": name,
            "configuration": config,
            "featureDim": dim,
            "outputs": classes,
            "lengths": lengths,
            "coordinates": hilbert,
            "cobraPooling": cobra,
            "inputSeed": 300 + index,
            "weightSeed": 30 + index,
        }
        model = build_classifier(
            "mamba2mil",
            in_dim=dim,
            num_classes=classes,
            model_cfg={
                **CONFIGS[config],
                "dropout": 0.0,
                "use_hilbert": hilbert,
                "use_cobra_inference": cobra,
            },
        )
        state = model.state_dict()
        case["tensors"] = []
        for position, (key, value) in enumerate(state.items()):
            shape = list(value.shape)
            case["tensors"].append([key, mapped(key), shape])
            state[key] = torch.from_numpy(tensor_values(key, shape, case["weightSeed"], position))
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        values, valid, coords = inputs(case)
        with torch.no_grad():
            output = model(values, mask=valid, coords=coords, return_attention=True)
        weights = torch.softmax(output.extras["attention_weights"].float(), dim=-1)
        case["eval"] = {
            "embedding": [floats(row) for row in output.slide_embedding],
            "logits": [floats(row) for row in output.logits],
            "attention": [floats(weights[slide, :n]) for slide, n in enumerate(lengths)],
        }
        if labels is not None:
            case["labels"] = labels
            model.train()
            model.zero_grad()
            output = model(values, mask=valid, coords=coords)
            loss = F.cross_entropy(output.logits.float(), torch.tensor(labels, device=device))
            loss.backward()
            case["fit"] = {
                "loss": floats(loss)[0],
                "gradNorms": {
                    mapped(key): float(parameter.grad.double().norm())
                    for key, parameter in model.named_parameters()
                    if parameter.grad is not None
                },
            }
        fixture["cases"].append(case)

    # OceanPath's random initialization from one seed, built on the CPU as it trains.
    fixture["initialization"] = []
    for config in ("small", "oceanpath"):
        dim = 24 if config == "small" else 1536
        torch.manual_seed(INIT_SEED)
        model = build_classifier(
            "mamba2mil", in_dim=dim, num_classes=2, model_cfg={**CONFIGS[config], "dropout": 0.1}
        )
        fixture["initialization"].append(
            {
                "configuration": config,
                "featureDim": dim,
                "outputs": 2,
                "seed": INIT_SEED,
                "norms": {
                    mapped(key): float(value.double().norm())
                    for key, value in model.state_dict().items()
                },
            }
        )

    if checkpoint is not None:
        aggregator = build_aggregator("mamba2mil", 1536, model_cfg=CONFIGS["oceanpath"])
        aggregator.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
        )
        aggregator.to(device).eval()
        bag = {**REAL_BAG, "coordinates": True}
        values, valid, coords = inputs(bag)
        with torch.no_grad():
            output = aggregator(values, mask=valid, coords=coords, return_attention=True)
            aggregator.use_cobra_inference = True
            cobra = aggregator(values, mask=valid, coords=coords)
        fixture["checkpoint"] = {
            **REAL_BAG,
            "file": Path(checkpoint).name,
            "sha256": digest(checkpoint),
            "bytes": Path(checkpoint).stat().st_size,
            "configuration": "oceanpath",
            "embedding": floats(output.slide_embedding[0]),
            "cobraEmbedding": floats(cobra.slide_embedding[0]),
            "attention": floats(torch.softmax(output.extras["attention_weights"][0], -1)),
        }
    return fixture


def differences(expected, actual, path="", tolerance=1e-5):
    """Paths whose structure differs, or whose numbers differ beyond the tolerance."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            return [path or "/"]
        return [
            item
            for key in expected
            for item in differences(expected[key], actual[key], f"{path}/{key}", tolerance)
        ]
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [path]
        return [
            item
            for index, (left, right) in enumerate(zip(expected, actual))
            for item in differences(left, right, f"{path}/{index}", tolerance)
        ]
    if isinstance(expected, float) and isinstance(actual, float):
        return [] if abs(expected - actual) <= tolerance * max(1.0, abs(expected)) else [path]
    return [] if expected == actual else [path]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--oceanpath", required=True, type=Path, help="OceanPath checkout")
    parser.add_argument("--checkpoint", type=Path, help="a pretrained aggregator state dict")
    parser.add_argument("--check", action="store_true", help="exit 1 if the fixture differs")
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="cuda: the fused kernels (the golden record); cpu: mamba-ssm's reference functions",
    )
    args = parser.parse_args()
    oceanpath = args.oceanpath.resolve()
    sys.path[:0] = [str(oceanpath / "src"), str(ROOT / "tests")]
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    with tempfile.TemporaryDirectory() as cache:
        os.environ["TRITON_F32_DEFAULT"] = "ieee"
        os.environ["TRITON_CACHE_DIR"] = cache
        fixture = record(oceanpath, args.checkpoint, args.device)
    rendered = json.dumps(fixture, indent=1, sort_keys=True) + "\n"
    if args.check:
        current = json.loads(TARGET.read_text(encoding="utf-8")) if TARGET.exists() else {}
        fresh = json.loads(rendered)
        # A later OceanPath commit is fine as long as the model's own files are unchanged.
        for document in (current, fresh):
            document.get("sources", {}).pop("OceanPath", None)
        found = differences(current, fresh)
        for path in found[:20]:
            print(f"differs: {path}")
        if found:
            print(f"{TARGET.relative_to(ROOT)} is out of date ({len(found)} differences).")
            return 1
        return 0
    TARGET.write_text(rendered, encoding="utf-8")
    print(
        f"Wrote {TARGET.relative_to(ROOT)} ({len(fixture['cases'])} cases, "
        f"{len(rendered) // 1024} KiB)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
