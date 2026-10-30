"""Golden outputs of OceanPath's Mamba-2 MIL, rebuilt without OceanPath or mamba-ssm.

``scripts/oceanpath_parity_fixtures.py`` ran OceanPath's model on its fused CUDA kernels,
with weights and inputs from these functions, and wrote its outputs to
``tests/fixtures/oceanpath_parity.json``. NumPy's legacy ``RandomState`` keeps the draws
identical on any platform. This module must stay importable by OceanPath's Python 3.10.
"""

import json
from pathlib import Path

import numpy as np

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "oceanpath_parity.json"
# Environment variable naming a local copy of the recorded real checkpoint; never a default.
CHECKPOINT_VARIABLE = "HISTOPILOT_OCEANPATH_CHECKPOINT"


def tensor_values(name, shape, seed, index):
    """Values for one OceanPath tensor, scaled like a trained model.

    Mamba-2's own parameters follow its initializers (A from 1 to 16, dt from 0.001 to 0.1
    through softplus, D near one), so slow-decaying heads carry state across chunks; other
    weights are fan-in scaled, biases small and norm scales near one.
    """
    rng = np.random.RandomState(seed * 1000 + index)
    leaf = name.rsplit(".", 1)[-1]
    if leaf == "A_log":
        values = np.log(rng.uniform(1, 16, shape))
    elif leaf == "dt_bias":
        dt = np.exp(rng.uniform(np.log(1e-3), np.log(1e-1), shape))
        values = dt + np.log(-np.expm1(-dt))
    elif leaf == "D":
        values = 1 + 0.1 * rng.standard_normal(shape)
    elif len(shape) >= 2:
        values = rng.standard_normal(shape) / np.sqrt(np.prod(shape[1:]))
    elif "norm" in name and leaf == "weight":
        values = 1 + 0.1 * rng.standard_normal(shape)
    else:
        values = 0.1 * rng.standard_normal(shape)
    return values.astype(np.float32)


def features(case):
    """[slides, longest bag, feature dim] features, zero after each slide's length."""
    lengths = case["lengths"]
    rng = np.random.RandomState(case["inputSeed"])
    values = np.zeros((len(lengths), max(lengths), case["featureDim"]), dtype=np.float32)
    for slide, length in enumerate(lengths):
        values[slide, :length] = rng.standard_normal((length, case["featureDim"]))
    return values


def mask(case):
    lengths = case["lengths"]
    return np.arange(max(lengths))[None, :] < np.asarray(lengths)[:, None]


def coordinates(case):
    """Distinct cells of a 256-pixel patch grid per slide, in shuffled order; zero padding."""
    lengths = case["lengths"]
    rng = np.random.RandomState(case["inputSeed"] + 1)
    values = np.zeros((len(lengths), max(lengths), 2), dtype=np.int64)
    for slide, length in enumerate(lengths):
        side = int(np.ceil(np.sqrt(length * 2)))
        cells = rng.choice(side * side, size=length, replace=False)
        values[slide, :length, 0] = (cells % side) * 256 + 1024
        values[slide, :length, 1] = (cells // side) * 256 + 512
    return values


def load():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))
