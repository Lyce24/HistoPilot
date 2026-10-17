"""Golden outputs of the authors' MIL code, rebuilt without that code.

``scripts/mil_parity_fixtures.py`` ran each official implementation once, with weights
and inputs drawn from NumPy's legacy, stream-stable ``RandomState``, and wrote its outputs
to ``tests/fixtures/mil_parity.json``. Each case lists every official tensor with its
shape and the name of its HistoPilot counterpart, so these helpers can rebuild the same
tensors for HistoPilot's modules and compare the outputs.
"""

import json
from pathlib import Path

import numpy as np

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "mil_parity.json"


def tensor_values(name, shape, seed, index):
    """Well-scaled values for one official tensor: fan-in scaled weights, small biases and
    LayerNorm scales near one, so every case has finite, non-degenerate outputs."""
    values = np.random.RandomState(seed * 1000 + index).standard_normal(shape)
    if len(shape) >= 2:
        values = values / np.sqrt(np.prod(shape[1:]))
    elif "norm" in name and name.endswith("weight"):
        values = 1 + 0.1 * values
    else:
        values = 0.1 * values
    return values.astype(np.float32)


def features(case):
    """The case's patch features, one slide of ``patches`` × ``featureDim``."""
    return (
        np.random.RandomState(case["inputSeed"])
        .standard_normal((case["patches"], case["featureDim"]))
        .astype(np.float32)
    )


def svm_inputs(case):
    rng = np.random.RandomState(case["inputSeed"])
    scales = np.repeat(
        np.asarray(case["scales"], dtype=np.float64), case["rows"] // len(case["scales"])
    )
    logits = rng.standard_normal((len(scales), 2)) * scales[:, None]
    targets = rng.randint(0, 2, len(scales))
    return logits.astype(np.float32), targets


def load():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))
