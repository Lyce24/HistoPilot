"""HistoPilot's Mamba-2 MIL reproduces OceanPath's model.

tests/fixtures/oceanpath_parity.json holds outputs of OceanPath's own ``Mamba2MIL`` and
``WSIClassifier``, recorded by scripts/oceanpath_parity_fixtures.py from weights and
inputs that tests/support/oceanpath_parity.py rebuilds here; its ``operations`` field
names the Mamba-2 implementation they ran on. Every OceanPath tensor maps to one
HistoPilot tensor and loads strictly, so a pass also shows the parameters match name for
name and shape for shape.

Tolerances. The fixture holds float32 results; the port computes the same functions in
float32 on the CPU with another summation order (a chunked scan where the reference
scans sequentially, or tiles differently on the GPU). Measured differences are about 1e-6
of an output's scale and a few 1e-6 of a gradient norm, and the port in float64 differs
from the fixture by as much, so they are the fixture's own float32 rounding; the fused
kernels' softplus, log(exp(x) + 1), adds about 1e-6 more. The bounds leave room for
that: outputs within 1e-5 of max(1, |largest value|), attention within 1e-4 of the
slide's largest weight, gradient norms within 1e-4 relative. Dropping the D skip, the
state carried between chunks or the gate's place before the norm moves these cases'
outputs by 1e-4 to 1, far outside them.
"""

import hashlib
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from support import oceanpath_parity  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from histopilot.models.mamba2 import OceanPathMamba2MIL  # noqa: E402

FIXTURE = oceanpath_parity.load()
CASES = FIXTURE["cases"]
FIT_CASES = [case for case in CASES if "fit" in case]


def port(configuration, feature_dim, outputs, **options):
    """OceanPath's configuration as this port's arguments (its attn_dim is attention_dim)."""
    values = dict(FIXTURE["configurations"][configuration])
    values["attention_dim"] = values.pop("attn_dim")
    return OceanPathMamba2MIL(feature_dim, outputs, **values, **options)


def build(case):
    model = port(
        case["configuration"],
        case["featureDim"],
        case["outputs"],
        dropout=0.0,
        hilbert=case["coordinates"],
        cobra_pooling=case["cobraPooling"],
    )
    state = {
        mapped: torch.from_numpy(
            oceanpath_parity.tensor_values(name, shape, case["weightSeed"], index)
        )
        for index, (name, mapped, shape) in enumerate(case["tensors"])
    }
    model.load_state_dict(state, strict=True)
    return model


def inputs(case):
    coordinates = (
        torch.from_numpy(oceanpath_parity.coordinates(case)) if case["coordinates"] else None
    )
    return (
        torch.from_numpy(oceanpath_parity.features(case)),
        torch.from_numpy(oceanpath_parity.mask(case)),
        coordinates,
    )


def assert_close(actual, expected, tolerance=1e-5, scale=None):
    expected = torch.tensor(expected, dtype=torch.float64).reshape(-1)
    actual = torch.as_tensor(actual).detach().double().reshape(-1)
    assert actual.shape == expected.shape
    scale = max(1.0, float(expected.abs().max())) if scale is None else scale
    assert float((actual - expected).abs().max()) <= tolerance * scale


def test_cases_cover_the_options_and_shapes_that_change_the_computation():
    assert {case["coordinates"] for case in CASES} == {True, False}
    assert any(case["cobraPooling"] for case in CASES)
    assert any(len(case["lengths"]) > 1 for case in CASES)
    # Bags shorter than a convolution, within one chunk and across chunks of 256.
    lengths = [length for case in CASES for length in case["lengths"]]
    assert min(lengths) < 4 and max(lengths) > 512
    assert "oceanpath" in {case["configuration"] for case in CASES}


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_embeddings_logits_and_attention_match_oceanpath(case):
    model = build(case).eval()
    features, mask, coordinates = inputs(case)
    with torch.no_grad():
        output = model(features, mask, coords=coordinates, return_attention=True)
    assert_close(output["embedding"], case["eval"]["embedding"])
    assert_close(output["logits"], case["eval"]["logits"])
    for slide, length in enumerate(case["lengths"]):
        expected = case["eval"]["attention"][slide]
        assert_close(output["attention"][slide, :length], expected, 1e-4, max(expected))
        # Padding takes no attention.
        assert not output["attention"][slide, length:].any()


@pytest.mark.parametrize("case", FIT_CASES, ids=lambda case: case["id"])
def test_training_loss_and_gradients_match_oceanpath(case):
    model = build(case).train()
    features, mask, coordinates = inputs(case)
    loss = F.cross_entropy(model(features, mask, coords=coordinates), torch.tensor(case["labels"]))
    loss.backward()
    assert_close(loss, [case["fit"]["loss"]])
    gradients = {name: parameter.grad for name, parameter in model.named_parameters()}
    expected = case["fit"]["gradNorms"]
    assert {name for name, grad in gradients.items() if grad is not None} == set(expected)
    # The attention score's bias has no gradient but rounding (a softmax ignores a shift);
    # it is compared against the largest gradient instead of itself.
    floor = 1e-5 * max(expected.values())
    for name, norm in expected.items():
        assert float(gradients[name].double().norm()) == pytest.approx(norm, rel=1e-4, abs=floor)


@pytest.mark.parametrize("record", FIXTURE["initialization"], ids=lambda row: row["configuration"])
def test_a_seed_draws_oceanpaths_random_initialization(record):
    torch.manual_seed(record["seed"])
    built = port(record["configuration"], record["featureDim"], record["outputs"])
    norms = {name: float(value.double().norm()) for name, value in built.state_dict().items()}
    assert set(norms) == set(record["norms"])
    for name, norm in record["norms"].items():
        assert norms[name] == pytest.approx(norm, rel=1e-6), name


def test_a_pretrained_checkpoint_reproduces_oceanpath_on_a_synthetic_bag():
    """Runs only where the recorded checkpoint is available; CI has no model weights."""
    recorded = FIXTURE.get("checkpoint")
    path = os.environ.get(oceanpath_parity.CHECKPOINT_VARIABLE)
    if recorded is None or not path or not Path(path).is_file():
        pytest.skip(f"Set {oceanpath_parity.CHECKPOINT_VARIABLE} to the recorded checkpoint.")
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != recorded["sha256"]:
        pytest.skip(f"{path} is not the checkpoint the fixture recorded ({recorded['file']}).")
    built = port(recorded["configuration"], recorded["featureDim"], 2).eval()
    built.load_pretrained(torch.load(path, map_location="cpu", weights_only=True))
    features, mask, coordinates = inputs({**recorded, "coordinates": True})
    with torch.no_grad():
        output = built(features, mask, coords=coordinates, return_attention=True)
        built.aggregator.use_cobra_inference = True
        cobra = built(features, mask, coords=coordinates, return_attention=True)
    assert_close(output["embedding"][0], recorded["embedding"])
    assert_close(cobra["embedding"][0], recorded["cobraEmbedding"])
    assert_close(output["attention"][0], recorded["attention"], 1e-4, max(recorded["attention"]))
