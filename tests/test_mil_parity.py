"""HistoPilot's CLAM, TransMIL, DSMIL, DTFD-MIL and ACMIL reproduce the authors' code.

tests/fixtures/mil_parity.json holds outputs of the official implementations, recorded by
scripts/mil_parity_fixtures.py from weights and inputs that tests/support/mil_parity.py
rebuilds here. Each official tensor maps to exactly one HistoPilot tensor (loading is
strict), so a pass also shows the architectures match parameter for parameter.
"""

import pytest

torch = pytest.importorskip("torch")

from support import mil_parity  # noqa: E402
from torch import nn  # noqa: E402

from histopilot.models import catalog, registry  # noqa: E402
from histopilot.models.clam import smooth_top1_svm  # noqa: E402
from histopilot.training.module import _BinaryLoss  # noqa: E402

FIXTURE = mil_parity.load()
CASES = FIXTURE["cases"]


def build(case, monkeypatch, orders=None):
    model = registry.build(case["model"], case["featureDim"], case["outputs"], case["recipe"])
    state = {
        mapped: torch.from_numpy(mil_parity.tensor_values(name, shape, case["weightSeed"], index))
        for index, (name, mapped, shape) in enumerate(case["tensors"])
    }
    model.load_state_dict(state, strict=True)
    if case["model"] == "dtfd":
        # Feed the recorded pseudo-bag splits instead of drawing them.
        chosen = [torch.tensor(order) for order in (orders or case["orders"])]
        monkeypatch.setattr(model, "_orders", lambda count, device: chosen)
    # Evaluation mode: dropout and ACMIL's masking are off, as when the outputs were recorded.
    return model.eval()


def assert_close(actual, expected, tolerance=2e-5):
    expected = torch.tensor(expected, dtype=torch.float64).reshape(-1)
    actual = torch.as_tensor(actual).detach().double().reshape(-1)
    assert actual.shape == expected.shape
    scale = max(1.0, float(expected.abs().max()))
    assert float((actual - expected).abs().max()) <= tolerance * scale


def test_every_new_architecture_has_golden_cases():
    assert {case["model"] for case in CASES} == {
        "clam_sb",
        "clam_mb",
        "transmil",
        "dsmil",
        "dtfd",
        "acmil",
    }
    assert all(case["source"] in FIXTURE["sources"] or "+" in case["source"] for case in CASES)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_predictions_and_attention_match_the_official_code(case, monkeypatch):
    model = build(case, monkeypatch)
    features = torch.from_numpy(mil_parity.features(case))[None]
    with torch.no_grad():
        output = model(features, return_attention=True)
    if "logits" in case["eval"]:
        assert_close(output["logits"][0], case["eval"]["logits"])
    if "probabilities" in case["eval"]:
        assert_close(torch.softmax(output["logits"][0].double(), -1), case["eval"]["probabilities"])
    if "attention" in case["eval"]:
        assert_close(output["attention"][0], case["eval"]["attention"])


@pytest.mark.parametrize(
    "case", [case for case in CASES if "fit" in case], ids=lambda case: case["id"]
)
def test_training_loss_and_gradients_match_the_official_code(case, monkeypatch):
    model = build(case, monkeypatch, orders=case.get("orders", [None])[:1])
    features = torch.from_numpy(mil_parity.features(case))[None]
    labels = torch.tensor([case["label"]])
    criterion = (
        _BinaryLoss(1, None) if case["loss"] == "bce" else nn.CrossEntropyLoss(reduction="none")
    )
    if catalog.auxiliary_loss(case["model"]):
        output = model.training_output(features, None, labels, criterion)
        loss = (
            output["bagLossWeight"] * criterion(output["logits"], labels) + output["auxiliaryLoss"]
        )
    else:
        loss = criterion(model(features), labels)
    loss.sum().backward()
    assert_close(loss.sum(), [case["fit"]["loss"]])
    gradients = {name: parameter.grad for name, parameter in model.named_parameters()}
    assert {name for name, grad in gradients.items() if grad is not None} == set(
        case["fit"]["gradNorms"]
    )
    # A gradient that is zero up to rounding (an attention-score bias, to which a softmax is
    # invariant) is compared against the model's largest gradient instead of itself.
    floor = 1e-5 * max(case["fit"]["gradNorms"].values())
    for name, norm in case["fit"]["gradNorms"].items():
        actual = float(gradients[name].double().norm())
        assert actual == pytest.approx(norm, rel=1e-4, abs=floor), name


def test_clam_instance_svm_matches_smooth_topk():
    case = FIXTURE["svm"]
    logits, targets = mil_parity.svm_inputs(case)
    x = torch.from_numpy(logits).requires_grad_()
    loss = smooth_top1_svm(x, torch.from_numpy(targets)).mean()
    loss.backward()
    assert_close(loss, [case["loss"]], 1e-6)
    assert_close(x.grad, case["gradient"], 1e-6)
