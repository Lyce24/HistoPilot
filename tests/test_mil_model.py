"""Numerical and masking contracts for the optional native attention model."""

import pytest

torch = pytest.importorskip("torch")

from histopilot.models.abmil import ABMIL  # noqa: E402


def model(**kwargs):
    torch.manual_seed(51)
    return ABMIL(6, 3, embed_dim=8, attention_dim=5, dropout=0, **kwargs)


def test_padding_including_nan_cannot_change_a_slide_prediction():
    network = model().eval()
    features = torch.randn(1, 7, 6)
    expected = network(features, return_attention=True)
    padded = torch.cat((features, torch.full((1, 5, 6), float("nan"))), dim=1)
    mask = torch.tensor([[True] * 7 + [False] * 5])
    actual = network(padded, mask, return_attention=True)
    torch.testing.assert_close(actual["logits"], expected["logits"])
    torch.testing.assert_close(actual["attention"][:, :7], expected["attention"])
    assert torch.count_nonzero(actual["attention"][:, 7:]) == 0


def test_attention_pooling_is_permutation_invariant_and_normalized():
    network = model().eval()
    features = torch.randn(2, 11, 6)
    order = torch.randperm(11)
    original = network(features, return_attention=True)
    permuted = network(features[:, order], return_attention=True)
    torch.testing.assert_close(original["logits"], permuted["logits"])
    torch.testing.assert_close(original["attention"][:, order], permuted["attention"])
    torch.testing.assert_close(original["attention"].sum(dim=1), torch.ones(2))


def test_attention_normalization_stays_float32_under_bfloat16_autocast():
    network = model().eval()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = network(torch.randn(2, 20, 6), return_attention=True)
    assert output["attention"].dtype == torch.float32
    assert torch.isfinite(output["logits"]).all()


def test_gradient_checkpointing_preserves_forward_and_parameter_gradients():
    standard = model()
    checkpointed = model(gradient_checkpointing=True)
    checkpointed.load_state_dict(standard.state_dict())
    features = torch.randn(2, 7, 6)
    mask = torch.tensor([[True] * 7, [True] * 3 + [False] * 4])
    for network in (standard, checkpointed):
        network(features, mask).square().sum().backward()
    for left, right in zip(standard.parameters(), checkpointed.parameters(), strict=True):
        assert left.grad is not None and right.grad is not None
        torch.testing.assert_close(left.grad, right.grad)


@pytest.mark.parametrize("layers", [1, 3])
def test_projection_dropout_only_regularizes_between_fc_layers(layers):
    network = ABMIL(6, 2, embed_dim=8, attention_dim=4, num_fc_layers=layers, dropout=0.5)
    projection = list(network.patch_embed)
    assert isinstance(projection[-1], torch.nn.ReLU)
    assert sum(isinstance(layer, torch.nn.Dropout) for layer in projection) == layers - 1
    # The default one-layer projection is deterministic even while the
    # attention scorer remains stochastic during fitting.
    if layers == 1:
        features = torch.randn(2, 30, 6)
        torch.testing.assert_close(network.patch_embed(features), network.patch_embed(features))
    assert isinstance(network.attention_tanh[-1], torch.nn.Dropout)
    assert isinstance(network.attention_gate[-1], torch.nn.Dropout)


@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("classes", [2, 4])
def test_gated_and_ungated_models_produce_one_logit_per_class(gated, classes):
    network = ABMIL(
        6, classes, embed_dim=8, attention_dim=4, num_fc_layers=2, gated_attention=gated
    )
    assert network(torch.randn(3, 5, 6)).shape == (3, classes)


def test_empty_or_incorrectly_masked_bags_fail_before_softmax():
    network = model()
    with pytest.raises(ValueError, match="at least one"):
        network(torch.randn(1, 3, 6), torch.zeros(1, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match="masks must match"):
        network(torch.randn(1, 3, 6), torch.ones(1, 4, dtype=torch.bool))
    with pytest.raises(ValueError, match="nonempty"):
        network(torch.empty(1, 0, 6))
