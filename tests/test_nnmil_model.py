"""Method-level checks against the nnMIL equations and inference contract."""

import math

import pytest

torch = pytest.importorskip("torch")
from torch.nn import functional as F  # noqa: E402

from histopilot.models.nnmil import NNMIL  # noqa: E402


@pytest.fixture(autouse=True)
def small_cpu_workload():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def reference_view(model, x, mask, indices):
    # Select matching coordinates and columns. The bag vectors used for
    # pooling remain full D, with no nonlinear feature projection.
    columns = torch.arange(model.in_dim) if indices is None else indices
    clean = x.masked_fill(~mask.unsqueeze(-1), 0)
    v = F.linear(
        clean[:, :, columns], model.attention_tanh.weight[:, columns], model.attention_tanh.bias
    )
    u = F.linear(
        clean[:, :, columns], model.attention_gate.weight[:, columns], model.attention_gate.bias
    )
    score = model.attention_score(v.tanh() * u.sigmoid()).squeeze(-1)
    attention = score.masked_fill(~mask, -torch.inf).softmax(1)
    embedding = torch.einsum("bn,bnd->bd", attention, clean)
    return model.classifier(embedding), attention, embedding


def test_training_matches_selected_projection_columns_and_full_dimensional_pooling():
    torch.manual_seed(51)
    model = NNMIL(9, 3, attention_dim=4, dropout=0)
    x = torch.randn(2, 6, 9)
    mask = torch.tensor([[True] * 6, [True, True, True, False, False, False]])
    state = torch.get_rng_state()
    output = model(x, mask, return_attention=True)
    torch.set_rng_state(state)
    columns = torch.randperm(9)[:4]
    expected, attention, embedding = reference_view(model, x, mask, columns)
    torch.testing.assert_close(output["logits"], expected)
    torch.testing.assert_close(output["attention"], attention)
    torch.testing.assert_close(output["embedding"], embedding)
    assert output["embedding"].shape == (2, 9)
    output["logits"].sum().backward()
    excluded = sorted(set(range(9)) - set(columns.tolist()))
    assert torch.count_nonzero(model.attention_tanh.weight.grad[:, excluded]) == 0
    assert torch.count_nonzero(model.attention_tanh.weight.grad[:, columns]) > 0
    assert bool((model.classifier.weight.grad[:, excluded] != 0).all())


@pytest.mark.parametrize("aggregation", ["mean_logits", "mean_probabilities"])
@pytest.mark.parametrize("outputs", [1, 3])
def test_evaluation_matches_explicit_window_predictions_and_uncertainty(aggregation, outputs):
    torch.manual_seed(10)
    model = NNMIL(
        9,
        outputs,
        attention_dim=4,
        dropout=0,
        window_stride_divisor=2,
        window_shuffle=False,
        window_aggregation=aggregation,
    ).eval()
    x = torch.randn(2, 5, 9)
    mask = torch.ones(2, 5, dtype=torch.bool)
    windows = [torch.arange(start, start + 4) for start in (0, 2, 4, 5)]
    views = [reference_view(model, x, mask, indices) for indices in windows]
    logits = torch.stack([view[0] for view in views]).double()
    probabilities = (
        torch.cat(((-logits).sigmoid(), logits.sigmoid()), -1)
        if outputs == 1
        else logits.softmax(-1)
    )
    output = model(x, mask, return_attention=True, return_uncertainty=True)
    expected_logits = logits.mean(0)
    if aggregation == "mean_probabilities":
        log_p = probabilities.mean(0).log()
        expected_logits = (log_p[:, 1] - log_p[:, 0]).unsqueeze(-1) if outputs == 1 else log_p
    torch.testing.assert_close(output["logits"], expected_logits)
    torch.testing.assert_close(
        output["attention"], torch.stack([view[1] for view in views]).mean(0)
    )
    torch.testing.assert_close(
        output["embedding"], torch.stack([view[2] for view in views]).mean(0)
    )
    uncertainty = output["window_uncertainty"]
    assert uncertainty["windowCount"] == 4
    mean_p = probabilities.mean(0)
    entropy_mean = -(mean_p * mean_p.log()).sum(-1)
    mean_entropy = -(probabilities * probabilities.log()).sum(-1).mean(0)
    torch.testing.assert_close(uncertainty["entropyOfMeanProbability"], entropy_mean)
    torch.testing.assert_close(uncertainty["meanWindowEntropy"], mean_entropy)
    torch.testing.assert_close(uncertainty["mutualInformation"], entropy_mean - mean_entropy)
    torch.testing.assert_close(
        uncertainty["probabilityVariance"], probabilities.var(0, correction=0)
    )


def test_single_window_bce_has_nonzero_entropy_and_zero_dispersion():
    model = NNMIL(3, 1, attention_dim=8, dropout=0).eval()
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.zero_()
    output = model(torch.randn(1, 2, 3), return_uncertainty=True)
    uncertainty = output["window_uncertainty"]
    assert uncertainty["windowCount"] == 1
    assert uncertainty["meanWindowEntropy"].item() == pytest.approx(math.log(2))
    assert uncertainty["mutualInformation"].item() == 0
    assert uncertainty["probabilityVariance"].tolist() == [[0, 0]]


def test_masked_nan_padding_does_not_change_any_view():
    model = NNMIL(7, 2, attention_dim=3, dropout=0).eval()
    original = torch.randn(1, 4, 7)
    padded = torch.cat((original, torch.full((1, 5, 7), torch.nan)), 1)
    mask = torch.tensor([[True] * 4 + [False] * 5])
    output = model(padded, mask, return_attention=True, return_uncertainty=True)
    base = model(original, return_attention=True, return_uncertainty=True)
    torch.testing.assert_close(output["logits"], base["logits"])
    torch.testing.assert_close(output["attention"][:, :4], base["attention"])
    assert output["attention"][:, 4:].sum().item() == 0
    for name, values in base["window_uncertainty"].items():
        if name != "windowCount":
            torch.testing.assert_close(output["window_uncertainty"][name], values)
    with pytest.raises(ValueError, match="at least one"):
        model(padded, torch.zeros_like(mask))


def test_feature_order_is_persistent_and_does_not_consume_rng_during_evaluation():
    model = NNMIL(11, 2, attention_dim=4, window_seed=42).eval()
    x = torch.randn(2, 3, 11)
    state = torch.get_rng_state().clone()
    initial = model(x)
    assert torch.equal(state, torch.get_rng_state())
    restored = NNMIL(11, 2, attention_dim=4, window_seed=99).eval()
    assert not torch.equal(model.feature_order, restored.feature_order)
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(x), initial, atol=0, rtol=0)
    assert sorted(torch.cat(restored.evaluation_windows()).unique().tolist()) == list(range(11))
    bad = model.state_dict()
    bad["feature_order"] = torch.zeros(11, dtype=torch.long)
    with pytest.raises(ValueError, match="permutation"):
        restored.load_state_dict(bad)


def test_feature_sampling_ablation_keeps_gating_and_one_full_dimensional_view():
    model = NNMIL(8, 2, attention_dim=3, feature_sampling=False, dropout=0)
    x, mask = torch.randn(1, 4, 8), torch.ones(1, 4, dtype=torch.bool)
    expected, _, _ = reference_view(model, x, mask, None)
    torch.testing.assert_close(model(x), expected)
    model.eval()
    output = model(x, return_uncertainty=True)
    torch.testing.assert_close(output["logits"], expected.double())
    assert output["window_uncertainty"]["windowCount"] == 1


def test_checkpointed_training_reproduces_dropout_and_gradients():
    a = NNMIL(9, 2, attention_dim=4, dropout=0.25)
    b = NNMIL(9, 2, attention_dim=4, dropout=0.25, gradient_checkpointing=True)
    b.load_state_dict(a.state_dict())
    x = torch.randn(2, 5, 9)
    torch.manual_seed(7)
    first = a(x)
    first.sum().backward()
    torch.manual_seed(7)
    second = b(x)
    second.sum().backward()
    torch.testing.assert_close(first, second)
    for p, q in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad)


def test_cpu_autocast_keeps_attention_normalization_and_pooling_finite():
    model = NNMIL(17, 2, attention_dim=5, dropout=0).eval()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model(torch.randn(2, 53, 17), return_attention=True, return_uncertainty=True)
    assert bool(torch.isfinite(output["logits"]).all())
    torch.testing.assert_close(output["attention"].sum(1), torch.ones(2))
    assert output["attention"].dtype == torch.float32


def test_finite_extreme_embeddings_do_not_overflow_when_combining_views():
    model = NNMIL(8, 2, attention_dim=2, dropout=0, window_shuffle=False).eval()
    with torch.no_grad():
        model.attention_tanh.weight.zero_()
        model.attention_gate.weight.zero_()
        model.classifier.weight.fill_(1e-38)
        model.classifier.bias.copy_(torch.tensor([1e38, -1e38]))
    features = torch.full((1, 2, 8), 2e38)
    output = model(features, return_attention=True, return_uncertainty=True)
    assert bool(torch.isfinite(output["embedding"]).all())
    assert bool(torch.isfinite(output["logits"]).all())
    torch.testing.assert_close(output["embedding"], features[:, 0, :])
    assert output["window_uncertainty"]["meanWindowEntropy"].item() == 0
    assert output["window_uncertainty"]["probabilityVariance"].tolist() == [[0, 0]]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"attention_dim": 0},
        {"window_stride_divisor": 0},
        {"window_seed": -1},
        {"window_aggregation": "majority"},
        {"dropout": 1},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        NNMIL(5, 2, **kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_checkpoint_feature_order_matches_cpu_inference():
    cpu = NNMIL(33, 2, attention_dim=8, dropout=0).eval()
    gpu = NNMIL(33, 2, attention_dim=8, dropout=0, window_seed=99).cuda().eval()
    gpu.load_state_dict(cpu.state_dict())
    features = torch.randn(2, 9, 33)
    with torch.inference_mode():
        expected = cpu(features, return_uncertainty=True)
        actual = gpu(features.cuda(), return_uncertainty=True)
    assert torch.equal(cpu.feature_order, gpu.feature_order.cpu())
    torch.testing.assert_close(actual["logits"].cpu(), expected["logits"], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        actual["window_uncertainty"]["meanWindowEntropy"].cpu(),
        expected["window_uncertainty"]["meanWindowEntropy"],
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_mixed_precision_training_and_inference_are_finite(dtype):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("GPU does not support bfloat16")
    model = NNMIL(64, 2, attention_dim=16, dropout=0.25, gradient_checkpointing=True).cuda()
    features = torch.randn(2, 31, 64, device="cuda")
    mask = torch.ones(2, 31, dtype=torch.bool, device="cuda")
    mask[0, 20:] = False
    features[0, 20:] = torch.nan
    with torch.autocast(device_type="cuda", dtype=dtype):
        loss = F.cross_entropy(model(features, mask).float(), torch.tensor([0, 1], device="cuda"))
    loss.backward()
    assert bool(torch.isfinite(loss))
    assert all(bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters())
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype):
        result = model(features, mask, return_attention=True, return_uncertainty=True)
    assert bool(torch.isfinite(result["logits"]).all())
    torch.testing.assert_close(result["attention"].sum(1), torch.ones(2, device="cuda"))
