"""Contracts for probes over one slide-encoder embedding per slide."""

import pytest

torch = pytest.importorskip("torch")

from histopilot.models import registry  # noqa: E402
from histopilot.models.slide import SlideProbe  # noqa: E402


def test_a_linear_probe_is_an_affine_read_out_of_the_embedding():
    torch.manual_seed(7)
    probe = SlideProbe(6, 3, dropout=0, input_dropout=0).eval()
    features = torch.randn(4, 1, 6)
    expected = probe.head(features[:, 0])
    torch.testing.assert_close(probe(features), expected)


def test_a_hidden_layer_is_used_when_one_is_requested():
    probe = SlideProbe(6, 2, hidden_dim=5, dropout=0, input_dropout=0)
    assert isinstance(probe.head, torch.nn.Sequential)
    assert probe(torch.randn(3, 1, 6)).shape == (3, 2)


def test_a_probe_refuses_a_patch_bag_instead_of_silently_pooling_it():
    probe = SlideProbe(6, 2, dropout=0, input_dropout=0).eval()
    with pytest.raises(ValueError, match="exactly one present embedding"):
        probe(torch.randn(2, 9, 6))


def test_a_probe_refuses_a_masked_out_embedding():
    probe = SlideProbe(6, 2, dropout=0, input_dropout=0).eval()
    features = torch.randn(2, 1, 6)
    with pytest.raises(ValueError):
        probe(features, torch.tensor([[True], [False]]))


def test_a_probe_has_no_attention_to_report():
    probe = SlideProbe(6, 2, dropout=0, input_dropout=0).eval()
    with pytest.raises(ValueError, match="no per-patch attention"):
        probe(torch.randn(1, 1, 6), return_attention=True)


def test_the_registry_builds_both_probes_from_a_recipe():
    linear = registry.build("slide_linear", 6, 2, {"dropout": 0, "inputDropout": 0})
    hidden = registry.build("slide_mlp", 6, 2, {"embedDim": 5, "dropout": 0, "inputDropout": 0})
    assert isinstance(linear.head, torch.nn.Linear)
    assert isinstance(hidden.head, torch.nn.Sequential)
    assert linear(torch.randn(2, 1, 6)).shape == (2, 2)


@pytest.mark.parametrize("architecture", ["slide_linear", "slide_mlp"])
@pytest.mark.parametrize("separate_rates", [False, True])
def test_training_optimizer_updates_probe_head_and_preserves_checkpoint_keys(
    architecture, separate_rates
):
    pytest.importorskip("lightning")
    from histopilot.schemas.development import TrainingRecipe
    from histopilot.training.module import MILTrainModule

    recipe = TrainingRecipe(
        model=architecture,
        embedDim=5,
        dropout=0,
        inputDropout=0,
        **({"headLearningRate": 0.01, "aggregatorLearningRate": 0.001} if separate_rates else {}),
    ).model_dump()
    target = {
        "task": "binary_classification",
        "unit": "patient",
        "classes": ["no", "yes"],
        "positiveClass": "yes",
    }
    module = MILTrainModule(6, target, recipe)
    assert module.model.classifier is module.model.head
    assert all(key.startswith("head.") for key in module.model.state_dict())
    optimizer = module.configure_optimizers()
    parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    assert len(parameters) == len({id(parameter) for parameter in parameters})
    assert {id(parameter) for parameter in parameters} == {
        id(parameter) for parameter in module.parameters()
    }
    original = {
        name: parameter.detach().clone() for name, parameter in module.model.named_parameters()
    }
    logits = module(torch.randn(4, 1, 6))
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 0, 1])).backward()
    optimizer.step()
    assert any(
        not torch.equal(original[name], parameter)
        for name, parameter in module.model.named_parameters()
    )
    if separate_rates:
        assert all(group["lr"] == 0.01 for group in optimizer.param_groups if group["params"])
    restored = registry.build(architecture, 6, 2, recipe)
    restored.load_state_dict(module.model.state_dict(), strict=True)
    module.eval()
    restored.eval()
    sample = torch.randn(3, 1, 6)
    torch.testing.assert_close(module(sample), restored(sample))
