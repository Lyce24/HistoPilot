"""Pooling baselines honour the recipe's dropout, so attention ablations compare like with like."""

import pytest

torch = pytest.importorskip("torch")

from histopilot.models.pooling import PoolingMIL  # noqa: E402


@pytest.mark.parametrize("pooling", ["mean", "max"])
def test_single_layer_pooling_applies_dropout_in_training_only(pooling):
    torch.manual_seed(0)
    features = torch.randn(4, 12, 8)
    outputs = {}
    for dropout in (0.0, 0.9):
        torch.manual_seed(1)
        model = PoolingMIL(8, 2, pooling=pooling, embed_dim=16, num_fc_layers=1, dropout=dropout)
        torch.manual_seed(2)
        model.train()
        training = model(features)
        model.eval()
        outputs[dropout] = (training, model(features))
    # Same initialization: evaluation ignores dropout, training does not.
    torch.testing.assert_close(outputs[0.0][1], outputs[0.9][1])
    assert not torch.allclose(outputs[0.0][0], outputs[0.9][0])


def test_pooled_dropout_keeps_saved_weight_layout():
    model = PoolingMIL(8, 2, embed_dim=16, num_fc_layers=1, dropout=0.25)
    assert set(model.state_dict()) == {
        "patch_embed.0.weight",
        "patch_embed.0.bias",
        "classifier.weight",
        "classifier.bias",
    }
