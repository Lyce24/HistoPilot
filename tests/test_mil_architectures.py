"""Numerical contracts of CLAM, TransMIL, DSMIL, DTFD-MIL and ACMIL.

Every architecture must give a slide the same prediction and map whether it is alone
or padded in a minibatch, return attention that is a distribution over the slide's own
patches, and fit through the same training interface.
"""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch.nn import functional as F  # noqa: E402

from histopilot.models import catalog, registry  # noqa: E402
from histopilot.models.clam import CLAM, smooth_top1_svm  # noqa: E402
from histopilot.models.dsmil import mean_probability_logits  # noqa: E402
from histopilot.models.dtfd import DTFDMIL, split_sizes  # noqa: E402
from histopilot.models.transmil import moore_penrose_pinv  # noqa: E402

NEW_MODELS = ("clam_sb", "clam_mb", "transmil", "dsmil", "dtfd", "acmil")
SMALL = {"embedDim": 8, "transmilHeads": 2, "transmilLandmarks": 4, "dropout": 0}


def build(name, classes=2, **options):
    torch.manual_seed(31)
    return registry.build(name, 6, classes, {**SMALL, **options})


def outputs(name, classes):
    """Logit counts each architecture supports: one-logit BCE except CLAM-MB."""
    return [count for count in (1, 2, 3) if count > 1 or name != "clam_mb"]


def criterion(classes):
    if classes == 1:
        from histopilot.training.module import _BinaryLoss

        return _BinaryLoss(1, None)
    return torch.nn.CrossEntropyLoss(reduction="none")


@pytest.mark.parametrize("name", NEW_MODELS)
def test_a_padded_slide_matches_the_slide_alone_and_attention_is_its_distribution(name):
    for classes in outputs(name, 2):
        network = build(name, classes).eval()
        features = torch.randn(3, 23, 6)
        mask = torch.ones(3, 23, dtype=torch.bool)
        mask[1, 15:] = mask[2, 4:] = False
        # Padding may hold anything, even NaN; it must never reach a prediction.
        features[~mask] = float("nan")
        with torch.inference_mode():
            batch = network(features, mask, return_attention=True)
            torch.testing.assert_close(network(features, mask), batch["logits"])
            assert batch["logits"].shape == (3, classes)
            attention = batch["attention"]
            assert attention.dtype == torch.float32 and attention.shape == (3, 23)
            assert torch.all(attention >= 0) and torch.count_nonzero(attention[~mask]) == 0
            torch.testing.assert_close(attention.sum(dim=1), torch.ones(3))
            for index, count in enumerate((23, 15, 4)):
                alone = network(features[index : index + 1, :count], return_attention=True)
                torch.testing.assert_close(alone["logits"][0], batch["logits"][index])
                torch.testing.assert_close(alone["attention"][0], attention[index, :count])


@pytest.mark.parametrize("name", ["clam_sb", "clam_mb", "dsmil", "acmil"])
def test_set_pooling_architectures_ignore_patch_order(name):
    network = build(name, 3).eval()
    features = torch.randn(2, 13, 6)
    order = torch.randperm(13)
    with torch.inference_mode():
        original = network(features, return_attention=True)
        permuted = network(features[:, order], return_attention=True)
    torch.testing.assert_close(original["logits"], permuted["logits"])
    torch.testing.assert_close(original["attention"][:, order], permuted["attention"])


@pytest.mark.parametrize("name", NEW_MODELS)
def test_evaluation_is_repeatable_and_fitting_reaches_every_trained_parameter(name):
    network = build(name, 3, clamSubtyping=True)
    features = torch.randn(4, 17, 6)
    mask = torch.ones(4, 17, dtype=torch.bool)
    mask[3, 9:] = False
    network.eval()
    with torch.inference_mode():
        torch.testing.assert_close(network(features, mask), network(features, mask), rtol=0, atol=0)
    network.train()
    labels = torch.tensor([0, 1, 2, 1])
    loss_fn = criterion(3)
    if catalog.auxiliary_loss(name):
        output = network.training_output(features, mask, labels, loss_fn)
        assert output["auxiliaryLoss"].shape == (4,)
        losses = (
            output["bagLossWeight"] * loss_fn(output["logits"], labels) + output["auxiliaryLoss"]
        )
    else:
        losses = loss_fn(network(features, mask), labels)
    losses.mean().backward()
    missing = [key for key, value in network.named_parameters() if value.grad is None]
    assert missing == []


@pytest.mark.parametrize("name", NEW_MODELS)
def test_gradient_checkpointing_preserves_forward_and_gradients(name):
    standard = build(name, 2)
    checkpointed = build(name, 2, gradientCheckpointing=True)
    checkpointed.load_state_dict(standard.state_dict())
    features = torch.randn(2, 11, 6)
    mask = torch.tensor([[True] * 11, [True] * 6 + [False] * 5])
    labels = torch.tensor([0, 1])
    for network in (standard, checkpointed):
        torch.manual_seed(5)
        if catalog.auxiliary_loss(name):
            output = network.training_output(features, mask, labels, criterion(2))
            value = output["logits"].square().sum() + output["auxiliaryLoss"].sum()
        else:
            value = network(features, mask).square().sum()
        value.backward()
    for left, right in zip(standard.parameters(), checkpointed.parameters(), strict=True):
        if left.grad is None:
            assert right.grad is None
            continue
        torch.testing.assert_close(left.grad, right.grad)


@pytest.mark.parametrize("name", NEW_MODELS)
def test_attention_stays_float32_under_bfloat16_autocast(name):
    network = build(name, 2).eval()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16), torch.inference_mode():
        output = network(torch.randn(2, 19, 6), return_attention=True)
    assert output["attention"].dtype == torch.float32
    assert torch.isfinite(output["logits"]).all()


def reference_smooth_svm(logits, targets, alpha=1.0, tau=1.0, threshold=1e3):
    """oval-group/smooth-topk SmoothTop1SVM, written out row by row."""
    losses = []
    for row, target in zip(logits.tolist(), targets.tolist(), strict=True):
        delta = [0.0 if index == target else alpha for index in range(len(row))]
        shifted = [value + margin - row[target] for value, margin in zip(row, delta, strict=True)]
        top = sorted(row, reverse=True)
        if top[0] - top[1] >= tau * math.log(threshold):
            losses.append(max(shifted))
        else:
            losses.append(tau * math.log(sum(math.exp(value / tau) for value in shifted)))
    return torch.tensor(losses, dtype=torch.float64)


def test_clam_svm_instance_loss_matches_smooth_topk_including_the_hard_switch():
    logits = torch.tensor(
        [[0.3, -0.2], [2.0, 1.5], [-1.0, 4.0], [20.0, -1.0], [-30.0, 5.0]], dtype=torch.float64
    )
    targets = torch.tensor([0, 1, 1, 0, 0])
    torch.testing.assert_close(
        smooth_top1_svm(logits, targets), reference_smooth_svm(logits, targets)
    )
    # Large margins use the hinge: zero loss for a confident correct row.
    assert smooth_top1_svm(logits, targets)[3] == 0


def test_clam_samples_extreme_patches_and_weights_the_instance_loss():
    network = CLAM(6, 2, embed_dim=8, attention_dim=4, dropout=0, instance_samples=3)
    features = torch.randn(1, 10, 6)
    labels = torch.tensor([1])
    output = network.training_output(features, None, labels, criterion(2))
    _, weights, _, embeddings, _ = network._forward(features, None)
    order = weights[0, 0].argsort(descending=True)
    chosen = torch.cat((order[:3], order[-3:]))
    targets = torch.tensor([1, 1, 1, 0, 0, 0])
    expected = smooth_top1_svm(network.instance_classifiers[1](embeddings[0, chosen]), targets)
    torch.testing.assert_close(output["auxiliaryLoss"][0], 0.3 * expected.mean())
    assert output["bagLossWeight"] == 0.7
    # Fewer than 2B patches: half of them, so the two sets never overlap.
    small = network.training_output(features[:, :4], None, labels, criterion(2))
    assert torch.isfinite(small["auxiliaryLoss"]).all()
    single = network.training_output(features[:, :1], None, labels, criterion(2))
    assert single["auxiliaryLoss"][0] == 0
    clustering_off = CLAM(6, 2, embed_dim=8, attention_dim=4, bag_weight=1)
    assert (
        clustering_off.training_output(features, None, labels, criterion(2))["auxiliaryLoss"][0]
        == 0
    )


def test_clam_subtyping_also_teaches_other_classes_and_averages():
    torch.manual_seed(3)
    network = CLAM(
        6, 3, embed_dim=8, attention_dim=4, dropout=0, instance_samples=2, subtyping=True
    )
    network.train()
    features = torch.randn(1, 9, 6)
    output = network.training_output(features, None, torch.tensor([2]), criterion(3))
    _, weights, _, embeddings, _ = network._forward(features, None)
    branch = weights[0, 0]
    total = 0
    for index, head in enumerate(network.instance_classifiers):
        if index == 2:
            chosen = torch.cat((branch.topk(2).indices, branch.topk(2, largest=False).indices))
            targets = torch.tensor([1, 1, 0, 0])
        else:
            chosen, targets = branch.topk(2).indices, torch.zeros(2, dtype=torch.long)
        total = total + smooth_top1_svm(head(embeddings[0, chosen]), targets).mean()
    torch.testing.assert_close(output["auxiliaryLoss"][0], 0.3 * total / 3)


def test_clam_mb_maps_follow_the_predicted_class_and_need_class_logits():
    network = build("clam_mb", 3).eval()
    features = torch.randn(2, 12, 6)
    with torch.inference_mode():
        output = network(features, return_attention=True)
        _, weights, _, _, _ = network._forward(features, None)
    predicted = output["logits"].argmax(-1)
    for index in range(2):
        torch.testing.assert_close(output["attention"][index], weights[index, predicted[index]])
    with pytest.raises(ValueError, match="one attention branch per class"):
        CLAM(6, 1, multi_branch=True)


def test_dsmil_streams_critical_patch_and_averaged_prediction():
    network = build("dsmil", 2).eval()
    features = torch.randn(1, 9, 6)
    with torch.inference_mode():
        bag, maximum, weights, _ = network._streams(features, None)
        embeddings = network.patch_embed(features)
        scores = network.instance_classifier(embeddings)
        critical = embeddings[0, scores[0].argmax(dim=0)]
        queries = network.query(embeddings[0])
        expected = torch.softmax(queries @ network.query(critical).T / math.sqrt(128), dim=0)
    torch.testing.assert_close(maximum, scores.max(dim=1).values)
    torch.testing.assert_close(weights[0], expected)
    averaged = build("dsmil", 2, dsmilAveragePrediction=True).eval()
    with torch.inference_mode():
        bag, maximum, _, _ = averaged._streams(features, None)
        mean = (bag.softmax(-1) + maximum.softmax(-1)) / 2
        torch.testing.assert_close(averaged(features).softmax(-1), mean)
    single = mean_probability_logits(torch.tensor([[0.4]]), torch.tensor([[-1.2]]))
    expected_probability = (
        torch.sigmoid(torch.tensor(0.4)) + torch.sigmoid(torch.tensor(-1.2))
    ) / 2
    torch.testing.assert_close(torch.sigmoid(single[0, 0]), expected_probability)


def test_dsmil_weights_the_max_instance_loss():
    network = build("dsmil", 2, dsmilMaxLossWeight=0.25)
    features = torch.randn(2, 7, 6)
    labels = torch.tensor([1, 0])
    output = network.training_output(features, None, labels, criterion(2))
    _, maximum, _, _ = network._streams(features, None)
    torch.testing.assert_close(
        output["auxiliaryLoss"], 0.25 * F.cross_entropy(maximum, labels, reduction="none")
    )
    assert output["bagLossWeight"] == 0.75


def test_dtfd_splits_like_numpy_and_tests_with_one_fixed_split():
    for count, groups in ((10, 4), (3, 4), (8, 8), (1, 3)):
        sizes = split_sizes(count, groups)
        assert sizes == [
            len(part) for part in np.array_split(np.arange(count), groups) if len(part)
        ]
    network = build("dtfd", 2).eval()
    features = torch.randn(1, 21, 6)
    with torch.inference_mode():
        first = network(features)
        torch.manual_seed(99)
        assert torch.equal(network(features), first)


@pytest.mark.parametrize("distill", ["afs", "maxmins", "maxs"])
def test_dtfd_attention_is_each_patch_weight_in_the_slide_embedding(distill):
    network = DTFDMIL(6, 2, embed_dim=8, attention_dim=4, distill=distill, distill_patches=2).eval()
    features = torch.randn(1, 17, 6)
    with torch.inference_mode():
        output = network(features, return_attention=True)
        embeddings = network.patch_embed(features)[0]
    torch.testing.assert_close(output["embedding"][0], output["attention"][0] @ embeddings)
    if distill != "afs":
        # Only the distilled patches carry weight: at most 2 per pseudo-bag and end.
        limit = 4 * (4 if distill == "maxmins" else 2)
        assert 0 < torch.count_nonzero(output["attention"]) <= limit


@pytest.mark.parametrize("distill", ["afs", "maxmins", "maxs"])
def test_dtfd_tiers_learn_from_their_own_losses_only(distill):
    # The paper minimises each tier's loss over its own parameters (Eq. 14), and the
    # reference's two optimizers step tier one before tier two's loss exists.
    network = build("dtfd", 2, dtfdDistill=distill)
    features = torch.randn(1, 30, 6)
    labels = torch.tensor([1])
    output = network.training_output(features, None, labels, criterion(2))
    tier2_loss = criterion(2)(output["logits"], labels).sum()
    tier1 = [
        *network.patch_embed.parameters(),
        *network.tier1_attention.parameters(),
        *network.tier1_classifier.parameters(),
    ]
    tier2 = [*network.tier2_attention.parameters(), *network.classifier.parameters()]
    assert all(
        grad is None
        for grad in torch.autograd.grad(tier2_loss, tier1, allow_unused=True, retain_graph=True)
    )
    assert all(
        grad is not None for grad in torch.autograd.grad(tier2_loss, tier2, retain_graph=True)
    )
    tier1_grads = torch.autograd.grad(output["auxiliaryLoss"].sum(), tier1, allow_unused=True)
    assert all(grad is not None for grad in tier1_grads[:2])


def test_dtfd_fits_each_pseudo_bag_with_the_slide_label():
    torch.manual_seed(2)
    network = build("dtfd", 3, dtfdPseudoBags=3)
    features = torch.randn(2, 10, 6)
    mask = torch.tensor([[True] * 10, [True] * 2 + [False] * 8])
    labels = torch.tensor([2, 0])
    output = network.training_output(features, mask, labels, criterion(3))
    assert output["bagLossWeight"] == 1.0 and output["auxiliaryLoss"].shape == (2,)
    assert torch.all(output["auxiliaryLoss"] > 0)


def test_acmil_masks_top_patches_only_while_fitting():
    network = build("acmil", 2, acmilBranches=3, acmilMaskedPatches=5, acmilMaskDrop=0.6)
    scores = torch.randn(1, 3, 8)
    mask = torch.tensor([[True] * 6 + [False] * 2])
    hidden = network._hidden(scores, mask)
    assert hidden.sum(dim=-1).tolist() == [[3, 3, 3]]
    assert not hidden[0, :, 6:].any()
    top = scores[0, :, :6].topk(5, dim=-1).indices
    for branch in range(3):
        assert set(hidden[0, branch].nonzero().squeeze(1).tolist()) <= set(top[branch].tolist())
    # A two-patch slide keeps one patch visible in every branch.
    assert network._hidden(scores, torch.tensor([[True] * 2 + [False] * 6])).sum(-1).tolist() == [
        [1, 1, 1]
    ]
    network.eval()
    features = torch.randn(1, 8, 6)
    with torch.inference_mode():
        torch.testing.assert_close(network(features), network(features), rtol=0, atol=0)


def test_acmil_adds_branch_losses_and_attention_similarity():
    network = build("acmil", 2, acmilBranches=2, acmilMaskedPatches=0)
    features = torch.randn(1, 9, 6)
    labels = torch.tensor([1])
    output = network.training_output(features, None, labels, criterion(2))
    _, branch_logits, weights, _, _ = network._forward(features, None)
    expected = F.cross_entropy(branch_logits[0], labels.repeat(2)) + F.cosine_similarity(
        weights[:, 0], weights[:, 1], dim=-1
    )
    torch.testing.assert_close(output["auxiliaryLoss"], expected)
    one = build("acmil", 2, acmilBranches=1)
    assert one.training_output(features, None, labels, criterion(2))["auxiliaryLoss"][0] == 0


def test_transmil_pseudo_inverse_converges_and_maps_fold_grid_copies():
    torch.manual_seed(4)
    matrix = torch.softmax(torch.randn(1, 2, 6, 6) * 0.5, dim=-1)
    torch.testing.assert_close(
        moore_penrose_pinv(matrix.double(), 12),
        torch.linalg.pinv(matrix.double()),
        rtol=1e-6,
        atol=1e-6,
    )
    network = build("transmil", 2, transmilLayers=3).eval()
    # Five patches fill a 3 × 3 grid with four copies; the map still has five entries.
    with torch.inference_mode():
        output = network(torch.randn(1, 5, 6), return_attention=True)
    assert output["attention"].shape == (1, 5)
    torch.testing.assert_close(output["attention"].sum(), torch.tensor(1.0))
    for layers, encoding in ((1, True), (2, False), (4, True)):
        network = build("transmil", 3, transmilLayers=layers, transmilPositionEncoding=encoding)
        assert network.eval()(torch.randn(2, 7, 6)).shape == (2, 3)


def test_transmil_order_and_position_encoding_matter():
    network = build("transmil", 2).eval()
    features = torch.randn(1, 16, 6)
    with torch.inference_mode():
        shuffled = network(features[:, torch.randperm(16)])
    assert not torch.allclose(network(features), shuffled)
    with pytest.raises(ValueError, match="divide evenly"):
        registry.build("transmil", 6, 2, {"embedDim": 10, "transmilHeads": 4})


def test_clam_attention_dropout_follows_the_published_rule_unless_switched():
    def rates(**recipe):
        network = registry.build("clam_sb", 6, 2, {"embedDim": 8, **recipe})
        return network.patch_embed[2].p, network.attention_a[-1].p, network.attention_b[-1].p

    # The reference appends Dropout(0.25) to both branches whenever dropout is on.
    assert rates(dropout=0.5) == (0.5, 0.25, 0.25)
    assert rates(dropout=0.1) == (0.1, 0.25, 0.25)
    assert rates(dropout=0.0) == (0.0, 0.0, 0.0)
    assert rates(dropout=0.5, clamRecipeAttentionDropout=True) == (0.5, 0.5, 0.5)


def test_dsmil_reads_features_directly_unless_its_input_layer_is_on():
    published = registry.build("dsmil", 6, 2, {"embedDim": 8})
    assert isinstance(published.patch_embed, torch.nn.Identity)
    assert published.instance_classifier.in_features == 6 and published.classifier.kernel_size == (
        6,
    )
    projected = registry.build(
        "dsmil", 6, 2, {"embedDim": 8, "numFcLayers": 2, "dsmilProjection": True}
    )
    assert sum(isinstance(layer, torch.nn.Linear) for layer in projected.patch_embed) == 2
    assert projected.instance_classifier.in_features == 8 and projected.classifier.kernel_size == (
        8,
    )
    features = torch.randn(2, 9, 6)
    for network in (published, projected):
        assert network.eval()(features).shape == (2, 2)


@pytest.mark.parametrize("name", ["dtfd", "acmil"])
def test_dtfd_and_acmil_use_the_published_bias_free_input_layer(name):
    network = registry.build(name, 6, 2, {"embedDim": 8, "numFcLayers": 3, "dtfdResidualBlocks": 2})
    assert network.patch_embed.fc1.bias is None
    # Shared fully connected layers do not apply; DTFD-MIL has the reference's residual blocks.
    expected = 2 if name == "dtfd" else 0
    assert len(network.patch_embed.resBlocks) == expected
    assert all(
        layer.bias is None
        for layer in network.patch_embed.modules()
        if isinstance(layer, torch.nn.Linear)
    )


def test_dtfd_averages_seeded_test_splits_like_the_reference_mean_inference():
    torch.manual_seed(12)
    single = build("dtfd", 3, dtfdDistill="maxmins", dtfdTestSplits=1).eval()
    averaged = build("dtfd", 3, dtfdDistill="maxmins", dtfdTestSplits=4).eval()
    averaged.load_state_dict(single.state_dict())
    features = torch.randn(1, 40, 6)
    with torch.inference_mode():
        one = single(features, return_attention=True)
        mean = averaged(features, return_attention=True)
        views = []
        for order in averaged._orders(40, features.device):
            embeddings = averaged.patch_embed(features[0])
            views.append(
                averaged._split(embeddings, averaged.tier1_attention(embeddings), order, True)
            )
    # The first split is the one-split prediction; four splits average probabilities and maps.
    torch.testing.assert_close(views[0][1], one["logits"][0])
    expected = torch.stack([torch.softmax(view[1].double(), -1) for view in views]).mean(0)
    torch.testing.assert_close(torch.softmax(mean["logits"][0].double(), -1), expected)
    torch.testing.assert_close(
        mean["attention"][0], torch.stack([view[3] for view in views]).mean(0)
    )
    torch.testing.assert_close(mean["attention"].sum(), torch.tensor(1.0))
    with torch.inference_mode():
        torch.testing.assert_close(averaged(features), mean["logits"], rtol=0, atol=0)
    # One-logit heads average sigmoids.
    binary = build("dtfd", 1, dtfdTestSplits=3).eval()
    with torch.inference_mode():
        logits = binary(features)
        orders = binary._orders(40, features.device)
        embeddings = binary.patch_embed(features[0])
        sigmoids = torch.stack(
            [
                torch.sigmoid(
                    binary._split(embeddings, binary.tier1_attention(embeddings), order)[1].double()
                )
                for order in orders
            ]
        )
    torch.testing.assert_close(torch.sigmoid(logits[0].double()), sigmoids.mean(0))
