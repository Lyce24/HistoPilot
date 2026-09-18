"""BCE attention and mean-logit ensembles agree with whole-bag predictions."""

import json
import runpy
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")
pytest.importorskip("h5py")
pytest.importorskip("PIL.Image")

from histopilot.application.predictors import checkpoint_snapshot  # noqa: E402
from histopilot.training.attention import _receipt, interpret  # noqa: E402
from histopilot.training.module import MILTrainModule  # noqa: E402

support = runpy.run_path(str(Path(__file__).with_name("test_attention_execution.py")))


def binary_ensemble(tmp_path):
    plan = support["attention_plan"](tmp_path)
    original = torch.load(plan["checkpoints"][0]["path"], map_location="cpu", weights_only=True)
    original["hyper_parameters"]["recipe"]["lossType"] = "bce"
    original["state_dict"]["model.classifier.weight"] = torch.zeros_like(
        original["state_dict"]["model.classifier.weight"][:1]
    )
    checkpoints = []
    for index, value in enumerate([4.0, -1.0]):
        original["state_dict"]["model.classifier.bias"] = torch.tensor([value])
        checkpoint_path = tmp_path / f"binary-{index}.ckpt"
        torch.save(original, checkpoint_path)
        checkpoints.append(checkpoint_snapshot(checkpoint_path, tmp_path))
    return {**plan, "method": "ensemble", "checkpoints": checkpoints, "aggregation": "mean_logit"}


def test_binary_mean_logit_attention_matches_predictor_and_resumes_stable_cache(
    tmp_path, monkeypatch
):
    plan = binary_ensemble(tmp_path)
    output = tmp_path / "attention"
    result = interpret(plan, output)
    positive = plan["target"]["classes"].index(plan["target"]["positiveClass"])
    assert result["slides"][0]["probabilities"][positive] == pytest.approx(
        torch.sigmoid(torch.tensor(1.5)).item()
    )
    maps = [
        json.loads((output / filename).read_text())
        for filename in (
            "slide-0-member-0.json",
            "slide-0-member-1.json",
            "slide-0.json",
        )
    ]
    for logit, member in zip([4.0, -1.0], maps[:2], strict=True):
        assert member["probabilities"][positive] == pytest.approx(
            torch.sigmoid(torch.tensor(logit)).item()
        )
        assert np.exp(member["logProbabilities"]).tolist() == pytest.approx(member["probabilities"])
    assert maps[2]["probabilities"] != pytest.approx(
        np.mean([member["probabilities"] for member in maps[:2]], axis=0)
    )
    assert [patch["weight"] for patch in maps[2]["patches"]] == pytest.approx(
        np.mean([[patch["weight"] for patch in member["patches"]] for member in maps[:2]], axis=0)
    )
    assert result["ensembleAggregation"] == "mean_logit"
    monkeypatch.setattr(
        MILTrainModule,
        "load_from_checkpoint",
        lambda *a, **k: pytest.fail("Must resume cached members"),
    )
    assert interpret(plan, output) == result


def test_changed_aggregation_cannot_reuse_attention_evidence(tmp_path):
    plan = binary_ensemble(tmp_path)
    output = tmp_path / "attention"
    interpret(plan, output)
    with pytest.raises(ValueError, match="evidence changed"):
        interpret({**plan, "aggregation": "mean_probability"}, output)


def test_mean_logit_attention_rejects_invalid_cached_log_probabilities(tmp_path):
    plan = binary_ensemble(tmp_path)
    output = tmp_path / "attention"
    interpret(plan, output)
    path = output / "slide-0-member-0.json"
    value = json.loads(path.read_text())
    value["logProbabilities"] = [-1, -1]
    path.write_text(json.dumps(value))
    receipt_path = output / ".slide-0-member-0.receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["artifact"] = _receipt(path)
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="log probabilities"):
        interpret(plan, output)
