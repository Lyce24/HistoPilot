"""Real CPU fold/refit/checkpoint/inference coverage for all clinical input arms."""

import copy
import json
import runpy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

from histopilot.application.predictors import checkpoint_snapshot  # noqa: E402
from histopilot.datasets.datamodule import MILDataModule  # noqa: E402
from histopilot.datasets.mil import SlideDataset  # noqa: E402
from histopilot.storage.packed import _stamp  # noqa: E402
from histopilot.training.fold import train_fold  # noqa: E402
from histopilot.training.inference import evaluate  # noqa: E402
from histopilot.training.module import MILTrainModule  # noqa: E402
from histopilot.training.refit import train_refit  # noqa: E402

support = runpy.run_path(str(Path(__file__).with_name("test_mil_training.py")))
FIELDS = [{"field": "age", "kind": "numeric"}, {"field": "site", "kind": "categorical"}]


def clinical_plan(tmp_path, mode, model="abmil"):
    plan = support["tiny_plan"](tmp_path)
    plan["recipe"].update(inputMode=mode, clinicalFields=FIELDS, model=model, maxEpochs=1)
    plan["data"]["clinicalValues"] = {
        row["slideId"]: {
            "age": (30 + i if row["partition"] == "train" else 1000 + i),
            "site": "fitting" if row["partition"] == "train" else "unseen",
        }
        for i, row in enumerate(plan["data"]["memberships"])
    }
    for entry in plan["data"]["featureFiles"].values():
        entry.update(_stamp(Path(entry["path"]).stat()))
    return plan


@pytest.mark.parametrize(
    "mode,model",
    [
        ("clinical", "abmil"),
        ("multimodal", "abmil"),
        ("multimodal", "nnmil"),
        ("multimodal", "mean_pool"),
    ],
)
def test_fold_checkpoint_external_inference_and_refit(tmp_path, monkeypatch, mode, model):
    plan = clinical_plan(tmp_path, mode, model)
    if mode == "clinical":
        monkeypatch.setattr(
            SlideDataset, "_native", lambda *args: pytest.fail("Clinical-only read images")
        )
        monkeypatch.setattr(
            SlideDataset, "_packed", lambda *args: pytest.fail("Clinical-only read a pack")
        )
    fitted = train_fold(plan, tmp_path / "fit")
    loaded = MILTrainModule.load_from_checkpoint(fitted["bestCheckpointPath"], weights_only=True)
    preprocessor = loaded.clinical_preprocessor
    assert preprocessor["trainingPatientCount"] == 4
    assert preprocessor["columns"][0]["median"] == 31.5
    assert preprocessor["columns"][1]["categories"] == ['"fitting"']
    image = torch.randn(2, 7, 4)
    clinical = [{"age": None, "site": "unseen"}, {"age": 45, "site": None}]
    loaded.eval()
    first = loaded.prediction_output(image, clinical=clinical)["logits"]
    assert torch.isfinite(first).all()
    if mode == "clinical":
        torch.testing.assert_close(
            first, loaded.prediction_output(image * 100, clinical=clinical)["logits"]
        )
    else:
        assert loaded.clinical_head.weight.grad is None  # Reloaded frozen predictor.
        assert loaded.clinical_head.weight.detach().abs().sum() > 0
    external = copy.deepcopy(plan["data"])
    external["memberships"] = [row for row in external["memberships"] if row["partition"] == "test"]
    ids = {row["slideId"] for row in external["memberships"]}
    external["featureFiles"] = {
        key: row for key, row in external["featureFiles"].items() if key in ids
    }
    external["sourceStamps"] = {row["path"]: row for row in external["featureFiles"].values()}
    external.update(inputMode=mode, clinicalFields=FIELDS)
    evaluation = {
        "runId": "clinical-eval",
        "method": "refit",
        "target": plan["target"],
        "data": external,
        "resources": plan["resources"],
        "device": "cpu",
        "checkpoints": [checkpoint_snapshot(fitted["bestCheckpointPath"], tmp_path / "fit")],
        "inference": {
            "loadingPolicy": "per_slide",
            "batchSize": 2,
            "numWorkers": 0,
            "precision": "float32",
            "patientAggregation": "mean",
            "decisionThreshold": 0.5,
        },
    }
    result = evaluate(evaluation, tmp_path / "external")
    assert result["patientCount"] == 4
    assert evaluate(evaluation, tmp_path / "external")["artifacts"] == result["artifacts"]
    predictions = json.loads((tmp_path / "external" / "predictions.json").read_text())
    assert len(predictions["patientRecords"]) == 4
    changed = copy.deepcopy(evaluation)
    changed["data"]["clinicalFields"] = FIELDS[::-1]
    with pytest.raises(ValueError, match="clinical schema"):
        evaluate(changed, tmp_path / "invalid")
    refit = copy.deepcopy(plan)
    refit["epochBudget"] = {"epochs": 1}
    refit["recipe"].update(maxEpochs=1, minEpochs=1, earlyStopping=False)
    for row in refit["data"]["memberships"]:
        row.update(partition="train", phase="refit", pool="development")
    result = train_refit(refit, tmp_path / "refit")
    refitted = MILTrainModule.load_from_checkpoint(result["bestCheckpointPath"], weights_only=True)
    assert refitted.clinical_preprocessor["trainingPatientCount"] == 12
    assert refitted.clinical_preprocessor != preprocessor
    assert not result["validationUsed"] and not result["testDataUsed"]


def test_validation_changes_do_not_change_fitted_clinical_transform(tmp_path):
    plan = clinical_plan(tmp_path, "multimodal")
    module = MILDataModule({**plan, **plan["data"]})
    expected = copy.deepcopy(module.clinical_preprocessor)
    for row in plan["data"]["memberships"]:
        if row["partition"] != "train":
            plan["data"]["clinicalValues"][row["slideId"]] = {
                "age": -900000,
                "site": "only-validation",
            }
    assert MILDataModule({**plan, **plan["data"]}).clinical_preprocessor == expected


def test_combined_attention_uses_frozen_clinical_values_and_invalidates_cache(tmp_path):
    import h5py

    from histopilot.training.attention import interpret

    attention_support = runpy.run_path(str(Path(__file__).with_name("test_attention_execution.py")))
    attention = attention_support["attention_plan"](tmp_path / "images")
    root = tmp_path / "combined"
    root.mkdir()
    plan = clinical_plan(root, "multimodal")
    result = train_fold(plan, root / "fit")
    attention["checkpoints"] = [checkpoint_snapshot(result["bestCheckpointPath"], root / "fit")]
    attention["slides"][0]["clinical"] = {"age": 30, "site": "fitting"}
    output = tmp_path / "attention"
    interpret(attention, output)
    first_hash = json.loads((output / ".slide-0-member-0.receipt.json").read_text())["inputHash"]
    actual = json.loads((output / "slide-0.json").read_text())
    model = MILTrainModule.load_from_checkpoint(result["bestCheckpointPath"], weights_only=True)
    model.eval()
    with h5py.File(attention["slides"][0]["featurePath"]) as handle, torch.inference_mode():
        expected = model.prediction_output(
            torch.tensor(handle["features"][:]).unsqueeze(0),
            clinical=[attention["slides"][0]["clinical"]],
            return_attention=True,
        )
    assert actual["probabilities"] == pytest.approx(
        torch.softmax(expected["logits"][0].double(), -1).tolist()
    )
    assert [row["weight"] for row in actual["patches"]] == pytest.approx(
        expected["attention"][0].tolist()
    )
    attention["slides"][0]["clinical"]["age"] = 900
    with pytest.raises(ValueError, match="evidence changed"):
        interpret(attention, output)
    changed_output = tmp_path / "attention-changed"
    interpret(attention, changed_output)
    changed_hash = json.loads((changed_output / ".slide-0-member-0.receipt.json").read_text())[
        "inputHash"
    ]
    assert changed_hash != first_hash
